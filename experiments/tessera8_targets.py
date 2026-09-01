"""Tessera-8 against its two targets -- and Tessera against Gridbook, 4 and 8 bit.

Rob, 2026-09-01: "for the tessera-8 bit analysis, I want you to have two
targets in mind: 1) the theoretical, and 2) how exl3 would perform with 8-bit
activations ... if we can beat exl3 using w8a8 instead of w4a16, we have a
major cross-platform opportunity"; and "compare Tessera to gridbook across
both 4 and 8 bit spaces".

One harness, so every number shares its rows: six GLM-5.3 routed-expert
projections (L5/20/42 gate/up), the 2026-09-01 pread capture, the LAST 1024
rows held out (the split ``exl3_reference_quantise.py`` used, so EXL3's
reconstructions are scored on the very same tokens).  Three legs per arm:

  out   ||X(W_hat - W)^T|| / ||XW^T||          the weight leg, A16
  W?A4  served NVFP4 activations (per-16 E4M3 scale + MSE global scale)
  W?A8  served per-token FP8 activations (``fp8_dynamic_activation_qdq_vllm``)

The 4-bit space: Tessera-4 (E2M1x2, the default wire) at its rungs, Gridbook
FP4-CB at its public rungs (layout v2, k/8 + 0.28125 bpw), NVFP4 RTN, EXL3 K4.
The 8-bit space: every arm whose decoded tile is E4M3 bytes x a scale --
FP8 RTN (the alphabet floor), Tessera-8 as it exists today (E4M3 grid, S6b
per-16 plane, the builder's sub-cap anchors), Tessera-8 on the minor-1 wire,
and the per-channel arm this analysis proposes: no block plane, anchors = a
rate-R Lloyd-Max codebook plus its midpoints (the TCQ codebook geometry that
tessera8_bounds.py found; the doubled Lloyd-Max codebook is WORSE than scalar
RTN at R=5), snapped to E4M3 values, the Ungerboeck-form conv code, a
least-squares row scale, and a scale-weighted trellis -- against Gridbook
FP8-CB (k/8 bpw) and EXL3 K4..K8.

Bounds carried alongside (from tessera8_bounds.py): 2^-R per unit variance
for a Gaussian source, and the E4M3 floor (per-channel FP8 RTN), which no
E4M3-tile format at any rate goes below.
"""
import argparse, json, math, sys, time
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, E4M3_GRID, PayloadGrid, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit, viterbi_columns
from tessera.export import _plan_for
from tessera.manifest import ScalePlaneKind
from tessera.trellis import ConvCode
from tessera8_bounds import e4m3_rtn, gaussian_sample, lloyd_max, ls_refit
from prismaquant.fp8_dynamic import fp8_dynamic_activation_qdq_vllm
from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)
from prismaquant.nvfp4_cb_formats import make_nvfp4_cb_qdq

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
EXL3 = "/home/rob/dq-runs/exl3-ref"
E4M3_MAX = 448.0
UNGERBOECK6 = (0o024, 0o103)
LARSEN6 = ConvCode(memory=6)
UNG6 = ConvCode(memory=6, generators=UNGERBOECK6)
E4M3_ALL = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
E4M3_ALL = E4M3_ALL[torch.isfinite(E4M3_ALL)].unique().sort().values   # 255 values incl. 0


def snap_unique(levels: torch.Tensor) -> torch.Tensor:
    """Nearest E4M3 value per level, resolving collisions to the nearest unused one."""
    grid = E4M3_ALL.to(levels.device)
    used: set = set()
    out = []
    for v in levels.sort().values.tolist():
        order = (grid - v).abs().argsort().tolist()
        for i in order:
            if i not in used:
                used.add(i); out.append(float(grid[i])); break
    return torch.tensor(sorted(out), device=levels.device)


def midpoint_codebook(z: torch.Tensor, R: int) -> torch.Tensor:
    """2^R Lloyd-Max levels of ``z`` plus the 2^R - 1 midpoints and one outer point."""
    lm = lloyd_max(z, 1 << R)
    mids = (lm[:-1] + lm[1:]) / 2
    outer = lm[:1] - (lm[1:2] - lm[:1]) / 2
    return torch.cat([lm, mids, outer]).sort().values


def per_channel_tcq(w, R, code, book, sigma_units, refits, weighted, log):
    """No-plane Tessera-8: row scale, E4M3-snapped codebook, trellis, LS refit."""
    rows, cols = w.shape
    rms = w.pow(2).mean(dim=1, keepdim=True).sqrt()
    s = rms / sigma_units                                    # decoded = value * s
    values = book                                           # in E4M3 units
    grid = PayloadGrid(f"pc{R}", tuple(float(v) for v in values.tolist()))
    forest = build_forest(R, grid=grid)
    hat = None
    for it in range(refits + 1):
        targets = (w / s).contiguous()
        weights = None
        if weighted:
            weights = (s / s.max()).pow(2).expand(rows, cols).contiguous()
        anchors, _, _ = viterbi_columns(targets, forest, code, completion=0, weights=weights)
        q = values.to(w.device)[anchors]
        if it < refits:
            num = (w * q).sum(dim=1, keepdim=True)
            den = (q * q).sum(dim=1, keepdim=True)
            s = torch.where(den > 0, num / den, s)
        hat = q * s
    clipped = float(((w / s).abs() > E4M3_MAX).float().mean())
    return hat, clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--sigma-units", type=float, nargs="+", default=[64.0])
    ap.add_argument("--refits", type=int, default=2)
    ap.add_argument("--tessera4-rungs", type=int, nargs="+", default=[896, 768, 640])
    ap.add_argument("--fp4cb", type=int, nargs="+", default=[24, 20, 16])
    ap.add_argument("--fp8cb", type=int, nargs="+", default=[32, 40, 48])
    ap.add_argument("--exl3", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--skip", default="", help="regex over arm names to skip")
    ap.add_argument("--out", default="experiments/results/tessera8_targets.json")
    a = ap.parse_args()
    import re
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"\n##### tessera8_targets {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(a)}")
    grid4 = tuple_grid(E2M1_GRID, 2)
    plans4 = {q: _plan_for(grid4, q, 4096) for q in a.tessera4_rungs}
    forests8 = {R: build_forest(R, grid=E4M3_GRID) for R in a.rates}
    z = gaussian_sample(1 << 16, "cuda")
    books_gauss = {R: midpoint_codebook(z, R) for R in a.rates}
    cb4 = {k: make_nvfp4_cb_qdq(k, "fp4", "product", scale_coding="two_tier") for k in a.fp4cb}
    cb8 = {k: make_nvfp4_cb_qdq(k, "fp8", "product") for k in a.fp8cb}
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "legs": [], "arms": {}, "args": vars(a),
           "bounds": {str(R): {"gaussian_rms_2^-R": 2.0 ** -R} for R in a.rates}}
    skip = re.compile(a.skip) if a.skip else None

    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
        g = select_mse_grid_input_global_scale([x_fit])
        xq4 = nvfp4_activation_qdq_served(x_ev, g).float()
        xq8 = fp8_dynamic_activation_qdq_vllm(x_ev).dequant.float()
        del x_fit, xa
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R_, C = w.shape
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            legs = {"a4": float((xq4 @ w.T - y).norm() / ny), "a8": float((xq8 @ w.T - y).norm() / ny)}
            out["tensors"].append(f"L{layer}.{proj}"); out["legs"].append(legs)
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  held-out {x_ev.shape[0]} rows  "
                f"act legs: A4 {legs['a4']:.5f}  A8 {legs['a8']:.5f}")
            log(f"    {'arm':<62} {'bpp':>6} {'wt':>8} {'out':>8} {'W?A4':>8} {'W?A8':>8} {'s':>5}")
            t_arm = [time.time()]

            def rec(arm, hat, bpp, **extra):
                if skip and skip.search(arm):
                    return
                r = {"bpp": bpp, "wt": float((hat - w).norm() / nw),
                     "out": float((x_ev @ hat.T - y).norm() / ny),
                     "a4": float((xq4 @ hat.T - y).norm() / ny),
                     "a8": float((xq8 @ hat.T - y).norm() / ny)}
                r.update(extra)
                out["arms"].setdefault(arm, []).append(r)
                now = time.time(); dt = now - t_arm[0]; t_arm[0] = now
                log(f"    {arm:<62} {bpp:6.3f} {r['wt']:8.5f} {r['out']:8.5f} {r['a4']:8.5f} {r['a8']:8.5f} {dt:5.1f}")

            # ---- references
            for K in a.exl3:
                p = Path(f"{EXL3}/L{layer}_{proj}_K{K}.pt")
                if p.exists():
                    rec(f"EXL3 K={K}", torch.load(p, map_location="cuda").float(), K + 0.0117)
            s = w.abs().amax(dim=1, keepdim=True) / E4M3_MAX
            s = ls_refit(w, s, 6, dim=1)
            rec("FP8 RTN per-channel LS-refit (the E4M3 floor)", e4m3_rtn(w / s) * s, 8.0 + 32 / C)
            blocks = w.reshape(R_, C // 16, 16)
            sb = blocks.abs().amax(dim=2, keepdim=True) / 6.0
            # NVFP4 RTN with an fp32 per-16 scale: the tile's own RTN reference
            e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda")
            tq = blocks / sb
            idx = (tq.abs().unsqueeze(-1) - e2m1).abs().argmin(dim=-1)
            rec("NVFP4 RTN per-16 amax (fp32 scale)", (e2m1[idx] * tq.sign() * sb).reshape(R_, C), 4.0 + 2.0)

            # ---- 4-bit space: Tessera-4 default wire at its rungs, Gridbook FP4-CB
            for q in a.tessera4_rungs:
                rates, forests = plans4[q]
                unit = encode_unit(w, forests, rates, LARSEN6, completion=0, span=2,
                                   scale_plane=ScalePlaneKind.LUT, scale_refit=4, trellis_weighting="scale")
                rec(f"Tessera-4 q256={q} (span2 LUT refit4 scale-wt)", reconstruct_unit(unit, forests, LARSEN6), q / 256 + 0.5)
                del unit
            for k in a.fp4cb:
                rec(f"Gridbook FP4-CB K{k} (v2 two-tier)", cb4[k](w), k / 8 + 0.28125)

            # ---- 8-bit space
            for R in a.rates:
                unit = encode_unit(w, forests8[R], (R,) * C, LARSEN6, completion=0, span=1,
                                   scale_plane=ScalePlaneKind.S6B, scale_refit=4, trellis_weighting="none")
                rec(f"Tessera-8 R={R} today (E4M3 S6b span1 refit4, builder anchors)", reconstruct_unit(unit, forests8[R], LARSEN6), R + 0.5)
                unit = encode_unit(w, forests8[R], (R,) * C, LARSEN6, completion=0, span=1,
                                   scale_plane=ScalePlaneKind.LUT, scale_refit=4, trellis_weighting="scale")
                rec(f"Tessera-8 R={R} LUT plane alone (span1 refit4 scale-wt)", reconstruct_unit(unit, forests8[R], LARSEN6), R + 0.25)
                unit = encode_unit(w, forests8[R], (R,) * C, LARSEN6, completion=0, span=2,
                                   scale_plane=ScalePlaneKind.LUT, scale_refit=4, trellis_weighting="scale")
                rec(f"Tessera-8 R={R} minor-1 wire (LUT span2 refit4 scale-wt)", reconstruct_unit(unit, forests8[R], LARSEN6), R + 0.75)
                del unit
                for su in a.sigma_units:
                    book_ideal = books_gauss[R] * su
                    book = snap_unique(book_ideal)
                    hat, clip = per_channel_tcq(w, R, UNG6, book, su, a.refits, True, log)
                    rec(f"Tessera-8 R={R} per-channel, LM+mid E4M3 anchors, Ungerboeck, s={su:.0f}", hat, R + 32 / C, clipped=clip)
                    if su == a.sigma_units[0]:
                        hat, _ = per_channel_tcq(w, R, LARSEN6, book, su, a.refits, True, log)
                        rec(f"Tessera-8 R={R} per-channel, same, Larsen code", hat, R + 32 / C)
                        hat, _ = per_channel_tcq(w, R, UNG6, book, su, a.refits, False, log)
                        rec(f"Tessera-8 R={R} per-channel, same, unweighted trellis", hat, R + 32 / C)
                        hat, _ = per_channel_tcq(w, R, UNG6, book_ideal, su, a.refits, True, log)
                        rec(f"Tessera-8 R={R} per-channel, IDEAL anchors (not E4M3; reference)", hat, R + 32 / C)
                        # codebook fit to the data's own row-normalised distribution
                        zs = (w / w.pow(2).mean(dim=1, keepdim=True).sqrt()).reshape(-1)
                        zs = zs[torch.randperm(zs.numel(), device="cuda")[: 1 << 18]]
                        book_data = snap_unique(midpoint_codebook(zs, R) * su)
                        hat, _ = per_channel_tcq(w, R, UNG6, book_data, su, a.refits, True, log)
                        rec(f"Tessera-8 R={R} per-channel, data-fit LM+mid E4M3 anchors", hat, R + 32 / C)
            for k in a.fp8cb:
                rec(f"Gridbook FP8-CB K{k}", cb8[k](w), k / 8 + 32 / C)
            json.dump(out, open(a.out, "w"), indent=1)
            del w, y
            torch.cuda.empty_cache()
        del x_ev, xq4, xq8
        torch.cuda.empty_cache()

    # ---- summary
    arms = out["arms"]
    m = lambda arm, k: sum(r[k] for r in arms[arm]) / len(arms[arm])
    legs = {k: sum(l[k] for l in out["legs"]) / len(out["legs"]) for k in ("a4", "a8")}
    out["summary"] = {arm: {k: m(arm, k) for k in ("bpp", "wt", "out", "a4", "a8")} for arm in arms}
    out["summary_legs"] = legs
    log(f"\n== mean over {len(out['tensors'])} tensors   act legs: A4 {legs['a4']:.5f}  A8 {legs['a8']:.5f}")
    log(f"    {'arm':<62} {'bpp':>6} {'wt':>8} {'out':>8} {'W?A4':>8} {'W?A8':>8}")
    for arm in arms:
        r = out["summary"][arm]
        log(f"    {arm:<62} {r['bpp']:6.3f} {r['wt']:8.5f} {r['out']:8.5f} {r['a4']:8.5f} {r['a8']:8.5f}")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
