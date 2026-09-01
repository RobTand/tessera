"""Does the conv code's set partitioning matter on the real 4-bit wire?

Found on synthetic Gaussian (tessera8_bounds.py, codebook battery, 2026-09-01):
the memory-6 code Tessera has always used, Larsen's maximum-free-distance
(0o133, 0o171), pairs the subsets {D0,D3}/{D1,D2} on the two branches leaving
a state.  Ungerboeck's rule for a four-way set partition pairs {D0,D2}/{D1,D3}
-- both branches from (and into) a state see the same "B" family, whose union
is every other level.  The Ungerboeck-form 64-state 1-D code (parity
polynomials h0 = 0o103, h1 = 0o024, taken as feedforward generators
(0o024, 0o103) so the state-only bit is the subset LSB) was 1.394x -> 1.355x
of the Shannon bound at R = 4 on Gaussian with an ideal codebook.

This runs it through ``encode_unit`` on the six GLM experts at the default
wire (span 2, LUT plane, refit 4, scale-weighted trellis) and the legacy wire,
plus the 256-state Ungerboeck code, scored on the capture's held-out 1024 rows
with EXL3's reconstructions (``exl3_reference_quantise.py``) scored on the
very same rows.  A conv code is wire: nothing here changes a default.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.manifest import ScalePlaneKind
from tessera.trellis import ConvCode
from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
EXL3 = "/home/rob/dq-runs/exl3-ref"
UNGERBOECK = {2: (0o2, 0o5), 3: (0o04, 0o13), 4: (0o04, 0o23), 5: (0o10, 0o45),
              6: (0o024, 0o103), 7: (0o126, 0o235), 8: (0o362, 0o515)}
CODES = {
    "Larsen m6 (today)": ConvCode(memory=6),
    "Ungerboeck m6": ConvCode(memory=6, generators=UNGERBOECK[6]),
    "Ungerboeck m8": ConvCode(memory=8, generators=UNGERBOECK[8]),
    "Larsen m8": ConvCode(memory=8),
}
WIRES = {
    "default (span2 LUT refit4 scale-wt)": dict(span=2, scale_plane=ScalePlaneKind.LUT, scale_refit=4, trellis_weighting="scale"),
    "legacy (span1 S6b refit4)": dict(span=1, scale_plane=ScalePlaneKind.S6B, scale_refit=4, trellis_weighting="none"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--codes", nargs="+", default=list(CODES))
    ap.add_argument("--wires", nargs="+", default=list(WIRES))
    ap.add_argument("--out", default="experiments/results/tessera_conv_code_check.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"\n##### tessera_conv_code_check {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(a)}")
    grid = tuple_grid(E2M1_GRID, 2)
    forests = {grid.rate_cap: build_forest(grid.rate_cap, grid=grid)}
    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "act": [], "arms": {}, "args": vars(a)}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] - a.eval_rows
        x_fit, x_ev = xa[:n_fit].contiguous().cuda(), xa[n_fit:].contiguous().cuda()
        g = select_mse_grid_input_global_scale([x_fit])
        xq_s = nvfp4_activation_qdq_served(x_ev, g).float()
        del x_fit
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R, C = w.shape
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            act = float((xq_s @ w.T - y).norm() / ny)
            out["tensors"].append(f"L{layer}.{proj}"); out["act"].append(act)
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  eval {x_ev.shape[0]} rows  act leg served {act:.5f}")
            log(f"    {'arm':<58} {'wt':>8} {'out':>8} {'W4A4':>8} {'enc s':>6}")

            def rec(arm, hat, enc=0.0, **extra):
                r = {"wt": float((hat - w).norm() / nw), "out": float((x_ev @ hat.T - y).norm() / ny),
                     "both_served": float((xq_s @ hat.T - y).norm() / ny), "act": act, "encode_s": enc}
                r.update(extra)
                out["arms"].setdefault(arm, []).append(r)
                log(f"    {arm:<58} {r['wt']:8.5f} {r['out']:8.5f} {r['both_served']:8.5f} {enc:6.1f}")

            for K in (4,):
                hat = torch.load(f"{EXL3}/L{layer}_{proj}_K{K}.pt", map_location="cuda").float()
                rec(f"EXL3 K={K} (4.0117 bpw, same held-out rows)", hat)
                del hat
            for wname in a.wires:
                kw = WIRES[wname]
                for cname in a.codes:
                    code = CODES[cname]
                    torch.cuda.synchronize(); t0 = time.time()
                    unit = encode_unit(w, forests, (grid.rate_cap,) * C, code, completion=0, **kw)
                    torch.cuda.synchronize(); enc = time.time() - t0
                    hat = reconstruct_unit(unit, forests, code)
                    rec(f"{wname} / {cname}", hat, enc, sse=unit.sse)
                    del hat, unit
            json.dump(out, open(a.out, "w"), indent=1)
            del w, y
            torch.cuda.empty_cache()
        del xa, x_ev, xq_s
        torch.cuda.empty_cache()

    log("\n== mean over tensors")
    log(f"    {'arm':<58} {'wt':>8} {'out':>8} {'W4A4':>8} {'out vs today':>12} {'W4A4 vs today':>13}")
    arms = out["arms"]
    today = next((k for k in arms if "default" in k and "Larsen m6" in k), None)
    m = lambda arm, k: sum(r[k] for r in arms[arm]) / len(arms[arm])
    for arm in arms:
        vs = f"{m(today, 'out') / m(arm, 'out'):12.3f}x" if today else ""
        vs2 = f"{m(today, 'both_served') / m(arm, 'both_served'):12.3f}x" if today else ""
        log(f"    {arm:<58} {m(arm, 'wt'):8.5f} {m(arm, 'out'):8.5f} {m(arm, 'both_served'):8.5f} {vs} {vs2}")
    out["summary"] = {arm: {k: m(arm, k) for k in ("wt", "out", "both_served")} for arm in arms}
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
