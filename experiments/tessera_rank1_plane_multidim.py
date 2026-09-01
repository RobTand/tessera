"""Delete the scale plane; spend its bits on the partition.  Still FP4-native.

Where the EXL3 gap lives, by the bit.  EXL3 spends all four bits of K=4 in its
trellis: one fp16 scale per tensor, weights made Gaussian by rotation, no block
scales.  Tessera spends 3.5 in the trellis and 0.5 on per-16 scales.  Half a
bit of trellis rate is worth far more than half a bit of block adaptation --
the overhead budget measured a rank-1 magnitude field (one gain per row, one
per column block) reproducing the block plane's error to within 1.7% on these
experts.  The plane is nearly worthless as *error*; it is expensive as *rate*.

Under the FP4-native constraint the hardware still consumes a per-16 E4M3 scale
-- but the artifact need not store what the loader can derive.  A rank-1
field ``s[n, b] = E4M3(r[n] * c[b])`` materialises into exactly the per-16 tile
NVFP4 wants, costs ~0.005 bpp to store, and the encoder prices the rounded
product so the codes are chosen for what the MMA will multiply (principle 8).

The freed half bit cannot go to the per-position rate: E2M1 has sixteen codes,
so an arity-2 trellis caps at 3.5 bits/weight.  Wei's multidimensional
partition lifts the cap to ``4 - 1/(2L)`` with the codebook untouched
(``tessera_multidim_partition.py``), which is what makes the plane's bits
spendable at all.  So the candidate format is

    rank-1 field  +  E2M1x2 trellis  +  multidim partition at L

at ``4 - 1/(2L) + ~0.005`` bpp: 3.75 (L=2), 3.88 (L=4), 3.94 (L=8) -- all
*below* today's 4.0.  Both the field and the block plane get the same
least-squares refit against the trellis's codes, so the comparison is between
formats and not between one refit and none.

Scored as `tessera_fp4_native_levers.py` scores: weight space, output-space
weight leg, W4A4 composite under the served activation quantiser, on the
held-out 128 tokens.  Decoder start state fixed at 0 throughout.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tessera_fp4_native_levers as F
from tessera.trellis import TCQ
from tessera_memory_and_codebook import labels
from prismaquant.nvfp4_activation_contract import (
    nvfp4_activation_qdq_served, select_mse_grid_input_global_scale)
from tessera_w4a4_projection import quant_a4

HALF = F.HALF


def fit_rank1(w):
    """``r[n] c[b]`` ~ RMS of block ``(n, b)``, least squares in the log domain."""
    R, C = w.shape
    rms = w.reshape(R, C // HALF, HALF).pow(2).mean(-1).sqrt().clamp_min(1e-12)
    log = rms.log()
    lr = log.mean(1, keepdim=True)
    lc = (log - lr).mean(0, keepdim=True)
    return lr.exp().squeeze(1), lc.exp().squeeze(0)


def materialise(r, c):
    """The per-16 E4M3 tile the MMA will see, from the stored field, behind
    one fp32 global scale of the field's own (the largest product lands at
    448, as `e4m3` does for any plane)."""
    s = r[:, None] * c[None, :]
    return F.e4m3(s.reshape(-1)).reshape(s.shape)


def refit_rank1(w, q, r, c, inner=4):
    """Alternating least squares on ``r`` and ``c`` for fixed codes ``q``:
    each update is the exact minimiser of sum (w - r c u)^2 over its factor."""
    R, C = w.shape
    U, W = q.reshape(R, C // HALF, HALF), w.reshape(R, C // HALF, HALF)
    A, B = (U * U).sum(-1), (W * U).sum(-1)                  # (R, blocks)
    for _ in range(inner):
        num = (B * c[None, :]).sum(1)
        den = (A * c[None, :] ** 2).sum(1)
        r = torch.where(den > 0, num / den.clamp_min(1e-30), r).clamp_min(1e-12)
        num = (B * r[:, None]).sum(0)
        den = (A * r[:, None] ** 2).sum(0)
        c = torch.where(den > 0, num / den.clamp_min(1e-30), c).clamp_min(1e-12)
    return r, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="experiments/results/tessera_rank1_plane_multidim.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    F.PROD = torch.tensor([F.GRID.vector(k) for k in range(F.GRID.size)],
                          dtype=torch.float32, device="cuda")
    F.COSET = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, F.GRID.size, "cuda")
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"tensors": [], "act": [], "arms": {}, "exl3_k4": F.EXL3_K4, "args": vars(a)}

    for layer in a.layers:
        blob = torch.load(f"{F.ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()
        n = xa.shape[0] // 2
        x_fit, x_ev = xa[:n].contiguous(), xa[n:].contiguous()
        g = select_mse_grid_input_global_scale([x_fit])
        xq_s = nvfp4_activation_qdq_served(x_ev, g).float()
        xq_a = quant_a4(x_ev)
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R, C = w.shape
            sc = F.Scorer(w, x_ev, xq_s, xq_a)
            act = {"served": float((xq_s @ w.T - sc.y).norm() / sc.ny),
                   "amax": float((xq_a @ w.T - sc.y).norm() / sc.ny)}
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  act leg served {act['served']:.5f}")
            log(f"    {'arm':<40} {'bpp':>6} {'wt':>7} {'out':>7} {'W4A4':>7}")
            res = {}
            t0 = time.time()

            def rec(nm, hat, bpp):
                res[nm] = sc(hat, bpp=bpp)
                log(f"    {nm:<40} {bpp:>6.3f} {res[nm]['wt']:.5f} {res[nm]['out']:.5f} "
                    f"{res[nm]['both_served']:.5f}  {time.time() - t0:5.1f}s")

            field_bpp = (16 * R + 16 * (C // HALF)) / (R * C)
            G = F.E4M3_MAX / F.amax_half(w).max()

            # --- the block plane, refit, at every L (today's format + its refit)
            for L in a.levels:
                eff = F.s6b(w)
                enc = (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else \
                      (lambda w_, s_, L=L: F.multidim(w_, s_, L))
                q = enc(w, F.expand(eff, R, C))
                for _ in range(a.iters):
                    eff = F.two_tier(F.ls_scale(w, q, eff))
                    q = enc(w, F.expand(eff, R, C))
                rec(f"S6b plane + LS x{a.iters}  L={L}", q * F.expand(eff, R, C),
                    4 - 1 / (2 * L) + 0.5)

            # --- the rank-1 field, amax-consistent start, then refit
            # Land the field where the block plane lands: amax_half/peak, i.e.
            # the half's amax on the top of the grid.  (An earlier run of this
            # file started 6x too large -- amax instead of amax/peak -- and then
            # clamped at E4M3's 448 behind a borrowed global scale; the LS
            # iterations could not recover, and every rank-1 row was void.)
            r, c = fit_rank1(w)
            kappa = float((F.amax_half(w).reshape(R, C // HALF) / (r[:, None] * c[None, :])).mean())
            r = r * kappa
            assert float(F.amax_half(w).max()) / 6 < float((r[:, None] * c[None, :]).max()) < float(F.amax_half(w).max()) * 6
            for L in a.levels:
                enc = (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else \
                      (lambda w_, s_, L=L: F.multidim(w_, s_, L))
                rr, cc = r.clone(), c.clone()
                s = torch.repeat_interleave(materialise(rr, cc), HALF, dim=1)
                q = enc(w, s)
                if L == 1:
                    rec("rank-1 field (amax start)  L=1", q * s, 3.5 + field_bpp)
                for _ in range(a.iters):
                    rr, cc = refit_rank1(w, q, rr, cc)
                    s = torch.repeat_interleave(materialise(rr, cc), HALF, dim=1)
                    q = enc(w, s)
                rec(f"rank-1 field + LS x{a.iters}  L={L}", q * s,
                    4 - 1 / (2 * L) + field_bpp)

            out["tensors"].append(f"L{layer}.{proj}")
            out["act"].append(act)
            for k, v in res.items():
                out["arms"].setdefault(k, []).append(v)
            json.dump(out, open(a.out, "w"), indent=1)
            del w, sc
            torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s, xq_a
        torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    act_s = [x["served"] for x in out["act"]]
    base = out["arms"].get("S6b plane + LS x%d  L=1" % a.iters)
    log(f"\nact leg served {mean(act_s):.5f} over {len(act_s)} tensors")
    log(f"{'arm':<40}{'bpp':>7}{'wt':>9}{'out':>9}{'W4A4':>9}{'out vs S6b+LS L=1':>20}"
        f"{'min':>7}{'max':>7}{'vs EXL3@A4':>12}")
    for k, v in out["arms"].items():
        bpp = v[0]["bpp"]
        wt, o, b = mean([x["wt"] for x in v]), mean([x["out"] for x in v]), \
            mean([x["both_served"] for x in v])
        ratio = [bb["out"] / rr["out"] for bb, rr in zip(base, v)] if base else [1.0]
        exl = [rr["both_served"] / (F.EXL3_K4 ** 2 + aa ** 2) ** 0.5 for rr, aa in zip(v, act_s)]
        log(f"{k:<40}{bpp:>7.3f}{wt:>9.5f}{o:>9.5f}{b:>9.5f}{mean(ratio):>19.3f}x"
            f"{min(ratio):>7.3f}{max(ratio):>7.3f}{mean(exl):>11.3f}x")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
