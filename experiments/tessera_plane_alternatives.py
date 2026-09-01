"""Two ways to spend less than 0.5 bpp on the plane, priced as they would serve.

`tessera_rank1_plane_multidim.py` showed a rank-1 field over (row, 16-column
block) is 1.32x worse than the block plane: the plane's bits are mostly buying
per-COLUMN magnitude structure, which a per-16 K-block can carry and a block
field cannot.  Two candidates remain:

  A. a rank-1 field over (row, COLUMN): ``s[n, k] = r[n] c[k]``.  The column
     factor cannot live in a per-16 hardware scale, so it must be folded into
     the activations: ``y = (X diag c) (W')^T diag r``.  That changes what the
     activation quantiser sees, so the W4A4 composite is scored on ``X diag c``
     quantised by the served quantiser -- the fold is priced, not assumed.
     For an MoE layer ``c`` would also have to be shared by every expert that
     reads ``X``; a per-tensor ``c`` is therefore an upper bound on this lane.
  B. one E4M3 scale per 32 K elements, stored once and served as two identical
     per-16 scales (rides the NVFP4 route unchanged).  0.25 bpp, which is what
     Wei's L=2 partition needs to reach 3.75 payload: 4.0 bpp total.

Both get the same least-squares refit as the block plane.  Scored as the lever
battery scores, on the held-out 128 tokens, decoder start state fixed.
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

HALF, GROUP = F.HALF, F.GROUP


def fit_rank1_cols(w):
    R, C = w.shape
    log = w.abs().clamp_min(1e-12).log()
    lr = log.mean(1, keepdim=True)
    lc = (log - lr).mean(0, keepdim=True)
    return lr.exp().squeeze(1), lc.exp().squeeze(0)


def refit_rank1_cols(w, q, r, c, inner=4):
    A, B = q * q, w * q                                      # (R, C)
    for _ in range(inner):
        num, den = (B * c[None, :]).sum(1), (A * c[None, :] ** 2).sum(1)
        r = torch.where(den > 0, num / den.clamp_min(1e-30), r).clamp_min(1e-12)
        num, den = (B * r[:, None]).sum(0), (A * r[:, None] ** 2).sum(0)
        c = torch.where(den > 0, num / den.clamp_min(1e-30), c).clamp_min(1e-12)
    return r, c


def full_ldlq_refit(w, Hreg, plane, requant, iters):
    """EXL3's LDLQ schedule with the shipping encoder inside each 32-column
    block: amax plane, trellis, then the least-squares refit iterated -- what
    `encode_unit` now does on every slice.  Do the two levers stack?"""
    from tessera.compensate import block_ldl, compensated_targets
    L = block_ldl(Hreg, GROUP)

    def encode(target, start, stop):
        eff = plane(target)
        q = F.trellis(target, F.expand(eff, *target.shape))
        for _ in range(iters):
            eff = requant(F.ls_scale(target, q, eff))
            q = F.trellis(target, F.expand(eff, *target.shape))
        return q * F.expand(eff, *target.shape)

    return compensated_targets(w, L, encode, block=GROUP)[1]


def per32(s_half):
    """Two halves -> one E4M3 scale per 32, duplicated."""
    return s_half.reshape(-1, 2).mean(1).repeat_interleave(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="experiments/results/tessera_plane_alternatives.json")
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
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R, C = w.shape
            y = x_ev @ w.T
            ny, nw = y.norm(), w.norm()
            act = float((xq_s @ w.T - y).norm() / ny)
            log(f"\n== L{layer} {proj} {tuple(w.shape)}  act leg served {act:.5f}")
            log(f"    {'arm':<40} {'bpp':>6} {'wt':>7} {'out':>7} {'W4A4':>7}  {'act':>7}")
            res = {}
            t0 = time.time()

            def rec(nm, hat, bpp, xq=xq_s, act_leg=act):
                res[nm] = {"wt": float((hat - w).norm() / nw),
                           "out": float((x_ev @ hat.T - y).norm() / ny),
                           "both_served": float((xq @ hat.T - y).norm() / ny),
                           "act": act_leg, "bpp": bpp}
                r_ = res[nm]
                log(f"    {nm:<40} {bpp:>6.3f} {r_['wt']:.5f} {r_['out']:.5f} "
                    f"{r_['both_served']:.5f}  {act_leg:.5f}  {time.time() - t0:5.1f}s")

            G = F.E4M3_MAX / F.amax_half(w).max()

            # reference: today's plane + refit, L=1 and L=2
            for L in (1, 2):
                enc = (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else \
                      (lambda w_, s_: F.multidim(w_, s_, 2))
                eff = F.s6b(w)
                q = enc(w, F.expand(eff, R, C))
                for _ in range(a.iters):
                    eff = F.two_tier(F.ls_scale(w, q, eff))
                    q = enc(w, F.expand(eff, R, C))
                rec(f"S6b plane + LS x{a.iters}  L={L}", q * F.expand(eff, R, C),
                    4 - 1 / (2 * L) + 0.5)

            # B: E4M3 per 32, duplicated per 16
            for L in (1, 2):
                enc = (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else \
                      (lambda w_, s_: F.multidim(w_, s_, 2))
                eff = F.e4m3(per32(F.amax_half(w).reshape(-1, 2).amax(1).repeat_interleave(2)), G)
                q = enc(w, F.expand(eff, R, C))
                for _ in range(a.iters):
                    eff = F.e4m3(per32(F.ls_scale(w, q, eff)), G)
                    q = enc(w, F.expand(eff, R, C))
                rec(f"E4M3 per-32 dup + LS x{a.iters}  L={L}", q * F.expand(eff, R, C),
                    4 - 1 / (2 * L) + 0.25)

            # C: compensation with the refit inside every block (do they stack?)
            from tessera.compensate import regularize_hessian
            Hreg = regularize_hessian((x_fit.T @ x_fit).contiguous(), count=n, sigma_reg=1.0)
            rec(f"full-LDLQ s=1.0 + in-block LS x{a.iters} (s6b)",
                full_ldlq_refit(w, Hreg, F.s6b, F.two_tier, a.iters), 4.0)
            rec(f"full-LDLQ s=1.0 + in-block LS x{a.iters} (e4m3)",
                full_ldlq_refit(w, Hreg, lambda t: F.e4m3(F.amax_half(t), G),
                                lambda s_: F.e4m3(s_, G), a.iters), 4.0)
            del Hreg

            # A: per-column rank-1 field, column factor folded into activations
            r, c = fit_rank1_cols(w)
            # land like the block plane: the half's amax on the top of the grid
            kappa = float((F.amax_half(w).reshape(R, C // HALF)
                           / (r[:, None] * c.reshape(-1, HALF).amax(1)[None, :])).mean())
            r = r * kappa
            field_bpp = (16 * R + 16 * C) / (R * C)
            for L in (1, 2, 8):
                enc = (lambda w_, s_: F.trellis(w_, s_)) if L == 1 else \
                      (lambda w_, s_, L=L: F.multidim(w_, s_, L))
                rr, cc = r.clone(), c.clone()
                s = F.e4m3(rr)[:, None] * cc[None, :]
                q = enc(w, s)
                for _ in range(a.iters):
                    rr, cc = refit_rank1_cols(w, q, rr, cc)
                    s = F.e4m3(rr)[:, None] * cc[None, :]
                    q = enc(w, s)
                # served composite on the folded activations X diag(c), weights q * r
                xc_fit, xc_ev = x_fit * cc[None, :], x_ev * cc[None, :]
                gc = select_mse_grid_input_global_scale([xc_fit])
                xqc = nvfp4_activation_qdq_served(xc_ev, gc).float()
                act_c = float((xqc @ (w / cc[None, :]).T - y).norm() / ny)
                hat = q * s
                res_name = f"rank-1 per-column (fold) + LS x{a.iters}  L={L}"
                res[res_name] = {"wt": float((hat - w).norm() / nw),
                                 "out": float((x_ev @ hat.T - y).norm() / ny),
                                 "both_served": float((xqc @ (q * F.e4m3(rr)[:, None]).T - y).norm() / ny),
                                 "act": act_c, "bpp": 4 - 1 / (2 * L) + field_bpp}
                r_ = res[res_name]
                log(f"    {res_name:<40} {r_['bpp']:>6.3f} {r_['wt']:.5f} {r_['out']:.5f} "
                    f"{r_['both_served']:.5f}  {act_c:.5f}  {time.time() - t0:5.1f}s")

            out["tensors"].append(f"L{layer}.{proj}")
            out["act"].append(act)
            for k, v in res.items():
                out["arms"].setdefault(k, []).append(v)
            json.dump(out, open(a.out, "w"), indent=1)
            del w
            torch.cuda.empty_cache()
        del xa, x_fit, x_ev, xq_s
        torch.cuda.empty_cache()

    mean = lambda v: sum(v) / len(v)
    base = out["arms"][f"S6b plane + LS x{a.iters}  L=1"]
    log(f"\n{'arm':<40}{'bpp':>7}{'wt':>9}{'out':>9}{'W4A4':>9}{'act':>9}{'out vs S6b+LS':>16}{'W4A4 vs':>9}{'vs EXL3@A4':>12}")
    for k, v in out["arms"].items():
        wt, o, b, ac = (mean([x[f] for x in v]) for f in ("wt", "out", "both_served", "act"))
        ro = mean([bb["out"] / rr["out"] for bb, rr in zip(base, v)])
        rb = mean([bb["both_served"] / rr["both_served"] for bb, rr in zip(base, v)])
        exl = mean([rr["both_served"] / (F.EXL3_K4 ** 2 + aa ** 2) ** 0.5 for rr, aa in zip(v, out["act"])])
        log(f"{k:<40}{v[0]['bpp']:>7.3f}{wt:>9.5f}{o:>9.5f}{b:>9.5f}{ac:>9.5f}{ro:>15.3f}x{rb:>8.3f}x{exl:>11.3f}x")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
