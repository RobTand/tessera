"""Theoretical limits for an FP4-native format at 4.0 bpp, on six GLM experts.

Rob (2026-09-01): "quantify the theoretical limits so we know when to stop
optimizing".  Everything here is a Shannon bound computed from the weights
and activations, or the MEASURED intrinsic loss of one component on synthetic
Gaussian data.  Rates are bits per weight; errors are RMS relative
(sqrt(MSE / mean(w^2))), the harness's weight-space `wt` unit.

Two worlds at a total rate R:

* rotation world (EXL3): whiten the weights, code them as i.i.d. Gaussian at
  the tensor's mean variance AM with every bit in the trellis:
      D_rot = AM * 2^(-2R)
* block-adaptive world (the FP4 tile): spend R_s bits/weight on per-block
  scales and code each block at its own variance.  The reverse-water-filling
  limit at high rate is the GEOMETRIC mean of the block variances:
      D_blk = GM_b * 2^(-2 (R - R_s))
  so the plane pays off iff AM/GM_b > 2^(2 R_s): 2.0x at R_s = 0.5.

Within a block the source is not Gaussian.  The Shannon lower bound
D >= 2^(2h) / (2 pi e) * 2^(-2R) uses the block-normalised differential
entropy h; a heavy-tailed block has LOWER h at unit variance than a Gaussian,
so it is easier, by 2^(h - h_gauss) in RMS.

Output space: an error e with E[e e^T] = D I costs D tr(H) at the output.
Ideal error feedback along an LDL order whitens it to D sum_i d_i (the LDL
pivots); an ideal transform coder reaches D n GM(lambda).  Both are ceilings
for LDLQ and for rotation respectively, computed from the capture's H.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tessera_fp4_native_levers as F  # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian  # noqa: E402
from tessera.trellis import TCQ  # noqa: E402
from tessera_memory_and_codebook import labels  # noqa: E402

HALF = 16


def am_over_gm(v):
    return float(v.mean() / v.clamp_min(1e-30).log().mean().exp())


def block_var(w, b):
    return w.reshape(w.shape[0], -1, b).pow(2).mean(-1)


def entropy_bits(z, bins=8192):
    """Differential entropy (bits) of the samples of z by histogram."""
    lim = float(z.abs().max()) + 1e-6
    h = torch.histc(z.float(), bins=bins, min=-lim, max=lim)
    p = h / h.sum()
    nz = p[p > 0]
    return float(-(nz * nz.log2()).sum()) + math.log2(2 * lim / bins)


def gauss_bits(var):
    return 0.5 * math.log2(2 * math.pi * math.e * var)


def discrete_entropy_bits(x):
    _, counts = torch.unique(x, return_counts=True)
    p = counts.float() / counts.sum()
    return float(-(p * p.log2()).sum())


def refit_e4m3(w, G, iters):
    """refit-only, flat E4M3 plane: the harness's arm.  Returns (q, eff)."""
    eff = F.e4m3(F.amax_half(w), G)
    q = F.trellis(w, F.expand(eff, *w.shape))
    for _ in range(iters):
        eff = F.e4m3(F.ls_scale(w, q, eff), G)
        q = F.trellis(w, F.expand(eff, *w.shape))
    return q, eff


def synthetic_trellis_loss(rows, cols, seed, Ls=(1, 2)):
    """Intrinsic loss of the E2M1x2 trellis on i.i.d. Gaussian at ONE optimal
    scale: RMS relative error over 2^(-R) for R = 3.5 (L=1), 3.75 (L=2)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(rows, cols, generator=g, device="cuda")
    out = {}
    for L in Ls:
        best = None
        for k in [2.2 + 0.1 * i for i in range(16)]:        # clip point k*sigma at code 6
            s = torch.full((rows * cols // HALF,), k / F.PEAK, device="cuda")
            sc = F.expand(s, rows, cols)
            q = F.trellis(z, sc) if L == 1 else F.multidim(z, sc, L)
            # one exact LS global scale for the chosen codes, then the error
            u = q * sc
            a = float((u * z).sum() / (u * u).sum().clamp_min(1e-30))
            err = float(((z - a * u).norm() / z.norm()))
            if best is None or err < best[0]:
                best = (err, k)
        R = 3.5 if L == 1 else 3.75
        out[f"L={L}"] = {"rms_rel": best[0], "clip_k": best[1], "rate": R,
                         "shannon_gauss": 2 ** (-R), "loss_x": best[0] / 2 ** (-R)}
    return out


def output_ceilings(H, block):
    """(transform gain, feedback gain at block 1, at `block`) in MSE units."""
    n = H.shape[0]
    lam = torch.linalg.eigvalsh(H).clamp_min(1e-12)
    g_tc = float(lam.mean() / lam.log().mean().exp())
    C = torch.linalg.cholesky(H)
    g_fb1 = float(H.diagonal().sum() / (C.diagonal() ** 2).sum())
    Lb = block_ldl(H, block)
    Linv = torch.linalg.solve_triangular(Lb, torch.eye(n, device=H.device), upper=False)
    D = Linv @ H @ Linv.T
    g_fbb = float(H.diagonal().sum() / D.diagonal().sum())
    return g_tc, g_fb1, g_fbb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--fit-docs", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--measured", default="experiments/results/tessera_ldlq_generalisation.json")
    ap.add_argument("--out", default="experiments/results/tessera_theoretical_limits.json")
    a = ap.parse_args()
    logf = open(a.out.replace(".json", ".log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    F.PROD = torch.tensor([F.GRID.vector(c) for c in range(F.GRID.size)], dtype=torch.float32, device="cuda")
    F.COSET = labels(TCQ(F.FOREST, F.CC).subsets, F.FOREST.anchors, F.GRID.size, "cuda")
    index = json.load(open(f"{F.SRC}/model.safetensors.index.json"))["weight_map"]
    out = {"args": vars(a), "tensors": [], "synthetic": {}}

    log("== synthetic i.i.d. Gaussian, 2048x4096, one optimal global scale (no plane)")
    syn = synthetic_trellis_loss(2048, 4096, seed=0)
    out["synthetic"] = syn
    for k, v in syn.items():
        log(f"   {k}: rate {v['rate']} rms_rel {v['rms_rel']:.5f} vs Shannon {v['shannon_gauss']:.5f}"
            f" -> intrinsic loss {v['loss_x']:.3f}x (clip at {v['clip_k']:.1f} sigma)")

    for layer in a.layers:
        blob = torch.load(f"{a.act}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().reshape(-1, a.seqlen, blob["inputs"].shape[-1])
        x_fit = xa[: a.fit_docs].reshape(-1, xa.shape[-1]).cuda()
        x_ev = xa[a.fit_docs:].reshape(-1, xa.shape[-1]).cuda()
        H_fit = (x_fit.T @ x_fit) / x_fit.shape[0]
        H_ev = (x_ev.T @ x_ev) / x_ev.shape[0]
        ceil = {}
        for tag, H, sig in (("fit s=1.0", H_fit, 1.0), ("fit s=0.025", H_fit, 0.025), ("eval s=0.025", H_ev, 0.025)):
            Hr = regularize_hessian(H, sigma_reg=sig)
            g_tc, g_fb1, g_fb32 = output_ceilings(Hr, 32)
            ceil[tag] = {"transform_gain_mse": g_tc, "feedback_gain_block1_mse": g_fb1,
                         "feedback_gain_block32_mse": g_fb32}
            log(f"== L{layer} H [{tag}]: output-space ceilings (RMS): transform {math.sqrt(g_tc):.3f}x,"
                f" ideal feedback block-1 {math.sqrt(g_fb1):.3f}x, block-32 {math.sqrt(g_fb32):.3f}x")
        del H_fit, H_ev, x_fit, x_ev
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{F.SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            t0 = time.time()
            rows, cols = w.shape
            am = float(w.pow(2).mean())
            r16 = am_over_gm(block_var(w, 16)); r32 = am_over_gm(block_var(w, 32))
            r_row = am_over_gm(w.pow(2).mean(1)); r_col = am_over_gm(w.pow(2).mean(0))
            # within-block shape: normalise each 16-block to unit RMS
            z16 = (w.reshape(rows, -1, 16) / block_var(w, 16).sqrt().unsqueeze(-1).clamp_min(1e-30)).reshape(-1)
            dh16 = entropy_bits(z16) - gauss_bits(float(z16.pow(2).mean()))
            zt = (w / math.sqrt(am)).reshape(-1)
            dh_t = entropy_bits(zt) - gauss_bits(float(zt.pow(2).mean()))
            kurt16 = float(z16.pow(4).mean() / z16.pow(2).mean() ** 2)
            # the plane's information content: entropy of the refit E4M3 bytes
            G = F.E4M3_MAX / F.amax_half(w).max()
            q, eff = refit_e4m3(w, G, a.iters)
            wt = float(((w - q * F.expand(eff, rows, cols)).norm() / w.norm()))
            bytes_ = (eff * G).to(torch.float8_e4m3fn).view(torch.uint8)
            h_plane = discrete_entropy_bits(bytes_)
            used = int(torch.unique(bytes_).numel())
            octaves = float(math.log2(float(eff.max() / eff.min())))
            R = 4.0
            b = {
                "rot_gauss_4.0": 2 ** (-R),
                "blk16_free_plane": math.sqrt(1 / r16) * 2 ** (-R),
                "blk16_Rs0.5": math.sqrt(1 / r16) * 2 ** (-(R - 0.5)),
                "blk16_Rs0.5_shape": math.sqrt(1 / r16) * 2 ** (-(R - 0.5)) * 2 ** dh16,
                "blk16_Rs_entropy": math.sqrt(1 / r16) * 2 ** (-(R - h_plane / 16)),
                "blk16_Rs_entropy_shape": math.sqrt(1 / r16) * 2 ** (-(R - h_plane / 16)) * 2 ** dh16,
                "blk32_Rs0.25": math.sqrt(1 / r32) * 2 ** (-(R - 0.25)),
                "blk32_Rs0.25_shape": math.sqrt(1 / r32) * 2 ** (-(R - 0.25)) * 2 ** dh16,
            }
            rec = {"layer": layer, "proj": proj, "am": am, "am_over_gm": {"blk16": r16, "blk32": r32, "row": r_row, "col": r_col},
                   "delta_h_bits": {"blk16": dh16, "tensor": dh_t}, "kurtosis_blk16": kurt16,
                   "plane": {"entropy_bits_per_block": h_plane, "bpp_entropy": h_plane / 16, "used_e4m3_values": used, "octaves": octaves},
                   "measured_refit_e4m3_wt": wt, "bounds_rms_rel": b, "ceilings": ceil}
            out["tensors"].append(rec)
            log(f"== L{layer} {proj}  AM/GM blk16 {r16:.2f} blk32 {r32:.2f} row {r_row:.2f} col {r_col:.2f}"
                f" | within-block dh {dh16:+.3f} bits (kurt {kurt16:.2f}); tensor dh {dh_t:+.3f}")
            log(f"   plane: {used} E4M3 values over {octaves:.2f} octaves, entropy {h_plane:.2f} bits/block = {h_plane/16:.3f} bpp"
                f" (E4M3 spends 0.500)")
            log(f"   bounds (RMS rel) @4.0 bpp: rot {b['rot_gauss_4.0']:.4f} | blk16 free {b['blk16_free_plane']:.4f}"
                f" Rs=.5 {b['blk16_Rs0.5']:.4f} (+shape {b['blk16_Rs0.5_shape']:.4f}) Rs=H {b['blk16_Rs_entropy']:.4f}"
                f" (+shape {b['blk16_Rs_entropy_shape']:.4f}) | blk32 Rs=.25 {b['blk32_Rs0.25']:.4f} (+shape {b['blk32_Rs0.25_shape']:.4f})")
            log(f"   measured refit-only flat-E4M3 plane wt {wt:.5f} = {wt / b['blk16_Rs0.5_shape']:.3f}x its bound,"
                f" {wt / b['blk16_Rs0.5']:.3f}x the Gaussian-block bound   {time.time()-t0:.1f}s")
            del w, q, eff
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
