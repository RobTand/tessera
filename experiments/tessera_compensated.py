"""Tessera has no error compensation.  EXL3's number was measured with it.

`exl3_rate_sweep.py:82` fits EXL3 against a real Hessian with LDL-ordered error
compensation and scores held-out ACTIVATION error; every Tessera arm ever run
against it scored uncompensated WEIGHT error.  `exl3_arm_glm_experts_v2.py:35`
records the asymmetry as "stated not netted".  This measures what it is worth,
on Tessera's side, where it can actually be fixed.

The two axes are orthogonal and that is the whole opportunity: Tessera's Viterbi
runs DOWN a column (pairs of output features), GPTQ propagates ACROSS columns
(input features).  So the trellis drops into the GPTQ loop unchanged -- quantise
column j with the existing encoder, divide the residual by `Hinv[j,j]`, push it
into columns > j.  Nothing about the format changes; the encoder just stops
pretending every column is independent.

`--block` is the honest knob.  Feedback is inherently sequential and a batched
Viterbi is not, so a block of B columns is quantised jointly and its residual
propagated to the remainder.  B=1 is exact GPTQ, B=columns is today's encoder,
and reporting the sweep rather than one B is what keeps the claim checkable.

Scored the way EXL3 was: `||x @ W_hat.T - x @ W.T|| / ||x @ W.T||` on the half of
the calibration tokens the Hessian never saw.
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import colour4, lloyd, viterbi
from tessera_memory_and_codebook import labels

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
CC = ConvCode(memory=6)
EXL3_K4 = 0.05653          # compensated, held-out activation error, same tensors


def quantize_block(w, cb, lab, peak, group=32):
    """Today's encoder on a block of columns: group-32 amax, pair rows, Viterbi."""
    R, C = w.shape
    s = (w.reshape(R // group, group, C).abs().amax(1, keepdim=True)
         .clamp(min=1e-12) / peak)
    xn = (w.reshape(R // group, group, C) / s).reshape(R, C)
    seq = xn.reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
    q = viterbi(seq, cb, lab, CC).permute(0, 2, 1).reshape(R, C)
    return (q.reshape(R // group, group, C) * s).reshape(R, C)


def gptq(w, H, cb, lab, peak, block, damp=0.01):
    """W (out, in); H (in, in).  Standard GPTQ, with the trellis as the rounder."""
    w = w.clone()
    n = w.shape[1]
    d = torch.arange(n, device=w.device)
    Hd = H.clone()
    Hd[d, d] += damp * float(torch.diag(H).mean())
    dead = torch.diag(Hd) == 0
    Hd[dead, dead] = 1.0
    w[:, dead] = 0
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    out = torch.empty_like(w)
    for i in range(0, n, block):
        j = min(i + block, n)
        out[:, i:j] = quantize_block(w[:, i:j], cb, lab, peak)
        # residual in the metric Hinv defines, pushed onto the columns not yet seen
        err = (w[:, i:j] - out[:, i:j]) @ torch.linalg.inv(
            Hinv[i:j, i:j]) if block > 1 else \
            (w[:, i:j] - out[:, i:j]) / Hinv[i, i]
        if j < n:
            w[:, j:] -= err @ Hinv[i:j, j:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--blocks", type=int, nargs="+", default=[128, 32, 8])
    ap.add_argument("--out", default="experiments/results/tessera_compensated.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forest = build_forest(grid.rate_cap, grid=grid)
    coset = labels(TCQ(forest, CC).subsets, forest.anchors, grid.size, "cuda")

    acc = {}
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()[:, :a.cols]
        half = xa.shape[0] // 2
        x_fit, x = xa[:half].contiguous(), xa[half:].contiguous()
        H = (x_fit.T @ x_fit).double().float().contiguous()
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name)[:a.rows, :a.cols].contiguous().cuda().float()
            ref = x @ w.T
            den = ref.norm()
            out_rel = lambda q: float((x @ q.T - ref).norm() / den)
            wt_rel = lambda q: float((q - w).norm() / w.norm())

            # the free-codebook lever, fitted once on the original weights
            R, C = w.shape
            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            pairs = xn.reshape(R // 2, 2, C).permute(0, 2, 1).reshape(-1, 2)
            cb = lloyd(pairs[torch.randperm(len(pairs))[:300000]], 256)
            lab = colour4(cb)

            for tag, book, lb in (("E2M1x2", vals, coset), ("Lloyd", cb, lab)):
                q = quantize_block(w, book, lb, peak)
                acc.setdefault(f"{tag:<7} uncompensated", []).append((out_rel(q), wt_rel(q)))
                for b in a.blocks:
                    q = gptq(w, H, book, lb, peak, b)
                    acc.setdefault(f"{tag:<7} GPTQ block={b:<4}", []).append(
                        (out_rel(q), wt_rel(q)))
            print(f"  {layer:>3} {proj:<10} done", flush=True)

    base = sum(v[0] for v in acc["E2M1x2  uncompensated"]) / len(acc["E2M1x2  uncompensated"])
    print(f"\n{'arm':<28}{'out err':>10}{'wt err':>10}{'vs base':>9}{'vs EXL3':>9}")
    for k in sorted(acc, key=lambda k: sum(v[0] for v in acc[k])):
        o = sum(v[0] for v in acc[k]) / len(acc[k])
        t = sum(v[1] for v in acc[k]) / len(acc[k])
        print(f"{k:<28}{o:>10.5f}{t:>10.5f}{base/o:>8.3f}x{o/EXL3_K4:>8.2f}x")
    print(f"{'EXL3 K=4 (compensated)':<28}{EXL3_K4:>10.5f}{'-':>10}"
          f"{base/EXL3_K4:>8.3f}x{1.0:>8.2f}x")
    json.dump({k: v for k, v in acc.items()}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}   all arms 4.00 bpp vs EXL3 4.0117")


if __name__ == "__main__":
    main()
