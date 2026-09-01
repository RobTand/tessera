"""What is error compensation worth to Tessera -- measured where it is measurable.

The first attempt made things worse, and the reason is not the method.  The
probe's cached activations hold **256 tokens**.  `exl3_rate_sweep.py` splits
them 128/128 and builds a Hessian over **4096 input features**: rank 128 of
4096, minimum eigenvalue negative.  At 1024 features the condition number is
9.0e7 before damping.  GPTQ compensating through that is steering by noise, and
it is why both weight AND output error degraded -- which is not how GPTQ fails
when it fails.

This measures the lever where the Hessian is actually determined.  `--cols 64`
against 128 fit tokens is 2x over-determined; `--cols 1024` reproduces the
broken regime on purpose, so the failure is exhibited rather than described.

Two scores, and the pair is the point:

  held-out   the honest number, on the 128 tokens the Hessian never saw
  in-sample  the same fit scored on its own fit tokens

Their ratio is the overfitting gap.  A compensation that only helps in-sample is
not a compensation, it is a memorised residual -- and with 128 tokens of a
single short probe the two halves are adjacent tokens of the same sequences, so
"held out" is a weaker guarantee here than the word suggests.

Act-order (columns quantised in decreasing `diag(H)`) is on by default: it is
the house recipe's `static_act_order`, and without it GPTQ on an ill-conditioned
H is unstable for reasons that have nothing to do with the rounder under test.
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
from tessera_compensated import quantize_block

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
CC = ConvCode(memory=6)


def gptq(w, H, cb, lab, peak, block, damp, act_order=True):
    w = w.clone()
    n = w.shape[1]
    perm = torch.argsort(torch.diag(H), descending=True) if act_order \
        else torch.arange(n, device=w.device)
    inv = torch.argsort(perm)
    w, H = w[:, perm], H[perm][:, perm]
    d = torch.arange(n, device=w.device)
    Hd = H.clone()
    Hd[d, d] += damp * float(torch.diag(H).mean())
    U = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    out = torch.empty_like(w)
    for i in range(0, n, block):
        j = min(i + block, n)
        out[:, i:j] = quantize_block(w[:, i:j], cb, lab, peak)
        err = torch.linalg.solve_triangular(
            U[i:j, i:j], w[:, i:j] - out[:, i:j], upper=True, left=False)
        if j < n:
            w[:, j:] -= err @ U[i:j, j:]
    return out[:, inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--damps", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--out", default="experiments/results/tessera_compensation_prize.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forest = build_forest(grid.rate_cap, grid=grid)
    coset = labels(TCQ(forest, CC).subsets, forest.anchors, grid.size, "cuda")

    acc, cond = {}, None
    for layer in a.layers:
        blob = torch.load(f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float().cuda()[:, :a.cols]
        h = xa.shape[0] // 2
        x_fit, x_ev = xa[:h].contiguous(), xa[h:].contiguous()
        H = (x_fit.T @ x_fit).double().float().contiguous()
        e = torch.linalg.eigvalsh(H.double())
        cond = float(e.max() / e.min().abs().clamp(min=1e-30))
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name)[:a.rows, :a.cols].contiguous().cuda().float()
            score = {}
            for tag, xs in (("held-out", x_ev), ("in-sample", x_fit)):
                ref = xs @ w.T
                score[tag] = (lambda q, xs=xs, ref=ref, d=ref.norm():
                              float((xs @ q.T - ref).norm() / d))
            R, C = w.shape
            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            pr = xn.reshape(R // 2, 2, C).permute(0, 2, 1).reshape(-1, 2)
            cb = lloyd(pr[torch.randperm(len(pr))[:min(300000, len(pr))]], 256)
            lab = colour4(cb)

            for tag, book, lb in (("E2M1x2", vals, coset), ("Lloyd ", cb, lab)):
                q = quantize_block(w, book, lb, peak)
                acc.setdefault(f"{tag} uncompensated", []).append(
                    (score["held-out"](q), score["in-sample"](q)))
                for dm in a.damps:
                    q = gptq(w, H, book, lb, peak, a.block, dm)
                    acc.setdefault(f"{tag} GPTQ damp={dm:<5}", []).append(
                        (score["held-out"](q), score["in-sample"](q)))
            print(f"  {layer:>3} {proj:<10} done", flush=True)

    base = sum(v[0] for v in acc["E2M1x2 uncompensated"]) / len(acc["E2M1x2 uncompensated"])
    print(f"\ncols={a.cols}  fit tokens={h}  cond(H)={cond:.2e}  block={a.block}")
    print(f"{'arm':<26}{'held-out':>11}{'in-sample':>11}{'gap':>7}{'vs base':>9}")
    for k in sorted(acc, key=lambda k: sum(v[0] for v in acc[k])):
        o = sum(v[0] for v in acc[k]) / len(acc[k])
        i = sum(v[1] for v in acc[k]) / len(acc[k])
        print(f"{k:<26}{o:>11.5f}{i:>11.5f}{o/i:>7.2f}{base/o:>8.3f}x")
    json.dump({"cols": a.cols, "cond": cond, "arms": acc}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
