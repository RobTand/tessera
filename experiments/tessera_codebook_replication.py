"""Does the free-codebook lever survive replication?  It has to be asked.

The 1.111x measured in `tessera_free_codebook_trellis.py` rests on four tensors
at 2048x1024 slices.  This repo has already been bitten by exactly that: the
learned-*values* lever measured 1.006x on one tensor and inverted to 0.990x
across eight.  A mean over four slices is a screen, not a result.

So: full width, no slicing, a spread of layers across the model, and all three
projections -- including `down_proj`, whose input is post-SwiGLU and whose
statistics nothing in today's work has touched.  Per-tensor ratios are printed
rather than only their mean, because the failure mode that matters is a lever
that wins on average while losing on a subset the allocator actually picks.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import colour4, lloyd
from tessera_memory_and_codebook import labels, run

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
CC = ConvCode(memory=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4, help="every Nth layer")
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--out", default="experiments/results/tessera_codebook_replication.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    layers = sorted({int(k.split("layers.")[1].split(".")[0])
                     for k in index if ".experts.0." in k})[::a.stride]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())
    forest = build_forest(grid.rate_cap, grid=grid)
    coset = labels(TCQ(forest, CC).subsets, forest.anchors, grid.size, "cuda")

    rows, t0 = [], time.time()
    print(f"{'layer':>6} {'proj':<11}{'shape':>13}{'E2M1x2':>10}{'Lloyd':>10}{'ratio':>8}")
    for layer in layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            if name not in index:
                continue
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().cuda().float()
            R, C = w.shape
            nrm = torch.linalg.norm(w)
            s = (w.reshape(R // 32, 32, C).abs().amax(1, keepdim=True)
                 .clamp(min=1e-12) / peak)
            xn = (w.reshape(R // 32, 32, C) / s).reshape(R, C)
            seq = xn.reshape(R // 2, 2, C).permute(0, 2, 1).contiguous()
            rel = lambda q: float(torch.linalg.norm(
                w - (q.permute(0, 2, 1).reshape(R, C)
                     .reshape(R // 32, 32, C) * s).reshape(R, C)) / nrm)

            flat = seq.reshape(-1, 2)
            cb = lloyd(flat[torch.randperm(len(flat), device=flat.device)[:300000]], 256)
            e_grid = rel(run(seq, vals, coset, CC))
            e_free = rel(run(seq, cb, colour4(cb), CC))
            rows.append({"layer": layer, "proj": proj, "shape": [R, C],
                         "grid": e_grid, "lloyd": e_free, "ratio": e_grid / e_free})
            print(f"{layer:>6} {proj:<11}{f'{R}x{C}':>13}{e_grid:>10.5f}"
                  f"{e_free:>10.5f}{e_grid/e_free:>7.3f}x", flush=True)
            del w, xn, seq, flat
            torch.cuda.empty_cache()

    r = sorted(x["ratio"] for x in rows)
    n = len(r)
    lose = [x for x in rows if x["ratio"] < 1.0]
    print(f"\n{n} tensors, full width, {time.time()-t0:.0f}s")
    print(f"  ratio   min {r[0]:.3f}x   median {r[n//2]:.3f}x   max {r[-1]:.3f}x")
    print(f"  mean    {sum(r)/n:.3f}x     geometric {torch.tensor(r).log().mean().exp():.3f}x")
    print(f"  tensors where the free codebook LOSES: {len(lose)}"
          + ("" if not lose else "  " + ", ".join(
              f"L{x['layer']}/{x['proj']} {x['ratio']:.3f}x" for x in lose[:6])))
    for p in a.projs:
        sub = [x["ratio"] for x in rows if x["proj"] == p]
        if sub:
            print(f"  {p:<11} n={len(sub):<3} median {sorted(sub)[len(sub)//2]:.3f}x")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
