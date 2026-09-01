"""Can a free codebook keep the embedded rate ladder?

The 1.106x was measured with a FLAT Lloyd codebook at the rate cap, where no
tree is needed.  Tessera's ladder needs one: `build_forest` lays anchors out in
"contiguous dyadic blocks over the value order" so that truncating completion
bits lands on an ancestor, which is what makes one deep encode serve every lower
rate with no second Viterbi.  A flat 2D Lloyd codebook has no value order and no
ancestors, so as measured the lever buys 1.106x and pays the ladder for it.

Tree-structured VQ is the construction that gets both.  Split recursively by
2-means to depth 8: the 2^l nodes at level l are exactly the anchor set a rate-
(l-1) trellis wants, every leaf's ancestors are its own coarser codes, and the
truncation property holds by construction rather than by convention.  The price
is that TSVQ is *constrained* Lloyd -- each centroid must live inside its
parent's cell -- so it cannot be better than flat Lloyd and the question is only
how much worse.

That number decides the design:

  * TSVQ keeps most of 1.106x  ->  build the tree codebook, ladder intact,
    strictly better than today at every rung.
  * TSVQ collapses             ->  the choice is codebook OR ladder, which is a
    different and much narrower proposal.

Both are measured against Tessera's own sub-cap machinery -- `build_forest` plus
`TCQ.subsets` at each rate -- so the ladder comparison is against what actually
ships, not against an idealisation of it.
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.trellis import TCQ, ConvCode
from tessera_free_codebook_trellis import colour4, lloyd, viterbi

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
CC = ConvCode(memory=6)


def two_means(p, iters=12):
    """Balanced-ish binary split: seed on the principal axis, then Lloyd."""
    m = p.mean(0)
    d = p - m
    # power iteration for the principal direction; no full SVD for 2 dims
    v = torch.linalg.eigh((d.T @ d).double())[1][:, -1].float()
    spread = (d @ v).std()
    c = torch.stack([m - v * spread, m + v * spread])
    for _ in range(iters):
        a = torch.cdist(p, c).argmin(1)
        for k in (0, 1):
            if int((a == k).sum()):
                c[k] = p[a == k].mean(0)
    return c


def tsvq(x, depth):
    """Levels of a binary VQ tree.  levels[l] has 2^l centroids; a leaf's
    ancestor at level l is its index >> (depth - l), which IS the truncation
    property `build_forest` promises."""
    levels = [x.mean(0, keepdim=True)]
    assign = torch.zeros(len(x), dtype=torch.long, device=x.device)
    for l in range(depth):
        cent = torch.zeros(2 << l, 2, device=x.device)
        for node in range(1 << l):
            p = x[assign == node]
            if len(p) < 2:
                cent[2 * node] = cent[2 * node + 1] = levels[-1][node]
                continue
            cent[2 * node:2 * node + 2] = two_means(p)
        d = torch.cdist(x, cent)
        # a point may only fall in its own parent's two children
        pair = torch.stack([d.gather(1, (2 * assign)[:, None]).squeeze(1),
                            d.gather(1, (2 * assign + 1)[:, None]).squeeze(1)], 1)
        assign = 2 * assign + pair.argmin(1)
        levels.append(cent)
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--out", default="experiments/results/tessera_tree_codebook.json")
    a = ap.parse_args()

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    grid = tuple_grid(E2M1_GRID, 2)
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                        dtype=torch.float32, device="cuda")
    peak = float(vals.abs().max())

    acc = {}
    for layer in a.layers:
        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name)[:, :a.cols].contiguous().cuda().float()
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
            samp = flat[torch.randperm(len(flat), device=flat.device)[:300000]]
            tree = tsvq(samp, 8)

            for rate in a.rates:
                # -- Tessera today: build_forest anchors + TCQ.subsets
                forest = build_forest(rate, grid=grid)
                anchors = list(forest.anchors)
                sub = TCQ(forest, CC).subsets
                lab = torch.zeros(len(anchors), dtype=torch.long, device="cuda")
                for u, g in enumerate(sub):
                    for p in g:
                        lab[p] = u
                cbg = vals[torch.tensor(anchors, device="cuda")]
                acc.setdefault(("product ", rate), []).append(rel(viterbi(seq, cbg, lab, CC)))

                # -- TSVQ: the 2^(rate+1) nodes at level rate+1 ARE the anchors
                cbt = tree[rate + 1]
                acc.setdefault(("tsvq    ", rate), []).append(
                    rel(viterbi(seq, cbt, colour4(cbt), CC)))

                # -- flat Lloyd at the same anchor count: the upper bound,
                #    and at rate < cap it has no ladder to offer
                cbf = lloyd(samp, 1 << (rate + 1))
                acc.setdefault(("flat    ", rate), []).append(
                    rel(viterbi(seq, cbf, colour4(cbf), CC)))
            print(f"  {layer:>3} {proj:<10} done", flush=True)

    print(f"\n{'rate':>5}{'bpp':>7}{'product':>10}{'tsvq':>10}{'flat':>10}"
          f"{'tsvq gain':>11}{'flat gain':>11}{'tsvq keeps':>12}")
    out = {}
    for rate in a.rates:
        p = sum(acc[("product ", rate)]) / len(acc[("product ", rate)])
        t = sum(acc[("tsvq    ", rate)]) / len(acc[("tsvq    ", rate)])
        f = sum(acc[("flat    ", rate)]) / len(acc[("flat    ", rate)])
        keep = (p / t - 1) / (p / f - 1) * 100 if p / f > 1.0001 else float("nan")
        out[rate] = {"product": p, "tsvq": t, "flat": f}
        print(f"{rate:>5}{rate/2+0.5:>7.2f}{p:>10.5f}{t:>10.5f}{f:>10.5f}"
              f"{p/t:>10.3f}x{p/f:>10.3f}x{keep:>11.0f}%")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
