"""Why the 8-bit rate ladder collapses: it is the tree, not the trellis.

Truncation error has two candidate sources -- the Viterbi PATH was chosen
against a deeper descendant table, or the forest's ANCESTORS are simply bad
representatives of their subtrees.  This isolates the second by removing the
trellis entirely: quantize to the nearest deep code, walk up to the level-c
ancestor, and compare against the best code in that same level-c set.

Result (2026-09-01): E2M1x2 nests at 1.24-1.56x, E4M3 at 2.2-14.8x.  The two
grids take different code paths in ``build_forest`` -- ``_build_forest_kd``
fires only when ``arity > 1``, so E4M3 gets the scalar "contiguous dyadic
blocks over the value order" construction.  On a log-spaced grid a contiguous
block spans orders of magnitude, and its single representative is dominated by
the largest values in it.

This is worth fixing rather than routing around: at full depth every code is
reachable regardless of tree shape (native error is identical across rungs),
so a tree built for nesting quality costs nothing at the top rate and is what
makes the 8-bit band priceable from one encode.
"""
import argparse, json
import torch
from tessera.alphabet import (E2M1_GRID, E4M3_GRID, build_forest, tuple_grid,
                              completion_capacity)


def probe(grid, tag, rungs, n):
    cap, ar = grid.rate_cap, grid.arity
    vals = torch.tensor([grid.vector(c) for c in range(grid.size)], dtype=torch.float32)
    peak = float(vals.abs().max())
    # The sigma/peak ratio ``build_forest`` fits its default samples against.
    x = torch.randn(n, ar) * (peak / 6.0)
    out = []
    print(f"\n{tag}  (arity {ar}, cap {cap}, builder={'kd' if ar > 1 else 'scalar'})")
    print(f"  {'rung':>5} {'c':>3} {'native-alpha':>13} {'trunc-alpha':>12} {'nesting':>9}")
    for rate in rungs:
        depth = completion_capacity(rate, cap)
        if depth == 0:
            continue
        blocks = torch.tensor(build_forest(rate, grid=grid).blocks)
        deep = vals[blocks.reshape(-1)]
        jdeep = torch.cdist(x, deep).argmin(1)
        for c in range(depth + 1):
            step = 1 << (depth - c)
            reach = blocks[:, ::step].contiguous()
            rvals = vals[reach.reshape(-1)]
            anc = (jdeep // blocks.shape[1]) * reach.shape[1] + (jdeep % blocks.shape[1]) // step
            et = ((x - rvals[anc]) ** 2).sum(1).mean().sqrt().item()
            en = ((x - rvals[torch.cdist(x, rvals).argmin(1)]) ** 2).sum(1).mean().sqrt().item()
            print(f"  {rate:5d} {c:3d} {en:13.5f} {et:12.5f} {et/en:8.3f}x")
            out.append({"grid": grid.name, "rung": rate, "completion": c,
                        "native_alphabet": en, "truncated_alphabet": et,
                        "nesting_penalty": et / en})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--out", default="experiments/results/tessera_alphabet_nesting.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    rows = probe(tuple_grid(E2M1_GRID, 2), "E2M1x2 (4-bit)", [2, 4], a.samples)
    rows += probe(E4M3_GRID, "E4M3 (8-bit)", [2, 4], a.samples)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
