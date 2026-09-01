"""Would a learned (Lloyd) alphabet help, or is it the anchor selection?

The gap-to-Lloyd measured earlier conflates two causes:

  * the fixed grid's VALUES are a poor fit to the weight distribution, or
  * the anchor SELECTION from those values (rank-blocked dyadic partition) is
    poor.

Only the first argues for learning the spacing.  Three quantizers at matched
code count separate them:

  A  actual anchors      -- what ``build_forest`` picks from the grid
  B  best subset of grid -- k-medoids restricted to the grid's own values
  C  free Lloyd          -- k-means, points anywhere

B vs C isolates the value set; A vs B isolates the selection.  If B ~ C the
grid's values are fine and the tree is the problem; if B >> C the values
themselves are limiting and learning them would pay.
"""
import argparse, json
import torch
from tessera.alphabet import (E2M1_GRID, E4M3_GRID, build_forest, tuple_grid,
                              alphabet_size)


def rms(x, c):
    return ((x - c[torch.cdist(x, c).argmin(1)]) ** 2).sum(1).mean().sqrt().item()


def lloyd(x, k, iters=80):
    c = x[torch.randperm(len(x))[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


def kmedoids_on_grid(x, vals, k, iters=25):
    """Best k-subset of ``vals``: greedy seed, then steepest-descent swaps."""
    D = torch.cdist(x, vals)                      # [n, |grid|]
    chosen = [int(D.mean(0).argmin())]
    best = D[:, chosen[0]].clone()
    while len(chosen) < k:                        # greedy: cheapest addition
        gain = torch.minimum(D, best[:, None]).mean(0)
        gain[torch.tensor(chosen)] = float("inf")
        j = int(gain.argmin()); chosen.append(j)
        best = torch.minimum(best, D[:, j])
    for _ in range(iters):                        # swap refinement
        improved = False
        for slot in range(k):
            rest = [c for i, c in enumerate(chosen) if i != slot]
            base = D[:, rest].min(1).values
            cand = torch.minimum(D, base[:, None]).mean(0)
            j = int(cand.argmin())
            if cand[j] < base.mean() and j not in rest and j != chosen[slot]:
                if cand[j] < torch.minimum(D[:, chosen[slot]], base).mean():
                    chosen[slot] = j; improved = True
        if not improved:
            break
    return vals[torch.tensor(chosen)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--out", default="experiments/results/tessera_learned_spacing.json")
    a = ap.parse_args()
    rows = []
    for grid, tag in [(tuple_grid(E2M1_GRID, 2), "E2M1x2 (4-bit)"),
                      (E4M3_GRID, "E4M3 (8-bit)")]:
        ar = grid.arity
        vals = torch.tensor([grid.vector(c) for c in range(grid.size)],
                            dtype=torch.float32)
        sigma = float(vals.abs().max()) / 6.0
        torch.manual_seed(0)
        x = torch.randn(a.samples, ar) * sigma
        print(f"\n{tag}  (sigma {sigma:.4g})")
        print(f"  {'k':>5} {'A anchors':>10} {'B best-subset':>14} {'C free Lloyd':>13}"
              f" {'A/B sel':>8} {'B/C val':>8}")
        for rate in [3, 4, 5]:
            k = alphabet_size(rate, grid.rate_cap)
            anchors = vals[torch.tensor(build_forest(rate, grid=grid).blocks)[:, 0]]
            A, B, C = rms(x, anchors), rms(x, kmedoids_on_grid(x, vals, k)), rms(x, lloyd(x, k))
            print(f"  {k:5d} {A/sigma:10.5f} {B/sigma:14.5f} {C/sigma:13.5f}"
                  f" {A/B:7.2f}x {B/C:7.2f}x")
            rows.append({"grid": grid.name, "rung": rate, "k": k,
                         "anchors": A / sigma, "best_subset": B / sigma,
                         "free_lloyd": C / sigma,
                         "selection_penalty": A / B, "value_penalty": B / C})
        # And the full alphabet: is the grid itself a good 256-point quantizer?
        full, cl = rms(x, vals), rms(x, lloyd(x, grid.size))
        print(f"  {grid.size:5d} {full/sigma:10.5f} {'(is the grid)':>14}"
              f" {cl/sigma:13.5f} {'-':>8} {full/cl:7.2f}x   <- whole grid")
        rows.append({"grid": grid.name, "rung": "full", "k": grid.size,
                     "anchors": full / sigma, "best_subset": full / sigma,
                     "free_lloyd": cl / sigma, "selection_penalty": 1.0,
                     "value_penalty": full / cl})
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
