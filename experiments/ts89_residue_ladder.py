"""What ``_default_sigma``'s forty rungs actually search, for #89.

The channel plane's per-grid constant is chosen by walking a quarter-binade
ladder ``peak * 2^(-k/4)``, forty rungs, and keeping the spread whose scalar
nearest-value error on a unit Gaussian is smallest.  #89 asks whether that
objective is the right one for a window body.  Before answering that, this
asks a cheaper question: **how many distinct things does the ladder try?**

The answer is four.  The encoder is exactly gauge-equivalent under
``sigma -> 2^k sigma`` (every row's pre-fp16 scale is proportional to
``1/sigma``, ``channel_global`` is a power of two, and the E4M3 grid is closed
under doubling away from its floor and peak), and so is this objective -- the
error and the spread scale together.  So the only free parameter is the dyadic
residue ``log2(sigma) mod 1``, the ladder's quarter-binade step visits exactly
four of them, and its forty rungs are ten copies each.

CPU only, seconds.  Writes JSON beside its ``--out``.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import torch

from tessera.alphabet import E4M3_GRID, E2M1_GRID, BF16_GRID
from tessera.scale_channel import GAUSSIAN_SOURCE, default_channel_sigma

GRIDS = {"E4M3": E4M3_GRID, "E2M1": E2M1_GRID, "BF16": BF16_GRID}


def rungs(grid) -> list[dict]:
    """``_default_sigma``'s own loop, reporting every rung instead of the argmin."""
    scalar = torch.tensor(sorted(set(grid.values)), dtype=torch.float64)
    peak = float(scalar.abs().max())
    sample = torch.tensor(GAUSSIAN_SOURCE(1 << 12, 1.0), dtype=torch.float64)
    out = []
    for k in range(40):
        sigma = peak * 2.0 ** (-k / 4)
        err = ((sample * sigma).unsqueeze(1) - scalar.unsqueeze(0)).abs().min(dim=1).values
        out.append({"k": k, "sigma": sigma,
                    "residue": round(math.log2(sigma) % 1, 6),
                    "rel": float((err * err).mean() / (sigma * sigma))})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    doc = {"stage": "residue_ladder", "grids": {}}
    lines = []
    for name, grid in GRIDS.items():
        rows = rungs(grid)
        best = min(rows, key=lambda r: r["rel"])
        by = collections.defaultdict(list)
        for r in rows:
            by[r["residue"]].append(r)
        classes = []
        for res in sorted(by):
            v = by[res]
            floor_ = min(v, key=lambda r: r["rel"])
            # Rungs that are gauge copies of the class winner: same objective
            # to the last bit, which is the invariance stated above.
            exact = sum(1 for r in v if r["rel"] == floor_["rel"])
            classes.append({"residue": res, "best_rel": floor_["rel"],
                            "best_k": floor_["k"], "best_sigma": floor_["sigma"],
                            "rungs": len(v), "gauge_copies": exact})
        classes.sort(key=lambda c: c["best_rel"])
        doc["grids"][name] = {
            "peak": float(max(abs(x) for x in grid.values)),
            "chosen_sigma": default_channel_sigma(grid),
            "chosen_residue": round(math.log2(default_channel_sigma(grid)) % 1, 6),
            "rungs": rows, "residue_classes": classes,
        }
        lines.append(f"\n== {name}  chosen sigma {best['sigma']:.4f} "
                     f"(k={best['k']}, residue {best['residue']})")
        lines.append(f"    {'residue':>9}{'best rel':>14}{'vs winner':>11}"
                     f"{'best k':>8}{'best sigma':>12}{'gauge copies':>14}")
        for c in classes:
            lines.append(f"    {c['residue']:>9.6f}{c['best_rel']:>14.6e}"
                         f"{c['best_rel'] / classes[0]['best_rel'] - 1:>10.3%}"
                         f"{c['best_k']:>8d}{c['best_sigma']:>12.4f}"
                         f"{c['gauge_copies']:>10d}/{c['rungs']}")
    text = "\n".join(lines)
    print(text)
    out = Path(a.out)
    out.write_text(json.dumps(doc, indent=1))
    out.with_suffix(".log").write_text(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
