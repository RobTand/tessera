"""Which table statistic orders the reach bands?  (CPU, no GPU, no encode.)

``ts89_dyadic_reach.py --stage ladder`` measured h as a step function of the
window table's **snapped reach**: flat to ~1% inside a band, stepping between
bands, minimum at the default's 384.  That is a mechanism you can predict with
-- given sigma, snap the reach and read the band, which gets every unclamped
arm of #89's table right -- but it is not an *explanation*, and the candidate
explanation the ladder printed in its ``predict`` column (relative table
resolution ``ulp/reach``, normalised to the default) is falsified by its own
rows: it holds to 2.5% across bands 288..384 and then **inverts**, calling 448
the best band where it measures as the second worst.

So this replays the table build alone, at the ladder's sigmas, and scores the
obvious candidates against the measured ordering.  Nothing here encodes: the
question is whether a property of the *alphabet* orders the bands, or whether
the ordering only exists once the alternation runs on a weight matrix.

A predictor that fails here is worth recording.  It is the difference between
"we know which band is best" -- measured, and enough to read #89's table --
and "we know why", which this report does not claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tessera.encode import (E4M3_GRID, _window_points_cpu, grid_vector_table,
                            window_table, window_table_reach)

WINDOW_BITS = 14
LADDER = [63.0, 65.0, 67.0, 68.5, 70.63529888131201, 74.5, 77.0, 79.5, 83.0,
          85.5, 88.0, 90.0, 92.5, 94.18039850841602, 99.0, 100.5, 103.0,
          106.5, 109.0, 110.5, 111.5]

# lower-is-better predictors, then the ones whose sign has to be flipped
DIRECT = ("snap_rel", "snap_rel_outer", "max_rel_snap", "saturated",
          "ulp_over_reach")
FLIPPED = ("reach", "distinct", "distinct_outer", "innermost_over_floor")


def stats(sigma: float) -> dict:
    ideal = _window_points_cpu(E4M3_GRID, WINDOW_BITS, sigma, 0, 16).double()
    codes = window_table(E4M3_GRID, WINDOW_BITS, sigma=sigma, seed=0, half=16)
    val = grid_vector_table(E4M3_GRID)[codes.long()].double().reshape(ideal.shape)
    reach = window_table_reach(E4M3_GRID, WINDOW_BITS, sigma=sigma, seed=0, half=16)
    err = val - ideal
    outer = ideal.abs() > 2.0 * sigma
    floor = min(abs(v) for v in E4M3_GRID.values if v != 0)
    nz = ideal.abs()[ideal.abs() > 0]
    return {
        "sigma": sigma,
        "reach": reach.realised,
        "requested": reach.requested,
        "delivered": reach.delivered,
        "saturated": reach.saturated,
        "ulp_over_reach": 32.0 / reach.realised,
        "snap_rel": float((err * err).sum() / (ideal * ideal).sum()),
        "snap_rel_outer": float((err[outer] ** 2).sum() / (ideal[outer] ** 2).sum()),
        "n_outer": int(outer.sum()),
        "distinct": int(torch.unique(val).numel()),
        "distinct_outer": int(torch.unique(val[outer]).numel()),
        "max_rel_snap": float((err.abs() / ideal.abs().clamp_min(1e-30)).max()),
        "innermost_over_floor": float(nz.min() / floor),
    }


def _ranks(xs: list[float]) -> list[float]:
    out = [0.0] * len(xs)
    for rank, i in enumerate(sorted(range(len(xs)), key=lambda i: xs[i])):
        out[i] = float(rank)
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ladder-json", required=True,
                   help="ts89_ladder.json, for the measured h ratios")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    doc = json.loads(Path(a.ladder_json).read_text())
    measured = {round(float(r["channel_sigma"]), 4): float(r["h_ratio"])
                for r in doc.get("arms", {}).values()
                if isinstance(r, dict) and "h_ratio" in r}

    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    log(f"{'sigma':>10}{'reach':>7}{'meas':>7}{'ulp/r':>8}{'snap_rel':>11}"
        f"{'snap_out':>11}{'maxsnap':>9}{'distinct':>9}{'d_out':>7}"
        f"{'sat':>5}{'in/floor':>9}")
    rows = []
    for sigma in LADDER:
        s = stats(sigma)
        s["measured"] = measured.get(round(sigma, 4))
        rows.append(s)
        m = s["measured"]
        log(f"{sigma:>10.4f}{s['reach']:>7.0f}"
            f"{(f'{m:.3f}' if m is not None else '-'):>7}"
            f"{s['ulp_over_reach']:>8.4f}{s['snap_rel']:>11.3e}"
            f"{s['snap_rel_outer']:>11.3e}{s['max_rel_snap']:>9.4f}"
            f"{s['distinct']:>9}{s['distinct_outer']:>7}{s['saturated']:>5}"
            f"{s['innermost_over_floor']:>9.2f}")

    have = [r for r in rows if r["measured"] is not None]
    if len(have) >= 3:
        ys = [r["measured"] for r in have]
        log(f"\nSpearman against the measured h ratio over {len(have)} sigmas "
            "(a predictor that orders the bands reads near +1):")
        for key in DIRECT:
            log(f"    {key:<24}{_spearman([r[key] for r in have], ys):>8.3f}")
        for key in FLIPPED:
            log(f"    -{key:<23}{_spearman([-r[key] for r in have], ys):>8.3f}")

    Path(a.out).write_text(json.dumps({"rows": rows}, indent=1))
    Path(a.out).with_suffix(".log").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
