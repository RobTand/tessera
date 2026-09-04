"""The sigma that gives one window width another's reach -- computed, not typed.

Issue #18, the one comparison its joint ``(L, ratio)`` grid could not make.
That grid found ``L`` and the window/channel ratio entangled, and named the
reason: at ratio 1 a wider table has more quantiles **and** its outermost
quantile sits further out, so ``L`` moves resolution and reach together and
only ``L`` costs bytes.  Every ``L`` comparison in it therefore prices a
bundle, and no cell says which half of the bundle the winner bought.

This module builds the arms that separate them.  For a target reach -- the
reach some *other* width has at the shipped ratio -- it finds the ratio at
which **this** width realises exactly that reach.  Encoding a width at each
of the three widths' reaches turns the one-dimensional ``L`` axis into a
2-D factorial:

* a **row** is a table entry count (``L``), which costs bytes;
* a **column** is a spread, which costs none -- it is a constant in the
  recipe, and the artifact is the same size at every value of it.

Read across a row and the entry count is fixed: the difference is spread.
Read down a column and the spread is fixed: the difference is entries, paid
for in bytes and byte-matched against the shipped pair by the sweep itself.

**Why a search and not a division.**  A window table's entries are grid
values, so its realised reach is the *snapped* outermost quantile: a step
function of the requested spread, not a linear one.  The naive ratio
``target / reach(L)`` lands on the wrong step about half the time -- at
``L=14`` on BF16 it delivers 3.6875 when 3.671875 was asked for, a 0.4%
miss that would be reported as an exact match by anyone who computed it
that way.  So the ratio is *searched* for, and the realised reach it
delivers is asserted equal to the target before it is returned.

**And the midpoint, not the edge.**  The set of ratios delivering one
target is an interval; its endpoints are one float from delivering the
neighbouring step (measured: ``0.9140110927067978`` gives 3.65625 and
``...79`` gives 3.671875).  A ratio recorded in a receipt and retyped into
a later run has to survive being read back at lower precision, so what is
returned is the interval's midpoint and what is reported beside it is the
interval, which is how much precision the number actually needs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.encode import window_table_reach  # noqa: E402


def realised_reach(grid, window_bits: int, sigma: float, *, seed: int = 0,
                   half: int = 16) -> float:
    """The reach a window table of this width actually delivers at ``sigma``.

    The grid's own value, not the quantile that was asked for: the table is
    built out of grid values, and it is the built table the encoder uses.
    """
    return window_table_reach(grid, int(window_bits), sigma=float(sigma),
                              seed=seed, half=half).realised


def _first_ratio_at_least(grid, window_bits, target, base_sigma, lo, hi,
                          *, strict: bool, seed=0, half=16) -> float:
    """Bisect the monotone step function for its first ratio past ``target``.

    ``realised_reach`` is non-decreasing in sigma -- every requested quantile
    scales with it and nearest-value snapping preserves order -- so the two
    edges of the interval delivering one value are two bisections.
    """
    def over(r: float) -> bool:
        got = realised_reach(grid, window_bits, r * base_sigma, seed=seed, half=half)
        return got > target if strict else got >= target

    if not over(hi):
        raise ValueError(
            f"L={window_bits}: reach {target!r} is above everything the "
            f"bracket reaches (ratio {hi} gives "
            f"{realised_reach(grid, window_bits, hi * base_sigma)!r})")
    if over(lo):
        raise ValueError(
            f"L={window_bits}: reach {target!r} is below the bracket's floor "
            f"(ratio {lo} already gives "
            f"{realised_reach(grid, window_bits, lo * base_sigma)!r})")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:      # the floats have met; the edge is hi
            break
        if over(mid):
            hi = mid
        else:
            lo = mid
    return hi


def matched_ratio(grid, window_bits: int, target_reach: float, *,
                  base_sigma: float, lo: float = 0.125, hi: float = 8.0,
                  seed: int = 0, half: int = 16) -> dict:
    """The ratio at which ``window_bits`` realises exactly ``target_reach``.

    Returns the midpoint of the interval that delivers it, the interval
    itself, and the realised reach -- which is **asserted** equal to the
    target, so a target no step of this width lands on raises instead of
    returning a near miss.
    """
    target = float(target_reach)
    first = _first_ratio_at_least(grid, window_bits, target, base_sigma, lo, hi,
                                  strict=False, seed=seed, half=half)
    past = _first_ratio_at_least(grid, window_bits, target, base_sigma, lo, hi,
                                 strict=True, seed=seed, half=half)
    ratio = 0.5 * (first + past)
    got = realised_reach(grid, window_bits, ratio * base_sigma, seed=seed, half=half)
    if got != target:
        raise ValueError(
            f"L={window_bits}: no ratio in [{lo}, {hi}] realises reach "
            f"{target!r} exactly; the nearest step delivers {got!r}. A width "
            "whose table cannot land on this reach has no matched arm, and a "
            "near miss is not one.")
    return {"window_bits": int(window_bits), "target_reach": target,
            "ratio": ratio, "realised": got,
            "interval": (first, past),
            "width": past - first}


def reach_grid(grid, widths, *, base_sigma: float, seed: int = 0,
               half: int = 16) -> dict:
    """The full factorial: every width, at every width's own shipped reach.

    The diagonal is ratio 1.0 by construction -- a width at its own reach is
    the shipped arm -- and that is asserted here rather than assumed, because
    a diagonal that came back at anything else would mean the target table
    and the encode disagree about what ``sigma`` means.
    """
    widths = [int(L) for L in widths]
    own = {L: realised_reach(grid, L, base_sigma, seed=seed, half=half)
           for L in widths}
    cells = {}
    for L in widths:
        for target_L in widths:
            m = matched_ratio(grid, L, own[target_L], base_sigma=base_sigma,
                              seed=seed, half=half)
            if target_L == L and not (m["interval"][0] <= 1.0 <= m["interval"][1]):
                raise ValueError(
                    f"L={L}: its own reach is delivered by ratios "
                    f"{m['interval']}, which does not contain 1.0")
            cells[(L, target_L)] = m
    return {"own": own, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="bf16")
    ap.add_argument("--widths", type=int, nargs="+", default=[12, 14, 16])
    ap.add_argument("--sigma", type=float, default=None,
                    help="the recipe's channel_sigma; default is BF16's")
    ap.add_argument("--ratios-for", type=int, default=None,
                    help="print only this width's ratio list, in target order, "
                         "for --pair-ratios")
    a = ap.parse_args()

    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.export import BF16_CHANNEL_SIGMA
    grids = {"bf16": BF16_GRID, "e4m3": E4M3_GRID}
    grid = grids[a.grid]
    sigma = BF16_CHANNEL_SIGMA if a.sigma is None else a.sigma

    g = reach_grid(grid, a.widths, base_sigma=sigma)
    if a.ratios_for is not None:
        # The diagonal is spelled exactly 1.0 and not its interval midpoint:
        # ratio 1.0 is the value the recipe stores and the sweep's
        # ``window_sigma=None`` path, so an arm one ULP away from it would be
        # a re-spelling of the shipped path rather than the shipped path.
        print(" ".join(
            repr(1.0 if t == a.ratios_for else g["cells"][(a.ratios_for, t)]["ratio"])
            for t in a.widths))
        return
    print(f"grid={a.grid}  channel_sigma={sigma!r}")
    for L in a.widths:
        print(f"  L={L:<3} own reach {g['own'][L]!r}")
    print("\n  ratio at which the row's width realises the column's reach")
    print("  " + "row L".ljust(8) + "".join(
        f"reach {g['own'][t]!r}".ljust(26) for t in a.widths))
    for L in a.widths:
        cells = [g["cells"][(L, t)] for t in a.widths]
        print("  " + f"L={L}".ljust(8) + "".join(
            f"{c['ratio']:.10f} (+-{c['width'] / 2:.1e})".ljust(26) for c in cells))


if __name__ == "__main__":
    main()
