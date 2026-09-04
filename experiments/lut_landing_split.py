#!/usr/bin/env python
"""Read the LANDING leg straight out of the refit's own diagnostics.

The held-out ``out`` geomean is what promotes, but it answers issue #50's
question through the whole encode -- trellis alternation, held-out noise and
all.  ``refit_diagnostics`` records the three legs of every refit call
separately, and the third one --

    continuous -> landed

-- is *exactly* the mechanism the issue is about: ``_fit_lut``'s separable
model plus its nearest-in-linear assignment, against the continuous per-block
optimum the refit had already computed.

**These costs are negative and must be differenced, never divided.**  The 1-D
metric's cost is ``sum(A*C^2 - 2*B*C)`` -- the quadratic with its constant
dropped -- so it is negative, lower is better, and a *ratio* of two of them is
meaningless.  The quantity with a meaning is the LOSS the landing adds,
``landed - continuous >= 0``, and the share of it a better fit removes:

    removed = (loss_control - loss_arm) / loss_control

which is the fraction of the landing leg the arm buys back **in the fit's own
objective**, with no held-out noise in the way.  Read beside the ``out``
column, it is the direct test of whether that objective is aligned with the
error that ships: an arm that removes landing loss and still raises ``out``
has shown the objective to be the problem, not the optimiser.

    python experiments/lut_landing_split.py \
        experiments/results/tessera_lut_exact_fit.json \
        [experiments/results/tessera_lut_landing_ceiling.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0.0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def legs(arm):
    """Per-refit-call LOSSES, in the fit's own cost.  Differences, not ratios.

    ``step`` is the gain the optimiser's step won (positive is progress),
    ``revert`` and ``landing`` are the losses the two later legs give back.
    """
    step, revert, landing = [], [], []
    for d in arm.get("refit", ()):
        b, s, c, l = d["before"], d["stepped"], d["continuous"], d["landed"]
        step.append(b - s)
        revert.append(c - s)
        landing.append(l - c)
    return step, revert, landing


def table(path: Path):
    doc = json.loads(path.read_text())
    units = doc["units"]
    per_arm: dict = {}
    for _u, arms in units.items():
        for arm_name, arm in arms.items():
            per_arm.setdefault(arm_name, []).append(arm)
    rows = {}
    for name, arms in per_arm.items():
        st = rv = ld = 0.0
        n = 0
        for a in arms:
            s_, r_, l_ = legs(a)
            st += sum(s_)
            rv += sum(r_)
            ld += sum(l_)
            n += len(l_)
        rows[name] = {"step": st, "revert": rv, "landing": ld, "passes": n,
                      "out": geomean([a["out"] for a in arms])}

    print(f"\n=== {path.name}   ({len(units)} units)")
    print("    summed over every refit pass of every unit, in the fit's own cost")
    print(f"\n{'arm':<56} {'step won':>10} {'landing lost':>13} "
          f"{'of step':>8} {'out':>9}")
    for name, r in rows.items():
        frac = r["landing"] / r["step"] if r["step"] else float("nan")
        print(f"{name:<56} {r['step']:>10.2f} {r['landing']:>13.2f} "
              f"{frac:>7.1%} {r['out']:>9.5f}")

    ctl = next((k for k in rows if k.startswith("drift control FIRST")), None)
    if ctl:
        # Only the FIRST pass is a fair comparison.  An arm that changes the
        # table in pass 1 hands pass 2 a different starting plane, so every
        # later pass is measured on inputs the two arms no longer share and
        # the summed columns above are not like for like.  Pass 1 is: both
        # arms reach the same ``continuous``, and the only thing that differs
        # is how sixteen entries were chosen from it.
        print(f"\n    pass 1 only -- the one pass both arms enter with the "
              f"same continuous optimum")
        print(f"\n{'unit':<40} {'control':>10} {'arm':>10} {'removed':>9}")
        for name in rows:
            if name == ctl:
                continue
            tc = ta = 0.0
            ok = True
            print(f"  {name}")
            for u, arms in units.items():
                if name not in arms or not arms[name].get("refit"):
                    ok = False
                    break
                c, a = arms[ctl]["refit"][0], arms[name]["refit"][0]
                if abs(c["continuous"] - a["continuous"]) > 1e-6:
                    ok = False
                    break
                lc, la = c["landed"] - c["continuous"], a["landed"] - a["continuous"]
                tc, ta = tc + lc, ta + la
                share = (lc - la) / lc if lc else float("nan")
                print(f"    {u:<38} {lc:>10.4f} {la:>10.4f} {share:>8.2%}")
            if ok and tc:
                print(f"    {'TOTAL':<38} {tc:>10.4f} {ta:>10.4f} "
                      f"{(tc - ta) / tc:>8.2%}")
            elif not ok:
                print("    -- not comparable pass for pass")

        # A unit whose table is unchanged in pass 1 hands pass 2 an identical
        # plane -- the encode is deterministic, so the two arms are still on
        # shared inputs -- and it may diverge later.  Walk forward to the FIRST
        # pass where the landed costs differ, and report it only if that pass's
        # ``continuous`` still agrees to 1e-6, which is the same like-for-like
        # test pass 1 passes trivially.  Without this a unit reads 0.00% above
        # while its held-out ``out`` moved, and the two look contradictory.
        print(f"\n    first pass whose tables differ, per unit "
              f"(reported only while both arms still share ``continuous``)")
        print(f"\n{'unit':<40} {'pass':>5} {'control':>10} {'arm':>10} "
              f"{'removed':>9}")
        for name in rows:
            if name == ctl:
                continue
            print(f"  {name}")
            for u, arms in units.items():
                if name not in arms or not arms[name].get("refit"):
                    continue
                C, A = arms[ctl]["refit"], arms[name]["refit"]
                hit = None
                for i, (c, a) in enumerate(zip(C, A)):
                    if c["landed"] != a["landed"]:
                        hit = (i, c, a)
                        break
                if hit is None:
                    print(f"    {u:<38} {'--':>5}   tables identical in every "
                          f"pass")
                    continue
                i, c, a = hit
                if abs(c["continuous"] - a["continuous"]) > 1e-6:
                    print(f"    {u:<38} {i + 1:>5}   diverged after the arms "
                          f"stopped sharing inputs -- not comparable")
                    continue
                lc = c["landed"] - c["continuous"]
                la = a["landed"] - a["continuous"]
                share = (lc - la) / lc if lc else float("nan")
                print(f"    {u:<38} {i + 1:>5} {lc:>10.4f} {la:>10.4f} "
                      f"{share:>8.2%}")
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        table(Path(p))


if __name__ == "__main__":
    main()
