#!/usr/bin/env python
"""Read the LANDING leg straight out of the refit's own diagnostics.

The held-out ``out`` geomean is what promotes, but it answers the question
through the whole encode.  ``refit_diagnostics`` records the three legs of
every refit call separately, and the third one --

    continuous -> landed

-- is *exactly* the mechanism issue #50 is about: ``_fit_lut``'s separable
model plus its nearest-in-linear assignment, against the continuous per-block
optimum the refit had already computed.  Under ``landing="grid"`` the same leg
is the sixteen-entry budget removed and the value set kept; under ``"none"``
it is identically 1.0 by construction.

So the split "how much is the solver, how much is the sixteen numbers" is
readable in the fit's own quadratic, per pass, with no held-out noise and no
trellis alternation in the way.  It is not a substitute for the ``out``
geomean -- a better fit cost is not a better unit -- but it is the direct
reading of the mechanism, and the two disagreeing is itself the finding.

    python experiments/lut_landing_split.py \
        experiments/results/tessera_lut_landing_ceiling.json \
        [experiments/results/tessera_lut_exact_fit.json]
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
    """Per-refit-call ratios for the three legs, in the fit's own cost."""
    step, revert, landing = [], [], []
    for d in arm.get("refit", ()):
        b, s, c, l = d["before"], d["stepped"], d["continuous"], d["landed"]
        if b > 0:
            step.append(s / b)
        if s > 0:
            revert.append(c / s)
        if c > 0:
            landing.append(l / c)
    return step, revert, landing


def table(path: Path):
    doc = json.loads(path.read_text())
    units = doc["units"]
    per_arm: dict = {}
    for _name, arms in units.items():
        for arm_name, arm in arms.items():
            per_arm.setdefault(arm_name, []).append(arm)
    print(f"\n=== {path.name}   ({len(units)} units)")
    print(f"{'arm':<52} {'landing':>9} {'step':>8} {'revert':>8} {'out':>9} "
          f"{'passes':>7}")
    for name, arms in per_arm.items():
        st, rv, ld = [], [], []
        for a in arms:
            s_, r_, l_ = legs(a)
            st += s_
            rv += r_
            ld += l_
        out = geomean([a["out"] for a in arms])
        print(f"{name:<52} {geomean(ld):>9.5f} {geomean(st):>8.5f} "
              f"{geomean(rv):>8.5f} {out:>9.5f} {len(ld):>7d}")
    return per_arm


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        table(Path(p))


if __name__ == "__main__":
    main()
