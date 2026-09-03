#!/usr/bin/env python
"""Read issue #35's sweep JSON into the three tables the verdict needs.

``ldlq_window_sweep.py --gauss-seidel --drift-control`` writes every number;
this prints the three views of them a reader has to see together:

1. **the drift control** -- the same encode first and last, per unit.  It is
   the noise floor, and an arm-to-arm gap under it is not a result.
2. **the per-unit table** -- ``out`` and ``hfit`` for the control, the Jacobi
   full-H arm and the Gauss-Seidel one.  The six units are one per role except
   ``down_proj``, which appears twice, so "per role" here is per unit; the
   dense 4-bit residual is known to live in ``q_proj`` and ``k_proj``
   (``tessera-bf16-gauge-and-dense4-residual-2026-09-02``) and a geomean is
   exactly the statistic that can hide what happened to them.
3. **the three-leg decomposition** -- from the refit's own diagnostic sink,
   pass by pass: how much of the error the STEP removes, how much the
   non-positive-target REVERT puts back, and how much the sixteen-entry
   LANDING puts back.  #35 is a question about the step alone, and the landed
   number cannot answer it.  Passes are compared at equal index only: the
   trellis moves codes between them, so pass 3's cost is not pass 1's scale.
"""
from __future__ import annotations

import argparse
import json
import math


def geo(units, arm, field):
    return math.exp(sum(math.log(units[u][arm][field]) for u in units) / len(units))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    units = d["units"]
    names = list(units)
    arms = sorted(set.intersection(*[{k for k in v if not k.startswith("_")}
                                     for v in units.values()]))
    ctl_first = next(x for x in arms if x.startswith("drift control FIRST"))
    ctl_last = next(x for x in arms if x.startswith("drift control LAST"))
    jac = next(x for x in arms if x.startswith("LDLQ") and x.endswith("refit full-H"))
    gs = next(x for x in arms if x.startswith("LDLQ") and x.endswith("(Gauss-Seidel)"))

    print("== 1. drift control -- the same encode, first arm and last arm, one process")
    print(f"{'unit':<38} {'out first':>10} {'out last':>10} {'delta':>9} "
          f"{'hfit first':>10} {'hfit last':>10} {'delta':>9}  bytes")
    for u in names:
        f, l = units[u][ctl_first], units[u][ctl_last]
        same = "IDENTICAL" if f["sha256"] == l["sha256"] else "DIFFER"
        print(f"{u.replace('model.layers.', 'L'):<38} {f['out']:10.5f} {l['out']:10.5f} "
              f"{l['out'] / f['out'] - 1:+8.4%} {f['hfit']:10.5f} {l['hfit']:10.5f} "
              f"{l['hfit'] / f['hfit'] - 1:+8.4%}  {same}")
    print(f"{'GEOMEAN':<38} {geo(units, ctl_first, 'out'):10.5f} "
          f"{geo(units, ctl_last, 'out'):10.5f} "
          f"{geo(units, ctl_last, 'out') / geo(units, ctl_first, 'out') - 1:+8.4%}")

    print("\n== 2. per unit -- control (served h^1.0 default), Jacobi full-H, Gauss-Seidel")
    print(f"{'unit':<30} {'ctl out':>9} {'jac out':>9} {'gs out':>9} {'gs/ctl':>8} "
          f"{'gs/jac':>8} | {'ctl hfit':>9} {'jac hfit':>9} {'gs hfit':>9} {'gs/jac':>8}")
    for u in names:
        c, j, g = units[u][ctl_first], units[u][jac], units[u][gs]
        print(f"{u.replace('model.layers.', 'L'):<30} {c['out']:9.5f} {j['out']:9.5f} "
              f"{g['out']:9.5f} {g['out'] / c['out']:8.4f} {g['out'] / j['out']:8.4f} | "
              f"{c['hfit']:9.5f} {j['hfit']:9.5f} {g['hfit']:9.5f} "
              f"{g['hfit'] / j['hfit']:8.4f}")
    gc, gj, gg = (geo(units, x, "out") for x in (ctl_first, jac, gs))
    hc, hj, hg = (geo(units, x, "hfit") for x in (ctl_first, jac, gs))
    print(f"{'GEOMEAN':<30} {gc:9.5f} {gj:9.5f} {gg:9.5f} {gg / gc:8.4f} {gg / gj:8.4f} | "
          f"{hc:9.5f} {hj:9.5f} {hg:9.5f} {hg / hj:8.4f}")

    print("\n== 3. where each refit pass's error goes -- step, revert, landing")
    print("   fractions of the pass's starting cost; 'survives' is the share of the")
    print("   step's own gain still there after the revert and the landing.")
    print(f"{'unit':<26} {'arm':<7} {'p':>2} {'step':>8} {'revert':>8} {'landing':>8} "
          f"{'survives':>9} {'rev':>5} {'candidate':>10}")
    for u in names:
        for label, arm in (("jacobi", jac), ("gs", gs)):
            for i, r in enumerate(units[u][arm].get("refit", [])):
                b, s, c, ld = r["before"], r["stepped"], r["continuous"], r["landed"]
                gain = b - s
                print(f"{u.replace('model.layers.', 'L'):<26} {label:<7} {i:2d} "
                      f"{(b - s) / b:8.4%} {(c - s) / b:8.4%} {(ld - c) / b:8.4%} "
                      f"{((b - ld) / gain if gain else float('nan')):9.2%} "
                      f"{r['reverted']:5d} {r['candidate']:>10}")

    if "verdict_issue_35" in d:
        print("\n== 4. the pre-registered verdict")
        print(json.dumps(d["verdict_issue_35"], indent=1))


if __name__ == "__main__":
    main()
