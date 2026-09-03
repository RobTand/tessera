#!/usr/bin/env python
"""The tables for issue #50, read out of `lut_landing_ceiling.py`'s JSON.

Three things a geomean cannot say and this prints instead: the per-unit spread,
whether the continuous ceiling on a unit needs a NEGATIVE scale (in which case
it is not a plane any wire could hold and overstates the prize on that unit),
and whether each landed row reproduces the published receipt it is supposed to.
"""
from __future__ import annotations

import json
import math
import sys

CTL = "control [LDLQ 1.0/32 + refit h^1.0]"
JAC = "LDLQ 1.0/32 + refit full-H (Jacobi)"
GS = "LDLQ 1.0/32 + refit full-H (Gauss-Seidel)"
ORACLES = ["free plane (continuous)", "free per-block E4M3",
           "oracle assign (own 16)", "oracle table+assign (16)"]
SHORT = {"free plane (continuous)": "free", "free per-block E4M3": "free-e4m3",
         "oracle assign (own 16)": "oracle-assign",
         "oracle table+assign (16)": "oracle-table"}


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main(path, ref=None, ref_map=None):
    d = json.load(open(path))
    units = d["units"]
    print(f"# {path}\n#   {len(units)} units: {', '.join(units)}")

    ref_d = json.load(open(ref)) if ref else None

    print("\n## drift control and cross-session reproduction")
    print(f"    {'unit':<44} {'bytes':>10} {'ctl out':>10} {'published':>11} {'delta':>9}")
    for u, r in units.items():
        pub = ""
        delta = ""
        if ref_d:
            key = ref_map.get(u, u) if ref_map else u
            block = (ref_d.get("units") or ref_d.get("experts", {})).get(key, {})
            cand = [v["out"] for k, v in block.items()
                    if isinstance(v, dict) and "out" in v and "refit h^1.0" in k
                    and "control" not in k and "drift" not in k]
            if cand:
                pub = f"{min(cand):11.6f}"
                delta = f"{r[CTL]['out'] / min(cand) - 1.0:+8.4%}"
        print(f"    {u:<44} {'IDENTICAL' if r['_drift']['bytes_identical'] else 'DIFFER':>10} "
              f"{r[CTL]['out']:10.6f} {pub:>11} {delta:>9}")

    for arm, tag in ((CTL, "the SERVED default (diagonal h^1.0)"),
                     (JAC, "full-H, Jacobi step"), (GS, "full-H, Gauss-Seidel step")):
        if arm not in next(iter(units.values())):
            continue
        print(f"\n## {tag} -- what the landing costs, per unit (out-space, held-out rows)")
        print(f"    {'unit':<40} {'landed':>9} " +
              " ".join(f"{SHORT[o]:>14}" for o in ORACLES) + f" {'neg':>6}")
        cols = {o: [] for o in ORACLES}
        landed = []
        for u, r in units.items():
            base = r[arm]["out"]
            landed.append(base)
            row = f"    {u.split('.')[-1] if '.' in u else u:<40} {base:9.5f} "
            for o in ORACLES:
                k = f"  {o:<24} [{arm}]"
                if k in r:
                    v = r[k]["out"]
                    cols[o].append(v)
                    row += f" {v:8.5f}/{v / base:.3f}x"
                else:
                    row += f" {'-':>14}"
            neg = r.get(f"  {'free plane (continuous)':<24} [{arm}]", {}).get("negatives", "")
            print(row + f" {neg:>6}")
        gl = geo(landed)
        print(f"    {'GEOMEAN':<40} {gl:9.5f} " + " ".join(
            f" {geo(cols[o]):8.5f}/{geo(cols[o]) / gl:.3f}x" if len(cols[o]) == len(units)
            else f"{'-':>14}" for o in ORACLES))

        print(f"    -- the same, on hfit (the fit-row quadratic the refit is monotone in)")
        colsh = {o: [] for o in ORACLES}
        lh = []
        for u, r in units.items():
            lh.append(r[arm]["hfit"])
            for o in ORACLES:
                k = f"  {o:<24} [{arm}]"
                if k in r:
                    colsh[o].append(r[k]["hfit"])
        gh = geo(lh)
        print(f"    {'GEOMEAN hfit':<40} {gh:9.5f} " + " ".join(
            f" {geo(colsh[o]):8.5f}/{geo(colsh[o]) / gh:.3f}x" if len(colsh[o]) == len(units)
            else f"{'-':>14}" for o in ORACLES))

    print("\n## the continuous solve, certified: exact per-row solve over coordinate descent")
    for arm in (CTL, JAC, GS):
        k = f"  {'free plane (continuous)':<24} [{arm}]"
        rows = [(u, r[k]) for u, r in units.items() if k in r]
        if not rows:
            continue
        print(f"    {arm}")
        for u, v in rows:
            print(f"      {u:<42} sweeps {v.get('sweeps'):>4}  "
                  f"cd {v.get('fit_cost_cd', float('nan')):.6e}  "
                  f"exact {v.get('fit_cost_exact', float('nan')):.6e}  "
                  f"exact/cd {v.get('exact_over_cd', float('nan')):.6f}")

    print("\n## did re-choosing the sixteen ENTRIES add anything over re-assigning them?")
    for arm in (CTL, JAC, GS):
        ka = f"  {'oracle assign (own 16)':<24} [{arm}]"
        kt = f"  {'oracle table+assign (16)':<24} [{arm}]"
        rows = [(u, r[ka], r[kt]) for u, r in units.items() if ka in r and kt in r]
        if not rows:
            continue
        print(f"    {arm}")
        for u, va, vt in rows:
            print(f"      {u:<42} assign hfit {va['hfit']:.6f}  table+assign {vt['hfit']:.6f}  "
                  f"{vt['hfit'] / va['hfit']:.6f}x  rounds {vt.get('rounds')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
         json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
