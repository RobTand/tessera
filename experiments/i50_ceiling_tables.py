#!/usr/bin/env python
"""The markdown tables of `tessera-lut-landing-ceiling-2026-09-03.md`, from the
two ceiling JSONs, so the doc's numbers are transcribed by a program.
"""
from __future__ import annotations

import json
import math
import sys

CTL = "control [LDLQ 1.0/32 + refit h^1.0]"
JAC = "LDLQ 1.0/32 + refit full-H (Jacobi)"
GS = "LDLQ 1.0/32 + refit full-H (Gauss-Seidel)"
O = ["free plane (continuous)", "free per-block E4M3",
     "oracle assign (own 16)", "oracle table+assign (16)"]
SH = ["free", "free-e4m3", "oracle-assign", "oracle-table"]


def geo(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def short(u):
    p = u.split(".")
    if u.startswith("model.layers"):
        return f"L{p[2]}.{p[-1]}"
    return u


def table(units, arm, field):
    out = [f"| unit | landed | " + " | ".join(SH) + " |",
           "|---|---|" + "---|" * len(SH)]
    cols = {o: [] for o in O}
    landed = []
    for u, r in units.items():
        b = r[arm][field]
        landed.append(b)
        cells = []
        for o in O:
            k = f"  {o:<24} [{arm}]"
            v = r[k][field]
            cols[o].append(v)
            cells.append(f"{v:.5f} ({v / b:.3f}x)")
        out.append(f"| `{short(u)}` | {b:.5f} | " + " | ".join(cells) + " |")
    gl = geo(landed)
    out.append(f"| **geomean** | **{gl:.5f}** | " + " | ".join(
        f"**{geo(cols[o]):.5f} ({geo(cols[o]) / gl:.3f}x)**" for o in O) + " |")
    return "\n".join(out), gl, {o: geo(cols[o]) for o in O}


def main(path, label):
    d = json.load(open(path))
    units = d["units"]
    print(f"\n### {label} -- {len(units)} units\n")
    for arm, name in ((CTL, "the SERVED default, `h^1.0` (`export.py:187`)"),
                      (JAC, "full-H refit, Jacobi step"),
                      (GS, "full-H refit, Gauss-Seidel step (#35's promoted arm)")):
        if arm not in next(iter(units.values())):
            continue
        t, gl, g = table(units, arm, "out")
        print(f"**{name}** -- out-space, held-out rows\n\n{t}\n")
        th, glh, gh = table(units, arm, "hfit")
        print(f"the same on `hfit`\n\n{th}\n")
        for f2, nm in (("plain", "plain weight error"), ("hweighted", "diagonal-h weighted")):
            t2, g2, gg = table(units, arm, f2)
            print(f"{nm}: landed {g2:.5f} -> " + ", ".join(
                f"{s} {gg[o]:.5f} ({gg[o] / g2:.3f}x)" for s, o in zip(SH, O)) + "\n")
    print("\ndrift control:")
    for u, r in units.items():
        print(f"  `{short(u)}` bytes "
              f"{'IDENTICAL' if r['_drift']['bytes_identical'] else 'DIFFER'}, "
              f"out {r['_drift']['out_first']:.6f} -> {r['_drift']['out_last']:.6f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
