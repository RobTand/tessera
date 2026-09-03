#!/usr/bin/env python
"""Read issue #50's ceiling run into the four tables the verdict needs.

``lut_landing_ceiling.py`` writes every number; this prints them in the order
a reader has to see them:

1. **the identity check** -- on every ``table`` arm, the sink's reconstruction
   against ``stock_dequant`` of the same unit.  A ceiling arm can only be
   scored off the sink, so if the wire arms do not agree with it there is no
   licence to read anything below.
2. **the drift control** -- the served default first and last, one process.
   An arm-to-arm gap under the control's own spread is not a result.
3. **the ceiling** -- per unit and geomean, for each refit objective:
   ``table`` (the wire), ``grid`` (every in-range E4M3 value, so the
   sixteen-entry budget removed and the E4M3 alphabet kept) and ``none``
   (continuous per-block scales).  ``none/table`` is the most any table fit
   could return; ``grid/table`` is what removing the entry budget alone did.
4. **the within-pass legs** -- the same question read off ``refit_diagnostics``
   on the ``table`` arms, which needs no ceiling run at all.  The last pass's
   ``continuous`` and ``landed`` are the plane the unit ends on with and
   without the landing, at FROZEN codes, so ``sqrt(continuous/landed)`` is the
   codes-frozen ``hfit`` ceiling and the end-to-end number above is the same
   quantity with the trellis allowed to re-adapt.
"""
from __future__ import annotations

import argparse
import json
import math


def geo(units, arm, field):
    return math.exp(sum(math.log(units[u][arm][field]) for u in units) / len(units))


def short(u):
    return u.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    units = d["units"]
    names = list(units)
    arms = list(units[names[0]])

    print("== 1. the sink IS the wire on every table arm")
    print(f"{'unit':<22} {'arm':<52} {'max abs':>10} {'rel':>10}  bit-identical")
    worst = 0.0
    for u in names:
        for arm, r in units[u].items():
            if r["landing"] != "table":
                continue
            worst = max(worst, r["sink_vs_wire_rel"])
            print(f"{short(u):<22} {arm[:52]:<52} {r['sink_vs_wire_max_abs']:10.3e} "
                  f"{r['sink_vs_wire_rel']:10.3e}  {r['sink_vs_wire_bit_identical']}")
    print(f"worst relative disagreement over all table arms: {worst:.3e}")

    ctl_first = next(x for x in arms if x.startswith("drift control FIRST"))
    ctl_last = next(x for x in arms if x.startswith("drift control LAST"))
    print("\n== 2. drift control -- the served default, first arm and last arm")
    print(f"{'unit':<22} {'out first':>10} {'out last':>10} {'delta':>9} "
          f"{'hfit first':>10} {'hfit last':>10} {'delta':>9}  recon")
    for u in names:
        f, l = units[u][ctl_first], units[u][ctl_last]
        same = "IDENTICAL" if f["sha256"] == l["sha256"] else "DIFFERS"
        print(f"{short(u):<22} {f['out']:10.5f} {l['out']:10.5f} "
              f"{l['out'] / f['out'] - 1:+8.4%} {f['hfit']:10.5f} {l['hfit']:10.5f} "
              f"{l['hfit'] / f['hfit'] - 1:+8.4%}  {same}")

    groups = []
    for arm in arms:
        if arm.startswith("drift control LAST") or "| landing=" in arm:
            continue
        base = ctl_first if arm.startswith("drift control FIRST") else arm
        label = arm[len("drift control FIRST ["):-1] if base is ctl_first else arm
        gr = {"table": arm}
        for landing in ("grid", "none"):
            cand = f"{label} | landing={landing}"
            if cand in arms:
                gr[landing] = cand
        groups.append((label, gr))

    print("\n== 3. the ceiling, end to end -- out (held out) and hfit (fit rows)")
    for label, gr in groups:
        print(f"\n-- {label}")
        head = f"{'unit':<22}"
        for k in ("table", "grid", "none"):
            if k in gr:
                head += f" {'out ' + k:>10}"
        for k in ("grid", "none"):
            if k in gr:
                head += f" {k + '/table':>11}"
        head += f" {'hfit table':>11} {'hfit none':>11} {'none/table':>11}"
        print(head)
        ratios = {k: [] for k in ("grid", "none")}
        hratio = []
        for u in names:
            row = f"{short(u):<22}"
            for k in ("table", "grid", "none"):
                if k in gr:
                    row += f" {units[u][gr[k]]['out']:10.5f}"
            for k in ("grid", "none"):
                if k in gr:
                    r = units[u][gr[k]]["out"] / units[u][gr["table"]]["out"]
                    ratios[k].append(r)
                    row += f" {r:11.4f}"
            ht = units[u][gr["table"]]["hfit"]
            hn = units[u][gr["none"]]["hfit"]
            hratio.append(hn / ht)
            row += f" {ht:11.5f} {hn:11.5f} {hn / ht:11.4f}"
            print(row)
        row = f"{'GEOMEAN':<22}"
        for k in ("table", "grid", "none"):
            if k in gr:
                row += f" {geo(units, gr[k], 'out'):10.5f}"
        for k in ("grid", "none"):
            if k in gr:
                row += f" {geo(units, gr[k], 'out') / geo(units, gr['table'], 'out'):11.4f}"
        row += (f" {geo(units, gr['table'], 'hfit'):11.5f}"
                f" {geo(units, gr['none'], 'hfit'):11.5f}"
                f" {geo(units, gr['none'], 'hfit') / geo(units, gr['table'], 'hfit'):11.4f}")
        print(row)

    print("\n== 4. the same ceiling read within the last pass, codes frozen")
    print("   (from refit_diagnostics on the table arms; no ceiling run needed)")
    print(f"{'unit':<22} {'arm':<40} {'hfit landed':>12} {'hfit cont':>10} {'ratio':>8}")
    for label, gr in groups:
        rs = []
        for u in names:
            r = units[u][gr["table"]]
            if not r.get("refit"):
                continue
            last = r["refit"][-1]
            den = last["landed"] / (r["hfit"] ** 2)
            hc = math.sqrt(max(last["continuous"], 0.0) / den)
            rs.append(hc / r["hfit"])
            print(f"{short(u):<22} {label[:40]:<40} {r['hfit']:12.5f} {hc:10.5f} "
                  f"{hc / r['hfit']:8.4f}")
        if rs:
            g = math.exp(sum(math.log(x) for x in rs) / len(rs))
            print(f"{'GEOMEAN':<22} {label[:40]:<40} {'':>12} {'':>10} {g:8.4f}")


if __name__ == "__main__":
    main()
