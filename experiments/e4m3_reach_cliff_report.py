#!/usr/bin/env python
"""Tables for ``docs/measurements/tessera-e4m3-reach-cliff-2026-09-03.md``.

Reads the JSON ``e4m3_reach_cliff.py`` writes and prints the receipt's tables
as markdown, so every number in the receipt is a re-runnable projection of
the raw arms::

    python experiments/e4m3_reach_cliff_report.py /mnt/shared/tessera-runs/e4m3/reach_cliff
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

AXES = ("wt", "h", "out")


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def arms_of(unit: dict, prefix: str):
    return {k: v for k, v in unit.items()
            if isinstance(v, dict) and "bpp" in v and k.startswith(prefix) and "[repeat]" not in k}


def tracked_table(doc):
    units = doc["units"]
    names = list(units)
    sigma0, ceiling = doc["sigma0"], doc["ceiling_sigma"]
    print(f"\n### tracked: `window_sigma = channel_sigma = sigma0 * 2^(k/4)`, R=4, L={doc['args']['window_bits']}, "
          f"{len(names)} units\n")
    print(f"sigma0 = {sigma0:.4f}; ceiling 448 / z_max = {ceiling:.3f} = sigma0 * 2^({math.log2(ceiling / sigma0):.3f}); "
          f"shipped reach {doc['shipped_reach_rms']:.4f} row-RMS.\n")
    print("| k | sigma | sigma/ceiling | reach (row-RMS) | rows over | table saturated | distinct | wt / shipped | h / shipped | tensor-identical |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    ks = sorted({v["k"] for u in units.values() for v in u.values() if isinstance(v, dict) and "k" in v})
    for k in ks:
        rows = []
        for n in names:
            for key, v in units[n].items():
                if isinstance(v, dict) and v.get("k") == k and "[repeat]" not in key:
                    rows.append((n, v))
        if len(rows) != len(names):
            continue
        ship = {n: next(v for key, v in units[n].items() if isinstance(v, dict) and v.get("k") == 0 and "[repeat]" not in key) for n in names}
        s = rows[0][1]["channel_sigma"]
        ident = sum(1 for n, v in rows if v["tsha"] == ship[n]["tsha"])
        print(f"| {k:+d} | {s:.4g} | {s / ceiling:.3f} | {geomean([v['reach_rms'] for _, v in rows]):.3f} | "
              f"{sum(v['over'] for _, v in rows) / len(rows):.3f} | {sum(v['saturated'] for _, v in rows) / len(rows):.5f} | "
              f"{min(v['distinct'] for _, v in rows)}-{max(v['distinct'] for _, v in rows)} | "
              f"{geomean([v['wt'] / ship[n]['wt'] for n, v in rows]):.4f} | {geomean([v['h'] / ship[n]['h'] for n, v in rows]):.4f} | "
              f"{ident}/{len(rows)} |")
    for n in names:
        c = units[n].get("control", {})
        print(f"\n- {n}: control bytes {'IDENTICAL' if c.get('bytes_identical') else 'DIFFER'}, tensor "
              f"{'IDENTICAL' if c.get('tensor_identical') else 'DIFFER'}; bytes across arms: {units[n].get('bytes')}")


def reach_table(doc, title):
    units = doc["units"]
    names = list(units)
    rungs = doc["args"]["rungs"]
    print(f"\n### {title}: table pinned at sigma0, rows moved; {len(names)} units\n")
    for q in rungs:
        R = q / 256
        keys = None
        per = {}
        for n in names:
            arms = arms_of(units[n], f"R{q} ")
            if not arms:
                continue
            per[n] = arms
        if len(per) != len(names):
            print(f"\n(R={R:g}: {len(per)}/{len(names)} units done)")
            if not per:
                continue
        names_q = list(per)
        ship = {n: per[n][f"R{q} shipped (r=4.08) cs={doc['sigma0']:.4g}"] for n in names_q}
        labels = list(per[names_q[0]])
        has_out = all("out" in per[n][l] for n in names_q for l in labels if l in per[n])
        print(f"\n**R = {R:g} (q{q})** -- geomean over {len(names_q)} units, every arm at identical bytes per unit\n")
        cols = "| reach (row-RMS) | rows over | wt / shipped | h / shipped |" + (" out / shipped |" if has_out else "")
        print(cols); print("|---:|---:|---:|---:|" + ("---:|" if has_out else ""))
        rows = []
        for l in labels:
            vs = [(n, per[n][l]) for n in names_q if l in per[n]]
            if len(vs) != len(names_q):
                continue
            r = vs[0][1]["reach_rms"]
            row = {"r": r, "over": sum(v["over"] for _, v in vs) / len(vs)}
            for ax in AXES:
                if all(ax in v for _, v in vs):
                    row[ax] = geomean([v[ax] / ship[n][ax] for n, v in vs])
            row["same_as_prev"] = None
            rows.append(row)
        rows.sort(key=lambda r: r["r"])
        for row in rows:
            print(f"| {row['r']:.3f} | {row['over']:.3f} | {row.get('wt', float('nan')):.4f} | {row.get('h', float('nan')):.4f} |"
                  + (f" {row.get('out', float('nan')):.4f} |" if has_out else ""))
        # the all-rows-clamped plateau: arms whose bytes are identical
        for n in names_q:
            shas = {}
            for l, v in per[n].items():
                shas.setdefault(v["sha"], []).append(round(v["reach_rms"], 2))
            dup = {s: rs for s, rs in shas.items() if len(rs) > 1}
            if dup:
                print(f"\n- {n}: byte-identical arms at reach {sorted(list(dup.values())[0])} (every row clamped: `over` = 1)")
        for n in names_q:
            o = units[n].get(f"R{q}_optimum", {})
            parts = []
            for ax in AXES:
                if ax in o and o[ax].get("reach"):
                    parts.append(f"{ax} {o[ax]['reach']:.2f} ({o[ax]['reach'] / o['shipped_reach']:.3f}x{', EDGE' if o[ax]['edge'] else ''})")
            c = units[n].get(f"R{q}_control", {})
            print(f"- {n}: optimum reach " + "; ".join(parts) + f"; control {'IDENTICAL' if c.get('tensor_identical') else 'DIFFER'}")
    law = doc.get("rung_law", {})
    if law:
        print("\n**Optimum reach per rung against `sqrt(R/4)` from the shipped 4.077** (parabola through the "
              "discrete minimum and its neighbours in log-log; EDGE = argmin on the grid's edge)\n")
        print("| R | axis | optimum (geomean) | per unit | edges | sqrt-law | optimum / law |")
        print("|---:|:---|---:|:---|---:|---:|---:|")
        for rung, blk in law.items():
            for ax, v in blk.items():
                print(f"| {int(rung[1:]) / 256:g} | {ax} | {v['reach_geomean']:.3f} | {', '.join(f'{x:.2f}' for x in v['per_unit'])} | "
                      f"{v['edges']} | {v['sqrt_law_from_shipped']:.3f} | {v['reach_geomean'] / v['sqrt_law_from_shipped']:.3f} |")


def overlay_table(doc):
    print(f"\n### overlay: tracked (table rebuilt at sigma, top pinned at 448) vs pinned (shipped table) at equal reach; {len(doc['units'])} units\n")
    print("| multiplier | reach (row-RMS) | rows over (tracked / pinned) | wt tracked/pinned | h tracked/pinned |")
    print("|---:|---:|:---|---:|---:|")
    ms = sorted({p["multiplier"] for u in doc["units"].values() for p in u.get("pairs", [])})
    for m in ms:
        ps = [p for u in doc["units"].values() for p in u.get("pairs", []) if p["multiplier"] == m]
        print(f"| x{m:g} | {geomean([p['reach_rms'] for p in ps]):.3f} | "
              f"{sum(p['tracked_over'] for p in ps) / len(ps):.3f} / {sum(p['pinned_over'] for p in ps) / len(ps):.3f} | "
              f"{geomean([p['wt_tracked_over_pinned'] for p in ps]):.4f} | {geomean([p['h_tracked_over_pinned'] for p in ps]):.4f} |")
    for n, u in doc["units"].items():
        c = u.get("control", {})
        print(f"- {n}: control {'IDENTICAL' if c.get('tensor_identical') else 'DIFFER'}; bytes {u.get('bytes')}")


def main():
    d = Path(sys.argv[1])
    t = load(d, "tracked")
    if t:
        tracked_table(t)
    for name, title in (("reach_wo", "reach, weights-only (the issue's instrument)"),
                        ("reach_prod", "reach, production encode (LDLQ + full-H refit; `out` on held-out rows)")):
        doc = load(d, name)
        if doc:
            reach_table(doc, title)
    o = load(d, "overlay")
    if o:
        overlay_table(o)


if __name__ == "__main__":
    main()
