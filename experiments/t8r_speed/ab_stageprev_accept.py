"""Check the single-rate STAGE_PREV A/B (arms master, fix) against its receipt's acceptance.

Acceptance (docs/measurements/2026-10-01-staged-history-single-rate.md):
* every single-rate (R1024) row, routed and dense: 1.00 in both passes (|ratio - 1| <= TOL_SAME);
* two-run routed (R832, R1088) at M = 2048: at or below about 0.94 of master;
* two-run routed at M = 1 and every two-run dense row: at or below about 0.98;
* every row bitwise in both passes, and no missing case.
With --old OLD_AB_DIR (the #763 A/B, arms master = pre-#763, port = #763), each
two-run row must also restore the pre-#763 time: the product of the fix's mean
ratio and #763's mean ratio within +-TOL_SAME of 1.  That is the criterion for
rows #763 slowed by less than the bars allow for.

Usage: ab_stageprev_accept.py AB_DIR [--old OLD_AB_DIR] [--tol-same 0.02]
Exit status 0 when every check holds, 1 otherwise.
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("ab_dir")
ap.add_argument("--tol-same", type=float, default=0.02)
ap.add_argument("--old", help="the #763 A/B directory (restore check)")
ap.add_argument("--slack", type=float, default=0.01, help="'about' allowance on the 0.94 / 0.98 bars")
a = ap.parse_args()
d = json.load(open(os.path.join(a.ab_dir, "ab_summary.json")))
arm = [x for x in d["arms"] if x != d["ref"]][0]
old = {}
if a.old:
    od = json.load(open(os.path.join(a.old, "ab_summary.json")))
    oarm = [x for x in od["arms"] if x != od["ref"]][0]
    for r in od["rows"]:
        v = [r.get(f"{oarm}_ratio"), r.get(f"{oarm}b_ratio")]
        if None not in v:
            old[(r["family"], r["group"], r["M"])] = sum(v) / 2
fails, lines = [], []
for r in d["rows"]:
    rate = r["group"].split(".")[1]
    two = rate != "R1024"
    rs = [r.get(f"{arm}_ratio"), r.get(f"{arm}b_ratio")]
    tag = f"{r['family']:6s} {r['group']:24s} M={r['M']:<5d} {rs[0]} {rs[1]} bitwise={r['bitwise']}"
    why = []
    if not r["bitwise"]:
        why.append("not bitwise")
    if r["missing"]:
        why.append(f"missing {r['missing']}")
    if None in rs:
        why.append("ratio missing")
    else:
        o = old.get((r["family"], r["group"], r["M"]))
        if o is not None:
            tag += f" restore={sum(rs) / 2 * o:.4f}"
        if not two:
            if any(abs(x - 1.0) > a.tol_same for x in rs):
                why.append(f"single-rate moved beyond +-{a.tol_same}")
        elif o is not None:
            if abs(sum(rs) / 2 * o - 1.0) > a.tol_same:
                why.append("two-run does not restore the pre-#763 time")
        elif r["family"] == "routed" and r["M"] == 2048:
            if any(x > 0.94 + a.slack for x in rs):
                why.append("two-run routed M=2048 above ~0.94")
        elif (r["family"] == "routed" and r["M"] == 1) or r["family"] == "dense":
            if any(x > 0.98 + a.slack for x in rs):
                why.append("two-run above ~0.98")
    lines.append(("FAIL " if why else "ok   ") + tag + ("  <- " + "; ".join(why) if why else ""))
    if why:
        fails.append(tag)
print(f"ref={d['ref']} arm={arm} kernel_sha={json.dumps(d.get('kernel_sha'))[:300]}")
print("\n".join(lines))
print(f"{len(d['rows'])} rows, {len(fails)} failing")
sys.exit(1 if fails else 0)
