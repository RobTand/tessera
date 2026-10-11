"""Markdown tables of ``bench_rung_forward.py`` runs, and of their real-wire controls (tessera#694).

Every number in ``docs/measurements/2026-10-11-mixed-rate-head-bench.md`` is a
cell of the tables this prints, so a reader can rerun the reduction on the
receipts instead of trusting a transcription.  A cell above the bar is bold.

    rung_report.py --run A=DIR/bench_rung_forward.json [--run B=...] [--real DIR/bench_t8r.json]
                   [--reference 1024] [--splits 1,8,512]

``--run`` may repeat; the label names the table.  ``--real`` is the
``bench_t8r.py`` run that timed the release artifact's own R1024, R1088 and
R832 stacks in the same job: its ratios sit beside the synthetic ones of the
same rungs, and the ratio of the two stacks' times is the harness's bias.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rung_summary import BAR, summarize_rungs  # noqa: E402

REAL_GROUPS = {1024: "experts.R1024.L10", 1088: "experts.R1088.L11", 832: "experts.R832.L42"}
KEYS = (("wall", "eager forward wall"), ("graph", "graph replay (M <= 8)"), ("kernel", "profiler device time"))


def cell(value, digits=3):
    if value is None:
        return "-"
    text = f"{value:.{digits}f}"
    return f"**{text}**" if digits == 3 and value > BAR else text


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def ratio_table(doc, summary, key, reference):
    rungs = list(doc["rungs"])
    ms = [str(m) for m in doc["meta"]["ms"]]
    rows = []
    for m in ms:
        if not any(key in summary[m].get(q, {}) for q in rungs):
            continue                      # graph replay stops at M 8: no row, no dashes
        row = [m]
        for q in rungs:
            entry = summary[m].get(q, {}).get(key)
            row.append("1" if int(q) == int(reference) and entry else cell(entry and entry["ratio"]))
        rows.append(row)
    worst = ["max"]
    for q in rungs:
        vals = [summary[m][q][key]["ratio"] for m in ms if key in summary[m].get(q, {})]
        worst.append("1" if int(q) == int(reference) and vals else cell(max(vals) if vals else None))
    return table(["M"] + [f"R{q}" for q in rungs], rows + [worst])


def real_cell(record, m, key):
    c = record["cells"].get(str(m))
    if not c:
        return None
    if key == "wall":
        return c["wall"]["median_ms"]
    if key == "graph":
        return c["graph"]["median_ms"] if "graph" in c else None
    return c["profile"]["kernel_us_per_call"] / 1000.0


def synthetic_time(doc, rung, m, key):
    c = doc["rungs"][str(rung)]["cells"].get(str(m))
    if not c:
        return None
    if key in ("wall", "graph"):
        f, r = c.get("F", {}).get(key), c.get("R", {}).get(key)
        return 0.5 * (f["median_ms"] + r["median_ms"]) if f and r else None
    prof = c.get("F", {}).get("profile")
    return prof["kernel_us_per_call"] / 1000.0 if prof else None


def real_table(real, doc, key, reference):
    groups = {r["group"]: r for r in real["results"]}
    ref = groups[REAL_GROUPS[reference]]
    header = ["M", "real R1088 / R1024", "synthetic R1088 / R1024", "real R832 / R1024", "synthetic R832 / R1024",
              "synthetic / real R1024", "synthetic / real R1088", "synthetic / real R832"]
    rows = []
    for m in doc["meta"]["ms"]:
        t0, s0 = real_cell(ref, m, key), synthetic_time(doc, reference, m, key)
        pairs = {rung: (real_cell(groups[REAL_GROUPS[rung]], m, key), synthetic_time(doc, rung, m, key))
                 for rung in (1088, 832)}
        if None in (t0, s0) or any(None in pair for pair in pairs.values()):
            continue
        row = [str(m)]
        for rung in (1088, 832):
            t, s = pairs[rung]
            row += [cell(t / t0), cell(s / s0)]
        row.append(f"{s0 / t0:.3f}")
        row += [f"{pairs[rung][1] / pairs[rung][0]:.3f}" for rung in (1088, 832)]
        rows.append(row)
    return table(header, rows)


def launch_split(doc, rung, m):
    """Device microseconds per call of the gate/up launch, the down launch, the token sum, and all kernels."""
    prof = doc["rungs"][str(rung)]["cells"][str(m)]["F"]["profile"]
    out = {"total": prof["kernel_us_per_call"]}
    for name, v in prof["top"].items():
        found = re.search(r"routed_fused_kernel<([^>]*)>", name)
        if found:
            out["mode" + found.group(1).split(",")[1].strip()] = v["us_per_call"]
        elif "token_sum" in name:
            out["token_sum"] = v["us_per_call"]
    return out


def split_table(doc, reference, m):
    base = launch_split(doc, reference, m)
    rows = []
    for q in doc["rungs"]:
        s = launch_split(doc, q, m)
        rows.append([f"R{q}", f"{s['mode0']:.1f}", cell(s["mode0"] / base["mode0"]), f"{s['mode2']:.1f}",
                     cell(s["mode2"] / base["mode2"]), f"{s['total']:.1f}", cell(s["total"] / base["total"])])
    return table(["Rung", "gate/up us", "/ R1024", "down us", "/ R1024", "all kernels us", "/ R1024"], rows)


def maxima_table(runs, key):
    """Per rung, each run's largest ratio over the reference and the M it falls at."""
    rungs = [q for q in runs[0][1]["rungs"] if int(q) != int(runs[0][2])]
    header = ["Rung"] + [label for label, _doc, _ref, _summary in runs]
    rows = []
    for q in rungs:
        row = [f"R{q}"]
        for _label, doc, _ref, summary in runs:
            found = [(summary[str(m)][q][key]["ratio"], m) for m in doc["meta"]["ms"] if key in summary[str(m)].get(q, {})]
            row.append(f"{cell(max(found)[0])} at M {max(found)[1]}" if found else "-")
        rows.append(row)
    return table(header, rows)


def power_table(doc, summary, m):
    row = summary[str(m)]
    rows = []
    for q in sorted(row, key=lambda r: row[r].get("rank", 99)):
        e = row[q]
        if "fwd_per_j" in e:
            rows.append([e["rank"], f"R{q}", f"{e['mean_w']:.1f}", f"{e['mean_w'] / 140.0:.2f}", f"{e['fwd_per_j']:.3f}"])
    return table(["Rank", "Rung", "Mean W", "Envelope fraction", "Forwards per joule"], rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--real")
    ap.add_argument("--reference", type=int, default=1024)
    ap.add_argument("--splits", default="1,8,512,2048")
    ap.add_argument("--maxima-only", action="store_true", help="print only the per-rung maxima over the runs")
    args = ap.parse_args(argv)
    real = json.load(open(args.real)) if args.real else None
    runs = []
    for item in args.run:
        label, path = item.split("=", 1)
        doc = json.load(open(path))
        summary = summarize_rungs(doc, args.reference)
        runs.append((label, doc, args.reference, summary))
        if args.maxima_only:
            continue
        print(f"### Run {label}: {doc['meta']['host']}, kernel {str(doc['meta']['kernel_sha'])[:8]}, "
              f"action {str(doc['meta']['pb_action'])[:8]}\n")
        layout = doc["rungs"][str(args.reference)].get("word_layouts", {}).get("gate", "unrecorded")
        for key, title in KEYS:
            print(f"Time over R{args.reference} ({layout} words), {title}:\n")
            print(ratio_table(doc, summary, key, args.reference) + "\n")
        if real is not None:
            for key, title in KEYS:
                print(f"Real wires against synthetic, {title} (real run {str(real['meta']['pb_action'])[:8]}):\n")
                print(real_table(real, doc, key, args.reference) + "\n")
        for m in (int(v) for v in args.splits.split(",")):
            if str(m) in summary:
                print(f"Launch split at M = {m} (forward pass, profiler us per call):\n")
                print(split_table(doc, args.reference, m) + "\n")
                print(f"Work per joule at M = {m}:\n")
                print(power_table(doc, summary, m) + "\n")
    if len(runs) > 1:
        print("### Largest ratio over M, per rung and run\n")
        for key, title in KEYS:
            print(f"{title}:\n")
            print(maxima_table(runs, key) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
