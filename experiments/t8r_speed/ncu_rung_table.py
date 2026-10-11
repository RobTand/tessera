"""Per-launch NCU table of a ``bench_rung_forward.py --ncu`` report (tessera#694).

``bench_t8r.sh`` with BENCH_NCU=1 wraps ``bench_rung_forward.py --ncu`` in
Nsight Compute: one call per (rung, M), between cudaProfilerStart and
cudaProfilerStop, and one report.  ``ncu --import REPORT --page raw --csv``
lists the kernels the report names (``routed_fused_kernel``, the token sum) in
launch order.  Every call launches the same number of them, so the launches
split evenly over the calls, and the calls are the JSON's (rung, M) cells in
the order the bench wrote them.  Each routed launch is named by its template:
the mode (0 gate/up, 2 down), the low rate, whether a second run exists, the
superblock width and whether the words are piece-major.

The table carries, per launch, the metrics the stall and occupancy reading
needs; a metric the report lacks is None, and the missing names are listed.

    ncu_rung_table.py --report t8r.ncu-rep --cells bench_rung_forward.json
    ncu_rung_table.py --csv raw.csv --cells bench_rung_forward.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys

NCU = os.path.join(os.environ.get("NCU_ROOT", "/opt/nvidia/nsight-compute/2025.3.1"), "ncu")
KERNEL = "routed_fused_kernel"
#: ``routed_fused_kernel<fp8, mode, dense, split, r_lo, two, bm, piece_major, paired>``
TEMPLATE = ("fp8", "mode", "dense", "split", "r_lo", "two", "bm", "piece_major", "paired")
STALLS = ("long_scoreboard", "barrier", "short_scoreboard", "mio_throttle", "wait", "math_pipe_throttle",
          "lg_throttle", "not_selected", "selected", "no_instruction", "branch_resolving", "dispatch_stall",
          "drain", "membar", "sleeping", "tex_throttle", "imc_miss", "misc")
METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "ipc": "smsp__inst_executed.avg.per_cycle_active",
    "inst_executed": "smsp__inst_executed.sum",
    "smem_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smem_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "dram_bytes_read": "dram__bytes_read.sum",
    "registers": "launch__registers_per_thread",
    "smem_dynamic_bytes": "launch__shared_mem_per_block_dynamic",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
}
METRICS.update({f"stall_{name}": f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio"
                for name in STALLS})
TO_US = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3,
         "second": 1e6, "s": 1e6}


def parse_template(name):
    """The template arguments of a routed launch, or None for another kernel."""
    found = re.search(KERNEL + r"<([^>]*)>", name)
    if not found:
        return None
    args = [a.strip() for a in found.group(1).split(",")]
    if len(args) != len(TEMPLATE):
        raise ValueError(f"{KERNEL} has {len(args)} template arguments, not {len(TEMPLATE)}: {name[:160]}")
    out = {}
    for key, value in zip(TEMPLATE, args):
        out[key] = (value == "true") if value in ("true", "false") else int(value)
    return out


def number(text):
    text = (text or "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def read_launches(csv_text, kernel_filter=("routed_fused_kernel", "token_sum")):
    """``(launches, missing)``: each named kernel in launch order, with its metrics."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    start = next((i for i, row in enumerate(rows) if "Kernel Name" in row), None)
    if start is None:
        raise ValueError("the report's CSV has no 'Kernel Name' column")
    head, units, body = rows[start], rows[start + 1], rows[start + 2:]
    col = {name: i for i, name in enumerate(head)}
    missing = sorted(key for key, metric in METRICS.items() if metric not in col)
    launches = []
    for row in body:
        if len(row) != len(head):
            continue
        name = row[col["Kernel Name"]]
        if not any(f in name for f in kernel_filter):
            continue
        record = {"id": int(row[col["ID"]]), "kernel": name[:140], "template": parse_template(name)}
        for key, metric in METRICS.items():
            if metric not in col:
                record[key] = None
                continue
            value = number(row[col[metric]])
            if value is not None and key == "duration_us":
                value *= TO_US[units[col[metric]]]
            record[key] = value
        launches.append(record)
    return launches, missing


def cell_order(doc):
    """The (rung, M) calls in the order the bench made them."""
    return [(int(rung), int(m)) for rung, record in doc["rungs"].items() for m in record["cells"]]


def split_by_call(launches, calls):
    """Every call launches the same kernels, so the launches split evenly."""
    if not calls or len(launches) % len(calls):
        raise ValueError(f"{len(launches)} launches do not split over {len(calls)} calls")
    per = len(launches) // len(calls)
    return [{"rung": rung, "M": m, "launches": launches[i * per:(i + 1) * per]}
            for i, (rung, m) in enumerate(calls)]


def check_pattern(groups):
    """Every call must show the same kernel pattern, or the split is wrong."""
    patterns = {tuple((l["template"] or {}).get("mode", l["kernel"][:24]) for l in g["launches"]) for g in groups}
    if len(patterns) != 1:
        raise ValueError(f"the calls launch different kernel patterns: {sorted(map(str, patterns))}")


def import_csv(report):
    done = subprocess.run([NCU, "--import", report, "--page", "raw", "--csv"], check=True,
                          capture_output=True, text=True)
    return done.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report")
    ap.add_argument("--csv")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if bool(args.report) == bool(args.csv):
        ap.error("give exactly one of --report and --csv")
    text = import_csv(args.report) if args.report else open(args.csv).read()
    if args.report and args.out:
        with open(os.path.splitext(args.out)[0] + ".raw.csv", "w") as handle:
            handle.write(text)
    launches, missing = read_launches(text)
    groups = split_by_call(launches, cell_order(json.load(open(args.cells))))
    check_pattern(groups)
    result = {"calls": groups, "missing_metrics": missing}
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    for g in groups:
        for launch in g["launches"]:
            t = launch["template"]
            label = (f"mode{t['mode']} r_lo={t['r_lo']} two={int(t['two'])} bm={t['bm']} pm={int(t['piece_major'])}"
                     if t else launch["kernel"][:40])
            print(json.dumps({"rung": g["rung"], "M": g["M"], "launch": label,
                              **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in launch.items()
                                 if k in ("duration_us", "sm_throughput_pct", "occupancy_pct", "issue_active_pct",
                                          "smem_bank_conflicts", "stall_barrier", "stall_long_scoreboard",
                                          "stall_short_scoreboard", "stall_mio_throttle", "stall_wait")}}))
    if missing:
        print(json.dumps({"missing_metrics": missing}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
