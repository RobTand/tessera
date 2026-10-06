"""Per-launch warp-stall table from the NCU reports ab_arms.sh writes.

Reads ``<out_root>/<arm>-ncu/t8r.ncu-rep`` for each arm through
``ncu --import ... --page raw --csv`` (the host's Nsight Compute, the one
bench_t8r.sh mounts) and prints, per ``routed_fused_kernel`` launch in profile
order, the duration and the average warps stalled per issue on the long
scoreboard, the barrier and the short scoreboard.  ab_arms.sh profiles the
same groups and M in the same order in every arm, so row i of each arm is the
same launch.  Writes ``<out_root>/ncu_stalls.json``.

    python3 ncu_stalls.py <out_root> <arm> [<arm> ...]
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

NCU = Path(os.environ.get("NCU_ROOT", "/opt/nvidia/nsight-compute/2025.3.1")) / "ncu"
KERNEL = "routed_fused_kernel"
METRICS = {
    "us": "gpu__time_duration.sum",
    "long_sb": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "short_sb": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "issued": "smsp__inst_executed.sum",
}


def launches(report: Path) -> list[dict]:
    raw = subprocess.run([str(NCU), "--import", str(report), "--page", "raw", "--csv"],
                         check=True, capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(raw)))
    head, units, body = rows[0], rows[1], rows[2:]
    col = {name: i for i, name in enumerate(head)}
    missing = [m for m in METRICS.values() if m not in col]
    if missing:
        raise SystemExit(f"{report}: metrics absent from the report: {missing}")
    out = []
    for r in body:
        name = r[col["Kernel Name"]]
        if KERNEL not in name:
            continue
        rec = {"id": int(r[col["ID"]]), "kernel": name[:120]}
        for key, metric in METRICS.items():
            value = float(r[col[metric]].replace(",", ""))
            unit = units[col[metric]]
            if key == "us":
                value *= {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0,
                          "msecond": 1e3, "ms": 1e3, "second": 1e6, "s": 1e6}[unit]
            rec[key] = value
        out.append(rec)
    return out


def main(argv: list[str]) -> int:
    root, arms = Path(argv[1]), argv[2:]
    table = {a: launches(root / f"{a}-ncu" / "t8r.ncu-rep") for a in arms}
    n = {len(v) for v in table.values()}
    if len(n) != 1:
        raise SystemExit(f"arms profiled different launch counts: { {a: len(v) for a, v in table.items()} }")
    (root / "ncu_stalls.json").write_text(json.dumps(table, indent=1))
    print("launch " + " | ".join(f"{a}: us long_sb barrier short_sb" for a in arms))
    for i in range(n.pop()):
        cells = []
        for a in arms:
            r = table[a][i]
            cells.append(f"{r['us']:9.1f} {r['long_sb']:6.2f} {r['barrier']:6.2f} {r['short_sb']:6.2f}")
        print(f"{i:3d} {table[arms[0]][i]['kernel'][:60]:60s} " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
