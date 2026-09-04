#!/usr/bin/env python3
"""Differences between route-trace snapshots: what each STAGE of a serve ran.

``TESSERA_ROUTE_TRACE`` accumulates launch counts for the life of the serve
process, so a single snapshot answers "what did this serve ever run" -- and
that is not the question. A served-KL receipt has to say what the SCORED
forwards were, separately from the model-load profile, the warm-ups and any
smoke generation. So the wrapper snapshots the file between stages and this
script subtracts consecutive snapshots.

usage::

    route_trace_delta.py stage-name=snapshot.json [stage-name=snapshot.json ...]

Each stage after the first is reported as its delta from the previous one, so
"decode dump" holds only the launches the decode dump caused. A stage that
should have run nothing at M > 1 shows it, in the table, as a number.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

KEY = ("policy", "shape", "symbol", "decoder")


def _entries(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "tessera.route_trace/1":
        raise SystemExit(f"{path} is not a tessera.route_trace/1 snapshot")
    return {tuple(e[k] for k in KEY): (e["launches"], e["modules"])
            for e in data["entries"]}


def _delta(before, after):
    out = {}
    for key, (launches, modules) in after.items():
        prior = before.get(key, (0, 0))[0]
        if launches - prior:
            out[key] = (launches - prior, modules)
    return out


def _print(title, table):
    print(f"--- {title} ---")
    if not table:
        print("  (no launches)")
        return
    width = max(len(" ".join(k)) for k in table)
    for key in sorted(table):
        launches, modules = table[key]
        policy, shape, symbol, decoder = key
        print(f"  {' '.join(key):<{width}}  launches={launches:<8d} "
              f"modules={modules}")
        del policy, shape, symbol, decoder


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    stages = []
    for arg in argv:
        if "=" not in arg:
            raise SystemExit(f"expected stage=path, got {arg!r}")
        name, path = arg.split("=", 1)
        stages.append((name, _entries(path)))
    _print(f"{stages[0][0]} (cumulative at the first snapshot)", stages[0][1])
    for (prev_name, prev), (name, current) in zip(stages, stages[1:]):
        _print(f"{name}  (delta from {prev_name})", _delta(prev, current))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
