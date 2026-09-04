#!/usr/bin/env python3
"""Can a lane read this checkpoint?  Asked of the BYTES, before a serve.

Issue #104: ``kernel_window_gemv`` repacks each column's code stream at that
column's own rate and has a 16-row lane only where R bits per code is a whole
number of bytes -- so a unit is readable iff every column rate is in
``SUPPORTED_RATES``.  A rung is a ROOT rate and ``bresenham_rate_schedule``
realises it by mixing the two rates bracketing it, so q256 1006 (root 3.93) is
columns at rate 3 and columns at rate 4, and every unit of such a checkpoint
refuses the lane at load -- through a substitution the route reports as a
served module.  Every one of the six allocated checkpoints under
``/mnt/shared/tessera-runs/allocated`` carried a rate outside the set.

This tool answers the question from the wire, read-only, with no serve:

* ``parse_fused`` every ``<module>.wire_bytes`` tensor and ``parse_unit_artifact``
  every member, so the rates reported are the rates on disk and not the rates a
  plan intended;
* the FULL rate histogram over every unit, per checkpoint -- not a sample:
  the failure it exists to catch is a rung whose rate set is uniform across
  the artifact, and a sample of eight units cannot tell that from a tail;
* per lane, which units it can read, and the offending rates by name.

It exits non-zero when a requested lane cannot read every unit, so it is
usable as a gate on an artifact somebody else built.

    tools/tessera_lane_preflight.py <checkpoint-dir> [...] \\
        [--lane tessera_window_gemv] [--device cpu] [--json OUT]

``--manifest-lanes`` restricts the check to the lanes the artifact itself
declares (``tessera_serving_manifest.json``'s ``requires_lanes``, stamped by
``experiments/export_tessera_serving.py --require-lane``), which is the form a
build pipeline wants: check what the producer promised.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def units_of(path: str, device: str):
    """``[(module, role, rates)]`` for every Tessera unit in a checkpoint.

    Reads the containers rather than the config: a ``config_groups`` scheme
    states the rung a producer meant, and the question here is what the bytes
    carry.
    """
    from safetensors import safe_open

    from tessera.fused import parse_fused
    from tessera.unit_artifact import parse_unit_artifact

    out = []
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"{path}: no .safetensors shard here")
    for shard in shards:
        with safe_open(os.path.join(path, shard), framework="pt") as handle:
            names = [n for n in handle.keys() if n.endswith(".wire_bytes")]
            for name in sorted(names):
                module = name[: -len(".wire_bytes")]
                blob = bytes(handle.get_tensor(name).cpu().numpy().tobytes())
                for member in parse_fused(blob):
                    parsed = parse_unit_artifact(member.blob, device=device)
                    out.append((module, member.name,
                                tuple(int(r) for r in parsed.unit.rates)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--lane", action="append", default=None,
                    help="lane to check, by the extension module_name_prefix the runtime "
                         "contract publishes it under (default: tessera_window_gemv)")
    ap.add_argument("--manifest-lanes", action="store_true",
                    help="check exactly the lanes each artifact DECLARES in its manifest's "
                         "requires_lanes, instead of --lane")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    from tessera.serving.scheme import lane_rate_report

    report = {"schema": "tessera.lane_preflight/1", "checkpoints": {}}
    bad = 0
    for path in args.checkpoints:
        path = os.path.abspath(path)
        lanes = list(args.lane or ())
        declared = []
        manifest = os.path.join(path, "tessera_serving_manifest.json")
        if os.path.isfile(manifest):
            with open(manifest) as fh:
                declared = list(json.load(fh).get("requires_lanes") or ())
        if args.manifest_lanes:
            lanes = declared
        elif not lanes:
            lanes = ["tessera_window_gemv"]

        t0 = time.time()
        units = units_of(path, args.device)
        hist = collections.Counter(tuple(sorted(set(r))) for _m, _r, r in units)
        per_rate = collections.Counter(r for _m, _r, rates in units for r in rates)
        entry = {
            "units": len(units),
            "declared_lanes": declared,
            "checked_lanes": lanes,
            # The full histogram: how many UNITS carry each distinct rate set,
            # and how many COLUMNS sit at each rate across the whole artifact.
            "rate_sets": {",".join(map(str, k)): v for k, v in sorted(hist.items())},
            "columns_by_rate": {str(k): v for k, v in sorted(per_rate.items())},
            "lanes": {},
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"\n=== {path}")
        print(f"    {len(units)} units in {entry['elapsed_s']}s; "
              f"rate sets {entry['rate_sets']}; columns by rate {entry['columns_by_rate']}")
        for lane in lanes:
            unreadable = []
            offending = collections.Counter()
            for module, role, rates in units:
                r = lane_rate_report(lane, rates)
                if not r["reachable"]:
                    unreadable.append(f"{module}.{role}")
                    offending.update(r["offending"])
            supported = lane_rate_report(lane, ())["supported"]
            ok = not unreadable
            entry["lanes"][lane] = {
                "supported_rates": supported,
                "units_readable": len(units) - len(unreadable),
                "units_unreadable": len(unreadable),
                "offending_rates": {str(k): v for k, v in sorted(offending.items())},
                "reachable": ok,
                "examples": unreadable[:3],
            }
            verdict = "READABLE" if ok else "UNREACHABLE"
            print(f"    lane {lane} (rates {supported}): {verdict} -- "
                  f"{len(units) - len(unreadable)}/{len(units)} units readable"
                  + ("" if ok else f"; offending rates "
                                   f"{entry['lanes'][lane]['offending_rates']}, "
                                   f"e.g. {unreadable[0]}"))
            if not ok:
                bad += 1
        report["checkpoints"][path] = entry

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print(f"\n-> {args.json}")
    if bad:
        print(f"\nREFUSED: {bad} (checkpoint, lane) pair(s) cannot take the lane. Any claim "
              "about that lane measured on these bytes is a claim about the fallback.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
