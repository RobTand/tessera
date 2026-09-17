#!/usr/bin/env python3
"""Bounded preparation measurement: compact loader vs the materialising reader.

The claim under test is that the compact path reads the same verified bytes
and produces the same rank-local planes with less startup work -- no expanded
parent BODY plane, no int64 scale expansion, no parent-sized tensor per rank
-- so this probe measures, on **actual** expert wires:

* ``parse_unit_metadata`` wall/CPU against ``parse_unit_artifact``;
* the compact rank cut (packed planes) against
  ``parse_unit_artifact`` + ``sharding.shard_parsed_roles`` +
  ``lane_planes.prepare_span2_planes``, reporting wall time, CPU time and the
  CUDA peak allocation delta of each arm, plus the resident bytes each arm's
  output holds;
* byte equality of every plane, and refusal on any mismatch.

The measurements are bounded and single-process; the wire bytes are read
once, outside the timed sections, and every timed section is synchronized.

Usage (through PrismaBuild; see the execution policy):
    python3 tools/compact_prep_probe.py --json-out receipt.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_EXPORT = (
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
    "first-artifact-exports/a4/merged-4c384e60"
)
TENSOR = "model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.wire"


def _read_wire(export: Path, layer: int, expert: int, projection: str) -> bytes:
    from safetensors import safe_open

    name = TENSOR.format(layer=layer, expert=expert, projection=projection)
    index = json.loads((export / "model.safetensors.index.json").read_text())
    shard = export / index["weight_map"][name]
    with safe_open(str(shard), framework="pt") as handle:
        return bytes(handle.get_tensor(name).detach().cpu().numpy().tobytes())


def _resident_bytes(planes) -> int:
    return sum(t.numel() * t.element_size() for t in planes.values()
               if torch.is_tensor(t))


def _timed(fn, repeat: int):
    """``(wall_ms, cpu_ms)`` medians over ``repeat`` synchronized runs."""
    wall, cpu = [], []
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall0, cpu0 = time.perf_counter(), time.process_time()
        fn()
        wall1, cpu1 = time.perf_counter(), time.process_time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall.append((wall1 - wall0) * 1e3)
        cpu.append((cpu1 - cpu0) * 1e3)
    return statistics.median(wall), statistics.median(cpu)


def _peak_delta(fn, repeat: int) -> int:
    """Peak CUDA allocation delta of one run, bytes."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - base)


def _equal(got: dict, want: dict) -> "tuple[bool, str]":
    if set(got) != set(want):
        return False, "key sets differ"
    for key, value in want.items():
        if torch.is_tensor(value):
            if not torch.equal(got[key], value):
                return False, key
        elif got[key] != value:
            return False, key
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", default=DEFAULT_EXPORT)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    from tessera import lane_planes as lp
    from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
    from tessera.fused import parse_fused
    from tessera.serving.sharding import plan_shard, shard_parsed_roles
    from tessera.unit_artifact import parse_unit_artifact, parse_unit_metadata

    export = Path(args.export)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {
        "export": str(export), "layer": args.layer, "expert": args.expert,
        "tp": args.tp, "device": str(device), "repeat": args.repeat, "roles": {},
    }

    # w13 (row cut over 2N) and w2 (column cut over K): the two MoE shapes.
    shapes = {
        "gate_proj": {"rows": 2048, "columns": 4096, "axis": "row"},
        "down_proj": {"rows": 4096, "columns": 2048, "axis": "column"},
    }
    wires, members = {}, {}
    for projection, shape in shapes.items():
        blob = _read_wire(export, args.layer, args.expert, projection)
        members[projection] = parse_fused(blob)[0]
        wires[projection] = blob

    for projection, shape in shapes.items():
        rows, columns, axis = shape["rows"], shape["columns"], shape["axis"]
        entry = {"bytes": len(wires[projection]), "rows": rows, "columns": columns,
                 "axis": axis, "ranks": {}}
        blob = members[projection].blob

        meta_wall, meta_cpu = _timed(lambda: parse_unit_metadata(blob, device), args.repeat)
        ref_wall, ref_cpu = _timed(lambda: parse_unit_artifact(blob, device), args.repeat)
        entry["parse_metadata_ms"] = {"wall": meta_wall, "cpu": meta_cpu}
        entry["parse_reference_ms"] = {"wall": ref_wall, "cpu": ref_cpu}

        for rank in range(args.tp):
            if axis == "row":
                plan = plan_shard("probe", roles=[(projection, rows)], columns=columns,
                                  out_partitions=[rows // args.tp], in_size=columns,
                                  tp_rank=rank, tp_size=args.tp,
                                  input_size=columns, output_size=rows)
                cut = {"rows": (plan.roles[0].lo, plan.roles[0].hi)}
            else:
                plan = plan_shard("probe", roles=[(projection, rows)], columns=columns,
                                  out_partitions=[rows], in_size=columns // args.tp,
                                  tp_rank=rank, tp_size=args.tp,
                                  input_size=columns, output_size=rows)
                cut = {"cols": (plan.roles[0].lo, plan.roles[0].hi)}

            def old_arm():
                parsed = parse_unit_artifact(blob, device)
                shard = shard_parsed_roles([(projection, parsed)], plan)[0][1]
                return lp.prepare_span2_planes(shard, device=device)

            wire = parse_compact_wire(blob, device=device, name=projection)

            def compact_arm():
                return prepare_span2_compact(wire, device=device, **cut)

            want = old_arm()
            got = compact_arm()
            ok, which = _equal(got, want)
            if not ok:
                print(json.dumps({"error": f"planes differ at {which}",
                                  "projection": projection, "rank": rank}))
                return 1
            old_wall, old_cpu = _timed(old_arm, args.repeat)
            new_wall, new_cpu = _timed(compact_arm, args.repeat)
            old_peak = _peak_delta(old_arm, args.repeat)
            new_peak = _peak_delta(compact_arm, args.repeat)
            entry["ranks"][str(rank)] = {
                "equal": True,
                "old_ms": {"wall": old_wall, "cpu": old_cpu},
                "compact_ms": {"wall": new_wall, "cpu": new_cpu},
                "old_peak_alloc_bytes": old_peak,
                "compact_peak_alloc_bytes": new_peak,
                "old_resident_bytes": _resident_bytes(want),
                "compact_resident_bytes": _resident_bytes(got),
                "speedup_wall": old_wall / new_wall if new_wall else None,
            }
        report["roles"][projection] = entry

    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
