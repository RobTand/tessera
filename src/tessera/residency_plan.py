"""Compute rank-local memory peaks from an explicit inventory and allocation plan.

This offline tool does not load tensors or change the live loader. Lifetimes
use half-open integer steps. The plan must declare copies and temporary buffers.
Native storage prices come from the existing serving byte accountant.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import json
from math import lcm, prod
import sys
from pathlib import Path

from .errors import GrammarError
from .serving_parts import (
    dense_resident_bytes_resident_mode,
    routed_fused_unit_bytes,
    routed_window_part_resident_bytes,
    routed_window_unit_resident_bytes,
)

REPORT_SCHEMA = "tessera.residency_report.v1"
DTYPE_BYTES = {
    "bool": 1, "uint8": 1, "int8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "uint16": 2, "int16": 2, "float16": 2, "bfloat16": 2,
    "uint32": 4, "int32": 4, "float32": 4,
    "uint64": 8, "int64": 8, "float64": 8,
}


class ResidencyRefusal(ValueError):
    """Retain a machine-readable report with named refusal reasons."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(reason["message"] for reason in report["reasons"]))


def _refuse(code: str, field: str, message: str) -> None:
    raise ResidencyRefusal({"schema": REPORT_SCHEMA, "fits": False, "ranks": [],
                            "reasons": [{"code": code, "field": field, "message": message}]})


def _object(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        _refuse("invalid_input", field, f"{field} must be an object")
    return value


def _fields(value, allowed: set[str], field: str) -> Mapping:
    value = _object(value, field)
    unknown = value.keys() - allowed
    if unknown:
        name = sorted(unknown, key=str)[0]
        _refuse("unknown_field", f"{field}.{name}", f"{field}.{name} is not a supported field")
    return value


def _integer(value, field: str, code: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _refuse(code, field, f"{field} must be an integer at least {minimum}")
    return value


def _name(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _refuse("invalid_input", field, f"{field} must be a nonempty name")
    return value


def _array(value, field: str) -> list:
    if not isinstance(value, list):
        _refuse("invalid_input", field, f"{field} must be an array")
    return value


def _inventory(structure_spec) -> dict:
    spec = _object(structure_spec, "structure_spec")
    tensors = _object(spec.get("tensors"), "structure_spec.tensors")
    inventory = {}
    for name, raw in tensors.items():
        _name(name, "structure_spec.tensors name")
        field = f"structure_spec.tensors.{name}"
        tensor = _fields(raw, {"shape", "dtype"}, field)
        shape = _array(tensor.get("shape"), field + ".shape")
        for i, dim in enumerate(shape):
            _integer(dim, f"{field}.shape[{i}]", "invalid_shape", 1)
        dtype = tensor.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            _refuse("unknown_dtype", field + ".dtype", f"{field}.dtype {dtype!r} has no byte width")
        inventory[name] = (tuple(shape), DTYPE_BYTES[dtype])
    return inventory


def _local_shapes(shape: tuple, allocation: Mapping, field: str, rank_count: int) -> list:
    ranks = _array(allocation.get("ranks"), field + ".ranks")
    if not ranks or any(type(rank) is not int or not 0 <= rank < rank_count for rank in ranks):
        _refuse("invalid_placement", field + ".ranks", f"{field}.ranks must name existing ranks")
    if len(set(ranks)) != len(ranks):
        _refuse("invalid_placement", field + ".ranks", f"{field}.ranks repeats a rank")
    axis = allocation.get("shard_axis")
    padding = allocation.get("padding_multiple")
    if axis is None:
        if padding is not None:
            _refuse("invalid_placement", field + ".padding_multiple", "padding_multiple requires shard_axis")
        return [(rank, shape, 0) for rank in ranks]
    if type(axis) is not int or not 0 <= axis < len(shape):
        _refuse("invalid_placement", field + ".shard_axis", f"{field}.shard_axis is outside the shape")
    extent = shape[axis]
    if padding is not None:
        padding = _integer(padding, field + ".padding_multiple", "invalid_placement", 1)
        multiple = lcm(padding, len(ranks))
        extent = -(-extent // multiple) * multiple
    elif extent % len(ranks):
        _refuse("invalid_placement", field + ".shard_axis", f"{field}.shard_axis does not divide into equal ranks")
    local = list(shape)
    local[axis] = extent // len(ranks)
    return [(rank, tuple(local), index * local[axis]) for index, rank in enumerate(ranks)]


def _storage_layout(raw, global_shape: tuple, field: str) -> Mapping:
    storage = _object(raw, field)
    kind = storage.get("kind")
    if kind not in ("dense_window", "routed_window", "dense_a4"):
        _refuse("unknown_storage", field + ".kind", f"{field}.kind {kind!r} is not supported")
    allowed = {"kind", "rates"}
    allowed |= ({"arity", "memory", "half", "lut_entries"} if kind == "dense_a4"
                else {"family", "window_bits", "tile_rows"})
    if kind == "routed_window":
        allowed.add("fused")
    _fields(storage, allowed, field)
    expected_ndim = 3 if kind == "routed_window" else 2
    if len(global_shape) != expected_ndim:
        _refuse("invalid_storage", field, f"{field} needs a {expected_ndim}-axis tensor")
    rates = _array(storage.get("rates"), field + ".rates")
    if len(rates) != global_shape[-1]:
        _refuse("invalid_storage", field + ".rates", f"{field}.rates does not cover the columns")
    for index, rate in enumerate(rates):
        _integer(rate, f"{field}.rates[{index}]", "invalid_storage", 1)
    numeric = ("arity", "memory", "half", "lut_entries") if kind == "dense_a4" else ("window_bits", "tile_rows")
    for key in numeric:
        _integer(storage.get(key), field + "." + key, "invalid_storage", 1)
    if kind == "dense_a4":
        # Each native trellis table entry occupies one int32.
        address_bits = (sys.maxsize // DTYPE_BYTES["int32"]).bit_length()
        if storage["memory"] + 1 >= address_bits:
            _refuse("invalid_storage", field + ".memory", f"{field}.memory exceeds the addressable table size")
    else:
        from .kernel_window_gemv import TILE_ROWS
        from .lane_planes import require_window_geometry

        if storage["tile_rows"] != TILE_ROWS:
            _refuse("invalid_storage", field + ".tile_rows",
                    f"{field}.tile_rows must match the compact loader row tile {TILE_ROWS}")
        try:
            require_window_geometry(storage["window_bits"], rates)
        except GrammarError as exc:
            key = ".rates" if max(rates) > storage["window_bits"] else ".window_bits"
            _refuse("invalid_storage", field + key, f"{field}{key}: {exc}")
    if kind != "dense_a4" and storage.get("family") not in ("TESSERA_BF16", "TESSERA_FP8"):
        _refuse("invalid_storage", field + ".family", f"{field}.family has no window byte accountant")
    if kind == "routed_window" and type(storage.get("fused")) is not bool:
        _refuse("invalid_storage", field + ".fused", f"{field}.fused must be a Boolean")
    return storage


def _storage_bytes(shape: tuple, width: int, allocation: Mapping, storage, offset: int, field: str) -> int:
    if storage is None:
        return prod(shape) * width
    axis = allocation.get("shard_axis")
    rates = storage["rates"]
    if axis == len(shape) - 1:
        rates = rates[offset:offset + shape[-1]]
    if len(rates) != shape[-1]:
        _refuse("invalid_storage", field + ".rates", f"{field}.rates cannot describe padded columns")
    rows, cols = shape[-2:]
    kind = storage["kind"]
    try:
        if kind in ("dense_window", "dense_a4"):
            role = {**storage, "rows": rows, "cols": cols, "rates": rates}
            family = "TESSERA_NVFP4" if kind == "dense_a4" else storage["family"]
            return dense_resident_bytes_resident_mode(family, rows, cols, native_roles=[role])
        experts = shape[0]
        unit = routed_window_unit_resident_bytes(storage["family"], rows, cols, rates,
                                                window_bits=storage["window_bits"], tile_rows=storage["tile_rows"])
        if storage["fused"]:
            unit += routed_fused_unit_bytes(storage["window_bits"], cols)
        return experts * unit + routed_window_part_resident_bytes(experts)
    except ValueError as exc:
        _refuse("invalid_storage", field, f"{field}: {exc}")


def plan_residency(structure_spec: Mapping, plan: Mapping) -> dict:
    """Return named per-rank peaks, or refuse invalid inputs and capacity excess."""
    inventory = _inventory(structure_spec)
    plan = _fields(plan, {"ranks", "allocations"}, "plan")
    budgets = _array(plan.get("ranks"), "plan.ranks")
    if not budgets:
        _refuse("invalid_budget", "plan.ranks", "plan.ranks needs at least one rank")
    ranks = []
    for rank, raw in enumerate(budgets):
        field = f"plan.ranks[{rank}]"
        budget = _fields(raw, {"capacity_bytes", "reserve_bytes"}, field)
        capacity = _integer(budget.get("capacity_bytes"), field + ".capacity_bytes", "invalid_budget")
        reserve = _integer(budget.get("reserve_bytes", 0), field + ".reserve_bytes", "invalid_budget")
        ranks.append({"rank": rank, "capacity_bytes": capacity, "reserve_bytes": reserve})
    events = [defaultdict(list) for _ in ranks]
    ids, placed = set(), set()
    for index, raw in enumerate(_array(plan.get("allocations"), "plan.allocations")):
        field = f"plan.allocations[{index}]"
        allocation = _fields(raw, {"id", "tensor", "ranks", "start", "stop", "shard_axis",
                                    "padding_multiple", "storage"}, field)
        name = _name(allocation.get("id"), field + ".id")
        if name in ids:
            _refuse("duplicate_allocation", field + ".id", f"{field}.id repeats {name!r}")
        ids.add(name)
        tensor = allocation.get("tensor")
        if not isinstance(tensor, str) or tensor not in inventory:
            _refuse("unknown_tensor", field + ".tensor", f"{field}.tensor {tensor!r} is not in the structure spec")
        placed.add(tensor)
        start = _integer(allocation.get("start"), field + ".start", "invalid_lifetime")
        stop = allocation.get("stop")
        if stop is not None:
            _integer(stop, field + ".stop", "invalid_lifetime", start + 1)
        shape, width = inventory[tensor]
        local_shapes = _local_shapes(shape, allocation, field, len(ranks))
        storage = None
        if "storage" in allocation:
            storage = _storage_layout(allocation["storage"], shape, field + ".storage")
        for rank, local_shape, offset in local_shapes:
            size = _storage_bytes(local_shape, width, allocation, storage, offset, field + ".storage")
            events[rank][start].append((name, size))
            if stop is not None:
                events[rank][stop].append((name, -size))
    missing = inventory.keys() - placed
    if missing:
        tensor = sorted(missing)[0]
        _refuse("unplaced_tensor", f"structure_spec.tensors.{tensor}", f"tensor {tensor!r} has no allocation")
    reasons = []
    for rank, steps in zip(ranks, events):
        current = peak = rank["reserve_bytes"]
        live, peak_live, peak_step = {}, {}, 0
        for step in sorted(steps):
            # All releases and starts at this step form one atomic boundary.
            for name, delta in steps[step]:
                current += delta
                if delta < 0:
                    del live[name]
                else:
                    live[name] = delta
            if current > peak:
                peak, peak_step, peak_live = current, step, dict(live)
        rank.update(peak_bytes=peak, peak_step=peak_step, peak_allocations=peak_live,
                    final_bytes=current, headroom_bytes=rank["capacity_bytes"] - peak)
        if peak > rank["capacity_bytes"]:
            excess = peak - rank["capacity_bytes"]
            reasons.append({"code": "capacity_exceeded", "rank": rank["rank"],
                            "field": f"plan.ranks[{rank['rank']}].capacity_bytes", "excess_bytes": excess,
                            "message": f"rank {rank['rank']} peak {peak} exceeds capacity {rank['capacity_bytes']} by {excess} bytes"})
    report = {"schema": REPORT_SCHEMA, "fits": not reasons, "ranks": ranks, "reasons": reasons}
    if reasons:
        raise ResidencyRefusal(report)
    return report


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def main(argv=None) -> int:
    """Write the report to standard output without a live-load side effect."""
    parser = argparse.ArgumentParser(description="Compute per-rank memory peaks before load.")
    parser.add_argument("--structure-spec", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = []
        for path in (args.structure_spec, args.plan):
            inputs.append(json.loads(path.read_text(), object_pairs_hook=_unique_pairs))
        report = plan_residency(*inputs)
    except ResidencyRefusal as exc:
        print(json.dumps(exc.report, sort_keys=True))
        return 2
    except (OSError, ValueError) as exc:
        report = {"schema": REPORT_SCHEMA, "fits": False, "ranks": [],
                  "reasons": [{"code": "invalid_json", "field": "input", "message": str(exc)}]}
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
