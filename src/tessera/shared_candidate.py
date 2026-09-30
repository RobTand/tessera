"""Strict additive NVFP4 replay-table pricing; legacy per-unit prices stay intact.

An artifact states membership and expected components. Only a same-process,
rank/device-local live-allocation witness can turn that price into a charge.
"""
from __future__ import annotations

import copy
import re

PRICING_SCHEMA = "tessera.shared_candidate_resident_pricing.v1"
COMPONENT_NAMES = ("subsets", "table_next", "table_sub")


def _refuse(reason):
    raise ValueError("shared-candidate: " + reason)


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        _refuse(f"{where}: unexpected or missing fields")


def _integer(value, minimum, where):
    if type(value) is not int or value < minimum:
        _refuse(f"{where}: expected integer >= {minimum}")
    return value


def _members(value):
    if (not isinstance(value, list) or not value
            or any(not isinstance(m, str) or not m for m in value)
            or value != sorted(set(value))):
        _refuse("members: expected sorted unique nonempty module names")


def validate_components(components):
    if (not isinstance(components, list) or any(not isinstance(c, dict) for c in components)
            or [c.get("name") for c in components] != list(COMPONENT_NAMES)):
        _refuse("components: expected subsets, table_next, table_sub in allocation order")
    for index, component in enumerate(components):
        _keys(component, ("name", "shape", "dtype", "layout", "bytes"), "component")
        shape = component["shape"]
        if (not isinstance(shape, list) or len(shape) != 2 or shape[0] != (4 if index == 0 else 2)
                or any(type(n) is not int or n < 1 for n in shape)
                or component["dtype"] != "int64" or component["layout"] != "contiguous"
                or type(component["bytes"]) is not int or component["bytes"] != shape[0] * shape[1] * 8):
            _refuse("component: exact int64 contiguous shape/bytes required")
    if components[1]["shape"] != components[2]["shape"]:
        _refuse("transition components have different state counts")
    return sum(c["bytes"] for c in components)


def validate_pricing(pricing):
    _keys(pricing, ("schema", "scope", "private_resident_bytes", "groups", "resident_bytes"), "pricing")
    if pricing["schema"] != PRICING_SCHEMA or pricing["scope"] != "per_process_rank_device":
        _refuse("unsupported pricing schema/scope")
    private = pricing["private_resident_bytes"]
    if not isinstance(private, dict) or not private or any(not isinstance(m, str) or not m for m in private):
        _refuse("private_resident_bytes: missing module roster")
    for m, n in private.items():
        _integer(n, 0, f"private_resident_bytes.{m}")
    if not isinstance(pricing["groups"], list) or not pricing["groups"]:
        _refuse("groups: missing explicit shared candidates")
    seen, total = set(), sum(private.values())
    for group in pricing["groups"]:
        _keys(group, ("group_id", "members", "components"), "group")
        ident = group["group_id"]
        if not isinstance(ident, str) or not re.fullmatch("[0-9a-f]{64}", ident) or ident in seen:
            _refuse("group_id: missing/duplicate trellis identity digest")
        seen.add(ident)
        _members(group["members"])
        if not set(group["members"]) <= set(private):
            _refuse("group members outside module roster")
        total += validate_components(group["components"])
    if type(pricing["resident_bytes"]) is not int or pricing["resident_bytes"] != total:
        _refuse("resident_bytes: private plus shared composition disagrees")
    return pricing


def build_pricing(modules, module_groups):
    """Compose additive artifact metadata, never rewrite legacy module prices."""
    groups, private = {}, {m: entry["resident_bytes_resident_mode"] for m, entry in modules.items()}
    if set(module_groups) != {m for m, entry in modules.items() if entry["family"] == "TESSERA_NVFP4"}:
        _refuse("every NVFP4 module must declare its replay groups")
    for module, specs in module_groups.items():
        if not isinstance(specs, list) or not specs:
            _refuse("module missing replay specifications")
        if module not in modules or modules[module]["family"] != "TESSERA_NVFP4":
            _refuse("replay tables require a declared NVFP4 module")
        seen = set()
        for spec in specs:
            _keys(spec, ("group_id", "components"), "trellis specification")
            ident = spec["group_id"]
            if ident in seen:
                _refuse("duplicate group within module")
            seen.add(ident)
            private[module] -= validate_components(spec["components"])
            group = groups.setdefault(ident, dict(copy.deepcopy(spec), members=[]))
            if group["components"] != spec["components"]:
                _refuse("one group_id carries conflicting components")
            group["members"].append(module)
    for group in groups.values():
        group["members"].sort()
    return validate_pricing({"schema": PRICING_SCHEMA, "scope": "per_process_rank_device",
                             "private_resident_bytes": private,
                             "groups": [groups[g] for g in sorted(groups)],
                             "resident_bytes": sum(private.values()) + sum(
                                 sum(c["bytes"] for c in g["components"]) for g in groups.values())})


def qualify_allocations(rows, views, dense, *, ready_index):
    """Join witnessed extents to exact resident allocation generations.

    No cache key or repeated address is deduplicated. Every component must name
    one live, never-freed, unaliased replay row, allocated before ready, with a
    plugin replay-table site and no private owner. The returned IDs are the
    only rows eligible for the shared class.
    """
    proof = dense.get("shared_candidate")
    _keys(proof, ("pricing", "process_id", "rank", "device_type", "device_id", "groups"), "proof")
    pricing = validate_pricing(proof["pricing"])
    _integer(ready_index, 1, "ready_index")
    for key, minimum in (("process_id", 1), ("rank", 0), ("device_id", 0)):
        _integer(dense.get(key), minimum, key)
        if proof[key] != dense[key] or type(proof[key]) is not int:
            _refuse(f"{key}: proof names another process/rank/device")
    if proof["device_type"] != dense.get("device_type") or proof["device_type"] != "cuda":
        _refuse("device_type: full-engine proof must name its CUDA device")
    units = dense.get("units")
    if not isinstance(units, dict) or any(not isinstance(entry, dict) for entry in units.values()):
        _refuse("missing unit roster")
    modules = {entry.get("module"): unit for unit, entry in units.items()}
    if len(modules) != len(dense["units"]) or set(modules) != set(pricing["private_resident_bytes"]):
        _refuse("unit/module roster missing or ambiguous")
    for module, unit in modules.items():
        if dense["units"][unit].get("manifest_resident_bytes_resident_mode") != pricing["private_resident_bytes"][module]:
            _refuse("unit private price differs from artifact composition")
    nvfp4_modules = {module for module, unit in modules.items()
                     if units[unit].get("family") == "TESSERA_NVFP4"}
    declared_members = {member for group in pricing["groups"] for member in group["members"]}
    if declared_members != nvfp4_modules:
        _refuse("group membership union must equal the complete NVFP4 module roster")
    expected = {g["group_id"]: g for g in pricing["groups"]}
    if (not isinstance(proof["groups"], list) or len(proof["groups"]) != len(expected)
            or any(not isinstance(g, dict) for g in proof["groups"])
            or {g.get("group_id") for g in proof["groups"]} != set(expected)):
        _refuse("missing/duplicate observed groups")
    if len(rows) != len(views) or len({r["allocation_id"] for r in rows}) != len(rows):
        _refuse("ambiguous allocation/view roster")
    by_id = {view["allocation_id"]: view for view in views}
    if set(by_id) != {r["allocation_id"] for r in rows}:
        _refuse("missing/duplicate allocation views")
    qualified, extents = {}, []
    for observed in proof["groups"]:
        _keys(observed, ("group_id", "members", "components"), "observed group")
        group = expected[observed["group_id"]]
        if observed["members"] != group["members"]:
            _refuse("observed group membership differs from artifact")
        if any(dense["units"][modules[m]].get("family") != "TESSERA_NVFP4" for m in group["members"]):
            _refuse("shared replay group contains a non-NVFP4 module")
        if not isinstance(observed["components"], list) or len(observed["components"]) != len(group["components"]):
            _refuse("missing component storage")
        for storage, component in zip(observed["components"], group["components"], strict=True):
            _keys(storage, (*component, "address"), "observed component")
            if {k: storage[k] for k in component} != component:
                _refuse("observed dtype/layout/shape/bytes differs from artifact")
            address = _integer(storage["address"], 1, "storage address")
            end = address + component["bytes"]
            if any(address < e and a < end for a, e in extents):
                _refuse("aliased shared extents")
            extents.append((address, end))
            matches = [r for r in rows if r.get("address") == address
                       and r["allocate_index"] < ready_index and r["free_completed_index"] is None]
            if len(matches) != 1:
                _refuse("missing/stale/ambiguous resident allocation generation")
            row = matches[0]
            view = by_id[row["allocation_id"]]
            site = view.get("site") or {}
            if (row["bytes"] != component["bytes"] or row.get("scope_stack") or row.get("observed_owners")
                    or row.get("free_requested_index") is not None
                    or row.get("observed_categories") not in ([], ["candidate"])
                    or view["class"] != "candidate" or view.get("rule") != "site:plugin"
                    or site.get("package") != "plugin"
                    or (site.get("relative"), site.get("name")) != (
                        ("encode.py", "_subset_table") if component["name"] == "subsets"
                        else ("decode.py", "_replay_tables"))):
                _refuse("table witness conflicts with allocation ownership/lifetime/site")
            qualified[row["allocation_id"]] = group["group_id"]
    return qualified
