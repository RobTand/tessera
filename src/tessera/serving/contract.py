"""The runtime contract this plugin packages, and what it is allowed to say.

Principle 14: a claim about what a serving runtime DOES is derived from a
machine-readable table the runtime publishes, never asserted in prose a gate
cannot read.  This package IS that runtime for Tessera bytes, so it publishes
its own ``runtime_contract.json`` and a producer (PrismaQuant) reads it through
``importlib.resources`` rather than hard-coding a route claim.

WHAT A CELL MEANS.  A ``lane_eligibility`` cell says: on this platform, for
this payload family, at these rungs, in this regime, the plugin executes this
activation contract on a route with this status.  A cell exists only where a
container receipt covers it; absence resolves ``unattested``, which is the
honest status and not a refusal.  Today the table is DENSE-ONLY on sm_121, at
one rung per family, both residency modes -- exactly the axes the served
receipts cover.  Routed-MoE experts, TP>1 and any other rung are not in it.

WHAT ``requires_plugin`` MEANS.  Every Tessera route is reachable only in a
process where this plugin is installed and registered (entry point
``vllm.general_plugins``).  Stock vLLM has no reader for these bytes, so the
route is not merely flag-gated: it is plugin-gated.  That is a machine-readable
cell field, not prose, because a producer's export gate has to be able to
refuse an artifact whose serve command would not install the plugin.

READING IT.  ``load_serving_contract()`` returns the parsed JSON;
``contract_path()`` the packaged file, for a consumer that wants to hash or
copy it verbatim.  Neither imports torch or vLLM: a producer reads this table
on a machine with no GPU.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

__all__ = [
    "CONTRACT_FILENAME",
    "CONTRACT_SCHEMA",
    "LANE_ELIGIBILITY_SCHEMA",
    "FORMAT_KIND",
    "REQUIRES_PLUGIN",
    "contract_path",
    "load_serving_contract",
    "validate_serving_contract",
]

CONTRACT_FILENAME = "runtime_contract.json"
CONTRACT_SCHEMA = "tessera.runtime-contract.v1"
LANE_ELIGIBILITY_SCHEMA = "tessera.lane-eligibility.v3"
#: The ``formats[]`` discriminator for a family addressed by a RATE (body bits
#: per 256 weights), not by a codebook size.  Every Tessera family is one.
FORMAT_KIND = "tessera_wire"
#: The value every Tessera cell publishes in ``requires_plugin``.
REQUIRES_PLUGIN = "tessera"

_ROUTE_STATUSES = frozenset({"backed", "backed_with_serve_flag", "unbacked", "fallback"})
_QUALIFICATIONS = frozenset({"device_qualified", "compile_only"})


def contract_path():
    """The packaged contract file as a ``Traversable``.

    ``importlib.resources`` so a wheel install, an editable install and an
    in-repo checkout all resolve identically; never repo-root arithmetic.
    """
    from importlib import resources

    return resources.files(__package__).joinpath(CONTRACT_FILENAME)


def load_serving_contract() -> dict[str, Any]:
    """The packaged contract, parsed and validated."""
    raw = contract_path().read_text(encoding="utf-8")
    contract = json.loads(raw)
    validate_serving_contract(contract)
    return contract


def _require_keys(payload: Mapping[str, Any], where: str, required: set[str],
                  optional: set[str] = frozenset()) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{where} is missing {missing}")
    unknown = sorted(set(payload) - required - set(optional))
    if unknown:
        raise ValueError(f"{where} carries unknown field(s) {unknown}")


def validate_serving_contract(contract: Mapping[str, Any]) -> None:
    """Refuse a contract this package would not itself honour.

    The point is not schema hygiene: it is that the packaged table and the code
    that serves are one artifact.  A cell naming a family no route serves, a
    structure the dispatch refuses, or a rung the reader will not accept would
    be a claim about a runtime that does not exist.
    """
    from .scheme import ROUTES, STRUCTURES

    _require_keys(contract, "runtime_contract",
                  required={"schema", "contract_version", "quant_method", "versions",
                            "formats", "lane_eligibility", "tensor_parallel",
                            "expert_parallel"})
    if contract["schema"] != CONTRACT_SCHEMA:
        raise ValueError(
            f"runtime_contract.schema must be {CONTRACT_SCHEMA!r}, got {contract['schema']!r}")
    if contract["quant_method"].get("canonical") != REQUIRES_PLUGIN:
        raise ValueError(
            f"runtime_contract.quant_method.canonical must be {REQUIRES_PLUGIN!r}: it is the "
            "checkpoint field that selects this plugin")

    families = {}
    for i, entry in enumerate(contract["formats"]):
        where = f"runtime_contract.formats[{i}]"
        _require_keys(entry, where,
                      required={"kind", "family", "name_pattern", "candidate_rungs_q256",
                                "reader_rate_range_q256", "native_terminal_q256",
                                "residency_modes"})
        if entry["kind"] != FORMAT_KIND:
            raise ValueError(f"{where}.kind must be {FORMAT_KIND!r}, got {entry['kind']!r}")
        low, high = entry["reader_rate_range_q256"]
        for rung in entry["candidate_rungs_q256"]:
            if not low <= rung <= high:
                raise ValueError(
                    f"{where}: candidate rung {rung} is outside the reader range [{low}, {high}]")
        families[entry["family"]] = entry

    block = contract["lane_eligibility"]
    _require_keys(block, "runtime_contract.lane_eligibility",
                  required={"schema", "platforms", "regimes", "structures", "cells"})
    if block["schema"] != LANE_ELIGIBILITY_SCHEMA:
        raise ValueError(
            f"runtime_contract.lane_eligibility.schema must be {LANE_ELIGIBILITY_SCHEMA!r}, "
            f"got {block['schema']!r}")
    unknown_structures = sorted(set(block["structures"]) - set(STRUCTURES))
    if unknown_structures:
        raise ValueError(
            f"runtime_contract.lane_eligibility.structures names {unknown_structures}, which "
            f"the dispatch refuses; this plugin serves {sorted(STRUCTURES)}. A cell may not "
            "claim a structure no route executes -- routed-MoE experts above all.")
    contracts_by_family = {
        # The route's own constant, not a copy: a cell that drifted from the
        # code would attest an activation contract the serve does not run.
        family: route["activation_contract"] for family, route in (
            (f, ROUTES[fam]) for f, fam in _FAMILY_TO_ROUTE.items())
    }
    for i, cell in enumerate(block["cells"]):
        where = f"runtime_contract.lane_eligibility.cells[{i}]"
        _require_keys(cell, where,
                      required={"id", "platform", "family", "structure", "regime",
                                "rungs_q256", "activation_contract", "route_status",
                                "qualification", "requires_plugin", "requires_serve_flags",
                                "predicates"})
        if cell["platform"] not in block["platforms"]:
            raise ValueError(f"{where}.platform {cell['platform']!r} is not declared")
        if cell["regime"] not in block["regimes"]:
            raise ValueError(f"{where}.regime {cell['regime']!r} is not declared")
        if cell["structure"] not in block["structures"]:
            raise ValueError(f"{where}.structure {cell['structure']!r} is not declared")
        if cell["family"] not in families:
            raise ValueError(f"{where}.family {cell['family']!r} is not published in formats[]")
        if cell["route_status"] not in _ROUTE_STATUSES:
            raise ValueError(f"{where}.route_status {cell['route_status']!r} is not a status")
        if cell["qualification"] not in _QUALIFICATIONS:
            raise ValueError(f"{where}.qualification {cell['qualification']!r} is not one")
        if cell["requires_plugin"] != REQUIRES_PLUGIN:
            raise ValueError(
                f"{where}.requires_plugin must be {REQUIRES_PLUGIN!r}: stock vLLM has no reader "
                "for these bytes, so every cell here is plugin-gated")
        if not cell["requires_serve_flags"]:
            raise ValueError(
                f"{where}: every route is reached through a declared residency, so a cell names "
                "the serve flag that selects one")
        expected = contracts_by_family[cell["family"]]
        if cell["activation_contract"] != expected:
            raise ValueError(
                f"{where}.activation_contract is {cell['activation_contract']!r} but the "
                f"{cell['family']} route executes {expected!r}")
        unknown_rungs = sorted(set(cell["rungs_q256"])
                               - set(families[cell["family"]]["candidate_rungs_q256"]))
        if unknown_rungs:
            raise ValueError(
                f"{where}.rungs_q256 names {unknown_rungs}, which the family does not publish")

    for i, unit in enumerate(contract["tensor_parallel"]["units"]):
        where = f"runtime_contract.tensor_parallel.units[{i}]"
        _require_keys(unit, where, required={"unit", "kind", "max_world_size"})
        if unit["max_world_size"] != 1:
            raise ValueError(
                f"{where}: a Tessera unit is one blob per vLLM module against a shared rate "
                "schedule; a sharded form needs per-rank wires, not a byte range, so no cell "
                "may admit a world size above one")
    if contract["expert_parallel"]["units"]:
        raise ValueError(
            "runtime_contract.expert_parallel.units must be empty: no served measurement covers "
            "routed-MoE experts, so the contract makes no expert-parallel claim")


#: ``formats[]`` family -> the ``scheme.ROUTES`` key that serves it.  Two names
#: for one thing, and they are deliberately different: the contract's family is
#: a PAYLOAD name a producer prices (grid + arity), the route key is what the
#: tile IS on the hardware.
_FAMILY_TO_ROUTE = {
    "TESSERA_E2M1_K2": "TESSERA_NVFP4",
    "TESSERA_E4M3_K1": "TESSERA_FP8",
}
