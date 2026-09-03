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

TWO CLAIMS ABOUT TENSOR PARALLELISM, AND THEY ARE NOT THE SAME.
``tensor_parallel.units[].max_world_size`` is an ATTESTATION: the largest world
size a served receipt covers.  It is 1, and it stays 1 until a multi-rank serve
has been run and measured.  Beside it, ``loader_axes`` says what this build's
LOADER does with a shard on each axis, which is a different question with a
different answer -- the E4M3 family cuts both axes, the E2M1x2 family cuts
columns only.  ``_validate_loader_axes`` checks that block against
``sharding.ROUTE_TP_AXES``, the dict the routes themselves gate on, so the
published table and the executed behaviour are one artifact rather than two
documents that can disagree about one runtime.

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


def _validate_loader_axes(axes: Any, family: str, where: str) -> None:
    """``loader_axes`` must BE ``sharding.ROUTE_TP_AXES``, not agree with it in prose.

    Principle 14 applied to this package's own table: the status is the value a
    gate reads and it is compared against the dict the routes themselves gate
    on, so a contract that said a family shards an axis its loader refuses --
    or the reverse -- is a refusal here rather than a surprise at load.  The
    ``reason`` beside it is prose: required where the status is ``refused``
    (a refusal an operator cannot act on is a wall), and ``null`` otherwise.

    Imported here rather than at module scope because ``sharding`` imports
    ``scheme``, and this module is read by a producer on a machine with no
    torch; neither import pulls torch, but keeping the contract reader's
    module-level dependencies to json is the property that keeps it true.
    """
    from .sharding import AXES, ROUTE_TP_AXES, TP_REFUSED, TP_STATUSES

    route = _FAMILY_TO_ROUTE.get(family)
    if route is None or route not in ROUTE_TP_AXES:
        raise ValueError(
            f"{where}.unit {family!r} names no route this package serves, so nothing here can "
            "say what its loader does with a shard axis")
    if not isinstance(axes, Mapping) or sorted(axes) != sorted(AXES):
        raise ValueError(
            f"{where}.loader_axes must describe exactly the axes {sorted(AXES)}, got "
            f"{sorted(axes) if isinstance(axes, Mapping) else type(axes).__name__}")
    for axis in AXES:
        entry = axes[axis]
        _require_keys(entry, f"{where}.loader_axes[{axis!r}]",
                      required={"status", "reason"})
        status = entry["status"]
        if status not in TP_STATUSES:
            raise ValueError(
                f"{where}.loader_axes[{axis!r}].status {status!r} is not one of "
                f"{list(TP_STATUSES)}")
        expected = ROUTE_TP_AXES[route][axis]
        if status != expected:
            raise ValueError(
                f"{where}.loader_axes[{axis!r}].status is {status!r} but the {route} loader "
                f"{expected} a {axis} cut (tessera.serving.sharding.ROUTE_TP_AXES). The table "
                "the routes gate on is the authority; a document that disagreed with it would "
                "be a claim about a runtime that does not exist.")
        reason = entry["reason"]
        if status == TP_REFUSED and not (isinstance(reason, str) and reason.strip()):
            raise ValueError(
                f"{where}.loader_axes[{axis!r}] is refused and carries no reason; a refusal an "
                "operator cannot act on is a wall, not a contract")
        if status != TP_REFUSED and reason is not None:
            raise ValueError(
                f"{where}.loader_axes[{axis!r}] is {status!r} and still carries a reason "
                f"{reason!r}; the field explains a refusal and is null otherwise")


def _validate_fused_module(block: Any, where: str) -> None:
    """``fused_module`` must BE ``scheme.FUSED_MODULE_FIELDS``, not agree with it.

    The same move ``loader_axes`` makes against ``ROUTE_TP_AXES``, for the
    question a producer's group allocator actually asks: inside one vLLM-fused
    module, which of a scheme's facts must every role share, and which is free
    per role?  The answer is a property of the DECODERS -- they read each role
    from that role's own manifest -- so it is published as a value the allocator
    reads, and checked here against the dict the loader itself gates on.  A
    document that said the rate was shared would cost a producer every
    per-member rung its objective picked; one that said a family was free would
    invite a checkpoint vLLM cannot build a method for.

    ``mixed_rung_receipt`` is the OTHER claim, and it is not this one.  It says
    whether a container receipt has covered a served module whose members took
    different rungs -- the distinction ``loader_axes`` and ``max_world_size``
    already draw between what the loader does and what has been measured.  It
    is a boolean a gate reads, never prose.
    """
    from .scheme import (FUSED_CONTAINER, FUSED_MODULE_FIELDS, FUSED_MODULE_SCHEMA,
                         FUSED_Q256_SPELLING)

    _require_keys(block, where,
                  required={"schema", "container", "fields", "sidecar_q256",
                            "mixed_rung_receipt"},
                  # Prose for a person, exactly as ``tensor_parallel.units_note``
                  # is; no gate reads a ``*_note``, and none may.
                  optional={"fields_note", "sidecar_q256_note", "mixed_rung_receipt_note"})
    if block["schema"] != FUSED_MODULE_SCHEMA:
        raise ValueError(f"{where}.schema must be {FUSED_MODULE_SCHEMA!r}, got {block['schema']!r}")
    if block["container"] != FUSED_CONTAINER:
        raise ValueError(
            f"{where}.container must be {FUSED_CONTAINER!r}: it is the magic "
            "``tessera.fused.parse_fused`` refuses anything else on")
    if block["sidecar_q256"] != FUSED_Q256_SPELLING:
        raise ValueError(
            f"{where}.sidecar_q256 must be {FUSED_Q256_SPELLING!r} "
            "(tessera.serving.scheme.FUSED_Q256_SPELLING); it names how a mixed-rung module is "
            "spelled in the checkpoint, and a producer that wrote another spelling would write "
            "a checkpoint this build refuses at load")
    fields = block["fields"]
    if not isinstance(fields, Mapping) or dict(fields) != FUSED_MODULE_FIELDS:
        raise ValueError(
            f"{where}.fields must equal tessera.serving.scheme.FUSED_MODULE_FIELDS "
            f"{FUSED_MODULE_FIELDS}, got {dict(fields) if isinstance(fields, Mapping) else fields!r}. "
            "The dict the loader gates on is the authority; a document that disagreed with it "
            "would tell a producer's allocator that a fused group may vary something this build "
            "refuses, or may not vary something it serves.")
    if not isinstance(block["mixed_rung_receipt"], bool):
        raise ValueError(
            f"{where}.mixed_rung_receipt must be a boolean: it says whether a container receipt "
            "covers a SERVED module whose members took different rungs, which is a different "
            "claim from what the loader accepts and must not be inferred from it")


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
                            "expert_parallel", "fused_module"},
                  # History, not a gate input: a consumer reads the version, and
                  # the changelog says what the version changed for a person.
                  optional={"changelog"})
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
                      required={"kind", "family", "grid", "name_pattern",
                                "activation_contract",
                                "reader_rate_range_q256", "reader_rate_step_q256",
                                "reader_rate_bound", "attested_rungs_q256",
                                "native_terminal_q256", "residency_modes"},
                      optional={"candidate_rungs_q256"})
        # The A side, where a gate can read it.  It belongs on the ROW and not
        # only on the cells because it is a claim about EXECUTION -- what this
        # family's route feeds the GEMM -- and a family can be decodable before
        # any receipt covers it.  Published only on cells, a family with no cell
        # yet had its activation contract in changelog prose, which a consumer
        # cannot tell apart from nobody having filled it in.  Checked against
        # the dict the routes dispatch on, exactly as ``loader_axes`` is checked
        # against ``ROUTE_TP_AXES``.
        route = _FAMILY_TO_ROUTE.get(entry["family"])
        if route is None or route not in ROUTES:
            raise ValueError(
                f"{where}.family {entry['family']!r} names no route this package serves, so "
                "nothing here can say what its A side executes")
        expected_contract = ROUTES[route]["activation_contract"]
        if entry["activation_contract"] != expected_contract:
            raise ValueError(
                f"{where}.activation_contract is {entry['activation_contract']!r} but the "
                f"{route} route executes {expected_contract!r} "
                "(tessera.serving.scheme.ROUTES). The route is the authority; a document that "
                "disagreed with it would price an A side the runtime does not execute.")
        if entry["kind"] != FORMAT_KIND:
            raise ValueError(f"{where}.kind must be {FORMAT_KIND!r}, got {entry['kind']!r}")
        alias = entry.get("candidate_rungs_q256")
        if alias is not None and list(alias) != list(entry["attested_rungs_q256"]):
            raise ValueError(
                f"{where}: candidate_rungs_q256 is a DEPRECATED ALIAS of "
                f"attested_rungs_q256 and must carry the same list, got {alias!r} vs "
                f"{entry['attested_rungs_q256']!r}. It exists only so a reader written "
                "against schema v1 keeps reading this document; two names disagreeing "
                "about one set is the confusion the rename was for.")
        low, high, step = _reader_rate(entry, where)
        for rung in entry["attested_rungs_q256"]:
            if not reader_accepts(rung, low, high, step):
                raise ValueError(
                    f"{where}: attested rung {rung} is not one the reader accepts "
                    f"([{low}, {high}] step {step}). An attested rung is a rung that WAS served; "
                    "it cannot lie outside what the decoder takes.")
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
                               - set(families[cell["family"]]["attested_rungs_q256"]))
        if unknown_rungs:
            raise ValueError(
                f"{where}.rungs_q256 names {unknown_rungs}, which the family does not publish")

    _require_keys(contract["tensor_parallel"], "runtime_contract.tensor_parallel",
                  required={"axis", "semantics", "units"},
                  # Prose for a person; no gate reads it (principle 14).
                  optional={"units_note"})
    for i, unit in enumerate(contract["tensor_parallel"]["units"]):
        where = f"runtime_contract.tensor_parallel.units[{i}]"
        _require_keys(unit, where,
                      required={"unit", "kind", "max_world_size", "loader_axes"})
        if unit["max_world_size"] != 1:
            raise ValueError(
                f"{where}.max_world_size is {unit['max_world_size']}, and it may only be 1. This "
                "field is an ATTESTATION -- the largest world size a served receipt covers -- and "
                "no multi-rank Tessera serve has been run. It is NOT a statement that the bytes "
                "cannot shard: they can, the artifact is TP-agnostic and the loader cuts a unit "
                "at load (tessera.layout.slice_unit). What this build's loader does with each "
                "axis is loader_axes, beside it; raising this number needs a two-rank serve with "
                "a per-rank census and a KL against the single-rank arm.")
        _validate_loader_axes(unit["loader_axes"], unit["unit"], where)
    if contract["expert_parallel"]["units"]:
        raise ValueError(
            "runtime_contract.expert_parallel.units must be empty: no served measurement covers "
            "routed-MoE experts, so the contract makes no expert-parallel claim")
    _validate_fused_module(contract["fused_module"], "runtime_contract.fused_module")


def _reader_rate(entry: Mapping[str, Any], where: str) -> tuple[int, int, int]:
    low, high = entry["reader_rate_range_q256"]
    step = entry["reader_rate_step_q256"]
    if not (isinstance(low, int) and isinstance(high, int) and isinstance(step, int)):
        raise ValueError(f"{where}: the reader rate range and step are integer q256 values")
    if low > high:
        raise ValueError(f"{where}: reader_rate_range_q256 is [{low}, {high}], which is empty")
    if step < 1:
        raise ValueError(f"{where}: reader_rate_step_q256 must be >= 1, got {step}")
    return low, high, step


def reader_accepts(q256: int, low: int, high: int, step: int) -> bool:
    """Is ``q256`` on the published grid of rates the decoder reads?"""
    return low <= q256 <= high and (q256 - low) % step == 0


#: ``(route, grid) -> (family, low, high, step)`` for the PACKAGED contract.
_PACKAGED_RATE_GRID: dict[tuple[str, str], tuple[str, int, int, int]] = {}


def _rate_grids(payload: Mapping[str, Any]) -> dict:
    out = {}
    for entry in payload["formats"]:
        route = _FAMILY_TO_ROUTE.get(entry["family"])
        if route is None:
            continue
        low, high, step = _reader_rate(entry, f"runtime_contract format {entry['family']}")
        out[(route, entry["grid"])] = (entry["family"], low, high, step)
    return out


def reader_rate_grid(route: str, grid: str, contract: Mapping[str, Any] | None = None):
    """``(family, low, high, step)`` for a (route, grid) pair, or ``None``.

    ``None`` is not "anything goes": it means this build publishes no decodable
    rate range for that pair, which the caller must treat as a refusal.  The
    ``TESSERA_NVFP4`` route holds two grids and the contract describes one of
    them, so resolving by route alone would hand an ``E2M1`` checkpoint the
    ``E2M1x2`` numbers.
    """
    if contract is None:
        # The gate runs once per Linear at load; re-reading and re-validating
        # the packaged file each time would put a file read on the load path
        # for no reason.  Only the packaged contract is cached -- a caller that
        # passes one in gets it read fresh, which is what the tests need.
        if not _PACKAGED_RATE_GRID:
            _PACKAGED_RATE_GRID.update(_rate_grids(load_serving_contract()))
        return _PACKAGED_RATE_GRID.get((route, grid))
    payload = contract
    for entry in payload["formats"]:
        if entry.get("grid") != grid:
            continue
        if _FAMILY_TO_ROUTE.get(entry["family"]) != route:
            continue
        low, high, step = _reader_rate(entry, f"runtime_contract format {entry['family']}")
        return entry["family"], low, high, step
    return None


#: ``formats[]`` family -> the ``scheme.ROUTES`` key that serves it.  Two names
#: for one thing, and they are deliberately different: the contract's family is
#: a PAYLOAD name a producer prices (grid + arity), the route key is what the
#: tile IS on the hardware.
_FAMILY_TO_ROUTE = {
    "TESSERA_E2M1_K2": "TESSERA_NVFP4",
    "TESSERA_E4M3_K1": "TESSERA_FP8",
    "TESSERA_BF16_K1": "TESSERA_BF16",
}
