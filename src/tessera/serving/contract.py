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

WHAT THE BYTES WERE.  A cell says a receipt covered a rung; it cannot say
*which bytes* at that rung, and two encoders can write two byte strings at
one rung -- the window table the decoder reads lives on the ALPHABET plane,
so anything that moves the table moves the bytes while the rung, the route
and the decoder stand still.  Beside ``attested_rungs_q256``, then, each
``formats[]`` row carries ``attested_wire``: one ``wire.recipes`` entry per
attested rung -- the checkpoint's own vocabulary, in the spelling a
checkpoint's config already records per rung -- saying which bytes the
receipt was cut on.  A producer preflight compares its checkpoint's entry at
that rung against the stamp and tells "attested on these bytes" from
"attested on this rung".  Coverage (one entry per attested rung, no other)
and the body/span/plane (the route's own ``scheme.ROUTES`` row) are refused
here; that the stamp is still what the exporter writes is pinned in
``tests/test_serving_attested_wire.py``, so a bytes-moving encode change
trips there instead of letting the attestation silently describe bytes no
fresh export writes.

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

WHAT IT EXECUTES AND WHAT IT LOADS ARE TWO CLAIMS.  ``formats`` and
``lane_eligibility`` say what a serve runs; ``native_extensions`` says which
libraries it can map into the serving process, and it is here because a
consumer needs the second one: PrismaQuant's serve fingerprint holds two KLs
comparable only across serves whose native-extension residency matches, so a
lane whose ``.so`` it cannot name reads as a stock serve.  With no table to
read it mirrored the basename in its own repository, which is this principle
failing one repository over.  The block is published as a prefix and a glob
rather than a basename because ``ext`` JIT-builds the module under a
build-identity hash, and it answers "what ran instead" per RESIDENCY rather
than with an ``optional`` flag -- see ``_validate_native_extensions``.

READING IT.  ``load_serving_contract()`` returns the parsed JSON;
``contract_path()`` the packaged file, for a consumer that wants to hash or
copy it verbatim.  Neither imports torch or vLLM: a producer reads this table
on a machine with no GPU.
"""
from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "CENSUS_PHASE_REGIMES",
    "CONTRACT_FILENAME",
    "CONTRACT_SCHEMA",
    "LANE_ELIGIBILITY_SCHEMA",
    "FORMAT_KIND",
    "REQUIRES_PLUGIN",
    "contract_path",
    "extension_lane",
    "lane_decoder",
    "lane_requirements",
    "load_serving_contract",
    "route_wire_spelling",
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

#: The serve-side observation's phase names, and the ``lane_eligibility``
#: regime each phase exercises.  ONE table, read by both sides.
#:
#: ``tools/tessera_route_census.py`` stands up two forwards and calls them by
#: the shape it drove -- a many-row ``prefill`` and a one-row ``decode`` -- and
#: it keys its receipt (``histogram``, ``records``) by those names, which
#: several served receipts already quote.  This contract calls the same two
#: things ``decode`` and ``batch``, because a regime is a *problem shape* and
#: the batch cell covers every M > 1 forward, not only a first prefill.  Both
#: vocabularies are load-bearing, so the pair is mapped here rather than
#: written twice: the census reads this to stamp the contract's word beside its
#: own phase, and :func:`validate_serving_contract` refuses a contract whose
#: declared regimes are not exactly this table's values -- so a third regime,
#: or a rename on either side, fails at contract load instead of at a
#: per-module ``KeyError`` after two full model loads.
CENSUS_PHASE_REGIMES: Mapping[str, str] = MappingProxyType(
    {"prefill": "batch", "decode": "decode"})

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


def _validate_native_extensions(entries: Any, where: str) -> None:
    """``native_extensions`` must BE ``ext.NATIVE_EXTENSIONS``, and be legible.

    The contract publishes what the plugin EXECUTES; this block publishes what
    it LOADS, because a consumer that keys reproducibility on native-extension
    residency has no other machine-readable source and was mirroring the name
    in its own repository -- principle 14's failure, one repository over.

    Two things are checked and they are different.  First AUTHORITY: the block
    equals :data:`ext.NATIVE_EXTENSIONS`, whose ``module_name_prefix`` is the
    very constant ``ext._load_locked`` asks ``cpp_extension.load`` for, so the
    published name is the loaded name by construction rather than by care.
    Second LEGIBILITY: a consumer must not have to guess whether the published
    string is a stem, a prefix or a pattern, so ``match`` names the rule and
    ``filename_glob`` is checked to actually match a filename this load path
    can produce -- the JIT module name carries a build-identity hash, so no
    exact basename exists to publish.

    ``when_unavailable`` is per RESIDENCY and not an ``optional`` boolean on
    purpose: "this build may not have compiled it" and "the route runs without
    it" are two facts, and the second differs by mode (resident substitutes a
    named decoder, streamed refuses).  A fingerprint that exists to tell those
    serves apart needs the substitute's name, and needs to know that an absent
    extension with a streamed route record is an impossible pair.

    Imported here rather than at module scope for the reason ``loader_axes``
    gives: this module is read by a producer with no torch, and ``ext`` and
    ``lane`` are torch-free, but keeping the contract reader's module-level
    dependencies to ``json`` is the property that keeps that true.
    """
    import fnmatch
    import os
    from importlib import util as importlib_util

    from .ext import FALLBACK_REFUSED, FALLBACK_STATUSES, LANE_FIELDS, \
        LANE_REQUIREMENT_FIELDS, MATCH_BASENAME_FNMATCH, NATIVE_EXTENSIONS, csrc_dir
    from .lane import MODES
    from .scheme import ROUTES

    if not isinstance(entries, list):
        raise ValueError(f"{where} must be a JSON array")
    if entries != list(NATIVE_EXTENSIONS):
        raise ValueError(
            f"{where} is not what this build loads. It must equal "
            "tessera.serving.ext.NATIVE_EXTENSIONS, whose module_name_prefix is the constant "
            "the JIT load path itself asks cpp_extension.load for; a document that named a "
            "different extension -- or forgot one -- would be a claim about a runtime that "
            f"does not exist. Published {entries!r}, loaded {list(NATIVE_EXTENSIONS)!r}.")
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        at = f"{where}[{i}]"
        _require_keys(entry, at,
                      required={"module_name_prefix", "filename_glob", "match", "source",
                                "loaded_by", "routes", "lane", "when_unavailable"})
        prefix = entry["module_name_prefix"]
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(f"{at}.module_name_prefix must be a non-empty string")
        if prefix in seen:
            raise ValueError(f"{at}.module_name_prefix {prefix!r} is declared twice")
        seen.add(prefix)
        if entry["match"] != MATCH_BASENAME_FNMATCH:
            raise ValueError(
                f"{at}.match is {entry['match']!r}; this contract publishes exactly "
                f"{MATCH_BASENAME_FNMATCH!r} -- fnmatch the glob against the BASENAME of a "
                "mapped .so. The rule is a value because a consumer cannot otherwise tell a "
                "stem from a prefix from a pattern.")
        glob = entry["filename_glob"]
        # The library torch writes is ``<module name><LIB_EXT>`` and the module
        # name is ``<prefix><build identity>``, so the pattern is the prefix
        # with the identity and the suffix globbed.  Checked by MEANING, not by
        # spelling: a name this load path can produce must match.
        if not isinstance(glob, str) or not fnmatch.fnmatch(f"{prefix}0123456789abcdef.so", glob):
            raise ValueError(
                f"{at}.filename_glob {glob!r} does not match a library name the load path can "
                f"produce ({prefix}<build identity>.so); a glob that matches nothing a serve "
                "maps is a fingerprint that reports every serve identical")
        source = entry["source"]
        if not isinstance(source, str) or not source.startswith("csrc/"):
            raise ValueError(
                f"{at}.source must be a path under 'csrc/' relative to this package, got "
                f"{source!r}")
        if not os.path.isfile(os.path.join(csrc_dir(), source[len("csrc/"):])):
            raise ValueError(
                f"{at}.source {source!r} is not packaged with this build; a contract may not "
                "publish an extension whose sources did not ship")
        loaded_by = entry["loaded_by"]
        if not isinstance(loaded_by, str) or not loaded_by.startswith(f"{__package__}."):
            raise ValueError(
                f"{at}.loaded_by is {loaded_by!r}; an entry belongs in this table only when the "
                f"library can be resident in a SERVING process, so it is loaded by a "
                f"{__package__}.* module. A producer-side extension does not belong here.")
        if importlib_util.find_spec(loaded_by) is None:
            raise ValueError(f"{at}.loaded_by names no module in this build: {loaded_by!r}")
        routes = entry["routes"]
        if not isinstance(routes, list) or not routes:
            raise ValueError(f"{at}.routes must name at least one route that needs it")
        unknown = sorted(set(routes) - set(ROUTES))
        if unknown:
            raise ValueError(
                f"{at}.routes names {unknown}, which this package does not serve "
                f"(tessera.serving.scheme.ROUTES)")
        # THE LANE, AND WHAT IT CAN READ.  ``when_unavailable`` names the
        # SUBSTITUTE's decoder; this names the extension's OWN, so a consumer
        # can tell "the lane ran" from "the fallback ran" without inverting a
        # fallback table.  ``requires`` is the lane's wire predicate, and it is
        # published because it is decidable from a PLAN: every field is a
        # function of ``(grid, q256)``, so a producer can refuse a checkpoint
        # that could never reach the lane before it encodes one unit (#104).
        # Absent means the lane has no predicate beyond its route's own -- an
        # empty object would be a claim of no constraint, which is different.
        lane = entry["lane"]
        _require_keys(lane, f"{at}.lane", required={"decoder"},
                      optional={"requires"})
        if not isinstance(lane["decoder"], str) or not lane["decoder"]:
            raise ValueError(
                f"{at}.lane.decoder must be the non-empty name this lane stamps in "
                "tessera.serving.telemetry's 'decoder' field; it is how a census counts the "
                "modules that actually took the lane")
        if "requires" in lane:
            spot = f"{at}.lane.requires"
            _require_keys(lane["requires"], spot, required=set(),
                          optional=set(LANE_REQUIREMENT_FIELDS))
            if not lane["requires"]:
                raise ValueError(
                    f"{spot} is empty; a lane with no wire predicate omits the block rather "
                    "than publishing an empty one, because 'no constraint' and 'nobody wrote "
                    "the constraint down' must not read the same to a gate")
            for field in ("column_rates", "window_bits"):
                if field not in lane["requires"]:
                    continue
                values = lane["requires"][field]
                if (not isinstance(values, list) or not values
                        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                                   for v in values)):
                    raise ValueError(
                        f"{spot}.{field} must be a non-empty list of positive integers -- the "
                        f"exact set the lane's kernel reads -- got {values!r}")
                if sorted(values) != list(values) or len(set(values)) != len(values):
                    raise ValueError(
                        f"{spot}.{field} must be ascending and without repeats, got {values!r}")
            # The CHECKPOINT's dialect for body and plane (``attested_wire``'s,
            # not ``ROUTES``'), because that is the vocabulary a plan speaks;
            # the map to the route's spelling is the one already written for
            # ``attested_wire``, so a rename fails in one place.
            for field, dialect in (("body", _ATTESTED_WIRE_BODY),
                                   ("plane", _ATTESTED_WIRE_PLANE)):
                if field not in lane["requires"]:
                    continue
                value = lane["requires"][field]
                if value not in dialect:
                    raise ValueError(
                        f"{spot}.{field} is {value!r}, which is not a {field} this package "
                        f"names ({sorted(dialect)}); a lane predicate a gate cannot resolve is "
                        "prose")
        if sorted(lane) != sorted(f for f in LANE_FIELDS if f in lane):
            raise ValueError(f"{at}.lane fields must come from {list(LANE_FIELDS)}")
        behaviours = entry["when_unavailable"]
        if not isinstance(behaviours, Mapping) or sorted(behaviours) != sorted(MODES):
            raise ValueError(
                f"{at}.when_unavailable must answer for exactly the residencies "
                f"{sorted(MODES)}: what a serve does without this library is a different answer "
                "per mode, which is why this is not an 'optional' boolean")
        for mode in MODES:
            spot = f"{at}.when_unavailable[{mode!r}]"
            _require_keys(behaviours[mode], spot, required={"status", "decoder"})
            status, decoder = behaviours[mode]["status"], behaviours[mode]["decoder"]
            if status not in FALLBACK_STATUSES:
                raise ValueError(
                    f"{spot}.status {status!r} is not one of {list(FALLBACK_STATUSES)}")
            if status == FALLBACK_REFUSED and decoder is not None:
                raise ValueError(
                    f"{spot} refuses and still names a decoder {decoder!r}; a refused serve does "
                    "not exist, so there is nothing for a census to have recorded")
            if status != FALLBACK_REFUSED and not (isinstance(decoder, str) and decoder):
                raise ValueError(
                    f"{spot} substitutes and names no decoder; the substitute's name is the "
                    "whole value of this field -- it is the value the route stamps in "
                    "tessera.serving.telemetry's 'decoder' field, and it is how a fingerprint "
                    "tells a native serve from a fallback one")


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
                            "native_extensions", "formats", "lane_eligibility",
                            "tensor_parallel", "expert_parallel", "fused_module"},
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

    _validate_native_extensions(contract["native_extensions"],
                                "runtime_contract.native_extensions")

    families = {}
    for i, entry in enumerate(contract["formats"]):
        where = f"runtime_contract.formats[{i}]"
        _require_keys(entry, where,
                      required={"kind", "family", "grid", "name_pattern",
                                "activation_contract",
                                "reader_rate_range_q256", "reader_rate_step_q256",
                                "reader_rate_bound", "attested_rungs_q256",
                                "attested_wire",
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
        # The residency, where a gate can read it.  It belongs on the ROW and
        # is checked against the tuple the serve gates on, exactly as
        # ``loader_axes`` is checked against ``ROUTE_TP_AXES``: a row claiming
        # a residency the build does not serve -- or inventing a mode, or a
        # typo -- would otherwise validate green while letting a producer
        # under-price or over-attest a residency.  Membership, not equality:
        # a family served in one residency only is an honest row.
        # Imported here rather than at module scope for the reason
        # ``loader_axes`` gives: this module is read by a producer with no
        # torch, and ``lane`` is torch-free, but keeping the contract
        # reader's module-level dependencies to ``json`` is the property that
        # keeps that true.
        from .lane import MODES

        modes = entry["residency_modes"]
        if (not isinstance(modes, list) or not modes
                or any(not isinstance(m, str) for m in modes)
                or len(set(modes)) != len(modes)
                or any(m not in MODES for m in modes)):
            raise ValueError(
                f"{where}.residency_modes must be the residencies this build serves: a "
                f"non-empty list of distinct modes from {list(MODES)} "
                "(tessera.serving.lane.MODES -- the set lane.serve_mode and "
                f"build_tessera_method gate on). Got {modes!r}.")
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
        _validate_attested_wire(entry, route, where)
        families[entry["family"]] = entry

    block = contract["lane_eligibility"]
    _require_keys(block, "runtime_contract.lane_eligibility",
                  required={"schema", "platforms", "regimes", "structures", "cells"})
    if block["schema"] != LANE_ELIGIBILITY_SCHEMA:
        raise ValueError(
            f"runtime_contract.lane_eligibility.schema must be {LANE_ELIGIBILITY_SCHEMA!r}, "
            f"got {block['schema']!r}")
    if set(block["regimes"]) != set(CENSUS_PHASE_REGIMES.values()):
        raise ValueError(
            f"runtime_contract.lane_eligibility.regimes declares {sorted(block['regimes'])}, "
            f"but the serve-side observation exercises "
            f"{sorted(set(CENSUS_PHASE_REGIMES.values()))} "
            f"(contract.CENSUS_PHASE_REGIMES, read by tools/tessera_route_census.py). A regime "
            "this document declares and the census never drives is a cell nothing observes, and "
            "a phase the census drives under a name this document does not declare cannot be "
            "joined to a cell at all -- which is how a per-(family, regime) expectation goes "
            "vacuously true on half the matrix. Add the regime to BOTH sides, in this table.")
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


def extension_lane(module_name_prefix: str, contract: Mapping[str, Any] | None = None) -> dict:
    """The ``lane`` block a native extension publishes, or raise.

    A lane is addressed by the extension's ``module_name_prefix`` because that
    is the string the load path itself asks ``cpp_extension.load`` for -- the
    one name a lane cannot be renamed behind.  Raises rather than returning
    ``None``: a caller asking about a lane this build does not publish has
    asked a question with no answer, and the honest outcome is a refusal, not
    an empty dict that reads like "no requirements".
    """
    payload = load_serving_contract() if contract is None else contract
    for entry in payload["native_extensions"]:
        if entry["module_name_prefix"] == module_name_prefix:
            return dict(entry["lane"])
    known = [e["module_name_prefix"] for e in payload["native_extensions"]]
    raise ValueError(
        f"this build publishes no native extension {module_name_prefix!r}, so it publishes no "
        f"lane for it. Known: {known}.")


def lane_decoder(module_name_prefix: str, contract: Mapping[str, Any] | None = None) -> str:
    """The ``telemetry`` decoder name a serve ON this lane stamps.

    Not the substitute's (``when_unavailable[mode].decoder``): this is what a
    census counts when it asks how many modules actually took the lane.
    """
    return extension_lane(module_name_prefix, contract)["decoder"]


def lane_requirements(module_name_prefix: str,
                      contract: Mapping[str, Any] | None = None) -> dict:
    """What a unit's wire must be for this lane to read it; ``{}`` if unstated.

    ``{}`` here means the lane publishes no predicate of its own -- its
    eligibility is the route's, already published in ``formats[]``.  The
    distinction between that and "the block exists and is empty" is enforced
    at contract load: an empty ``requires`` is refused.
    """
    return dict(extension_lane(module_name_prefix, contract).get("requires", {}))


#: ``formats[]`` family -> the ``scheme.ROUTES`` key that serves it.  Two names
#: for one thing, and they are deliberately different: the contract's family is
#: a PAYLOAD name a producer prices (grid + arity), the route key is what the
#: tile IS on the hardware.
_FAMILY_TO_ROUTE = {
    "TESSERA_E2M1_K2": "TESSERA_NVFP4",
    "TESSERA_E4M3_K1": "TESSERA_FP8",
    "TESSERA_BF16_K1": "TESSERA_BF16",
}

#: The checkpoint's spelling of a body/plane beside the route's spelling of
#: the same fact.  ``attested_wire`` speaks ``wire.recipes`` -- the
#: vocabulary a checkpoint's config records per rung
#: (``export._BODY_NAMES`` / ``export._PLANE_NAMES``) -- while the authority
#: it is checked against speaks ``scheme.ROUTES``.  The map between the two
#: dialects is written once here, and
#: ``tests/test_serving_attested_wire.py`` compares both ends against their
#: sources, so a rename on either side fails there rather than drifting here.
_ATTESTED_WIRE_BODY = {"tcq": "TCQ", "window": "WINDOW"}
_ATTESTED_WIRE_PLANE = {"s6b": "S6B", "lut16": "LUT", "channel": "CHANNEL"}


def route_wire_spelling(field: str, value: str) -> str:
    """``scheme.ROUTES``' spelling of a checkpoint-dialect body or plane name.

    The contract states a wire in the vocabulary a checkpoint's own config
    records (``window``, ``channel``); the table a loader gates on spells the
    same facts ``WINDOW`` and ``CHANNEL``.  One map, the one
    ``attested_wire`` already uses, so a caller comparing a recipe against a
    published predicate does not carry a second copy of it.
    """
    dialect = {"body": _ATTESTED_WIRE_BODY, "plane": _ATTESTED_WIRE_PLANE}.get(field)
    if dialect is None:
        raise ValueError(f"no wire dialect for field {field!r}; known: ['body', 'plane']")
    if value not in dialect:
        raise ValueError(f"{value!r} is not a {field} this package names ({sorted(dialect)})")
    return dialect[value]


def _validate_attested_wire(entry: Mapping[str, Any], route: str, where: str) -> None:
    """``attested_wire`` must be the wire each attested rung was cut on (#55).

    Checked in two halves, because they are two different claims.  COVERAGE:
    exactly one entry per rung in ``attested_rungs_q256`` -- a stamp that
    forgot a rung leaves bytes undescribed, and one that names a rung the
    family does not attest describes nothing.  SUBSTANCE: a body/span/plane
    the route decodes, read off ``scheme.ROUTES`` -- the table the loader
    itself gates on -- so a stamp no decoder in this build would read is a
    refusal here rather than a receipt for bytes that cannot serve.  The
    spreads (``sigma``/``channel_sigma``) are typed -- a number, or null for
    the pinned wire -- but not re-derived: they transcribe the receipt, and
    the receipt is prose.  What keeps the transcription honest is the
    tripwire in ``tests/test_serving_attested_wire.py``, which compares every
    stamp against what the exporter writes at that rung today.
    """
    from .scheme import ROUTES

    stamped = entry["attested_wire"]
    if not isinstance(stamped, list):
        raise ValueError(
            f"{where}.attested_wire must be a JSON array, one entry per attested rung")
    rungs = entry["attested_rungs_q256"]
    for i, item in enumerate(stamped):
        at = f"{where}.attested_wire[{i}]"
        _require_keys(item, at,
                      required={"q256", "body", "span", "plane",
                                "window_bits", "seed", "sigma", "channel_sigma"})
        q256 = item["q256"]
        if not isinstance(q256, int) or q256 not in set(rungs):
            raise ValueError(
                f"{at}.q256 is {q256!r}, which {entry['family']} does not attest "
                f"(attested_rungs_q256 {list(rungs)}); a wire stamp beside a rung no receipt "
                "covers is a receipt for nothing")
    if sorted(item["q256"] for item in stamped) != sorted(rungs):
        raise ValueError(
            f"{where}.attested_wire must carry one entry per attested rung "
            f"{sorted(rungs)}, got {[item['q256'] for item in stamped]}; a rung with no stamp "
            "has bytes no receipt describes, and a stamp is meaningless anywhere else")
    expected = ROUTES[route]
    for i, item in enumerate(stamped):
        at = f"{where}.attested_wire[{i}]"
        body = _ATTESTED_WIRE_BODY.get(item["body"])
        if body is None:
            raise ValueError(
                f"{at}.body {item['body']!r} is not a wire body the exporter spells "
                "(tessera.export._BODY_NAMES); a stamp outside the checkpoint's own "
                "vocabulary is nothing a preflight can compare")
        if body != expected["body"] or item["span"] != expected["span"]:
            raise ValueError(
                f"{at} stamps {item['body']!r} span {item['span']!r}, but {route} decodes the "
                f"span-{expected['span']} {expected['body']} body (tessera.serving.scheme.ROUTES). "
                "A stamp the route does not decode cannot be what a served receipt was cut on.")
        plane = _ATTESTED_WIRE_PLANE.get(item["plane"])
        if plane is None:
            raise ValueError(
                f"{at}.plane {item['plane']!r} is not a scale plane the exporter spells "
                "(tessera.export._PLANE_NAMES); a stamp outside the checkpoint's own "
                "vocabulary is nothing a preflight can compare")
        if plane != expected["plane"]:
            raise ValueError(
                f"{at} stamps the {item['plane']!r} plane, but {route} decodes the "
                f"{expected['plane']} plane to its {expected['tile']} tile "
                "(tessera.serving.scheme.ROUTES). "
                "A stamp the route does not decode cannot be what a served receipt was cut on.")
        for field in ("window_bits", "seed"):
            if not isinstance(item[field], int):
                raise ValueError(
                    f"{at}.{field} is {item[field]!r}; a wire.recipes entry carries integers "
                    "here, and a stamp that does not spell one is nothing a preflight can compare")
        for field in ("sigma", "channel_sigma"):
            value = item[field]
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(
                    f"{at}.{field} is {value!r}; a modelled source spread in grid units is a "
                    "number, or null for the pinned wire (sigma unset). A stamp that spells "
                    "neither is nothing a preflight can compare")
