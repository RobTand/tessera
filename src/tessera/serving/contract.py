"""The runtime contract this plugin packages, and what it is allowed to say.

Runtime attestation: a claim about what a serving runtime DOES is derived from a
machine-readable table the runtime publishes, never asserted in prose a gate
cannot read.  This package IS that runtime for Tessera bytes, so it publishes
its own ``runtime_contract.json`` and a producer (PrismaQuant) reads it through
``importlib.resources`` rather than hard-coding a route claim.

WHAT A CELL MEANS.  A ``lane_eligibility`` cell says: on this platform, for
this payload family, at these rungs, in this regime, at this residency, on
this exact runtime image and measured execution mode, the
plugin executes these LAUNCHES under this activation contract on a route with
this status.  A cell exists only where a container receipt covers it; absence
resolves ``unattested``, which is the honest status and not a refusal. The table
preserves the measured dense cells and adds routed-MoE E4M3/q1024 resident eager
on its own exact EUGR image. TP>1, expert parallelism and unmeasured runtime,
residency or rung combinations are not attested.

THE LAUNCH IS A VALUE, AND THE RESIDENCY IS A CONDITION (schema v4, #111).
``executes`` is a list of ``{symbol, decoder}`` and it is DERIVED here from
``scheme.ROUTE_LAUNCHES`` -- the table the routes' own ``census_expected`` is
built from -- narrowed by the cell's regime, by the residency its
``TESSERA_SERVE_MODE`` flag names, and by the lanes each of its rungs reaches
under ``native_extensions[].lane.requires``.  The REGIME there is this
module's (``CENSUS_PHASE_REGIMES`` below: ``decode`` is the one-row forward
and ``batch`` is every M > 1), never the kernel's word for M <= its GEMV max
-- reading the second into a cell is how the batch cell first published the
prefill launch alone and left out the GEMV the same regime runs at two rows.  Before v4 the launch appeared
only in the cell's ``id``, so an E4M3 decode published the materialised FP8
pair in every case; that was accidentally true while the window-GEMV lane was
unreachable and false the moment a rate-constrained artifact was served, with
the lane's own op on 112 of 112 modules.  The residency carries the condition
because both window routes set ``layer.tessera_gemv = None`` in ``resident``
-- the lane exists in ``streamed`` alone -- so two cells of one ``(platform,
family, structure, regime, runtime image, execution mode)`` must cover DISJOINT
residencies. A cell ``id`` names its scope and never a launch; v5 permits a
runtime-derived suffix for disjoint variants while retaining existing IDs.

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

import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "CENSUS_PHASE_REGIMES",
    "CELL_PREDICATE_FACTS",
    "CELL_PREDICATE_OPS",
    "EVIDENCE_GRADES",
    "EVIDENCE_KL_KINDS",
    "EVIDENCE_RECEIPT_ROOT",
    "EVIDENCE_SMOKE_STATUSES",
    "EXECUTION_MODES",
    "PLUGIN_ENTRY_POINT",
    "RUNTIME_SCOPE_KEYS",
    "RUNTIME_VERSION_KEYS",
    "VERSIONS_KEYS",
    "cell_evidence",
    "cell_runtime_versions",
    "derive_evidence_grade",
    "CONSTRUCTION_SCHEMA",
    "CONSTRUCTION_CENSUS_SCHEMA",
    "classify_construction",
    "construction_entry",
    "construction_entry_from_receipt",
    "normalise_module",
    "vllm_module_name",
    "CONTRACT_FILENAME",
    "CONTRACT_SCHEMA",
    "PAYLOAD_FAMILY_BY_ROUTE",
    "LANE_ELIGIBILITY_SCHEMA",
    "FORMAT_KIND",
    "REQUIRES_PLUGIN",
    "contract_path",
    "cell_executes",
    "cell_predicates",
    "cell_residency_modes",
    "cell_runtime_scope",
    "cell_runtime_id_suffix",
    "require_runtime_image",
    "extension_lane",
    "lane_decoder",
    "lane_requirements",
    "load_serving_contract",
    "route_wire_spelling",
    "validate_serving_contract",
]

CONTRACT_FILENAME = "runtime_contract.json"
CONTRACT_SCHEMA = "tessera.runtime-contract.v1"
LANE_ELIGIBILITY_SCHEMA = "tessera.lane-eligibility.v6"
#: Execution is a separate axis from token-count regime and residency. These
#: are the two modes selected by a serving invocation's enforce_eager flag.
EXECUTION_MODES = ("eager", "compiled")
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

_ROUTE_STATUSES = frozenset({"backed", "backed_with_serve_flag", "unbacked"})
_QUALIFICATIONS = frozenset({"device_qualified", "compile_only"})

#: The closed grammar of a cell's ``predicates`` (#134).  A predicate narrows
#: the cell to units for which ``fact op value`` holds, and it is a
#: ``{fact, op, value}`` object over the STRUCTURAL facts a producer's unit
#: carries before it is encoded -- never a wire predicate (that is
#: ``native_extensions[].lane.requires``, decidable from a plan, and a
#: different table).  The set is closed on both axes because a consumer that
#: could not resolve a predicate must refuse the cell, not skip the rule: an
#: unknown predicate that no-ops would let a narrower cell read as
#: unconditionally eligible.  The vocabulary is the one
#: ``docs/measurements/tessera-lane-eligibility-executes-2026-09-04.md``
#: records and PrismaQuant's ``lane_eligibility._PREDICABLE_FACTS`` /
#: ``_PREDICATE_OPS`` resolve; it is owned here so the publisher refuses what
#: the reader could not read.  Every cell published today carries ``[]``.
CELL_PREDICATE_FACTS = ("payload_family", "k", "n_sub", "rate_q256", "role_split",
                        "in_features", "out_features")
CELL_PREDICATE_OPS = ("equals", "in", "multiple_of", "at_least", "at_most")

#: A cell's ``runtime`` (schema v6, #131).  The SCOPE half -- the exact image
#: and the execution modes a receipt covers -- arrived with v5 and is the
#: census's join key.  The TOOLCHAIN half is the vLLM and torch build the
#: cell's own receipt records, verbatim; until v6 the only versions in the
#: document were a global ``versions.attested_on`` block, which named vLLM
#: 0.28.0 for two cells measured on ``0.28.1rc1.dev397``.  One home for both
#: key sets: :func:`cell_runtime_scope` reads the first and tolerates the
#: second (an observation-side fixture carries only a scope);
#: :func:`cell_runtime_versions` requires the whole closed object.
RUNTIME_SCOPE_KEYS = frozenset({"image", "execution_modes"})
RUNTIME_VERSION_KEYS = frozenset({"vllm", "torch"})

#: ``versions`` (schema v6, #131) says one thing per field and nothing about
#: a measured runtime: ``tessera`` is the distribution version,
#: ``plugin_entry_point`` the entry point the wheel declares, and
#: ``default_serve_image`` the ONE serve-image pin every harness reads
#: (``runtime_image.PIN_CONTRACT_FIELD``) -- which must be an image some cell
#: attests, because a default nobody measured is not a pin.
VERSIONS_KEYS = frozenset({"tessera", "plugin_entry_point", "default_serve_image"})
PLUGIN_ENTRY_POINT = "tessera = tessera.serving:register"

#: A cell's ``evidence`` (schema v6, #133): what GRADE of evidence the cell
#: rests on, in a field a gate can read.  Every served KL this repository
#: holds -- dense and MoE -- is a ``kl_tool`` top-K teacher/student-
#: intersection LOWER BOUND; what separates the cells is which regime it was
#: scored in, under which execution modes, and whether a greedy smoke is on
#: record.  So a ``kl`` entry is ``{kind, top_k, regime, execution_modes,
#: receipt}`` (no number: the receipt holds it with its bounds), ``smoke`` is
#: ``{status, receipt}`` in the receipt's own words, and ``grade`` is DERIVED
#: from the entries in the cell's own regime the way ``executes`` is derived
#: from the route table -- stored so a reader needs no derivation, checked so
#: it cannot drift.  ``receipt`` is a repository path under
#: :data:`EVIDENCE_RECEIPT_ROOT`; the validator checks its grammar and a tree
#: test checks the file (a wheel does not ship docs).
EVIDENCE_KL_KINDS = ("topk_intersection_lower_bound", "full_vocab")
EVIDENCE_SMOKE_STATUSES = ("recorded", "repetitive", "not_recorded")
EVIDENCE_GRADES = ("route_only", "kl_lower_bound", "kl_full_vocab")
EVIDENCE_RECEIPT_ROOT = "docs/measurements/"


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


def require_runtime_image(value: Any, where: str = "runtime.image") -> str:
    """An exact manifest reference, using the serving image parser's grammar."""
    from .runtime_image import parse_reference

    if isinstance(value, str):
        repository, tag, digest = parse_reference(value)
        if digest is not None and tag is None and value == f"{repository}@{digest}":
            return value
    raise ValueError(
        f"{where} must be an exact digest reference (repository@sha256:<64 lowercase hex>), "
        f"got {value!r}; a floating tag or local image id does not identify an attested runtime")


def cell_runtime_scope(cell: Mapping[str, Any],
                       where: str = "lane_eligibility cell") -> tuple[str, tuple[str, ...]]:
    """The explicit runtime scope a cell attests; no global image fallback."""
    runtime = cell.get("runtime")
    at = f"{where}.runtime"
    _require_keys(runtime, at, required=set(RUNTIME_SCOPE_KEYS), optional=RUNTIME_VERSION_KEYS)
    image = require_runtime_image(runtime["image"], f"{at}.image")
    modes = runtime["execution_modes"]
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) or mode not in EXECUTION_MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError(
            f"{at}.execution_modes must be a non-empty list of distinct modes from "
            f"{list(EXECUTION_MODES)}, got {modes!r}")
    return image, tuple(mode for mode in EXECUTION_MODES if mode in modes)


def cell_runtime_versions(cell: Mapping[str, Any],
                          where: str = "lane_eligibility cell") -> tuple[str, str]:
    """The ``(vllm, torch)`` a cell's own receipt records, or raise (#131).

    Requires the WHOLE closed ``runtime`` object -- scope and toolchain -- so
    the validator, which calls this, refuses a cell that still relies on a
    global version block; :func:`cell_runtime_scope` is the lenient reader
    for the census join.
    """
    runtime = cell.get("runtime")
    at = f"{where}.runtime"
    _require_keys(runtime, at, required=set(RUNTIME_SCOPE_KEYS | RUNTIME_VERSION_KEYS))
    out = []
    for field in ("vllm", "torch"):
        value = runtime[field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{at}.{field} must be a non-empty string: the build the cell's receipt "
                f"records, verbatim. Got {value!r}.")
        out.append(value)
    return out[0], out[1]


def cell_runtime_id_suffix(cell: Mapping[str, Any]) -> str:
    """An optional scope-derived suffix when multiple runtimes need distinct ids.

    Derived from the SCOPE alone: a toolchain is a property of the image, so
    it must not fork the id (two ids for one scope would be the table-order
    defect the suffix exists to close).
    """
    image, modes = cell_runtime_scope(cell)
    encoded = json.dumps({"image": image, "execution_modes": list(modes)},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "_runtime_" + hashlib.sha256(encoded).hexdigest()


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
                            "tensor_parallel", "expert_parallel", "fused_module",
                            "construction"},
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
    # ``versions`` (v6, #131): closed, every field checked, nothing about a
    # measured runtime -- that lives on the cells.  A reader that still wrote
    # ``attested_on`` learns here, at load, that it no longer means anything.
    versions = contract["versions"]
    _require_keys(versions, "runtime_contract.versions", required=set(VERSIONS_KEYS))
    if not isinstance(versions["tessera"], str) or not versions["tessera"]:
        raise ValueError(
            "runtime_contract.versions.tessera must be a non-empty string, the distribution "
            f"version; got {versions['tessera']!r}")
    if versions["plugin_entry_point"] != PLUGIN_ENTRY_POINT:
        raise ValueError(
            f"runtime_contract.versions.plugin_entry_point must be {PLUGIN_ENTRY_POINT!r}, the "
            f"entry point this package registers; got {versions['plugin_entry_point']!r}")
    default_serve_image = require_runtime_image(
        versions["default_serve_image"], "runtime_contract.versions.default_serve_image")

    _validate_native_extensions(contract["native_extensions"],
                                "runtime_contract.native_extensions")
    _validate_construction(contract["construction"], "runtime_contract.construction")

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
    declared_structures = block["structures"]
    if (not isinstance(declared_structures, list) or not declared_structures
            or any(not isinstance(s, str) for s in declared_structures)
            or len(set(declared_structures)) != len(declared_structures)):
        raise ValueError(
            "runtime_contract.lane_eligibility.structures must be a non-empty list of "
            f"distinct strings, got {declared_structures!r}")
    # ``scheme.STRUCTURES`` is the DISPATCH-capability bound, not an
    # attestation source.  A new dispatch structure must not become eligible
    # here merely because nobody remembered to add it to an ``unserved``
    # denylist: only a cell below is the published authority that a served
    # receipt exists.  First refuse anything the build cannot execute; after
    # validating the cells, derive the positive attested set from them.
    refused = sorted(set(declared_structures) - set(STRUCTURES))
    if refused:
        raise ValueError(
            f"runtime_contract.lane_eligibility.structures names {refused}, which "
            "scheme.STRUCTURES says the dispatch refuses outright")
    contracts_by_family = {
        # The route's own constant, not a copy: a cell that drifted from the
        # code would attest an activation contract the serve does not run.
        family: route["activation_contract"] for family, route in (
            (f, ROUTES[fam]) for f, fam in _FAMILY_TO_ROUTE.items())
    }
    _cell_scope: dict = {}
    cell_ids: set[str] = set()
    cell_structures: list[str] = []
    toolchains_by_image: dict[str, tuple[tuple[str, str], str]] = {}
    for i, cell in enumerate(block["cells"]):
        where = f"runtime_contract.lane_eligibility.cells[{i}]"
        _require_keys(cell, where,
                      required={"id", "platform", "family", "structure", "regime",
                                "rungs_q256", "activation_contract", "executes",
                                "route_status", "qualification", "requires_plugin",
                                "requires_serve_flags", "predicates", "runtime", "evidence"})
        if not isinstance(cell["id"], str) or not cell["id"]:
            raise ValueError(f"{where}.id must be a non-empty string")
        if cell["id"] in cell_ids:
            raise ValueError(f"{where} repeats cell id {cell['id']!r}; every cell has one identity")
        cell_ids.add(cell["id"])
        runtime_image, execution_modes = cell_runtime_scope(cell, where)
        # ONE IMAGE, ONE TOOLCHAIN (v6, #131): a digest names bytes, so two
        # cells naming one image and two vLLM builds would be two runtimes
        # under one digest, which a digest cannot be.
        toolchain = cell_runtime_versions(cell, where)
        known = toolchains_by_image.setdefault(runtime_image, (toolchain, cell["id"]))
        if known[0] != toolchain:
            raise ValueError(
                f"{where} ({cell['id']!r}) records (vllm, torch) {toolchain} on "
                f"{runtime_image}, but {known[1]!r} records {known[0]} on the same digest; "
                "one image cannot be two runtimes. One of the two receipts is misread.")
        if cell["platform"] not in block["platforms"]:
            raise ValueError(f"{where}.platform {cell['platform']!r} is not declared")
        if cell["regime"] not in block["regimes"]:
            raise ValueError(f"{where}.regime {cell['regime']!r} is not declared")
        if cell["structure"] not in block["structures"]:
            raise ValueError(f"{where}.structure {cell['structure']!r} is not declared")
        if cell["structure"] not in cell_structures:
            cell_structures.append(cell["structure"])
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
        cell_predicates(cell, where)
        cell_evidence(cell, where, regimes=block["regimes"])
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
        # WHAT IT EXECUTES (schema v4).  Shape first, then the value.
        executes = cell["executes"]
        if (not isinstance(executes, list) or not executes
                or any(not isinstance(e, Mapping) for e in executes)):
            raise ValueError(
                f"{where}.executes must be a non-empty list of {{symbol, decoder}} objects: a "
                "cell that names no launch attests a route nobody can identify, and an empty "
                "list would read as 'this regime runs nothing'")
        for j, launch in enumerate(executes):
            _require_keys(launch, f"{where}.executes[{j}]", required={"symbol", "decoder"})
        if len(cell_executes(cell)) != len(executes):
            raise ValueError(
                f"{where}.executes repeats a (symbol, decoder) pair; the field is a SET of "
                "launches and a duplicate would double-count a route in any histogram "
                "built from it")
        _validate_cell_executes(cell, _FAMILY_TO_ROUTE[cell["family"]],
                                families[cell["family"]], contract, where)
        # THE RESIDENCY IS A CONDITION, so two cells of one (platform, family,
        # structure, regime, runtime image, execution mode) must not both claim
        # one residency: a reader resolving
        # "what runs here" would otherwise get whichever cell it read first,
        # and that is the failure this schema version exists to close.
        modes = cell_residency_modes(cell, where)
        # THE ID NAMES THE SCOPE, NEVER THE LAUNCH.  Until v4 every id ended in
        # the launch it claimed (``..._decode_scaled_mm_w8a8``) -- an assertion
        # in a string no gate parses, which went stale the day the window-GEMV
        # lane became reachable while the id went on saying ``scaled_mm``
        # (#111). The launch is ``executes`` now. The id retains its base
        # scope, optionally with the v5 runtime suffix; explicit cell fields
        # remain the authority for matching either spelling.
        expected_id = "_".join(
            [cell["family"].lower(), cell["structure"], cell["platform"].replace("_", ""),
             cell["regime"]]
            + ([] if len(modes) == len(_all_modes()) else list(modes)))
        scoped_id = expected_id + cell_runtime_id_suffix(cell)
        if cell["id"] not in (expected_id, scoped_id):
            raise ValueError(
                f"{where}.id is {cell['id']!r}; a cell id is its SCOPE and must be "
                f"{expected_id!r}, optionally followed by its derived runtime suffix "
                "(family, structure, platform, regime, plus the residency where the cell "
                "covers only some). An id that names a launch is a second, "
                "unparsed spelling of `executes` -- exactly the one that went stale.")
        scope = (cell["platform"], cell["family"], cell["structure"], cell["regime"], runtime_image)
        for mode in modes:
            for execution_mode in execution_modes:
                key = (scope, mode, execution_mode)
                clash = _cell_scope.get(key)
                if clash is not None:
                    raise ValueError(
                        f"{where} ({cell['id']!r}) and {clash!r} both cover "
                        f"{scope} at residency {mode!r} and execution mode {execution_mode!r}. "
                        "A cell is resolved by these facts plus the rung, so two cells "
                        "claiming one of them would make the answer depend on table order.")
                _cell_scope[key] = cell["id"]

    # THE PIN IS AN ATTESTED IMAGE (v6, #131).  ``default_serve_image`` is the
    # one digest every harness reads; a default no cell was measured on would
    # be a pin to a runtime nothing here says anything about.
    if default_serve_image not in toolchains_by_image:
        raise ValueError(
            f"runtime_contract.versions.default_serve_image is {default_serve_image!r}, but no "
            f"lane_eligibility cell attests that image (cells attest "
            f"{sorted(toolchains_by_image)}). The default serve image is the pin harnesses "
            "read; it must be a runtime some receipt covers.")

    # The structure axis is a projection of the receipt-bearing cells, never
    # of the dispatch roster.  This is intentionally positive authority: when
    # a future structure enters ``scheme.STRUCTURES`` it remains unattested
    # until a cell is published for an actual serve, without requiring a
    # second hand-maintained list of every runnable-but-unserved structure.
    if declared_structures != cell_structures:
        without_cells = sorted(set(declared_structures) - set(cell_structures))
        raise ValueError(
            "runtime_contract.lane_eligibility.structures names "
            f"{declared_structures}, but its receipt-bearing cells project exactly to "
            f"{cell_structures}. "
            + (f"no receipt-bearing cell names {without_cells}; lane_eligibility is where "
               "served facts go, so dispatch capability alone cannot attest a structure. "
               if without_cells else
               "The structure axis is ordered by first occurrence in the cells, so a second "
               "spelling cannot describe the same published contract. ")
            + "Publish the served cell only after its census, artifact, and quality receipt "
              "exist, then derive this axis from those cells.")

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
            "expert-parallel execution, so the contract makes no expert-parallel claim")
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


#: How a cell's ``requires_serve_flags`` names a residency.  The env var is
#: ``lane.TESSERA_MODE_ENV``'s and the values are ``lane.MODES``'; the pipe is
#: an OR, so ``TESSERA_SERVE_MODE=resident|streamed`` is a cell that covers
#: both.  Parsed rather than pattern-matched because since schema v4 the
#: residency is a CONDITION on the cell, not decoration: two cells of one
#: ``(platform, family, structure, regime)`` are told apart by it.
_MODE_FLAG_SEPARATOR = "|"


def cell_residency_modes(cell: Mapping[str, Any],
                         where: str = "lane_eligibility cell") -> tuple[str, ...]:
    """The residencies a cell's serve flags select, as a value a gate reads.

    A ``lane_eligibility`` cell is scoped to a ``(platform, family, structure,
    regime)`` and, since v4, to a set of RESIDENCIES: the window-GEMV lane
    exists in ``streamed`` alone (both window routes set
    ``layer.tessera_gemv = None`` in ``resident``), so the decode regime of one
    family executes two different launches depending on this flag.  It was
    always in the cell -- as a string a reader had to pattern-match.  This is
    the same string, read.
    """
    from .lane import MODES, TESSERA_MODE_ENV

    head = f"{TESSERA_MODE_ENV}="
    named = [f for f in cell.get("requires_serve_flags", ()) if str(f).startswith(head)]
    if len(named) != 1:
        raise ValueError(
            f"{where}.requires_serve_flags must name exactly one {TESSERA_MODE_ENV} flag "
            f"(got {named!r}): the residency is the axis that decides which launch a regime "
            "makes, so a cell that does not scope itself to one cannot be resolved")
    modes = tuple(str(named[0])[len(head):].split(_MODE_FLAG_SEPARATOR))
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError(
            f"{where}.requires_serve_flags names residencies {list(modes)}; they must be "
            f"distinct values of {list(MODES)} (tessera.serving.lane.MODES, the set "
            "lane.serve_mode gates on)")
    return modes


def _all_modes() -> tuple:
    """``lane.MODES``, imported lazily so this module stays torch-free to read."""
    from .lane import MODES

    return MODES


def cell_executes(cell: Mapping[str, Any]) -> set:
    """``{(symbol, decoder)}`` a cell publishes -- the census's own shape."""
    return {(str(e["symbol"]), str(e["decoder"])) for e in cell["executes"]}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def cell_predicates(cell: Mapping[str, Any],
                    where: str = "lane_eligibility cell") -> tuple[tuple[str, str, Any], ...]:
    """The ``(fact, op, value)`` triples a cell's ``predicates`` states, or raise.

    The field was required on every cell and read by no line of this package
    (#134): a cell could carry ``["anything"]`` and validate, so the first
    gate to start reading it would have inherited unvalidated content.  This
    is the grammar, refused where the bytes are decided: a JSON array of
    closed ``{fact, op, value}`` objects, ``fact`` from
    :data:`CELL_PREDICATE_FACTS`, ``op`` from :data:`CELL_PREDICATE_OPS`, and
    ``value`` typed by the op -- ``in`` takes a non-empty list of scalars,
    ``multiple_of`` a positive integer (a consumer evaluates ``actual % value``
    and treats 0 as false), ``at_least``/``at_most`` an integer, ``equals`` a
    scalar.  One ``(fact, op)`` pair at most once: two ``at_least`` rows on one
    fact are one bound written twice, or two bounds disagreeing.
    """
    payload = cell.get("predicates")
    at = f"{where}.predicates"
    if not isinstance(payload, list):
        raise ValueError(
            f"{at} must be a JSON array of {{fact, op, value}} objects, got {payload!r}")
    out: list[tuple[str, str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, item in enumerate(payload):
        spot = f"{at}[{i}]"
        _require_keys(item, spot, required={"fact", "op", "value"})
        fact, op, value = item["fact"], item["op"], item["value"]
        if fact not in CELL_PREDICATE_FACTS:
            raise ValueError(
                f"{spot}.fact {fact!r} is not a structural fact a cell may predicate on; the "
                f"closed set is {list(CELL_PREDICATE_FACTS)}. A predicate a consumer cannot "
                "resolve is a cell it must refuse, never a rule it may skip.")
        if op not in CELL_PREDICATE_OPS:
            raise ValueError(
                f"{spot}.op {op!r} is not one of {list(CELL_PREDICATE_OPS)}")
        scalar = (str, int, float)
        if op == "in":
            if (not isinstance(value, list) or not value
                    or any(not isinstance(v, scalar) or isinstance(v, bool) for v in value)):
                raise ValueError(
                    f"{spot}: 'in' takes a non-empty list of scalars, got {value!r}")
        elif op == "multiple_of":
            if not _is_int(value) or value <= 0:
                raise ValueError(
                    f"{spot}: 'multiple_of' takes a positive integer, got {value!r}; a consumer "
                    "evaluates actual % value and a zero modulus holds for nothing")
        elif op in ("at_least", "at_most"):
            if not _is_int(value):
                raise ValueError(f"{spot}: {op!r} takes an integer, got {value!r}")
        elif not isinstance(value, scalar) or isinstance(value, bool):
            raise ValueError(f"{spot}: 'equals' takes a scalar, got {value!r}")
        if (fact, op) in seen:
            raise ValueError(
                f"{spot} repeats ({fact!r}, {op!r}); one bound per fact and op, or two bounds "
                "disagree about one cell")
        seen.add((fact, op))
        out.append((str(fact), str(op), value))
    return tuple(out)


def _require_receipt_path(value: Any, where: str) -> str:
    """A repository path under :data:`EVIDENCE_RECEIPT_ROOT`, or raise.

    The grammar only: the wheel does not ship docs, so whether the file exists
    is a tree test's business (``tests/test_cell_evidence.py``).
    """
    if (not isinstance(value, str) or not value.startswith(EVIDENCE_RECEIPT_ROOT)
            or len(value) == len(EVIDENCE_RECEIPT_ROOT)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(
            f"{where}.receipt must be a repository path under {EVIDENCE_RECEIPT_ROOT!r} "
            f"(the receipt that holds the number and its caveats), got {value!r}")
    return value


def derive_evidence_grade(cell: Mapping[str, Any]) -> str:
    """The grade a cell's ``evidence.kl`` entries DERIVE, from kinds alone.

    Every entry is in the cell's own regime (:func:`cell_evidence` refuses
    any other), so the derivation is: no entry, ``route_only`` -- the census
    attests dispatch and nothing attests quality in this regime; any top-K
    intersection bound, ``kl_lower_bound``; any full-vocabulary KL,
    ``kl_full_vocab``.  ``qualification`` is deliberately not overloaded
    with this: one home per fact.
    """
    kinds = {entry["kind"] for entry in cell["evidence"]["kl"]}
    if "full_vocab" in kinds:
        return "kl_full_vocab"
    if "topk_intersection_lower_bound" in kinds:
        return "kl_lower_bound"
    return "route_only"


def cell_evidence(cell: Mapping[str, Any], where: str = "lane_eligibility cell",
                  regimes: Any = None) -> dict[str, Any]:
    """The parsed ``evidence`` a cell states, or raise (#133).

    Closed at every level -- a prose ``detail`` beside the entries is exactly
    the field a gate cannot read.  ``kl`` entries are ``{kind, top_k, regime,
    execution_modes, receipt}``: ``kind`` from :data:`EVIDENCE_KL_KINDS`,
    ``top_k`` a positive integer for a top-K bound and ``null`` for a
    full-vocabulary KL, ``regime`` the CELL'S OWN regime (a prefill bound
    written into a decode cell is the confusion this field exists to refuse),
    ``execution_modes`` a non-empty distinct subset of the cell's, ``receipt``
    a repository path.  ``smoke`` is ``{status, receipt}`` with ``status``
    from :data:`EVIDENCE_SMOKE_STATUSES` and a receipt exactly when a smoke
    was recorded.  ``grade`` must equal :func:`derive_evidence_grade`.
    """
    payload = cell.get("evidence")
    at = f"{where}.evidence"
    _require_keys(payload, at, required={"grade", "kl", "smoke"})
    grade = payload["grade"]
    if grade not in EVIDENCE_GRADES:
        raise ValueError(f"{at}.grade {grade!r} is not one of {list(EVIDENCE_GRADES)}")
    entries = payload["kl"]
    if not isinstance(entries, list) or any(not isinstance(e, Mapping) for e in entries):
        raise ValueError(
            f"{at}.kl must be a JSON array of {{kind, top_k, regime, execution_modes, receipt}} "
            f"objects, got {entries!r}")
    cell_modes = None
    if "runtime" in cell:
        _, cell_modes = cell_runtime_scope(cell, where)
    cell_regime = cell.get("regime")
    seen: set[tuple] = set()
    parsed_kl = []
    for i, entry in enumerate(entries):
        spot = f"{at}.kl[{i}]"
        _require_keys(entry, spot,
                      required={"kind", "top_k", "regime", "execution_modes", "receipt"})
        kind = entry["kind"]
        if kind not in EVIDENCE_KL_KINDS:
            raise ValueError(f"{spot}.kind {kind!r} is not one of {list(EVIDENCE_KL_KINDS)}")
        top_k = entry["top_k"]
        if kind == "topk_intersection_lower_bound":
            if not _is_int(top_k) or top_k <= 0:
                raise ValueError(
                    f"{spot}.top_k must be a positive integer for a top-K intersection bound, "
                    f"got {top_k!r}")
        elif top_k is not None:
            raise ValueError(
                f"{spot}.top_k must be null for a full-vocabulary KL, got {top_k!r}")
        regime = entry["regime"]
        if regimes is not None and regime not in regimes:
            raise ValueError(
                f"{spot}.regime {regime!r} is not a declared regime {sorted(regimes)} "
                "(the contract's own word, not the census phase name)")
        if cell_regime is not None and regime != cell_regime:
            raise ValueError(
                f"{spot}.regime {regime!r} is not the cell's regime {cell_regime!r}: a bound "
                "scored in another regime is another cell's evidence, and reading it here is "
                "how a prefill number came to stand in for decode quality")
        modes = entry["execution_modes"]
        if (not isinstance(modes, list) or not modes
                or any(not isinstance(m, str) or m not in EXECUTION_MODES for m in modes)
                or len(set(modes)) != len(modes)):
            raise ValueError(
                f"{spot}.execution_modes must be a non-empty list of distinct modes from "
                f"{list(EXECUTION_MODES)}, got {modes!r}")
        if cell_modes is not None:
            outside = sorted(set(modes) - set(cell_modes))
            if outside:
                raise ValueError(
                    f"{spot} claims execution_modes {outside} the cell does not cover "
                    f"({list(cell_modes)}); a KL under a mode the census never joined attests "
                    "a runtime this cell does not scope")
        receipt = _require_receipt_path(entry["receipt"], spot)
        key = (kind, top_k, regime, tuple(sorted(modes)), receipt)
        if key in seen:
            raise ValueError(f"{spot} repeats an entry; the field is a set of receipts")
        seen.add(key)
        parsed_kl.append({"kind": kind, "top_k": top_k, "regime": regime,
                          "execution_modes": list(modes), "receipt": receipt})
    smoke = payload["smoke"]
    _require_keys(smoke, f"{at}.smoke", required={"status", "receipt"})
    status = smoke["status"]
    if status not in EVIDENCE_SMOKE_STATUSES:
        raise ValueError(
            f"{at}.smoke.status {status!r} is not one of {list(EVIDENCE_SMOKE_STATUSES)}")
    if status == "not_recorded":
        if smoke["receipt"] is not None:
            raise ValueError(
                f"{at}.smoke: status not_recorded names a receipt {smoke['receipt']!r}; a "
                "receipt is where a recorded smoke lives, so one here says the status is wrong")
        smoke_receipt = None
    else:
        if smoke["receipt"] is None:
            raise ValueError(
                f"{at}.smoke: status {status!r} names no receipt; a smoke nobody recorded "
                "is not_recorded")
        smoke_receipt = _require_receipt_path(smoke["receipt"], f"{at}.smoke")
    derived = derive_evidence_grade({"evidence": {"kl": parsed_kl}})
    if grade != derived:
        raise ValueError(
            f"{at}.grade is {grade!r} but its kl entries derive {derived!r}; the grade is read "
            "off the entries, never asserted beside them")
    return {"grade": grade, "kl": parsed_kl, "smoke": {"status": status, "receipt": smoke_receipt}}


def _lanes_a_rung_reaches(route: str, contract: Mapping[str, Any], wire: Mapping[str, Any],
                          rates: "tuple[int, ...]") -> tuple[str, ...]:
    """Which of ``route``'s extension lanes can read a rung, by the published predicate.

    The predicate is the one the extension itself publishes at
    ``native_extensions[].lane.requires`` -- column rates, window bits, body and
    plane -- which ``tests/test_lane_reachability.py`` ties to
    ``kernel_window_gemv``'s own constants.  So a cell whose ``executes`` names
    a lane launch is bound to the kernel's constants transitively, and the day
    the kernel drops a rate the contract stops validating.
    """
    from .scheme import lane_rate_report

    out = []
    for entry in contract["native_extensions"]:
        lane = entry.get("lane")
        if not lane or route not in entry.get("routes", ()):
            continue
        requires = lane.get("requires") or {}
        if not lane_rate_report(entry["module_name_prefix"], rates, contract)["reachable"]:
            continue
        if int(wire["window_bits"]) not in [int(b) for b in requires.get("window_bits", ())]:
            continue
        if str(wire["body"]) != str(requires.get("body")):
            continue
        if str(wire["plane"]) != str(requires.get("plane")):
            continue
        out.append(entry["module_name_prefix"])
    return tuple(out)


def _validate_cell_executes(cell: Mapping[str, Any], route: str, entry: Mapping[str, Any],
                            contract: Mapping[str, Any], where: str) -> None:
    """``executes`` must BE the launches this build makes, not agree with them.

    Principle 14 applied to the launch: until schema v4 a cell said which A-side
    contract ran and which rungs a receipt covered, and the only place the
    LAUNCH appeared was the cell's ``id`` -- so the contract's machine-readable
    answer to "what does an E4M3 decode execute" was the materialised FP8 pair
    in every case, which stopped being true the moment an artifact at a
    lane-readable rung was served (#111).  The value is derived here from
    ``scheme.ROUTE_LAUNCHES``, the table the routes' own ``census_expected``
    is built from, narrowed by exactly the axes the cell already carries: the
    structure, regime, the residency its serve flag names, and the lanes each of its
    rungs can reach.
    """
    from ..grammar import rate_set, root_from_q256
    from .scheme import launch_pairs

    wires = {int(w["q256"]): w for w in entry["attested_wire"]}
    # The family's own published terminal rate, so a rung above what this
    # family can encode raises here rather than resolving to a rate set.
    cap = int(entry["native_terminal_q256"]) // 256
    modes = cell_residency_modes(cell, where)
    want: set = set()
    for rung in cell["rungs_q256"]:
        rates = rate_set(root_from_q256(int(rung)), cap=cap)
        lanes = _lanes_a_rung_reaches(route, contract, wires[int(rung)], rates)
        for mode in modes:
            want |= launch_pairs(route, structure=cell["structure"],
                                 regime=cell["regime"], mode=mode, lanes=lanes)
    got = cell_executes(cell)
    if got != want:
        raise ValueError(
            f"{where}.executes is {sorted(got)} but the {route} route makes "
            f"{sorted(want)} for structure {cell['structure']!r} in the "
            f"{cell['regime']!r} regime at residency {list(modes)} "
            f"on rung(s) {list(cell['rungs_q256'])} (tessera.serving.scheme.ROUTE_LAUNCHES, "
            "the table the routes' own census_expected is derived from). A cell states what "
            "the runtime EXECUTES; it is derived from the dispatch's table or it is a claim "
            "about a runtime nobody read.")


#: ``formats[]`` family -> the ``scheme.ROUTES`` key that serves it.  Two names
#: for one thing, and they are deliberately different: the contract's family is
#: a PAYLOAD name a producer prices (grid + arity), the route key is what the
#: tile IS on the hardware.
_FAMILY_TO_ROUTE = {
    "TESSERA_E2M1_K2": "TESSERA_NVFP4",
    "TESSERA_E4M3_K1": "TESSERA_FP8",
    "TESSERA_BF16_K1": "TESSERA_BF16",
}

#: The inverse, and it is a map a CONSUMER needs: a route record's ``policy``
#: names the ROUTE (``TESSERA_FP8:streamed``) while a ``lane_eligibility``
#: cell names the PAYLOAD FAMILY (``TESSERA_E4M3_K1``), so joining a served
#: record to the cell that covers it goes through here rather than through a
#: second table in the tool.  Derived, and asserted injective at import: the
#: day one route publishes two families, this map stops being the join and a
#: caller must say which family it means.
PAYLOAD_FAMILY_BY_ROUTE = {route: family for family, route in _FAMILY_TO_ROUTE.items()}
assert len(PAYLOAD_FAMILY_BY_ROUTE) == len(_FAMILY_TO_ROUTE), (
    "two payload families share a route, so route -> family is no longer a join")

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


# ---------------------------------------------------------------------------
# construction: which Linears the runtime OFFERS to a quant config at all
# ---------------------------------------------------------------------------
#
# The rest of this file answers "what does the plugin EXECUTE".  This block
# answers a question that comes earlier and that the plugin structurally cannot
# answer for itself: **is the plugin asked about this module at all**.
#
# ``LinearBase.__init__`` takes ``UnquantizedLinearMethod()`` in the
# ``quant_config is None`` branch *without calling* ``get_quant_method``
# (vLLM 0.28, ``model_executor/layers/linear.py:258``).  A model implementation
# that builds a projection with ``quant_config=None`` therefore takes vLLM's own
# BF16 method, and no plugin can refuse, warn, or even see the prefix.  On
# ``Glm5NextForConditionalGeneration`` that is every attention projection, the
# whole KDA layer, the sparse indexer and the entire vision tower.
#
# A producer that writes a wire there deletes the ``<module>.weight`` the
# runtime wants and puts bytes in its place that nothing decodes.  So the
# producer needs this fact BEFORE it encodes -- and principle 14 says it is
# derived from the runtime, never hand-kept beside it.  ``tools/
# tessera_construction_census.py`` derives it, by building the model the way the
# loader does with a probe quant config that records every prefix it is offered;
# the receipts it writes live in ``docs/measurements/construction/`` and this
# block is generated from them by :func:`construction_entry_from_receipt`,
# which ``tests/test_serving_construction.py`` re-derives and compares -- the
# same "table DERIVED from the code path, not typed beside it" rule
# ``native_extensions`` follows.

CONSTRUCTION_SCHEMA = "tessera.construction.v1"
CONSTRUCTION_CENSUS_SCHEMA = "tessera.construction-census.v1"

#: Any purely numeric path segment in a module prefix.  Must match
#: ``tools/tessera_construction_census.py``'s ``NUMERIC_SEGMENT``: a repeated
#: block is a repeated block whether the model spells its stack ``layers.N`` or
#: ``blocks.N``, and the census and the lookup have to normalise identically or
#: the join is silently empty.
_NUMERIC_SEGMENT = re.compile(r"(?<=\.)\d+(?=\.|$)")


def normalise_module(prefix: str) -> str:
    """A module prefix with its repeat indices collapsed to ``*``."""
    return _NUMERIC_SEGMENT.sub("*", prefix)


def construction_entry_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The contract row a construction census receipt implies.

    Generated, not transcribed: the receipt is the observation, this is the
    subset a gate reads, and the test that compares them is what keeps the two
    from drifting.
    """
    if receipt.get("schema") != CONSTRUCTION_CENSUS_SCHEMA:
        raise ValueError(
            f"construction census must declare schema {CONSTRUCTION_CENSUS_SCHEMA!r}, "
            f"got {receipt.get('schema')!r}")
    architectures = list(receipt["model"].get("architectures") or ())
    if len(architectures) != 1:
        raise ValueError(
            f"a construction census covers exactly one architecture; {architectures} "
            "does not say which module tree was walked")
    linears = receipt["linears"]
    return {
        "architecture": architectures[0],
        "runtime": {key: receipt["runtime"][key] for key in ("image", "image_id", "vllm")},
        "model": {key: receipt["model"][key] for key in
                  ("model_type", "num_hidden_layers", "layer_types", "mlp_layer_types")},
        "supports_quant": bool(receipt["supports_quant"]),
        "hf_to_vllm_mapper_unstacked": receipt["hf_to_vllm_mapper_unstacked"] or {},
        "offered": sorted(row["prefix_pattern"] for row in linears
                          if row["offered_to_quant_config"]),
        "never_offered": [
            {"prefix_pattern": row["prefix_pattern"], "class": row["class"],
             "quant_method": row["quant_method"]}
            for row in sorted(linears, key=lambda r: r["prefix_pattern"])
            if not row["offered_to_quant_config"]],
        "offered_non_linear": [dict(row) for row in receipt["offered_non_linear"]],
    }


def _validate_construction(block: Any, where: str) -> None:
    """Refuse a construction block a gate could misread."""
    _require_keys(block, where, required={"schema", "architectures"}, optional={"note"})
    if block["schema"] != CONSTRUCTION_SCHEMA:
        raise ValueError(f"{where}.schema must be {CONSTRUCTION_SCHEMA!r}")
    seen: set[str] = set()
    for i, entry in enumerate(block["architectures"]):
        row = f"{where}.architectures[{i}]"
        _require_keys(entry, row,
                      required={"architecture", "runtime", "model", "supports_quant",
                                "hf_to_vllm_mapper_unstacked", "offered", "never_offered",
                                "offered_non_linear"},
                      optional={"receipt"})
        name = entry["architecture"]
        if name in seen:
            raise ValueError(
                f"{row}: {name!r} is censused twice. Two censuses of one architecture are two "
                "claims about one runtime; merge them or drop the stale one")
        seen.add(name)
        overlap = sorted(set(entry["offered"]) &
                         {r["prefix_pattern"] for r in entry["never_offered"]})
        if overlap:
            raise ValueError(
                f"{row}: {overlap} are both offered and never offered; a census cannot say both")
        for key in ("image", "image_id", "vllm"):
            if not entry["runtime"].get(key):
                raise ValueError(
                    f"{row}.runtime.{key} is empty: a construction answer is a property of the "
                    "image it was observed in, and an unstamped one cannot be checked against "
                    "the image a serve actually runs")


def construction_entry(architectures, contract: Mapping[str, Any] | None = None):
    """The censused entry for a checkpoint's ``architectures``, or ``None``."""
    payload = contract if contract is not None else load_serving_contract()
    block = payload.get("construction")
    if not block:
        return None
    wanted = [architectures] if isinstance(architectures, str) else list(architectures or ())
    for entry in block["architectures"]:
        if entry["architecture"] in wanted:
            return entry
    return None


#: The ``WeightsMapper`` fields ``vllm_module_name`` replays, in vLLM's own
#: order (``_map_name_with_shard``, vLLM 0.28 ``model_executor/models/
#: utils.py``).  Everything else in that dataclass -- ``orig_to_new_renaming``,
#: ``orig_to_new_regex``, ``orig_to_new_stacked``, and whatever vLLM adds next
#: -- is REFUSED rather than skipped, because a rule silently not applied is a
#: wrong module name and a wrong module name is a dead wire.
_MAPPER_FIELDS_REPLAYED = ("orig_to_new_substr", "orig_to_new_prefix", "orig_to_new_suffix")


def _require_replayable_mapper(table: Mapping[str, Any]) -> None:
    """Refuse a mapper table carrying a rule this producer cannot replay.

    Principle 14 has no exemption for "the algorithm is code, not a table".
    vLLM publishes the TABLE (the census reads it off the model class); the
    ALGORITHM that consumes it lives in ``WeightsMapper._map_name_with_shard``
    and cannot be derived from the table.  So the producer replays the three
    fields whose semantics are pinned in the pure suite and attested against
    the real mapper in ``tests/test_serving_name_mapping.py`` -- and refuses,
    by name, on every other field.

    ``orig_to_new_stacked`` is in the refused set on purpose.  vLLM applies it
    inside ``_map_name``, and ``get_unstacked_mapper()`` -- the variant
    ``configure_quant_config`` hands a quant config -- empties it.  A receipt
    field named ``hf_to_vllm_mapper_unstacked`` carrying a populated stacked
    map therefore says the census fell back to the raw mapper, which is a
    contradiction in the receipt rather than a case to implement.
    """
    unknown = sorted(field for field, value in table.items()
                     if value and field not in _MAPPER_FIELDS_REPLAYED)
    if unknown:
        raise ValueError(
            f"the census recorded hf_to_vllm_mapper fields this producer does not replay: "
            f"{unknown}. vLLM applies them in WeightsMapper._map_name_with_shard, so the "
            f"vLLM-side module name computed here would be wrong and the wire written to it "
            f"dead. Extend vllm_module_name AND its attestation in "
            f"tests/test_serving_name_mapping.py, or export this architecture through a "
            f"runtime whose mapper uses only {list(_MAPPER_FIELDS_REPLAYED)}.")


def vllm_module_name(entry: Mapping[str, Any], checkpoint_module: str) -> str:
    """A checkpoint module name in the namespace the runtime builds it under.

    A line-for-line replay of vLLM's ``WeightsMapper._map_name_with_shard``
    over the three fields ``_MAPPER_FIELDS_REPLAYED`` names, in its order and
    with its semantics: a substring rule replaces ONE occurrence
    (``key.replace(substr, new_key, 1)``), and the prefix and suffix loops fall
    THROUGH -- each rule sees the key the previous rule rewrote.  Only the
    unstacked fields matter, which is exactly what ``configure_quant_config``
    hands a quant config and therefore exactly what
    ``TesseraConfig.apply_vllm_mapper`` will apply to this name at load.

    The replay is a claim about another runtime, so it is attested rather than
    asserted: ``tests/test_serving_name_mapping.py`` runs these same names
    through the real ``WeightsMapper`` inside the serving image and fails on
    any disagreement, and ``_require_replayable_mapper`` refuses a table
    carrying a rule the replay does not cover.
    """
    table = entry.get("hf_to_vllm_mapper_unstacked") or {}
    _require_replayable_mapper(table)
    name = checkpoint_module

    def _dropped() -> None:
        # vLLM's WeightsMapper spells "discard this weight" as a None
        # replacement (``_map_name`` returns None and ``apply_list`` drops the
        # entry).  A module the runtime maps away is a module it never builds,
        # so a wire written there is dead -- the same refusal
        # ``TesseraConfig.apply_vllm_mapper`` raises at load, taken here at
        # export instead.
        raise ValueError(
            f"the runtime's hf_to_vllm_mapper DROPS {checkpoint_module!r}, so it builds no "
            "module for this name at all. A wire written here is dead weight; that is a "
            "refusal, not a warning.")

    for old, new in (table.get("orig_to_new_substr") or {}).items():
        if old in name:
            if new is None:
                _dropped()
            name = name.replace(old, new, 1)
    for old, new in (table.get("orig_to_new_prefix") or {}).items():
        if name.startswith(old):
            if new is None:
                _dropped()
            name = name.replace(old, new, 1)
    for old, new in (table.get("orig_to_new_suffix") or {}).items():
        if name.endswith(old):
            if new is None:
                _dropped()
            name = new.join(name.rsplit(old, 1))
    return name


def classify_construction(entry: Mapping[str, Any], checkpoint_module: str) -> tuple[str, str]:
    """``(verdict, vllm module pattern)`` for one checkpoint module name.

    ``offered`` -- the runtime builds this module and offers it to the quant
    config, so a wire written here is executed.  ``never_offered`` -- it builds
    it with ``quant_config=None``, so the plugin is never asked and the wire is
    dead.  ``absent`` -- the census walked the whole module tree and this name
    is not a module the runtime builds at all (a fused role named at its leaf,
    a name from another architecture).  The last two are the same outcome for a
    producer and are told apart only so the refusal can say which it is.

    ``offered`` IS TWO LISTS, because "was this prefix offered a quant config"
    and "is this prefix a Linear" are different questions and the census
    records them separately.  ``offered`` holds the ``LinearBase`` rows;
    ``offered_non_linear`` holds every prefix the probe config WAS asked about
    that is not one -- the LM head, and the ``RoutedExperts`` stack a routed
    MoE layer builds.  An expert stack read only the first list resolved
    ``absent``, which would have refused the one module the expert route
    exists to serve while the census receipt said, in the same file, that the
    runtime asks about it.  The census is the attestation either way
    (principle 14); this reads all of what it wrote.
    """
    pattern = normalise_module(vllm_module_name(entry, checkpoint_module))
    if pattern in set(entry["offered"]):
        return "offered", pattern
    if pattern in {row["prefix_pattern"] for row in entry.get("offered_non_linear", ())}:
        return "offered", pattern
    if pattern in {row["prefix_pattern"] for row in entry["never_offered"]}:
        return "never_offered", pattern
    return "absent", pattern
