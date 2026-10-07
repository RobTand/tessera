"""Does a full-engine capture's evidence say its native routes ran by module kind?

WHAT THIS QUALIFIES.  A resource capture measures allocation, and two decoders
of the same bytes do not allocate alike, so a ledger is about a decoder and
the record has to say which one.  The serve's own ``TESSERA_ROUTE_TRACE``
histogram (``telemetry._RouteTrace``) keys every served dispatch on
``(policy, shape, symbol, decoder, contract, kind)``; the qualification reads
that file and refuses unless every dispatch on every family the artifact
carries is the ONE launch that family's declared module kind makes. Manifest
``dense`` (including the legacy absent structure) maps to trace ``dense``;
``routed_moe`` maps to trace ``moe``. Neither is inferred from module names.

THE LAUNCHES. Each route can dispatch through several current native lanes.
The data below mirrors the route owners and includes experimental launches for capture checks.
The producer contract publishes those pairs in scheme.ROUTE_LAUNCHES.
The mixed-kind regression checks this mirror against the producer without a fixed roster.
The BF16 decode-once pair preserves folded weights and uses shared scratch on each admitted step.
Capture qualification does not promote an experimental serving cell.
This module reads completed JSON and does not import Torch, vLLM or Tessera.

LIBRARY OBSERVATION. Dense window routes can load the fused CUDA libraries.
The capture also records any mapped GEMV library.
A mapped library does not prove that a module used it.
The dispatch histogram is the proof this qualifier reads.
The library census remains a recorded observation, not a qualification gate.

it served: within ONE M every module of the contract appears exactly once, so
the count is the sum within an M group, maximised over the M groups (the
2026-09-18 capture: 110 per M group where a max over keys returned 28).
``M*`` marks a record written under ``torch.compile`` tracing and is refused,
never counted.  Since ``identity_version`` 1 each entry also names the modules
it counted (``module_names``) and how many it could not name
(``unnamed_modules``); an unnamed module is refused, and when the caller
supplies the manifest's names the two sets must agree.

A missing or unreadable trace is NOT VERIFIED, and not verified is a refusal,
never a pass.  Nothing here imports Torch, vLLM or ``tessera``.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

__all__ = [
    "WINDOW_GEMM_SYMBOL",
    "FUSED_WINDOW_DENSE_SYMBOL",
    "A4_DENSE_GEMM_SYMBOL",
    "NATIVE_WINDOW_GEMM_DECODER",
    "NATIVE_WINDOW_GEMM_FOLDED_DECODER",
    "NATIVE_FUSED_WINDOW_DENSE_DECODER",
    "NATIVE_FUSED_WINDOW_DENSE_FOLDED_DECODER",
    "NATIVE_SPAN2_GEMM_DECODER",
    "FP8_ACTIVATION_CONTRACT",
    "BF16_ACTIVATION_CONTRACT",
    "NVFP4_ACTIVATION_CONTRACT",
    "DENSE_LAUNCHES",
    "MOE_LAUNCHES",
    "KIND_LAUNCHES",
    "expected_module_kinds",
    "WINDOW_GEMV_LIBRARY_GLOB",
    "QUALIFICATION_SCHEMA",
    "mapped_native_libraries",
    "trace_launches_by_contract",
    "qualify_native_route",
    "QualificationRefused",
    "refusal_record",
]

#: ``scheme.WINDOW_GEMM_SYMBOL`` / ``scheme.FUSED_WINDOW_DENSE_SYMBOL`` /
#: ``scheme.A4_DENSE_GEMM_SYMBOL``.
WINDOW_GEMM_SYMBOL = "tessera::window_gemm_dense"
FUSED_WINDOW_DENSE_SYMBOL = "tessera::fused_window_dense"
A4_DENSE_GEMM_SYMBOL = "tessera.kernel_a4.a4_span2_gemm"
#: ``telemetry.DECODER_NATIVE_WINDOW_GEMM`` / ``DECODER_NATIVE_WINDOW_GEMM_FOLDED``
#: / ``DECODER_NATIVE_FUSED_WINDOW_DENSE`` / ``DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED``
#: / ``DECODER_NATIVE_SPAN2_GEMM``.
NATIVE_WINDOW_GEMM_DECODER = "native_window_gemm"
NATIVE_WINDOW_GEMM_FOLDED_DECODER = "native_window_gemm_folded"
NATIVE_FUSED_WINDOW_DENSE_DECODER = "native_fused_window_dense"
NATIVE_FUSED_WINDOW_DENSE_FOLDED_DECODER = "native_fused_window_dense_folded"
NATIVE_SPAN2_GEMM_DECODER = "native_span2_gemm"
#: Contract v46: the E4M3 family's tensor-core instruction library
#: (``routed_fused.library_for``), experimental; TESSERA_FP8 only.
NATIVE_FUSED_WINDOW_DENSE_E4M3MMA_DECODER = "native_fused_window_dense_e4m3mma"
#: Contract v56 (tessera#931): ``scheme.DECODE_ONCE_DENSE_SYMBOL`` /
#: ``telemetry.DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3``, experimental,
#: default-off, resident only; TESSERA_FP8 only.
DECODE_ONCE_DENSE_SYMBOL = "tessera.serving.e4m3_prefill.prefill_apply"
NATIVE_WINDOW_DECODE_ONCE_E4M3_DECODER = "native_window_decode_once_e4m3"
#: The default-off BF16 per-step scratch lane, on unchanged folded arithmetic.
BF16_DECODE_ONCE_DENSE_SYMBOL = "tessera.serving.bf16_prefill.prefill_apply"
NATIVE_WINDOW_DECODE_ONCE_BF16_FOLDED_DECODER = "native_window_decode_once_bf16_folded"
#: ``scheme.{FP8,BF16,NVFP4}_ACTIVATION_CONTRACT``.
FP8_ACTIVATION_CONTRACT = "fp8_per_token_dynamic"
BF16_ACTIVATION_CONTRACT = "bf16_unquantized"
NVFP4_ACTIVATION_CONTRACT = "e2m1_group16_ue4m3_static"

#: family -> (activation contract, EVERY (symbol, decoder) pair its dense
#: route may stamp).  ``fp8_route.DENSE_LAUNCHES``, ``bf16_route.DENSE_LAUNCHES``
#: and ``nvfp4_route.process_weights_after_loading`` (``tessera_symbol`` /
#: ``tessera_decoder``) are the owners; ``scheme.ROUTE_LAUNCHES`` publishes
#: the same pairs and the contract test ties them.  Since contract v43 the two
#: window families carry two: the Triton window GEMM and the fused window
#: kernel's dense identity, which ``native_window.prepare_dense_native_module``
#: picks per module on the module's own wire
#: (``routed_fused.fused_dense_window_supported``).  A pair outside the tuple
#: is a foreign launch and refuses the capture.
DENSE_LAUNCHES = {
    "TESSERA_FP8": (FP8_ACTIVATION_CONTRACT, (
        (WINDOW_GEMM_SYMBOL, NATIVE_WINDOW_GEMM_DECODER),
        (FUSED_WINDOW_DENSE_SYMBOL, NATIVE_FUSED_WINDOW_DENSE_DECODER),
        (FUSED_WINDOW_DENSE_SYMBOL, NATIVE_FUSED_WINDOW_DENSE_E4M3MMA_DECODER),
        (DECODE_ONCE_DENSE_SYMBOL, NATIVE_WINDOW_DECODE_ONCE_E4M3_DECODER))),
    "TESSERA_BF16": (BF16_ACTIVATION_CONTRACT, (
        (WINDOW_GEMM_SYMBOL, NATIVE_WINDOW_GEMM_FOLDED_DECODER),
        (FUSED_WINDOW_DENSE_SYMBOL, NATIVE_FUSED_WINDOW_DENSE_FOLDED_DECODER),
        (BF16_DECODE_ONCE_DENSE_SYMBOL, NATIVE_WINDOW_DECODE_ONCE_BF16_FOLDED_DECODER))),
    "TESSERA_NVFP4": (NVFP4_ACTIVATION_CONTRACT, (
        (A4_DENSE_GEMM_SYMBOL, NATIVE_SPAN2_GEMM_DECODER),)),
}

#: family -> (activation contract, EVERY (symbol, decoder) pair its routed
#: route may stamp, resident only), as scheme.ROUTE_LAUNCHES publishes them.
#: Since contract v42 (tessera#640) the two window families carry two: the
#: compact adapter and the fused routed window lane, which
#: ``PackedWindowMoeBundles.adapter`` picks per module on the bundle's own
#: shape (``routed_fused.fused_routed_window_supported``).  A served MoE
#: family may therefore dispatch on either or both; a pair outside this
#: tuple is a foreign launch and refuses the capture.
COMPACT_WINDOW_MOE_SYMBOL = "tessera.native_window_moe.NativeWindowMoE.__call__"
FUSED_WINDOW_MOE_SYMBOL = "tessera.routed_fused.FusedRoutedWindowMoE.__call__"
MOE_LAUNCHES = {
    "TESSERA_FP8": (FP8_ACTIVATION_CONTRACT, (
        (COMPACT_WINDOW_MOE_SYMBOL, "native_window_moe_compact"),
        (FUSED_WINDOW_MOE_SYMBOL, "native_routed_fused_window"),
        (FUSED_WINDOW_MOE_SYMBOL, "native_routed_fused_window_e4m3mma"))),
    "TESSERA_BF16": (BF16_ACTIVATION_CONTRACT, (
        (COMPACT_WINDOW_MOE_SYMBOL, "native_window_moe_compact_folded"),
        (FUSED_WINDOW_MOE_SYMBOL, "native_routed_fused_window_folded"))),
    "TESSERA_NVFP4": (NVFP4_ACTIVATION_CONTRACT, (
        ("tessera.kernel_a4.a4_span2_grouped_gemm", "native_span2_grouped"),)),
}
#: kind -> family -> (contract, admissible pairs); both kinds read the same way.
KIND_LAUNCHES = {
    "dense": DENSE_LAUNCHES,
    "moe": MOE_LAUNCHES,
}

#: ``ext.NATIVE_EXTENSIONS[0]["filename_glob"]``: the one extension the package
#: still builds.  Recorded, not required -- no dense launch names its lane.
WINDOW_GEMV_LIBRARY_GLOB = "tessera_window_gemv*.so"

QUALIFICATION_SCHEMA = "tessera.step4_native_route_qualification.v2"


class QualificationRefused(Exception):
    """The capture's evidence does not establish the native routes."""


def mapped_native_libraries(runtime_observation, glob=WINDOW_GEMV_LIBRARY_GLOB):
    """``{path: sha256}`` of the mapped libraries whose basename matches ``glob``.

    ``base.native_libraries`` is the worker process's actual mapped shared
    objects (``full_engine_worker.native_library_observation`` over
    ``/proc/self/maps``), each with its sha256.
    """
    libraries = runtime_observation["base"]["native_libraries"]
    matched = {}
    for path, value in libraries.items():
        if not fnmatch.fnmatch(Path(path).name, glob):
            continue
        # ``base.native_libraries`` maps a path to its sha256 string; the
        # worker's own ``native_library_observation`` keeps {sha256, bytes}.
        # Read both shapes rather than requiring one, and refuse anything else.
        if isinstance(value, str):
            matched[path] = value
        elif isinstance(value, dict) and isinstance(value.get("sha256"), str):
            matched[path] = value["sha256"]
        else:
            raise QualificationRefused(
                f"mapped library {path} carries no readable sha256; it cannot be recorded")
    return matched


def _pair_key(symbol, decoder):
    return f"{symbol} / {decoder}"


def trace_launches_by_contract(route_trace, contract, *, policy=None, kind=None):
    """``{"<symbol> / <decoder>": {...}}`` for one activation contract.

    Each value carries ``symbol``, ``decoder``, ``launches`` (summed over
    every entry), ``entries``, ``modules`` (the per-M-group maximum described
    in the module docstring), ``module_names`` (the union, sorted) and
    ``unnamed_modules`` (the per-M-group maximum).  ``policy``, when given,
    is the ``<family>:<mode>`` stamp every entry on the contract must carry;
    another policy on the same contract is refused, because a contract served
    under a residency the configuration did not name is not the serve the
    configuration describes. ``kind`` restricts the counted dispatches after
    policy validation; the caller separately checks the complete kind roster.
    """
    entries = route_trace.get("entries")
    if entries is None:
        raise QualificationRefused("route trace carries no entries; the dispatch leg is not verified")
    totals = {}
    per_m = {}
    seen_contract = False
    for entry in entries:
        if entry.get("contract") != contract:
            continue
        if policy is not None and entry.get("policy") != policy:
            raise QualificationRefused(
                f"route-trace entry on {contract} carries policy {entry.get('policy')!r}, the "
                f"configuration serves {policy!r}: {entry!r}")
        if kind is not None and entry.get("kind") != kind:
            continue
        seen_contract = True
        symbol = entry.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise QualificationRefused(
                f"route-trace entry on {contract} names no symbol: {entry!r}")
        decoder = entry.get("decoder")
        if not isinstance(decoder, str) or not decoder:
            raise QualificationRefused(
                f"route-trace entry on {contract} names no decoder: {entry!r}")
        if "launches" not in entry:
            raise QualificationRefused(
                f"route-trace entry on {contract} counts no launches: {entry!r}")
        shape = entry.get("shape")
        if not isinstance(shape, str) or not shape:
            raise QualificationRefused(
                f"route-trace entry on {contract} names no shape: {entry!r}")
        token = shape.split(":")[0]
        if token == "M*":
            raise QualificationRefused(
                f"route-trace entry on {contract} was written under torch.compile tracing "
                f"({shape}), where a count is not a launch count: {entry!r}")
        key = _pair_key(symbol, decoder)
        bucket = totals.setdefault(key, {"symbol": symbol, "decoder": decoder, "launches": 0,
                                         "modules": 0, "entries": 0, "module_names": set(),
                                         "unnamed_modules": 0})
        bucket["launches"] += int(entry["launches"])
        bucket["entries"] += 1
        names = entry.get("module_names")
        if names is not None:
            if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
                raise QualificationRefused(
                    f"route-trace entry on {contract} carries unreadable module_names: {entry!r}")
            bucket["module_names"].update(names)
        group = per_m.setdefault((key, token), [0, 0])
        group[0] += int(entry.get("modules", 0))
        group[1] += int(entry.get("unnamed_modules", 0))
    for (key, _token), (count, unnamed) in per_m.items():
        totals[key]["modules"] = max(totals[key]["modules"], count)
        totals[key]["unnamed_modules"] = max(totals[key]["unnamed_modules"], unnamed)
    for bucket in totals.values():
        bucket["module_names"] = sorted(bucket["module_names"])
    if not seen_contract:
        raise QualificationRefused(
            f"route trace records no dispatch on {contract}; the capture never served the route it prices")
    return totals


def _expected_count(expected, family):
    value = expected[family]
    if isinstance(value, dict):
        return int(value["count"]), value.get("names")
    return int(value), None


def expected_module_kinds(expected_modules):
    """Normalize legacy dense expectations or a manifest's explicit kind partition.

    A mixed family adds ``kinds: {dense|moe: {count, names}}`` beside its
    aggregate count/names. That partition must be disjoint and exhaustive.
    Explicit kinds require names; an old count-only dense caller remains valid.
    """
    result = {}
    for family, value in expected_modules.items():
        if family not in DENSE_LAUNCHES:
            raise QualificationRefused(f"the artifact names an unknown family: {family!r}")
        count, names = _expected_count(expected_modules, family)
        if count < 0:
            raise QualificationRefused(f"negative module count for {family}")
        if not isinstance(value, dict) or "kinds" not in value:
            result[family] = {"dense": {"count": count, "names": names}} if count else {}
            continue
        kinds = value["kinds"]
        if not isinstance(kinds, dict) or not kinds or set(kinds) - set(KIND_LAUNCHES):
            raise QualificationRefused(f"unknown or empty manifest module kinds for {family}: {kinds!r}")
        union = set()
        for kind, members in kinds.items():
            if not isinstance(members, dict):
                raise QualificationRefused(f"unreadable manifest module kind {family}/{kind}")
            roster, size = members.get("names"), members.get("count")
            if (type(size) is not int or size <= 0 or not isinstance(roster, list)
                    or not all(isinstance(n, str) and n for n in roster)
                    or len(roster) != size or len(set(roster)) != size or union.intersection(roster)):
                raise QualificationRefused(f"invalid or overlapping manifest module names for {family}/{kind}")
            union.update(roster)
        if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
                or len(names) != count or len(union) != count or union != set(names)):
            raise QualificationRefused(f"manifest kind partition does not cover {family}'s count/names")
        result[family] = kinds
    return result


def _qualify_kind(route_trace, *, family, kind, mode, members, require_names):
    count, names = members["count"], members.get("names")
    if kind == "moe" and mode != "resident":
        raise QualificationRefused("routed MoE has no streamed native launch")
    contract, pairs = KIND_LAUNCHES[kind][family]
    policy = f"{family}:{mode}"
    launches = trace_launches_by_contract(route_trace, contract, policy=policy, kind=kind)
    expected_keys = [_pair_key(symbol, decoder) for symbol, decoder in pairs]
    foreign = sorted(key for key in launches if key not in expected_keys)
    if foreign:
        admissible = " or ".join(expected_keys)
        raise QualificationRefused(
            f"dispatches on {contract} ({family}/{kind}) used {foreign}, not {admissible}: "
            + json.dumps({key: {k: launches[key][k] for k in ("launches", "modules")}
                          for key in foreign}, sort_keys=True))
    native = _merge_admissible_launches(launches, expected_keys)
    if native["launches"] < 1:
        raise QualificationRefused(f"no served dispatch on {contract} ({family}/{kind}) was counted")
    if native["unnamed_modules"]:
        raise QualificationRefused(
            f"{native['unnamed_modules']} module(s) dispatching on {contract} ({family}/{kind}) carried "
            "no prefix; a per-module claim needs unnamed_modules == 0")
    if native["modules"] != count:
        raise QualificationRefused(
            f"{native['modules']} modules dispatched on {contract} ({family}/{kind}), the artifact "
            f"assigns {count}; the remainder did not serve this route")
    if require_names and not native["module_names"]:
        raise QualificationRefused(f"{family}/{kind} names no modules; a kind claim needs exact prefixes")
    if names is not None and native["module_names"]:
        expected_names = sorted(set(names))
        if native["module_names"] != expected_names:
            missing = sorted(set(expected_names) - set(native["module_names"]))
            extra = sorted(set(native["module_names"]) - set(expected_names))
            raise QualificationRefused(
                f"the modules dispatching on {contract} ({family}/{kind}) are not the manifest's: "
                f"missing {missing}, unexpected {extra}")
    names_checked = bool(names is not None and native["module_names"])
    if len(pairs) == 1:
        # The one-launch record every dense reader has consumed since v1, unchanged.
        ((symbol, decoder),) = pairs
        expected = {"symbol": symbol, "decoder": decoder, "modules": count, "names_checked": names_checked}
    else:
        expected = {"launches": [{"symbol": symbol, "decoder": decoder} for symbol, decoder in pairs],
                    "modules": count, "names_checked": names_checked}
    return {"contract": contract, "policy": policy, "expected": expected, "observed": native}


def _merge_admissible_launches(launches, expected_keys):
    """One observation over the admissible pairs a kind may dispatch on.

    A module runs exactly one adapter, so the per-pair ``modules`` (each a
    per-M-group maximum) add across pairs, as do launches, entries and the
    unnamed count; ``module_names`` is the union.  ``by_launch`` keeps each
    pair's own bucket; ``symbol``/``decoder`` are set only when one pair was
    observed, so a dense record reads exactly as before.
    """
    present = [key for key in expected_keys if key in launches]
    merged = {"launches": 0, "entries": 0, "modules": 0, "unnamed_modules": 0,
              "module_names": [], "by_launch": {}}
    names = set()
    for key in present:
        bucket = launches[key]
        merged["by_launch"][key] = dict(bucket)
        for field in ("launches", "entries", "modules", "unnamed_modules"):
            merged[field] += bucket[field]
        names.update(bucket["module_names"])
    merged["module_names"] = sorted(names)
    if len(present) == 1:
        merged["symbol"] = launches[present[0]]["symbol"]
        merged["decoder"] = launches[present[0]]["decoder"]
    return merged


def qualify_dispatch(route_trace, *, mode, expected_modules):
    """Qualify exact native launches separately for every manifest family/kind.

    Legacy count/names callers describe dense modules. Explicit ``kinds``
    callers get a per-kind result and require exact trace identities.
    """
    if mode not in ("resident", "streamed"):
        raise QualificationRefused(f"unknown residency mode {mode!r}")
    families = {}
    for family, kinds in expected_module_kinds(expected_modules).items():
        if not kinds:
            continue
        contract = DENSE_LAUNCHES[family][0]
        entries = route_trace.get("entries")
        if not isinstance(entries, list):
            raise QualificationRefused("route trace carries no entries; the dispatch leg is not verified")
        for entry in entries:
            if str(entry.get("policy", "")).split(":")[0] == family and entry.get("contract") != contract:
                raise QualificationRefused(f"unexpected activation contract for {family}: {entry.get('contract')!r}")
            if entry.get("contract") == contract and entry.get("kind") not in kinds:
                raise QualificationRefused(f"unexpected route kind for {family}: {entry.get('kind')!r}")
        explicit = isinstance(expected_modules[family], dict) and "kinds" in expected_modules[family]
        claims = {kind: _qualify_kind(route_trace, family=family, kind=kind, mode=mode,
                                      members=members, require_names=explicit)
                  for kind, members in kinds.items()}
        families[family] = {"kinds": claims} if explicit else claims["dense"]
    if not families:
        raise QualificationRefused("the artifact assigns no module to any family; nothing to qualify")
    return families


def qualify_native_route(runtime_observation, route_trace, *, mode, expected_modules):
    """The dispatch leg over every family, plus the recorded library census.

    ``runtime_observation`` is the worker's ``runtime-observation.json``; its
    mapped-library census is RECORDED (was the GEMV lane's extension mapped at
    all?) and refuses nothing, because no dense launch depends on a shared
    object.  The record says so in ``scope``.
    """
    mapped = mapped_native_libraries(runtime_observation)
    families = qualify_dispatch(route_trace, mode=mode, expected_modules=expected_modules)
    return {"schema": QUALIFICATION_SCHEMA, "mode": mode,
            "families": families,
            "mapped_extension_libraries": mapped,
            "trace_identity": {key: route_trace.get(key) for key in
                               ("schema", "identity_version", "rank", "world_size", "rank_source",
                                "rank_conflict", "platform", "pid")},
            "scope": ("every counted dispatch on every family the artifact carries is the one "
                      "native launch its manifest module kind makes; no library leg -- no qualified launch on this "
                      "tree loads a cpp_extension, so a mapped library proves nothing here and is "
                      "recorded only; no timing, fixed-resource or release admission follows from it"),
            "qualified": True}


def refusal_record(phase, message, **context):
    """The record a refused run writes beside its ledgers."""
    return {"schema": "tessera.step4_capture_refusal.v1", "phase": phase,
            "refusal": message, "qualified": False,
            "scope": "the run was refused before or after measurement; nothing here is a price",
            **context}


def read_json(path):
    return json.loads(Path(path).read_text())
