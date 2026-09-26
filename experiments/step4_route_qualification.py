"""Does a full-engine capture's own evidence say the native dense routes ran?

WHAT THIS QUALIFIES.  A resource capture measures allocation, and two decoders
of the same bytes do not allocate alike, so a ledger is about a decoder and
the record has to say which one.  The serve's own ``TESSERA_ROUTE_TRACE``
histogram (``telemetry._RouteTrace``) keys every served dispatch on
``(policy, shape, symbol, decoder, contract, kind)``; the qualification reads
that file and refuses unless every dispatch on every family the artifact
carries is the ONE launch that family's dense route makes.

THE LAUNCHES.  Since ``37e89f576`` (contract v30) and ``1b767a207`` (#538),
each dense route module owns exactly one launch and stamps it at its one
``emit_route`` call:

* ``TESSERA_FP8`` and ``TESSERA_BF16`` -- ``fp8_route.DENSE_LAUNCH`` =
  ``(tessera::window_gemm_dense, native_window_gemm)`` and
  ``bf16_route.DENSE_LAUNCH`` = ``(tessera::window_gemm_dense,
  native_window_gemm_folded)``: the compact loader's packed window unit
  multiplied by ``tessera.window_gemm``'s Triton kernel through
  ``serving.native_window``, on the epilogue arithmetic for FP8 and the folded
  one for BF16 (tessera#614).  ``apply`` raises when the module was not
  prepared rather than falling back to a materialised tile.
* ``TESSERA_NVFP4`` -- ``(tessera.kernel_a4.a4_span2_gemm, native_span2_gemm)``:
  the packed span-2 planes decoded in-kernel by ``tessera.kernel_a4``, whose
  ``require_native_fp4_mma`` refuses a Triton that cannot lower block-scaled
  FP4 MMA rather than emulating it.

``scheme.ROUTE_LAUNCHES`` publishes these pairs and
``tests/test_serving_contract.py`` ties each route's ``DENSE_LAUNCH`` to the
table.  The strings are spelled here as data because this module may not
import the serving package (it reads finished JSON on any host).

WHAT RETIRED.  The pre-retirement driver proved a ``cpp_extension`` JIT
(``tessera_nvfp4_*.so``) built before the engine started and bound the
worker's mapped-library census to those bytes, because ``ext.NATIVE_EXTENSIONS``
published a silent ``torch_materialize_stock`` substitute for a container
that could not compile.  On master no dense route loads a ``cpp_extension``:
``native_window.prepare_dense_native_module`` builds the window unit through
``compact_prep.prepare_window_compact``, which constructs
``kernel_window_gemv.WindowGemvUnit`` directly and never calls that module's
``_ext()``; the only extension the package still builds
(``tessera_window_gemv``, substitute ``torch_window``) is reached from the
GEMV lane alone, which no dense dispatch makes.  The library leg therefore has
no subject; ``mapped_native_libraries`` is kept to RECORD whether that lane's
library was mapped anyway, and the dispatch leg is the whole proof.

COUNTING MODULES IS NOT A MAX AND NOT A SUM.  ``shape`` carries both the
call's M and the module's ``N:K``, so one module appears under one key per M
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
    "A4_DENSE_GEMM_SYMBOL",
    "NATIVE_WINDOW_GEMM_DECODER",
    "NATIVE_WINDOW_GEMM_FOLDED_DECODER",
    "NATIVE_SPAN2_GEMM_DECODER",
    "FP8_ACTIVATION_CONTRACT",
    "BF16_ACTIVATION_CONTRACT",
    "NVFP4_ACTIVATION_CONTRACT",
    "DENSE_LAUNCHES",
    "WINDOW_GEMV_LIBRARY_GLOB",
    "QUALIFICATION_SCHEMA",
    "mapped_native_libraries",
    "trace_launches_by_contract",
    "qualify_native_route",
    "QualificationRefused",
    "refusal_record",
]

#: ``scheme.WINDOW_GEMM_SYMBOL`` / ``scheme.A4_DENSE_GEMM_SYMBOL``.
WINDOW_GEMM_SYMBOL = "tessera::window_gemm_dense"
A4_DENSE_GEMM_SYMBOL = "tessera.kernel_a4.a4_span2_gemm"
#: ``telemetry.DECODER_NATIVE_WINDOW_GEMM`` / ``DECODER_NATIVE_WINDOW_GEMM_FOLDED``
#: / ``DECODER_NATIVE_SPAN2_GEMM``.
NATIVE_WINDOW_GEMM_DECODER = "native_window_gemm"
NATIVE_WINDOW_GEMM_FOLDED_DECODER = "native_window_gemm_folded"
NATIVE_SPAN2_GEMM_DECODER = "native_span2_gemm"
#: ``scheme.{FP8,BF16,NVFP4}_ACTIVATION_CONTRACT``.
FP8_ACTIVATION_CONTRACT = "fp8_per_token_dynamic"
BF16_ACTIVATION_CONTRACT = "bf16_unquantized"
NVFP4_ACTIVATION_CONTRACT = "e2m1_group16_ue4m3_static"

#: family -> (activation contract, the one (symbol, decoder) its dense route
#: stamps).  ``fp8_route.DENSE_LAUNCH``, ``bf16_route.DENSE_LAUNCH`` and
#: ``nvfp4_route.process_weights_after_loading`` (``tessera_symbol`` /
#: ``tessera_decoder``) are the owners; ``scheme.ROUTE_LAUNCHES`` publishes
#: the same pairs and the contract test ties them.
DENSE_LAUNCHES = {
    "TESSERA_FP8": (FP8_ACTIVATION_CONTRACT, (WINDOW_GEMM_SYMBOL, NATIVE_WINDOW_GEMM_DECODER)),
    "TESSERA_BF16": (BF16_ACTIVATION_CONTRACT,
                     (WINDOW_GEMM_SYMBOL, NATIVE_WINDOW_GEMM_FOLDED_DECODER)),
    "TESSERA_NVFP4": (NVFP4_ACTIVATION_CONTRACT, (A4_DENSE_GEMM_SYMBOL, NATIVE_SPAN2_GEMM_DECODER)),
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


def trace_launches_by_contract(route_trace, contract, *, policy=None):
    """``{"<symbol> / <decoder>": {...}}`` for one activation contract.

    Each value carries ``symbol``, ``decoder``, ``launches`` (summed over
    every entry), ``entries``, ``modules`` (the per-M-group maximum described
    in the module docstring), ``module_names`` (the union, sorted) and
    ``unnamed_modules`` (the per-M-group maximum).  ``policy``, when given,
    is the ``<family>:<mode>`` stamp every entry on the contract must carry;
    another policy on the same contract is refused, because a contract served
    under a residency the configuration did not name is not the serve the
    configuration describes.
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
        seen_contract = True
        if policy is not None and entry.get("policy") != policy:
            raise QualificationRefused(
                f"route-trace entry on {contract} carries policy {entry.get('policy')!r}, the "
                f"configuration serves {policy!r}: {entry!r}")
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


def qualify_dispatch(route_trace, *, mode, expected_modules):
    """The dispatch leg over every family in ``expected_modules``, or a refusal.

    ``expected_modules`` is ``{family: count}`` or ``{family: {"count": n,
    "names": [...]}}`` from the artifact's ``tessera_serving_manifest.json``.
    A family the artifact does not carry (count 0) is skipped.  Returns
    ``{family: {"contract", "policy", "expected", "observed", "launches"}}``.
    """
    if mode not in ("resident", "streamed"):
        raise QualificationRefused(f"unknown residency mode {mode!r}")
    unknown = sorted(set(expected_modules) - set(DENSE_LAUNCHES))
    if unknown:
        raise QualificationRefused(
            f"the artifact names families this qualifier has no dense launch for: {unknown}")
    families = {}
    for family in sorted(expected_modules):
        count, names = _expected_count(expected_modules, family)
        if count == 0:
            continue
        contract, (symbol, decoder) = DENSE_LAUNCHES[family]
        policy = f"{family}:{mode}"
        launches = trace_launches_by_contract(route_trace, contract, policy=policy)
        expected_key = _pair_key(symbol, decoder)
        foreign = sorted(key for key in launches if key != expected_key)
        if foreign:
            raise QualificationRefused(
                f"dispatches on {contract} ({family}) used {foreign}, not {expected_key}: "
                + json.dumps({key: {k: launches[key][k] for k in ("launches", "modules")}
                              for key in foreign}, sort_keys=True))
        native = launches[expected_key]
        if native["launches"] < 1:
            raise QualificationRefused(f"no served dispatch on {contract} ({family}) was counted")
        if native["unnamed_modules"]:
            raise QualificationRefused(
                f"{native['unnamed_modules']} module(s) dispatching on {contract} ({family}) carried "
                "no prefix; a per-module claim needs unnamed_modules == 0")
        if native["modules"] != count:
            raise QualificationRefused(
                f"{native['modules']} modules dispatched on {contract} ({family}), the artifact "
                f"assigns {count}; the remainder did not serve this route")
        if names is not None and native["module_names"]:
            expected_names = sorted(set(names))
            if native["module_names"] != expected_names:
                missing = sorted(set(expected_names) - set(native["module_names"]))
                extra = sorted(set(native["module_names"]) - set(expected_names))
                raise QualificationRefused(
                    f"the modules dispatching on {contract} ({family}) are not the manifest's: "
                    f"missing {missing}, unexpected {extra}")
        families[family] = {"contract": contract, "policy": policy,
                            "expected": {"symbol": symbol, "decoder": decoder, "modules": count,
                                         "names_checked": bool(names is not None and native["module_names"])},
                            "observed": native}
    if not families:
        raise QualificationRefused("the artifact assigns no module to any dense family; nothing to qualify")
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
                      "native launch its dense route makes; no library leg -- no dense launch on this "
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
