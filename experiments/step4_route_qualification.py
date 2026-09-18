"""Does a full-engine capture's own evidence say the native fp4 route ran?

In ``resident`` residency the NVFP4 route prepares its decoder with
``allow_torch_fallback=substitutes_when_unavailable(self._mode)``
(``nvfp4_route.py``), and ``ext.NATIVE_EXTENSIONS`` publishes
``{"resident": {"status": "substituted", "decoder": "torch_materialize_stock"}}``
for ``tessera_nvfp4_``.  A container that cannot build the extension therefore
**serves**: the request succeeds, the ledger closes, and the capture reads as
qualified while the E2M1 units decoded through stock Torch.  The resident serve
is numerically untouched by that substitution, which is why nothing downstream
notices -- but a resource capture measures allocation, and the two decoders do
not allocate alike, so the ledger is about the wrong decoder.

So the qualification is refused unless the capture's OWN evidence carries both
legs:

* the library leg -- ``runtime-observation.json`` ``base.native_libraries`` is
  the worker process's actual mapped shared objects with their sha256
  (``full_engine_worker.native_library_observation``).  A ``tessera_nvfp4_*.so``
  must appear there, and its bytes must be the bytes the launcher's preflight
  built and recorded before the engine started.
* the dispatch leg -- ``TESSERA_ROUTE_TRACE``'s counter file
  (``telemetry._RouteTrace``) keys every served dispatch on
  ``(policy, shape, symbol, decoder, contract, kind)`` and reports it as
  ``launches``/``modules`` (``_RouteTrace.snapshot``).  Every entry on the
  NVFP4 activation contract must name ``native_span2``; one
  ``torch_materialize_stock`` entry refuses the capture.

Neither leg is restated by the launcher: the first is the worker's census, the
second is the serving route's own telemetry.  The library leg alone proves the
extension was mapped, not that a given module used it; the dispatch leg is what
carries the per-module claim, and its ``modules`` count is how many distinct
modules reported each key.  A missing or unreadable trace is NOT VERIFIED, and
not verified is a refusal, never a pass.

Nothing here imports Torch, vLLM or ``tessera``: it reads finished JSON, so the
refusal logic is testable on any host.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

__all__ = [
    "NVFP4_LIBRARY_GLOB",
    "NATIVE_DECODER",
    "SUBSTITUTED_DECODER",
    "mapped_native_libraries",
    "trace_decoders_by_contract",
    "qualify_native_route",
    "QualificationRefused",
]

#: ``ext.NVFP4_MODULE_PREFIX`` + ``"*.so"``; the built module name carries a
#: build-identity hash, so no exact basename exists to compare against.
NVFP4_LIBRARY_GLOB = "tessera_nvfp4_*.so"

#: ``telemetry.DECODER_NATIVE_SPAN2`` / ``DECODER_TORCH_STOCK``.  Spelled here
#: as data because this module may not import the serving package.
NATIVE_DECODER = "native_span2"
SUBSTITUTED_DECODER = "torch_materialize_stock"


class QualificationRefused(Exception):
    """The capture's evidence does not establish the native route."""


def mapped_native_libraries(runtime_observation, glob=NVFP4_LIBRARY_GLOB):
    """``{path: sha256}`` of the mapped libraries whose basename matches ``glob``."""
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
                f"mapped library {path} carries no readable sha256; the library leg cannot be checked")
    return matched


def trace_decoders_by_contract(route_trace, contract):
    """``{decoder: {"count": int, "modules": int}}`` for one activation contract.

    ``_RouteTrace.snapshot`` writes its counters as a list of ``entries``; each
    carries the six key fields plus ``launches`` (one per module per served
    dispatch) and ``modules`` (how many distinct modules reported the key).  An
    entry that names no decoder is not a decoder this may silently drop.
    """
    entries = route_trace.get("entries")
    if entries is None:
        raise QualificationRefused("route trace carries no entries; the dispatch leg is not verified")
    totals = {}
    seen_contract = False
    for entry in entries:
        if entry.get("contract") != contract:
            continue
        seen_contract = True
        decoder = entry.get("decoder")
        if not isinstance(decoder, str) or not decoder:
            raise QualificationRefused(
                f"route-trace entry on {contract} names no decoder: {entry!r}")
        if "launches" not in entry:
            raise QualificationRefused(
                f"route-trace entry on {contract} counts no launches: {entry!r}")
        bucket = totals.setdefault(decoder, {"launches": 0, "modules": 0, "entries": 0})
        bucket["launches"] += int(entry["launches"])
        bucket["entries"] += 1
        # NOT a sum: one module that serves both a prefill and a decode shape
        # appears under two keys, and adding them would report twice as many
        # modules as the artifact has.  The widest single key is how many
        # distinct modules were seen dispatching on this decoder.
        bucket["modules"] = max(bucket["modules"], int(entry.get("modules", 0)))
    if not seen_contract:
        raise QualificationRefused(
            f"route trace records no dispatch on {contract}; the capture never served the route it prices")
    return totals


def qualify_native_route(runtime_observation, route_trace, *, expected_library_sha256,
                         activation_contract, expected_modules=None):
    """Both legs, or a refusal.  Returns the qualification record.

    ``expected_library_sha256`` is what the launcher's in-container preflight
    built and digested BEFORE the engine started: the same bytes must be the
    ones the worker mapped.  ``expected_modules``, when given, is how many
    distinct modules the artifact assigns to this route -- a trace that reports
    fewer served the rest on something else.
    """
    matched = mapped_native_libraries(runtime_observation)
    if not matched:
        raise QualificationRefused(
            f"no {NVFP4_LIBRARY_GLOB} is mapped in the worker process; the NVFP4 decode ran on "
            f"{SUBSTITUTED_DECODER} (ext.NATIVE_EXTENSIONS publishes that substitution for resident)")
    if len(matched) != 1:
        raise QualificationRefused(f"several NVFP4 decode libraries are mapped: {sorted(matched)}")
    (library_path, library_sha256), = matched.items()
    if library_sha256 != expected_library_sha256:
        raise QualificationRefused(
            f"mapped {library_path} is sha256 {library_sha256}, the preflight built "
            f"{expected_library_sha256}; the engine did not run the proven library")
    decoders = trace_decoders_by_contract(route_trace, activation_contract)
    foreign = sorted(name for name in decoders if name != NATIVE_DECODER)
    if foreign:
        raise QualificationRefused(
            f"dispatches on {activation_contract} used {foreign}, not {NATIVE_DECODER}: "
            + json.dumps({name: decoders[name] for name in foreign}, sort_keys=True))
    native = decoders[NATIVE_DECODER]
    if native["launches"] < 1:
        raise QualificationRefused(f"no served dispatch on {activation_contract} was counted")
    if expected_modules is not None and native["modules"] != expected_modules:
        raise QualificationRefused(
            f"{native['modules']} modules dispatched on {activation_contract}, the artifact "
            f"assigns {expected_modules}; the remainder did not serve this route")
    return {"schema": "tessera.step4_native_route_qualification.v1",
            "activation_contract": activation_contract,
            "library": {"path": library_path, "sha256": library_sha256},
            "decoders": decoders, "expected_modules": expected_modules,
            "scope": "the mapped decode library and every counted dispatch on this contract; "
                     "no timing, fixed-resource or release admission follows from it",
            "qualified": True}


def refusal_record(phase, message, **context):
    """The record a refused run writes beside its ledgers."""
    return {"schema": "tessera.step4_capture_refusal.v1", "phase": phase,
            "refusal": message, "qualified": False,
            "scope": "the run was refused before or after measurement; nothing here is a price",
            **context}


def read_json(path):
    return json.loads(Path(path).read_text())
