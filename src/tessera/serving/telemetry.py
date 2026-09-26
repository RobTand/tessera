"""The route record a served Tessera module writes, and how a census reads it.

A ``lane_eligibility`` cell in ``runtime_contract.json`` states which route a
module *executes*.  This module is the serve-side observation behind such a
cell: every ``apply()`` writes one Python scalar per ``ROUTE_FIELDS`` entry
onto its layer, naming the kernel it invoked, the activation contract that
ran, the platform it ran on, the problem shape and whether the launch
returned.  ``tools/tessera_route_census.py`` reads them back
from inside the worker, so a receipt is the record the serve wrote and not a
log line someone parsed.

One ``setattr`` of a Python scalar per field -- no tensor is touched, so this
sits on the hot path without a synchronisation and cannot perturb what
executed.  The platform is a process constant resolved once, never probed
per call (see ``record_platform``).

``TESSERA_ROUTE_TRACE=<abs path>`` additionally keeps a counting histogram of
what the serve executed, keyed by route AND problem shape -- the question a
per-module "latest record" cannot answer, and the one a served KL needs
answered before it can claim to have measured a decode-path kernel
(tessera#102, ``_RouteTrace``).  It is off by default and eager-only.

Since tessera#509 each histogram entry also names the modules it counted
(``module_names``, sorted, real prefixes only), reports how many distinct
modules had no usable prefix (``unnamed_modules``, plus
``dispatches_without_prefix``), and the file header carries its own ``rank``,
``world_size``, ``rank_source`` and ``platform``.  ``modules`` stays what it
always counted -- distinct named prefixes plus distinct unnamed objects, so a
legacy consumer sees the same number -- but it is now equal to
``len(module_names) + unnamed_modules`` and therefore checkable against the
names beside it.  Before this, an entry's ``modules`` was the size of a set
whose members the file never wrote (an unnamed layer was counted by
``hex(id(layer))``), so a consumer could compare module COUNTS and nothing
else: two modules that swapped contracts left the histogram identical.  A
complete per-module identity requires ``unnamed_modules == 0``.  The headers
are read from ``torch.distributed`` when it is initialized and are JSON
``null`` with ``rank_source: "unavailable"`` when it is not, because a
defaulted rank 0 is indistinguishable from a real one.  The object identity
that keeps two unnamed modules distinct is a bare ``id``, never a reference:
telemetry does not keep a module -- and through it, its weights -- alive after
the model is unloaded.  The rank identity is observed once and kept for the
trace's lifetime -- the atexit flush runs after
``destroy_process_group()``, so re-reading there would erase it -- and a later
observation that disagrees is reported as ``rank_conflict`` instead of being
adopted.  Nothing is queried per dispatch.  The header's ``platform`` is READ
from the token already latched at model build, never probed for: the plugin
writes this file at IMPORT, in the API-server process, before vLLM forks the
engine core, so a device probe there would initialise CUDA in the wrong
process.  Until it is latched the header says ``""``, the same "does not say"
the per-route records use.  Both additions are
additive: ``schema`` is unchanged, no field was renamed or removed, and a
reader that knows only the histogram still reads exactly what it read before.
``identity_version`` is compared for equality -- see its definition.

This is Gridbook's ``nvfp4_activation_contract`` telemetry, reduced to the
Tessera routes and owned here.  The attribute prefix is ``_tessera_route_``
(Gridbook's is ``_cb_route_``): the two records must never be mistaken for one
another if both plugins are ever installed in one process.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

import torch

from .flags import latched_path
from .scheme import (
    BF16_ACTIVATION_CONTRACT, FP8_ACTIVATION_CONTRACT, NVFP4_ACTIVATION_CONTRACT, ROUTES)

__all__ = [
    "ROUTE_FIELDS",
    "ROUTE_STATES",
    "ROUTE_CONTRACTS",
    "DECODERS",
    "DECODER_NATIVE_SPAN2",
    "DECODER_TORCH_STOCK",
    "DECODER_TORCH_WINDOW",
    "DECODER_WINDOW_GEMV",
    "DECODER_NATIVE_WINDOW_GEMM",
    "DECODER_NATIVE_WINDOW_GEMM_FOLDED",
    "DECODER_NATIVE_SPAN2_GEMM",
    "DECODER_NATIVE_SPAN2_GROUPED",
    "DECODER_NATIVE_WINDOW_MOE_COMPACT",
    "DECODER_NATIVE_WINDOW_MOE_COMPACT_FOLDED",
    "ATTR_PREFIX",
    "ROUTE_TRACE_ENV",
    "ROUTE_TRACE_SCHEMA",
    "start_route_trace",
    "stop_route_trace",
    "route_trace",
    "route_trace_snapshot",
    "NVFP4_ACTIVATION_CONTRACT",
    "FP8_ACTIVATION_CONTRACT",
    "BF16_ACTIVATION_CONTRACT",
    "emit_route",
    "record_platform",
    "latched_platform",
    "reset_platform_for_tests",
    "read_route",
    "note_lane_refusal",
    "read_lane_refusal",
    "LANE_REFUSAL_ATTR",
    "route_shape",
    "IDENTITY_VERSION",
]

#: Stamped on every record so a served route can be compared against a priced
#: one instead of assumed equal.  DERIVED from ``ROUTES``, like the family
#: list: a hand-written tuple here was a place a third family could be added
#: and then emit a contract this set did not contain, so a census reading the
#: record against it would call a served route unrecognised.  The strings
#: themselves live in ``scheme``, which is torch-free.
ROUTE_CONTRACTS = frozenset(route["activation_contract"] for route in ROUTES.values())
ROUTE_STATES = frozenset(("served", "fallback", "error"))

#: Which implementation produced the weight tile this route multiplied.  A
#: receipt must never claim the native decoder for a serve that took the
#: pure-torch fallback, even though the two produce identical bytes: the
#: fallback is a different residency contract (resident only) and a different
#: load-time cost, and a census that cannot see the difference cannot attest
#: the route.  ``torch_window`` is pure torch by construction and needs no
#: extension -- one value for one decoder, since it is the same
#: ``serving.window`` object in the BF16 route and in the FP8 route's fallback,
#: carrying a table of E4M3 bytes in one and of bf16 values in the other.  The
#: FP8 route's GEMV lane stamps ``window_gemv`` instead (below).
DECODER_NATIVE_SPAN2 = "native_span2"
DECODER_TORCH_STOCK = "torch_materialize_stock"
DECODER_TORCH_WINDOW = "torch_window"
#: The streamed FP8 route's window-GEMV lane (``fp8_gemv``): the wire read
#: directly by ``tessera_window_gemv``, with no decoded tile anywhere -- and,
#: on the same lane's prefill path, the tile the lane's kernel decode produced
#: for ``_scaled_mm``.  A distinct value because neither launch runs the torch
#: window decoder, and stamping ``torch_window`` for one would claim a decoder
#: that did not run, the same defect the ``torch_materialize_stock`` value
#: exists to prevent on the NVFP4 route.
DECODER_WINDOW_GEMV = "window_gemv"
#: The dense native window GEMM (``serving.native_window``): the compact
#: loader's ``WindowGemvUnit`` decoded inside the packed bitstream GEMM, with
#: no weight tile produced anywhere -- not at load, not per forward.  A
#: distinct value because no other decoder ran, and a census that read
#: ``torch_window`` here would claim one.
DECODER_NATIVE_WINDOW_GEMM = "native_window_gemm"
#: The same dense GEMM on the BF16 family's FOLDED weight arithmetic: one bf16
#: rounding of ``value * row_scale`` per weight in registers before the dot,
#: with no scale in the epilogue (``window_gemm``'s ``arithmetic="folded"``,
#: tessera#614).  ``native_window_gemm`` stays the epilogue arithmetic -- the
#: FP8 family's, and the BF16 family's until #614 -- so a cell or a census can
#: name which numerical function of the wire it attests.
DECODER_NATIVE_WINDOW_GEMM_FOLDED = "native_window_gemm_folded"
#: The native A4 lanes (``tessera.kernel_a4``): the span-2 GEMM decodes the
#: compact loader's packed planes in-kernel -- densely, and per selected expert
#: in the grouped form.  Distinct from ``native_span2``, which names the
#: load-time span-2 DECODE into a stock tile.
DECODER_NATIVE_SPAN2_GEMM = "native_span2_gemm"
DECODER_NATIVE_SPAN2_GROUPED = "native_span2_grouped"
#: The compact window MoE adapter (``tessera.native_window_moe``): routed
#: experts served from the loader's packed ``WindowGemvUnit``s with no decoded
#: tile, on the FP8 family's contract (per-token native A quant, row scale on
#: the fp32 accumulator).
DECODER_NATIVE_WINDOW_MOE_COMPACT = "native_window_moe_compact"
#: The same adapter on the BF16 family, whose weight arithmetic is FOLDED: one
#: bf16 rounding of ``value * row_scale`` per weight in registers before the
#: dot, with no scale in the epilogue (``window_gemm_grouped``'s
#: ``arithmetic="folded"``).  A distinct value because it is a distinct
#: numerical function of the same wire: a census or a cell that read the
#: epilogue decoder here would attest the arithmetic that did not run.
DECODER_NATIVE_WINDOW_MOE_COMPACT_FOLDED = "native_window_moe_compact_folded"
DECODERS = frozenset((DECODER_NATIVE_SPAN2, DECODER_TORCH_STOCK, DECODER_TORCH_WINDOW,
                      DECODER_WINDOW_GEMV, DECODER_NATIVE_WINDOW_GEMM,
                      DECODER_NATIVE_WINDOW_GEMM_FOLDED,
                      DECODER_NATIVE_SPAN2_GEMM, DECODER_NATIVE_SPAN2_GROUPED,
                      DECODER_NATIVE_WINDOW_MOE_COMPACT,
                      DECODER_NATIVE_WINDOW_MOE_COMPACT_FOLDED))

ATTR_PREFIX = "_tessera_route_"

#: Absolute path to a JSON file the serve keeps a per-(route, shape) launch
#: histogram in.  Unset (the default) means the histogram does not exist:
#: ``emit_route`` writes the same record it always did and counts nothing.
#: Read once, at import, which is what latches it for the process.
ROUTE_TRACE_ENV = "TESSERA_ROUTE_TRACE"
ROUTE_TRACE_SCHEMA = "tessera.route_trace/1"

#: Version of the ADDITIVE identity block this file writes: entry
#: ``module_names`` / ``unnamed_modules`` / ``dispatches_without_prefix``, and
#: the header's ``rank`` / ``world_size`` / ``rank_source`` / ``platform``.
#: ``schema`` stays ``tessera.route_trace/1`` because nothing was removed or
#: renamed: a reader that knows only the histogram still reads exactly what it
#: read before, and a reader that wants per-module identity checks this number
#: instead of guessing from the presence of a field.
#:
#: ``== IDENTITY_VERSION`` is the only supported comparison.  A consumer must
#: NOT treat a future ``> 1`` as if it had v1 semantics: the fields it knows
#: may have been redefined, and reading an unknown version with known names is
#: how a schema silently changes meaning under a caller.
IDENTITY_VERSION = 1

#: The record's field names, in report order.  The census reads exactly these.
#: Since #573 the record also names the executed kernel schedule where the
#: dispatch names one (``kernel_schedule``): ``None`` is unobserved, a
#: nonempty string names the schedule (a CUTLASS tag or, on the fused native
#: routes, the op node itself).  Absence is never a refusal -- every receipt
#: in the field predates the stamp -- and a present-but-empty or non-string
#: value is a defect the consumer refuses, never a placeholder to emit.
ROUTE_FIELDS = (
    "kind",       # "dense" | "moe"
    "policy",     # "<family>:<residency mode>"
    "symbol",     # the kernel entry point actually invoked
    "tile_m",     # int; 0 where the route has no tile of its own
    "shape",      # compact problem-shape key
    "contract",   # what RAN, from ROUTE_CONTRACTS
    "state",      # from ROUTE_STATES
    "reason",     # exact refusal reason; None when served
    "decoder",    # from DECODERS: which decoder produced the weight tile
    "platform",   # the device's own token: "sm_121", "gfx1201"; "" if none
    "kernel_schedule",  # executed schedule where named; None if unobserved
)


#: The platform token every record on this process carries, resolved once.
#:
#: WHY THE RECORD NEEDS IT (#457).  A cell in ``lane_eligibility`` is keyed by
#: ``(platform, family, structure, regime, residency, rung)``, and until this
#: field existed the record it is compared against carried every one of those
#: except the first: a gfx1151 serve and an sm_121 serve of the SAME artifact
#: wrote byte-identical records, so a census could join either one to the
#: sm_121 cell and report agreement.  The platform is the coordinate that
#: tells them apart, and it belongs on the observation rather than on the
#: tool's command line, where it is whatever the operator typed.
#:
#: WHY IT IS RESOLVED ONCE, LAZILY, AND NEVER INSIDE ``apply()``.
#: ``emit_route`` runs on the forward -- under ``torch.compile`` it runs
#: inside the traced body -- and that surface has broken a serve before
#: (``LANE_REFUSAL_ATTR`` above records why a load fact is not a route field).
#: A device probe there would be a CUDA/HIP call per module per forward on a
#: path whose whole premise is that it touches no tensor.  So it is a process
#: constant: one process serves one device, and the value is the PROBED token
#: (never ``TESSERA_PLATFORM_TOKEN``, which is a build-only override -- a
#: receipt that could be moved by an environment variable is not a receipt).
#: A box that cannot name a platform stamps ``""``, which a census reads as
#: "this record does not say" rather than as a claim about a platform.
_PLATFORM: "str | None" = None


def record_platform() -> str:
    """This process's platform token for the route record; ``""`` if unknown.

    Cached after the first call.  Never raises: telemetry that can break a
    serve is not telemetry.
    """
    global _PLATFORM
    if _PLATFORM is not None:
        return _PLATFORM
    # NEVER PROBE A DEVICE INSIDE A TRACED BODY (#113 is the precedent).
    # ``emit_route`` is called from ``apply()``, which vLLM captures with
    # ``aot_compile_fullgraph``; ``torch.cuda.get_device_capability`` then
    # becomes an FX node over fake tensors and Dynamo raises while COMPILING
    # -- where ``emit_route``'s ``except Exception`` cannot reach it, so the
    # engine core never initialises.  ``is_compiling()`` is constant-folded,
    # so under compile this is a literal "" and the probe is dead code.  In a
    # real serve it is never reached: ``lane.build_tessera_method`` latches
    # the token at model build, eagerly, before any forward is traced.
    if torch.compiler.is_compiling():
        return ""
    try:
        from .backend import platform_of_this_process

        _PLATFORM = platform_of_this_process(torch) or ""
    except Exception:  # noqa: BLE001 -- a record with no platform is honest
        _PLATFORM = ""
    return _PLATFORM


def latched_platform() -> str:
    """The platform token IF it has already been latched; ``""`` otherwise.

    READ-ONLY by contract, and deliberately not :func:`record_platform`: the
    route trace's header is written from ``flush()``, which the plugin runs at
    IMPORT -- in the API-server process, before vLLM forks the engine core --
    and again from the atexit flush.  A device probe there would either
    initialise CUDA in a process that must never have it, or freeze ``""``
    before the engine core had a device to name, and the header would then be
    wrong in both processes.  The probe stays where it belongs: eagerly, at
    model build, in ``lane.build_tessera_method``.
    """
    return _PLATFORM or ""


def reset_platform_for_tests() -> None:
    """Forget the cached token (tests only)."""
    global _PLATFORM
    _PLATFORM = None


#: Where a load-time lane refusal is parked on a layer.  A SEPARATE attribute
#: from the route record, and not a thirteenth ``ROUTE_FIELDS`` entry, for one
#: reason: the record is written from ``apply()`` on every forward, and that is
#: the exact surface a compiled forward has broken before (a Python branch on
#: the token dim, an ``lru_cache``d build on the call path -- issue #52).  A
#: refusal is a LOAD fact, written once, so it is read once by a census and
#: never touched inside a traced graph.
LANE_REFUSAL_ATTR = f"{ATTR_PREFIX}lane_refusal"


def note_lane_refusal(layer, lane: str, refusal) -> None:
    """Record, at LOAD, that ``lane`` could not prepare for this module.

    The route still serves -- the fallback produces the same bytes -- so the
    route record says ``served`` and says so honestly.  What the record cannot
    say is that the lane the artifact was built to exercise took nothing, and
    a stderr warning is not a value a gate reads: 112 of those scrolled past
    under four censuses that each reported ``problems: []`` (issue #104).
    ``None`` clears the note, so a module whose lane prepared carries no stale
    refusal from an earlier load in the same process.
    """
    try:
        setattr(layer, LANE_REFUSAL_ATTR,
                None if refusal is None else f"{lane}: {refusal}")
    except Exception:  # noqa: BLE001 -- telemetry must never break a load
        pass


def read_lane_refusal(layer):
    """The load-time lane refusal on ``layer``, or ``None``.

    ``None`` covers both "the lane prepared" and "no lane was ever asked
    for"; the two are told apart by the route record's decoder, which is the
    field that says what actually ran.
    """
    return getattr(layer, LANE_REFUSAL_ATTR, None)


def route_shape(x2, rows, cols) -> str:
    """The ``M:N:K`` string a route record carries, safe under compile.

    ``M`` is the token count of THIS call, so in eager mode the record says
    which problem shape ran (prefill and decode differ, and the census reads
    that difference).  Under ``torch.compile`` the token dimension is symbolic
    and the forward is shape-polymorphic: formatting the SymInt would pin it to
    one value and vLLM's compiled forward then refuses to start
    (``ConstraintViolationError`` on ``input_ids.size()[0]``, seen 2026-09-02
    serving these routes without ``--enforce-eager``).  A compiled record
    therefore carries ``M*`` -- the honest statement that one graph serves every
    M -- and never the value.  ``N`` and ``K`` are weight facts.
    """
    m = "*" if torch.compiler.is_compiling() else str(int(x2.shape[0]))
    return f"M{m}:N{int(rows)}:K{int(cols)}"


def _process_rank():
    """``(rank, world_size, source)`` for THIS process, never a fabricated one.

    Read at snapshot time only -- never per dispatch, and never per forward.
    ``torch.distributed`` is initialized in the engine-core process before the
    first forward, so a real serve reports its real rank and world size.  A
    process that never joined a group reports ``(None, None, "unavailable")``
    rather than a plausible-looking ``0``: a defaulted rank is
    indistinguishable from a genuine rank 0, which is precisely the confusion
    a consumer binding a trace file to a rank has to avoid (tessera#509).
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size()), "torch.distributed"
    except Exception:  # noqa: BLE001 -- metadata must never break a serve
        pass
    return None, None, "unavailable"


def emit_route(layer, *, kind: str, policy: str, symbol: str, tile_m: int = 0,
               shape: str = "", contract: str = "", state: str = "served",
               reason=None, decoder: str = "", platform: "str | None" = None,
               kernel_schedule=None) -> None:
    """Record the latest dispatch route on ``layer``.  Never raises.

    TWO-PHASE USE.  Write ``state="error"`` with a reason before a launch and
    rewrite ``state="served"`` after it returns; then "raised mid-launch" is
    distinguishable from "never launched", and ``symbol`` stays an honest
    record of what was INVOKED even when it threw.

    The record is the LATEST dispatch, which answers "what does this module
    serve on" and not "what did this serve run".  ``TESSERA_ROUTE_TRACE``
    answers the second question by counting; see ``_RouteTrace``.

    ``kernel_schedule`` names the executed schedule where the dispatch names
    one (tessera#573): a CUTLASS tag where the mainloop has one, the op node
    itself on the fused native routes.  Unlike ``platform`` -- a process
    constant stamped centrally -- it differs per dispatch and must come from
    the call site.  ``None`` (the default) is unobserved and reads back as
    ``None``; pass nothing when nothing names one, never a placeholder: the
    consumer treats missing-or-null as unknown and refuses a present-but-empty
    or non-string value as a defect.
    """
    try:
        values = {
            "kind": str(kind), "policy": str(policy), "symbol": str(symbol),
            "tile_m": int(tile_m), "shape": str(shape), "contract": str(contract),
            "state": str(state), "reason": None if reason is None else str(reason),
            "decoder": str(decoder),
            # NOT a caller's fact.  Every route module stamps the same value,
            # so it is read here rather than threaded through nine call sites
            # that would each have to remember it; the keyword exists for a
            # test that wants to write a record for another platform.
            "platform": record_platform() if platform is None else str(platform),
            # THE caller's fact.  Stored verbatim -- no coercion, no
            # placeholder: a non-string or empty value is a defect the
            # consumer refuses downstream, and coercing it here would hide
            # the producer bug behind an honest-looking string.  ``None``
            # stays ``None`` so records written before #573 read as
            # unobserved rather than as a claim.
            "kernel_schedule": kernel_schedule,
        }
        for field in ROUTE_FIELDS:
            setattr(layer, f"{ATTR_PREFIX}{field}", values[field])
        trace = _TRACE
        if trace is not None:
            trace.count(layer, values)
    except Exception:  # noqa: BLE001 -- telemetry must never break a request
        pass


class _RouteTrace:
    """Counts of what a serve ACTUALLY executed, keyed by route and shape.

    ``read_route`` answers "what does this module serve on": the latest
    record, which is what a census asserts against.  It cannot answer "what
    did this SERVE run", and that is the question a served KL needs answered.
    A prefill-regime KL dump scores 512-row forwards, so a decode-only kernel
    never executes on a scored forward and a two-arm A/B over it returns a
    bit-identical null -- clean, precise, and about the wrong path
    (tessera#102).  The fix on the instrument side is a decode-regime dump;
    the fix on THIS side is being able to show, from the serve's own
    telemetry, which shapes the scored forwards actually took.

    So: one counter per ``(policy, shape, symbol, decoder, contract, kind)``,
    incremented on every served dispatch, plus the number of distinct modules
    that reported it.  ``shape`` carries the M of the call
    (``route_shape``), which is the discriminator that matters: the streamed
    FP8 route's fallback arm reports ``torch._scaled_mm`` in BOTH regimes, so
    a symbol alone cannot tell a prefill launch from a decode launch, and a
    trace that could not tell them apart would be one lane wearing two names.

    OFF BY DEFAULT, and absent means absent: with no ``TESSERA_ROUTE_TRACE``
    the counter object does not exist and ``emit_route`` does the same
    ``setattr``s it always did.  Enabled, it costs one dict lookup and one set
    insert per module per forward, under a lock -- no tensor is touched and no
    synchronisation happens, so it cannot perturb what executed.

    EAGER ONLY, and the file says so.  Under vLLM's compiled forward the
    dispatch's Python body runs at TRACE time, not per launch, so the counts
    would describe compilation rather than serving; ``route_shape`` already
    degrades to ``M*`` there, which is what marks such an entry in the file.
    Since tessera#113 that is enforced rather than described: ``count``
    declines while ``torch.compiler.is_compiling()``, so a compiled serve with
    the trace enabled records only its eager startup and **serves**.  It used
    to die instead -- Dynamo cannot enter this class's lock under vLLM 0.28's
    full-graph capture, and that error is raised while compiling the traced
    body, where ``emit_route``'s ``except Exception`` cannot reach it.
    """

    #: How often the flusher thread writes, when anything changed.  A
    #: time-throttled write inside ``emit_route`` would never flush the LAST
    #: forward of a run, which is exactly the one a caller wants to read.
    FLUSH_SECONDS = 1.0

    def __init__(self, path):
        self.path = Path(path)
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.flushes = 0
        self._lock = threading.Lock()
        self._counts: dict[tuple, list] = {}
        self._dirty = False
        #: The FIRST rank/world/source observed while distributed was
        #: initialized, kept for the trace's whole lifetime.  ``None`` until
        #: observed -- reported as JSON null with an "unavailable" source, never
        #: guessed.  Never rewritten: ``dist.destroy_process_group()`` runs
        #: before the atexit flush, and a header that went null there would
        #: throw away the only identity the counts in this file ever had.
        self._rank_identity = None
        #: A LATER observation that disagrees with the cached one.  Recorded and
        #: reported, never adopted: the counts belong to the first identity.
        self._rank_conflict = None
        # Write NOW: the point of failure for a mis-set path must be the
        # serve's startup, loudly, and not a silent no-op discovered when the
        # receipt is being written.  ``emit_route`` swallows exceptions by
        # contract, so nothing later in the run could report this.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="tessera-route-trace")
        self._thread.start()

    # -- hot path ----------------------------------------------------------
    def count(self, layer, values) -> None:
        # DECLINE UNDER COMPILE, and decline BEFORE the lock (tessera#113).
        # The class docstring already says this counter is eager-only because
        # its counts would describe compilation; what it did not do is behave
        # that way.  vLLM 0.28 captures the forward with
        # ``aot_compile_fullgraph``, and Dynamo cannot enter a ``threading
        # .Lock`` context manager under a full-graph capture:
        #
        #   torch._dynamo.exc.Unsupported: Unsupported context manager
        #     Explanation: Dynamo does not know how to enter a `lock` context
        #     manager.
        #
        # ``emit_route``'s ``except Exception: pass`` does not catch that --
        # it is raised while COMPILING the traced body, not while running it
        # -- so the whole engine core failed to initialise and the serve never
        # came up.  A telemetry switch must never be able to stop a serve.
        # ``torch.compiler.is_compiling()`` is constant-folded by Dynamo, so
        # under compile this method is dead code and the lock is never traced;
        # eager is unchanged, byte for byte.
        if torch.compiler.is_compiling():
            return
        if values.get("state") != "served":
            return
        key = (values["policy"], values["shape"], values["symbol"],
               values["decoder"], values["contract"], values["kind"])
        # A layer's prefix, when it HAS one.  Anything else -- missing, empty,
        # or not a string -- is UNNAMED: it is never given a made-up name, and
        # it is never sorted into the same list as real prefixes (which would
        # compare a str against whatever else arrived).
        #
        # The object's own ``id`` is kept PRIVATELY, in the count, and is never
        # written to the file.  That preserves what the pre-#509 ``modules``
        # counted -- one per distinct unnamed object, because that was
        # ``len(set(layer.prefix or hex(id(layer))))`` -- so a legacy consumer
        # sees the same number it always saw.  Collapsing every unnamed layer
        # into one bucket would have been a silent SEMANTIC change in the
        # backwards-compatible direction I claimed, and a rarer but real
        # miscount: two unknown modules would read as one.
        prefix = getattr(layer, "prefix", None)
        named = prefix if isinstance(prefix, str) and prefix else None
        with self._lock:
            entry = self._counts.get(key)
            if entry is None:
                entry = self._counts[key] = [0, set(), set(), 0]
            entry[0] += 1
            if named is None:
                # The private id only, exactly as the pre-#509 count did: a
                # trace must never hold a reference to a module (and through
                # it, to GPU weights) for telemetry.  The consequence is the
                # legacy one and it is bounded: two unnamed layers whose
                # lifetimes DO NOT OVERLAP can share a freed address and be
                # counted once.  Any unnamed module already fails per-module
                # qualification (unnamed_modules must be 0), so this cannot
                # turn an unqualified run into a qualified one -- it can only
                # understate how much was unnamed, which the presence of
                # unnamed_modules itself reports.
                entry[2].add(id(layer))
                entry[3] += 1
            else:
                entry[1].add(named)
            self._dirty = True

    # -- readout -----------------------------------------------------------
    def _identity(self):
        """The rank identity for this trace: observed once, then stable.

        The observation is re-taken at snapshot time (once per flush, never per
        dispatch), but a non-null identity is never replaced.  That matters
        because the last write a serve makes is the atexit flush, which runs
        AFTER ``torch.distributed.destroy_process_group()``: re-reading there
        would report null and erase the identity the counts were recorded
        under.  A later observation that *disagrees* is reported as
        ``rank_conflict`` rather than silently adopted.
        """
        if self._rank_identity is not None and self._rank_identity[2] != "unavailable":
            observed = _process_rank()
            if observed[0] is not None and (observed[0], observed[1]) != (
                    self._rank_identity[0], self._rank_identity[1]):
                self._rank_conflict = {"rank": observed[0],
                                       "world_size": observed[1],
                                       "source": observed[2]}
            return self._rank_identity
        observed = _process_rank()
        if observed[0] is not None:
            self._rank_identity = observed
        return self._rank_identity or observed

    def snapshot(self) -> dict:
        with self._lock:
            entries = []
            for (policy, shape, symbol, decoder, contract, kind), (
                    count, names, unnamed, unprefixed) in sorted(self._counts.items()):
                named = sorted(names)
                entries.append({
                    "policy": policy, "shape": shape, "symbol": symbol,
                    "decoder": decoder, "contract": contract, "kind": kind,
                    "launches": count,
                    # The count is the SAME fact as the names plus the number
                    # of unnamed modules, always, and it is unchanged from the
                    # pre-#509 meaning: distinct named prefixes plus distinct
                    # unnamed objects.
                    "modules": len(named) + len(unnamed),
                    "module_names": named,
                    "unnamed_modules": len(unnamed),
                    "dispatches_without_prefix": unprefixed,
                })
        rank, world_size, rank_source = self._identity()
        return {
            "schema": ROUTE_TRACE_SCHEMA,
            "identity_version": IDENTITY_VERSION,
            # The header's own identity, so binding a trace to a rank is a read
            # and not an inference from the file's path.
            "rank": rank,
            "world_size": world_size,
            "rank_source": rank_source,
            "rank_conflict": self._rank_conflict,
            # READ the latched token, never probe for one: this runs at PLUGIN
            # IMPORT, in the API-server process, before vLLM forks the engine
            # core.  A device probe here is a CUDA initialisation in the wrong
            # process (or a frozen "" in the right one) for a header field.
            "platform": latched_platform(),
            "pid": os.getpid(),
            "started_utc": self.started_utc,
            "flushed_utc": datetime.now(timezone.utc).isoformat(),
            "flushes": self.flushes,
            "note": ("launches counted per module per served dispatch; a "
                     "shape of M* means the record was written while "
                     "torch.compile was tracing, where one graph serves every "
                     "M and a count is not a launch count.  Since "
                     f"identity_version {IDENTITY_VERSION} each entry also "
                     "names the modules it counted (module_names, sorted).  "
                     "module_names holds real prefixes ONLY: a layer with no "
                     "usable prefix is unnamed, reported as the count "
                     "unnamed_modules (with dispatches_without_prefix for how "
                     "many dispatches came from them) and never given a made-"
                     "up name.  'modules' is len(module_names) + "
                     "unnamed_modules, which is the same number it carried "
                     "before this version.  A complete per-module identity "
                     "needs unnamed_modules == 0; otherwise the names present "
                     "are exact and the rest are honestly unknown."),
            "entries": entries,
        }

    def flush(self) -> None:
        # vLLM runs the API server and the engine core as SEPARATE processes,
        # and a general plugin is loaded by both.  Only the process holding
        # the model ever counts anything, so an empty histogram here is the
        # other process: it must still prove it can write the path -- that is
        # what the startup write is for -- but it must never overwrite a
        # populated file.  Without this guard the histogram a census reads is
        # whichever process wrote last, and the failure mode is a file full of
        # zeros that looks exactly like a lane that never ran.
        if not self._counts and self.path.exists():
            return
        payload = self.snapshot()
        self.flushes += 1
        tmp = Path(f"{self.path}.tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n")
        os.replace(tmp, self.path)

    def _loop(self) -> None:
        while True:
            time.sleep(self.FLUSH_SECONDS)
            with self._lock:
                dirty, self._dirty = self._dirty, False
            if dirty:
                try:
                    self.flush()
                except Exception:  # noqa: BLE001 -- a trace never breaks a serve
                    pass


def start_route_trace(path) -> "_RouteTrace":
    """Install the route trace at ``path``.  Raises if it cannot be written."""
    global _TRACE
    _TRACE = _RouteTrace(path)
    return _TRACE


def stop_route_trace() -> None:
    """Uninstall the trace (tests only; the flusher thread is a daemon)."""
    global _TRACE
    _TRACE = None


def route_trace():
    """The installed trace, or ``None``."""
    return _TRACE


def route_trace_snapshot():
    """The installed trace's counts, or ``None`` when tracing is off."""
    return None if _TRACE is None else _TRACE.snapshot()


def _route_trace_from_env():
    """Read the flag ONCE, at import, which is what latches it for the run."""
    path = latched_path(ROUTE_TRACE_ENV, meaning="the route-trace JSON")
    if path is None:
        return None
    trace = start_route_trace(path)
    atexit.register(_flush_at_exit)
    return trace


def _flush_at_exit() -> None:
    trace = _TRACE
    if trace is not None:
        try:
            trace.flush()
        except Exception:  # noqa: BLE001
            pass


_TRACE = None
_route_trace_from_env()


def read_route(layer):
    """The latest route record as a plain dict, or ``None`` if never written.

    Pure ``getattr`` over Python scalars.  Returning ``None`` (rather than a
    partial dict) is what lets a consumer count a MISSING record as a probe
    error instead of silently passing a gate that never observed a route.
    """
    if getattr(layer, f"{ATTR_PREFIX}state", None) is None:
        return None
    return {f: getattr(layer, f"{ATTR_PREFIX}{f}", None) for f in ROUTE_FIELDS}
