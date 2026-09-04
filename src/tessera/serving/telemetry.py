"""The route record a served Tessera module writes, and how a census reads it.

A ``lane_eligibility`` cell in ``runtime_contract.json`` states which route a
module *executes*.  This module is the serve-side observation behind such a
cell: every ``apply()`` writes twelve Python scalars onto its layer naming the
kernel it invoked, the activation contract that ran, the problem shape and
whether the launch returned.  ``tools/tessera_route_census.py`` reads them back
from inside the worker, so a receipt is the record the serve wrote and not a
log line someone parsed.

Twelve ``setattr``s of Python scalars -- no tensor is touched, so this sits on
the hot path without a synchronisation and cannot perturb what executed.

``TESSERA_ROUTE_TRACE=<abs path>`` additionally keeps a counting histogram of
what the serve executed, keyed by route AND problem shape -- the question a
per-module "latest record" cannot answer, and the one a served KL needs
answered before it can claim to have measured a decode-path kernel
(tessera#102, ``_RouteTrace``).  It is off by default and eager-only.

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
    "read_route",
    "note_lane_refusal",
    "read_lane_refusal",
    "LANE_REFUSAL_ATTR",
    "route_shape",
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
DECODERS = frozenset((DECODER_NATIVE_SPAN2, DECODER_TORCH_STOCK, DECODER_TORCH_WINDOW,
                      DECODER_WINDOW_GEMV))

ATTR_PREFIX = "_tessera_route_"

#: Absolute path to a JSON file the serve keeps a per-(route, shape) launch
#: histogram in.  Unset (the default) means the histogram does not exist:
#: ``emit_route`` writes the same record it always did and counts nothing.
#: Read once, at import, which is what latches it for the process.
ROUTE_TRACE_ENV = "TESSERA_ROUTE_TRACE"
ROUTE_TRACE_SCHEMA = "tessera.route_trace/1"

#: The record's field names, in report order.  The census reads exactly these.
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
)


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


def emit_route(layer, *, kind: str, policy: str, symbol: str, tile_m: int = 0,
               shape: str = "", contract: str = "", state: str = "served",
               reason=None, decoder: str = "") -> None:
    """Record the latest dispatch route on ``layer``.  Never raises.

    TWO-PHASE USE.  Write ``state="error"`` with a reason before a launch and
    rewrite ``state="served"`` after it returns; then "raised mid-launch" is
    distinguishable from "never launched", and ``symbol`` stays an honest
    record of what was INVOKED even when it threw.

    The record is the LATEST dispatch, which answers "what does this module
    serve on" and not "what did this serve run".  ``TESSERA_ROUTE_TRACE``
    answers the second question by counting; see ``_RouteTrace``.
    """
    try:
        values = {
            "kind": str(kind), "policy": str(policy), "symbol": str(symbol),
            "tile_m": int(tile_m), "shape": str(shape), "contract": str(contract),
            "state": str(state), "reason": None if reason is None else str(reason),
            "decoder": str(decoder),
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
    the counter object does not exist and ``emit_route`` does the same twelve
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
        module = getattr(layer, "prefix", "") or hex(id(layer))
        with self._lock:
            entry = self._counts.get(key)
            if entry is None:
                entry = self._counts[key] = [0, set()]
            entry[0] += 1
            entry[1].add(module)
            self._dirty = True

    # -- readout -----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            entries = [
                {"policy": policy, "shape": shape, "symbol": symbol,
                 "decoder": decoder, "contract": contract, "kind": kind,
                 "launches": count, "modules": len(modules)}
                for (policy, shape, symbol, decoder, contract, kind),
                    (count, modules) in sorted(self._counts.items())
            ]
        return {
            "schema": ROUTE_TRACE_SCHEMA,
            "pid": os.getpid(),
            "started_utc": self.started_utc,
            "flushed_utc": datetime.now(timezone.utc).isoformat(),
            "flushes": self.flushes,
            "note": ("launches counted per module per served dispatch; a "
                     "shape of M* means the record was written while "
                     "torch.compile was tracing, where one graph serves every "
                     "M and a count is not a launch count"),
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
