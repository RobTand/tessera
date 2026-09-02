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

This is Gridbook's ``nvfp4_activation_contract`` telemetry, reduced to the two
Tessera routes and owned here.  The attribute prefix is ``_tessera_route_``
(Gridbook's is ``_cb_route_``): the two records must never be mistaken for one
another if both plugins are ever installed in one process.
"""
from __future__ import annotations

import torch

from .scheme import FP8_ACTIVATION_CONTRACT, NVFP4_ACTIVATION_CONTRACT

__all__ = [
    "ROUTE_FIELDS",
    "ROUTE_STATES",
    "ROUTE_CONTRACTS",
    "DECODERS",
    "DECODER_NATIVE_SPAN2",
    "DECODER_TORCH_STOCK",
    "DECODER_TORCH_WINDOW",
    "ATTR_PREFIX",
    "NVFP4_ACTIVATION_CONTRACT",
    "FP8_ACTIVATION_CONTRACT",
    "emit_route",
    "read_route",
    "route_shape",
]

#: Stamped on every record so a served route can be compared against a priced
#: one instead of assumed equal.  The strings live in ``scheme`` (torch-free).
ROUTE_CONTRACTS = frozenset((NVFP4_ACTIVATION_CONTRACT, FP8_ACTIVATION_CONTRACT))
ROUTE_STATES = frozenset(("served", "fallback", "error"))

#: Which implementation produced the weight tile this route multiplied.  A
#: receipt must never claim the native decoder for a serve that took the
#: pure-torch fallback, even though the two produce identical bytes: the
#: fallback is a different residency contract (resident only) and a different
#: load-time cost, and a census that cannot see the difference cannot attest
#: the route.  ``torch_window`` is the FP8 route's decoder, which is pure
#: torch by construction and needs no extension.
DECODER_NATIVE_SPAN2 = "native_span2"
DECODER_TORCH_STOCK = "torch_materialize_stock"
DECODER_TORCH_WINDOW = "torch_window"
DECODERS = frozenset((DECODER_NATIVE_SPAN2, DECODER_TORCH_STOCK, DECODER_TORCH_WINDOW))

ATTR_PREFIX = "_tessera_route_"

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
    except Exception:  # noqa: BLE001 -- telemetry must never break a request
        pass


def read_route(layer):
    """The latest route record as a plain dict, or ``None`` if never written.

    Pure ``getattr`` over Python scalars.  Returning ``None`` (rather than a
    partial dict) is what lets a consumer count a MISSING record as a probe
    error instead of silently passing a gate that never observed a route.
    """
    if getattr(layer, f"{ATTR_PREFIX}state", None) is None:
        return None
    return {f: getattr(layer, f"{ATTR_PREFIX}{f}", None) for f in ROUTE_FIELDS}
