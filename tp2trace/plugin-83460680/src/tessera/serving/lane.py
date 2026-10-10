"""The residency mode, and the dispatch from a scheme's family to its route.

ONE FLAG.  The checkpoint selects this plugin (``quantization_config.
quant_method: "tessera"``), so there is no enable flag to get wrong: if the
bytes are Tessera's, the plugin serves them.  What the operator does choose is
the RESIDENCY, ``TESSERA_SERVE_MODE=resident|streamed``, and the plugin will
not choose it for the operator: an unset mode is a named refusal, not a
default.  The mode is latched for the process (``flags``) and included in
vLLM's compile-cache key (``compile_identity``).

WHAT THE MODES PREPARE.  The current dense routes -- NVFP4, FP8 and BF16 --
prepare the SAME packed native units in both modes: the compact reader expands
no weight plane and no ``[rows, columns]`` tile is materialised at load or per
forward, so the mode is a declared dispatch fact rather than a choice between
an expanded and a compressed weight footprint.  A routed mixture-of-experts
stack is resident-only and refuses ``streamed``.

WHICH ROUTE a module takes is the CHECKPOINT's fact, not the operator's: the
scheme's ``family`` (FAMILY = ROUTE, ``scheme.ROUTES``) picks ``TESSERA_NVFP4``
(an E2M1-based grid over a LUT plane -> the NVFP4 tile, W4A4), ``TESSERA_FP8``
(E4M3 over the CHANNEL plane -> the per-channel FP8 pair, W8A8) or
``TESSERA_BF16`` (BF16 over the CHANNEL plane -> a plain bf16 tile, W16A16).
One checkpoint may carry all three, module by module, and a single serve
executes each on its own path -- which is what makes this a product an
allocator can target across the whole 3-to-8-bit range rather than a set of
lanes an operator has to choose between.
"""
from __future__ import annotations

from typing import Mapping

from .flags import latched_mode, reset_for_tests as _reset_flag
from .scheme import ROUTES, TESSERA_FAMILIES

__all__ = [
    "TESSERA_MODE_ENV",
    "MODE_RESIDENT",
    "MODE_STREAMED",
    "MODES",
    "serve_mode",
    "reset_for_tests",
    "build_tessera_method",
]

TESSERA_MODE_ENV = "TESSERA_SERVE_MODE"
MODE_RESIDENT = "resident"
MODE_STREAMED = "streamed"
MODES = (MODE_RESIDENT, MODE_STREAMED)

_UNSET_HELP = (
    "Select the residency explicitly: 'resident' or 'streamed'. Dense routes prepare "
    "the same packed native units in both modes; a routed mixture-of-experts stack "
    "requires 'resident'.")


def serve_mode() -> str:
    """The declared residency mode.  No default: an unset mode is an error."""
    return latched_mode(TESSERA_MODE_ENV, modes=MODES,
                        meaning="the Tessera residency", unset_help=_UNSET_HELP)


def reset_for_tests() -> None:
    _reset_flag(TESSERA_MODE_ENV)


def build_tessera_method(scheme: Mapping, prefix: str = "<tessera>", mode: str | None = None):
    """The vLLM linear method for a Tessera module, by the scheme's family."""
    resolved = mode or serve_mode()
    if resolved not in MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    family = scheme.get("family") if isinstance(scheme, Mapping) else None
    route = ROUTES.get(family) if isinstance(family, str) else None
    if route is None:
        raise ValueError(
            f"tessera target {prefix!r}: family must be one of {TESSERA_FAMILIES}, got {family!r}")
    # FAMILY = ROUTE, and the route says which module serves it.  Dispatching
    # off the table rather than an if-chain is what makes a third family one
    # route module plus one ROUTES entry: an if-chain here would be a second
    # place to remember, and the one that fails at SERVE time rather than at
    # import time.  Imported lazily so a producer reading the contract on a
    # box with no torch never pulls a route in.
    module_name, builder_name = route["builder"]
    # THE PLATFORM GATE, ASKED BEFORE THE ROUTE MODULE IS IMPORTED (#457).
    # This is the earliest seam a dense Tessera module passes through --
    # ``TesseraConfig.get_quant_method`` calls straight into here -- so an
    # artifact whose family the pinned contract publishes as unbacked on this
    # device is refused with the contract's own word before a weight is
    # created, before ``process_weights_after_loading``, and before the route
    # module has imported a kernel or a vLLM quantizer.  The alternative is a
    # HIP failure three layers down describing a missing operator, which is a
    # true statement about the wrong thing: the absence is attested, not
    # accidental.  ``unstated`` refuses nothing, so sm_121 and every contract
    # written before the platform axis are byte-for-byte unchanged.
    from .backend import require_platform_backs
    from .contract import PAYLOAD_FAMILY_BY_ROUTE
    from .telemetry import record_platform

    # ``.get``, not ``[]``: a route this contract publishes no payload family
    # for is a route the platform table cannot have attested anything about,
    # so it reads through as ``unstated`` and refuses nothing.  That keeps
    # "a new family needs only a ROUTES entry" true, which is a property
    # ``test_serving_dispatch`` pins with a synthetic route.
    payload_family = PAYLOAD_FAMILY_BY_ROUTE.get(family)
    if payload_family is not None:
        require_platform_backs(payload_family, f"tessera target {prefix!r}")
    # LATCH THE PLATFORM HERE, where it is cheap and eager.  ``emit_route``
    # stamps a process constant, and resolving it for the first time inside a
    # traced forward would put a device probe in the FX graph; this runs once
    # per module at build, long before any forward.
    record_platform()
    from importlib import import_module
    return getattr(import_module(module_name), builder_name)(scheme, prefix, resolved)
