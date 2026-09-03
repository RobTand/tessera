"""The residency mode, and the dispatch from a scheme's family to its route.

ONE FLAG.  The checkpoint selects this plugin (``quantization_config.
quant_method: "tessera"``), so there is no enable flag to get wrong: if the
bytes are Tessera's, the plugin serves them.  What the operator does choose is
the RESIDENCY, ``TESSERA_SERVE_MODE=resident|streamed``:

* ``resident`` decodes every module once at load and holds its stock tile
  (4.5 bpw for an NVFP4 module, 8.0 plus one fp32 per row for an FP8 one,
  16.0 plus one fp32 per row for a BF16 one -- the wire's smaller bytes are
  then on disk only.  On the BF16 route that is the source precision, so it is
  the correctness path and not a size claim);
* ``streamed`` holds the wire's own bytes per module and decodes every forward
  into a transient tile.

That is a real difference in the footprint the artifact occupies, so the
plugin will not choose it for the operator: an unset mode is a named refusal,
not a default.  The mode is latched for the process (``flags``) and folded into
vLLM's compile-cache key (``compile_identity``), because two modes trace the
same files into different forwards and would otherwise share one cached
function.

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
    "The plugin will not choose a residency for you: 'resident' decodes at load and holds "
    "each module's stock tile (4.5 bpw for an NVFP4 module, 8.0 plus one fp32 per row for an "
    "FP8 one -- the wire's bytes are then on disk only) and 'streamed' holds the wire's own "
    "bytes per module and decodes every forward into a transient tile. The mode changes the "
    "footprint the artifact actually occupies, so it is declared, not defaulted.")


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
    from importlib import import_module
    return getattr(import_module(module_name), builder_name)(scheme, prefix, resolved)
