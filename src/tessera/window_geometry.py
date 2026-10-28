"""Shared window geometry and rate checks, without a tensor runtime.

MIN_BLOCK is the minimum matrix operand extent required by tl.dot; smaller
token counts are served by masking that extent rather than reducing it.
"""
from __future__ import annotations

from .errors import GrammarError
from .manifest import WINDOW_BITS_MAX

__all__ = ["MIN_BLOCK", "TILE_ROWS", "require_window_geometry"]

TILE_ROWS = 512
MIN_BLOCK = 16


def require_window_geometry(window_bits: int, rates) -> None:
    """Refuse a window width or column rate that the wire cannot represent."""
    if not 1 <= int(window_bits) <= WINDOW_BITS_MAX:
        raise GrammarError(
            f"window_bits {window_bits} outside 1..{WINDOW_BITS_MAX}, the widest "
            "window the wire carries"
        )
    rates = tuple(rates)
    if rates and max(rates) > int(window_bits):
        raise GrammarError(
            f"rate {max(rates)} does not fit a {window_bits}-bit window"
        )
