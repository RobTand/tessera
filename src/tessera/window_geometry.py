"""Shared window row tiles and rate checks, without a tensor runtime."""
from __future__ import annotations

from .errors import GrammarError
from .manifest import WINDOW_BITS_MAX

__all__ = ["TILE_ROWS", "require_window_geometry"]

TILE_ROWS = 512


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
