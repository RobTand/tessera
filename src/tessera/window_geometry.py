"""Shared window row tiles and rate checks, without a tensor runtime."""
from __future__ import annotations

from .errors import GrammarError
from .manifest import WINDOW_BITS_MAX

__all__ = ["TILE_ROWS", "require_window_geometry", "MIN_M_BLOCK", "DECODE_MAX_M",
           "decode_schedule"]

TILE_ROWS = 512

#: The dense GEMM's smallest M block: ``tl.dot`` needs a 16-row minimum
#: operand, so a smaller batch is served by masking (tessera#617).
MIN_M_BLOCK = 16

#: Batch rows at or below this run the dense GEMM's decode-regime schedule:
#: the packed-weight decode costs the same for any small M, so the launch
#: drops to the minimum M block instead of padding a prefill-shaped tile
#: (tessera#617).
DECODE_MAX_M = 16


def decode_schedule(m: int, block_m: int, block_n: int, block_k: int) -> tuple:
    """The dense GEMM's effective ``(BM, BN, BK)`` for a call with ``m`` rows.

    At or below ``DECODE_MAX_M`` the call is weight-bound -- M = 8 costs the
    same as M = 1 -- so it runs the minimum M block over the prepared N/K
    blocks.  The N/K blocks never move here, so the K-loop order each output
    accumulates in is unchanged and the numerics stay inside the oracle
    tolerance.  Above it the prepared blocks serve unchanged.
    """
    if int(m) <= DECODE_MAX_M:
        return (MIN_M_BLOCK, block_n, block_k)
    return (block_m, block_n, block_k)


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
