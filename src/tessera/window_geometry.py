"""Shared window row tiles and rate checks, without a tensor runtime."""
from __future__ import annotations

from .errors import GrammarError
from .manifest import WINDOW_BITS_MAX

__all__ = ["TILE_ROWS", "require_window_geometry", "MIN_M_BLOCK", "DECODE_MAX_M",
           "DECODE_BLOCK_N", "decode_schedule"]

TILE_ROWS = 512

#: The dense GEMM's smallest M block: ``tl.dot`` needs a 16-row minimum
#: operand, so a smaller batch is served by masking (tessera#617).
MIN_M_BLOCK = 16

#: Batch rows at or below this run the dense GEMM fallback's decode-regime
#: schedule: the minimum M block and a narrow N block (tessera#617).
DECODE_MAX_M = 16

#: The N block selected for the decode-regime fallback. The GB10 sweep
#: (``experiments/sweep_window_gemm_decode.py``) measured improved timings
#: with BM16/BN32 versus BM64/BN64 at the tested shapes at M = 1 and M = 8.
#: Those observations do not establish the M = 1 limiter.
DECODE_BLOCK_N = 32


def decode_schedule(m: int, block_m: int, block_n: int, block_k: int) -> tuple:
    """The dense GEMM's effective ``(BM, BN, BK)`` for a call with ``m`` rows.

    At or below ``DECODE_MAX_M`` the fallback runs the minimum M block over
    a narrow N block, following the isolated decode timing improvement. This
    is not a limiter diagnosis or a claim about the default fused serving
    lane. The K block never moves here, so the K-loop order each output
    accumulates in is unchanged. The N block never grows beyond the prepared
    one. Above ``DECODE_MAX_M`` the prepared blocks serve unchanged.
    """
    if int(m) <= DECODE_MAX_M:
        return (MIN_M_BLOCK, min(block_n, DECODE_BLOCK_N), block_k)
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
