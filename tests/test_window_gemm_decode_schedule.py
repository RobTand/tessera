"""The dense GEMM's decode-regime schedule (tessera#617): small M runs the
minimum M block, large M keeps the prepared blocks, N/K blocks never move.

These tests read only ``tessera.window_geometry``: no tensor runtime, no
Triton import, so they run on the CPU fleet.  The GPU oracle tests own the
numerics of the launched schedule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.window_geometry import (  # noqa: E402
    DECODE_MAX_M,
    MIN_M_BLOCK,
    decode_schedule,
)


def test_minimum_m_block_is_the_dot_minimum():
    assert MIN_M_BLOCK == 16


def test_decode_regime_drops_to_the_minimum_m_block():
    assert decode_schedule(1, 64, 64, 64) == (16, 64, 64)


def test_decode_regime_holds_through_the_boundary():
    for m in (2, 8, DECODE_MAX_M):
        assert decode_schedule(m, 64, 64, 64) == (16, 64, 64)


def test_prefill_keeps_the_prepared_blocks():
    for m in (DECODE_MAX_M + 1, 64, 512, 2048):
        assert decode_schedule(m, 64, 64, 64) == (64, 64, 64)


def test_schedule_never_moves_the_n_or_k_blocks():
    assert decode_schedule(1, 64, 128, 32) == (16, 128, 32)
    assert decode_schedule(512, 32, 256, 128) == (32, 256, 128)
