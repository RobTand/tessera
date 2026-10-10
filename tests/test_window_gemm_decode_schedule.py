"""The dense GEMM's decode-regime schedule (tessera#617): small M runs the
minimum M block over a narrow N block, large M keeps the prepared blocks,
the K block never moves.

These tests read only ``tessera.window_geometry``: no tensor runtime, no
Triton import, so they run on the CPU fleet.  The GPU sweep owns the
numbers behind the schedule; the GPU oracle tests own its numerics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.window_geometry import (  # noqa: E402
    DECODE_BLOCK_N,
    DECODE_MAX_M,
    MIN_M_BLOCK,
    decode_schedule,
)


def test_minimum_m_block_is_the_dot_minimum():
    assert MIN_M_BLOCK == 16


def test_narrow_n_block_is_a_wire_tile_divisor():
    from tessera.window_geometry import TILE_ROWS  # noqa: E402
    assert TILE_ROWS % DECODE_BLOCK_N == 0


def test_decode_regime_selects_the_minimum_m_block_and_narrow_n():
    assert decode_schedule(1, 64, 64, 64) == (16, 32, 64)


def test_decode_regime_holds_through_the_boundary():
    for m in (2, 8, DECODE_MAX_M):
        assert decode_schedule(m, 64, 64, 64) == (16, 32, 64)


def test_prefill_keeps_the_prepared_blocks():
    for m in (DECODE_MAX_M + 1, 64, 512, 2048):
        assert decode_schedule(m, 64, 64, 64) == (64, 64, 64)


def test_schedule_never_grows_beyond_a_narrower_prepared_n():
    assert decode_schedule(1, 64, 16, 64) == (16, 16, 64)


def test_schedule_never_moves_the_k_block():
    assert decode_schedule(1, 64, 128, 32) == (16, 32, 32)
    assert decode_schedule(512, 32, 256, 128) == (32, 256, 128)
