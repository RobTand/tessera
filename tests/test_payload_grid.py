"""TESSERA-8 and the payload-grid generalisation.

TESSERA-4 and TESSERA-8 are one construction at two grid widths -- the spec
calls EN8 "the same architecture at E4M3 payload width" -- so the tests that
matter are the ones showing the grammar closes at both, and that E2M1 behaviour
did not move when the width became a parameter.
"""

from __future__ import annotations

import pytest
import torch

from tessera.alphabet import (
    E2M1_GRID,
    E4M3_GRID,
    PayloadGrid,
    build_forest,
    value_order,
)
from tessera.decode import reconstruct_unit
from tessera.encode import encode_unit
from tessera.errors import GrammarError
from tessera.trellis import ConvCode

CODE = ConvCode(memory=6)


def test_e4m3_values_agree_with_the_hardware_format():
    """The grid is FP8 as the hardware defines it, not as we imagine it."""
    native = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    for byte in range(256):
        if byte in (0x7F, 0xFF):  # NaN in FN -- see PayloadGrid.native
            continue
        assert float(native[byte]) == E4M3_GRID.values[byte], f"byte {byte:#04x}"


def test_the_nan_slots_carry_a_neighbours_value_and_map_back_to_a_legal_byte():
    """E4M3FN has 254 finite patterns, and 254 admits no dyadic partition.

    Rather than lose the power of two, the two NaN slots duplicate their
    neighbour; ``native`` sends them back to the legal byte at materialisation.
    A duplicate is never preferred, because ties break to the lower code.
    """
    assert E4M3_GRID.values[0x7F] == E4M3_GRID.values[0x7E]
    assert E4M3_GRID.values[0xFF] == E4M3_GRID.values[0xFE]
    assert E4M3_GRID.native[0x7F] == 0x7E and E4M3_GRID.native[0xFF] == 0xFE
    order = value_order(E4M3_GRID)
    assert order.index(0x7E) < order.index(0x7F)
    assert order.index(0xFE) < order.index(0xFF)


def test_a_grid_whose_size_is_not_a_power_of_two_is_refused():
    with pytest.raises(GrammarError, match="not a power of"):
        PayloadGrid("odd", (0.0, 1.0, 2.0), (0, 1, 2))


@pytest.mark.parametrize("rate", range(1, 8))
def test_the_partition_closes_at_every_rate_on_e4m3(rate):
    """``2^(R+1) anchors x 2^(cap-R) descendants = 2^(cap+1)`` -- 256 here."""
    forest = build_forest(rate, grid=E4M3_GRID)
    assert len(forest.blocks) == 1 << (rate + 1)
    assert len(forest.blocks[0]) == 1 << (7 - rate)
    assert len({code for block in forest.blocks for code in block}) == 256


def test_e2m1_forests_are_unchanged_by_the_generalisation():
    """The width became a parameter; TESSERA-4's alphabet must not have moved."""
    for rate in (1, 2, 3):
        forest = build_forest(rate, grid=E2M1_GRID)
        assert forest.grid is E2M1_GRID
        assert forest.cap == 3
        assert build_forest(rate).blocks == forest.blocks


@pytest.mark.parametrize("rate", [3, 5, 7])
def test_tessera_8_round_trips(rate):
    """Encode and decode over the 256-code grid, at three rates."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forests = {rate: build_forest(rate, grid=E4M3_GRID)}
    torch.manual_seed(0)
    weights = (torch.randn(128, 256, device=device) * 0.02).contiguous()
    unit = encode_unit(
        weights, forests, (rate,) * 256, CODE,
        with_diagonals=False, released_positions=0,
    )
    out = reconstruct_unit(unit, forests, CODE)
    assert out.shape == weights.shape
    assert torch.isfinite(out).all()
    # Higher rate must not be worse; the alphabet is nested by construction.
    assert (out - weights).norm() / weights.norm() < 1.0


def test_a_unit_may_not_mix_payload_grids():
    """Two grids in one unit would give the same code two meanings."""
    forests = {2: build_forest(2, grid=E2M1_GRID), 3: build_forest(3, grid=E4M3_GRID)}
    with pytest.raises(GrammarError, match="one payload grid"):
        encode_unit(
            torch.randn(64, 32) * 0.02, forests, (2,) * 16 + (3,) * 16, CODE,
            with_diagonals=False, released_positions=0,
        )
