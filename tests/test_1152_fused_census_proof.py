"""tessera#1142: fused census proof for R1152 and the priced pairs.

R1152 is admitted by run-table cover on the base routed E4M3 cells.
Its table [4, 5] is the table stub B served at R1088 on the
e4m3mma fused launch. This file proves that mapping from the
packaged contract, the route source and the held receipt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.control import grid_for_name
from tessera.export import wire_recipe
from tessera.manifest import body_rate_cap
from tessera.routed_fused import (
    fused_launch_for_rates,
    routed_lane_rates,
    slot_words_for_rate,
)
from tessera.serving.contract import (
    cell_covers_rung,
    cell_executes,
    format_entry,
    load_serving_contract,
    rung_rates,
)
from tessera.serving.scheme import (
    STRUCTURE_ROUTED_MOE,
    refuse_unreachable_lane,
    refuse_unserveable_wire,
)
from tessera.serving import ext
from tessera.structure import STRUCTURE_ROUTED_MOE as ROUTED

E4M3 = "TESSERA_E4M3_K1"
LIBRARY = "e4m3mma"
BASE_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
)
TARGET = 1152
NEIGHBOR = 1088
FUSED_SYMBOL = "tessera.routed_fused.FusedRoutedWindowMoE.__call__"
ROOT = Path(__file__).resolve().parents[1]
E4M3MMA_RECEIPT = (
    ROOT / "experiments" / "results" / "glm53_u1_stub_b_e4m3mma_tp1_eager_census.json"
)


def _contract():
    return load_serving_contract()


def _row(contract):
    return format_entry(E4M3, contract)


def _cell(contract, cell_id):
    return next(c for c in contract["lane_eligibility"]["cells"] if c["id"] == cell_id)


def _rule_pairs(row):
    return [tuple(int(r) for r in t) for t in row["allowable_rungs"]["run_tables"] if len(t) == 2]


def _representative(pair, cell, row):
    """The lowest rung with this table that the cell covers, or None."""
    return min(
        (q for q in range(256, 2049) if rung_rates(row, q) == pair and cell_covers_rung(cell, q, row)),
        default=None,
    )


def _priced_pairs(cell, row):
    """The rule pairs the lane reaches and the cell covers: the pairs the allocator can price."""
    lane_rates = set(routed_lane_rates(LIBRARY))
    return [t for t in _rule_pairs(row) if set(t) <= lane_rates and _representative(t, cell, row) is not None]


def _e4m3mma_decoder(cell, row, q):
    """The fused e4m3mma decoder the scoped executes name at rung q."""
    scoped = cell_executes(cell, q256=q, entry=row)
    found = sorted(d for (s, d) in scoped if s == FUSED_SYMBOL and LIBRARY in d)
    assert len(found) == 1, (q, sorted(scoped))
    return found[0]


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_priced_pairs_come_from_the_rule_not_a_roster(cell_id):
    contract = _contract()
    row = _row(contract)
    cell = _cell(contract, cell_id)
    priced = _priced_pairs(cell, row)
    assert (3, 4) in priced
    assert (4, 5) in priced


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_each_priced_pair_maps_to_a_fused_launch_and_a_cell(cell_id):
    contract = _contract()
    row = _row(contract)
    cell = _cell(contract, cell_id)
    decoder = _e4m3mma_decoder(cell, row, NEIGHBOR)
    for pair in _priced_pairs(cell, row):
        launch = fused_launch_for_rates(pair, LIBRARY)
        assert launch.fits_sm121
        assert launch.slot_words >= max(slot_words_for_rate(r) for r in pair)
        rep = _representative(pair, cell, row)
        assert rep is not None
        assert (FUSED_SYMBOL, decoder) in cell_executes(cell, q256=rep, entry=row)


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_r1152_shares_the_held_r1088_pair(cell_id):
    contract = _contract()
    row = _row(contract)
    cell = _cell(contract, cell_id)
    assert rung_rates(row, TARGET) == rung_rates(row, NEIGHBOR) == (4, 5)
    assert NEIGHBOR in cell["rungs_q256"]
    launch = fused_launch_for_rates(rung_rates(row, TARGET), LIBRARY)
    held = fused_launch_for_rates(rung_rates(row, NEIGHBOR), LIBRARY)
    assert (launch.slot_words, launch.smem_gate_up, launch.smem_down) == (
        held.slot_words, held.smem_gate_up, held.smem_down)


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_r1152_admission_comes_from_the_covered_range(cell_id):
    contract = _contract()
    row = _row(contract)
    cell = _cell(contract, cell_id)
    covered = [q for q in range(256, 2049) if cell_covers_rung(cell, q, row)]
    assert covered == list(range(768, 1280))
    assert TARGET in covered
    decoder = _e4m3mma_decoder(cell, row, TARGET)
    assert (FUSED_SYMBOL, decoder) in cell_executes(cell, q256=TARGET, entry=row)
    assert fused_launch_for_rates((4, 5), LIBRARY).fits_sm121


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_export_gate_admits_r1152(cell_id):
    assert (
        refuse_unserveable_wire(
            "E4M3",
            TARGET,
            "WINDOW",
            "CHANNEL",
            family="TESSERA_FP8",
            span=1,
            target="stack.probe",
            structure=STRUCTURE_ROUTED_MOE,
        )
        == "TESSERA_FP8"
    )


def test_fused_lanes_read_r1152_as_4_5():
    grid = grid_for_name("E4M3")
    recipe = wire_recipe(grid, TARGET)
    cap = body_rate_cap(recipe.body, grid)
    for lane in (
        ext.ROUTED_FUSED_E4M3_MODULE_NAME,
        ext.ROUTED_FUSED_MMA_E4M3_MODULE_NAME,
    ):
        assert (
            refuse_unreachable_lane(
                lane,
                grid="E4M3",
                q256=TARGET,
                rate_cap=cap,
                body=recipe.body.name,
                plane=recipe.scale_plane.name,
                window_bits=int(recipe.window_bits),
                target="stack.probe",
                structure=ROUTED,
            )
            == (4, 5)
        )


@pytest.mark.parametrize("phase", ("decode", "prefill"))
def test_stub_b_served_the_shared_pair_on_the_fused_launch(phase):
    contract = _contract()
    row = _row(contract)
    cell = _cell(contract, BASE_IDS[0])
    decoder = _e4m3mma_decoder(cell, row, NEIGHBOR)
    receipt = json.loads(E4M3MMA_RECEIPT.read_text(encoding="utf-8"))
    routes = receipt["histogram"][phase]["routes"]
    match = [r for r in routes if (r["symbol"], r["decoder"]) == (FUSED_SYMBOL, decoder)]
    assert len(match) == 1
    assert match[0]["kind"] == "moe"
    assert match[0]["contract"] == "fp8_per_token_dynamic"
    assert match[0]["state"] == "served"
    assert match[0]["modules"] == 4
