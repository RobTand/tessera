"""Routed T-8 cells admit each measured whole-bit rung, and refuse the rest.

Contract v59 admits run table [3] (rung 768) on the two base routed E4M3
cells. Rung 768 is D41-measured, not receipt-served: its evidence is the
committed D41 slice experiments/results/t8_d41_r768_census.json (status
measured, supported, 20 of 20 cells; gate_up M1 241.96 us) with the
run-table oracle beside it
(docs/measurements/2026-09-30-t8-run-tables.md). Rungs above 1152 carry no
D41 speed row (audit matrix section 2a), so the cells refuse them until the
sweep extends. Rung 640 sits below the sweep range and stays refused too.
The runtime twins stay pinned: no image census ran at 768 there.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    cell_covers_rung,
    format_entry,
    load_serving_contract,
)

E4M3 = "TESSERA_E4M3_K1"
BASE_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
)
TWIN_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
)
MEASURED_CENSUS = [768, 832, 864, 896, 928, 944, 960, 1024, 1088]
UNMEASURED = (640, 1280, 1300, 1536, 1792, 2048)


@pytest.fixture(scope="module")
def contract():
    return load_serving_contract()


def _cell(contract, cell_id):
    return next(c for c in contract["lane_eligibility"]["cells"] if c["id"] == cell_id)


def _row(contract):
    return next(e for e in contract["formats"] if e["family"] == E4M3)


def test_base_routed_cells_carry_table_3_with_census_rung_768(contract):
    for cell_id in BASE_IDS:
        cell = _cell(contract, cell_id)
        assert cell["rungs_q256"] == MEASURED_CENSUS, cell_id
        assert cell["run_tables"] == [[3], [3, 4], [4], [4, 5]], cell_id


def test_measured_rungs_are_covered_through_the_rule(contract):
    row = _row(contract)
    for cell_id in BASE_IDS:
        cell = _cell(contract, cell_id)
        covered = [q for q in range(256, 2049) if cell_covers_rung(cell, q, row)]
        assert covered == [768] + list(range(769, 1280)), cell_id


def test_unmeasured_rungs_are_refused_by_every_base_routed_cell(contract):
    row = _row(contract)
    for cell_id in BASE_IDS:
        cell = _cell(contract, cell_id)
        for q in UNMEASURED:
            assert not cell_covers_rung(cell, q, row), (cell_id, q)


def test_the_export_gate_admits_a_routed_stack_at_768(contract):
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, refuse_unserveable_wire

    assert refuse_unserveable_wire(
        "E4M3", 768, "WINDOW", "CHANNEL", family="TESSERA_FP8", span=1,
        target="stack.probe", structure=STRUCTURE_ROUTED_MOE,
        contract=contract) == "TESSERA_FP8"


@pytest.mark.parametrize("q", [1280, 1536, 2048])
def test_the_export_gate_refuses_a_routed_stack_above_the_sweep(contract, q):
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, refuse_unserveable_wire

    with pytest.raises(ValueError) as caught:
        refuse_unserveable_wire(
            "E4M3", q, "WINDOW", "CHANNEL", family="TESSERA_FP8", span=1,
            target="stack.probe", structure=STRUCTURE_ROUTED_MOE, contract=contract)
    assert BASE_IDS[0] in str(caught.value)


def test_runtime_twins_stay_pinned_without_an_image_census(contract):
    for cell_id in TWIN_IDS:
        cell = _cell(contract, cell_id)
        assert cell["rungs_q256"] == [896, 928, 1024, 1088], cell_id
        assert cell["run_tables"] == [[3, 4], [4], [4, 5]], cell_id


CENSUS = Path(__file__).resolve().parents[1] / "experiments" / "results" / "t8_d41_r768_census.json"


def _census():
    return json.loads(CENSUS.read_text(encoding="utf-8"))


def test_the_committed_r768_census_carries_the_admission_evidence():
    """The committed D41 slice is the evidence the cells cite: rung 768,
    measured and supported with no anomaly flags, 20 of 20 cells, on the
    kernel build the table names. Its admits block states the exact scope
    the cells may take from it."""
    from tessera.serving.contract import rung_rates

    doc = _census()
    assert doc["table_version"] == 9
    assert doc["kernel_build"]["id"] == "e4m3mma-sm_121-d12fba61b3467f3e"
    rung = doc["rung"]
    assert rung["rung"] == 768
    assert rung["measurement_status"] == "measured" and rung["supported"] is True
    assert rung["anomaly_flags"] == []
    evidence = rung["measurements"]
    assert len(evidence) == 20
    assert {e["measurement_status"] for e in evidence} == {"measured"}
    assert {e["measurement_build_id"] for e in evidence} == {doc["kernel_build"]["id"]}
    gate_up_m1 = next(e for e in evidence if e["cell_id"] == "routed:gate_up:M1")
    assert gate_up_m1["kernel_time_us"] == pytest.approx(241.96, abs=1.0)
    assert doc["admits"] == {
        "family": "TESSERA_E4M3_K1", "structure": "routed_moe", "rungs": [768],
        "run_tables": [[3]],
        "cell_ids": ["tessera_e4m3_k1_routed_moe_sm121_decode_resident",
                     "tessera_e4m3_k1_routed_moe_sm121_batch_resident"],
        "pending": "TP2 served census at 768"}
    row = {"grid": "E4M3", "native_terminal_q256": 2048}
    assert rung_rates(row, 768) == (3,)
