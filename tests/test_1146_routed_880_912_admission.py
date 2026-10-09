"""tessera#1146: routed E4M3 880/912 through the true admission path.

Replaces stopped tessera#689. Positive admission needs a GPU route
census on image X (blocked). This file drives each rung solo through
the true path and names the missing layer. Derived cover, the export
gate and the run table already admit both rungs. Census rungs and
per-launch decoder tags do not. A cell-wide launch union is not read
as rung proof: each launch tag is checked per rung.
"""

from __future__ import annotations

import pytest

from tessera.serving.contract import (
    cell_covers_rung,
    format_entry,
    load_serving_contract,
    rung_allowable,
    rung_rates,
)
from tessera.serving.scheme import (
    STRUCTURE_ROUTED_MOE,
    refuse_unserveable_wire,
)

IMAGE_X = (
    "localhost/prismaquant/spark-vllm-nccl230@sha256:"
    "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5"
)
BASE_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
)
TARGETS = (880, 912)
NEIGHBOR = 896
MAIN_DECODERS = (
    "native_window_moe_compact",
    "native_routed_fused_window",
    "native_routed_fused_window_e4m3mma",
)


def _contract():
    return load_serving_contract()


def _cell(contract, cell_id):
    return next(
        c for c in contract["lane_eligibility"]["cells"] if c["id"] == cell_id
    )


def _row(contract):
    return format_entry("TESSERA_E4M3_K1", contract)


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_observed_image_is_held_launch_image(cell_id):
    cell = _cell(_contract(), cell_id)
    assert cell["runtime"]["image"] == IMAGE_X
    assert list(cell["runtime"]["execution_modes"]) == ["eager"]


@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_derived_cover_admits_each_rung_solo(cell_id, q):
    contract = _contract()
    assert cell_covers_rung(_cell(contract, cell_id), q, _row(contract))


@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_census_rungs_name_each_rung_solo(cell_id, q):
    assert q in _cell(_contract(), cell_id)["rungs_q256"]


@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_decoder_tags_name_each_rung_solo(cell_id, q):
    tags = {
        e["decoder"]: e.get("rungs_q256", [])
        for e in _cell(_contract(), cell_id)["executes"]
    }
    for decoder in MAIN_DECODERS:
        assert q in tags[decoder], decoder
    assert q not in tags["native_routed_window_classes_e4m3mma"]


@pytest.mark.parametrize("q", TARGETS)
def test_export_gate_admits_each_rung_solo(q):
    assert (
        refuse_unserveable_wire(
            "E4M3",
            q,
            "WINDOW",
            "CHANNEL",
            family="TESSERA_FP8",
            span=1,
            target="stack.probe",
            structure=STRUCTURE_ROUTED_MOE,
        )
        == "TESSERA_FP8"
    )


@pytest.mark.parametrize("q", TARGETS)
def test_run_table_matches_served_neighbor(q):
    contract = _contract()
    row = _row(contract)
    assert rung_allowable(row, q)
    assert rung_rates(row, q) == rung_rates(row, NEIGHBOR) == (3, 4)
