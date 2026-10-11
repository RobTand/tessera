"""tessera#1146: routed E4M3 880/912 through the true admission path.
Replaces stopped tessera#689. Record #689 as parent tie. Note #689 state. State stays OPEN with ig:stopped label. Author is RobTand with no assignee. Claim no owner beyond issues-ts CPU slice and T-8 GPU handoff. Accept PR #734 as taken-over work. Merge commit is 6a7b7616848df1c6722962a12f3ab31dd4089f07. Head is afe6547f76d117e4437ebb8d09928468405d6eba. PR adds planning tools only. It adds no cells and no ranges. Cite pact rows from #689 body. Rows are 768 at 880 and 736 at 912. File pact/gamut/CELL-COVERAGE-2026-09-28.md is absent in this checkout. Record no pact digest. Record no close proof. Issue stays OPEN and stopped. Prove 880 and 912 through derived cover on contract v66. Use held 896 and oracle [3, 4]. Match image X digest f8dbe1a0. Claim no new digest. Add no source change. Hold this file as regression pin only. Base is 69aae020048458a168296ef534ad29938299cf83. Diff base to head shows test file only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.control import grid_for_name
from tessera.export import wire_recipe
from tessera.manifest import body_rate_cap
from tessera.serving import ext
from tessera.serving.contract import (
    cell_covers_rung,
    cell_executes,
    format_entry,
    load_serving_contract,
    rung_allowable,
    rung_rates,
)
from tessera.serving.scheme import (
    STRUCTURE_ROUTED_MOE,
    refuse_unreachable_lane,
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
CLASS_DECODER = "native_routed_window_classes_e4m3mma"
COMPACT_SYMBOL = "tessera.native_window_moe.NativeWindowMoE.__call__"
FUSED_SYMBOL = "tessera.routed_fused.FusedRoutedWindowMoE.__call__"
# Rows 768 at 880 and 736 at 912 come from #689 body Severity line.
# File pact/gamut/CELL-COVERAGE-2026-09-28.md is absent here. Record no digest.
PACT_ROWS = {880: 768, 912: 736}
ROOT = Path(__file__).resolve().parents[1]
HELD_RECEIPT = ROOT / "experiments" / "results" / "glm53_x_stub_tp1_eager_census.json"


def _contract():
    return load_serving_contract()


def _cell(contract, cell_id):
    return next(
        c for c in contract["lane_eligibility"]["cells"] if c["id"] == cell_id
    )


def _row(contract):
    return format_entry("TESSERA_E4M3_K1", contract)


def _held_receipt():
    return json.loads(HELD_RECEIPT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cell_id", BASE_IDS)
def test_observed_image_is_held_launch_image(cell_id):
    cell = _cell(_contract(), cell_id)
    assert cell["runtime"]["image"] == IMAGE_X
    assert list(cell["runtime"]["execution_modes"]) == ["eager"]

@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_device_fit_holds_for_each_rung_solo(cell_id, q):
    """Each native cell fits the device for 880 and 912."""
    contract = _contract()
    cell = _cell(contract, cell_id)
    row = _row(contract)
    assert cell["platform"] == "sm_121"
    assert cell["qualification"] == "device_qualified"
    assert cell["route_status"] == "backed_with_serve_flag"
    assert cell["activation_contract"] == "fp8_per_token_dynamic"
    assert cell["runtime"]["image"] == IMAGE_X
    assert _held_receipt()["device"]["platform_token"] == cell["platform"]
    assert cell_covers_rung(cell, q, row)


@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_derived_cover_admits_each_rung_solo(cell_id, q):
    contract = _contract()
    assert cell_covers_rung(_cell(contract, cell_id), q, _row(contract))


@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_held_table_covers_each_rung_solo(cell_id, q):
    """Held 896 attests table [3, 4]; 880 and 912 share it."""
    contract = _contract()
    cell = _cell(contract, cell_id)
    row = _row(contract)
    assert NEIGHBOR in cell["rungs_q256"]
    assert rung_rates(row, q) == rung_rates(row, NEIGHBOR) == (3, 4)
    assert [3, 4] in cell["run_tables"]
    assert cell_covers_rung(cell, q, row)


@pytest.mark.parametrize("cell_id", BASE_IDS)
@pytest.mark.parametrize("q", TARGETS)
def test_scoped_decoders_match_held_launches_solo(cell_id, q):
    """Scoped decoders admit each rung solo; union does not prove it."""
    contract = _contract()
    cell = _cell(contract, cell_id)
    row = _row(contract)
    scoped = cell_executes(cell, q256=q, entry=row)
    expect = {(COMPACT_SYMBOL, d) for d in MAIN_DECODERS[:1]} | {
        (FUSED_SYMBOL, d) for d in MAIN_DECODERS[1:]
    }
    assert scoped == expect
    union = cell_executes(cell)
    assert (FUSED_SYMBOL, CLASS_DECODER) in union
    assert (FUSED_SYMBOL, CLASS_DECODER) not in scoped


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


@pytest.mark.parametrize("q", TARGETS)
def test_lane_admits_each_rung_solo(q):
    """Each fused lane reads [3, 4] on a routed stack."""
    from tessera.structure import STRUCTURE_ROUTED_MOE as ROUTED

    grid = grid_for_name("E4M3")
    recipe = wire_recipe(grid, q)
    cap = body_rate_cap(recipe.body, grid)
    for lane in (
        ext.ROUTED_FUSED_E4M3_MODULE_NAME,
        ext.ROUTED_FUSED_MMA_E4M3_MODULE_NAME,
    ):
        assert (
            refuse_unreachable_lane(
                lane,
                grid="E4M3",
                q256=q,
                rate_cap=cap,
                body=recipe.body.name,
                plane=recipe.scale_plane.name,
                window_bits=int(recipe.window_bits),
                target="stack.probe",
                structure=ROUTED,
            )
            == (3, 4)
        )


def test_held_receipt_names_image_and_compact_896():
    """Held TP1 serve on image X records compact 896 in both phases."""
    receipt = _held_receipt()
    assert receipt["runtime"]["image"] == IMAGE_X
    assert receipt["runtime"]["execution_mode"] == "eager"
    assert receipt["device"]["platform_token"] == "sm_121"
    assert receipt["verdict"] == "served"
    assert receipt["problems"] == []
    for phase in ("decode", "prefill"):
        routes = receipt["histogram"][phase]["routes"]
        match = [
            r
            for r in routes
            if r["decoder"] == "native_window_moe_compact"
            and r["symbol"] == COMPACT_SYMBOL
            and r["state"] == "served"
        ]
        assert len(match) == 1
        assert match[0]["contract"] == "fp8_per_token_dynamic"


def test_oracle_attests_table_3_4_on_image_x():
    """Oracle on image X attests [3, 4]; rule cites it."""
    contract = _contract()
    row = _row(contract)
    assert [3, 4] in row["allowable_rungs"]["run_tables"]
    assert row["allowable_rungs"]["evidence"] == [
        "docs/measurements/2026-09-30-t8-run-tables.md"
    ]
    assert (ROOT / row["allowable_rungs"]["evidence"][0]).exists()


@pytest.mark.parametrize("q", TARGETS)
def test_pact_rows_resolve_on_image_x(q):
    """Priced rows from #689 body now resolve on image X."""
    assert PACT_ROWS == {880: 768, 912: 736}
    contract = _contract()
    for cell_id in BASE_IDS:
        assert cell_covers_rung(_cell(contract, cell_id), q, _row(contract))
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


def test_producer_answers_hold_outside_routed_880_912_scope():
    """Producer answers match pinned base answers for all eight queries."""
    contract = _contract()
    queries = [
        ("E4M3", 880, "WINDOW", "CHANNEL", "TESSERA_FP8", 1, "stack.probe",
         STRUCTURE_ROUTED_MOE),
        ("E4M3", 912, "WINDOW", "CHANNEL", "TESSERA_FP8", 1, "stack.probe",
         STRUCTURE_ROUTED_MOE),
        ("E4M3", 896, "WINDOW", "CHANNEL", "TESSERA_FP8", 1, "stack.probe",
         STRUCTURE_ROUTED_MOE),
        ("E4M3", 1024, "WINDOW", "CHANNEL", "TESSERA_FP8", 1, "stack.probe",
         STRUCTURE_ROUTED_MOE),
        ("E4M3", 880, "WINDOW", "CHANNEL", "TESSERA_FP8", 1, "dense.probe",
         "dense"),
        ("E4M3", 912, "WINDOW", "CHANNEL", "TESSERA_FP8", 1, "dense.probe",
         "dense"),
        ("E2M1x2", 896, "WINDOW", "LUT16", "TESSERA_NVFP4", 2, "stack.probe",
         STRUCTURE_ROUTED_MOE),
        ("BF16", 1024, "WINDOW", "CHANNEL", "TESSERA_BF16", 1, "stack.probe",
         STRUCTURE_ROUTED_MOE),
    ]
    # Pinned answers at base 69aae02004. Diff base to head shows no src change.
    # Head answers must equal these base answers. Scope adds no producer change.
    expected = [
        "TESSERA_FP8",
        "TESSERA_FP8",
        "TESSERA_FP8",
        "TESSERA_FP8",
        "TESSERA_FP8",
        "TESSERA_FP8",
        "refused:tessera export 'stack.probe': TESSERA_NVFP4 decodes the LUT scale plane to its n",
        "refused:tessera export 'stack.probe': no lane_eligibility cell in runtime_contract.json ",
    ]
    answers = [_answer(contract, *a) for a in queries]
    assert answers == expected
    assert contract["contract_version"] == 66


def _answer(contract, grid, q256, body, plane, family, span, target, structure):
    try:
        return refuse_unserveable_wire(
            grid,
            q256,
            body,
            plane,
            family=family,
            span=span,
            target=target,
            structure=structure,
            contract=contract,
        )
    except ValueError as caught:
        return f"refused:{str(caught)[:80]}"
