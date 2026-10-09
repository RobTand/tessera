"""The R768 TP2 served census qualifies the twin routed cells, rung for rung.

A two-layer GLM-5.3-Flash stub with one routed stack (288 E4M3
experts at q256 768, run table [3]) served at tensor parallel 2
on the twin runtime image. Both ranks record the stack on the
E4M3 fused pair in both phases. The receipt is committed byte for
byte beside the stub config. This module replays the census
tool's own join over the receipt against the PACKAGED table.

What it pins:

1. the receipt is the one the contract cites: sha256, verdict
   ``served`` with no problems, the twin image and toolchain, the
   same checkpoint config, and a world of two that answered as
   two ranks on two hosts;
2. every rank, in both phases, ran the E4M3 fused routed pair on
   the served stack -- no fallback on either rank;
3. the joined records cover the twin cells at 768 through run
   table [3], with no D41 allowance and no hard-coded exception;
4. the twin cells carry exactly the rungs the receipts carried
   for their image, and the base cells still refuse 768.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES,
    PAYLOAD_FAMILY_BY_ROUTE,
    cell_covers_rung,
    load_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TOOL = ROOT / "tools" / "tessera_route_census.py"
RECEIPT_PATH = RESULTS / "glm53_r768_stub_tp2_eager_census.json"
CONFIG_PATH = RESULTS / "glm53_r768_stub_config.json"
RECEIPT_SHA256 = "350d07b68b40a3ee5bfea77d09641f40eb67a61397b0dee0129e0e7c16d0cd46"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
TWIN_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
)
BASE_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
)
FUSED = ("tessera.routed_fused.FusedRoutedWindowMoE.__call__",
         "native_routed_fused_window_e4m3mma")
TWIN_RUNGS = [768, 896, 928, 1024, 1088]


def _tool():
    spec = importlib.util.spec_from_file_location("r768_census_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_committed_receipt_is_the_two_rank_serve_it_says_it_is():
    raw = RECEIPT_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(raw)
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"execution_mode": "eager", "image": IMAGE}
    assert receipt["device"]["platform_token"] == "sm_121"
    assert receipt["checkpoint_sidecars"]["config.json"] == \
        hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    assert receipt["topology"]["requested_tensor_parallel_size"] == 2
    assert receipt["topology"]["observed_world_size"] == 2
    ranks = receipt["ranks"]
    assert sorted(r["rank"] for r in ranks) == [0, 1]
    assert sorted(r["node"] for r in ranks) == ["sparklina", "sparky"]
    for rank in ranks:
        assert rank["world_size"] == 2
        assert rank["runtime_image"] == IMAGE
        assert rank["platform_token"] == "sm_121"
        assert rank["lane_refusals"] == {}
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in TWIN_IDS:
        runtime = cells[cell_id]["runtime"]
        assert runtime["image"] == IMAGE, cell_id
        assert runtime["execution_modes"] == ["eager"], cell_id
        assert runtime["vllm"] == receipt["versions"]["vllm"], cell_id
        assert runtime["torch"] == receipt["versions"]["torch"], cell_id


def test_every_rank_ran_the_fused_pair_in_both_phases():
    receipt = _load(RECEIPT_PATH)
    for rank in receipt["ranks"]:
        for phase, records in rank["records"].items():
            assert len(records) == 1, (rank["rank"], phase)
            for name, record in records.items():
                assert record["kind"] == "moe", (rank["rank"], phase, name)
                assert (record["symbol"], record["decoder"]) == FUSED, (
                    rank["rank"], phase, name)
                assert record["contract"] == "fp8_per_token_dynamic"
                assert record["policy"] == "TESSERA_FP8:resident"
                assert record["state"] == "served"
    for phase, histogram in receipt["histogram"].items():
        assert histogram["tessera_modules"] == 2, phase
        assert histogram["other_route_modules"] == 0, phase


def test_the_served_stack_declares_rung_768():
    tool = _tool()
    groups = _load(CONFIG_PATH)["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    assert rungs == {"model.language_model.layers.1.mlp.experts": 768}


def test_the_joined_records_cover_the_twin_cells_without_an_allowance():
    tool = _tool()
    receipt = _load(RECEIPT_PATH)
    groups = _load(CONFIG_PATH)["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    mapping = receipt["declared_name_mapping"]
    rungs = {mapping.get(target) or target: rung for target, rung in rungs.items()}
    block, problems = tool.all_structure_agreement(
        receipt["records"], cells=load_serving_contract()["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        declared_rungs=rungs, record_owners=receipt["record_owner"],
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"])
    assert problems == []
    assert block["agrees"] is True, json.dumps(block, indent=1)[:2000]
    phases = block["structures"]["routed_moe"]["phases"]
    assert phases["decode"]["covered_by_cell"] == 1
    assert phases["decode"]["unattested"] == 0
    assert phases["prefill"]["covered_by_cell"] == 1
    assert phases["prefill"]["unattested"] == 0


def test_the_twin_cells_carry_768_and_table_3_and_the_base_cells_refuse_it():
    contract = load_serving_contract()
    row = next(e for e in contract["formats"] if e["family"] == "TESSERA_E4M3_K1")
    cells = {c["id"]: c for c in contract["lane_eligibility"]["cells"]}
    for cell_id in TWIN_IDS:
        cell = cells[cell_id]
        assert cell["rungs_q256"] == TWIN_RUNGS, cell_id
        assert cell["run_tables"] == [[3], [3, 4], [4], [4, 5]], cell_id
        assert cell_covers_rung(cell, 768, row), cell_id
        for q in (640, 1280, 1536, 2048):
            assert not cell_covers_rung(cell, q, row), (cell_id, q)
    for cell_id in BASE_IDS:
        cell = cells[cell_id]
        assert cell["rungs_q256"] == [832, 864, 896, 928, 944, 960, 1024, 1088], cell_id
        assert cell["run_tables"] == [[3, 4], [4], [4, 5]], cell_id
        assert not cell_covers_rung(cell, 768, row), cell_id


def test_the_export_gate_admits_a_routed_stack_at_768_through_the_twins():
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, refuse_unserveable_wire

    contract = load_serving_contract()
    assert refuse_unserveable_wire(
        "E4M3", 768, "WINDOW", "CHANNEL", family="TESSERA_FP8", span=1,
        target="stack.probe", structure=STRUCTURE_ROUTED_MOE,
        contract=contract) == "TESSERA_FP8"
    for q in (640, 1280, 1536, 2048):
        try:
            refuse_unserveable_wire(
                "E4M3", q, "WINDOW", "CHANNEL", family="TESSERA_FP8", span=1,
                target="stack.probe", structure=STRUCTURE_ROUTED_MOE,
                contract=contract)
        except ValueError:
            continue
        raise AssertionError(f"rung {q} is admitted without a served receipt")
