"""Bind the IMG-N class scope to its TP2 census."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE,
    load_serving_contract, validate_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
IMAGE = "localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a"
PAIR = ("tessera.routed_fused.FusedRoutedWindowMoE.__call__",
        "native_routed_window_classes_e4m3mma")
RUNGS = [896, 928, 1024, 1088]
SPEC = importlib.util.spec_from_file_location("img_n_census", ROOT / "tools/tessera_route_census.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def scoped_cells(contract):
    return [c for c in contract["lane_eligibility"]["cells"]
            if c["family"] == "TESSERA_E4M3_K1" and c["structure"] == "routed_moe"
            and c["runtime"]["image"] == IMAGE]


def declared_rungs(receipt, config):
    groups = config["quantization_config"]["config_groups"]
    return {receipt["declared_name_mapping"].get(t, t): TOOL.declared_rung(g["scheme"])
            for g in groups.values() for t in g["targets"]}


def agreement(receipt, records, rungs, contract):
    return TOOL.all_structure_agreement(
        records, cells=contract["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121", declared_rungs=rungs,
        record_owners=receipt["record_owner"], families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt["runtime"]["image"], execution_mode="eager")


def test_receipt_identity_and_both_rank_replay():
    scope = json.loads((RESULTS / "glm53_img_n_v70_scope.json").read_text())
    assert scope["status"] == "served", "The approved TP2 census is not committed."
    raw = (ROOT / scope["receipt"]).read_bytes()
    config_raw = (ROOT / scope["config"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == scope["sha256"]
    assert hashlib.sha256(config_raw).hexdigest() == scope["config_sha256"]
    receipt, config = json.loads(raw), json.loads(config_raw)
    assert receipt["checkpoint_sidecars"]["config.json"] == scope["config_sha256"]
    assert receipt["versions"]["tessera_commit"] == scope["commit"]
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"image": IMAGE, "execution_mode": "eager"}
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    assert receipt["topology"]["requested_tensor_parallel_size"] == 2
    assert receipt["topology"]["observed_world_size"] == 2
    assert sorted(r["rank"] for r in receipt["ranks"]) == [0, 1]
    assert sorted(r["node"] for r in receipt["ranks"]) == ["sparklina", "sparky"]
    contract = load_serving_contract()
    rungs = declared_rungs(receipt, config)
    for rank in receipt["ranks"]:
        assert set(rank["records"]) == set(CENSUS_PHASE_REGIMES)
        assert rank["world_size"] == 2 and rank["runtime_image"] == IMAGE
        assert rank["platform_token"] == "sm_121" and rank["lane_refusals"] == {}
        block, problems = agreement(receipt, rank["records"], rungs, contract)
        assert problems == [] and block["agrees"] is True
        for phase, records in rank["records"].items():
            assert len(records) == 21
            routed = {name: r for name, r in records.items()
                      if r["kind"] == "moe" and r["policy"] == "TESSERA_FP8:resident"}
            assert len(routed) == 4
            owners = receipt["record_owner"][phase]
            assert {rungs[owners[name]] for name in routed} == set(RUNGS)
            assert {(r["symbol"], r["decoder"]) for r in routed.values()} == {PAIR}
            assert all(r["state"] == "served" for r in routed.values())
        damaged = copy.deepcopy(rank["records"])
        for rows in damaged.values():
            for record in rows.values():
                if record["kind"] == "moe" and record["policy"].startswith("TESSERA_FP8:"):
                    record["decoder"] = "native_routed_fused_window_e4m3mma"
        assert len(agreement(receipt, damaged, rungs, contract)[1]) == 8
        wrong_image = copy.deepcopy(receipt)
        wrong_image["runtime"]["image"] = "example.invalid/other@sha256:" + "0" * 64
        unseen, problems = agreement(wrong_image, rank["records"], rungs, contract)
        assert problems == []
        assert all(row["covered_by_cell"] == 0 for per in unseen["structures"].values()
                   for row in per["phases"].values())


def test_historical_img_n_records_refuse():
    receipt = json.loads((RESULTS / "glm53_u1_stub_b_nightly_tp1_eager_census.json").read_text())
    config = json.loads((RESULTS / "glm53_u1_stub_b_config.json").read_text())
    records = {phase: {name: r for name, r in rows.items()
                       if r["kind"] == "moe" and r["policy"].startswith("TESSERA_FP8:")}
               for phase, rows in receipt["records"].items()}
    assert all(len(rows) == 4 for rows in records.values())
    block, problems = agreement(receipt, records, declared_rungs(receipt, config), load_serving_contract())
    assert block["agrees"] is False and len(problems) == 8
    assert all("does not publish" in p for p in problems)


@pytest.mark.parametrize("axis", ["executes", "image", "execution_modes", "rungs_q256"])
def test_scope_cannot_borrow_historical_receipts(axis):
    contract = load_serving_contract()
    cells = scoped_cells(contract)
    assert len(cells) == 2
    validate_serving_contract(contract)
    for cell in cells:
        assert cell["executes"] == [{"symbol": PAIR[0], "decoder": PAIR[1], "rungs_q256": RUNGS}]
        assert cell["evidence"]["grade"] == "route_only"
        if axis == "executes":
            cell[axis][0]["decoder"] = "native_routed_fused_window_e4m3mma"
        elif axis == "rungs_q256":
            cell["executes"][0][axis] = [896]
        elif axis == "image":
            cell["runtime"][axis] = "example.invalid/other@sha256:" + "0" * 64
        else:
            cell["runtime"][axis] = ["eager", "compiled"]
    with pytest.raises(ValueError):
        validate_serving_contract(contract)
