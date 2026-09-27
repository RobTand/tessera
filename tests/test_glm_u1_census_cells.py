"""The GLM-image cells rest on committed receipts, all of them, rung for rung.

Contract v39 (tessera#604, second half) minted the four ``TESSERA_E2M1_K2``
cells on the GLM serving image and widened the six E4M3/BF16 dense and routed
E4M3 cells v38 minted there.  The receipts are eight route censuses of
eight-layer GLM-5.3-Flash stubs (three dense-MLP and five MoE layers), TP 1,
eager, resident, each committed byte for byte under ``experiments/results/``
beside the stub's ``config.json``.  Every wire in them was cut from an existing
encode campaign's receipts; none was encoded for the census.  This module
replays the census tool's own join over each receipt's records against the
PACKAGED table, the way ``test_glm_x_census_cells`` does for the v38 receipt.

What it pins:

1. every served module of every receipt, in both phases, joins a cell (the
   fail-before: drop the v39 E2M1 cells and the all-E2M1 stub is unattested);
2. each GLM-image cell covers EXACTLY the rungs the nine receipts (these
   eight and v38's) carried for its family and structure -- a cell widened
   past its receipts, or a receipt rung dropped from a cell, fails here;
3. each receipt is the one the contract cites: same checkpoint config, same
   image and toolchain, the serve's backends recorded, and the E2M1 modules on
   the two native A4 launches.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES,
    PAYLOAD_FAMILY_BY_ROUTE,
    load_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TOOL = ROOT / "tools" / "tessera_route_census.py"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
RECEIPTS = {
    "a": "512106785e597729a6458576d8bca03d658dd47f9eef3ac987b32e576ce6f3c3",
    "b": "1de904a5a0852e126b36ee1cee9e0b9072264c8049e6ef7959f0c9e99b0e933d",
    "d": "11c6c2a3cc534889f26ae6e3337035ae76997099d16224215c43c750eec9dc96",
    "s1": "ac37c0adeb96e1cca049f924f29313c21ed9218a5d241d6882c778809052a320",
    "s2": "f21480e479bf7a00190c512849721fff8e4fe6fe44b12757c0f99a46380a5f3e",
    "s3": "5b574f675ba68846a8c39e08ef66a57597ee5ed3074edfa9105f802b7e802dfe",
    "s4": "c80b789af160bd5708e28f93c071d8154c3ee1b1f0e3cadb8d24e0b96723a77c",
    "s5": "bba21efae20ff37a52720ce2df08affbf1deab1b3e3308db84c7a210ba930344",
}
#: The v38 receipt the six widened cells were first minted on; its rungs are
#: part of what each cell must cover.
V38 = ("glm53_x_stub_tp1_eager_census.json", "glm53_x_stub_config.json")
MINTED = sorted(f"tessera_e2m1_k2_{structure}_sm121_{regime}_resident"
                for structure in ("dense", "routed_moe") for regime in ("decode", "batch"))
GLM_CELLS = sorted(
    [f"tessera_{family}_{structure}_sm121_{regime}_resident"
     for family in ("e4m3_k1", "bf16_k1")
     for structure in ("dense", "routed_moe")
     for regime in ("decode", "batch")] + MINTED)
A4_LAUNCH = {"dense": ("tessera.kernel_a4.a4_span2_gemm", "native_span2_gemm"),
             "routed_moe": ("tessera.kernel_a4.a4_span2_grouped_gemm", "native_span2_grouped")}


def _tool():
    spec = importlib.util.spec_from_file_location("glm_u1_census_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(stub):
    return (RESULTS / f"glm53_u1_stub_{stub}_tp1_eager_census.json",
            RESULTS / f"glm53_u1_stub_{stub}_config.json")


def _load(receipt_path):
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _declared_rungs(tool, receipt, config_path) -> dict:
    """The rung per MODULE, derived as the tool derives it on a serve."""
    groups = json.loads(config_path.read_text())["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    mapping = receipt["declared_name_mapping"]
    return {mapping.get(target) or target: rung for target, rung in rungs.items()}


def _agreement(tool, contract, receipt, config_path):
    return tool.all_structure_agreement(
        receipt["records"], cells=contract["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        declared_rungs=_declared_rungs(tool, receipt, config_path),
        record_owners=receipt["record_owner"],
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"])


def _carried(tool, receipt, config_path) -> dict:
    rungs = _declared_rungs(tool, receipt, config_path)
    carried: dict = {}
    for phase, records in receipt["records"].items():
        for name, record in records.items():
            owner = receipt["record_owner"][phase][name]
            family = PAYLOAD_FAMILY_BY_ROUTE[record["policy"].partition(":")[0]]
            structure = "routed_moe" if record["kind"] == "moe" else "dense"
            carried.setdefault((family, structure), set()).add(rungs[owner])
    return carried


@pytest.mark.parametrize("stub", sorted(RECEIPTS))
def test_each_committed_receipt_is_the_one_the_contract_cites(stub):
    receipt_path, config_path = _paths(stub)
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == RECEIPTS[stub]
    receipt = _load(receipt_path)
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"execution_mode": "eager", "image": IMAGE}
    assert receipt["device"]["platform_token"] == "sm_121"
    assert receipt["checkpoint_sidecars"]["config.json"] == \
        hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    assert receipt["env"]["TESSERA_RESEARCH_GLM53_NOPE"] == "1"
    assert receipt["engine_backends"] == {
        "attention_backend": "CUSTOM", "kv_cache_dtype": "fp8_ds_mla",
        "moe_backend": "triton", "kernel_config": {"enable_flashinfer_autotune": False},
        "trust_remote_code": True}
    # 3 dense MLPs x 2 + 5 shared-expert blocks x 2 + 5 routed stacks.
    assert receipt["decoder_coverage"]["phases"]["decode"]["modules"] == 21
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in GLM_CELLS:
        runtime = cells[cell_id]["runtime"]
        assert runtime["image"] == IMAGE, cell_id
        assert runtime["execution_modes"] == ["eager"], cell_id
        assert runtime["vllm"] == receipt["versions"]["vllm"], cell_id
        assert runtime["torch"] == receipt["versions"]["torch"], cell_id


@pytest.mark.parametrize("stub", sorted(RECEIPTS))
def test_every_served_module_joins_a_cell_in_both_phases(stub):
    tool = _tool()
    receipt_path, config_path = _paths(stub)
    block, problems = _agreement(tool, load_serving_contract(), _load(receipt_path), config_path)
    assert problems == []
    assert block["agrees"] is True, json.dumps(block, indent=1)[:2000]
    seen = 0
    for structure, per in block["structures"].items():
        assert per["agrees"] is True, structure
        for phase, row in per["phases"].items():
            assert row["unattested"] == 0, (structure, phase, row)
            assert row["covered_by_cell"] == row["modules"] > 0, (structure, phase, row)
            seen += row["modules"]
    assert seen == 42  # 21 modules, two phases


def test_the_all_e2m1_stub_is_unattested_without_the_minted_cells():
    """The fail-before, as a mutation of the packaged table: drop the four
    E2M1 cells v39 minted and no module of the all-E2M1 stub is covered."""
    tool = _tool()
    contract = load_serving_contract()
    contract["lane_eligibility"]["cells"] = [
        c for c in contract["lane_eligibility"]["cells"] if c["id"] not in MINTED]
    receipt_path, config_path = _paths("d")
    block, _problems = _agreement(tool, contract, _load(receipt_path), config_path)
    for per in block["structures"].values():
        for row in per["phases"].values():
            assert row["covered_by_cell"] == 0
            assert row["unattested"] == row["modules"] > 0


def test_the_e2m1_modules_ran_the_native_a4_launches():
    receipt = _load(_paths("d")[0])
    for phase, records in receipt["records"].items():
        assert len(records) == 21, phase
        for name, record in records.items():
            structure = "routed_moe" if record["kind"] == "moe" else "dense"
            assert (record["symbol"], record["decoder"]) == A4_LAUNCH[structure], (phase, name)
            assert record["contract"] == "e2m1_group16_ue4m3_static", (phase, name)
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in MINTED:
        cell = cells[cell_id]
        assert [(e["symbol"], e["decoder"]) for e in cell["executes"]] == [
            A4_LAUNCH[cell["structure"]]], cell_id


def test_the_glm_cells_cover_exactly_the_rungs_the_nine_receipts_carried():
    tool = _tool()
    carried: dict = {}
    sources = [_paths(stub) for stub in sorted(RECEIPTS)]
    sources.append((RESULTS / V38[0], RESULTS / V38[1]))
    for receipt_path, config_path in sources:
        for key, rungs in _carried(tool, _load(receipt_path), config_path).items():
            carried.setdefault(key, set()).update(rungs)
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in GLM_CELLS:
        cell = cells[cell_id]
        assert set(cell["rungs_q256"]) == carried[(cell["family"], cell["structure"])], cell_id
        assert cell["evidence"]["grade"] == "route_only", cell_id
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"], cell_id
