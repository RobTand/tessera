"""The eight GLM-image cells rest on a committed receipt, not on a summary of it.

Contract v38 (tessera#604) minted dense and routed ``TESSERA_E4M3_K1`` and
``TESSERA_BF16_K1`` cells on the GLM serving image from one route census of a
GLM-5.3-Flash stub (one dense and three MoE layers), TP 1, eager, resident.
The receipt and the stub's ``config.json`` are committed byte for byte under
``experiments/results/``; this module replays the census tool's own join --
``all_structure_agreement``, the function the tool calls on a live serve --
over the receipt's records against the PACKAGED table.

What it pins:

1. every one of the nine served modules, in both phases, joins a cell, and the
   join agrees (the fail-before: on contract v37 every module is
   ``unattested``, because no cell names this image and three of the four
   pairs were experimental);
2. the cells cover exactly the rungs the stub carried per family and
   structure -- a cell widened past its receipt, or a receipt rung dropped
   from a cell, fails here;
3. the receipt is the one the contract cites: same checkpoint config, same
   image, same toolchain, the serve's backends recorded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from tessera.serving.contract import (
    CENSUS_PHASE_REGIMES,
    PAYLOAD_FAMILY_BY_ROUTE,
    load_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
RECEIPT = RESULTS / "glm53_x_stub_tp1_eager_census.json"
CONFIG = RESULTS / "glm53_x_stub_config.json"
TOOL = ROOT / "tools" / "tessera_route_census.py"
RECEIPT_SHA256 = "d10738cd5692588a4afd4b8f3aeeb33924b99a8e144f84bb125624c660a34edc"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
CELL_IDS = sorted(
    f"tessera_{family}_{structure}_sm121_{regime}_resident"
    for family in ("e4m3_k1", "bf16_k1")
    for structure in ("dense", "routed_moe")
    for regime in ("decode", "batch"))


def _tool():
    spec = importlib.util.spec_from_file_location("glm_x_census_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _declared_rungs(receipt) -> dict:
    """The rung per MODULE, derived the way the tool derives it on a serve.

    ``declared_rung`` over the checkpoint's own ``config_groups``, then the
    runtime's name mapping the receipt recorded, so the replay reads the same
    two facts the live join read.
    """
    tool = _tool()
    groups = json.loads(CONFIG.read_text())["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    mapping = receipt["declared_name_mapping"]
    return {mapping.get(target) or target: rung for target, rung in rungs.items()}


def _agreement(contract):
    receipt = _receipt()
    return _tool().all_structure_agreement(
        receipt["records"], cells=contract["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        declared_rungs=_declared_rungs(receipt), record_owners=receipt["record_owner"],
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"])


def test_the_committed_receipt_is_the_one_the_contract_cites():
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == RECEIPT_SHA256
    receipt = _receipt()
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"execution_mode": "eager", "image": IMAGE}
    assert receipt["device"]["platform_token"] == "sm_121"
    assert receipt["checkpoint_sidecars"]["config.json"] == \
        hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    # The serve's attention and MoE-backend selection, recorded where the
    # cells' requires_serve_flags do not name it (the E2M1 precedent).
    assert receipt["env"]["TESSERA_RESEARCH_GLM53_NOPE"] == "1"
    assert receipt["engine_backends"] == {
        "attention_backend": "CUSTOM", "kv_cache_dtype": "fp8_ds_mla",
        "moe_backend": "triton", "kernel_config": {"enable_flashinfer_autotune": False},
        "trust_remote_code": True}
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in CELL_IDS:
        runtime = cells[cell_id]["runtime"]
        assert runtime["image"] == IMAGE, cell_id
        assert runtime["execution_modes"] == ["eager"], cell_id
        assert runtime["vllm"] == receipt["versions"]["vllm"], cell_id
        assert runtime["torch"] == receipt["versions"]["torch"], cell_id


def test_every_served_module_joins_a_cell_in_both_phases():
    block, problems = _agreement(load_serving_contract())
    assert problems == []
    assert block["agrees"] is True, json.dumps(block, indent=1)[:2000]
    seen = 0
    for structure, per in block["structures"].items():
        assert per["agrees"] is True, structure
        for phase, row in per["phases"].items():
            assert row["unattested"] == 0, (structure, phase, row)
            assert row["covered_by_cell"] == row["modules"] > 0, (structure, phase, row)
            seen += row["modules"]
    # Nine modules, two phases.
    assert seen == 18


def test_the_join_fails_on_the_table_before_these_cells():
    """The fail-before, as a mutation of the packaged table: drop the eight
    cells and no served module is covered any more."""
    contract = load_serving_contract()
    contract["lane_eligibility"]["cells"] = [
        c for c in contract["lane_eligibility"]["cells"] if c["id"] not in CELL_IDS]
    block, _problems = _agreement(contract)
    for per in block["structures"].values():
        for row in per["phases"].values():
            assert row["covered_by_cell"] == 0
            assert row["unattested"] == row["modules"]


def test_the_cells_cover_exactly_the_rungs_the_stub_carried():
    receipt = _receipt()
    rungs = _declared_rungs(receipt)
    carried: dict = {}
    for phase, records in receipt["records"].items():
        for name, record in records.items():
            owner = receipt["record_owner"][phase][name]
            family = PAYLOAD_FAMILY_BY_ROUTE[record["policy"].partition(":")[0]]
            structure = "routed_moe" if record["kind"] == "moe" else "dense"
            carried.setdefault((family, structure), set()).add(rungs[owner])
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in CELL_IDS:
        cell = cells[cell_id]
        assert set(cell["rungs_q256"]) == carried[(cell["family"], cell["structure"])], cell_id
        assert cell["evidence"]["grade"] == "route_only", cell_id
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"], cell_id
