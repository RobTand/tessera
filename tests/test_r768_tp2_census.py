"""Qualify the base-image R768 TP2 dispatch without an allowance.

Both ranks served the current class decoder in decode and batch.
The historical pairs keep their original rung scope.
The runtime twins remain unchanged and refuse R768.
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
    cell_covers_rung,
    load_serving_contract,
    validate_serving_contract,
    _SCOPED_EXECUTION_RECEIPTS,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TOOL = ROOT / "tools" / "tessera_route_census.py"
IMAGE_PREFIX = "localhost/prismaquant/spark-vllm-nccl230@sha256:"
TWIN_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
)
BASE_IDS = (
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
)
RECEIPTS = (
    ("glm53_r768_stub_tp2_eager_census.json", "glm53_r768_stub_config.json",
     "350d07b68b40a3ee5bfea77d09641f40eb67a61397b0dee0129e0e7c16d0cd46",
     IMAGE_PREFIX + "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a",
     "native_routed_fused_window_e4m3mma", TWIN_IDS),
    (Path(_SCOPED_EXECUTION_RECEIPTS[0]["receipt"]).name, "glm53_r768_stub_base_config.json",
     "3ece1c1fe96310531e2fa3496b4aec9387027144f4064ac11905d85d4f70d1c5",
     IMAGE_PREFIX + "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5",
     "native_routed_window_classes_e4m3mma", BASE_IDS),
)
SYMBOL = "tessera.routed_fused.FusedRoutedWindowMoE.__call__"
PHASES = {"decode", "prefill"}


def _tool():
    spec = importlib.util.spec_from_file_location("r768_census_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rungs(tool, receipt, config):
    groups = config["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    mapping = receipt["declared_name_mapping"]
    return {mapping.get(target) or target: rung for target, rung in rungs.items()}


def _join(tool, receipt, config, records, cells):
    return tool.all_structure_agreement(
        records, cells=cells, phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        declared_rungs=_rungs(tool, receipt, config), record_owners=receipt["record_owner"],
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE, runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"])


def test_the_committed_receipt_is_the_two_rank_serve_it_says_it_is():
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for filename, config_name, sha, image, _decoder, ids in RECEIPTS:
        raw = (RESULTS / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha, filename
        receipt = json.loads(raw)
        assert receipt["verdict"] == "served" and receipt["problems"] == [], filename
        assert receipt["runtime"] == {"execution_mode": "eager", "image": image}, filename
        assert receipt["device"]["platform_token"] == "sm_121"
        assert receipt["checkpoint_sidecars"]["config.json"] == \
            hashlib.sha256((RESULTS / config_name).read_bytes()).hexdigest()
        assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
        assert receipt["topology"]["requested_tensor_parallel_size"] == 2
        assert receipt["topology"]["observed_world_size"] == 2
        assert sorted(r["rank"] for r in receipt["ranks"]) == [0, 1]
        assert sorted(r["node"] for r in receipt["ranks"]) == ["sparklina", "sparky"]
        for rank in receipt["ranks"]:
            assert rank["world_size"] == 2
            assert rank["runtime_image"] == image
            assert rank["platform_token"] == "sm_121"
            assert rank["lane_refusals"] == {}
        for cell_id in ids:
            runtime = cells[cell_id]["runtime"]
            assert runtime["image"] == image, cell_id
            assert runtime["execution_modes"] == ["eager"], cell_id
            assert runtime["vllm"] == receipt["versions"]["vllm"], cell_id
            assert runtime["torch"] == receipt["versions"]["torch"], cell_id


def test_every_rank_ran_the_fused_pair_in_both_phases():
    for filename, _config, _sha, _image, decoder, _ids in RECEIPTS:
        receipt = _load(RESULTS / filename)
        for rank in receipt["ranks"]:
            assert set(rank["records"]) == PHASES, (filename, rank["rank"])
            for phase, records in rank["records"].items():
                assert len(records) == 1, (filename, rank["rank"], phase)
                for name, record in records.items():
                    assert record["kind"] == "moe", (rank["rank"], phase, name)
                    assert (record["symbol"], record["decoder"]) == (SYMBOL, decoder)
                    assert record["contract"] == "fp8_per_token_dynamic"
                    assert record["policy"] == "TESSERA_FP8:resident"
                    assert record["state"] == "served"
                    expected_m = 1 if phase == "decode" else 64
                    assert record["shape"] == f"M{expected_m}:N2048:K4096"
        assert set(receipt["histogram"]) == PHASES
        for histogram in receipt["histogram"].values():
            assert histogram["tessera_modules"] == 2
            assert histogram["other_route_modules"] == 0


def test_the_served_stack_declares_rung_768():
    tool = _tool()
    for filename, config_name, _sha, _image, _decoder, _ids in RECEIPTS:
        receipt = _load(RESULTS / filename)
        config = _load(RESULTS / config_name)
        assert _rungs(tool, receipt, config) == {"language_model.model.layers.1.mlp.experts": 768}


def test_the_base_records_join_their_exact_decoder_scope_without_an_allowance():
    import copy

    tool = _tool()
    cells = load_serving_contract()["lane_eligibility"]["cells"]
    for filename, config_name, _sha, _image, _decoder, ids in RECEIPTS:
        receipt, config = _load(RESULTS / filename), _load(RESULTS / config_name)
        qualified = ids == BASE_IDS
        for rank in receipt["ranks"]:
            block, problems = _join(tool, receipt, config, rank["records"], cells)
            assert problems == [], (filename, rank["rank"], problems)
            assert block["agrees"] is (True if qualified else None)
            phases = block["structures"]["routed_moe"]["phases"]
            assert set(phases) == PHASES
            for phase, row in phases.items():
                assert row["modules"] == 1
                assert row["covered_by_cell"] == int(qualified)
                assert row["unattested"] == int(not qualified)
                expected = BASE_IDS[0 if phase == "decode" else 1]
                assert row["cells"] == ({expected: 1} if qualified else {})
            if not qualified:
                continue
            # A receipt cannot qualify its decoder on another run table.
            changed_config = copy.deepcopy(config)
            groups = changed_config["quantization_config"]["config_groups"]
            for group in groups.values():
                for role in group["scheme"]["groups"].values():
                    role["q256"] = 896
            assert set(_rungs(tool, receipt, changed_config).values()) == {896}
            mismatch, failures = _join(tool, receipt, changed_config, rank["records"], cells)
            assert mismatch["agrees"] is False and len(failures) == 2
            # Historical launches cannot borrow the new class receipt.
            for decoder in ("native_window_moe_compact", "native_routed_fused_window",
                            "native_routed_fused_window_e4m3mma"):
                changed_records = copy.deepcopy(rank["records"])
                for records in changed_records.values():
                    for record in records.values():
                        record["decoder"] = decoder
                        if decoder == "native_window_moe_compact":
                            record["symbol"] = "tessera.native_window_moe.NativeWindowMoE.__call__"
                mismatch, failures = _join(tool, receipt, config, changed_records, cells)
                assert mismatch["agrees"] is False and len(failures) == 2
            # A contract cannot remove or widen the explicit decoder scope.
            for scope in (None, [768, 896], [896]):
                changed_contract = copy.deepcopy(load_serving_contract())
                for cell in changed_contract["lane_eligibility"]["cells"]:
                    if cell["id"] not in BASE_IDS:
                        continue
                    launch = next(e for e in cell["executes"] if e["decoder"] == _decoder)
                    if scope is None:
                        del launch["rungs_q256"]
                    else:
                        launch["rungs_q256"] = scope
                with pytest.raises(ValueError, match="rungs_q256"):
                    validate_serving_contract(changed_contract)


def test_only_the_base_cells_gain_the_served_rung():
    contract = load_serving_contract()
    row = next(e for e in contract["formats"] if e["family"] == "TESSERA_E4M3_K1")
    cells = {c["id"]: c for c in contract["lane_eligibility"]["cells"]}
    for ids, rungs in ((TWIN_IDS, [896, 928, 1024, 1088]),
                       (BASE_IDS, [768, 832, 864, 896, 928, 944, 960, 1024, 1088])):
        for cell_id in ids:
            cell = cells[cell_id]
            assert cell["rungs_q256"] == rungs, cell_id
            tables = ([[3]] if ids == BASE_IDS else []) + [[3, 4], [4], [4, 5]]
            assert cell["run_tables"] == tables, cell_id
            assert cell_covers_rung(cell, 768, row) is (ids == BASE_IDS)
            for q in (640, 1280, 1536, 2048):
                assert not cell_covers_rung(cell, q, row), (cell_id, q)
