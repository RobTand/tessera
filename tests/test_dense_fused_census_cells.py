"""The two pinned-image E4M3 dense cells name the fused dense identity on their
OWN receipts (contract v43).

``tessera_e4m3_k1_dense_sm121_{decode,batch}`` were attested by the v34 census
of ``qwen3-0.6b-uniform-R1024`` -- 112 E4M3 dense modules, every one q256 1024
with rows a multiple of 128 -- on the pinned ``vllm/vllm-openai`` image, in both
residencies.  Contract v43 adds ``tessera::fused_window_dense`` /
``native_fused_window_dense`` to their ``executes`` (the table derives the pair
for every window dense cell whose rungs the lane reaches), which a cell may
carry only if a serve on its runtime recorded it.  Withdrawing the two cells
instead would have moved ``versions.default_serve_image`` onto a build no
registry serves, so the artifact was served again on the pinned image with the
fused lane as the dispatch, once per residency, and the two receipts are
committed byte for byte under ``experiments/results/`` beside the artifact's
``config.json``.  This module replays the census tool's own join over each
receipt against the PACKAGED table, the way ``test_glm_u1_census_cells`` does
for the GLM-image receipts.

What it pins:

1. each receipt is the one the changelog cites (sha256), was served on the
   two cells' runtime (image, vLLM, torch), in the residency its name says,
   and recorded the fused pair on all 112 modules in both phases;
2. every served module of both receipts joins one of the two cells in both
   phases and agrees with it, and on a table whose two cells lack the fused
   pair (the fail-before: v42's executes) every joined record DISAGREES --
   the join is by scope, the verdict is by launch;
3. the two cells' rungs are exactly what the receipts carried (1024).
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
from tessera.serving.runtime_image import pinned_reference
from tessera.serving.scheme import FUSED_WINDOW_DENSE_SYMBOL, WINDOW_GEMM_SYMBOL

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TOOL = ROOT / "tools" / "tessera_route_census.py"
CONFIG = RESULTS / "qwen3_0_6b_uniform_r1024_config.json"
RECEIPTS = {
    "resident": ("qwen3_0_6b_uniform_r1024_fused_resident_eager_census.json",
                 "9b7c0eb5b57ba5cb323c16e55196368682a9e734c07901bb7daf2072abda4684"),
    "streamed": ("qwen3_0_6b_uniform_r1024_fused_streamed_eager_census.json",
                 "4405c4c1e7fa77c235e6db344e3c70ac219ad44eb6a5378d8cad9cc17c40abfb"),
}
CELLS = ("tessera_e4m3_k1_dense_sm121_decode", "tessera_e4m3_k1_dense_sm121_batch")
FUSED_PAIR = (FUSED_WINDOW_DENSE_SYMBOL, "native_fused_window_dense")
TRITON_PAIR = (WINDOW_GEMM_SYMBOL, "native_window_gemm")
MODULES = 112


def _tool():
    spec = importlib.util.spec_from_file_location("dense_fused_census_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(mode):
    name, _sha = RECEIPTS[mode]
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _declared_rungs(tool, receipt) -> dict:
    groups = json.loads(CONFIG.read_text())["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    mapping = receipt["declared_name_mapping"]
    return {mapping.get(target) or target: rung for target, rung in rungs.items()}


def _agreement(tool, contract, receipt):
    return tool.all_structure_agreement(
        receipt["records"], cells=contract["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        declared_rungs=_declared_rungs(tool, receipt),
        record_owners=receipt["record_owner"],
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"])


@pytest.mark.parametrize("mode", sorted(RECEIPTS))
def test_each_committed_receipt_is_the_one_the_changelog_cites(mode):
    name, sha = RECEIPTS[mode]
    assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == sha
    receipt = _load(mode)
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"execution_mode": "eager", "image": pinned_reference()}
    assert receipt["device"]["platform_token"] == "sm_121"
    assert receipt["checkpoint_sidecars"]["config.json"] == \
        hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert receipt["env"]["TESSERA_SERVE_MODE"] == mode
    assert receipt["decoder_coverage"]["required"] == ["native_fused_window_dense"]
    for phase in ("decode", "prefill"):
        records = receipt["records"][phase]
        assert len(records) == MODULES, phase
        pairs = {(rec["symbol"], rec["decoder"]) for rec in records.values()}
        assert pairs == {FUSED_PAIR}, (phase, pairs)
        assert {rec["policy"] for rec in records.values()} == {f"TESSERA_FP8:{mode}"}
        assert {rec["contract"] for rec in records.values()} == {"fp8_per_token_dynamic"}
        assert {rec["kind"] for rec in records.values()} == {"dense"}
        assert receipt["decoder_coverage"]["phases"][phase]["decoders"] == {
            "native_fused_window_dense": MODULES}
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in CELLS:
        runtime = cells[cell_id]["runtime"]
        assert runtime["image"] == receipt["runtime"]["image"], cell_id
        assert runtime["execution_modes"] == ["eager"], cell_id
        assert runtime["vllm"] == receipt["versions"]["vllm"] == "0.28.0", cell_id
        assert runtime["torch"] == receipt["versions"]["torch"], cell_id
        assert cells[cell_id]["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident|streamed"]


@pytest.mark.parametrize("mode", sorted(RECEIPTS))
def test_every_served_module_joins_a_pinned_image_cell_in_both_phases(mode):
    tool = _tool()
    block, problems = _agreement(tool, load_serving_contract(), _load(mode))
    assert problems == []
    assert block["agrees"] is True, json.dumps(block, indent=1)[:2000]
    dense = block["structures"]["dense"]
    assert set(block["structures"]) == {"dense"}
    assert dense["agrees"] is True
    for phase, cell_id in (("decode", CELLS[0]), ("prefill", CELLS[1])):
        row = dense["phases"][phase]
        assert row["unattested"] == 0 and row["unsupported_records"] == 0, (phase, row)
        assert row["covered_by_cell"] == row["modules"] == MODULES, (phase, row)
        assert row["cells"] == {cell_id: MODULES}, (phase, row)


def test_the_join_fails_on_the_table_before_v43():
    """The fail-before, as a mutation of the packaged table: strip the fused
    pair from the two cells' ``executes`` (v42's rows).  The join is by scope
    (family, residency, regime, rung, runtime), so every record still lands on
    its cell -- and every one of them then DISAGREES with it: 112 problems per
    phase naming the fused pair the cell does not publish, and the dense block
    reads ``agrees: False`` for both receipts."""
    tool = _tool()
    contract = load_serving_contract()
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["id"] in CELLS:
            cell["executes"] = [e for e in cell["executes"]
                                if (e["symbol"], e["decoder"]) != FUSED_PAIR]
            assert [(e["symbol"], e["decoder"]) for e in cell["executes"]] == [TRITON_PAIR]
    for mode in RECEIPTS:
        block, problems = _agreement(tool, contract, _load(mode))
        dense = block["structures"]["dense"]
        assert dense["agrees"] is False and block["agrees"] is False, mode
        for row in dense["phases"].values():
            assert row["covered_by_cell"] == MODULES, (mode, row)
        assert len(problems) == 2 * MODULES, (mode, len(problems))
        for problem in problems:
            assert repr(FUSED_PAIR) in problem, problem
            assert any(f"cell {cell_id!r}" in problem for cell_id in CELLS), problem
            assert "does not publish" in problem, problem


def test_the_two_cells_name_both_launches_and_exactly_the_receipts_rungs():
    tool = _tool()
    carried = set()
    for mode in RECEIPTS:
        receipt = _load(mode)
        rungs = _declared_rungs(tool, receipt)
        for phase, records in receipt["records"].items():
            for name in records:
                carried.add(rungs[receipt["record_owner"][phase][name]])
    assert carried == {1024}
    cells = {c["id"]: c for c in load_serving_contract()["lane_eligibility"]["cells"]}
    for cell_id in CELLS:
        cell = cells[cell_id]
        assert set(cell["rungs_q256"]) == carried, cell_id
        pairs = [(e["symbol"], e["decoder"]) for e in cell["executes"]]
        assert pairs == [TRITON_PAIR, FUSED_PAIR], cell_id
        assert cell["family"] == "TESSERA_E4M3_K1" and cell["structure"] == "dense", cell_id
        assert cell["activation_contract"] == "fp8_per_token_dynamic", cell_id
