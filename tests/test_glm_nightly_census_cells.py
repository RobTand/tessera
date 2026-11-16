"""Keep the historical IMG-N census and check its current contract agreement.

V70 gives routed E4M3 cells a separate class receipt at the same four rungs.
The historical routed launches now refuse. Dense E4M3 records still agree.
V68 BF16 receipts also retain separate scope from the old folded records.
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
    cell_runtime_id_suffix,
    cell_executes,
    format_entry,
    load_serving_contract,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TOOL = ROOT / "tools" / "tessera_route_census.py"
RECEIPT = RESULTS / "glm53_u1_stub_b_nightly_tp1_eager_census.json"
RECEIPT_SHA256 = "a5f1a4a198ef77ae77c86b4eccae28179621667e68d6577c6d4e0fc4b66d0c19"
SERVE_LOG = RESULTS / "glm53_u1_stub_b_nightly_tp1_eager_census.log"
CONFIG = RESULTS / "glm53_u1_stub_b_config.json"
#: The f8dbe1a0 receipt of the same stub on the E4M3 instruction (contract v47).
TWIN = RESULTS / "glm53_u1_stub_b_e4m3mma_tp1_eager_census.json"
MEASUREMENT = "docs/measurements/2026-09-30-glm-nightly-cells-and-graph-equivalence.md"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
VLLM = "0.30.1rc1.dev336+gaf5b4857e.d20260929"
TORCH = "2.13.0+cu130"
SCOPES = [("TESSERA_E4M3_K1", structure) for structure in ("dense", "routed_moe")]


def _tool():
    spec = importlib.util.spec_from_file_location("glm_nightly_census_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nightly_cells(contract) -> dict:
    return {c["id"]: c for c in contract["lane_eligibility"]["cells"]
            if c["runtime"]["image"] == IMAGE and c["family"] == "TESSERA_E4M3_K1"}


def _declared_rungs(tool, receipt, config_path=CONFIG) -> dict:
    groups = _load(config_path)["quantization_config"]["config_groups"]
    rungs = {target: tool.declared_rung(group["scheme"])
             for group in groups.values() for target in group["targets"]}
    mapping = receipt["declared_name_mapping"]
    return {mapping.get(target) or target: rung for target, rung in rungs.items()}


# v68 reuses these four IDs with new receipts; retain all other historical scopes.
_V68_BF16_IDS = {
    f"tessera_bf16_k1_{structure}_sm121_{regime}_resident_runtime_"
    "1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e"
    for structure in ("dense", "routed_moe") for regime in ("decode", "batch")
}


def _agreement(tool, contract, receipt):
    """Replay v48 E4M3 qualification; folded BF16 remains historical."""
    return tool.all_structure_agreement(
        receipt["records"], cells=[c for c in contract["lane_eligibility"]["cells"]
                                  if c["id"] not in _V68_BF16_IDS],
        phase_regimes=CENSUS_PHASE_REGIMES, platform="sm_121",
        declared_rungs=_declared_rungs(tool, receipt),
        record_owners=receipt["record_owner"],
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"])


def test_the_historical_nightly_receipt_retains_its_identity():
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == RECEIPT_SHA256
    receipt = _load(RECEIPT)
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"execution_mode": "eager", "image": IMAGE}
    assert receipt["compiled"] is False
    assert (receipt["versions"]["vllm"], receipt["versions"]["torch"]) == (VLLM, TORCH)
    assert receipt["device"]["platform_token"] == "sm_121"
    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert receipt["checkpoint_sidecars"]["config.json"] == config_sha
    assert _load(TWIN)["checkpoint_sidecars"] == receipt["checkpoint_sidecars"]
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    # The nightly's own GLM attention, not the NoPE plugin's CUSTOM backend
    # every f8dbe1a0 receipt recorded.
    assert receipt["env"]["TESSERA_RESEARCH_GLM53_NOPE"] == "0"
    assert receipt["engine_backends"] == {
        "kv_cache_dtype": "fp8_ds_mla", "moe_backend": "triton",
        "kernel_config": {"enable_flashinfer_autotune": False}, "trust_remote_code": True}
    log = SERVE_LOG.read_text(encoding="utf-8")
    assert "Using FLASHINFER_MLA_SPARSE_SM120 attention backend" in log
    assert f"(v{VLLM})" in log
    assert receipt["decoder_coverage"]["phases"]["decode"]["modules"] == 21
    cells = _nightly_cells(load_serving_contract())
    assert len(cells) == 4
    for cell in cells.values():
        assert cell["runtime"] == {"image": IMAGE, "execution_modes": ["eager"],
                                   "vllm": VLLM, "torch": TORCH,
                                   "kernel_build": f"legacy-toolchain/{VLLM}/{TORCH}"}, cell["id"]
        assert cell["id"].endswith(cell_runtime_id_suffix(cell)), cell["id"]
    assert (ROOT / MEASUREMENT).is_file()


def test_historical_routed_records_disagree_in_both_phases():
    """V70 refuses historical E4M3 routed launches on IMG-N."""
    tool = _tool()
    receipt = _load(RECEIPT)
    want = {}
    for phase, records in receipt["records"].items():
        for name, record in records.items():
            family = PAYLOAD_FAMILY_BY_ROUTE[record["policy"].partition(":")[0]]
            structure = "routed_moe" if record["kind"] == "moe" else "dense"
            counts = want.setdefault((structure, phase), {})
            counts[family] = counts.get(family, 0) + 1
    block, problems = _agreement(tool, load_serving_contract(), receipt)
    assert len(problems) == 8 and all("does not publish" in p for p in problems)
    assert block["agrees"] is False, json.dumps(block, indent=1)[:2000]
    nightly = _nightly_cells(load_serving_contract())
    for structure, per in block["structures"].items():
        for phase, row in per["phases"].items():
            counts = want[(structure, phase)]
            assert row["modules"] == sum(counts.values()), (structure, phase, row)
            assert row["covered_by_cell"] == counts.get("TESSERA_E4M3_K1", 0), (
                structure, phase, row)
            assert row["unattested"] == counts.get("TESSERA_BF16_K1", 0), (
                structure, phase, row)
            assert set(row["cells"]) <= set(nightly), row


def test_no_module_is_attested_on_the_nightly_without_the_minted_cells():
    """The fail-before, as a mutation of the packaged table: the f8dbe1a0 GLM
    E4M3 cells name the same scopes, launches and rungs, and still cover
    nothing here, because a cell is a receipt from ONE image."""
    tool = _tool()
    contract = load_serving_contract()
    nightly = _nightly_cells(contract)
    contract["lane_eligibility"]["cells"] = [
        c for c in contract["lane_eligibility"]["cells"] if c["id"] not in nightly]
    block, _problems = _agreement(tool, contract, _load(RECEIPT))
    for per in block["structures"].values():
        for row in per["phases"].values():
            assert row["covered_by_cell"] == 0
            assert row["unattested"] == row["modules"] > 0


def test_historical_twins_keep_their_launches_without_qualifying_img_n_classes():
    receipt, twin = _load(RECEIPT), _load(TWIN)
    for phase, records in receipt["records"].items():
        assert len(records) == 21, phase
        assert {name: (rec["symbol"], rec["decoder"]) for name, rec in records.items()} == {
            name: (rec["symbol"], rec["decoder"])
            for name, rec in twin["records"][phase].items()}, phase
    engagement = receipt["lane_engagement"]
    assert engagement["all_required_engaged"] is True
    assert engagement["required_lanes"] == [
        "tessera_routed_fused_mma_e4m3", "tessera_routed_fused_value"]
    contract = load_serving_contract()
    shipped = {c["id"]: c for c in contract["lane_eligibility"]["cells"]}
    for cell in _nightly_cells(contract).values():
        base = cell["id"][: -len(cell_runtime_id_suffix(cell))]
        entry = format_entry(cell["family"], contract)
        for rung in cell["rungs_q256"]:
            old = cell_executes(shipped[base], q256=rung, entry=entry)
            current = cell_executes(cell, q256=rung, entry=entry)
            if cell["structure"] == "routed_moe":
                assert current.isdisjoint(old), (cell["id"], rung)
            else:
                assert current == old, (cell["id"], rung)


def test_the_nightly_cells_cover_exactly_the_rungs_the_receipts_carried():
    tool = _tool()
    carried: dict = {}
    for receipt_path, config_path in ((RECEIPT, CONFIG),):
        receipt = _load(receipt_path)
        rungs = _declared_rungs(tool, receipt, config_path)
        for phase, records in receipt["records"].items():
            for name, record in records.items():
                owner = receipt["record_owner"][phase][name]
                family = PAYLOAD_FAMILY_BY_ROUTE[record["policy"].partition(":")[0]]
                structure = "routed_moe" if record["kind"] == "moe" else "dense"
                carried.setdefault((family, structure), set()).add(rungs[owner])
    assert sorted(k for k in carried if k[0] != "TESSERA_BF16_K1") == sorted(SCOPES)
    cells = _nightly_cells(load_serving_contract())
    assert sorted({(c["family"], c["structure"]) for c in cells.values()}) == sorted(SCOPES)
    # v68 re-mints BF16 scopes; the old folded claims remain withdrawn at v62.
    assert ("TESSERA_BF16_K1", "dense") in carried and ("TESSERA_BF16_K1", "routed_moe") in carried
    for cell in cells.values():
        assert set(cell["rungs_q256"]) == carried[(cell["family"], cell["structure"])], cell["id"]
        assert cell["evidence"]["grade"] == "route_only", cell["id"]
        assert cell["requires_serve_flags"] == ["TESSERA_SERVE_MODE=resident"], cell["id"]


@pytest.mark.parametrize("regime", ["decode", "batch"])
def test_the_release_rung_is_covered_in_every_scope(regime):
    """The GLM-5.3 release artifact carries q256 1024 in both E4M3 scopes."""
    for cell in _nightly_cells(load_serving_contract()).values():
        if cell["regime"] == regime:
            assert 1024 in cell["rungs_q256"], cell["id"]
