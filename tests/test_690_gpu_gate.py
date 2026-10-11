"""Gate record for tessera#690 GPU work (tessera#1141).

The gate file must name the three approvals every GPU child obeys,
cite the runtime pin verbatim, keep the v6 cell-runtime leads, and
hold E2M1 parked. This test fails while the gate file is missing.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "docs/measurements/2026-10-09-690-gpu-gate.md"
CONTRACT = ROOT / "src/tessera/serving/runtime_contract.json"
E2M1_SOURCE = ROOT / "src/tessera/routed_fused_e2m1.py"


def _gate_text():
    assert GATE.is_file(), f"gate file missing: {GATE}"
    return GATE.read_text(encoding="utf-8")


def test_gate_names_qualification_adoption_and_tuple():
    text = _gate_text()
    assert "qualification_action" in text
    assert "adoption_note" in text
    assert "exact_tuple" in text


def test_gate_status_closed():
    text = _gate_text()
    assert "Status: CLOSED" in text


def test_gate_cites_runtime_pin_verbatim():
    text = _gate_text()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["contract_version"] == 66
    pin = contract["versions"]["default_serve_image"]
    assert pin.startswith("vllm/vllm-openai@sha256:")
    assert pin in text
    assert str(contract["contract_version"]) in text


def test_gate_keeps_all_routed_cell_images():
    text = _gate_text()
    for lead in ("tessera.lane-eligibility.v12", "execution_modes", "kernel_build"):
        assert lead in text
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    images = {
        c["runtime"]["image"]
        for c in contract["lane_eligibility"]["cells"]
        if c.get("structure") == "routed_moe"
    }
    assert len(images) >= 2
    for image in images:
        assert image in text


def test_e2m1_source_stays_parked():
    assert E2M1_SOURCE.is_file(), "E2M1 source must stay in the tree, parked"
    text = _gate_text()
    assert "parked" in text
    assert "routed_fused_e2m1" in text


def test_gate_names_filed_parked_e2m1_issue():
    text = _gate_text()
    assert "tessera#1149" in text
    assert "2026-09-28" in text
