"""A runtime receipt does not attest the output of every later encoder (#198)."""
import copy
import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    cell_evidence, load_serving_contract, validate_serving_contract,
)


def affected(cell):
    return cell["family"] == "TESSERA_E4M3_K1" and cell["structure"] == "dense"


def artifact():
    return {
        "id": "gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook",
        "encoder_commit": "8070ec6c4e0448826cda3f3f8d9401a125444e3b",
        "reencode": {
            "encoder_commit": "b" * 40,
            "unit": "model.layers.0.mlp.down_proj",
            "payload": "different", "metric": "weight_sse",
            "weight_error": "lower",
            "receipt": "docs/measurements/encoder-evidence-scope-2026-09-05.md",
        },
    }


def test_historical_e4m3_cells_scope_their_encoder_evidence():
    doc = load_serving_contract()
    root = Path(__file__).resolve().parents[1]
    measurement = json.loads((root / "docs/measurements/encoder-evidence-scope-minor7-2026-09-05.json").read_text())
    for cell in doc["lane_eligibility"]["cells"]:
        scope = cell["evidence"]["artifact"]
        if affected(cell):
            assert scope["id"] == artifact()["id"]
            assert scope["encoder_commit"] == artifact()["encoder_commit"]
            assert scope["reencode"]["payload"] == "different"
            assert scope["reencode"]["metric"] == "weight_sse"
            assert scope["reencode"]["unit"] == "model.layers.0.mlp.down_proj"
            assert (root / scope["reencode"]["receipt"]).is_file()
            assert scope["reencode"]["encoder_commit"] == measurement["encoder_commit"]
            assert scope["reencode"]["unit"] == measurement["target"]
            assert measurement["same_payload"] is False
            old = measurement["arms"]["historical"]["sse"]
            new = measurement["arms"]["current"]["sse"]
            relation = "lower" if new < old else "equal" if new == old else "higher"
            assert scope["reencode"]["weight_error"] == relation
        else:
            assert scope is None  # No reproduction measurement was taken for these artifacts.
        assert cell_evidence(cell)["artifact"] == scope


def test_missing_artifact_scope_is_not_read_as_current_encoder_evidence():
    doc = load_serving_contract()
    doc["lane_eligibility"]["cells"][0]["evidence"].pop("artifact", None)
    with pytest.raises(ValueError, match="evidence is missing.*artifact"):
        validate_serving_contract(doc)


def test_weight_screen_never_promotes_the_served_evidence_grade():
    cell = copy.deepcopy(load_serving_contract()["lane_eligibility"]["cells"][0])
    cell["evidence"]["artifact"] = artifact()
    parsed = cell_evidence(cell)
    assert parsed["grade"] == "route_only"
    assert parsed["kl"] == []
    assert parsed["artifact"] == artifact()


@pytest.mark.parametrize("path,value,field", [
    (("id",), "/box/checkpoint", "id"),
    (("id",), "../checkpoint", "id"),
    (("encoder_commit",), "8070ec6", "encoder_commit"),
    (("reencode", "encoder_commit"), "master", "encoder_commit"),
    (("reencode", "unit"), "", "unit"),
    (("reencode", "payload"), "probably_same", "payload"),
    (("reencode", "metric"), "served_kl", "metric"),
    (("reencode", "weight_error"), "conservative_kl", "weight_error"),
    (("reencode", "receipt"), "/box/run.json", "receipt"),
    (("reencode", "receipt"), "docs/measurements/../escape.md", "receipt"),
    (("reencode", "payload"), "identical", "weight_error"),
])
def test_malformed_artifact_scope_is_refused_by_field(path, value, field):
    cell = copy.deepcopy(load_serving_contract()["lane_eligibility"]["cells"][0])
    scope = artifact()
    target = scope
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    cell["evidence"]["artifact"] = scope
    with pytest.raises(ValueError, match=rf"artifact\..*{field}"):
        cell_evidence(cell)
