"""Research execution snapshots bind literal inputs without becoming wire identity."""
import hashlib
import json

import pytest

from tessera.moe_execution import ResearchSelectedMoeConfig, ResearchSelectedMoeInput


def _block():
    return {"schema": "tessera.research_selected_moe.v1", "max_experts_per_chunk": 2,
            "decode_backend": "torch", "expected_tensor_parallel_size": 1}


def test_snapshot_round_trip_binds_bytes_but_preserves_execution_values(tmp_path):
    raw = (json.dumps(_block(), indent=2) + "\n").encode()
    path = tmp_path / "execution.json"
    path.write_bytes(raw)
    snapshot = ResearchSelectedMoeInput.read(path)
    path.unlink()
    record = snapshot.record()
    assert record["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert ResearchSelectedMoeInput.from_record(record) == snapshot
    compact = ResearchSelectedMoeInput.from_bytes(json.dumps(_block()).encode())
    assert compact.config == snapshot.config
    assert compact.record()["input_sha256"] != record["input_sha256"]
    record["config"]["max_experts_per_chunk"] = 17
    assert snapshot.config.max_experts_per_chunk == 2
    with pytest.raises(ValueError, match="digest/content"):
        ResearchSelectedMoeInput.from_record(record)


@pytest.mark.parametrize("raw", [b"\xff", b"{", b"null", b"[]",
    b'{"schema":"first","schema":"second"}'])
def test_invalid_literal_input_refuses(raw):
    with pytest.raises(ValueError, match="research_selected_moe"):
        ResearchSelectedMoeInput.from_bytes(raw)


@pytest.mark.parametrize("field,value", [("input_sha256", "0" * 64),
    ("input_utf8", "{}"), ("config", {}), ("unexpected", True)])
def test_changed_input_record_refuses(field, value):
    record = ResearchSelectedMoeInput.from_bytes(json.dumps(_block()).encode()).record()
    record[field] = value
    with pytest.raises(ValueError, match="research_selected_moe"):
        ResearchSelectedMoeInput.from_record(record)


def test_shared_config_incompatible_targets():
    config = ResearchSelectedMoeConfig.from_checkpoint(_block())
    with pytest.raises(ValueError, match="requires TESSERA_FP8/E4M3"):
        config.require_targets({"m": {"structure": "routed_moe", "family": "TESSERA_BF16", "grid": "BF16"}}, "resident")


@pytest.mark.parametrize("field,value", [("expected_tensor_parallel_size", True),
    ("expected_tensor_parallel_size", 1.0), ("max_experts_per_chunk", 2.0)])
def test_carried_config_refuses_numeric_type_aliases(field, value):
    record = ResearchSelectedMoeInput.from_bytes(json.dumps(_block()).encode()).record()
    record["config"][field] = value
    with pytest.raises(ValueError, match="research_selected_moe"):
        ResearchSelectedMoeInput.from_record(record)
