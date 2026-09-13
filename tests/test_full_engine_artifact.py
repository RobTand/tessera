"""CPU regressions for artifact-mode observation inputs; no engine or GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from experiments.capture_full_engine_resources import read_calibration_prompt
from experiments.full_engine_artifact import read_tessera_artifact, ASSIGNMENT_SCHEMA
from experiments.full_engine_resource_partition import (
    CHECKPOINT_CENSUS_ARTIFACT, SUPPORTED_EXECUTION, assemble_full_engine_resource_report)
from experiments.full_engine_resources import analyze_engine_resource_ledger


def _artifact(tmp_path, *, quant_method="tessera", roles=None):
    roles = roles or {"model.layers.0.self_attn.qkv_proj": ["q_proj", "k_proj", "v_proj"],
                      "model.layers.0.mlp.down_proj": ["down_proj"]}
    (tmp_path / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3ForCausalLM"],
         "quantization_config": {"quant_method": quant_method, "ignore": ["lm_head"]}}))
    modules = {}
    for module, names in roles.items():
        prefix = module.rsplit(".", 1)[0]
        modules[module] = {"family": "TESSERA_FP8",
                           "roles": [{"tensor": f"{prefix}.{name}.weight", "role": name} for name in names]}
    (tmp_path / "tessera_serving_manifest.json").write_text(json.dumps({"modules": modules, "totals": {"units": 4}}))
    (tmp_path / "model.safetensors").write_bytes(b"not a real checkpoint")
    return tmp_path


def test_artifact_roster_and_assignment_come_from_the_manifest(tmp_path):
    source, roster, assignment = read_tessera_artifact(_artifact(tmp_path))
    assert [(row["unit_id"], row["module"], row["family"]) for row in roster] == [
        ("g:model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.qkv_proj", "TESSERA_FP8"),
        ("l:model.layers.0.mlp.down_proj", "model.layers.0.mlp.down_proj", "TESSERA_FP8")]
    assert roster[0]["members"] == ["model.layers.0.self_attn.q_proj.weight",
                                    "model.layers.0.self_attn.k_proj.weight",
                                    "model.layers.0.self_attn.v_proj.weight"]
    assert assignment["schema"] == ASSIGNMENT_SCHEMA
    assert assignment["units"] == {row["unit_id"]: "TESSERA_FP8" for row in roster}
    # The identity is the bytes the engine loads, never the directory name.
    assert set(source["files"]) == {"config.json", "tessera_serving_manifest.json", "model.safetensors"}
    assert source["files"]["model.safetensors"] == hashlib.sha256(b"not a real checkpoint").hexdigest()
    assert "artifact" not in source and source["ignore"] == ["lm_head"]


def test_artifact_observation_refuses_a_checkpoint_that_is_not_tessera(tmp_path):
    with pytest.raises(ValueError, match="quantization_config names tessera"):
        read_tessera_artifact(_artifact(tmp_path, quant_method="compressed-tensors"))


def test_artifact_observation_refuses_a_role_claimed_by_two_modules(tmp_path):
    roles = {"model.layers.0.mlp.down_proj": ["down_proj"], "model.layers.0.mlp.other": ["down_proj"]}
    with pytest.raises(ValueError, match="empty or duplicated"):
        read_tessera_artifact(_artifact(tmp_path, roles=roles))


def _fixture(tmp_path, ids):
    path = tmp_path / "calibration.safetensors"
    save_file({"calibration_ids": ids}, str(path))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_reads_row_zero_of_any_row_count_and_records_the_shape(tmp_path):
    ids = np.arange(3 * 512, dtype=np.int64).reshape(3, 512)
    path, sha = _fixture(tmp_path, ids)
    workload = read_calibration_prompt(path, sha)
    assert workload["prompt_token_ids"] == list(range(512))
    assert workload["calibration"] == {"path": str(path.resolve()), "sha256": sha,
                                       "key": "calibration_ids", "row": 0, "rows": 3}


@pytest.mark.parametrize("ids", [np.zeros((0, 512), dtype=np.int64), np.zeros((2, 511), dtype=np.int64),
                                 np.zeros(512, dtype=np.int64), np.zeros((2, 512), dtype=np.int32)])
def test_calibration_refuses_a_fixture_that_is_not_int64_rows_of_512(tmp_path, ids):
    path, sha = _fixture(tmp_path, ids)
    with pytest.raises(ValueError, match="int64"):
        read_calibration_prompt(path, sha)


def test_calibration_refuses_bytes_that_differ_from_the_declared_digest(tmp_path):
    path, _ = _fixture(tmp_path, np.zeros((1, 512), dtype=np.int64))
    with pytest.raises(ValueError, match="differ"):
        read_calibration_prompt(path, "0" * 64)


@pytest.fixture
def capture():
    return json.loads((Path(__file__).parent / "fixtures/full_engine_resource_ledger.json").read_text())


def _host_owner(pinned):
    owner = {"address": 8192, "bytes": 64, "category": "shared", "device_id": None, "device_type": "cpu",
             "dtype": "torch.int32", "owner_id": "runner:req_states.prompt_len.cpu",
             "provenance": "synthetic host owner", "shape": [16], "storage_offset_bytes": 0,
             "stride": [1], "view_extent_bytes": 64}
    if pinned is not None:
        owner["pinned"] = pinned
    return owner


@pytest.mark.parametrize("pinned,expected", [(False, "scoped_out"), (True, "issue"), (None, "issue")])
def test_a_pageable_host_owner_is_scoped_out_and_a_pinned_or_unstated_one_is_a_join_gap(capture, pinned, expected):
    raw = copy.deepcopy(capture)
    raw["checkpoints"][0]["owners"].append(_host_owner(pinned))
    result = analyze_engine_resource_ledger(raw)
    first = result["checkpoints"][0]
    gaps = [issue for issue in result["issues"] if "runner:req_states.prompt_len.cpu" in issue]
    # owner_count is the count over live device storages, never the host rows.
    assert first["owner_count"] == 2
    if expected == "scoped_out":
        assert not gaps and not first["unmatched_storage_observations"]
        assert [row["owner_id"] for row in first["pageable_host_observations"]] == ["runner:req_states.prompt_len.cpu"]
    else:
        assert gaps == ["owner runner:req_states.prompt_len.cpu: no matching live Torch or pinned-host backing allocation"]
        assert first["pageable_host_observations"] == []
        assert [row["owner_id"] for row in first["unmatched_storage_observations"]] == ["runner:req_states.prompt_len.cpu"]


def test_the_report_carries_the_checkpoint_census_as_an_artifact_not_a_checkpoint_field(capture):
    raw = copy.deepcopy(capture)
    raw["checkpoints"][0]["owners"].append(_host_owner(False))
    ledger = analyze_engine_resource_ledger(raw)
    members = {"reference": {"canonical_census": "synthetic", "runtime_binding": "synthetic",
                             "selected_rows": ["synthetic"]},
               "workload": {"calibration": "synthetic", "prompt_ids": ["synthetic"], "sampling": "synthetic"},
               "execution": dict(SUPPORTED_EXECUTION)}
    report = assemble_full_engine_resource_report(ledger, **members)
    frozen = {"label", "owner_count", "pinned_host_storages", "storages", "trace_index",
              "unique_owned_storage_bytes", "unique_pinned_host_backing_bytes", "unmatched_storage_observations"}
    assert all(set(row) == frozen for row in report["observations"]["checkpoints"])
    carried = [item for item in report["observations"]["artifacts"] if item.get("schema") == CHECKPOINT_CENSUS_ARTIFACT]
    assert len(carried) == 1
    assert carried[0]["checkpoints"][0] == {
        "label": "startup", "trace_index": 3, "census_owner_count": 2, "census_storage_count": 1,
        "pageable_host": {"count": 1, "bytes": 64, "owner_ids": ["runner:req_states.prompt_len.cpu"]}}


def test_a_checkpoint_states_every_live_storage_the_capture_ever_bound_to_an_owner(capture):
    # An owner bound at a later checkpoint owns the same storage while it is
    # live at an earlier one; the consumer recomputes checkpoints that way.
    ledger = analyze_engine_resource_ledger(capture)
    rows = {row["allocation_id"]: row for row in ledger["torch_allocations"]}
    for checkpoint in ledger["checkpoints"]:
        index = checkpoint["trace_index"]
        live = sorted(identity for identity, row in rows.items()
                      if row["allocate_index"] <= index and row["observed_owners"]
                      and (row["free_completed_index"] is None or index < row["free_completed_index"]))
        assert [entry["allocation_id"] for entry in checkpoint["storages"]] == live
        assert checkpoint["owner_count"] == sum(len(rows[identity]["observed_owners"]) for identity in live)
        assert checkpoint["unique_owned_storage_bytes"] == sum(rows[identity]["bytes"] for identity in live)
        assert checkpoint["census_owner_count"] >= 0 and checkpoint["census_storage_count"] >= 0
