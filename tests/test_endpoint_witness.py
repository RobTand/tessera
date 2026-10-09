"""Test the complete byte contract; fixtures do not qualify a native serving window."""
from __future__ import annotations

import copy

import pytest

from tessera import endpoint_witness as ew
from _endpoint_fixture import receipt, resign


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "artifact"
    return root, receipt(root)


def verify(root, value, live=None):
    return ew.verify_witness(value, served_dir=root, live=value if live is None else live)


def test_a_complete_join_needs_both_bytes_and_a_live_listener(sample):
    root, value = sample
    assert verify(root, value) is None
    assert "live runtime binding" in ew.verify_witness(value, served_dir=root)
    assert "artifact directory" in ew.verify_witness(value, live=value)
    assert "expected ranks" in ew.verify_witness(value, served_dir=root, live=value, ranks=[0])


@pytest.mark.parametrize("key", ["listener", "launch", "lifetime", "artifacts", "tokenizer",
                                  "byte_coverage", "fingerprint", "qualification_scope"])
def test_absent_join_fields_are_refused(sample, key):
    root, value = sample
    del value[key]
    assert verify(root, value) is not None


@pytest.mark.parametrize("bad", [None, [], 1, {}, {"schema": ew.SCHEMA}])
def test_malformed_witnesses_return_a_refusal(sample, bad):
    root, _ = sample
    assert ew.verify_witness(bad, served_dir=root) is not None


def test_schema_and_qualification_cannot_change(sample):
    root, value = sample
    bad = copy.deepcopy(value)
    bad["schema"] = "unknown"
    assert "schema" in verify(root, resign(bad))
    bad = copy.deepcopy(value)
    bad["qualification_scope"] = "Serving qualified"
    assert "qualification" in verify(root, resign(bad))


def test_an_edited_body_is_refused_by_its_byte_fingerprint(sample):
    root, value = sample
    value["listener"]["served_alias"] = "other"
    assert "fingerprint" in verify(root, value)


@pytest.mark.parametrize("ranks", [[], [0], [1, 0], [0, 0], [0, 2], [True, 1]])
def test_launch_ranks_must_cover_the_complete_world(sample, ranks):
    root, value = sample
    value["launch"]["ranks"] = ranks
    assert "ranks" in verify(root, resign(value))


def test_missing_duplicate_and_inconsistent_worker_ranks_are_refused(sample):
    root, value = sample
    bad = copy.deepcopy(value)
    bad["artifacts"].pop()
    assert "incomplete" in verify(root, resign(bad))
    bad = copy.deepcopy(value)
    bad["artifacts"][1]["rank"] = 0
    assert "ranks" in verify(root, resign(bad))
    bad = copy.deepcopy(value)
    bad["artifacts"][1]["world_size"] = 3
    assert "world" in verify(root, resign(bad))
    bad = copy.deepcopy(value)
    bad["artifacts"][1]["owner"] = bad["artifacts"][0]["owner"]
    assert "same worker" in verify(root, resign(bad))


@pytest.mark.parametrize("part", ["rank", "tokenizer"])
def test_mixed_attempt_observations_are_refused(sample, part):
    root, value = sample
    item = value["artifacts"][0] if part == "rank" else value["tokenizer"]
    item["request_id"] = "another-runtime-request"
    assert "mixes runtime" in verify(root, resign(value))


@pytest.mark.parametrize("stamp", [0, -1, float("inf"), float("nan"), True, "500", 399, 601])
def test_observations_outside_the_runtime_lifetime_are_refused(sample, stamp):
    root, value = sample
    value["artifacts"][0]["observed_unix"] = stamp
    if isinstance(stamp, float) and not __import__("math").isfinite(stamp):
        assert verify(root, value) is not None
    else:
        assert verify(root, resign(value)) is not None


def test_loaded_byte_times_must_belong_to_the_worker_process(sample):
    root, value = sample
    value["artifacts"][0]["models"][0]["load_started_unix"] = 90
    assert "process lifetime" in verify(root, resign(value))


def test_loaded_input_times_must_belong_to_the_load(sample):
    root, value = sample
    value["artifacts"][0]["models"][0]["inputs"][0]["loaded_unix"] = 301
    assert "load lifetime" in verify(root, resign(value))


@pytest.mark.parametrize("change", ["missing_input", "gap", "source_digest", "source_file",
                                   "source_tensor", "bounds", "resident", "model"])
def test_incomplete_or_unrelated_loaded_byte_observations_are_refused(sample, change):
    root, value = sample
    for rank in value["artifacts"]:
        model = rank["models"][0]
        if change == "missing_input":
            model["inputs"] = []
        elif change == "gap":
            model["inputs"][0]["end"] = 3
        elif change == "source_digest":
            model["inputs"][0]["source_sha256"] = "f" * 64
        elif change == "source_file":
            model["inputs"][0]["file"] = "unrelated.safetensors"
        elif change == "source_tensor":
            model["inputs"][0]["tensor"] = "unrelated"
        elif change == "bounds":
            model["inputs"][0]["start"] = -1
        elif change == "resident":
            model["resident"] = {}
        else:
            rank["models"] = []
    assert verify(root, resign(value)) is not None


def test_rank_file_byte_disagreement_is_refused(sample):
    root, value = sample
    value["artifacts"][1]["models"][0]["files"]["model.safetensors"]["sha256"] = "f" * 64
    assert "disagree" in verify(root, resign(value))


@pytest.mark.parametrize("change", ["size", "shape", "dtype", "offsets", "path", "coverage"])
def test_false_source_coverage_is_refused(sample, change):
    root, value = sample
    for rank in value["artifacts"]:
        source = rank["models"][0]["files"]["model.safetensors"]
        if change == "size":
            source["bytes"] += 1
        elif change == "shape":
            source["tensors"]["weight"]["shape"] = [3]
        elif change == "dtype":
            source["tensors"]["weight"]["dtype"] = "F32"
        elif change == "offsets":
            source["tensors"]["weight"]["data_offsets"] = [1, 5]
        elif change == "path":
            rank["models"][0]["files"]["../other.safetensors"] = source
    if change == "coverage":
        value["byte_coverage"]["tensor_payload_bytes"] += 32
    assert verify(root, resign(value)) is not None


@pytest.mark.parametrize("change", ["mapping", "backend", "special_id", "missing_file", "missing_mapping", "client_metadata"])
def test_client_or_inconsistent_tokenizer_facts_are_refused(sample, change):
    root, value = sample
    tokenizer = value["tokenizer"]
    if change == "mapping":
        tokenizer["vocab"] = {"<s>": 1, "a": 0}
    elif change == "backend":
        tokenizer["backend"] = copy.deepcopy(tokenizer["backend"])
        tokenizer["backend"]["model"]["vocab"] = {"<s>": 1, "a": 0}
    elif change == "special_id":
        tokenizer["special_ids"]["bos"] = 1
    elif change == "missing_file":
        del tokenizer["files"]["tokenizer.json"]
    elif change == "missing_mapping":
        del tokenizer["vocab"]
    else:
        value["tokenizer"] = {"vocab_size": 2, "source": "client"}
    assert verify(root, resign(value)) is not None


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer.json", "tokenizer_config.json"])
def test_changed_or_missing_artifact_bytes_are_refused(sample, name):
    root, value = sample
    path = root / name
    raw = path.read_bytes()
    path.write_bytes(raw + b"x")
    assert "bytes differ" in verify(root, value)
    path.unlink()
    assert verify(root, value) is not None


def test_unrelated_complete_directory_cannot_replace_loaded_artifact_bytes(sample, tmp_path):
    root, value = sample
    other = tmp_path / "other"
    receipt(other)
    raw = (other / "model.safetensors").read_bytes()
    (other / "model.safetensors").write_bytes(raw[:-1] + b"\xff")
    assert "artifact file bytes differ" in ew.verify_witness(value, served_dir=other, live=value)


def test_new_colocated_unloaded_files_do_not_become_loaded_evidence(sample):
    root, value = sample
    (root / "unused.bin").write_bytes(b"not loaded")
    assert verify(root, value) is None
    assert value["byte_coverage"]["files"] == ["model.safetensors"]


@pytest.mark.parametrize("change", ["listener", "rank", "resident", "alias"])
def test_current_binding_refuses_replacement_processes_and_changed_loaded_state(sample, change):
    root, value = sample
    live = copy.deepcopy(value)
    if change == "listener":
        live["listener"]["owner"]["start_ticks"] += 1
        live["launch"]["attempt_id"] = ew.attempt_id(live["listener"]["owner"])
    elif change == "rank":
        live["artifacts"][0]["owner"]["start_ticks"] += 1
    elif change == "resident":
        live["artifacts"][0]["models"][0]["resident"]["parameter:weight"]["sha256"] = "f" * 64
    else:
        live["listener"]["served_alias"] = "replacement"
    assert "binding differs" in verify(root, value, resign(live))


def test_the_live_request_stamp_can_change_without_changing_runtime_state(sample):
    root, value = sample
    live = copy.deepcopy(value)
    live["lifetime"]["request_id"] = "fresh-request"
    live["lifetime"]["started_unix"] = 700.0
    live["lifetime"]["finished_unix"] = 900.0
    for rank in live["artifacts"]:
        rank["request_id"] = "fresh-request"
        rank["observed_unix"] = 800.0
    live["tokenizer"]["request_id"] = "fresh-request"
    live["tokenizer"]["observed_unix"] = 800.0
    assert verify(root, value, resign(live)) is None
