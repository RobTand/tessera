"""Synthetic CPU ledger regressions; these fixtures are not engine/GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def capture():
    return json.loads((Path(__file__).parent / "fixtures/full_engine_resource_ledger.json").read_text())


def test_storage_aliases_are_counted_once_without_fixed_resource_admission(capture):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    result = analyze_engine_resource_ledger(capture)
    assert result["status"] == "observed_raw_ledger", result
    first = result["checkpoints"][0]
    assert first["unique_owned_storage_bytes"] == 512
    assert first["owner_count"] == 2
    assert first["storages"][0]["owners"] == ["embedding.weight", "lm_head.weight"]
    assert result["fixed_resources"] is None
    assert result["timings"] is None
    assert result["full_model_fixed_resources_complete"] is False
    assert result["admission"] == "not_implemented"


def test_pointer_reuse_is_a_new_lifetime_and_escaped_output_is_preserved(capture):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    result = analyze_engine_resource_ledger(capture)
    rows = [r for r in result["torch_allocations"] if r["address"] == 4608]
    assert [r["generation"] for r in rows] == [1, 2]
    assert rows[0]["lifetime_scope"] == "inside_unit"
    assert rows[1]["lifetime_scope"] == "escapes_unit"
    assert rows[1]["allocation_id"] in result["escaping_allocation_ids"]
    assert result["torch_observed_live_peak_bytes"] == 1536
    assert result["fixed_resources"] is None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def returned_snapshot_capture(capture):
    """Synthetic ownership, actual stock Torch's observed snapshot convention."""
    from experiments.full_engine_resources import history_revision
    history = capture["torch_snapshot"]["device_traces"][0]
    history.pop()  # A snapshot records its marker after returning its history.
    previous = None
    for number, checkpoint in enumerate(capture["checkpoints"], 1):
        checkpoint["trace_index"] -= 1
        checkpoint["history_boundary"] = "before_current_snapshot_marker"
        version = copy.deepcopy(history[:checkpoint["trace_index"]])
        version[0]["time_us"] = 100 * number
        checkpoint["history_prefix_sha256"] = _digest(version)
        if previous is not None:
            checkpoint["previous_history_revision"] = history_revision(previous, version)
        previous = version
    history[0]["time_us"] = 300
    return capture


def test_returned_snapshot_boundaries_reconcile_exact_timestamp_revisions(returned_snapshot_capture):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    result = analyze_engine_resource_ledger(returned_snapshot_capture)
    assert result["status"] == "observed_raw_ledger", result["issues"]
    assert result["history_join"]["revised_time_fields"] == 2
    assert result["history_join"]["timing_eligible"] is False
    assert len(result["checkpoints"]) == 3
    assert result["fixed_resources"] is None and result["timings"] is None


@pytest.mark.parametrize("defect", ["missing_revision", "wrong_after", "wrong_before",
    "duplicate_revision", "missing_marker", "ownership_field", "missing_rows", "unknown_boundary"])
def test_revision_join_never_accepts_missing_or_ownership_changes(returned_snapshot_capture, defect):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    raw = returned_snapshot_capture
    checkpoint = raw["checkpoints"][1]
    revision = checkpoint["previous_history_revision"]
    if defect == "missing_revision":
        checkpoint.pop("previous_history_revision")
    elif defect == "wrong_after":
        revision["changes"][0]["fields"]["time_us"]["after"] += 1
    elif defect == "wrong_before":
        revision["changes"][0]["fields"]["time_us"]["before"] += 1
    elif defect == "duplicate_revision":
        revision["changes"].append(copy.deepcopy(revision["changes"][0]))
    elif defect == "missing_marker":
        raw["torch_snapshot"]["device_traces"][0][checkpoint["trace_index"]]["action"] = "alloc"
    elif defect == "ownership_field":
        revision["changes"][0]["fields"]["addr"] = {
            "before_present": True, "before": 8192, "after_present": True, "after": 4096}
    elif defect == "missing_rows":
        revision["missing_previous_rows"] = [{}]
    elif defect == "unknown_boundary":
        checkpoint["history_boundary"] = "guessed"
    result = analyze_engine_resource_ledger(raw)
    assert result["status"] == "incomplete", result
    assert result["issues"] and result["fixed_resources"] is None


def test_requested_storage_and_rounded_allocator_block_are_distinct(capture):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    history = capture["torch_snapshot"]["device_traces"][0]
    history[1]["size"] = 32
    for checkpoint in capture["checkpoints"]:
        checkpoint["history_prefix_sha256"] = _digest(history[:checkpoint["trace_index"]])
        checkpoint["segments"][0]["blocks"][0]["requested_size"] = 32
        for owner in checkpoint["owners"]:
            if owner["address"] == 4096:
                owner.update(bytes=32, shape=[32], view_extent_bytes=32)
    capture["torch_snapshot"]["segments"][0]["blocks"][0]["requested_size"] = 32
    result = analyze_engine_resource_ledger(capture)
    assert result["status"] == "observed_raw_ledger", result["issues"]
    assert result["checkpoints"][0]["unique_owned_storage_bytes"] == 32
    assert result["torch_allocations"][0]["allocator_block_bytes_observed"] == [512]
    assert result["torch_observed_live_peak_scope"] == "requested_allocation_bytes_excluding_allocator_rounding"


@pytest.mark.parametrize("defect", ["missing_history", "history_not_early", "history_full",
                                    "missing_free_pair", "conflicting_alias", "owner_outside_allocation",
                                    "checkpoint_prefix_changed", "unmatched_free", "unknown_external",
                                    "dropped_cupti", "missing_cupti", "unknown_action", "unknown_owner",
                                    "unclosed_unit", "crossed_units", "unknown_persistent_block",
                                    "api_without_memory", "missing_api", "final_snapshot_changed",
                                    "unbound_checkpoint_marker"])
def test_incomplete_capture_stays_unknown(capture, defect):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    data = copy.deepcopy(capture)
    history = data["torch_snapshot"]["device_traces"][0]
    if defect == "missing_history":
        data.pop("torch_snapshot")
    elif defect == "history_not_early":
        data["capture"]["history_started_before_cuda_initialization"] = False
    elif defect == "history_full":
        data["capture"]["max_history_entries"] = len(history)
    elif defect == "missing_free_pair":
        history[4:6] = []
    elif defect == "conflicting_alias":
        data["checkpoints"][0]["owners"][1]["category"] = "candidate"
    elif defect == "owner_outside_allocation":
        data["checkpoints"][0]["owners"][0]["address"] += 1
    elif defect == "checkpoint_prefix_changed":
        data["checkpoints"][0]["history_prefix_sha256"] = "f" * 64
    elif defect == "unmatched_free":
        history[5]["addr"] += 16
    elif defect == "unknown_external":
        row = copy.deepcopy(data["cupti_trace"]["memory_events"][0])
        row.update(address=99999, bytes=64, correlation_id=900, timestamp_ns=150)
        data["cupti_trace"]["memory_events"].append(row)
    elif defect == "dropped_cupti":
        data["cupti_trace"]["dropped_records"][0]["count"] = 1
    elif defect == "missing_cupti":
        data.pop("cupti_trace")
    elif defect == "unknown_action":
        history[0]["action"] = "unmapped_pool_import"
    elif defect == "unknown_owner":
        data["checkpoints"][0]["owners"][0]["category"] = "unknown"
    elif defect == "unclosed_unit":
        data["unit_intervals"][0]["end_checkpoint"] = None
    elif defect == "crossed_units":
        data["unit_intervals"].append({"invocation_id": "other", "unit_id": "other", "begin_checkpoint": "unit_end", "end_checkpoint": "capture_end"})
        data["unit_intervals"][0]["end_checkpoint"] = "capture_end"
        data["unit_intervals"][0]["begin_checkpoint"] = "startup"
        data["unit_intervals"][1]["end_checkpoint"] = "startup"
    elif defect == "unknown_persistent_block":
        for checkpoint in data["checkpoints"]:
            checkpoint["owners"] = [o for o in checkpoint["owners"] if o["address"] != 4096]
    elif defect == "api_without_memory":
        api = copy.deepcopy(data["cupti_trace"]["api_events"][0])
        api.update(correlation_id=900, start_ns=120, end_ns=130)
        data["cupti_trace"]["api_events"].append(api)
    elif defect == "missing_api":
        data["cupti_trace"]["api_events"] = []
    elif defect == "final_snapshot_changed":
        data["torch_snapshot"]["segments"] = []
    elif defect == "unbound_checkpoint_marker":
        data["checkpoints"][0]["cupti_timestamp_ns"] += 1
    result = analyze_engine_resource_ledger(data)
    assert result["status"] == "incomplete", result
    assert result["issues"]
    assert result["fixed_resources"] is None
    assert result["full_model_fixed_resources_complete"] is False


def test_raw_whole_engine_peak_and_supplied_times_never_become_fixed_prices(capture):
    from experiments.full_engine_resources import analyze_engine_resource_ledger
    capture["whole_engine_peak_bytes"] = 90000000
    capture["prefill_ms"] = 23.0
    capture["operator_medians_ms"] = [2.0, 3.0]
    result = analyze_engine_resource_ledger(capture)
    assert result["fixed_resources"] is None
    assert result["timings"] is None


def test_recorder_validates_identity_before_starting_a_collector(monkeypatch):
    from experiments import full_engine_resources as module
    monkeypatch.setattr(module, "NativeMemoryCollector", lambda _: pytest.fail("collector started"))
    with pytest.raises(ValueError, match="identity"):
        module.FullEngineResourceRecorder("unused.so", {}, max_checkpoints=1)


def test_history_revision_preserves_changes_without_guessing_their_meaning():
    from experiments.full_engine_resources import history_revision
    before = [{"action": "alloc", "addr": 1024, "time_us": 100, "frames": [{"name": "old"}]},
              {"action": "snapshot", "size": 16}]
    after = [{"action": "alloc", "addr": 1024, "time_us": 101, "frames": [{"name": "new"}]},
             {"action": "snapshot", "size": 16}, {"action": "alloc", "addr": 2048}]
    result = history_revision(before, after)
    assert result["previous_length"] == 2 and result["current_length"] == 3
    assert result["missing_previous_rows"] == []
    assert result["changes"] == [{"index": 0, "fields": {
        "frames": {"before_present": True, "before": [{"name": "old"}],
                   "after_present": True, "after": [{"name": "new"}]},
        "time_us": {"before_present": True, "before": 100,
                    "after_present": True, "after": 101}}}]
    assert history_revision(after, before)["missing_previous_rows"] == [after[-1]]


def test_recorder_publishes_hash_bound_raw_inputs_and_keeps_unknowns(capture, monkeypatch, tmp_path):
    from experiments import full_engine_resources as module
    calls = []
    snapshots = []
    for checkpoint in capture["checkpoints"]:
        snapshot = copy.deepcopy(capture["torch_snapshot"])
        snapshot["device_traces"][0] = snapshot["device_traces"][0][:checkpoint["trace_index"]]
        snapshot["segments"] = checkpoint["segments"]
        snapshots.append(snapshot)
    last = snapshots[-1]
    class Collector:
        start_code = 0
        def __init__(self, library):
            calls.append("cupti_start")
        def mark(self, label):
            return 200 + len(calls)
        def current_context_id(self):
            return 1
        def finish(self, path):
            value = copy.deepcopy(capture["cupti_trace"])
            Path(path).write_text(json.dumps(value))
            calls.append("cupti_stop")
            return value
    def record_history(**kwargs):
        calls.append("history_start" if kwargs["enabled"] else "history_stop")
    cuda = SimpleNamespace(is_initialized=lambda: False, init=lambda: None,
                           synchronize=lambda device: None,
                           get_allocator_backend=lambda: "native",
                           memory=SimpleNamespace(_record_memory_history=record_history,
                                                  _snapshot=lambda: snapshots.pop(0) if snapshots else last))
    monkeypatch.setattr(module, "NativeMemoryCollector", Collector)
    monkeypatch.setattr(module, "_torch", lambda: SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(module.os, "getpid", lambda: capture["process_id"])
    recorder = module.FullEngineResourceRecorder("unused.so", capture["identity"], max_checkpoints=2)
    recorder.snapshot("startup")
    with pytest.raises(RuntimeError, match="checkpoint budget"):
        recorder.snapshot("unplanned")
    with pytest.raises(RuntimeError, match="checkpoint budget"):
        with recorder.unit_scope("unplanned-unit"):
            pytest.fail("unit entered without budget for both boundaries")
    receipt = recorder.finish(tmp_path / "capture")
    assert calls[:2] == ["cupti_start", "history_start"]
    assert "cupti_stop" in calls
    assert receipt["fixed_resources"] is None
    assert receipt["status"] == "incomplete"
    assert receipt["full_model_fixed_resources_complete"] is False
    for reference in receipt["artifacts"].values():
        path = tmp_path / "capture" / reference["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    raw = json.loads((tmp_path / "capture" / "capture.json").read_text())
    assert raw["identity"] == capture["identity"]
    assert raw["torch_snapshot"]["device_traces"]
    assert raw["capture"]["max_checkpoints"] == 2
    costs = raw["observer_cost"]["snapshot_attempts"]
    assert len(costs) == 2
    assert all(c["host_observer_elapsed_ns"] > 0 and c["serialized_snapshot_bytes"] > 0 for c in costs)
    assert raw["observer_cost"]["gpu_timing_eligible"] is False
    with pytest.raises(RuntimeError, match="closed"):
        recorder.snapshot("late")
