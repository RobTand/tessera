"""Synthetic protocol fixtures; no GPU or whole-engine qualification claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from experiments.full_engine_timings import analyze_profile_partition
from experiments.full_engine_timing_observation import build_timing_observation
from experiments.full_engine_timing_world import POLICY_SCHEMA, SCHEMA
from experiments.capture_full_engine_resources import require_observed_decode_contract

BASE = datetime(2026, 9, 26, 18, 0, tzinfo=timezone.utc).timestamp()
DIGESTS = {name: character * 64 for name, character in zip(
    ("configuration_sha256", "model_sha256", "runtime_manifest_sha256",
     "workload_sha256", "assignment_sha256", "canonical_units_sha256"), "abcdef")}
SYNTHETIC_SOURCE = {"schema": "tessera.artifact_observer_source.v1",
                    "files": {"config.json": "1" * 64,
                              "tessera_serving_manifest.json": "2" * 64},
                    "manifest_totals": {"units": 1}, "ignore": []}
DIGESTS["model_sha256"] = hashlib.sha256(json.dumps(
    SYNTHETIC_SOURCE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_unmeasured_draft_forward_cannot_enter_the_target_only_observer():
    require_observed_decode_contract({"engine_args": {"speculative_config": None}})
    with pytest.raises(ValueError, match="draft owner/first-call regime"):
        require_observed_decode_contract({"engine_args": {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1}}})


def _events(rank, step_index, *, segmented):
    offset = 1000 * step_index
    step = f"step.{step_index}"
    events = [{"ph": "X", "cat": "user_annotation", "name": step,
               "pid": 10 + rank, "tid": 20, "ts": offset, "dur": 900}]
    for index, (name, at, gpu_name) in enumerate(((f"apply.{step_index}", 100, "native"),
                                                    (f"reduce.{step_index}", 400, "ncclAllReduce"))):
        if segmented:
            events.append({"ph": "X", "cat": "user_annotation", "name": name,
                           "pid": 10 + rank, "tid": 20, "ts": offset + at, "dur": 100})
        correlation = 10000 * rank + offset + index
        events.extend([{"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                        "pid": 10 + rank, "tid": 20, "ts": offset + at + 10, "dur": 1,
                        "args": {"correlation": correlation}},
                       {"ph": "X", "cat": "kernel", "name": gpu_name,
                        "ts": offset + at + 20, "dur": 5,
                        "args": {"correlation": correlation, "device": 0, "stream": 7}}])
    correlation = 10000 * rank + offset + 3
    events.extend([{"ph": "X", "cat": "cuda_runtime", "name": "cudaMemcpyAsync",
                    "pid": 10 + rank, "tid": 20, "ts": offset + 700, "dur": 1,
                    "args": {"correlation": correlation}},
                   {"ph": "X", "cat": "gpu_memcpy", "name": "output D2H",
                    "ts": offset + 710, "dur": 5,
                    "args": {"correlation": correlation, "device": 0, "stream": 8}}])
    return events


def _arm(rank, arm, pid):
    partition = arm == "partition"
    host = {"ip": f"192.168.1.{107 + rank}", "interface": "eth0",
            "source": "synthetic CPU fixture; no actual interface observed"}
    identity = dict(DIGESTS, rank=rank, world_size=2, device_id=0,
                    device_uuid=f"GPU-rank-{rank}", host=host)
    capture = {"schema": "tessera.full_engine_timing_capture.v2", "arm": arm,
               "identity": identity, "device_id": 0, "errors": [],
               "canonical_unit_ids": ["s:moe"], "ranges": {}, "steps": [],
               "observer_span": {"started_unix_ns": int(BASE * 1e9),
                                 "finished_unix_ns": int((BASE + 1) * 1e9)},
               "exclusivity": {"own_pid": pid,
                               "at_arm": {"available": True, "count": 1,
                                          "processes": [{"pid": pid}], "device_index": 0},
                               "at_finish": {"available": True, "count": 1,
                                             "processes": [{"pid": pid}], "device_index": 0}},
               "profiler_collection_health": {"available": True,
                   "streams": [{"stream_id": stream, "dropped_records": 0} for stream in (7, 8, 9)]}}
    events = []
    for step_index, (phase, count) in enumerate((("prefill", 512), ("decode", 1))):
        capture["ranges"][f"step.{step_index}"] = {"kind": "step", "step_id": str(step_index)}
        if partition:
            capture["ranges"][f"apply.{step_index}"] = {
                "kind": "owner_segment", "step_id": str(step_index), "unit_id": "s:moe", "role": "apply"}
            capture["ranges"][f"reduce.{step_index}"] = {
                "kind": "owner_segment", "step_id": str(step_index), "unit_id": "s:moe", "role": "final_reduce"}
        reduction = {"input_shape": [512, 2048], "input_dtype": "torch.bfloat16",
                     "output_shape": [512, 2048], "output_dtype": "torch.bfloat16",
                     "trunc_size": None, "output_is_reduced": False,
                     "effective_output_is_reduced": False, "tp_size": 2,
                     "skip_final_all_reduce": False, "is_sequence_parallel": False,
                     "collective_calls": [{"input_shape": [512, 2048],
                                           "input_dtype": "torch.bfloat16"}],
                     "collective_site": "vllm.model_executor.layers.fused_moe.runner.moe_runner.tensor_model_parallel_all_reduce"}
        segments = ([{"unit_id": "s:moe", "role": "apply", "range_name": f"apply.{step_index}",
                      "elapsed_ms": 2.0},
                     {"unit_id": "s:moe", "role": "final_reduce", "range_name": f"reduce.{step_index}",
                      "elapsed_ms": 1.0, "reduction": reduction}] if partition else [])
        capture["steps"].append({"step_id": str(step_index), "phase": phase,
                                 "scheduled_tokens": count, "main_stream_id": 7,
                                 "copy_stream_id": 8, "copy_event_observed": True,
                                 "completion_join": "main_event_and_stock_copy_event",
                                 "units": ([{"unit_id": "s:moe", "range_name": f"apply.{step_index}",
                                            "elapsed_ms": 3.0}] if partition else []),
                                 "segments": segments,
                                 "fixed_gaps_ms": [1.0, 0.5, 1.0] if partition else [4.0],
                                 "whole_step_ms": 5.5 if partition else 4.0})
        events.extend(_events(rank, step_index, segmented=partition))
    profile = {"traceEvents": events}
    return capture, profile


def _write_world(tmp_path, *, mutate=None, policy=True):
    capture_dir, launch_dir = tmp_path / "capture", tmp_path / "launch"
    capture_dir.mkdir()
    launch_dir.mkdir()
    plan = {"identity": DIGESTS, "canonical_source": SYNTHETIC_SOURCE,
            "world_size": 2, "timing_samples": 3,
            "selected_configuration": {"engine_args": {"tensor_parallel_size": 2,
                                                        "speculative_config": None},
                                       "environment": {"TESSERA_SERVE_MODE": "resident"}}}
    if policy:
        path = tmp_path / "policy.json"
        path.write_text(json.dumps({"schema": POLICY_SCHEMA, "max_abs_relative_difference": 1.0}))
        plan["observer_impact_policy"] = {"path": str(path),
                                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    run = {"schema": "tessera.stock_engine_raw_timing_run.v1", "plan_sha256": "9" * 64,
           "fixture_provenance": "synthetic_cpu_protocol_fixture",
           "warmup_requests": 1, "warmup_tokens": [7, 8], "arms": []}
    rows = {}
    for sample in range(3):
        for arm in ("control", "partition"):
            entry = {"arm": arm, "sample": sample, "tokens": [7, 8], "armed": [], "workers": []}
            for rank in (0, 1):
                pid = 4242 + rank
                directory = capture_dir / f"rank-{rank}-{sample}-{arm}"
                capture, profile = _arm(rank, arm, pid)
                rows[(arm, sample, rank)] = {"directory": directory, "capture": capture,
                                              "profile": profile, "entry": entry}
                entry["armed"].append({"pid": pid, "rank": rank, "world_size": 2,
                                       "device_id": 0, "device_uuid": f"GPU-rank-{rank}",
                                       "host": capture["identity"]["host"]})
                entry["workers"].append({"pid": pid, "rank": rank, "world_size": 2,
                                         "device_id": 0, "device_uuid": f"GPU-rank-{rank}",
                                         "host": capture["identity"]["host"],
                                         "directory": str(directory)})
            run["arms"].append(entry)
    if mutate:
        mutate(plan, run, rows)
    (capture_dir / "observer-plan.json").write_text(json.dumps(plan))
    run["plan_sha256"] = hashlib.sha256((capture_dir / "observer-plan.json").read_bytes()).hexdigest()
    (capture_dir / "run.json").write_text(json.dumps(run))
    for row in rows.values():
        directory = row["directory"]
        directory.mkdir()
        profile_path = directory / "profile.json"
        profile_path.write_text(json.dumps(row["profile"]))
        row["capture"]["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        (directory / "capture.json").write_text(json.dumps(row["capture"]))
        partition = analyze_profile_partition(row["capture"], row["profile"])
        (directory / "partition.json").write_text(json.dumps(partition))
    stamp = datetime.fromtimestamp(BASE, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for rank in (0, 1):
        directory = launch_dir / f"rank-{rank}"
        directory.mkdir()
        (directory / "host-vitals.log").write_text(
            f"{stamp} host_ip=192.168.1.{107 + rank} gpu_uuid=GPU-rank-{rank} "
            f"compute_apps={4242 + rank}\n")
    return capture_dir, launch_dir


def test_tp2_world_has_two_raw_rank_vectors_and_direct_gap_aggregation(tmp_path):
    capture_dir, launch_dir = _write_world(tmp_path)
    observed = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert observed["schema"] == SCHEMA
    assert observed["fixture_provenance"] == "synthetic_cpu_protocol_fixture"
    assert observed["partition"]["established"] is False
    assert "profiler_collection_health" in observed["partition"]["reason"]
    assert [rank["device_uuid"] for rank in observed["ranks"]] == ["GPU-rank-0", "GPU-rank-1"]
    assert observed["partition"]["terms"] is None
    assert {str(rank["rank"]): rank["raw_terms"]["prefill"]["fixed_gap_sum_ms"]
            for rank in observed["ranks"]} == {"0": [2.5] * 3, "1": [2.5] * 3}
    assert all(len(rank["arms"]) == 6 for rank in observed["ranks"])
    assert observed["raw_inputs"]["plan"]["sha256"] == observed["plan_sha256"]
    assert all(rank["host_vitals"]["sha256"] for rank in observed["ranks"])


@pytest.mark.parametrize("mutate,failed", [
    (lambda _p, run, _r: run["arms"][0]["workers"].pop(), "complete_world"),
    (lambda _p, run, _r: run["arms"][0].update(tokens=[7, 99]), "identical_tokens"),
    (lambda _p, _run, rows: rows[("partition", 0, 1)]["capture"]["identity"].update(world_size=1),
     "rank1.raw_evidence"),
    (lambda _p, _run, rows: next(event for event in rows[("partition", 0, 1)]["profile"]["traceEvents"]
                                 if event.get("name") == "native")["args"].update(stream=99),
     "rank1.stream_coverage"),
    (lambda _p, _run, rows: rows[("partition", 0, 1)]["capture"]["steps"][0]["segments"].pop(),
     "rank1.stream_coverage"),
    (lambda _p, _run, rows: rows[("partition", 0, 1)]["capture"]["steps"][0]["segments"][1]["reduction"]["collective_calls"].append(
        {"input_shape": [512, 2048], "input_dtype": "torch.bfloat16"}), "rank1.stream_coverage"),
])
def test_tp2_refuses_incomplete_or_mismatched_world(tmp_path, mutate, failed):
    capture_dir, launch_dir = _write_world(tmp_path, mutate=mutate)
    observed = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert observed["partition"]["established"] is False
    assert failed in observed["partition"]["reason"]


def test_tp2_refuses_missing_raw_profile_and_absent_policy(tmp_path):
    capture_dir, launch_dir = _write_world(tmp_path, policy=False)
    (capture_dir / "rank-1-0-partition" / "profile.json").unlink()
    observed = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert observed["partition"]["established"] is False
    assert "rank1.raw_evidence" in observed["partition"]["reason"]
    assert "rank0.observer_impact" in observed["partition"]["reason"]


def test_reset_on_read_zero_is_not_cumulative_collection_health(tmp_path):
    def mutate(_plan, _run, rows):
        # A previous Kineto buffer callback may already have consumed one
        # dropped-record count. Its post-stop read is then zero despite loss.
        rows[("partition", 0, 1)]["capture"]["profiler_collection_health"] = {
            "available": True, "counter_semantics": "reset_on_read",
            "consumption_owner": "Kineto callback, not observed here",
            "streams": [{"stream_id": stream, "dropped_records": 0} for stream in (7, 8, 9)]}

    capture_dir, launch_dir = _write_world(tmp_path, mutate=mutate)
    observed = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert observed["partition"]["established"] is False
    assert "rank1.profiler_collection_health" in observed["partition"]["reason"]


def test_tp2_raw_run_plan_and_host_logs_are_content_bound(tmp_path):
    capture_dir, launch_dir = _write_world(tmp_path)
    run_path = capture_dir / "run.json"
    run = json.loads(run_path.read_text())
    run["plan_sha256"] = "0" * 64
    run_path.write_text(json.dumps(run))
    (launch_dir / "rank-1" / "host-vitals.log").unlink()
    observed = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert observed["partition"]["established"] is False
    assert "plan_binding" in observed["partition"]["reason"]
    assert "rank1.device_exclusivity" in observed["partition"]["reason"]
    assert observed["ranks"][1]["host_vitals"]["sha256"] is None


def test_tp2_reduction_is_separate_and_double_charge_is_refused():
    capture, profile = _arm(0, "partition", 4242)
    assert analyze_profile_partition(capture, profile)["status"] == "observed_same_run_partition"
    capture["steps"][0]["units"][0]["elapsed_ms"] = 4.0
    result = analyze_profile_partition(capture, profile)
    assert result["status"] == "incomplete"
    assert "owner elapsed time disagrees" in result["issues"][0]
