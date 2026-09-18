"""The same-run timing observation of one synthetic timing pass (tessera#399).

Every fixture here is a directory of hand-written JSON: no torch, no vLLM, no
CUDA, no profiler. What a passing suite establishes is the qualification
contract -- ``partition.established`` is true only when every check passed,
each check fails on its own evidence and names itself, a witness that could
not be taken is a failed check rather than a pass, and the terms are listed
per sample and never aggregated. It establishes nothing about a real device.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.full_engine_timing_observation import (
    OBSERVATION_SCHEMA, build_timing_observation, count_gpu_operations,
    host_exclusivity, worker_exclusivity,
)

UNITS = ["l:dense", "s:moe"]
PID = 4242
BASE = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc).timestamp()
IDENTITY = {
    "configuration_sha256": "a" * 64, "model_sha256": "b" * 64,
    "runtime_manifest_sha256": "c" * 64, "workload_sha256": "d" * 64,
    "assignment_sha256": "e" * 64, "canonical_units_sha256": "f" * 64,
}
#: The partition arm carries the observer's own overhead, so its whole-step
#: time is the larger of the two; the difference is disclosed, never removed.
WHOLE_STEP_MS = {"control": [4.0, 5.0], "partition": [5.0, 6.0]}


def _census(available=True, pid=PID):
    if not available:
        return {"available": False, "error": "pynvml unavailable", "processes": None}
    return {"available": True, "device_index": 0, "count": 1,
            "processes": [{"pid": pid, "used_gpu_memory_bytes": 1024}]}


def _stamp(offset_s=0.0):
    return datetime.fromtimestamp(BASE + offset_s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _arm_files(arm, sample):
    """One arm's capture, profile and partition analysis, in the recorder's shape."""
    capture = {"schema": "tessera.full_engine_timing_capture.v1", "arm": arm, "errors": [],
               "canonical_unit_ids": list(UNITS), "device_id": 0, "ranges": {}, "steps": [],
               "observer_span": {"started_unix_ns": int(BASE * 1e9),
                                 "finished_unix_ns": int((BASE + 1) * 1e9)},
               "exclusivity": {"at_arm": _census(), "at_finish": _census()}}
    events = []
    for index, (step_id, phase, tokens) in enumerate([("0", "prefill", 512), ("1", "decode", 1)]):
        offset = index * 1000
        step_name = "step." + step_id
        capture["ranges"][step_name] = {"kind": "step", "step_id": step_id}
        events.append({"ph": "X", "cat": "user_annotation", "name": step_name,
                       "pid": 10, "tid": 20, "ts": offset, "dur": 900})
        units = []
        for position, unit in enumerate(UNITS):
            name = f"unit.{step_id}.{position}"
            capture["ranges"][name] = {"kind": "unit", "step_id": step_id, "unit_id": unit}
            units.append({"unit_id": unit, "range_name": name, "elapsed_ms": 1.0 + position})
            correlation = offset + position
            events.extend([
                {"ph": "X", "cat": "user_annotation", "name": name, "pid": 10, "tid": 20,
                 "ts": offset + 100 + position * 200, "dur": 100},
                {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10,
                 "tid": 20, "ts": offset + 120 + position * 200, "dur": 1,
                 "args": {"correlation": correlation}},
                {"ph": "X", "cat": "kernel", "name": "native",
                 "ts": offset + 130 + position * 200, "dur": 5,
                 "args": {"correlation": correlation, "device": 0, "stream": 7}}])
        events.extend([
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaMemcpyAsync", "pid": 10, "tid": 20,
             "ts": offset + 500, "dur": 1, "args": {"correlation": offset + 3}},
            {"ph": "X", "cat": "gpu_memcpy", "name": "output D2H", "ts": offset + 510, "dur": 5,
             "args": {"correlation": offset + 3, "device": 0, "stream": 8}}])
        capture["steps"].append({
            "step_id": step_id, "phase": phase, "scheduled_tokens": tokens,
            "main_stream_id": 7, "copy_stream_id": 8, "units": units,
            "whole_step_ms": WHOLE_STEP_MS[arm][index], "fixed_gaps_ms": [1.0, 1.0, 1.0],
            "copy_event_observed": True,
            "completion_join": "main_event_and_stock_copy_event"})
    partition = {"schema": "tessera.full_engine_timing_partition.v1",
                 "status": "observed_same_run_partition", "issues": []}
    return {"capture": capture, "profile": {"traceEvents": events}, "partition": partition}


def build_pass(tmp_path, *, samples=1, planned=None, mutate=None):
    """Write one synthetic timing pass and return ``(capture_dir, launch_dir)``.

    ``mutate`` receives the whole in-memory tree before it is written, so a
    defect is introduced in exactly one place and everything else stays the
    nominal pass.
    """
    capture_dir, launch_dir = tmp_path / "capture", tmp_path / "launch"
    arms, files = [], {}
    for sample in range(samples):
        for arm in ("control", "partition"):
            directory = capture_dir / f"{arm}-{sample}"
            files[(arm, sample)] = _arm_files(arm, sample)
            arms.append({"arm": arm, "sample": sample, "armed": [{"pid": PID}],
                         "tokens": [7, 8, 9], "workers": [{"directory": str(directory)}]})
    tree = {
        "plan": {"schema": "tessera.stock_engine_resource_observer_plan.v1",
                 "identity": dict(IDENTITY),
                 "timing_samples": samples if planned is None else planned},
        "run": {"schema": "tessera.stock_engine_raw_timing_run.v1", "plan_sha256": "9" * 64,
                "warmup_tokens": [7, 8, 9], "arms": arms},
        "files": files,
        "vitals": [f"{_stamp()} gpu_w=20 compute_apps={PID}"],
    }
    if mutate is not None:
        mutate(tree)
    capture_dir.mkdir(parents=True)
    launch_dir.mkdir(parents=True)
    (capture_dir / "observer-plan.json").write_text(json.dumps(tree["plan"]))
    (capture_dir / "run.json").write_text(json.dumps(tree["run"]))
    for entry in tree["run"]["arms"]:
        directory = Path(entry["workers"][0]["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        payload = tree["files"][(entry["arm"], entry["sample"])]
        for name in ("capture", "profile", "partition"):
            (directory / f"{name}.json").write_text(json.dumps(payload[name]))
    (launch_dir / "host-vitals.log").write_text("\n".join(tree["vitals"]) + "\n")
    return capture_dir, launch_dir


def _failed(observation):
    return {name for name, check in observation["qualification"].items() if not check["passed"]}


# --- the established pass ----------------------------------------------------


def test_every_qualification_check_passes_on_a_nominal_pass(tmp_path):
    capture_dir, launch_dir = build_pass(tmp_path)
    observation = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert observation["schema"] == OBSERVATION_SCHEMA
    assert _failed(observation) == set()
    assert observation["partition"]["established"] is True
    assert observation["partition"]["reason"] is None
    assert observation["run_identity"] == IDENTITY
    assert observation["process_id"] == PID
    assert observation["canonical_unit_ids"] == UNITS
    assert observation["timing_samples"] == 1


def test_the_terms_are_listed_per_sample_and_never_aggregated(tmp_path):
    capture_dir, launch_dir = build_pass(tmp_path, samples=2)
    terms = build_timing_observation(capture_dir, launch_dir=launch_dir)["partition"]["terms"]
    assert set(terms) == {"prefill", "decode"}
    assert terms["prefill"]["samples"] == [0, 1]
    assert terms["prefill"]["whole_step_ms"] == [5.0, 5.0]
    assert terms["decode"]["whole_step_ms"] == [6.0, 6.0]
    assert terms["prefill"]["fixed_gap_sum_ms"] == [3.0, 3.0]
    assert terms["prefill"]["fixed_gaps_ms"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert terms["prefill"]["candidate_ms"] == {"l:dense": [1.0, 1.0], "s:moe": [2.0, 2.0]}
    assert terms["prefill"]["unit_order"] == UNITS


def test_the_observer_overhead_is_disclosed_beside_the_terms_and_never_subtracted(tmp_path):
    capture_dir, launch_dir = build_pass(tmp_path)
    observation = build_timing_observation(capture_dir, launch_dir=launch_dir)
    overhead = observation["observer_overhead"]["prefill"]
    assert overhead == [{"sample": 0, "control_whole_step_ms": 4.0,
                         "partition_whole_step_ms": 5.0, "difference_ms": 1.0}]
    terms = observation["partition"]["terms"]["prefill"]
    # The term is the partition arm's own measured step, with the control arm's
    # beside it: nothing here is a difference.
    assert terms["whole_step_ms"] == [5.0]
    assert terms["control_whole_step_ms"] == [4.0]


# --- one failed check at a time ----------------------------------------------


def _token_mismatch(tree):
    tree["run"]["arms"][0]["tokens"] = [7, 8, 99]


def _wrong_step_shape(tree):
    tree["files"][("partition", 0)]["capture"]["steps"][0]["scheduled_tokens"] = 511


def _partition_incomplete(tree):
    partition = tree["files"][("partition", 0)]["partition"]
    partition.update(status="incomplete", issues=["a step's GPU work is on an unjoined stream"])


def _gpu_counts_differ(tree):
    tree["files"][("partition", 0)]["profile"]["traceEvents"].extend([
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10, "tid": 20,
         "ts": 600, "dur": 1, "args": {"correlation": 777}},
        {"ph": "X", "cat": "kernel", "name": "observer_extra", "ts": 610, "dur": 5,
         "args": {"correlation": 777, "device": 0, "stream": 7}}])


def _nvml_unavailable(tree):
    tree["files"][("partition", 0)]["capture"]["exclusivity"]["at_finish"] = _census(False)


def _vitals_without_compute_apps(tree):
    tree["vitals"] = [f"{_stamp()} gpu_w=20"]


def _two_compute_apps(tree):
    tree["vitals"] = [f"{_stamp()} gpu_w=20 compute_apps={PID};9999"]


def _two_worker_pids(tree):
    tree["run"]["arms"][0]["armed"] = [{"pid": PID + 1}]


@pytest.mark.parametrize("mutate,check,planned", [
    (_token_mismatch, "identical_tokens", None),
    (None, "arms_complete", 2),
    (_wrong_step_shape, "step_shape", None),
    (_partition_incomplete, "stream_coverage", None),
    (_gpu_counts_differ, "gpu_operation_counts_agree", None),
    (_nvml_unavailable, "device_exclusivity", None),
    (_vitals_without_compute_apps, "device_exclusivity", None),
    (_two_compute_apps, "device_exclusivity", None),
    (_two_worker_pids, "single_worker", None),
])
def test_one_defect_fails_exactly_its_check_and_establishes_nothing(
        tmp_path, mutate, check, planned):
    capture_dir, launch_dir = build_pass(tmp_path, planned=planned, mutate=mutate)
    observation = build_timing_observation(capture_dir, launch_dir=launch_dir)
    assert _failed(observation) == {check}
    assert observation["partition"]["established"] is False
    assert check in observation["partition"]["reason"]
    assert observation["partition"]["terms"] is None


def test_a_launcher_directory_that_carries_no_vitals_log_cannot_witness(tmp_path):
    # A witness that could not be taken is a failed check, never a pass.
    capture_dir, _ = build_pass(tmp_path)
    observation = build_timing_observation(capture_dir)
    assert _failed(observation) == {"device_exclusivity"}
    assert observation["qualification"]["device_exclusivity"]["host"]["passed"] is False
    assert observation["partition"]["established"] is False


# --- the two exclusivity witnesses, read directly ----------------------------


def test_the_host_witness_reads_the_samples_within_one_cadence_of_a_span():
    samples = [(BASE - 20, [str(PID)]), (BASE - 4, [str(PID)]), (BASE + 0.5, [str(PID)]),
               (BASE + 12, [str(PID)])]
    result = host_exclusivity(samples, [(BASE, BASE + 1)], cadence_s=5.0)
    assert result["passed"] is True
    assert result["samples"] == 2
    assert result["compute_apps"] == [str(PID)]


def test_the_host_witness_refuses_a_span_no_sample_falls_near():
    samples = [(BASE - 100, [str(PID)])]
    result = host_exclusivity(samples, [(BASE, BASE + 1)], cadence_s=5.0)
    assert result["passed"] is False
    assert "no host sample falls within one cadence" in result["reason"]


def test_the_host_witness_refuses_a_second_compute_process():
    samples = [(BASE, [str(PID), "9999"])]
    assert host_exclusivity(samples, [(BASE, BASE + 1)])["passed"] is False


def test_the_worker_witness_names_the_arm_whose_census_is_missing():
    arms = [{"arm": "partition", "sample": 0,
             "capture": {"exclusivity": {"at_arm": _census(), "at_finish": _census(False)}}}]
    result = worker_exclusivity(arms)
    assert result["passed"] is False
    assert "partition/0/at_finish" in result["reason"]


def test_the_worker_witness_refuses_a_second_pid_in_the_census():
    arms = [{"arm": "partition", "sample": 0,
             "capture": {"exclusivity": {"at_arm": _census(), "at_finish": _census(pid=99)}}}]
    assert worker_exclusivity(arms)["passed"] is False


# --- the GPU operation count -------------------------------------------------


def test_every_gpu_operation_is_counted_against_the_step_its_launch_lies_in():
    files = _arm_files("partition", 0)
    counts = count_gpu_operations(files["profile"], files["capture"])
    assert counts["by_step"] == {"0": 3, "1": 3}
    assert counts["outside_steps"] == 0
    assert counts["total"] == 6


def test_a_gpu_operation_whose_launch_is_outside_every_step_is_not_attributed():
    files = _arm_files("partition", 0)
    files["profile"]["traceEvents"].extend([
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10, "tid": 20,
         "ts": 5000, "dur": 1, "args": {"correlation": 4242}},
        {"ph": "X", "cat": "kernel", "name": "stray", "ts": 5010, "dur": 5,
         "args": {"correlation": 4242, "device": 0, "stream": 7}}])
    counts = count_gpu_operations(files["profile"], files["capture"])
    assert counts["by_step"] == {"0": 3, "1": 3}
    assert counts["outside_steps"] == 1


def test_a_foreign_run_schema_is_refused_rather_than_parsed(tmp_path):
    def mutate(tree):
        tree["run"]["schema"] = "tessera.stock_engine_raw_timing_run.v2"

    capture_dir, launch_dir = build_pass(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="unsupported timing run schema"):
        build_timing_observation(capture_dir, launch_dir=launch_dir)


def test_an_arm_capture_that_is_not_the_arm_it_is_listed_as_is_refused(tmp_path):
    def mutate(tree):
        tree["files"][("partition", 0)]["capture"]["arm"] = "control"

    capture_dir, launch_dir = build_pass(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="is not the partition arm"):
        build_timing_observation(capture_dir, launch_dir=launch_dir)
