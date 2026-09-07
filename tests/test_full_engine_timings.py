"""Synthetic event/launch joins; real full-engine qualification is separate."""
import copy

import pytest

from experiments.full_engine_timings import analyze_profile_partition


@pytest.fixture
def evidence():
    capture = {"schema": "tessera.full_engine_timing_capture.v1", "arm": "partition", "errors": [],
               "canonical_unit_ids": ["l:dense", "s:moe"], "device_id": 0, "ranges": {}, "steps": []}
    events = []
    for sid, phase, tokens in [("0", "prefill", 512), ("1", "decode", 1)]:
        offset = int(sid) * 1000
        step_name = "step." + sid
        capture["ranges"][step_name] = {"kind": "step", "step_id": sid}
        events.append({"ph": "X", "cat": "user_annotation", "name": step_name, "pid": 10, "tid": 20, "ts": offset, "dur": 900})
        units = []
        for index, unit in enumerate(capture["canonical_unit_ids"]):
            name = f"unit.{sid}.{index}"
            capture["ranges"][name] = {"kind": "unit", "step_id": sid, "unit_id": unit}
            units.append({"unit_id": unit, "range_name": name, "elapsed_ms": 1.0})
            events.append({"ph": "X", "cat": "user_annotation", "name": name, "pid": 10, "tid": 20, "ts": offset + 100 + index * 200, "dur": 100})
            correlation = offset + index
            events.extend([
                {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10, "tid": 20,
                 "ts": offset + 120 + index * 200, "dur": 1, "args": {"correlation": correlation}},
                {"ph": "X", "cat": "kernel", "name": "native", "ts": offset + 130 + index * 200, "dur": 5,
                 "args": {"correlation": correlation, "device": 0, "stream": 7}}])
        events.extend([
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaMemcpyAsync", "pid": 10, "tid": 20,
             "ts": offset + 500, "dur": 1, "args": {"correlation": offset + 3}},
            {"ph": "X", "cat": "gpu_memcpy", "name": "output D2H", "ts": offset + 510, "dur": 5,
             "args": {"correlation": offset + 3, "device": 0, "stream": 8}}])
        capture["steps"].append({"step_id": sid, "phase": phase, "scheduled_tokens": tokens,
            "main_stream_id": 7, "copy_stream_id": 8, "units": units,
            "whole_step_ms": 5.0, "fixed_gaps_ms": [1.0, 1.0, 1.0],
            "copy_event_observed": True, "completion_join": "main_event_and_stock_copy_event"})
    return capture, {"traceEvents": events}


def test_same_run_partition_preserves_direct_gaps_and_async_copy(evidence):
    result = analyze_profile_partition(*evidence)
    assert result["status"] == "observed_same_run_partition", result
    assert result["timings"] is None
    assert result["admission"] == "not_implemented"
    for step in result["steps"]:
        assert step["fixed_gap_sum_ms"] == 3.0
        assert step["observed_stream_ids"] == [7, 8]
        assert [op["unit_id"] for op in step["gpu_operations"]] == ["l:dense", "s:moe", None]


@pytest.mark.parametrize("defect", ["missing_unit", "extra_unit", "unjoined_stream", "unit_side_stream",
                                    "missing_copy_join", "missing_launch", "duplicate_launch", "missing_range",
                                    "outside_step", "overlap", "wrong_phase", "wrong_sum", "invalid_number",
                                    "missing_gpu", "copy_overlaps_units", "foreign_device"])
def test_incomplete_event_or_gpu_coverage_stays_unpriced(evidence, defect):
    capture, profile = evidence
    events = profile["traceEvents"]
    if defect == "missing_unit":
        capture["steps"][0]["units"].pop()
    elif defect == "extra_unit":
        capture["steps"][0]["units"].append(capture["steps"][0]["units"][0])
    elif defect in {"unjoined_stream", "unit_side_stream", "foreign_device"}:
        gpu = next(event for event in events if event.get("cat") == "kernel")
        gpu["args"]["device" if defect == "foreign_device" else "stream"] = 1 if defect == "foreign_device" else 8 if defect == "unit_side_stream" else 9
    elif defect == "missing_copy_join":
        capture["steps"][0]["copy_event_observed"] = False
    elif defect in {"missing_launch", "duplicate_launch", "copy_overlaps_units"}:
        launch = next(event for event in events if event.get("cat") == "cuda_runtime" and
                      (event["name"] == "cudaMemcpyAsync" if defect == "copy_overlaps_units" else True))
        if defect == "missing_launch":
            events.remove(launch)
        elif defect == "duplicate_launch":
            events.append(copy.deepcopy(launch))
        else:
            launch["ts"] = 250
    elif defect == "missing_range":
        events.pop(0)
    elif defect == "outside_step":
        events[0]["dur"] = 50
    elif defect == "overlap":
        name = capture["steps"][0]["units"][1]["range_name"]
        span = next(event for event in events if event["name"] == name)
        span.update(ts=100, dur=400)
    elif defect == "wrong_phase":
        capture["steps"][0]["scheduled_tokens"] = 511
    elif defect == "wrong_sum":
        capture["steps"][0]["whole_step_ms"] = 6.0
    elif defect == "invalid_number":
        capture["steps"][0]["whole_step_ms"] = float("nan")
    else:
        profile["traceEvents"] = [event for event in events if event.get("cat") not in {"kernel", "gpu_memcpy"}]
    result = analyze_profile_partition(capture, profile)
    assert result["status"] == "incomplete", (defect, result)
    assert result["issues"]
    assert result["timings"] is None


def test_zero_token_cleanup_does_not_consume_prefill_step():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from experiments.full_engine_timings import FullEngineTimingRecorder
    recorder = FullEngineTimingRecorder.__new__(FullEngineTimingRecorder)
    recorder.current, recorder.ranges, recorder.steps = None, {}, []
    recorder.torch = SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name: nullcontext()))
    scheduler = SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids={"warmup"})
    with recorder.housekeeping(scheduler):
        assert recorder.current is None
    assert recorder.steps == []
    assert list(recorder.ranges.values()) == [{"kind": "housekeeping", "scheduled_tokens": 0,
                                               "finished_request_ids": ["warmup"]}]
    scheduler.total_num_scheduled_tokens = 1
    with pytest.raises(RuntimeError, match="schedules tokens"):
        with recorder.housekeeping(scheduler):
            pass


@pytest.mark.parametrize("gpu_work", [False, True])
def test_cleanup_scope_is_retained_and_cannot_hide_gpu_work(evidence, gpu_work):
    capture, profile = evidence
    capture["ranges"]["cleanup"] = {"kind": "housekeeping", "scheduled_tokens": 0,
                                    "finished_request_ids": ["warmup"]}
    profile["traceEvents"].append({"ph": "X", "cat": "user_annotation", "name": "cleanup", "pid": 10, "tid": 20,
                                    "ts": -100, "dur": 50})
    if gpu_work:
        profile["traceEvents"].extend([
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10, "tid": 20,
             "ts": -90, "dur": 1, "args": {"correlation": 5000}},
            {"ph": "X", "cat": "kernel", "name": "cleanup_gpu", "ts": -80, "dur": 5,
             "args": {"correlation": 5000, "device": 0, "stream": 7}}])
    result = analyze_profile_partition(capture, profile)
    assert result["status"] == ("incomplete" if gpu_work else "observed_same_run_partition")
    assert result["timings"] is None


@pytest.mark.parametrize("cpu_range_state", ["unique", "missing", "duplicate"])
def test_gpu_annotation_projection_cannot_duplicate_or_replace_cpu_range(evidence, cpu_range_state):
    capture, profile = evidence
    events = profile["traceEvents"]
    cpu_ranges = [event for event in events if event.get("cat") == "user_annotation"]
    # Kineto emits GPU projections with the same record_function names.
    events.extend(dict(event, cat="gpu_user_annotation", pid=0, tid=7)
                  for event in cpu_ranges)
    if cpu_range_state == "missing":
        events.remove(cpu_ranges[0])
    elif cpu_range_state == "duplicate":
        events.append(dict(cpu_ranges[0]))
    result = analyze_profile_partition(capture, profile)
    assert result["status"] == ("observed_same_run_partition" if cpu_range_state == "unique" else "incomplete"), result
    assert result["timings"] is None and result["admission"] == "not_implemented"
