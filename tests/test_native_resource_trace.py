"""CPU contract fixtures for the continuous-trace window analyzer.

No native/GPU reuse decision is asserted here: these tests pin the published
window derivation over one continuous trace (baseline reconstruction, marker
pairing, defect refusals, static carve-outs) and the analyzer CLI contract.
"""
import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    return importlib.import_module("experiments.native_resource_trace")

def _trace():
    configuration = ["register_callbacks", "allocation_source", "enable_memory2",
                     "enable_memory_pool", "enable_runtime", "enable_driver",
                     "flush_before_disable", "flush_after_disable"]
    value = {"schema": "tessera.cupti_memory_trace.v1", "process_id": 7,
             "cupti_version": 130001, "start_ns": 1, "end_ns": 1000,
             "capture": {"started_before_cuda_libraries": True,
                         "collector_library_sha256": "a" * 64, "start_code": 0, "stop_code": 0},
             "errors": [], "pool_records": 0, "completed_buffers": 1,
             "configuration": [{"operation": name, "code": 0} for name in configuration],
             "dropped_records": [{"code": 0, "count": 0}] * 2,
             "markers": [], "memory_events": [], "api_events": []}
    def event(t, address, size, operation, source):
        correlation = t
        value["memory_events"].append({"timestamp_ns": t, "process_id": 7,
            "device_id": 0, "context_id": 42, "stream_id": 1, "address": address,
            "bytes": size, "correlation_id": correlation, "operation": operation,
            "device_memory": True, "memory_kind": 3, "async": False, "pool_type": 0,
            "source": source})
        value["api_events"].append({"name": "cudaMalloc" if operation == "allocate" else "cudaFree",
            "start_ns": t - 1, "end_ns": t + 1, "process_id": 7,
            "correlation_id": correlation, "return_value": 0})
    event(10, 1000, 100, "allocate", "nccl-init-fixture")
    for i, size in enumerate((40, 60)):
        begin = 100 + i * 200
        value["markers"] += [{"name": f"rate:r{i}:begin", "timestamp_ns": begin},
                             {"name": f"rate:r{i}:end", "timestamp_ns": begin + 100}]
        event(begin + 20, 2000, size, "allocate", "torch-fixture")
        event(begin + 80, 2000, size, "free", "")
    return value


def _index(trace=None):
    return _module().ContinuousTrace(_trace() if trace is None else trace,
                                     device_id=0, context_id=42)


def test_windows_reconstruct_persistent_baseline_not_just_in_window_events():
    index = _index()
    for i, size in enumerate((40, 60)):
        result = index.window(f"rate:r{i}")
        assert result["baseline_bytes"] == 100
        assert result["window_peak_bytes"] == 100 + size
        assert result["transient_peak_bytes"] == size
        assert result["baseline_live"] == result["end_live"]
        assert result["baseline_live"][0]["source"] == "nccl-init-fixture"
        assert result["allocation_requests"] == [{"bytes": size, "source": "torch-fixture",
                                                   "memory_kind": 3, "count": 1}]
        assert result["trace_sha256"] == index.trace_sha256


@pytest.mark.parametrize("defect", ["late_start", "unfinished", "drop", "pool", "duplicate",
    "unmatched_free", "wrong_size", "unknown_source", "wrong_context", "async",
    "missing_api", "missing_memory", "ambiguous_api", "crossing_api", "graph", "bad_api_after_window"])
def test_continuous_trace_never_hides_invalid_history_or_missing_observation(defect):
    trace = _trace()
    row = trace["memory_events"][1]
    if defect == "late_start":
        trace["capture"]["started_before_cuda_libraries"] = False
    elif defect == "unfinished":
        trace["capture"]["stop_code"] = -1
    elif defect == "drop":
        trace["dropped_records"] = [{"code": 0, "count": 1}] * 2
    elif defect == "pool":
        trace["pool_records"] = 1
    elif defect == "duplicate":
        trace["memory_events"].append(copy.deepcopy(row))
    elif defect == "unmatched_free":
        trace["memory_events"].pop(1)
    elif defect == "wrong_size":
        trace["memory_events"][2]["bytes"] += 1
    elif defect == "unknown_source":
        row["source"] = ""
    elif defect == "wrong_context":
        row["context_id"] = 0
    elif defect == "async":
        row["async"] = True
    elif defect == "missing_api":
        trace["api_events"].pop(1)
    elif defect == "missing_memory":
        trace["memory_events"] = [trace["memory_events"][0]]
    elif defect == "ambiguous_api":
        trace["api_events"].append(copy.deepcopy(trace["api_events"][1]))
    elif defect == "crossing_api":
        trace["api_events"][1]["start_ns"] = 99
    elif defect == "graph":
        trace["api_events"][1]["name"] = "cudaGraphLaunch"
    else:
        trace["api_events"].append({**trace["api_events"][1], "start_ns": 800, "end_ns": 799})
    if defect == "late_start":
        message = "bootstrap"
    elif defect == "unfinished":
        message = "completion"
    elif defect == "drop":
        message = "dropped"
    elif defect == "pool":
        message = "pool"
    elif defect == "duplicate":
        message = "duplicate"
    elif defect == "unmatched_free":
        message = "unmatched|matching allocation"
    elif defect == "wrong_size":
        message = "bytes"
    elif defect == "unknown_source":
        message = "source"
    elif defect == "wrong_context":
        message = "context"
    elif defect == "async":
        message = "domain"
    elif defect == "missing_api":
        message = "no unambiguous in-window API correlation|lacks its matching memory operation"
    elif defect == "missing_memory":
        message = "lacks its matching memory operation"
    elif defect == "ambiguous_api":
        message = "ambiguous"
    elif defect == "crossing_api":
        message = "crosses"
    elif defect == "graph":
        message = "graph"
    else:
        message = "reversed|outside continuous collection"
    with pytest.raises(ValueError, match=message):
        _index(trace).window("rate:r0")


def test_retained_rate_allocation_cannot_be_reclassified_as_one_time_init():
    trace = _trace()
    trace["memory_events"] = trace["memory_events"][:2]
    trace["api_events"] = trace["api_events"][:2]
    with pytest.raises(ValueError, match="baseline|retained"):
        _index(trace).window("rate:r0")


def test_duplicate_or_unpaired_rate_markers_refuse():
    trace = _trace()
    trace["markers"].append(copy.deepcopy(trace["markers"][0]))
    with pytest.raises(ValueError, match="paired|marker"):
        _index(trace).window("rate:r0")


def test_module_static_allocation_inside_a_rate_window_is_refused():
    trace = _trace()
    static = copy.deepcopy(trace["memory_events"][1])
    static.update({"address": 3000, "context_id": 0, "memory_kind": 6,
                   "stream_id": 0, "source": "module-static-fixture",
                   "correlation_id": 0})
    trace["memory_events"].append(static)
    api = copy.deepcopy(trace["api_events"][1])
    api.update({"start_ns": 119, "end_ns": 121, "correlation_id": 2999})
    trace["api_events"].append(api)
    with pytest.raises(ValueError, match="static"):
        _index(trace).window("rate:r0")


# --- round-4: the drift axis is derived, the table is unmixed, the closed
# --- forms are pinned, and every surviving mutant has a killing test.




def test_cli_derives_the_same_windows_as_the_library(tmp_path):
    module = _module()
    trace = _trace()
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    cli = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "native_resource_trace.py"),
         str(path), "--device-id", "0", "--context-id", "42",
         "--interval", "rate:r0", "--interval", "rate:r1"],
        check=True, capture_output=True, text=True)
    report = json.loads(cli.stdout)
    assert report["schema"] == module.CLI_SCHEMA
    index = module.ContinuousTrace(trace, device_id=0, context_id=42)
    for interval, window in report["intervals"].items():
        assert window == index.window(interval)


def test_cli_refuses_intervals_the_trace_does_not_contain(tmp_path):
    trace = _trace()
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    cli = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "native_resource_trace.py"),
         str(path), "--device-id", "0", "--context-id", "42",
         "--interval", "rate:r9"],
        capture_output=True, text=True)
    assert cli.returncode != 0
    assert "unknown intervals" in cli.stderr



def test_resource_row_vocabulary_helpers_classify_by_their_own_fields():
    from experiments.native_operator_resources import api_base_name
    from experiments.native_operator_resources import memory_row_domain
    from experiments.native_operator_resources import ownership_operation

    assert api_base_name("cuMemAlloc_v2") == "cuMemAlloc"
    assert api_base_name("cudaMallocAsync") == "cudaMallocAsync"
    assert ownership_operation("cudaMalloc", 0) == "allocate"
    assert ownership_operation("cudaFree", 0) == "free"
    assert ownership_operation("cudaLaunchKernel", 0) is None
    assert ownership_operation("cudaMalloc", 1) == "unsupported"

    def row(kind, *, async_=False, pool=0):
        return {"memory_kind": kind, "async": async_, "pool_type": pool}

    assert memory_row_domain(row(3)) == "device"
    assert memory_row_domain(row(6)) == "static"
    assert memory_row_domain(row(1)) == "host"
    assert memory_row_domain(row(2)) == "host"
    with pytest.raises(ValueError, match="domain"):
        memory_row_domain(row(3, async_=True))
    with pytest.raises(ValueError, match="domain"):
        memory_row_domain(row(3, pool=2))
    with pytest.raises(ValueError, match="domain"):
        memory_row_domain(row(9))


def test_a_pinned_host_allocation_between_windows_is_not_an_unpriced_gap():
    trace = _trace()
    host = copy.deepcopy(trace["memory_events"][1])
    host.update({"address": 9000, "memory_kind": 1, "correlation_id": 77,
                 "timestamp_ns": 250})  # strictly between rate:r0 [100,200] and rate:r1 [300,400]
    trace["memory_events"].append(host)
    index = _index(trace)  # a host row between the two rate windows must not refuse
    assert index.window("rate:r0")["interval"] == "rate:r0"
