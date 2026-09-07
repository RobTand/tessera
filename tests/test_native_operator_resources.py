"""Resource-observer refusal regressions; CPU inputs are not GPU evidence."""
import copy
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def real_evidence():
    path = os.environ.get("TESSERA_NATIVE_RESOURCE_FIXTURE") or (
        Path(__file__).parent / "fixtures" / "native_operator_resource_qualification.json")
    return json.loads(Path(path).read_text())


def test_missing_collection_is_unknown_not_zero():
    from experiments.native_operator_resources import analyze_trace
    result = analyze_trace({}, interval="apply", torch_observation={})
    assert result["status"] == "incomplete"
    assert result["peak_scratch_bytes"] is None
    assert result["reasons"]


def test_missing_library_refuses_before_cuda(tmp_path):
    from experiments.native_operator_resources import NativeMemoryCollector
    with pytest.raises(FileNotFoundError):
        NativeMemoryCollector(tmp_path / "missing.so")


def test_real_transient_native_allocation_is_counted(real_evidence):
    from experiments.native_operator_resources import analyze_trace
    actual = analyze_trace(real_evidence["trace"], interval="qualification",
                           torch_observation=real_evidence["torch_observation"])
    assert actual["status"] == "complete_operator_bound", actual
    assert actual["external_native_peak_bytes"] == real_evidence["expected_external_bytes"]
    assert actual["peak_scratch_bytes"] == (actual["external_native_peak_bytes"]
                                              + actual["torch_peak_increment_bytes"])
    assert actual["full_model_fixed_resources_complete"] is False


@pytest.mark.parametrize("defect", ["dropped", "missing_drop_query", "missing_enable", "wrong_context",
                                    "missing_free", "changed_reservations", "missing_correlation",
                                    "duplicate_allocate", "async", "graph", "no_memory_records",
                                    "unobserved_external_mapping"])
def test_incomplete_real_trace_never_produces_resource_zero(real_evidence, defect):
    from experiments.native_operator_resources import analyze_trace
    data = copy.deepcopy(real_evidence)
    trace, observation = data["trace"], data["torch_observation"]
    start = next(r["timestamp_ns"] for r in trace["markers"] if r["name"] == "qualification:begin")
    rows = [r for r in trace["memory_events"] if r["timestamp_ns"] >= start]
    assert len(rows) == 2, "fixture must contain the observed native malloc/free pair"
    if defect == "dropped":
        trace["dropped_records"][0]["count"] = 1
    elif defect == "missing_drop_query":
        trace["dropped_records"].pop()
    elif defect == "missing_enable":
        trace["configuration"] = [r for r in trace["configuration"] if r["operation"] != "enable_memory2"]
    elif defect == "wrong_context":
        rows[0]["context_id"] += 1
    elif defect == "missing_free":
        trace["memory_events"].remove(rows[1])
    elif defect == "changed_reservations":
        observation["after_segments"].pop()
    elif defect == "missing_correlation":
        trace["api_events"] = [r for r in trace["api_events"] if r["correlation_id"] != rows[0]["correlation_id"]]
    elif defect == "duplicate_allocate":
        trace["memory_events"].append(copy.deepcopy(rows[0]))
    elif defect == "async":
        rows[0]["async"] = True
    elif defect == "graph":
        api = next(r for r in trace["api_events"] if r["correlation_id"] == rows[0]["correlation_id"])
        api["name"] = "cudaGraphLaunch_v10000"
    elif defect == "no_memory_records":
        trace["memory_events"] = []
    elif defect == "unobserved_external_mapping":
        api = copy.deepcopy(next(r for r in trace["api_events"] if r["correlation_id"] == rows[0]["correlation_id"]))
        api.update(name="cudaExternalMemoryGetMappedBuffer_v10000", correlation_id=900000)
        trace["api_events"].append(api)
    actual = analyze_trace(trace, interval="qualification", torch_observation=observation)
    assert actual["status"] == "incomplete", actual
    assert actual["peak_scratch_bytes"] is None
    assert actual["reasons"]
