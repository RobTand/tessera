"""Resource-observer refusal regressions; CPU inputs are not GPU evidence."""
import copy
import ctypes
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def test_context_lookup_uses_the_collectors_loaded_cupti(monkeypatch):
    from experiments.native_operator_resources import NativeMemoryCollector

    def current_context(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        return 0

    def context_id(context, pointer):
        assert context.value == 123
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32))[0] = 42
        return 0

    def load(name):
        assert name == "libcuda.so.1", "CUPTI must use the collector's linked dependency"
        return SimpleNamespace(cuCtxGetCurrent=current_context)

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        memory_snapshot=lambda: [], synchronize=lambda device: None,
        reset_peak_memory_stats=lambda device: None, memory_allocated=lambda device: 0,
        max_memory_allocated=lambda device: 0, get_allocator_backend=lambda: "native")))
    collector = object.__new__(NativeMemoryCollector)
    collector.start_code = 0
    collector._lib = SimpleNamespace(cuptiGetContextId=context_id)
    collector.mark = lambda name: 1
    sentinel = object()
    output, observation = collector.observe_apply(lambda: sentinel, "fixture")
    assert output is sentinel
    assert observation["context_id"] == 42


@pytest.fixture
def real_evidence():
    path = os.environ.get("TESSERA_NATIVE_RESOURCE_FIXTURE") or (
        Path(__file__).parent / "fixtures" / "native_operator_resource_qualification.json")
    return json.loads(Path(path).read_text())


@pytest.fixture
def real_static_evidence():
    path = Path(__file__).parent / "fixtures" / "native_operator_resource_static_startup.json"
    return json.loads(path.read_text())


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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_real_preexisting_static_storage_is_separate_from_operator_scratch(real_static_evidence, phase):
    from experiments.native_operator_resources import analyze_trace
    actual = analyze_trace(real_static_evidence["trace"], interval=phase,
                           torch_observation=real_static_evidence["torch_observations"][phase])
    assert actual["status"] == "complete_operator_bound", actual
    assert actual["external_native_peak_bytes"] == 0
    assert actual["peak_scratch_bytes"] == real_static_evidence["expected_scratch_bytes"][phase]
    assert actual["startup_static"]["live_bytes"] == real_static_evidence["expected_startup_bytes"]
    assert actual["startup_static"]["live_allocation_count"] == 2
    assert actual["startup_static"]["raw_context_ids"] == [0]
    assert actual["full_model_fixed_resources_complete"] is False


@pytest.mark.parametrize("defect", ["static_allocate_inside", "static_free_inside", "static_wrong_device",
                                    "static_unknown_context", "static_unknown_source", "static_duplicate",
                                    "dynamic_zero_context", "dynamic_wrong_device"])
def test_static_domain_does_not_relax_dynamic_or_in_apply_ownership(real_static_evidence, defect):
    from experiments.native_operator_resources import analyze_trace
    data = copy.deepcopy(real_static_evidence)
    trace = data["trace"]
    static = next(r for r in trace["memory_events"] if r["memory_kind"] == 6)
    dynamic = next(r for r in trace["memory_events"] if r["memory_kind"] == 3)
    begin = next(r["timestamp_ns"] for r in trace["markers"] if r["name"] == "prefill:begin")
    if defect == "static_allocate_inside":
        static["timestamp_ns"] = begin + 1
    elif defect == "static_free_inside":
        freed = copy.deepcopy(static)
        freed.update(operation="free", timestamp_ns=begin + 1)
        trace["memory_events"].append(freed)
    elif defect == "static_wrong_device":
        static["device_id"] += 1
    elif defect == "static_unknown_context":
        static["context_id"] = 17
    elif defect == "static_unknown_source":
        static["source"] = ""
    elif defect == "static_duplicate":
        trace["memory_events"].append(copy.deepcopy(static))
    elif defect == "dynamic_zero_context":
        dynamic["context_id"] = 0
    elif defect == "dynamic_wrong_device":
        dynamic["device_id"] += 1
    actual = analyze_trace(trace, interval="prefill", torch_observation=data["torch_observations"]["prefill"])
    assert actual["status"] == "incomplete", actual
    assert actual["peak_scratch_bytes"] is None
    assert actual["reasons"]


@pytest.mark.parametrize("defect", ["dropped", "missing_drop_query", "missing_enable", "wrong_context",
                                    "missing_free", "changed_reservations", "missing_correlation",
                                    "duplicate_allocate", "async", "graph", "no_memory_records",
                                    "unobserved_external_mapping", "missing_transient_memory_pair",
                                    "allocation_api_outside_interval", "memory_outside_api",
                                    "ambiguous_api_correlation"])
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
    elif defect == "missing_transient_memory_pair":
        trace["memory_events"] = [r for r in trace["memory_events"] if r not in rows]
    elif defect == "allocation_api_outside_interval":
        api = next(r for r in trace["api_events"] if r["correlation_id"] == rows[0]["correlation_id"])
        api["start_ns"] = start - 2
        api["end_ns"] = start - 1
    elif defect == "memory_outside_api":
        api = next(r for r in trace["api_events"] if r["correlation_id"] == rows[0]["correlation_id"])
        api["end_ns"] = rows[0]["timestamp_ns"] - 1
    elif defect == "ambiguous_api_correlation":
        api = next(r for r in trace["api_events"] if r["correlation_id"] == rows[0]["correlation_id"])
        trace["api_events"].append(copy.deepcopy(api))
    elif defect == "unobserved_external_mapping":
        api = copy.deepcopy(next(r for r in trace["api_events"] if r["correlation_id"] == rows[0]["correlation_id"]))
        api.update(name="cudaExternalMemoryGetMappedBuffer_v10000", correlation_id=900000)
        trace["api_events"].append(api)
    actual = analyze_trace(trace, interval="qualification", torch_observation=observation)
    assert actual["status"] == "incomplete", actual
    assert actual["peak_scratch_bytes"] is None
    assert actual["reasons"]
