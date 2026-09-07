"""Synthetic callback controls; actual CUPTI callback qualification is separate."""
import copy

import pytest

from experiments.full_engine_cuda_domains import analyze_memory_api_arguments


@pytest.fixture
def trace():
    apis, arguments = [], []
    calls = [("cudaHostAlloc", {"host_address": 4096, "bytes": 64, "flags": 2}),
             ("cudaHostGetDevicePointer", {"host_address": 4112, "device_address": 8208, "flags": 0}),
             ("cudaFree", {"device_address": 0}), ("cudaFreeHost", {"host_address": 4096})]
    for index, (name, fields) in enumerate(calls, 1):
        apis.append({"name": name + "_v3020", "process_id": 10, "correlation_id": index,
                     "start_ns": index * 20 - 10, "end_ns": index * 20, "return_value": 0})
        # Callback timestamp may be outside the measured API body interval.
        arguments.append({"name": name, "process_id": 10, "correlation_id": index,
                          "callback_id": index, "site": "exit", "timestamp_ns": index * 20 + 1,
                          "return_value": 0, **fields})
    common = {"process_id": 10, "memory_kind": 2, "device_id": 0, "context_id": 1,
              "pool_type": 0, "async": False, "address": 4096, "bytes": 64, "source": "fixture.so"}
    return {"process_id": 10, "start_ns": 1, "end_ns": 100,
        "argument_schema": "tessera.cuda_memory_api_arguments.v1", "api_events": apis,
        "api_argument_events": arguments,
        "configuration": [{"operation": "subscribe_arguments", "code": 0},
                          {"operation": "unsubscribe_arguments", "code": 0}]
                         + [{"operation": f"enable_argument_callback_{i}", "code": 0} for i in range(1, 5)],
        "memory_events": [{**common, "correlation_id": 1, "timestamp_ns": 15, "operation": "allocate"},
                          {**common, "correlation_id": 4, "timestamp_ns": 75, "operation": "free"}]}


def test_pointer_arguments_bind_pinned_lifetime_mapping_and_null_free(trace):
    result = analyze_memory_api_arguments(trace)
    assert result["status"] == "observed_argument_domains", result
    assert result["handled_api_keys"] == [[10, 1], [10, 2], [10, 3], [10, 4]]
    assert result["host_allocations"][0]["freed_ns"] == 75
    mapping = result["host_mappings"][0]
    assert (mapping["host_address"], mapping["device_address"], mapping["bytes_from_queried_offset"]) == (4112, 8208, 48)
    assert result["null_device_frees"][0]["device_address"] == 0


@pytest.mark.parametrize("defect", ["missing_callback", "missing_host_pair", "wrong_mapping", "nonnull_free",
                                    "failed_free", "wrong_allocation_size", "map_after_free"])
def test_missing_or_conflicting_ownership_is_not_exempted(trace, defect):
    if defect == "missing_callback":
        trace["api_argument_events"].pop(2)
    elif defect == "missing_host_pair":
        trace["memory_events"] = []
    elif defect == "wrong_mapping":
        trace["api_argument_events"][1]["host_address"] = 9999
    elif defect == "nonnull_free":
        trace["api_argument_events"][2]["device_address"] = 1234
    elif defect == "failed_free":
        trace["api_argument_events"][2]["return_value"] = 1
        trace["api_events"][2]["return_value"] = 1
    elif defect == "wrong_allocation_size":
        trace["api_argument_events"][0]["bytes"] = 63
    elif defect == "map_after_free":
        trace["api_events"][1].update(start_ns=80, end_ns=85)
        trace["api_argument_events"][1]["timestamp_ns"] = 86
    result = analyze_memory_api_arguments(trace)
    assert result["status"] == "incomplete"
    assert result["issues"]
    assert len(result["handled_api_keys"]) < 4


@pytest.mark.parametrize("defect", ["duplicate", "foreign", "disabled", "return_mismatch", "boolean_null"])
def test_ambiguous_callback_identity_fails_closed(trace, defect):
    if defect == "duplicate":
        trace["api_argument_events"].append(copy.deepcopy(trace["api_argument_events"][0]))
    elif defect == "foreign":
        trace["api_argument_events"][0]["process_id"] = 11
    elif defect == "disabled":
        trace["api_argument_events"][0]["callback_id"] = 999
    elif defect == "return_mismatch":
        trace["api_argument_events"][0]["return_value"] = 1
    elif defect == "boolean_null":
        trace["api_argument_events"][2]["device_address"] = False
    with pytest.raises(ValueError):
        analyze_memory_api_arguments(trace)


def test_old_capture_has_no_argument_based_exemptions(trace):
    trace.pop("argument_schema")
    result = analyze_memory_api_arguments(trace)
    assert result["status"] == "unavailable"
    assert result["handled_api_keys"] == []


def test_host_and_cuda_aliases_join_one_lifetime_at_checkpoint(trace):
    from experiments.full_engine_cuda_domains import match_host_owner
    from experiments.full_engine_resources import _checkpoint_owners
    domains = analyze_memory_api_arguments(trace)
    def owner(name, kind, address):
        return {"owner_id": name, "category": "shared", "provenance": "fixture pointer view",
                "device_type": kind, "device_id": 0 if kind == "cuda" else None,
                "address": address, "bytes": 16, "storage_offset_bytes": 0, "view_extent_bytes": 16}
    cpu, cuda = owner("cpu", "cpu", 4112), owner("cuda", "cuda", 8208)
    cp = {"owners": [cpu, cuda], "cupti_timestamp_ns": 50, "label": "mapped", "trace_index": 1}
    issues = []
    result = _checkpoint_owners(cp, {}, 0, issues, domains)
    assert issues == []
    assert result["unique_pinned_host_backing_bytes"] == 64
    assert result["pinned_host_storages"][0]["owners"] == ["cpu", "cuda"]
    assert match_host_owner(cuda, 30, domains, 0) is None
    assert match_host_owner(cuda, 75, domains, 0) is None
    assert match_host_owner({**cuda, "bytes": 49}, 50, domains, 0) is None
    assert match_host_owner(cuda, 50, domains, 1) is None
