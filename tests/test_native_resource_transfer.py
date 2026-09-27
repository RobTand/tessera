"""CPU contract fixtures. No native/GPU qualification is asserted by these tests."""
import copy
import importlib

import pytest


def _module():
    return importlib.import_module("experiments.native_resource_transfer")


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


def test_stratification_includes_first_median_last_and_at_least_five():
    rates = list(range(128, 897))
    sample = _module().stratified_rates(rates, 5)
    assert len(sample) >= 5
    assert {rates[0], rates[len(rates) // 2], rates[-1]} <= set(sample)
    assert sample == sorted(set(sample))


def test_stratified_sampler_never_drops_the_last_rate():
    mod = _module()
    rates = list(range(256, 266))
    sample = mod.stratified_rates(rates, 5)
    assert len(sample) == 5
    assert sample[0] == 256 and sample[-1] == 265
    assert rates[(len(rates) - 1) // 2] in sample


def _identity():
    return {"schema": "tessera.native_resource_transfer_runtime.v1",
            "image_digest": "sha256:" + "a" * 64, "tessera_package_sha256": "b" * 64,
            "vllm_package_sha256": "c" * 64, "nccl_version": [2, 30, 4],
            "family": "TESSERA_BF16_K1", "world_size": 1,
            "native_runtime_sha256": "d" * 64}


PHASES = ("prefill", "decode")
FRESH_MEDIANS = [1.00, 1.01, 1.02, 1.03, 1.04]   # per rep, distinct -> s_i > 0
PERSISTENT_FACTOR = 1.05                          # common factor: stamped, never refused
EPS_SAMPLES = [1e-4, 2e-4]


def _base_samples(scale):
    return [0.98 * scale, 1.00 * scale, 1.02 * scale]


def _persistent_samples(scale):
    return _base_samples(scale * PERSISTENT_FACTOR)


def _proc_trace(pid, rates, *, init_extra=0, window_bytes=None):
    """One continuous trace whose windows bracket exactly the given rates."""
    value = {"schema": "tessera.cupti_memory_trace.v1", "process_id": pid,
             "cupti_version": 130001, "start_ns": 1, "end_ns": 1,
             "capture": {"started_before_cuda_libraries": True,
                         "collector_library_sha256": "a" * 64, "start_code": 0, "stop_code": 0},
             "errors": [], "pool_records": 0, "completed_buffers": 1,
             "configuration": [{"operation": name, "code": 0} for name in
                               ("register_callbacks", "allocation_source", "enable_memory2",
                                "enable_memory_pool", "enable_runtime", "enable_driver",
                                "flush_before_disable", "flush_after_disable")],
             "dropped_records": [{"code": 0, "count": 0}] * 2,
             "markers": [], "memory_events": [], "api_events": []}

    def event(t, address, size, operation, source):
        correlation = t
        value["memory_events"].append({"timestamp_ns": t, "process_id": pid,
            "device_id": 0, "context_id": 42, "stream_id": 1, "address": address,
            "bytes": size, "correlation_id": correlation, "operation": operation,
            "device_memory": True, "memory_kind": 3, "async": False, "pool_type": 0,
            "source": source})
        value["api_events"].append({"name": "cudaMalloc" if operation == "allocate" else "cudaFree",
            "start_ns": t - 1, "end_ns": t + 1, "process_id": pid,
            "correlation_id": correlation, "return_value": 0})

    event(10, 1000, 100, "allocate", "nccl-init-fixture")
    if init_extra:
        event(12, 1500, init_extra, "allocate", "nccl-init-fixture")
    clock = 100
    for index, rate in enumerate(rates):
        size = 40 if window_bytes is None else window_bytes.get(rate, 40)
        name = f"rate:TESSERA_BF16_K1_R{rate}"
        value["markers"] += [{"name": f"{name}:begin", "timestamp_ns": clock},
                             {"name": f"{name}:end", "timestamp_ns": clock + 100}]
        event(clock + 20, 2000 + index, size, "allocate", "torch-fixture")
        event(clock + 80, 2000 + index, size, "free", "")
        clock += 200
    value["end_ns"] = clock + 100
    return value


def _binding(rate, world=1):
    return {"format": f"TESSERA_BF16_K1_R{rate}",
            "operator": {"wire_sha256": "1" * 64, "wire_record_sha256": "2" * 64,
                         "source_weight": {"sha256": "3" * 64},
                         "rendered_weight": {"sha256": "4" * 64},
                         "native_tensors": {"weight": {"sha256": "5" * 64}},
                         "scheme_sha256": "6" * 64,
                         "declared_route": {"symbol": "fixture", "decoder": "fixture"}},
            "runtime": {"execution": {"tensor_parallel": world}}, "phase_tensors": {}}


def _evidence(*, count=5, world=1, init_extra=0, fresh_bytes=None):
    """A full passing evidence set: traces, cases and the raw FOLLOWUP-7 band."""
    mod = _module()
    identity = _identity()
    identity["world_size"] = world
    rates = list(range(256, 265))
    selected = mod.stratified_rates(rates, count)
    resource_trace = _proc_trace(200, selected, init_extra=init_extra)
    cases, fresh_raw = [], {phase: {} for phase in PHASES}
    persistent_raw = {phase: {"process": None, "rates": []} for phase in PHASES}
    timing_process = {"pid": 300, "boot_id": "CPU-fixture", "start_ticks": 300}
    for index, rate in enumerate(selected):
        rate_name = f"TESSERA_BF16_K1_R{rate}"
        fresh_trace = _proc_trace(100 + index, [rate],
                                  window_bytes=(fresh_bytes or {}))
        fresh_process = {"pid": 100 + index, "boot_id": "CPU-fixture",
                         "start_ticks": 100 + index}
        binding = _binding(rate, world)
        for rep, median in enumerate(FRESH_MEDIANS):
            for phase, scale in (("prefill", 1.0), ("decode", 0.5)):
                fresh_raw[phase].setdefault(str(rate), []).append(
                    {"process": {"pid": 900 + index * 10 + rep, "boot_id": "CPU-fixture",
                                 "start_ticks": 900 + index * 10 + rep},
                     "samples_ms": _base_samples(median * scale)})
        for phase, scale in (("prefill", 1.02), ("decode", 0.51)):
            persistent_raw[phase]["process"] = copy.deepcopy(timing_process)
            persistent_raw[phase]["rates"].append(
                {"q256": rate, "samples_ms": _persistent_samples(scale),
                 "time_in_process": index + 1})
        cases.append({"rate": rate, "cut_axis": None,
                      "fresh": {"trace": fresh_trace, "device_id": 0, "context_id": 42,
                                "interval": f"rate:{rate_name}",
                                "binding": copy.deepcopy(binding),
                                "process": copy.deepcopy(fresh_process)},
                      "resource": {"interval": f"rate:{rate_name}",
                                   "device_id": 0, "context_id": 42,
                                   "binding": copy.deepcopy(binding),
                                   "process": {"pid": 200, "boot_id": "CPU-fixture",
                                               "start_ticks": 200}},
                      "timing": {"samples_ms": {phase: _persistent_samples(scale)
                                                for phase, scale in
                                                (("prefill", 1.02), ("decode", 0.51))},
                                 "collector_started": False,
                                 "time_in_process": index + 1,
                                 "binding": copy.deepcopy(binding),
                                 "process": copy.deepcopy(timing_process)}})
    band = {"source": "fresh_process_repeat_r5_pooled_log", "gate_kind": "not_detected",
            "raw": {"eps": {"samples_ms": list(EPS_SAMPLES),
                            "eps_source": "CPU fixture timer (fixture only)"},
                    "phases": {phase: {"fresh": fresh_raw[phase],
                                       "persistent": persistent_raw[phase]}
                               for phase in PHASES}}}
    return {"mod": mod, "identity": identity, "rates": rates, "selected": selected,
            "resource_trace": resource_trace, "cases": cases, "band": band}


def _qualify(evidence, *, change=None, band_change=None):
    mod = evidence["mod"]
    cases = copy.deepcopy(evidence["cases"])
    band = copy.deepcopy(evidence["band"])
    if change:
        change(cases, evidence)
    if band_change:
        band_change(band, evidence)
    return mod.qualify_transfer(evidence["identity"], rates=evidence["rates"],
                                cases=cases, noise_band=band,
                                resource_trace=evidence["resource_trace"])


def _resource_records(evidence):
    mod = evidence["mod"]
    index = mod.ContinuousTrace(evidence["resource_trace"], device_id=0, context_id=42)
    records = []
    for case in evidence["cases"]:
        rate_name = f"TESSERA_BF16_K1_R{case['rate']}"
        records.append({"schema": "tessera.native_resource_transfer_record.v1",
                        "status": "observed", "rate": rate_name, "q256": case["rate"],
                        "runtime_identity": copy.deepcopy(evidence["identity"]),
                        "collector": {"started": True, "library_sha256": "a" * 64},
                        "process": copy.deepcopy(case["resource"]["process"]),
                        "binding": copy.deepcopy(case["resource"]["binding"]),
                        "device_id": 0, "context_id": 42,
                        "window": index.window(f"rate:{rate_name}")})
    return records


def test_matching_three_way_observations_qualify_and_bind_the_runtime_identity():
    pytest.importorskip("scipy")
    evidence = _evidence()
    qualification = _qualify(evidence)
    assert qualification["status"] == "passed", qualification["reasons"]
    assert qualification["timing_gate"]["gate_kind"] == "not_detected"
    for phase in PHASES:
        report = qualification["timing_gate"]["phases"][phase]
        # the fixture's fresh m_i is a mean of logs, so d_bar carries the tiny
        # Jensen gap of the fixture medians, not exactly log(factor)
        import math as _math
        assert abs(report["d_bar_log"] - _math.log(PERSISTENT_FACTOR)) < 1.5e-4
        assert report["d_bar_interval_log"] is not None
        assert report["mde80_log"] > 0
        assert report["fallback_rates"] == []
    transfer = {"qualification_id": qualification["qualification_id"],
                "runtime_identity": _identity()}
    evidence["mod"].require_transfer(
        transfer, runtime_identity=_identity(), qualifications=[qualification],
        resource_records=_resource_records(evidence),
        resource_trace=evidence["resource_trace"], device_id=0, context_id=42)


def test_tp2_is_refused_until_cases_are_bound_to_their_axis_and_rank():
    pytest.importorskip("scipy")
    evidence = _evidence(world=2)
    identity = evidence["identity"]
    for case in evidence["cases"]:
        case["cut_axis"] = "input"
        for leg in ("fresh", "resource", "timing"):
            case[leg]["binding"]["runtime"]["execution"]["tensor_parallel"] = 2
    with pytest.raises(ValueError, match="cut axis and rank"):
        evidence["mod"].qualify_transfer(identity, rates=evidence["rates"],
                                         cases=evidence["cases"],
                                         noise_band=evidence["band"],
                                         resource_trace=evidence["resource_trace"])


@pytest.mark.parametrize("field", ["allocation_requests", "transient_peak_bytes"])
def test_exact_derived_window_comparison_cannot_be_tolerated(field):
    pytest.importorskip("scipy")
    def change(cases, evidence):
        rate = evidence["selected"][0]
        case = next(c for c in cases if c["rate"] == rate)
        if field == "allocation_requests":
            # the resource trace prices a different allocation size for this
            # rate: bump the first window's alloc AND its paired free together
            begin = evidence["resource_trace"]["markers"][0]["timestamp_ns"]
            end = evidence["resource_trace"]["markers"][1]["timestamp_ns"]
            target = None
            for event in evidence["resource_trace"]["memory_events"]:
                if (event.get("source") == "torch-fixture" and begin < event["timestamp_ns"] < end):
                    target = event["address"]
                    event["bytes"] += 1
            for event in evidence["resource_trace"]["memory_events"]:
                if (event.get("operation") == "free" and event["address"] == target):
                    event["bytes"] += 1
        else:
            for marker in evidence["resource_trace"]["markers"][:2]:
                pass
            # transient peak: a second, larger in-window allocation that frees
            # before the window ends
            base = evidence["resource_trace"]["memory_events"]
            first_alloc = next(e for e in base if e.get("source") == "torch-fixture")
            begin = evidence["resource_trace"]["markers"][0]["timestamp_ns"]
            entry = copy.deepcopy(first_alloc)
            entry.update({"timestamp_ns": begin + 40, "address": 9000, "bytes": 90,
                          "correlation_id": 9000})
            base.append(entry)
            api = copy.deepcopy(evidence["resource_trace"]["api_events"][1])
            api.update({"start_ns": begin + 39, "end_ns": begin + 41,
                        "correlation_id": 9000})
            evidence["resource_trace"]["api_events"].append(api)
            free = copy.deepcopy(entry)
            free.update({"timestamp_ns": begin + 60, "operation": "free", "source": "",
                         "correlation_id": 9001})
            base.append(free)
            free_api = copy.deepcopy(api)
            free_api.update({"name": "cudaFree", "start_ns": begin + 59,
                             "end_ns": begin + 61, "correlation_id": 9001})
            evidence["resource_trace"]["api_events"].append(free_api)
    result = _qualify(_evidence(), change=change)
    assert result["status"] == "failed"
    assert result["fallback"] == "fresh_process"
    assert any(field in reason for reason in result["reasons"])


def test_resource_initialization_multiset_must_match_the_fresh_leg():
    pytest.importorskip("scipy")
    def change(cases, evidence):
        evidence["resource_trace"]["memory_events"].append(
            {**evidence["resource_trace"]["memory_events"][0],
             "timestamp_ns": 12, "address": 1500, "bytes": 64, "correlation_id": 12})
        evidence["resource_trace"]["api_events"].append(
            {**evidence["resource_trace"]["api_events"][0],
             "start_ns": 11, "end_ns": 13, "correlation_id": 12})
    result = _qualify(_evidence(), change=change)
    assert result["status"] == "failed"
    assert any("initialization_requests" in reason for reason in result["reasons"])


@pytest.mark.parametrize("defect", ["collector_started", "world_binding",
                                    "missing_process", "leg_missing"])
def test_round2_mutants_m1_to_m3_are_refused(defect):
    pytest.importorskip("scipy")
    def change(cases, evidence):
        case = cases[0]
        if defect == "collector_started":
            case["timing"]["collector_started"] = True
        elif defect == "world_binding":
            for leg in ("fresh", "resource", "timing"):
                case[leg]["binding"]["runtime"]["execution"]["tensor_parallel"] = 2
        elif defect == "missing_process":
            case["timing"].pop("process")
        else:
            case.pop("resource")
    result = _qualify(_evidence(), change=change)
    assert result["status"] == "failed"
    expected = {"collector_started": "timing collector was started",
                "world_binding": "binding world differs",
                "missing_process": "process identity is missing",
                "leg_missing": "a qualification leg is missing"}[defect]
    assert any(expected in reason for reason in result["reasons"])


def test_world_one_persistence_checks_bite():
    pytest.importorskip("scipy")
    def resource_change(cases, evidence):
        cases[1]["resource"]["process"] = {"pid": 201, "boot_id": "CPU-fixture",
                                           "start_ticks": 201}
        cases[1]["resource"]["binding"] = copy.deepcopy(cases[0]["resource"]["binding"])
    result = _qualify(_evidence(), change=resource_change)
    assert result["status"] == "failed"
    assert any("resource pass must be one persistent process" in r for r in result["reasons"])

    # the timing bucket's own one-process rule: a second pass-T process at
    # world 1 is refused even though each leg still matches the band rows
    def timing_two_processes(cases, evidence):
        cases[1]["timing"]["process"] = {"pid": 301, "boot_id": "CPU-fixture",
                                         "start_ticks": 301}
    result = _qualify(_evidence(), change=timing_two_processes)
    assert result["status"] == "failed"
    assert any("timing pass must be one persistent process" in r for r in result["reasons"])


def test_timing_leg_is_bound_to_the_raw_band():
    pytest.importorskip("scipy")
    def change(cases, evidence):
        cases[0]["timing"]["samples_ms"]["prefill"] = [9.9, 9.8, 9.7]
    result = _qualify(_evidence(), change=change)
    assert result["status"] == "failed"
    assert any("timing samples differ from the raw band" in r for r in result["reasons"])


def test_fresh_and_resource_windows_must_come_from_distinct_traces():
    pytest.importorskip("scipy")
    def change(cases, evidence):
        cases[0]["fresh"]["trace"] = evidence["resource_trace"]
        cases[0]["fresh"]["device_id"] = 0
        cases[0]["fresh"]["context_id"] = 42
    result = _qualify(_evidence(), change=change)
    assert result["status"] == "failed"
    assert any("distinct traces" in reason for reason in result["reasons"])


@pytest.mark.parametrize("defect", ["missing_record", "unknown_rate", "foreign_trace",
                                    "fabricated_window", "duplicate"])
def test_consumer_refuses_unbound_or_changed_transfer(defect):
    pytest.importorskip("scipy")
    evidence = _evidence()
    qualification = _qualify(evidence)
    assert qualification["status"] == "passed"
    records = _resource_records(evidence)
    if defect == "missing_record":
        records = records[1:]
    elif defect == "unknown_rate":
        records.append(copy.deepcopy(records[0]))
        records[-1]["rate"] = "TESSERA_BF16_K1_R9999"
        records[-1]["q256"] = 9999
    elif defect == "foreign_trace":
        records.append(copy.deepcopy(records[0]))
        records[-1]["rate"] = "TESSERA_BF16_K1_R900"
        records[-1]["q256"] = 900
    elif defect == "fabricated_window":
        records[0]["window"]["allocation_requests"] = "fabricated"
        records[0]["window"]["transient_peak_bytes"] = -1
    else:
        records.append(copy.deepcopy(records[0]))
    transfer = {"qualification_id": qualification["qualification_id"],
                "runtime_identity": _identity()}
    with pytest.raises(ValueError):
        evidence["mod"].require_transfer(
            transfer, runtime_identity=_identity(), qualifications=[qualification],
            resource_records=records, resource_trace=evidence["resource_trace"],
            device_id=0, context_id=42)


def test_consumer_refuses_a_failed_or_changed_qualification():
    pytest.importorskip("scipy")
    evidence = _evidence()

    def change(cases, _evidence):
        cases[0]["timing"]["collector_started"] = True

    failed = _qualify(evidence, change=change)
    assert failed["status"] == "failed"
    transfer = {"qualification_id": failed["qualification_id"], "runtime_identity": _identity()}
    with pytest.raises(ValueError, match="no passing runtime qualification"):
        evidence["mod"].require_transfer(
            transfer, runtime_identity=_identity(), qualifications=[failed],
            resource_records=_resource_records(evidence),
            resource_trace=evidence["resource_trace"], device_id=0, context_id=42)


def test_qualification_rejects_prepared_identity_drift():
    pytest.importorskip("scipy")
    fields = ["wire_sha256", "wire_record_sha256", "scheme_sha256"]

    def change(cases, evidence):
        case = cases[0]
        case["resource"]["binding"]["operator"][fields[0]] = "e" * 64

    result = _qualify(_evidence(), change=change)
    assert result["status"] == "failed"
    assert any("binding differs between legs" in reason for reason in result["reasons"])


def test_unknown_runtime_packages_cannot_be_substituted():
    pytest.importorskip("scipy")
    evidence = _evidence()
    other = _identity()
    other["vllm_package_sha256"] = "f" * 64
    transfer = {"qualification_id": "0" * 64, "runtime_identity": other}
    with pytest.raises(ValueError, match="runtime identity differs|one known qualification"):
        evidence["mod"].require_transfer(
            transfer, runtime_identity=other, qualifications=[_qualify(evidence)],
            resource_records=_resource_records(evidence),
            resource_trace=evidence["resource_trace"], device_id=0, context_id=42)


def _gate_band(*, k=9, sigma=0.02, spread=1.0, persistent_bias=None, drift=0.0,
               hetero_rate=None, hetero_factor=3.0, eps_samples=None,
               persistent_process_offset=0, fresh_process_tag="p"):
    """Raw FOLLOWUP-7 band built from seeded draws, no cases or traces."""
    import random

    rng = random.Random(20260927)
    rates = list(range(256, 256 + k))
    phases = {}
    persistent_processes = []
    for phase in ("prefill", "decode"):
        fresh = {}
        x = {}
        for i, rate in enumerate(rates):
            legs, logs = [], []
            for rep in range(5):
                noise = rng.gauss(0.0, sigma * (hetero_factor if hetero_rate == i else 1.0))
                logs.append(noise)
                legs.append({"process": {"pid": 1000 + i * 20 + rep,
                                         "boot_id": f"gate-{fresh_process_tag}",
                                         "start_ticks": 1000 + i * 20 + rep},
                             "samples_ms": _base_samples(1.0 + noise)})
            fresh[str(rate)] = legs
            x[i] = logs
        entries = []
        y = []
        for i, rate in enumerate(rates):
            bias = 0.0
            if persistent_bias is not None and i in persistent_bias:
                bias = persistent_bias[i]
            value = drift * i + bias + rng.gauss(0.0, sigma)
            y.append(value)
            entries.append({"q256": rate,
                            "samples_ms": _base_samples(1.0 + value),
                            "time_in_process": i + 1})
        persistent_process = {"pid": 2000 + persistent_process_offset,
                              "boot_id": "gate-persistent", "start_ticks": 2000}
        persistent_processes.append(persistent_process)
        phases[phase] = {"fresh": fresh,
                         "persistent": {"process": persistent_process,
                                        "rates": entries}}
    return {"source": "fresh_process_repeat_r5_pooled_log", "gate_kind": "not_detected",
            "raw": {"eps": {"samples_ms": eps_samples or [1e-4],
                            "eps_source": "gate fixture timer"},
                    "phases": phases}}


def _gate(band, selected):
    return _module()._timing_gate(band, selected)


def test_gate_holds_its_level_on_a_true_null():
    pytest.importorskip("scipy")
    import math

    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02)
    report, reasons = _gate(band, selected)
    assert reasons == [], reasons
    assert report["phases"]["prefill"]["fallback_rates"] == []
    # the common factor is stamped with an interval that covers zero
    assert report["phases"]["prefill"]["d_bar_interval_log"][0] <= 0.0
    assert report["phases"]["prefill"]["d_bar_interval_log"][1] >= 0.0


def test_gate_false_fail_rate_on_a_true_null_is_below_two_percent():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    rng = numpy.random.default_rng(20260927)
    failures = 0
    draws = 2000
    for draw in range(draws):
        rates = list(range(256, 265))
        phases = {}
        for phase in ("prefill", "decode"):
            fresh = {}
            for i, rate in enumerate(rates):
                legs = []
                for rep in range(5):
                    noise = float(rng.normal(0.0, 0.02))
                    legs.append({"process": {"pid": 1000 + i * 20 + rep,
                                             "boot_id": f"mc-{draw}",
                                             "start_ticks": 1000 + i * 20 + rep},
                                 "samples_ms": _base_samples(1.0 + noise)})
                fresh[str(rate)] = legs
            entries = []
            for i, rate in enumerate(rates):
                value = float(rng.normal(0.0, 0.02))
                entries.append({"q256": rate,
                                "samples_ms": _base_samples(1.0 + value),
                                "time_in_process": i + 1})
            phases[phase] = {"fresh": fresh,
                             "persistent": {"process": {"pid": 2000,
                                                        "boot_id": f"mc-{draw}",
                                                        "start_ticks": 2000},
                                            "rates": entries}}
        band = {"source": "fresh_process_repeat_r5_pooled_log",
                "gate_kind": "not_detected",
                "raw": {"eps": {"samples_ms": [1e-4], "eps_source": "mc timer"},
                        "phases": phases}}
        _, reasons = _module()._timing_gate(band, rates)
        if reasons:
            failures += 1
    assert failures / draws < 0.02, failures


@pytest.mark.parametrize("defect", ["empty_phase", "one_rate", "identical_medians",
                                    "eps_above_spread"])
def test_gate_refuses_evidence_it_cannot_resolve(defect):
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9)
    if defect == "empty_phase":
        band["raw"]["phases"]["decode"]["fresh"] = {}
    elif defect == "one_rate":
        for phase in ("prefill", "decode"):
            band["raw"]["phases"][phase]["fresh"] = {
                "256": band["raw"]["phases"][phase]["fresh"]["256"]}
    elif defect == "identical_medians":
        for phase in ("prefill", "decode"):
            for legs in band["raw"]["phases"][phase]["fresh"].values():
                for leg in legs:
                    leg["samples_ms"] = [1.0, 1.0, 1.0]
    else:
        band["raw"]["eps"]["samples_ms"] = [0.5]
    _, reasons = _gate(band, selected)
    assert any("no raw fresh timing evidence" in r or
               "at least two sampled rates" in r or
               "timer_cannot_resolve_process_noise" in r for r in reasons), reasons


def test_gate_refuses_a_single_rate_bias_of_five_process_noise_units():
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, persistent_bias={0: 5 * 0.02})
    _, reasons = _gate(band, selected)
    assert any("residual exceeds the not-detected gate" in r for r in reasons), reasons


def test_gate_refuses_a_monotone_drift_in_time_in_process():
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, drift=0.8 * 0.02)
    _, reasons = _gate(band, selected)
    assert any("persistent_mode_drifts" in r for r in reasons), reasons


def test_gate_screens_a_three_sigma_rate_onto_the_heteroscedastic_fallback():
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, hetero_rate=0)
    report, reasons = _gate(band, selected)
    assert "256" in report["phases"]["prefill"]["fallback_rates"]
    assert report["phases"]["prefill"]["fallback_note"] is not None


def test_gate_refuses_persistent_samples_from_two_processes():
    """Cross-phase persistent identity: one pass-T process names both phases."""
    pytest.importorskip("scipy")
    selected = list(range(256, 265))
    band = _gate_band(k=9, persistent_process_offset=7)
    band["raw"]["phases"]["decode"]["persistent"]["process"] = {
        "pid": 3007, "boot_id": "other", "start_ticks": 3007}
    _, reasons = _gate(band, selected)
    assert any("persistent samples must come from one process" in r for r in reasons)


def test_caller_asserted_band_numbers_are_refused():
    pytest.importorskip("scipy")
    evidence = _evidence()
    band = copy.deepcopy(evidence["band"])
    band["per_rate"] = {"256": {"prefill": [1.0] * 5, "decode": [0.5] * 5}}
    with pytest.raises(ValueError, match="caller-asserted|raw"):
        evidence["mod"].qualify_transfer(evidence["identity"], rates=evidence["rates"],
                                         cases=evidence["cases"], noise_band=band,
                                         resource_trace=evidence["resource_trace"])


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


def test_gate_pins_the_pooled_threshold_and_mde80_closed_forms():
    """threshold/s = t(1-a/2K, k(r-1))*sqrt((1+1/r)(1-1/k)); K = 2k+2.

    REVIEW-658-659-666-r3 Q01 pins 3.9510/4.8202 at k=9 and 4.0052/4.8879
    at k=12. Kills K=2k (3.9131 at k=9), MDE without z_0.8, and the
    d_bar interval read at t(0.975) instead of t(0.995).
    """
    import math
    import statistics as st

    from scipy import stats

    pytest.importorskip("scipy")
    for k, pooled_pin, mde_pin in ((9, 3.9510, 4.8202), (12, 4.0052, 4.8879)):
        selected = list(range(256, 256 + k))
        band = _gate_band(k=k, sigma=0.02)
        report, reasons = _gate(band, selected)
        assert reasons == [], reasons
        phase = report["phases"]["prefill"]
        fallback = set(phase["fallback_rates"])
        # pin a pooled-path rate (at k=12 the seeded draws screen rate 256)
        key = str(next(q for q in selected if str(q) not in fallback))
        s = phase["s_log"]
        m = st.fmean(math.log(v) for v in phase["fresh_medians_ms"][key])
        eps_log = math.log(1.0 + report["eps_ms"] / math.exp(m))
        ratio = (phase["residuals"][key]["threshold_log"] - eps_log) / s
        assert abs(ratio - pooled_pin) < 5e-4, (k, ratio)
        assert abs(phase["mde80_log"] / s - mde_pin) < 5e-4, (k, phase["mde80_log"] / s)
        # d_bar interval half-width is t(0.995, k-1) on the stdev of the d_i
        spread = st.stdev(list(phase["d_log"].values())) / math.sqrt(k)
        hw_pin = float(stats.t.ppf(0.995, k - 1)) * spread
        assert abs((phase["d_bar_interval_log"][1] - phase["d_bar_log"]) - hw_pin) < 1e-12


def _null_gate_state(k=9, sigma=0.02):
    selected = list(range(256, 256 + k))
    report, reasons = _gate(_gate_band(k=k, sigma=sigma), selected)
    assert reasons == [], reasons
    return selected, report


def test_gate_threshold_includes_eps_log_on_the_pooled_path():
    """A residual inside the eps_log margin passes; without eps_log it would
    refuse. Kills dropping +eps_log from the pooled threshold."""
    import math
    import statistics as st

    from scipy import stats

    selected, report = _null_gate_state()
    r, k = 5, 9
    factor = math.sqrt((1.0 + 1.0 / r) * (1.0 - 1.0 / k))
    t_crit = float(stats.t.ppf(1.0 - 0.01 / (2.0 * (2 * k + 2)), k * (r - 1)))
    # a single-rate bias moves its residual by bias*(k-1)/k in BOTH phases,
    # so target the tightest phase for the pass and the loosest for the bite
    key = str(selected[0])
    # the null run's own residual for the target rate rides on top of the
    # bias, so subtract it per phase; a single bias must clear both phases
    pass_bias, bite_bias = [], []
    for name, phase in report["phases"].items():
        m = st.fmean(math.log(v) for v in phase["fresh_medians_ms"][key])
        eps_log = math.log(1.0 + report["eps_ms"] / math.exp(m))
        threshold = t_crit * factor * phase["s_log"] + eps_log
        base = phase["residuals"][key]["d_minus_d_bar"]
        pass_bias.append((threshold - 0.5 * report["eps_ms"] - base) * k / (k - 1))
        bite_bias.append((threshold + 0.6 * report["eps_ms"] - base) * k / (k - 1))
    bias = min(pass_bias)
    band = _gate_band(k=9, sigma=0.02, persistent_bias={0: bias})
    gate, reasons = _gate(band, selected)
    assert reasons == [], reasons
    assert gate["phases"]["prefill"]["residuals"][key]["passed"] is True
    assert gate["phases"]["decode"]["residuals"][key]["passed"] is True
    # and just outside the margin the gate bites in both phases
    bias = max(bite_bias)
    band = _gate_band(k=9, sigma=0.02, persistent_bias={0: bias})
    _, reasons = _gate(band, selected)
    assert any("residual exceeds the not-detected gate" in reason for reason in reasons)


def test_gate_fallback_threshold_uses_the_full_satterthwaite_forms():
    """The fallback variance keeps the sum term and the df is Satterthwaite.
    A residual placed between the dropped-sum threshold and the full one
    passes only when both closed forms are intact. Kills the v_i sum-term
    drop and df_i = r-1."""
    import math
    import statistics as st

    from scipy import stats

    selected = list(range(256, 265))
    r, k = 5, 9
    band = _gate_band(k=9, sigma=0.02, hetero_rate=0)
    report, reasons = _gate(band, selected)
    assert reasons == [], reasons
    phase = report["phases"]["prefill"]
    key = str(selected[0])
    assert key in phase["fallback_rates"]
    logs = {int(q): [math.log(v) for v in vals]
            for q, vals in phase["fresh_medians_ms"].items()}
    first = selected[0]
    s_i = {q: st.stdev(vals) for q, vals in logs.items()}
    v_full = (1.0 + 1.0 / r) * ((1.0 - 1.0 / k) ** 2 * s_i[first] ** 2
                                + sum(s_i[o] ** 2 for o in s_i if o != first) / k ** 2)
    v_dropped = (1.0 + 1.0 / r) * (1.0 - 1.0 / k) ** 2 * s_i[first] ** 2
    c_ii = (1.0 + 1.0 / r) * (1.0 - 1.0 / k) ** 2
    c_ij = (1.0 + 1.0 / r) / k ** 2
    den = ((c_ii * s_i[first] ** 2) ** 2
           + sum((c_ij * s_i[o] ** 2) ** 2 for o in s_i if o != first))
    df_i = (r - 1) * v_full ** 2 / den
    m = st.fmean(logs[first])
    eps_log = math.log(1.0 + report["eps_ms"] / math.exp(m))
    K = 2 * k + 2
    t_full = float(stats.t.ppf(1.0 - 0.01 / (2.0 * K), df_i))
    t_dropped = float(stats.t.ppf(1.0 - 0.01 / (2.0 * K), df_i))
    threshold_full = t_full * math.sqrt(v_full) + eps_log
    threshold_dropped = t_dropped * math.sqrt(v_dropped) + eps_log
    assert threshold_full > threshold_dropped
    # pin the module's own numbers against the closed forms: dropping the
    # sum term moves variance_log, df_i = r-1 moves threshold_log
    assert abs(phase["residuals"][key]["variance_log"] - v_full) < 1e-12
    assert abs(phase["residuals"][key]["threshold_log"] - threshold_full) < 1e-9
    assert 4.0 < df_i < 4.2, df_i


def test_gate_screens_at_alpha_over_k_not_alpha():
    """A variance screen p inside (alpha/k, alpha) keeps the rate on the
    pooled path; screening at alpha would push it to the fallback."""
    selected = list(range(256, 265))
    band = _gate_band(k=9, sigma=0.02, hetero_rate=0, hetero_factor=1.1)
    report, reasons = _gate(band, selected)
    assert reasons == [], reasons
    p = report["phases"]["prefill"]["variance_screen"]["256"]["p"]
    assert 0.01 / 9 < p < 0.01, p
    assert report["phases"]["prefill"]["fallback_rates"] == []


def test_qualify_refuses_a_timing_leg_without_samples():
    evidence = _evidence()

    def change(cases, _e):
        del cases[0]["timing"]["samples_ms"]

    result = _qualify(evidence, change=change)
    assert any("timing samples must carry both phases" in r for r in result["reasons"])


def test_qualify_refuses_a_timing_leg_with_non_dictionary_samples():
    evidence = _evidence()

    def change(cases, _e):
        cases[1]["timing"]["samples_ms"] = "n/a"

    result = _qualify(evidence, change=change)
    assert any("timing samples must carry both phases" in r for r in result["reasons"])


def test_qualify_binds_the_timing_leg_ordinal_to_the_band():
    evidence = _evidence()

    def change(cases, _e):
        cases[0]["timing"]["time_in_process"] = 99

    result = _qualify(evidence, change=change)
    assert any("timing time in process differs from the raw band" in r
               for r in result["reasons"])


def test_qualify_pins_the_timing_process_to_the_band():
    evidence = _evidence()

    def change(cases, _e):
        cases[0]["timing"]["process"] = {"pid": 301, "boot_id": "CPU-fixture",
                                         "start_ticks": 301}

    result = _qualify(evidence, change=change)
    assert any("timing process differs from the raw band" in r for r in result["reasons"])


def test_qualify_refuses_a_relabelled_pass_t_order():
    """D06: relabelling time in_process hides a drift the gate detected; the
    pass-R trace is the authority for run order, and it disagrees here."""
    evidence = _evidence()
    selected = evidence["selected"]

    def band_change(band, _e):
        for phase in PHASES:
            entries = band["raw"]["phases"][phase]["persistent"]["rates"]
            for entry in entries:
                entry["time_in_process"] = len(selected) - selected.index(entry["q256"])

    def change(cases, _e):
        for case in cases:
            case["timing"]["time_in_process"] = len(selected) - selected.index(case["rate"])

    result = _qualify(evidence, change=change, band_change=band_change)
    assert any("persistent timing order disagrees with the pass-R trace" in r
               for r in result["reasons"])


def test_qualify_refuses_fresh_repeats_shared_across_rates():
    evidence = _evidence()

    def band_change(band, _e):
        stolen = copy.deepcopy(
            band["raw"]["phases"]["prefill"]["fresh"]["256"][0]["process"])
        band["raw"]["phases"]["prefill"]["fresh"]["258"][2]["process"] = stolen

    result = _qualify(evidence, band_change=band_change)
    assert any("fresh repeats must be separate processes across rates" in r
               for r in result["reasons"])


def test_qualify_refuses_a_within_rate_duplicate_fresh_repeat():
    evidence = _evidence()

    def band_change(band, _e):
        first = copy.deepcopy(
            band["raw"]["phases"]["decode"]["fresh"]["256"][0]["process"])
        band["raw"]["phases"]["decode"]["fresh"]["256"][1]["process"] = first

    result = _qualify(evidence, band_change=band_change)
    assert any("fresh repeats must be separate processes" in r
               and "across rates" not in r for r in result["reasons"])


def test_qualify_refuses_a_persistent_table_mixing_fresh_rows():
    """A fresh-process row inside the persistent table is a mixed-source
    table: the pass-T evidence must not be timed by a fresh repeat."""
    evidence = _evidence()

    def band_change(band, _e):
        fresh = copy.deepcopy(band["raw"]["phases"]["prefill"]["fresh"]["256"][0]["process"])
        for phase in PHASES:
            band["raw"]["phases"][phase]["persistent"]["process"] = copy.deepcopy(fresh)

    result = _qualify(evidence, band_change=band_change)
    assert any("persistent table mixes fresh-process rows" in r for r in result["reasons"])


def test_qualify_refuses_a_fresh_reference_reused_across_rates():
    evidence = _evidence()

    def change(cases, _e):
        cases[1]["fresh"]["process"] = copy.deepcopy(cases[0]["fresh"]["process"])
        cases[1]["fresh"]["trace"] = copy.deepcopy(cases[0]["fresh"]["trace"])

    result = _qualify(evidence, change=change)
    assert any("fresh reference process was reused across rates" in r
               for r in result["reasons"])


def test_qualify_refuses_a_case_fresh_leg_sharing_a_persistent_process():
    evidence = _evidence()

    def change(cases, _e):
        cases[0]["fresh"]["process"] = {"pid": 300, "boot_id": "CPU-fixture",
                                        "start_ticks": 300}

    result = _qualify(evidence, change=change)
    assert any("fresh reference process was reused across rates" in r
               for r in result["reasons"])


def test_qualify_names_the_floor_for_a_rate_with_one_distinct_median():
    evidence = _evidence()

    def band_change(band, _e):
        leg = band["raw"]["phases"]["prefill"]["fresh"]["258"][0]
        for other in band["raw"]["phases"]["prefill"]["fresh"]["258"]:
            other["samples_ms"] = list(leg["samples_ms"])

    result = _qualify(evidence, band_change=band_change)
    assert any("fewer than two distinct fresh medians" in r for r in result["reasons"])
