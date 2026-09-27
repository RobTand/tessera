"""CPU protocol doubles for pass-R/pass-T runners. No GPU claim is made.

The fakes share one clock and the engine logs its device-byte events into the
collector's trace, so marker-vs-evict ordering is not asserted on call lists
alone: an evict that ran after its end marker leaves the free outside the rate
window and the window itself refuses.
"""
import copy
import json
from contextlib import contextmanager

import pytest

from experiments.native_resource_passes import (
    PASS_R_SCHEMA,
    PASS_T_SCHEMA,
    run_pass_resource,
    run_pass_timing,
)

RATES = ["TESSERA_BF16_K1_R256", "TESSERA_BF16_K1_R512"]


class FakeClock:
    def __init__(self):
        self.now = 1000

    def tick(self, step):
        self.now += step
        return self.now


def _runtime(family="TESSERA_BF16_K1"):
    return {"schema": "tessera.native_resource_transfer_runtime.v1",
            "image_digest": "sha256:" + "a" * 64, "tessera_package_sha256": "b" * 64,
            "vllm_package_sha256": "c" * 64, "nccl_version": [2, 30, 4],
            "family": family, "world_size": 1,
            "native_runtime_sha256": "d" * 64}


class FakeCollector:
    """Sequential marker timestamps; one finish() builds the continuous trace."""

    library_sha256 = "a" * 64

    def __init__(self, clock, engine, *, drift=False, retain=False, stall=False):
        self.clock = clock
        self.engine = engine
        self.marks = []
        self.finished = []
        self.drift = drift
        self.retain = retain
        self.stall = stall
        self.late_free = None

    def mark(self, name):
        point = self.clock.tick(5)
        self.marks.append((name, point))
        if name.endswith(":end") and self.stall:
            return point - 100
        return point

    def finish(self, path):
        self.finished.append(str(path))
        trace = self._trace_dict()
        import json as _json
        import os as _os
        _os.makedirs(_os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            _json.dump(trace, handle)
        return trace

    def _trace_dict(self):
        configuration = ["register_callbacks", "allocation_source", "enable_memory2",
                         "enable_memory_pool", "enable_runtime", "enable_driver",
                         "flush_before_disable", "flush_after_disable"]
        pid = self.engine.pid
        trace = {"schema": "tessera.cupti_memory_trace.v1", "process_id": pid,
                 "cupti_version": 130001, "start_ns": 1, "end_ns": self.clock.now + 1000,
                 "capture": {"started_before_cuda_libraries": True,
                             "collector_library_sha256": self.library_sha256,
                             "start_code": 0, "stop_code": 0},
                 "errors": [], "pool_records": 0, "completed_buffers": 1,
                 "configuration": [{"operation": name, "code": 0} for name in configuration],
                 "dropped_records": [{"code": 0, "count": 0}] * 2,
                 "markers": [], "memory_events": [], "api_events": []}
        for event in self.engine.memory_log:
            if event["kind"] == "init":
                _emit(trace, event["ts"], 1000, 100, "allocate", "nccl-init-fixture", pid)
        def size_for(name):
            q256 = int(name.split("_R", 1)[1].split(":", 1)[0])
            return 40 if q256 < 384 else 60

        begin_ordinal = -1
        for index, (name, point) in enumerate(self.marks):
            recorded = point + 1 if self.drift and name.endswith(":end") else point
            trace["markers"].append({"name": name, "timestamp_ns": recorded})
            if name.endswith(":begin"):
                begin_ordinal += 1
                size = size_for(name)
                _emit(trace, point + 1, 2000, size, "allocate", "torch-fixture", pid)
                if not (self.retain and index == len(self.marks) - 2):
                    _emit(trace, self.engine.freed_at.get(begin_ordinal, point + 3), 2000,
                          size, "free", "", pid)
        return trace


def _emit(trace, ts, address, size, operation, source, pid):
    trace["memory_events"].append({"timestamp_ns": ts, "process_id": pid,
        "device_id": 0, "context_id": 42, "stream_id": 1, "address": address,
        "bytes": size, "correlation_id": ts, "operation": operation,
        "device_memory": True, "memory_kind": 3, "async": False, "pool_type": 0,
        "source": source})
    trace["api_events"].append({"name": "cudaMalloc" if operation == "allocate" else "cudaFree",
        "start_ns": ts - 1, "end_ns": ts + 1, "process_id": pid,
        "correlation_id": ts, "return_value": 0})


class FakeEngine:
    """Two-arg engine protocol double with warmup and device-byte logging."""

    def __init__(self, clock, *, drift=False, late_evict=False, pid=4242):
        self.clock = clock
        self.opened = []
        self.warmups = []
        self.loads = []
        self.primed = []
        self.settles = 0
        self.evicted = []
        self.timed = []
        self.memory_log = []
        self.freed_at = {}
        self.drift = drift
        self.late_evict = late_evict
        self.pid = pid
        self.memory_log.append({"kind": "init", "ts": self.clock.tick(10)})

    @contextmanager
    def family(self, name):
        self.opened.append(name)
        yield

    def warmup(self, rate):
        self.warmups.append(rate)
        payload = self.load(rate)
        prepared = self.prepare(rate, payload)
        self.prime(prepared, payload)
        self.evict(prepared, payload)
        self.settle()

    def load(self, rate):
        self.loads.append(rate)
        return "payload:" + rate

    def prepare(self, rate, payload):
        return {"rate": rate, "payload": payload}

    def prime(self, prepared, payload):
        self.primed.append(prepared["rate"])

    def identity(self, prepared, payload):
        return {"format": prepared["rate"], "wire_sha256": "b" * 64,
                "runtime": {"execution": {"tensor_parallel": 1}},
                "drifted": self.drift}

    def settle(self):
        self.settles += 1

    def evict(self, prepared, payload):
        self.evicted.append(prepared["rate"])
        # The free lands in the trace wherever it happens: before the end
        # marker under a correct runner, after it under a late one.
        index = len([name for name, _ in _collector_of(self).marks
                     if name.endswith(":begin")]) - 1
        ts = self.clock.tick(2)
        if self.late_evict:
            _collector_of(self).late_free = ts
        else:
            self.freed_at[index] = ts

    def process(self):
        return {"pid": self.pid, "boot_id": "fake-pass", "start_ticks": self.pid}

    def time(self, prepared, payload):
        self.timed.append(prepared["rate"])
        scale = 1.0 + (self.pid % 7) * 2e-3
        return {"prefill": [1.0 * scale, 1.1 * scale, 0.9 * scale],
                "decode": [0.4 * scale, 0.5 * scale, 0.6 * scale]}


_COLLECTORS = {}


def _collector_of(engine):
    return _COLLECTORS[id(engine)]


def _pair(*, drift=False, late_evict=False, retain=False, stall=False, pid=4242):
    clock = FakeClock()
    engine = FakeEngine(clock, drift=drift, late_evict=late_evict, pid=pid)
    collector = FakeCollector(clock, engine, drift=drift, retain=retain, stall=stall)
    _COLLECTORS[id(engine)] = collector
    return engine, collector


def test_pass_resource_marks_each_rate_and_derives_windows_from_one_trace(tmp_path):
    engine, collector = _pair()
    records = run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                                trace_path=tmp_path / "trace.json", device_id=0, context_id=42)
    assert [name for name, _ in collector.marks] == [
        "rate:TESSERA_BF16_K1_R256:begin", "rate:TESSERA_BF16_K1_R256:end",
        "rate:TESSERA_BF16_K1_R512:begin", "rate:TESSERA_BF16_K1_R512:end"]
    assert collector.finished == [str(tmp_path / "trace.json")]
    assert engine.opened == ["TESSERA_BF16_K1"]
    # The warm-up is a real load/prepare/prime/evict cycle on the first rate,
    # so it appears in the engine's own logs ahead of every marked rate.
    assert engine.warmups == RATES[:1], "warm-up runs once before the first marker"
    first_begin = collector.marks[0][1]
    assert engine.loads == RATES[:1] + RATES and engine.primed == RATES[:1] + RATES
    assert engine.evicted == RATES[:1] + RATES
    assert engine.settles == len(RATES) + 1
    assert set(records) == set(RATES)
    for size, rate in zip((40, 60), RATES, strict=True):
        record = records[rate]
        assert record["schema"] == PASS_R_SCHEMA and record["status"] == "observed"
        assert record["rate"] == rate
        assert record["q256"] == int(rate.rsplit("_R", 1)[1])
        assert record["collector"] == {"started": True, "library_sha256": "a" * 64}
        assert record["process"] == {"pid": 4242, "boot_id": "fake-pass", "start_ticks": 4242}
        assert record["binding"] == {"format": rate, "wire_sha256": "b" * 64,
                                     "runtime": {"execution": {"tensor_parallel": 1}},
                                     "drifted": False}
        assert record["window"]["allocation_requests"] == [
            {"bytes": size, "source": "torch-fixture", "memory_kind": 3, "count": 1}]
        assert record["window"]["initialization_requests"] == [
            {"bytes": 100, "source": "nccl-init-fixture", "memory_kind": 3, "count": 1}]
        assert record["window"]["begin_ns"] > first_begin - 1


def test_pass_resource_refuses_an_evict_that_runs_after_the_end_marker(tmp_path):
    engine, collector = _pair(late_evict=True)

    def late_finish(path):
        trace = FakeCollector.finish(collector, path)
        # A late evict leaves the free after this rate's end marker but inside
        # the next rate's window (an operation between windows would itself be
        # an unpriced gap): the retention is what the window must refuse. The
        # orphaned API row goes with its memory row, or coverage fails on the
        # API side instead of the retention.
        removed = [row for row in trace["memory_events"]
                   if row["operation"] == "free" and row["bytes"] == 40]
        dead = {row["correlation_id"] for row in removed}
        trace["memory_events"] = [row for row in trace["memory_events"] if row not in removed]
        trace["api_events"] = [api for api in trace["api_events"]
                               if api["correlation_id"] not in dead]
        next_begin = collector.marks[2][1]
        _emit(trace, next_begin, 2000, 40, "free", "", engine.pid)
        return trace

    collector.finish = late_finish
    with pytest.raises(ValueError, match="retained allocations"):
        run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                          trace_path=tmp_path / "t.json", device_id=0, context_id=42)


def test_pass_resource_windows_must_be_this_process_markers(tmp_path):
    engine, collector = _pair(drift=True)
    with pytest.raises(ValueError, match="not this process's"):
        run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                          trace_path=tmp_path / "t.json", device_id=0, context_id=42)


def test_pass_resource_refuses_a_trace_that_retains_rate_memory(tmp_path):
    engine, collector = _pair(retain=True)
    with pytest.raises(ValueError, match="retained allocations"):
        run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                          trace_path=tmp_path / "t.json", device_id=0, context_id=42)


def test_pass_resource_refuses_an_end_marker_at_or_before_its_begin(tmp_path):
    engine, collector = _pair(stall=True)
    with pytest.raises(ValueError, match="does not follow"):
        run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                          trace_path=tmp_path / "t.json", device_id=0, context_id=42)


def test_pass_resource_still_flushes_the_trace_when_a_rate_raises(tmp_path):
    engine, collector = _pair()
    original_mark = collector.mark

    def exploding_mark(name):
        if name.endswith(":end"):
            raise RuntimeError("boom")
        return original_mark(name)

    collector.mark = exploding_mark
    with pytest.raises(RuntimeError, match="boom"):
        run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                          trace_path=tmp_path / "t.json", device_id=0, context_id=42)
    assert collector.finished, "finish() ran despite the mid-loop failure"


def test_pass_timing_never_collects_and_binds_the_pass_r_identity(tmp_path):
    engine, collector = _pair()
    resource = run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                                 trace_path=tmp_path / "t.json", device_id=0, context_id=42)
    timing_engine, _ = _pair()
    timing_engine.pid = 9191
    records = run_pass_timing(RATES, timing_engine, resource, runtime_identity=_runtime())
    assert timing_engine.opened == ["TESSERA_BF16_K1"] and timing_engine.timed == RATES
    assert timing_engine.warmups == []
    assert set(records) == set(RATES)
    for rate in RATES:
        record = records[rate]
        assert record["schema"] == PASS_T_SCHEMA and record["status"] == "observed"
        assert record["collector_started"] is False
        assert record["binding"] == resource[rate]["binding"]
        assert record["process"]["pid"] == 9191 != resource[rate]["process"]["pid"]
        assert record["samples_ms"]["prefill"] == [1.0, 1.1, 0.9]


def test_pass_timing_refuses_identity_drift_against_the_pass_r_record(tmp_path):
    engine, collector = _pair()
    resource = run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                                 trace_path=tmp_path / "t.json", device_id=0, context_id=42)
    drift_engine, _ = _pair(drift=True)
    with pytest.raises(ValueError, match="differs from its pass-R record"):
        run_pass_timing(RATES, drift_engine, resource, runtime_identity=_runtime())


def test_pass_timing_refuses_foreign_runtime_family_world_or_record(tmp_path):
    engine, collector = _pair()
    resource = run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                                 trace_path=tmp_path / "t.json", device_id=0, context_id=42)
    world2 = _runtime()
    world2["world_size"] = 2
    with pytest.raises(ValueError, match="different runtime identity|not an observed"):
        run_pass_timing(RATES, _pair()[0], resource, runtime_identity=world2)
    unobserved = copy.deepcopy(resource)
    unobserved[RATES[0]]["status"] = "replayed"
    with pytest.raises(ValueError, match="not an observed"):
        run_pass_timing(RATES, _pair()[0], unobserved, runtime_identity=_runtime())
    rekeyed = copy.deepcopy(resource)
    rekeyed[RATES[0]]["rate"] = "TESSERA_BF16_K1_R999"
    with pytest.raises(ValueError, match="not an observed"):
        run_pass_timing(RATES, _pair()[0], rekeyed, runtime_identity=_runtime())
    with pytest.raises(ValueError, match="must key every roster rate exactly"):
        run_pass_timing(RATES, _pair()[0], {"schema": json.dumps("not records")},
                        runtime_identity=_runtime())


def test_passes_refuse_a_roster_outside_the_runtime_family(tmp_path):
    engine, collector = _pair()
    with pytest.raises(ValueError, match="runtime identity family"):
        run_pass_resource(["TESSERA_E4M3_K1_R256"], engine, collector,
                          runtime_identity=_runtime(), trace_path=tmp_path / "t.json",
                          device_id=0, context_id=42)
    with pytest.raises(ValueError, match="runtime identity family"):
        run_pass_timing(["TESSERA_E4M3_K1_R256"], _pair()[0],
                        {"TESSERA_E4M3_K1_R256": {}}, runtime_identity=_runtime())


@pytest.mark.parametrize("rates", [
    ["TESSERA_BF16_K1_R256", "TESSERA_E4M3_K1_R256"],
    ["TESSERA_BF16_K1_R256", "TESSERA_BF16_K1_R256"],
    ["BF16_R256"],
    [],
])
def test_passes_require_one_nonrepeating_family_roster(rates, tmp_path):
    engine, collector = _pair()
    with pytest.raises(ValueError, match="exactly one valid format family|"
                       "nonempty roster|runtime identity family"):
        run_pass_resource(rates, engine, collector, runtime_identity=_runtime(),
                          trace_path=tmp_path / "t.json", device_id=0, context_id=42)
    with pytest.raises(ValueError, match="exactly one valid format family|"
                       "nonempty roster|runtime identity family"):
        run_pass_timing(rates, _pair()[0], {"any": "records"},
                        runtime_identity=_runtime())


def test_pass_resource_refusal_still_flushes_the_one_trace(tmp_path):
    """Round-2 P11: validation inside the try, so finish() keeps its trace.

    A roster refusal must not skip the collector's stop: the trace is the
    process's one durable record, and losing it to a validation raise leaves
    the collector half-open with no evidence of what ran.
    """
    engine, collector = _pair()
    with pytest.raises(ValueError, match="runtime identity family"):
        run_pass_resource(["TESSERA_E4M3_K1_R256"], engine, collector,
                          runtime_identity=_runtime(),
                          trace_path=tmp_path / "trace.json", device_id=0, context_id=42)
    assert collector.finished == [str(tmp_path / "trace.json")]
    assert engine.opened == [], "a refused roster never enters the family context"


def test_pass_timing_refuses_a_record_without_its_observed_window(tmp_path):
    """Round-2 P13: pass T consumes pass-R records; the window is not optional."""
    engine, collector = _pair()
    records = run_pass_resource(RATES, engine, collector, runtime_identity=_runtime(),
                                trace_path=tmp_path / "t.json", device_id=0, context_id=42)
    damaged = copy.deepcopy(records)
    del damaged[RATES[0]]["window"]
    damaged[RATES[1]]["window"] = {"status": "unobserved", "interval": f"rate:{RATES[1]}"}
    with pytest.raises(ValueError, match="no observed window"):
        run_pass_timing(RATES, _pair()[0], damaged, runtime_identity=_runtime())
    relabelled = copy.deepcopy(records)
    relabelled[RATES[0]]["window"] = {**records[RATES[0]]["window"],
                                      "interval": f"rate:{RATES[1]}"}
    with pytest.raises(ValueError, match="no observed window"):
        run_pass_timing(RATES, _pair()[0], relabelled, runtime_identity=_runtime())


def test_runner_records_feed_qualify_transfer_without_adaptation(tmp_path):
    pytest.importorskip("scipy")
    """Round-1 #659-3: the records' own shapes ARE the qualification cases.

    Pass-R records, pass-T records and fresh legs produced by THESE runners
    (fresh windows are single-rate pass-R runs, fresh timing single-rate
    pass-T runs) assemble into qualify_transfer cases with no adapter layer,
    the raw FOLLOWUP-7 band assembles from the same records, and the
    qualification passes. This is the end-to-end contract between the
    runners and the transfer authority.
    """
    import json

    from experiments.native_resource_transfer import qualify_transfer

    domain = [256, 257, 258, 259, 260]
    rates = [f"TESSERA_BF16_K1_R{q256}" for q256 in domain]
    identity = _runtime()
    paths = iter(range(10_000))
    phases = ("prefill", "decode")

    def one_pass(roster, pid, trace_name):
        engine, collector = _pair(pid=pid)
        trace = tmp_path / f"{trace_name}{next(paths)}.json"
        records = run_pass_resource(roster, engine, collector, runtime_identity=identity,
                                    trace_path=trace, device_id=0, context_id=42)
        return records, json.loads(trace.read_text())

    resource, resource_trace = one_pass(rates, pid=100, trace_name="resource")
    timing_engine, _ = _pair(pid=200)
    timing = run_pass_timing(rates, timing_engine, resource, runtime_identity=identity)

    cases, fresh_raw = [], {phase: {} for phase in phases}
    persistent_raw = {phase: {"process": None, "rates": []} for phase in phases}
    for index, rate in enumerate(rates):
        fresh_records, fresh_trace = one_pass([rate], pid=300 + index,
                                              trace_name="fresh")
        record = fresh_records[rate]
        reps = []
        for rep in range(5):
            rep_engine, _ = _pair(pid=400 + index * 8 + rep)
            rep_records = run_pass_timing([rate], rep_engine,
                                          {rate: record}, runtime_identity=identity)
            reps.append(rep_records[rate])
        for phase in phases:
            fresh_raw[phase][str(domain[index])] = [
                {"process": rep["process"], "samples_ms": rep["samples_ms"][phase]}
                for rep in reps]
        for phase in phases:
            persistent = persistent_raw[phase]
            if persistent["process"] is None:
                persistent["process"] = timing[rate]["process"]
            persistent["rates"].append({"q256": timing[rate]["q256"],
                                        "samples_ms": timing[rate]["samples_ms"][phase],
                                        "time_in_process": timing[rate]["time_in_process"]})
        cases.append({"rate": domain[index], "cut_axis": None,
                      "fresh": {"trace": fresh_trace,
                                "device_id": record["device_id"],
                                "context_id": record["context_id"],
                                "interval": f"rate:{rate}",
                                "binding": record["binding"],
                                "process": record["process"]},
                      "resource": {"interval": f"rate:{rate}",
                                   "device_id": resource[rate]["device_id"],
                                   "context_id": resource[rate]["context_id"],
                                   "binding": resource[rate]["binding"],
                                   "process": resource[rate]["process"]},
                      "timing": {"samples_ms": timing[rate]["samples_ms"],
                                 "collector_started": timing[rate]["collector_started"],
                                 "binding": timing[rate]["binding"],
                                 "process": timing[rate]["process"],
                                 "time_in_process": timing[rate]["time_in_process"]}})
    band = {"source": "fresh_process_repeat_r5_pooled_log", "gate_kind": "not_detected",
            "raw": {"eps": {"samples_ms": [0.001], "eps_source": "test fixture timer"},
                    "phases": {phase: {"fresh": fresh_raw[phase],
                                       "persistent": persistent_raw[phase]}
                               for phase in phases}}}
    qualification = qualify_transfer(identity, rates=domain, cases=cases,
                                     noise_band=band, resource_trace=resource_trace)
    assert qualification["status"] == "passed", qualification["reasons"]
