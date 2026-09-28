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
    return {"schema": "tessera.native_resource_identity.v1",
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


def test_run_report_binds_both_passes_to_the_single_trace(tmp_path):
    """The published run report binds pass-R windows, pass-T samples and the one trace.

    Round-1 #659 follow-up for the re-home: the runners' records assemble into
    a ``tessera.native_persistent_run.v1`` report with no adapter layer. Every
    window in the report names the report's trace digest, every pass-T record
    disclaims a collector, and the report refuses a roster mismatch.
    """
    from experiments.native_resource_passes import assemble_run_report
    from experiments.native_resource_trace import digest

    domain = [256, 257, 258]
    rates = [f"TESSERA_BF16_K1_R{q256}" for q256 in domain]
    identity = _runtime()
    engine, collector = _pair(pid=100)
    trace = tmp_path / "resource_trace.json"
    resource = run_pass_resource(rates, engine, collector, runtime_identity=identity,
                                 trace_path=trace, device_id=0, context_id=42)
    timing_engine, _ = _pair(pid=200)
    timing = run_pass_timing(rates, timing_engine, resource, runtime_identity=identity)
    report = assemble_run_report(resource, timing, trace_path=trace)
    assert report["schema"] == "tessera.native_persistent_run.v1"
    assert report["schema_version"] == 1
    import hashlib
    import json as _json
    with open(trace, "rb") as handle:
        file_sha256 = hashlib.sha256(handle.read()).hexdigest()
    assert report["trace"]["file_sha256"] == file_sha256
    with open(trace) as handle:
        assert report["trace"]["sha256"] == digest(_json.load(handle))
    assert set(report["pass_r"]) == set(rates) == set(report["pass_t"])
    for rate in rates:
        window = report["pass_r"][rate]["window"]
        assert window["trace_sha256"] == report["trace"]["sha256"]
        assert window["interval"] == f"rate:{rate}"
        assert report["pass_t"][rate]["collector_started"] is False
        assert report["pass_t"][rate]["samples_ms"].keys() == {"prefill", "decode"}


def test_run_report_refuses_roster_or_trace_disagreement(tmp_path):
    from experiments.native_resource_passes import assemble_run_report

    rates = ["TESSERA_BF16_K1_R256", "TESSERA_BF16_K1_R257"]
    identity = _runtime()
    engine, collector = _pair(pid=100)
    trace = tmp_path / "trace.json"
    resource = run_pass_resource(rates, engine, collector, runtime_identity=identity,
                                 trace_path=trace, device_id=0, context_id=42)
    timing_engine, _ = _pair(pid=200)
    timing = run_pass_timing(rates, timing_engine, resource, runtime_identity=identity)
    with pytest.raises(ValueError, match="same roster"):
        assemble_run_report({rates[0]: resource[rates[0]]},
                            {rates[1]: timing[rates[1]]}, trace_path=trace)
    with pytest.raises(ValueError, match="started collector"):
        polluted = {rate: dict(record, collector_started=True)
                    for rate, record in timing.items()}
        assemble_run_report(resource, polluted, trace_path=trace)
    with pytest.raises(ValueError, match="more than one trace"):
        tampered = {rate: dict(record) for rate, record in resource.items()}
        tampered[rates[0]]["window"] = dict(tampered[rates[0]]["window"],
                                            trace_sha256="0" * 64)
        assemble_run_report(tampered, timing, trace_path=trace)


def _cli(tmp_path, *argv, pid=None):
    import os
    import subprocess
    import sys as _sys
    from pathlib import Path as _Path
    env = dict(os.environ)
    if pid is not None:
        env["PASSES_FAKE_PID"] = str(pid)
    return subprocess.run(
        [_sys.executable, "-m", "experiments.native_resource_passes", *argv],
        capture_output=True, text=True, env=env, cwd=str(_Path(__file__).resolve().parents[1]))


def _identity_file(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(_runtime()))
    return path


def test_cli_produces_the_run_report_end_to_end(tmp_path):
    """pass-r -> pass-t -> assemble as subprocesses; paths in, paths out."""
    identity = _identity_file(tmp_path)
    rates = "TESSERA_BF16_K1_R256,TESSERA_BF16_K1_R257"
    engine = "tests._native_passes_cli_fakes:engine_factory"
    collector = "tests._native_passes_cli_fakes:collector_factory"
    first = _cli(tmp_path, "pass-r", "--rates", rates, "--engine", engine,
                 "--collector", collector, "--fixtures", str(tmp_path / "unused.json"),
                 "--collector-library", str(tmp_path / "lib.so"),
                 "--runtime-identity", str(identity),
                 "--trace", str(tmp_path / "trace.json"),
                 "--out", str(tmp_path / "resource.json"),
                 "--device-id", "0", "--context-id", "42", pid=100)
    assert first.returncode == 0, first.stderr
    second = _cli(tmp_path, "pass-t", "--rates", rates, "--engine", engine,
                  "--fixtures", str(tmp_path / "unused.json"),
                  "--records", str(tmp_path / "resource.json"),
                  "--runtime-identity", str(identity),
                  "--out", str(tmp_path / "timing.json"), pid=1100)
    assert second.returncode == 0, second.stderr
    third = _cli(tmp_path, "assemble",
                 "--resource", str(tmp_path / "resource.json"),
                 "--timing", str(tmp_path / "timing.json"),
                 "--trace", str(tmp_path / "trace.json"),
                 "--out", str(tmp_path / "report.json"))
    assert third.returncode == 0, third.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["schema"] == "tessera.native_persistent_run.v1"
    assert set(report["pass_r"]) == {"TESSERA_BF16_K1_R256", "TESSERA_BF16_K1_R257"}
    resource = json.loads((tmp_path / "resource.json").read_text())
    assert resource["TESSERA_BF16_K1_R256"]["window"]["trace_sha256"] == \
        report["trace"]["sha256"]


def test_cli_one_rate_roster_is_the_fresh_report_producer(tmp_path):
    identity = _identity_file(tmp_path)
    engine = "tests._native_passes_cli_fakes:engine_factory"
    collector = "tests._native_passes_cli_fakes:collector_factory"
    run = _cli(tmp_path, "pass-r", "--rates", "TESSERA_BF16_K1_R256",
               "--engine", engine, "--collector", collector,
               "--fixtures", str(tmp_path / "unused.json"),
               "--collector-library", str(tmp_path / "lib.so"),
               "--runtime-identity", str(identity),
               "--trace", str(tmp_path / "f.json"),
               "--out", str(tmp_path / "fr.json"),
               "--device-id", "0", "--context-id", "42", pid=300)
    assert run.returncode == 0, run.stderr
    timing = _cli(tmp_path, "pass-t", "--rates", "TESSERA_BF16_K1_R256",
                  "--engine", engine, "--fixtures", str(tmp_path / "unused.json"),
                  "--records", str(tmp_path / "fr.json"),
                  "--runtime-identity", str(identity),
                  "--out", str(tmp_path / "ft.json"), pid=1300)
    assert timing.returncode == 0, timing.stderr
    fresh = _cli(tmp_path, "assemble", "--resource", str(tmp_path / "fr.json"),
                 "--timing", str(tmp_path / "ft.json"),
                 "--trace", str(tmp_path / "f.json"),
                 "--out", str(tmp_path / "fresh.json"))
    assert fresh.returncode == 0, fresh.stderr
    report = json.loads((tmp_path / "fresh.json").read_text())
    assert len(report["pass_r"]) == 1 and len(report["pass_t"]) == 1


def test_cli_refusals_exit_nonzero_with_a_named_error(tmp_path):
    identity = _identity_file(tmp_path)
    engine = "tests._native_passes_cli_fakes:engine_factory"
    collector = "tests._native_passes_cli_fakes:collector_factory"
    bad_roster = _cli(tmp_path, "pass-r", "--rates", "TESSERA_E4M3_K1_R256,TESSERA_BF16_K1_R257",
                      "--engine", engine, "--collector", collector,
                      "--fixtures", str(tmp_path / "unused.json"),
                      "--collector-library", str(tmp_path / "lib.so"),
                      "--runtime-identity", str(identity),
                      "--trace", str(tmp_path / "t.json"),
                      "--out", str(tmp_path / "r.json"),
                      "--device-id", "0", "--context-id", "42")
    assert bad_roster.returncode == 2
    assert "belong to the runtime identity family" in bad_roster.stderr
    missing = _cli(tmp_path, "assemble", "--resource", str(tmp_path / "nope.json"),
                   "--timing", str(tmp_path / "nope2.json"),
                   "--trace", str(tmp_path / "nope3.json"),
                   "--out", str(tmp_path / "o.json"))
    assert missing.returncode == 2
    assert "cannot read" in missing.stderr
    bad_spec = _cli(tmp_path, "pass-t", "--rates", "TESSERA_BF16_K1_R256",
                    "--engine", "tests._native_passes_cli_fakes",
                    "--fixtures", str(tmp_path / "unused.json"),
                    "--records", str(tmp_path / "x.json"),
                    "--runtime-identity", str(identity),
                    "--out", str(tmp_path / "y.json"))
    assert bad_spec.returncode == 2
    assert "module:attr" in bad_spec.stderr


def test_cli_default_wiring_builds_collector_before_engine(monkeypatch, tmp_path):
    """The production path resolves the real classes, collector first."""
    import experiments.native_resource_passes as producer

    order = []
    identity = _runtime()
    identity["tessera_package_sha256"] = "1" * 64
    identity["vllm_package_sha256"] = "2" * 64
    identity["native_runtime_sha256"] = "3" * 64

    class RecorderCollector:
        library_sha256 = "a" * 64
        ticks = 100

        def __init__(self, library):
            order.append(("collector", str(library)))

        def mark(self, name):
            RecorderCollector.ticks += 5
            return RecorderCollector.ticks

        def finish(self, path):
            order.append(("finish", str(path)))
            raise ValueError("collector trace refusal fixture")

    class RecorderEngine:
        def __init__(self, fixtures):
            order.append(("engine", sorted(fixtures)))

        def family(self, name):
            from contextlib import nullcontext
            return nullcontext()

        def warmup(self, rate):
            pass

        def load(self, rate):
            return rate

        def prepare(self, rate, payload):
            return rate

        def prime(self, prepared, payload):
            pass

        def identity(self, prepared, payload):
            return {"format": prepared}

        def settle(self):
            pass

        def evict(self, prepared, payload):
            pass

        def process(self):
            return {"pid": 1, "boot_id": "b", "start_ticks": 1}

        def time(self, prepared, payload):
            return {"prefill": [1.0, 1.1, 0.9], "decode": [0.4, 0.5, 0.6]}

    # a trace the ContinuousTrace validation will refuse is fine: the test
    # asserts construction ORDER, so we stop at the first window derivation
    # by letting finish() return an invalid trace and expecting the refusal.
    monkeypatch.setattr(producer, "_load_entry", lambda spec: (_ for _ in ()).throw(
        AssertionError("test-only override must not run on the default path")))
    import experiments.native_operator_resources as resources
    import experiments.native_transfer_engine as engine_module
    monkeypatch.setattr(resources, "NativeMemoryCollector", RecorderCollector)
    monkeypatch.setattr(engine_module, "NativeTransferEngine", RecorderEngine)

    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({"TESSERA_BF16_K1_R256": str(tmp_path)}))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="trace refusal fixture"):
        producer._main(["pass-r", "--rates", "TESSERA_BF16_K1_R256",
                        "--fixtures", str(fixtures),
                        "--collector-library", str(tmp_path / "lib.so"),
                        "--runtime-identity", str(identity_path),
                        "--trace", str(tmp_path / "t.json"),
                        "--out", str(tmp_path / "r.json"),
                        "--device-id", "0", "--context-id", "42"])
    kinds = [kind for kind, _ in order[:2]]
    assert kinds == ["collector", "engine"]


def test_run_report_refuses_a_null_collector_started(tmp_path):
    from experiments.native_resource_passes import assemble_run_report

    rates = ["TESSERA_BF16_K1_R256"]
    identity = _runtime()
    engine, collector = _pair(pid=100)
    trace = tmp_path / "trace.json"
    resource = run_pass_resource(rates, engine, collector, runtime_identity=identity,
                                 trace_path=trace, device_id=0, context_id=42)
    timing_engine, _ = _pair(pid=200)
    timing = run_pass_timing(rates, timing_engine, resource, runtime_identity=identity)
    nulled = {rate: dict(record, collector_started=None)
              for rate, record in timing.items()}
    with pytest.raises(ValueError, match="started collector"):
        assemble_run_report(resource, nulled, trace_path=trace)
