"""Self-contained protocol doubles for the producer CLI subprocess tests.

``experiments/native_resource_passes`` exposes a subprocess entry point
(pass-r / pass-t / assemble). Real engines bind the native bench; these
doubles implement the same engine/collector protocol on the CPU so the CLI
contract itself is testable without any device. The engine factory reads
``PASSES_FAKE_PID`` so a test can give each process a distinct identity.
"""
from __future__ import annotations

import os
from contextlib import contextmanager


class _Clock:
    def __init__(self):
        self.now = 100

    def tick(self, step):
        self.now += step
        return self.now


class FakeEngine:
    """Two-arg engine protocol double with warmup and device-byte logging."""

    def __init__(self, clock, *, pid=None):
        self.clock = clock
        self.pid = int(os.environ.get("PASSES_FAKE_PID", "4242"))
        self.opened = []
        self.warmups = []
        self.loads = []
        self.primed = []
        self.settles = 0
        self.evicted = []
        self.timed = []
        self.memory_log = [{"kind": "init", "ts": self.clock.tick(10)}]
        self.freed_at = {}
        self.collector = None  # bound by FakeCollector at construction

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
                "runtime": {"execution": {"tensor_parallel": 1}}}

    def settle(self):
        self.settles += 1

    def evict(self, prepared, payload):
        # The paired free is emitted by the collector just after the rate's
        # begin marker (point + 4), deterministically inside the window.
        self.evicted.append(prepared["rate"])

    def process(self):
        return {"pid": self.pid, "boot_id": "fake-pass", "start_ticks": self.pid}

    def time(self, prepared, payload):
        self.timed.append(prepared["rate"])
        scale = 1.0 + (self.pid % 7) * 2e-3
        return {"prefill": [1.0 * scale, 1.1 * scale, 0.9 * scale],
                "decode": [0.4 * scale, 0.5 * scale, 0.6 * scale]}


class FakeCollector:
    """Sequential marker timestamps; one finish() builds the continuous trace."""

    library_sha256 = "a" * 64

    def __init__(self, engine):
        self.engine = engine
        self.marks = []
        engine.collector = self

    def mark(self, name):
        point = self.engine.clock.tick(5)
        self.marks.append((name, point))
        return point

    def finish(self, path):
        import json
        import os
        trace = self._trace_dict()
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(trace, handle)
        return trace

    def _trace_dict(self):
        configuration = ["register_callbacks", "allocation_source", "enable_memory2",
                         "enable_memory_pool", "enable_runtime", "enable_driver",
                         "flush_before_disable", "flush_after_disable"]
        pid = self.engine.pid
        trace = {"schema": "tessera.cupti_memory_trace.v1", "process_id": pid,
                 "cupti_version": 130001, "start_ns": 1,
                 "end_ns": self.engine.clock.now + 1000,
                 "capture": {"started_before_cuda_libraries": True,
                             "collector_library_sha256": self.library_sha256,
                             "start_code": 0, "stop_code": 0},
                 "errors": [], "pool_records": 0, "completed_buffers": 1,
                 "configuration": [{"operation": name, "code": 0} for name in configuration],
                 "dropped_records": [{"code": 0, "count": 0}] * 2,
                 "markers": [], "memory_events": [], "api_events": []}
        for event in self.engine.memory_log:
            _emit(trace, event["ts"], 1000, 100, "allocate", "nccl-init-fixture", pid)
        for name, point in self.marks:
            trace["markers"].append({"name": name, "timestamp_ns": point})
            if name.endswith(":begin"):
                q256 = int(name.split("_R", 1)[1].split(":", 1)[0])
                size = 40 if q256 < 384 else 60
                ordinal = len([n for n, _ in self.marks if n.endswith(":begin")]) - 1
                _emit(trace, point + 1, 2000, size, "allocate", "torch-fixture", pid)
                _emit(trace, point + 4, 2000, size, "free", "", pid)
                del ordinal
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


def engine_factory():
    return FakeEngine(_Clock())


def collector_factory(engine):
    return FakeCollector(engine)
