"""Measure row wall time and whole-device kernel-resident time from NVML samples.

utilization.gpu is the percent of a sample interval with a resident kernel. It
is not SM throughput. Integrate adjacent samples by the trapezoidal rule; keep
sample coverage and errors so an incomplete measurement cannot claim zero use.
The row owns the Flux GPU slot. Power and memory samples are retained as well.
"""
from __future__ import annotations

from contextlib import contextmanager
import resource
import subprocess
import threading
import time


def summarize(samples, start, end, errors=()):
    samples = [s for s in samples if start <= s["t"] <= end]
    busy = sum((b["t"] - a["t"]) * (a["busy_percent"] + b["busy_percent"]) / 200 for a, b in zip(samples, samples[1:]))
    return {"wall_minutes": (end - start) / 60, "gpu_busy_minutes": busy / 60 if len(samples) > 1 else None,
            "method": "whole-device NVML kernel-resident percent; sampled trapezoidal integral; not SM throughput",
            "sample_interval_seconds": 1, "sampled_seconds": samples[-1]["t"] - samples[0]["t"] if len(samples) > 1 else 0,
            "peak_device_memory_mib": max((s["memory_mib"] for s in samples), default=None),
            "samples": samples, "errors": list(errors)}


class PhaseTimes:
    """Accumulate exclusive monotonic wall seconds, including exceptional exits."""
    def __init__(self, row):
        self.seconds = row["phase_wall_seconds"] = dict.fromkeys(("setup", "identity_audit", "gate_cases", "diagnostics"), 0.0)
        self.current, self.since = None, time.monotonic()

    def change(self, name):
        now = time.monotonic()
        if self.current is not None:
            self.seconds[self.current] += now - self.since
        self.current, self.since = name, now

    def __enter__(self):
        self.change("setup")
        return self

    def __exit__(self, *error):
        self.change(None)

    @contextmanager
    def measure(self, name):
        previous = self.current
        self.change(name)
        try:
            yield
        finally:
            self.change(previous)


class RowMeter:
    def __init__(self):
        self.start = time.monotonic()
        self.samples, self.errors = [], []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while True:
            try:
                line = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,power.draw,memory.used",
                                                "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                # One Flux slot must expose one device. Do not sum unrelated GPUs.
                if len(line.splitlines()) != 1:
                    raise ValueError("row telemetry needs exactly one visible device")
                uuid, busy, power, memory = (v.strip() for v in line.split(","))
                reading = {"t": time.monotonic(), "uuid": uuid, "busy_percent": float(busy), "power_watts": float(power), "memory_mib": float(memory)}
                if not 0 <= reading["busy_percent"] <= 100 or self.samples and uuid != self.samples[0]["uuid"]:
                    raise ValueError("invalid or changed GPU telemetry device")
                self.samples.append(reading)
            except Exception as error:
                self.errors.append(str(error))
                break
            if self.done.wait(1):
                break

    def finish(self):
        self.done.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            self.errors.append("GPU telemetry did not stop within its command timeout")
        result = summarize(self.samples, self.start, time.monotonic(), self.errors)
        result["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return result
