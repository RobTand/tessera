"""Measure row wall time and whole-device kernel-resident time from NVML samples.

utilization.gpu is the percent of a sample interval with a resident kernel. It
is not SM throughput. Integrate adjacent samples by the trapezoidal rule; keep
sample coverage and errors so an incomplete measurement cannot claim zero use.
The row owns the Flux GPU slot. Power and memory samples are retained as well.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import resource
import subprocess
import threading
import time


def number(value):
    """NVML fields such as GB10 memory.used can be unavailable independently."""
    value = value.strip()
    return None if value in ("[N/A]", "N/A", "[Not Supported]") else float(value)


def gpu_reading(line):
    if len(line.strip().splitlines()) != 1:
        raise ValueError("row telemetry needs exactly one visible device")
    uuid, busy, power, memory = (v.strip() for v in line.split(","))
    reading = {"uuid": uuid, "busy_percent": number(busy), "power_watts": number(power), "memory_mib": number(memory)}
    if any(v is not None and not math.isfinite(v) for k, v in reading.items() if k != "uuid") or (
            reading["busy_percent"] is not None and not 0 <= reading["busy_percent"] <= 100):
        raise ValueError("invalid GPU telemetry value")
    return reading


def compute_memory(text):
    values = [number(line) for line in text.splitlines() if line.strip()]
    return sum(values) if all(v is not None and math.isfinite(v) and v >= 0 for v in values) else None


def summarize(samples, start, end, errors=()):
    samples = [s for s in samples if start <= s["t"] <= end]
    intervals = [(a, b) for a, b in zip(samples, samples[1:]) if a["busy_percent"] is not None and b["busy_percent"] is not None]
    busy = sum((b["t"] - a["t"]) * (a["busy_percent"] + b["busy_percent"]) / 200 for a, b in intervals)
    power = [s["power_watts"] for s in samples if s.get("power_watts") is not None]
    memory = [s["compute_apps_memory_mib"] for s in samples if s.get("compute_apps_memory_mib") is not None]
    return {"wall_minutes": (end - start) / 60, "gpu_busy_minutes": busy / 60 if intervals else None,
            "method": "whole-device NVML kernel-resident percent; sampled trapezoidal integral; not SM throughput",
            "sample_interval_seconds": 1, "sampled_seconds": samples[-1]["t"] - samples[0]["t"] if len(samples) > 1 else 0,
            "busy_sampled_seconds": sum(b["t"] - a["t"] for a, b in intervals),
            "peak_device_memory_mib": max(memory, default=None), "device_memory_method": "sum of NVML compute-apps used_memory",
            "peak_power_watts": max(power, default=None), "mean_sample_power_watts": sum(power) / len(power) if power else None,
            "power_envelope_watts": 140,
            "samples": samples, "errors": list(errors)}


class PhaseTimes:
    """Accumulate exclusive monotonic wall seconds, including exceptional exits."""
    def __init__(self, row, meter=None):
        self.meter = meter
        self.seconds = row["phase_wall_seconds"] = dict.fromkeys(("setup", "identity_audit", "gate_cases", "diagnostics"), 0.0)
        self.current, self.since = None, time.monotonic()

    def change(self, name):
        now = time.monotonic()
        if self.current is not None:
            self.seconds[self.current] += now - self.since
        self.current, self.since = name, now
        if self.meter is not None:
            self.meter.phase = name or "finalization"

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
        self.phase = "setup"
        self.samples, self.errors = [], []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def query(self, query):
        return subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=5)

    def sample(self):
        while True:
            try:
                reading = gpu_reading(self.query("--query-gpu=uuid,utilization.gpu,power.draw,memory.used"))
                if self.samples and reading["uuid"] != self.samples[0]["uuid"]:
                    raise ValueError("changed GPU telemetry device")
                reading.update(t=time.monotonic(), compute_apps_memory_mib=None, phase=getattr(self, "phase", "setup"))
                try:
                    reading["compute_apps_memory_mib"] = compute_memory(self.query("--query-compute-apps=used_memory"))
                except Exception as error:
                    self.errors.append(str(error))
                self.samples.append(reading)
            except Exception as error:
                self.errors.append(str(error))
            if self.done.wait(1):
                break

    def snapshot(self, *, include_samples=True):
        end = time.monotonic()
        samples = list(self.samples)
        result = summarize(samples, self.start, end, self.errors)
        result["phases"] = {}
        for phase in sorted({s.get("phase", "setup") for s in samples}):
            pairs = [(a, b) for a, b in zip(samples, samples[1:]) if a.get("phase", "setup") == b.get("phase", "setup") == phase]
            valid = [(a, b) for a, b in pairs if a["busy_percent"] is not None and b["busy_percent"] is not None]
            resident = sum((b["t"] - a["t"]) * (a["busy_percent"] + b["busy_percent"]) / 200 for a, b in valid)
            wall = sum(b["t"] - a["t"] for a, b in pairs)
            result["phases"][phase] = {"sampled_wall_seconds": wall, "gpu_resident_seconds": resident if valid else None,
                "wall_without_resident_kernels_seconds": wall - resident if valid else None,
                "method": "same-phase adjacent NVML samples; wall minus kernel-resident time, not CPU utilization"}
        if not include_samples:
            result.pop("samples")
        result["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["whole_job_peak_memory_bytes"] = cgroup_peak()
        result["whole_job_peak_memory_method"] = "memory.peak of the inherited Flux cgroup; null if unavailable"
        result.update(torch_peak_allocated_bytes=None, torch_peak_reserved_bytes=None)
        torch = __import__("sys").modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            result.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return result

    def finish(self):
        if hasattr(self, "result"):
            return self.result
        self.done.set()
        self.thread.join(timeout=11)
        if self.thread.is_alive():
            self.errors.append("GPU telemetry did not stop within its command timeout")
        self.result = self.snapshot()
        return self.result


def cgroup_peak(root="/sys/fs/cgroup", membership="/proc/self/cgroup"):
    """Read the inherited Flux cgroup peak, which includes the container children."""
    from pathlib import Path
    for line in Path(membership).read_text().splitlines():
        if line.startswith("0::"):
            path = Path(root) / line[3:].lstrip("/") / "memory.peak"
            if path.is_file():
                return int(path.read_text())
    return None
