"""Measure row wall time and kernel-resident time while a compute process exists.

utilization.gpu is the percent of a sample interval with a resident kernel. It
is not SM throughput. Integrate adjacent samples by the trapezoidal rule; keep
sample coverage and errors so an incomplete measurement cannot claim zero use.
GB10 can retain a stale utilization counter after all compute processes exit.
No process means zero busy time. PID values are diagnostic and are never matched.
The row owns the Flux GPU slot and checks that it starts free.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
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


def compute_processes(text):
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pid, memory = line.split(",")
        pid, memory = int(pid), number(memory)
        if pid <= 0 or memory is not None and (not math.isfinite(memory) or memory < 0):
            raise ValueError("invalid NVML compute process")
        records.append({"pid": pid, "memory_mib": memory})
    return records


def process_memory(apps):
    values = [a["memory_mib"] for a in apps]
    return sum(values) if all(v is not None for v in values) else None


def query(query):
    return subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=5)


def vllm_processes(root=Path("/proc")):
    found = []
    for path in root.glob("[0-9]*"):
        try:
            name = (path / "comm").read_text().strip().lower()
            arguments = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if name.startswith("vllm") or any(Path(a).name == "vllm" or a.startswith("vllm.") for a in arguments):
                found.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def preflight(reading=query, processes=vllm_processes):
    gpu = gpu_reading(reading("--query-gpu=uuid,utilization.gpu,power.draw,memory.used"))
    apps = compute_processes(reading("--query-compute-apps=pid,used_memory"))
    memory = process_memory(apps)
    used = gpu["memory_mib"] if gpu["memory_mib"] is not None else memory
    servers = processes()
    return {"verdict": "FREE" if not apps and used is not None and 0 <= used < 0.3 * 1024 and not servers else "STOPPED",
            "compute_process_count": len(apps), "compute_apps_memory_mib": memory,
            "device_memory_mib": used, "device_memory_limit_mib": 0.3 * 1024,
            "device_memory_method": "NVML memory.used" if gpu["memory_mib"] is not None else "NVML compute-apps used_memory; GB10 memory.used unavailable",
            "vllm_process_count": len(servers), "uuid": gpu["uuid"]}


def active_busy(sample):
    count = sample.get("compute_process_count")
    return None if count is None else 0 if count == 0 else sample.get("nvml_busy_percent", sample["busy_percent"])


def self_visibility(torch):
    _allocation = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    try:
        apps = compute_processes(query("--query-compute-apps=pid,used_memory"))
    except Exception as error:
        return {"verdict": "STOPPED", "reason": "NVML self-visibility query failed", "exception_type": type(error).__name__}
    return {"verdict": "VISIBLE" if apps else "STOPPED", "reason": None if apps else "NVML self-visibility found no compute process",
            "compute_process_count": len(apps), "compute_apps_memory_mib": process_memory(apps)}


def summarize(samples, start, end, errors=()):
    samples = [{**s, "busy_percent": active_busy(s)} for s in samples if start <= s["t"] <= end]
    intervals = [(a, b) for a, b in zip(samples, samples[1:]) if a["busy_percent"] is not None and b["busy_percent"] is not None]
    busy = sum((b["t"] - a["t"]) * (a["busy_percent"] + b["busy_percent"]) / 200 for a, b in intervals)
    power = [s["power_watts"] for s in samples if s.get("power_watts") is not None]
    memory = [s["compute_apps_memory_mib"] for s in samples if s.get("compute_apps_memory_mib") is not None]
    return {"wall_minutes": (end - start) / 60, "gpu_busy_minutes": busy / 60 if intervals else None,
            "method": "NVML resident percent gated by any compute process; trapezoidal integral; no PID matching; not SM throughput",
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
        self.nvml_pids = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def query(self, request):
        return query(request)

    def sample(self):
        while True:
            try:
                reading = gpu_reading(self.query("--query-gpu=uuid,utilization.gpu,power.draw,memory.used"))
                if self.samples and reading["uuid"] != self.samples[0]["uuid"]:
                    raise ValueError("changed GPU telemetry device")
                reading.update(t=time.monotonic(), nvml_busy_percent=reading["busy_percent"], compute_process_count=None,
                               compute_process_memory_mib=None, compute_apps_memory_mib=None, phase=getattr(self, "phase", "setup"))
                try:
                    apps = compute_processes(self.query("--query-compute-apps=pid,used_memory"))
                    reading["compute_process_count"] = len(apps)
                    reading["compute_process_memory_mib"] = [a["memory_mib"] for a in apps]
                    reading["compute_apps_memory_mib"] = process_memory(apps)
                    if apps and getattr(self, "nvml_pids", None) is None:
                        self.nvml_pids = [a["pid"] for a in apps]
                except Exception as error:
                    self.errors.append(str(error))
                reading["busy_percent"] = active_busy(reading)
                self.samples.append(reading)
            except Exception as error:
                self.errors.append(str(error))
            if self.done.wait(1):
                break

    def snapshot(self, *, include_samples=True):
        end = time.monotonic()
        samples = [{**s, "busy_percent": active_busy(s)} for s in self.samples]
        result = summarize(samples, self.start, end, self.errors)
        result["pid_diagnostic"] = {"in_container_pid": os.getpid(), "nvml_pids": getattr(self, "nvml_pids", None) or [],
                                    "scope": "first nonempty NVML sample; diagnostic only; no PID matching"}
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


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    try:
        report = preflight()
    except Exception as error:
        report = {"verdict": "STOPPED", "exception_type": type(error).__name__}
    args.preflight.write_text(json.dumps(report, indent=2) + "\n")
    if report["verdict"] != "FREE":
        now = datetime.now(timezone.utc).isoformat()
        receipt = {"schema": "tessera.numerics_row.v2", "canary": args.canary, "units": [],
                   "row": {"outcome": "STOPPED", "utc_start": now, "utc_end": now, "exit_status": 3,
                           "finalization_finished": False, "stop_cause": "GPU preflight not free", "gpu_preflight": report}}
        (args.preflight.parent / "numerics_row.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise SystemExit(3)
