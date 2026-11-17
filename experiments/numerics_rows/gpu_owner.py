"""Resolve NVML host PIDs through the capture container's host process tree."""
import json
from pathlib import Path
import subprocess
import time

DIRECTORY = Path("/run/numerics/gpu-owner")
HOST_PROC = Path("/run/numerics/host-proc")


def process(pid, root):
    fields = (root / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[1]), int(fields[19])


class ProcessOwner:
    def __init__(self, record, root=HOST_PROC):
        self.pid, self.start, self.root = record["host_pid"], record["start_ticks"], Path(root)
        if type(self.pid) is not int or self.pid <= 1 or process(self.pid, self.root)[1] != self.start:
            raise RuntimeError("capture host worker identity is invalid")

    @classmethod
    def wait(cls):
        deadline = time.monotonic() + 30
        while not (DIRECTORY / "worker.json").is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError("capture host worker identity is missing")
            time.sleep(0.1)
        return cls(json.loads((DIRECTORY / "worker.json").read_text()))

    def contains(self, pid):
        if process(self.pid, self.root)[1] != self.start:
            raise RuntimeError("capture host worker PID was reused")
        seen = set()
        while pid > 1 and pid not in seen:
            if pid == self.pid:
                return True
            seen.add(pid)
            pid, _ = process(pid, self.root)
        if pid in seen:
            raise RuntimeError("capture host process ancestry has a cycle")
        return False

    def counts(self, apps):
        counts = dict.fromkeys(("owned_compute_process_count", "foreign_compute_process_count", "unresolved_compute_process_count"), 0)
        for app in apps:
            try:
                key = "owned_compute_process_count" if self.contains(app["pid"]) else "foreign_compute_process_count"
            except (OSError, ValueError, IndexError, RuntimeError):
                key = "unresolved_compute_process_count"
            counts[key] += 1
        return counts


def run_container(name, directory, command):
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=True)
    child = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                return child.returncode
            try:
                pid = int(subprocess.check_output(["podman", "inspect", "--format", "{{.State.Pid}}", name],
                                                 text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
                if pid > 1:
                    _, start = process(pid, Path("/proc"))
                    pending = directory / "worker.pending"
                    pending.write_text(json.dumps({"host_pid": pid, "start_ticks": start}) + "\n")
                    pending.replace(directory / "worker.json")
                    return child.wait()
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            time.sleep(0.1)
        raise RuntimeError("capture container host PID was not observed")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
