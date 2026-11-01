"""The class benchmark stops owned work after its launcher exits."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_guard():
    spec = importlib.util.spec_from_file_location("class_measurement_guard", ROOT / "experiments/t8r_speed/memory_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        # The entry can vanish mid-read (ESRCH) while the task is reaped.
        return False


def test_running_treats_a_vanishing_process_as_not_running(monkeypatch):
    # A process that is being reaped can raise ESRCH from the read, not ENOENT.
    def vanish(self, *args, **kwargs):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(Path, "read_text", vanish)
    assert running(os.getpid()) is False


@pytest.mark.parametrize("trigger", ["floor", "read_error", "launcher_exit"])
def test_guard_stops_workload_after_launcher_exit(tmp_path, monkeypatch, trigger):
    guard = load_guard()
    pidfile = tmp_path / "workload.pid"
    termfile = tmp_path / "workload.term"
    leaf = ("import os,signal,time;from pathlib import Path;"
            f"signal.signal(signal.SIGTERM,lambda *_:Path({str(termfile)!r}).write_text('TERM'));"
            f"Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(120)")
    launcher = ("import signal,subprocess,sys,time;"
                "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0));"
                f"p=subprocess.Popen([sys.executable,'-c',{leaf!r}]);"
                + ("time.sleep(.1)" if trigger == "launcher_exit" else "p.wait()"))
    reads = 0

    def memory():
        nonlocal reads
        reads += 1
        if reads == 1:
            return guard.FLOOR_BYTES + 1
        deadline = time.monotonic() + 10
        while not pidfile.exists():
            assert time.monotonic() < deadline, "The workload did not start."
            time.sleep(.01)
        if trigger == "read_error":
            raise OSError("memory reader failed")
        return guard.FLOOR_BYTES + 1 if trigger == "launcher_exit" else 0

    monkeypatch.setattr(guard, "available_bytes", memory)
    monkeypatch.setattr(guard, "TERM_GRACE_SECONDS", .2)
    monkeypatch.setattr(sys, "argv", ["memory_guard", "--out", str(tmp_path / "memory.json"),
                                      "--", sys.executable, "-c", launcher])
    pid = None
    try:
        if trigger == "read_error":
            with pytest.raises(OSError, match="memory reader failed"):
                guard.main()
        else:
            status = guard.main()
            assert status == (1 if trigger == "floor" else 0)
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 1
        while running(pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not running(pid), "D30 left the workload alive after its launcher exited."
        assert termfile.read_text() == "TERM"
    finally:
        if pid is None and pidfile.exists():
            pid = int(pidfile.read_text())
        if pid is not None and running(pid):
            os.kill(pid, signal.SIGKILL)
