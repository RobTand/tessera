"""Payload exceptions retain their cause while cleaning owned process groups."""
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_payload_exception_stops_owned_group_and_preserves_cleanup_error(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("payload_flux", ROOT / "tools/_suite_flux.py")
    flux = importlib.util.module_from_spec(spec); spec.loader.exec_module(flux)
    directory = tmp_path / "record"; directory.mkdir()
    scratch_root = tmp_path / "scratch"; scratch_root.mkdir()
    class Paths:
        def __new__(cls, value):
            return scratch_root if str(value) == "/home/rob/tl-gates/flux-validation" else Path(value)
        cwd = Path.cwd
    monkeypatch.setattr(flux, "Path", Paths)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLUX_JOB_ID", "123")
    identity = {"snapshot_commit": "a" * 40, "sha256": "b" * 64}
    intent = {"checkout": str(tmp_path), "source": identity, "application": None,
              "command": ["python", "payload"], "timeout_s": 60, "runtime": {"sha256": "c" * 64}}
    flux.save(directory / "launch.json", intent)
    monkeypatch.setattr(flux, "source", lambda path: identity)
    monkeypatch.setattr(flux, "runtime_identity", lambda *args, **kwargs: intent["runtime"])
    monkeypatch.setattr(flux.subprocess, "check_output", lambda *args, **kwargs: "xfs")
    handlers = {}
    monkeypatch.setattr(flux.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    monkeypatch.setattr(flux.time, "sleep", lambda seconds: None)
    signals = []
    child = SimpleNamespace(pid=777, returncode=None)
    def wait(*args, **kwargs):
        if not signals: raise OSError("payload wait failed")
        child.returncode = -signal.SIGTERM
        return child.returncode
    child.wait = wait
    child.poll = lambda: child.returncode
    child.send_signal = lambda sig: signals.append((child.pid, sig))
    monkeypatch.setattr(flux.os, "killpg", lambda group, sig: signals.append((group, sig)))
    def popen(*args, **kwargs):
        assert kwargs.get("start_new_session") is True, "payload must own its child process group"
        return child
    monkeypatch.setattr(flux.subprocess, "Popen", popen)
    def cleanup(path): raise PermissionError("owned scratch cleanup failed")
    monkeypatch.setattr(flux.shutil, "rmtree", cleanup)
    with pytest.raises(OSError, match="payload wait failed"):
        flux.payload(str(directory), flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest())
    record = json.loads((directory / "payload.json").read_bytes())
    assert (777, signal.SIGTERM) in signals
    assert record["scratch_removed"] is False
    assert "owned scratch cleanup failed" in record["cleanup_error"]
