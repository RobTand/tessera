"""Payload exceptions retain their cause while cleaning owned process groups."""
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("container_evidence", ["none", "malformed"])
@pytest.mark.parametrize("machine,filesystem,qualified", [
    ("x86_64", "xfs", True), ("aarch64", "ext4", True),
    ("x86_64", "ext4", False), ("aarch64", "xfs", False),
])
def test_payload_exception_stops_owned_group_and_preserves_cleanup_error(
        tmp_path, monkeypatch, container_evidence, machine, filesystem, qualified):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("payload_flux", ROOT / "tools/_suite_flux.py")
    flux = importlib.util.module_from_spec(spec); spec.loader.exec_module(flux)
    directory = tmp_path / "record"; directory.mkdir()
    scratch_root = tmp_path / "scratch"; scratch_root.mkdir()
    monkeypatch.setattr(flux, "payload_scratch_root", lambda: scratch_root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLUX_JOB_ID", "f123")
    identity = {"snapshot_commit": "a" * 40, "sha256": "b" * 64}
    intent = {"directory": str(directory), "checkout": str(tmp_path), "source": identity, "application": None,
              "command": ["python", "payload"], "timeout_s": 60, "runtime": {"sha256": "c" * 64},
              "qualification": {"broker_starttime": "fixture-instance"}}
    if container_evidence == "malformed":
        intent["container"] = {"schema": "tessera.suite_container.v1"}
        (directory / "container.json").write_text("invalid container evidence")
    flux.save(directory / "launch.json", intent)
    digest = flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest()
    flux.save(directory / "submission.json", {"state": "submitted", "job_id": "f123", "launch_sha256": digest})
    original = {"tasks": [{"command": ["/usr/bin/python3", "-s", "tools/_suite_flux.py", "payload", str(directory), digest]}]}
    flux.save(directory / "jobspec.json", original)
    monkeypatch.setattr(flux, "checked", lambda args: "f123" if args[:2] == ["job", "id"] else json.dumps(original))
    monkeypatch.setattr(flux, "broker_starttime", lambda **kw: "fixture-instance")
    monkeypatch.setattr(flux, "source", lambda path: identity)
    monkeypatch.setattr(flux, "runtime_identity", lambda *args, **kwargs: intent["runtime"])
    monkeypatch.setattr(flux.os, "uname", lambda: SimpleNamespace(machine=machine))
    monkeypatch.setattr(flux.subprocess, "check_output", lambda *args, **kwargs: filesystem)
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
    if not qualified:
        with pytest.raises(ValueError, match="unqualified local scratch filesystem"):
            flux.payload(str(directory), digest)
        assert not signals
        assert not (scratch_root / "f123").exists()
        assert not (directory / "payload.json").exists()
        return
    with pytest.raises(OSError, match="payload wait failed"):
        flux.payload(str(directory), flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest())
    record = json.loads((directory / "payload.json").read_bytes())
    assert (777, signal.SIGTERM) in signals
    assert record["scratch_removed"] is False
    assert "owned scratch cleanup failed" in record["cleanup_error"]


@pytest.mark.parametrize("native_id,ascii_id", [("f123", "f123"), ("ƒ123", "f123")])
@pytest.mark.parametrize("interrupt_at", ["wait", "source", "runtime"])
@pytest.mark.parametrize("machine,filesystem", [("x86_64", "xfs"), ("aarch64", "ext4")])
def test_interrupt_publishes_cleanup_without_waiting_for_an_exit_seal(
        tmp_path, monkeypatch, interrupt_at, machine, filesystem, native_id, ascii_id):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("interrupt_flux", ROOT / "tools/_suite_flux.py")
    flux = importlib.util.module_from_spec(spec); spec.loader.exec_module(flux)
    directory = tmp_path / "record"; directory.mkdir()
    scratch_root = tmp_path / "scratch"; scratch_root.mkdir()
    monkeypatch.setattr(flux, "payload_scratch_root", lambda: scratch_root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLUX_JOB_ID", native_id)
    identity = {"snapshot_commit": "a" * 40, "sha256": "b" * 64}
    intent = {"directory": str(directory), "checkout": str(tmp_path), "source": identity, "application": None,
              "command": ["python", "payload"], "timeout_s": 60, "runtime": {"sha256": "c" * 64},
              "qualification": {"broker_starttime": "fixture-instance"}}
    flux.save(directory / "launch.json", intent)
    digest = flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest()
    flux.save(directory / "submission.json", {"state": "submitted", "job_id": ascii_id, "launch_sha256": digest})
    original = {"tasks": [{"command": ["/usr/bin/python3", "-s", "tools/_suite_flux.py", "payload", str(directory), digest]}]}
    flux.save(directory / "jobspec.json", original)
    monkeypatch.setattr(flux, "checked", lambda args: "f123" if args[:2] == ["job", "id"] else json.dumps(original))
    monkeypatch.setattr(flux, "broker_starttime", lambda **kw: "fixture-instance")
    observations = []
    def source(path):
        if "source" in observations:
            if interrupt_at == "source":
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            assert interrupt_at == "runtime", "cancellation must not delay cleanup with an exit source seal"
        observations.append("source")
        return identity
    def runtime(*args, **kwargs):
        if "runtime" in observations and interrupt_at == "runtime":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert "runtime" not in observations, "cancellation must not delay cleanup with an exit runtime seal"
        observations.append("runtime")
        return intent["runtime"]
    monkeypatch.setattr(flux, "source", source)
    monkeypatch.setattr(flux, "runtime_identity", runtime)
    monkeypatch.setattr(flux.os, "uname", lambda: SimpleNamespace(machine=machine))
    monkeypatch.setattr(flux.subprocess, "check_output", lambda *args, **kwargs: filesystem)
    handlers = {}
    monkeypatch.setattr(flux.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    signals = []
    child = SimpleNamespace(pid=777, returncode=None)
    def wait():
        if interrupt_at == "wait":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        child.returncode = 0
        return 0
    child.wait = wait
    child.poll = lambda: child.returncode
    child.send_signal = signals.append
    monkeypatch.setattr(flux.subprocess, "Popen", lambda *args, **kwargs: child)
    assert flux.payload(str(directory), flux.hashlib.sha256((directory / "launch.json").read_bytes()).hexdigest()) == 143
    record = json.loads((directory / "payload.json").read_bytes())
    assert signals == ([signal.SIGTERM] if interrupt_at == "wait" else [])
    assert record["interrupted_signal"] == signal.SIGTERM
    assert record["scratch_removed"] and not (scratch_root / ascii_id).exists()
    assert record["runtime_verified"] is False
    assert record["scratch"] == str(scratch_root / ascii_id)
