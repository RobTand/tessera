"""Deterministic teardown probes for the control/PM rank test adapter.

These exercise the adapter used by test_graph_attest_determinism_control's
child entry point, not LocalArm or a real host process. A scripted process
makes TERM noncompletion observable without a real-time wait.
The real-rank integration tests remain responsible for process isolation,
restart evidence and the OFF population comparison.
"""
from pathlib import Path
import math
import os
import runpy
import signal
import subprocess
import sys

import pytest

import test_graph_attest_determinism_control as scenario
import managed_window as window
import rank_window


class ScriptedServer:
    """Inert process boundary: TERM does not complete; KILL makes it waitable.

    No host PID is used. In the live case the first bounded wait reports
    that TERM did not complete; a later wait can reap only a stopped process.
    """

    pid = 424242
    args = ["inert-control-server"]

    def __init__(self, already_exited=False):
        self.running = not already_exited
        self.returncode = 0 if already_exited else None
        self.reaped = False
        self.term_requested = False
        self.wait_calls = 0

    def poll(self):
        if not self.running:
            self.reaped = True
        return self.returncode

    def signal_group(self, pid, sig):
        if pid != self.pid:
            raise AssertionError("cleanup touched a process other than its owned server")
        if not self.running:
            raise ProcessLookupError("owned process group is already absent")
        if sig == signal.SIGTERM:
            self.term_requested = True
        elif sig == signal.SIGKILL:
            assert self.term_requested, "cleanup must attempt graceful termination before KILL"
            self.running = False
            self.returncode = -signal.SIGKILL
        elif sig != 0:
            raise AssertionError(f"unexpected process-group signal {sig}")

    def terminate(self):
        self.signal_group(self.pid, signal.SIGTERM)

    def kill(self):
        self.signal_group(self.pid, signal.SIGKILL)

    def wait(self, timeout=None):
        assert timeout is not None and math.isfinite(timeout) and timeout >= 0, (
            "owned-process waits need a finite failure bound")
        self.wait_calls += 1
        assert self.wait_calls <= 4, "cleanup kept retrying without stopping the server"
        if self.running:
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.reaped = True
        return self.returncode


class CleanupBudget:
    """Constant, generous failure budget; it is never a success oracle."""

    def remaining(self, *, cleanup=False):
        return 120.0


def control_adapter(tmp_path, monkeypatch, phase):
    """Capture the actual child adapter without starting the rank protocol.

    The current harness defines ControlArm inside its script entry point.
    Running that entry point keeps this regression coupled to the real adapter
    rather than testing a copied approximation of its inherited cleanup.
    """
    parent = tmp_path / phase
    parent.mkdir()
    window.atomic_json(parent / "identity0.json", dict(
        rank=0, scope_id="inert-owned-scope", window_end_unix=1_800_000_000.0))
    captured = []

    def capture(config, owned, queue, root, arms, adapter, envelope, **kwargs):
        captured.append((adapter, arms[0]))
        return 0

    monkeypatch.setattr(rank_window, "run_rank", capture)
    monkeypatch.setattr(window, "Envelope", lambda *args, **kwargs: CleanupBudget())
    monkeypatch.setattr(sys, "argv", [str(Path(scenario.__file__)), "rank",
                                      str(parent), "0", "0", phase])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(scenario.__file__, run_name="__main__")
    assert exit_info.value.code == 0
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("phase", ["control", "pm"])
@pytest.mark.parametrize("already_exited", [False, True], ids=["term-not-completed", "already-exited"])
def test_control_rank_cleanup_proves_exit_before_claiming_empty(
        tmp_path, monkeypatch, phase, already_exited):
    adapter, arm = control_adapter(tmp_path, monkeypatch, phase)
    server = ScriptedServer(already_exited)
    adapter.server = server
    adapter.active = arm["arm"]
    # Both modules use the same os module. This boundary never signals a host PID.
    monkeypatch.setattr(os, "killpg", server.signal_group)

    physical = adapter.cleanup(arm)

    assert not server.running, "cleanup claimed emptiness while its server still runs"
    assert server.reaped, "cleanup must reap the owned server, including an already-exited one"
    assert adapter.server is None, "the next arm must not adopt the previous server"
    assert physical["containers_empty"] is True
    assert physical["gpu_descendants_empty"] is True
