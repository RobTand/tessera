"""Deterministic teardown and isolation probes for the control/PM test adapter.

The scripted process models the adapter's stdin stop/acknowledgement protocol,
not LocalArm's TERM protocol. No host PID or real-time wait is used in teardown
probes. The real-rank tests still prove process death, restart and OFF records.
"""
from pathlib import Path
import io
import math
import runpy
import signal
import subprocess
import sys

import pytest

import test_graph_attest_determinism_control as scenario
import managed_window as window
import rank_window


class ScriptedServer:
    """Stateful pipe boundary; only stop or KILL permits a successful reap."""

    args = ["inert-control-server"]

    def __init__(self, behavior):
        self.behavior = behavior
        self.running = behavior != "already-exited"
        self.returncode = None if self.running else 1
        self.reaped = False
        self.events = []
        self.stdin, self.stdout = io.StringIO(), io.StringIO()

    @staticmethod
    def bounded(timeout):
        assert timeout is not None and math.isfinite(timeout) and timeout > 0, (
            "owned-process operations need a finite failure bound")

    def communicate(self, command, timeout=None):
        self.bounded(timeout)
        assert not self.events, "cleanup retried its stop handshake"
        assert command == "stop\n", "cleanup did not use the payload's stop protocol"
        self.events.append("stop")
        if self.behavior == "unresponsive":
            raise subprocess.TimeoutExpired(self.args, timeout)
        if self.running:
            self.running = False
            self.returncode = 0
        self.reaped = True
        if self.behavior == "acknowledged":
            return "stopped\n", None
        return "", None  # no successful stop acknowledgement

    def poll(self):
        return self.returncode

    def kill(self):
        assert self.events == ["stop"], "KILL must follow the owned stop attempt"
        assert self.running, "cleanup escalated after the process had exited"
        self.events.append("kill")
        self.running = False
        self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        self.bounded(timeout)
        self.events.append("reap")
        assert not self.running, "cleanup tried to reap without stopping the server"
        self.reaped = True
        return self.returncode


class CleanupBudget:
    """Constant, generous failure budget; it is never a success oracle."""

    def remaining(self, *, cleanup=False):
        return 120.0


def control_adapter(tmp_path, monkeypatch, phase, *, interrupt=False):
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
        if interrupt:
            raise RuntimeError("capture interrupted")
        return 0

    import eager_benchmark as benchmark
    with monkeypatch.context() as patch:
        # runpy executes direct assignments in the entry point. Register their
        # predecessors first: ordinary monkeypatch teardown cannot infer them.
        patch.setattr(benchmark, "CLIENT", benchmark.CLIENT)
        patch.setattr(window, "time", window.time)
        patch.setattr(rank_window, "time", rank_window.time)
        patch.setattr(sys, "path", list(sys.path))
        patch.setattr(rank_window, "run_rank", capture)
        patch.setattr(window, "Envelope", lambda *args, **kwargs: CleanupBudget())
        patch.setattr(sys, "argv", [str(Path(scenario.__file__)), "--control-rank",
                                  str(parent), "0", "0", phase])
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(scenario.__file__, run_name="__main__")
        assert exit_info.value.code == 0
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("phase", ["control", "pm"])
@pytest.mark.parametrize("behavior", ["acknowledged", "already-exited", "missing-ack", "unresponsive"])
def test_control_rank_cleanup_proves_exit_before_claiming_empty(
        tmp_path, monkeypatch, phase, behavior):
    adapter, arm = control_adapter(tmp_path, monkeypatch, phase)
    server = ScriptedServer(behavior)
    adapter.server = server
    adapter.active = arm["arm"]

    if behavior == "acknowledged":
        physical = adapter.cleanup(arm)
        assert physical["containers_empty"] is True
        assert physical["gpu_descendants_empty"] is True
    else:
        error = subprocess.TimeoutExpired if behavior == "unresponsive" else AssertionError
        with pytest.raises(error):
            adapter.cleanup(arm)  # no successful cleanup record without an ACK

    assert not server.running, "cleanup left its server running"
    assert server.reaped, "cleanup must reap even an already-exited server"
    assert adapter.server is None, "the next arm must not adopt the previous server"
    expected = ["stop", "kill", "reap"] if behavior == "unresponsive" else ["stop", "reap"]
    assert server.events == expected
    assert server.stdin.closed and server.stdout.closed


@pytest.mark.parametrize("interrupt", [False, True], ids=["normal-exit", "capture-failure"])
def test_control_adapter_capture_restores_shared_state(tmp_path, monkeypatch, interrupt):
    import eager_benchmark as benchmark
    previous_client = benchmark.CLIENT
    previous_window_clock, previous_rank_clock = window.time, rank_window.time
    previous_envelope, previous_runner = window.Envelope, rank_window.run_rank
    previous_argv, previous_path = sys.argv, sys.path
    if interrupt:
        with pytest.raises(RuntimeError, match="capture interrupted"):
            control_adapter(tmp_path, monkeypatch, "control", interrupt=True)
    else:
        control_adapter(tmp_path, monkeypatch, "control")
    assert benchmark.CLIENT is previous_client
    assert window.time is previous_window_clock
    assert rank_window.time is previous_rank_clock
    assert window.Envelope is previous_envelope
    assert rank_window.run_rank is previous_runner
    assert sys.argv is previous_argv and sys.path is previous_path


def test_direct_pm_lifecycle_owns_its_validator_source(tmp_path, monkeypatch):
    import eager_benchmark as benchmark
    unavailable = tmp_path / "absent-box-client.py"
    monkeypatch.setattr(benchmark, "CLIENT", unavailable)
    # Direct invocation deliberately bypasses scenario.local_client, as the
    # protected regression does. Both real controllers and all assertions run.
    scenario.test_real_rank_lifecycle_restarts_both_ranks_and_retains_mismatch_without_failing(
        tmp_path, "pm", False)
    assert benchmark.CLIENT is unavailable
