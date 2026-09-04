"""The stage finalizer owns only its launch and one bounded cleanup interval."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cleanup_function():
    path = ROOT / "experiments" / "ts5_stage_cleanup.py"
    spec = importlib.util.spec_from_file_location("ts5_stage_cleanup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cleanup_stage


class CleanupHarness:
    def __init__(self, *, existing="owned-container", gpu="", delays=(), failure=None):
        self.now = 100.0
        self.existing = existing
        self.gpu = gpu
        self.delays = iter(delays)
        self.failure = failure
        self.calls = []
        self.joins = []
        self.stopped = False

    def clock(self):
        return self.now

    def stop(self):
        self.stopped = True

    def join(self, *, timeout):
        assert self.stopped, "stop telemetry before waiting on it"
        self.joins.append(timeout)
        self.now += timeout

    def run(self, command, **kwargs):
        timeout = kwargs["timeout"]
        self.calls.append((list(command), timeout, self.now))
        assert timeout > 0, "never start an operation after its deadline"
        duration = next(self.delays, 0)
        self.now += min(duration, timeout)
        if duration >= timeout:
            raise subprocess.TimeoutExpired(command, timeout)
        if self.failure is not None:
            raise self.failure
        if command[:3] == ["docker", "rm", "-f"]:
            self.existing = ""
            output = ""
        elif command[:2] == ["docker", "ps"]:
            output = self.existing
        elif command[0] == "nvidia-smi":
            output = self.gpu
        else:
            raise AssertionError(f"unexpected cleanup command: {command}")
        return SimpleNamespace(stdout=output, stderr="", returncode=0)

    def cleanup(self, *, launched=True, completed=True):
        return cleanup_function()(
            "stage-container", launched=launched, completed=completed,
            stop=SimpleNamespace(set=self.stop), monitor=SimpleNamespace(join=self.join),
            run=self.run, clock=self.clock,
        )


@pytest.mark.parametrize("completed", [False, True])
def test_owned_container_removed_and_release_verified(completed):
    harness = CleanupHarness()
    result = harness.cleanup(completed=completed)
    assert result["launched"] is True
    assert result["measurement_completed"] is completed
    assert result["container_before_cleanup"] == "owned-container"
    assert result["container_after_cleanup"] == ""
    assert result["gpu_compute_processes"] == ""
    assert result["safe_to_release"] is True
    removes = [(command, timeout) for command, timeout, _ in harness.calls
               if command[:3] == ["docker", "rm", "-f"]]
    assert len(removes) == 1
    assert removes[0][0] == ["docker", "rm", "-f", "stage-container"]
    assert 0 < removes[0][1] <= 45
    assert len(harness.joins) == 1 and 0 < harness.joins[0] <= 2


def test_unlaunched_stage_does_not_remove_preexisting_container():
    harness = CleanupHarness(existing="someone-elses-container")
    result = harness.cleanup(launched=False, completed=False)
    assert result["launched"] is False
    assert result["safe_to_release"] is False
    assert not any(command[:3] == ["docker", "rm", "-f"]
                   for command, _, _ in harness.calls)
    assert harness.existing == "someone-elses-container"
    assert harness.stopped


def test_existing_gpu_work_prevents_release_even_after_container_removed():
    harness = CleanupHarness(gpu="999, remaining-worker, 4096 MiB")
    result = harness.cleanup()
    assert result["container_after_cleanup"] == ""
    assert result["gpu_compute_processes"] == harness.gpu
    assert result["safe_to_release"] is False


def test_absent_owned_container_needs_no_removal_but_still_checks_gpu():
    harness = CleanupHarness(existing="")
    result = harness.cleanup()
    assert result["safe_to_release"] is True
    assert not any(command[:3] == ["docker", "rm", "-f"]
                   for command, _, _ in harness.calls)
    assert any(command[0] == "nvidia-smi" for command, _, _ in harness.calls)


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired(["docker", "ps"], 30),
    subprocess.CalledProcessError(1, ["docker", "ps"], stderr="daemon unavailable"),
    OSError("docker executable missing"),
])
def test_subprocess_failure_becomes_unsafe_receipt(failure):
    harness = CleanupHarness(failure=failure)
    result = harness.cleanup()
    assert result["safe_to_release"] is False
    assert result["error"] and type(failure).__name__ in result["error"]
    assert harness.stopped and harness.joins


def test_inspection_removal_and_gpu_share_one_deadline():
    # Each operation fits its individual timeout on the old serial budget,
    # but their combined duration cannot fit the one 90-second interval.
    harness = CleanupHarness(delays=(28, 44, 28, 28))
    result = harness.cleanup()
    assert result["safe_to_release"] is False
    assert result["error"]
    assert harness.now <= 190.0
    for _command, timeout, started in harness.calls:
        assert started + timeout <= 190.0
    assert harness.joins[0] <= 2


def test_telemetry_join_error_is_recorded_not_raised():
    harness = CleanupHarness()
    def failed_join(*, timeout):
        raise RuntimeError("telemetry join failed")
    result = cleanup_function()(
        "stage-container", launched=True, completed=False,
        stop=SimpleNamespace(set=harness.stop), monitor=SimpleNamespace(join=failed_join),
        run=harness.run, clock=harness.clock,
    )
    assert result["safe_to_release"] is False
    assert "telemetry join failed" in result["error"]
