"""The D30 grace tests must model cleanup, not benchmark the host scheduler.

Run the actual named tests with ambient clock/process boundaries unavailable.
Fixtures may replace these module-local facades with controlled implementations;
pytest itself and unrelated standard-library users retain their real modules.
The probe also requires entry into the original production cleanup routine, so
removing the exercise of Envelope is not a way to satisfy the contract.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from child_env import child_env


ROOT = Path(__file__).resolve().parents[1]
TARGET = "tests/test_graph_attest_d30_memory_guard.py"
CASES = (
    "test_cooperative_child_exits_within_grace_without_a_kill",
    "test_finite_cleanup_deadline_shortens_the_kill_grace",
    "test_term_ignoring_child_group_is_killed_only_after_the_full_grace",
)


class ControlledBoundary:
    """Forward constants/types, but fail before an uncontrolled physical action."""

    def __init__(self, real, blocked):
        self._real = real
        self._blocked = blocked

    def __getattr__(self, name):
        if name not in self._blocked:
            return getattr(self._real, name)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                f"D30 cleanup tests require a controlled boundary: "
                f"{self._real.__name__}.{name}; inject clock/process events"
            )

        return forbidden


class CleanupProbe:
    def __init__(self):
        self.calls = []
        self.cleanup_entries = 0

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        module = item.module
        window = module.window
        saved = []
        for owner in (module, window):
            for name, blocked in (
                ("time", {"time", "monotonic", "sleep", "perf_counter"}),
                ("subprocess", {"Popen", "run", "call", "check_call", "check_output"}),
                ("os", {"kill", "killpg"}),
            ):
                real = getattr(owner, name)
                saved.append((owner, name, real))
                setattr(owner, name, ControlledBoundary(real, blocked))
        original = window.Envelope._terminate

        def terminate(envelope, process, argv):
            self.cleanup_entries += 1
            return original(envelope, process, argv)

        window.Envelope._terminate = terminate
        try:
            yield
        finally:
            window.Envelope._terminate = original
            for owner, name, value in reversed(saved):
                setattr(owner, name, value)

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.calls.append(dict(nodeid=report.nodeid, outcome=report.outcome,
                                   xfail=hasattr(report, "wasxfail")))


def _probe(nodeid, scratch):
    plugin = CleanupProbe()
    status = pytest.main([
        nodeid, "-q", "--noconftest", "-p", "no:cacheprovider",
        "--basetemp=" + str(Path(scratch) / "pytest"),
    ], plugins=[plugin])
    print("D30_PROBE=" + json.dumps(dict(status=int(status), calls=plugin.calls,
                                        cleanup_entries=plugin.cleanup_entries)))


@pytest.mark.parametrize("case", CASES, ids=CASES)
def test_cleanup_test_uses_controlled_boundaries(tmp_path, case):
    nodeid = TARGET + "::" + case
    # An isolated interpreter prevents imported test modules, clocks or signal
    # replacements from leaking to another test/xdist worker. Timeout is only a
    # generous deadlock backstop, never a performance expectation.
    script = (
        "import importlib.util, pathlib, sys; "
        "sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent)); "
        "spec = importlib.util.spec_from_file_location('d30_probe', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); module._probe(sys.argv[2], sys.argv[3])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), nodeid, str(tmp_path)],
        cwd=ROOT,
        env=child_env(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""),
        text=True, capture_output=True, timeout=120,
    )
    diagnostic = result.stdout + result.stderr
    assert result.returncode == 0, diagnostic
    lines = [line for line in result.stdout.splitlines() if line.startswith("D30_PROBE=")]
    assert len(lines) == 1, diagnostic
    report = json.loads(lines[0].removeprefix("D30_PROBE="))
    assert report["status"] == 0, diagnostic
    assert report["calls"] == [dict(nodeid=nodeid, outcome="passed", xfail=False)], diagnostic
    assert report["cleanup_entries"] >= 1, "must exercise real Envelope._terminate: " + diagnostic
