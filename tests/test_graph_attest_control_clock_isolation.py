"""The CPU control lifecycle must not spend a real serving-work deadline.

Exercise the existing test entry point, not a second implementation of the
protocol. A deterministic advance of the controller's ambient monotonic clock
at its first payload launch models descheduling without sleeping or creating
load. Real subprocess waits remain generous failure bounds. The success-path
fixture must own an injected protocol clock; production deadlines must not be
removed or relaxed to satisfy this test.
"""
from pathlib import Path
import subprocess
import sys

import pytest

import test_graph_attest_determinism_control as lifecycle


BOOTSTRAP = r'''
from pathlib import Path
import runpy
import subprocess
import sys
import time

script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
sys.argv = sys.argv[1:]
real_monotonic = time.monotonic
real_popen = subprocess.Popen
offset = 0.0


def monotonic():
    return real_monotonic() + offset


def launch(*args, **kwargs):
    global offset
    # This hook runs in a controller, not in the parent pytest process.
    # Its first Popen is the local simulated payload. Advance once, at an
    # observed lifecycle transition; never wait for an elapsed duration.
    if not offset:
        offset = 60.0
        print("CONTROL_CLOCK_ADVANCED_AT_PAYLOAD_LAUNCH", flush=True)
    return real_popen(*args, **kwargs)


time.monotonic = monotonic
subprocess.Popen = launch
runpy.run_path(str(script), run_name="__main__")
'''


@pytest.mark.parametrize("phase", ["control", "pm"])
def test_matching_off_lifecycle_is_independent_of_ambient_work_clock(tmp_path, monkeypatch, phase):
    bootstrap = tmp_path / "controller_clock_probe.py"
    bootstrap.write_text(BOOTSTRAP)
    real_popen = subprocess.Popen
    script = Path(lifecycle.__file__).resolve()
    controllers = []

    def launch(argv, *args, **kwargs):
        command = list(argv)
        if (len(command) > 2 and command[0] == sys.executable
                and command[1] == str(script) and command[2] == "--control-rank"):
            controllers.append(command)
            command = [sys.executable, str(bootstrap), *command[1:]]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    try:
        lifecycle.test_real_rank_lifecycle_restarts_both_ranks_and_retains_mismatch_without_failing(
            tmp_path, phase, False)
    except (AssertionError, subprocess.TimeoutExpired) as exc:
        diagnostics = "\n".join(
            f"rank {rank}:\n" + (tmp_path / f"controller{rank}.log").read_text()
            for rank in (0, 1))
        pytest.fail(f"CPU lifecycle depends on ambient work-clock progress: {exc}\n{diagnostics}")

    assert len(controllers) == 2, "both real controller processes must execute"
    for rank in (0, 1):
        log = (tmp_path / f"controller{rank}.log").read_text()
        assert "CONTROL_CLOCK_ADVANCED_AT_PAYLOAD_LAUNCH" in log, log

    # Independently inspect the real protocol records as well as invoking the
    # existing lifecycle assertions (including owned child death and restarts).
    import json
    root = tmp_path / "run"
    names = (("control_off_first", "control_off_restart") if phase == "control"
             else ("control_pm_off", "control_piece_major"))
    for rank in (0, 1):
        outcome = json.loads((root / f"outcome-rank{rank}.json").read_text())
        assert outcome["returncode"] == 0, outcome
        assert outcome["completed_arms"] == list(names), outcome
        assert outcome["simulation"] is True and outcome["ownership_released"] is False
        assert "cleanup_error" not in outcome, outcome
        for name in names:
            cleaned = json.loads((root / f"{name}-cleaned-rank{rank}.json").read_text())
            assert cleaned["containers_empty"] is True
            assert cleaned["gpu_descendants_empty"] is True
