"""The extra-pass verifier regression must not depend on a producer child.

Run the existing case in an isolated pytest process so the launch prohibition
and verifier observer cannot leak to unrelated producer integration tests. No
clock, load generator, retry or real producer failure is needed to expose the
coupling. The observer delegates every decision to the real verifier.
"""
from pathlib import Path
import json
import subprocess
import sys

from child_env import child_env


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    "tests/test_graph_receipt_population.py::"
    "test_verify_rejects_forged_population_and_independence_claims[extra-pass]"
)

PROBE = r'''
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tessera import graph_receipt as gr

root, scratch, case = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
real_verify = gr.verify
observed = []
reports = []
patch = pytest.MonkeyPatch()


def unavailable(*args, **kwargs):
    raise AssertionError(
        "verifier-only extra-pass regression launched a subprocess; "
        "use an independent in-memory valid receipt, not the producer"
    )


def observe(receipt, serve):
    reason = real_verify(receipt, serve)
    graph = next(a for a in receipt["arms"] if a["name"] == "g")
    observed.append({
        "passes": [p["name"] for p in graph["passes"]],
        "reason": reason,
        "verdict": receipt.get("verdict"),
        "equal": graph.get("equal"),
        "replayed": graph.get("replayed_everything"),
        "attests": bool(receipt.get("attests")),
    })
    return reason


class Guard:
    def pytest_runtest_setup(self, item):
        # Before fixture setup as well as the test body: moving the producer
        # launch into another fixture must not conceal this dependency.
        patch.setattr(subprocess, "Popen", unavailable)
        patch.setattr(gr, "verify", observe)

    def pytest_runtest_logreport(self, report):
        reports.append({"when": report.when, "outcome": report.outcome,
                        "detail": str(report.longrepr) if report.failed else ""})


try:
    status = pytest.main([
        str(root / case), "-q", "--noconftest", "-o", "addopts=",
        "-p", "no:cacheprovider", "--basetemp=" + str(scratch / "pytest"),
    ], plugins=[Guard()])
finally:
    patch.undo()
print("POPULATION_ISOLATION=" + json.dumps({
    "status": int(status), "reports": reports, "observed": observed,
}))
'''


def test_extra_pass_exercises_real_verifier_without_producer_process(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT), str(tmp_path), CASE],
        cwd=ROOT,
        env=child_env(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line.removeprefix("POPULATION_ISOLATION=")
             for line in result.stdout.splitlines()
             if line.startswith("POPULATION_ISOLATION=")]
    assert len(lines) == 1, result.stdout + result.stderr
    evidence = json.loads(lines[0])
    assert evidence["status"] == 0, result.stdout + result.stderr
    assert [(r["when"], r["outcome"]) for r in evidence["reports"]] == [
        ("setup", "passed"), ("call", "passed"), ("teardown", "passed"),
    ], evidence

    # A missing/skipped/no-op attack is not an isolation fix: retain a genuine
    # positive control and an actual refusal of the forged duplicate pass.
    calls = evidence["observed"]
    assert any(c["passes"] == ["g", "g-r2"] and c["reason"] is None
               for c in calls), evidence
    assert any(
        c["passes"] == ["g", "g-r2", "g-r2"]
        and c["reason"] == (
            "malformed population for arm g: incomplete or invalid required equality passes"
        )
        and c["verdict"] == "equal" and c["equal"] is True
        and c["replayed"] is True and c["attests"]
        for c in calls
    ), evidence
