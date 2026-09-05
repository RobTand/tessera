"""A trailing-refit stage must fail when its tool fails, and only then.

``experiments/refit_trailing_serve.sh serve ARM`` dumps one arm's logprobs and
compares them against the pair's teacher.  The script runs under
``set -uo pipefail`` and not errexit, so a failed comparison was recorded by
pipefail for its pipeline -- and then discarded, because the pipeline is the
branch's last command and the unconditional ``echo "STEP_DONE ..."; date``
after the ``case`` became the script's exit status.  A missing or corrupt
teacher, or a comparator that refuses the pair's metadata, therefore reported a
completed stage with status 0, and an orchestrator or a later gate could accept
an absent -- or an earlier attempt's -- KL receipt (tessera#251).

``compare-drift`` had the same discarded status and could not simply propagate
it, because its expected reading IS a refusal: ``refit_trailing_bytes.py``
returns "NOT the matched pair" for the 2026-09-02 bytes, and that reading is
the whole point of the stage.  So the tool now says which happened -- 0 the
matched pair, 3 the computed NOT-matched verdict, 1 the tool failing -- and
this stage accepts the two verdicts, refuses everything else by name, and
requires that the receipt standing at
``experiments/results/refit_trailing_encoder_drift.json`` was written by this
attempt (tessera#269).

No serve is launched here: the wrapper is copied into a fake repository whose
dump helper and Python are stubs, which is how ``test_serve_wrapper_cleanup``
exercises the teacher wrapper's exit paths.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "experiments" / "refit_trailing_serve.sh"

ARM = "bjac"

#: Stands in for both Pythons the wrapper runs: ``kl_tool.py compare`` for the
#: ``serve`` stage and ``refit_trailing_bytes.py`` for ``compare-drift``.  Each
#: writes this attempt's receipt at whatever ``--out`` names, or refuses.
PY_STUB = r'''#!/bin/bash
out=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"
  prev="$a"
done
case "$1" in
*refit_trailing_bytes.py)
  echo "refit_trailing_bytes $*"
  if [ "$DRIFT_RC" = 1 ]; then
    echo "Traceback (most recent call last): FileNotFoundError" >&2
    exit 1
  fi
  if [ "$DRIFT_RECEIPT" = 1 ] && [ -n "$out" ]; then
    mkdir -p "$(dirname "$out")"
    printf '%s\n' '{"verdict":"NOT the matched pair","attempt":"this"}' > "$out"
  fi
  exit "$DRIFT_RC"
  ;;
esac
echo "kl_tool $*"
if [ "$FAIL_COMPARE" = 1 ]; then
  echo "REFUSED: teacher and student metadata disagree" >&2
  exit 7
fi
[ -z "$out" ] || printf '%s\n' '{"schema":"prismaquant.kl_compare/2","attempt":"this"}' > "$out"
'''

DUMP_STUB = r'''#!/bin/bash
# serve_and_dump_kl.sh <model-dir> <out.json> <role>
[ "$FAIL_DUMP" = 0 ] || { echo "dump FAILED" >&2; exit 3; }
printf 'npz\n' > "$2.npz"
'''


def fake_repo(tmp_path):
    """The wrapper, its helpers stubbed, and the two directories it writes."""
    repo = tmp_path / "repo"
    experiments = repo / "experiments"
    experiments.mkdir(parents=True)
    shutil.copyfile(WRAPPER, experiments / WRAPPER.name)
    (experiments / "runtime_image.sh").write_text(
        "runtime_image_pin() { echo image@sha256:stub; }\n")
    dump = experiments / "serve_and_dump_kl.sh"
    dump.write_text(DUMP_STUB)
    dump.chmod(0o755)
    py = tmp_path / "py"
    py.write_text(PY_STUB)
    py.chmod(0o755)
    return repo, experiments, py, tmp_path / "runs"


@pytest.fixture
def harness(tmp_path):
    """A fake repo, runs dir and KL dir; returns a runner for `serve ARM`."""
    repo, experiments, py, runs = fake_repo(tmp_path)
    kldir = tmp_path / "kl"
    kldir.mkdir()
    twin = runs / f"{ARM}-stock-twin"
    twin.mkdir(parents=True)
    (twin / "model.safetensors").write_text("bytes")
    teacher = kldir / "teacher.json.npz"
    teacher.write_text("teacher")

    def run(*, fail_compare=False, fail_dump=False, dump_present=True):
        if dump_present:
            (kldir / f"qwen_ts75_{ARM}.json.npz").write_text("npz")
        env = os.environ | {
            "TESSERA_REPO": str(repo),
            "TESSERA_PY": str(py),
            "TESSERA_RUNS": str(runs),
            "TESSERA_KL_DIR": str(kldir),
            "TESSERA_TEACHER": str(teacher),
            "FAIL_COMPARE": str(int(fail_compare)),
            "FAIL_DUMP": str(int(fail_dump)),
        }
        return subprocess.run(
            ["bash", str(experiments / WRAPPER.name), "serve", ARM],
            env=env, text=True, capture_output=True)

    return run, runs, kldir


def test_a_successful_compare_publishes_its_receipt_and_completes(harness):
    run, runs, _ = harness
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STEP_DONE serve" in result.stdout
    receipt = runs / f"kl_{ARM}.json"
    assert '"attempt":"this"' in receipt.read_text()
    assert not list(runs.glob(f"kl_{ARM}.json.attempt.*")), "attempt not published"


def test_a_failed_compare_fails_the_stage(harness):
    """The comparator's status is the stage's status; STEP_DONE is not
    printed by a step that did not complete."""
    run, runs, _ = harness
    result = run(fail_compare=True)
    assert result.returncode != 0, result.stdout
    assert "STEP_DONE" not in result.stdout
    assert "KL compare for bjac exited 7" in result.stderr
    assert not (runs / f"kl_{ARM}.json").exists()


def test_a_failed_compare_does_not_leave_an_older_receipt_certifying_it(harness):
    """A receipt from an earlier attempt sits at the path a reader looks in,
    and nothing in it says which attempt wrote it."""
    run, runs, _ = harness
    runs.mkdir(parents=True, exist_ok=True)
    stale = runs / f"kl_{ARM}.json"
    stale.write_text('{"attempt":"earlier"}')
    result = run(fail_compare=True)
    assert result.returncode != 0, result.stdout
    assert not stale.exists(), "the earlier attempt's receipt still certifies this one"
    assert (runs / f"kl_{ARM}.json.stale").read_text() == '{"attempt":"earlier"}'
    assert "moved to" in result.stderr


def test_a_failed_compare_leaves_no_partial_receipt(harness):
    run, runs, _ = harness
    result = run(fail_compare=True)
    assert result.returncode != 0
    assert list(runs.glob(f"kl_{ARM}.json.attempt.*")) == []


def test_the_comparison_log_is_written_either_way(harness):
    run, runs, _ = harness
    run(fail_compare=True)
    assert "REFUSED" in (runs / f"kl_{ARM}.log").read_text()


def test_a_failed_dump_still_fails_the_stage(harness):
    """The dump leg was already propagated; it stays that way."""
    run, runs, _ = harness
    result = run(fail_dump=True, dump_present=False)
    assert result.returncode != 0, result.stdout
    assert "STEP_DONE" not in result.stdout


# --- compare-drift: the verdict is the reading, the tool failing is not ------

DRIFT_RECEIPT = "experiments/results/refit_trailing_encoder_drift.json"


@pytest.fixture
def drift_harness(tmp_path):
    """The same fake repo; returns a runner for `compare-drift`, the runs dir
    and the receipt path the stage names."""
    repo, experiments, py, runs = fake_repo(tmp_path)

    def run(*, rc=3, receipt_written=True):
        env = os.environ | {
            "TESSERA_REPO": str(repo),
            "TESSERA_PY": str(py),
            "TESSERA_RUNS": str(runs),
            "DRIFT_RC": str(rc),
            "DRIFT_RECEIPT": str(int(receipt_written)),
        }
        return subprocess.run(
            ["bash", str(experiments / WRAPPER.name), "compare-drift"],
            env=env, text=True, capture_output=True)

    return run, runs, repo / DRIFT_RECEIPT


def test_a_drift_verdict_completes_the_stage(drift_harness):
    """"NOT the matched pair" is what this stage exists to report: it is a
    reading, not a failure, and it must not be turned into one."""
    run, _, receipt = drift_harness
    result = run(rc=3)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STEP_DONE compare-drift" in result.stdout
    assert '"attempt":"this"' in receipt.read_text()


def test_a_matched_verdict_also_completes_the_stage(drift_harness):
    run, _, receipt = drift_harness
    result = run(rc=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STEP_DONE compare-drift" in result.stdout
    assert receipt.exists()


def test_a_failed_tool_is_not_read_as_a_drift_reading(drift_harness):
    """A missing export, an unreadable manifest or any uncaught exception
    exits 1, which is not a verdict; the stage refuses it by name."""
    run, _, receipt = drift_harness
    result = run(rc=1)
    assert "STEP_DONE" not in result.stdout, "a failed tool completed the stage"
    assert result.returncode == 1, result.stdout
    assert ("refit_trailing_bytes failed (exit 1), not a drift reading"
            in result.stderr), result.stderr
    assert not receipt.exists()


def test_a_failed_tool_does_not_leave_an_older_receipt_certifying_it(
        drift_harness):
    """A receipt from an earlier run sits at the path a reader looks in, and
    nothing in it says which run wrote it (the pattern tessera#251 gave the
    serve stage)."""
    run, _, receipt = drift_harness
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"attempt":"earlier"}')
    result = run(rc=1)
    assert not receipt.exists(), "an earlier run's receipt still certifies this one"
    assert result.returncode != 0, result.stdout
    assert Path(f"{receipt}.stale").read_text() == '{"attempt":"earlier"}'


def test_a_verdict_that_wrote_no_receipt_is_refused(drift_harness):
    """The status alone is not the reading: the stage names a receipt, so it
    fails when nothing was written by this attempt."""
    run, _, receipt = drift_harness
    result = run(rc=3, receipt_written=False)
    assert result.returncode != 0, "a stage that produced no receipt succeeded"
    assert "STEP_DONE" not in result.stdout
    assert "wrote no" in result.stderr, result.stderr
    assert not receipt.exists()


def test_the_drift_log_is_written_either_way(drift_harness):
    run, runs, _ = drift_harness
    run(rc=3)
    assert "refit_trailing_bytes" in (runs / "compare_drift.log").read_text()
    run(rc=1)
    assert "Traceback" in (runs / "compare_drift.log").read_text()
