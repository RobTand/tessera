"""The trailing-refit serve stage must fail when its KL comparison fails.

``experiments/refit_trailing_serve.sh serve ARM`` dumps one arm's logprobs and
compares them against the pair's teacher.  The script runs under
``set -uo pipefail`` and not errexit, so a failed comparison was recorded by
pipefail for its pipeline -- and then discarded, because the pipeline is the
branch's last command and the unconditional ``echo "STEP_DONE ..."; date``
after the ``case`` became the script's exit status.  A missing or corrupt
teacher, or a comparator that refuses the pair's metadata, therefore reported a
completed stage with status 0, and an orchestrator or a later gate could accept
an absent -- or an earlier attempt's -- KL receipt (tessera#251).

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

#: Stands in for ``$PY /home/rob/dq-runs/kl_tool.py compare ... --out PATH``.
#: It writes this attempt's receipt at whatever ``--out`` names, or refuses.
PY_STUB = r'''#!/bin/bash
out=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"
  prev="$a"
done
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


@pytest.fixture
def harness(tmp_path):
    """A fake repo, runs dir and KL dir; returns a runner for `serve ARM`."""
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

    runs = tmp_path / "runs"
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
            "TESSERA_KL_CORPUS": str(kldir / "corpus.json"),
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
