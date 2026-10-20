"""Submission comparisons stay attached to their checkout, even after HEAD moves."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "merge_suite.py"


def _module():
    spec = importlib.util.spec_from_file_location("_merge_suite_submission", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("at_master", [True, False, None])
@pytest.mark.parametrize("restart", [False, True])
def test_master_comparison_belongs_to_the_submission_checkout(tmp_path, at_master, restart):
    """An arm placed after HEAD moves cannot borrow the earlier comparison.

    Before the fix, the moved checkout's row incorrectly inherited ``yes`` or
    ``no``, and the original checkout's row lost it. The unknown case also lost
    the captured checkout association entirely. Pre-fix failure line::

        E AssertionError: ('a moved checkout must not inherit the submission comparison', ...)
    """

    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
             *args], text=True).strip()

    git("init", "-q", "--initial-branch=measured")
    git("commit", "-q", "--allow-empty", "-m", "submitted checkout")
    submitted = git("rev-parse", "HEAD")
    moved = git("commit-tree", "HEAD^{tree}", "-p", submitted, "-m", "later checkout")
    if at_master is not None:
        git("update-ref", "refs/heads/master", submitted if at_master else moved)
    run = tmp_path / "run"
    run.mkdir()
    module = _module()
    module._write_submission_master(checkout, run)
    git("reset", "--hard", moved)
    if restart:
        module = _module()
    arms = [
        {"arm": name, "returncode": 0, "surface": {
            "commit": commit, "device": "fixture CPU", "counts": {
                "passed": 3, "failed": 0, "skipped": 0}, "not_collected": []}}
        for name, commit in (("x86", submitted), ("gpu", moved))
    ]
    receipt = module._assemble_receipt(
        arms, checkout, assembled_by="resume" if restart else "submit",
        submission_master=module._read_submission_master(run))
    ledger = tmp_path / "ledger.md"
    module._record_markdown(ledger, receipt)
    rows = [line for line in ledger.read_text().splitlines() if "fixture CPU" in line]
    original = next(row for row in rows if "| x86 |" in row)
    later = next(row for row in rows if "| gpu |" in row)
    assert "| unknown | gpu |" in later, (
        "a moved checkout must not inherit the submission comparison", later)
    expected = "unknown" if at_master is None else ("yes" if at_master else "no")
    assert f"| {expected} | x86 |" in original, original
    assert receipt["population"]["checkout_commit_at_submit"] == submitted
    assert receipt["population"]["commit"] == moved
    assert receipt["population"]["is_master_head"] is at_master
    assert receipt["arms"] == arms


@pytest.mark.parametrize("association,measured", [(None, "a" * 40), ("a" * 40, None)])
def test_a_ledger_comparison_needs_an_established_subject(tmp_path, association, measured):
    """Unknown associations and assumed measured commits cannot establish yes.

    Before the fix both cases emitted ``| yes | x86 |`` from the current commit.
    Pre-fix failure line: ``assert "| unknown | x86 |" in row, row``.
    """

    module = _module()
    ledger = tmp_path / "ledger.md"
    module._record_markdown(ledger, {
        "generated_utc": "now",
        "population": {"commit": "a" * 40, "checkout_commit_at_submit": association,
                       "is_master_head": True},
        "arms": [{"arm": "x86", "returncode": 0, "surface": {
            "commit": measured, "device": "fixture CPU", "counts": {"passed": 3}}}],
    })
    row = next(line for line in ledger.read_text().splitlines() if "fixture CPU" in line)
    assert "| unknown | x86 |" in row, row
