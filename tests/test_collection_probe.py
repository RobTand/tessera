"""The torch-free collection probe (tests/conftest.py) and its classifier.

The ``pure`` CI job has pytest and nothing else.  Which test modules it can
collect is a question each module answers by importing; the probe asks in ONE
child interpreter so the collecting process executes no test module before
collection, and it classifies a failed import by whether the missing name is
one this tree provides (tessera#154).
"""

import json
import subprocess
import sys
from pathlib import Path

import conftest


def test_the_own_name_set_is_read_off_the_trees_import_roots():
    own = conftest._own_import_roots()
    assert {"tessera", "box_artifacts", "conftest"} <= own, sorted(own)
    assert "torch" not in own and "numpy" not in own


def test_a_missing_own_name_is_a_failure_and_a_missing_dependency_is_a_skip():
    own = frozenset({"tessera", "tests"})
    assert conftest._third_party_import_failure(ModuleNotFoundError(name="torch"), own)
    assert not conftest._third_party_import_failure(
        ModuleNotFoundError(name="tessera.decode"), own)
    assert not conftest._third_party_import_failure(ModuleNotFoundError(name="tests.x"), own)
    assert not conftest._third_party_import_failure(ImportError("no name at all"), own)


def test_the_probe_runs_in_one_child_and_reports_only_dependency_failures(tmp_path):
    (tmp_path / "test_needs_dependency.py").write_text("import nosuchdependency_xyz\n")
    (tmp_path / "test_own_name_moved.py").write_text("import mine.moved\n")
    (tmp_path / "test_collects.py").write_text("def test_ok():\n    pass\n")
    skipped = conftest._probe_uncollectable(tmp_path, frozenset({"mine"}))
    assert skipped == ["test_needs_dependency.py"], skipped


def test_the_probe_is_one_subprocess_not_one_per_module(monkeypatch, tmp_path):
    for i in range(5):
        (tmp_path / f"test_{i}.py").write_text("import nosuchdependency_xyz\n")
    calls = []
    real_run = subprocess.run

    def counting_run(*args, **kwargs):
        calls.append(args)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counting_run)
    skipped = conftest._probe_uncollectable(tmp_path, frozenset())
    assert len(skipped) == 5 and len(calls) == 1, (len(skipped), len(calls))


def test_a_probe_child_does_not_probe_again(monkeypatch):
    """This module imports the conftest, so the child that executes it would
    otherwise start a probe of its own, and so on without end."""
    def refuse(*args, **kwargs):
        raise AssertionError("the child probed")
    monkeypatch.setenv(conftest._PROBE_MARK, "1")
    monkeypatch.setattr(subprocess, "run", refuse)
    assert conftest._uncollectable() == []
