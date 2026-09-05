"""The torch-free collection probe (tests/conftest.py) and its classifier.

The ``pure`` CI job has pytest and nothing else.  Which test modules it can
collect is a question each module answers by importing; the probe asks in ONE
child interpreter so the collecting process executes no test module before
collection, and it classifies a failed import by whether the missing name is
one this tree provides (tessera#154).
"""

import json
import os
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


def test_the_conftest_collects_without_xdist_installed():
    """The ``pure`` job has no xdist, so a hook only xdist declares must be
    optional: a plain ``pytest_testnodedown`` made pluggy refuse the whole
    conftest at collection (``PluginValidationError: unknown hook``), and the
    job exited 3 having run nothing (tessera#290).  ``-p no:xdist`` is that
    job's plugin set on a box where xdist is installed."""
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:xdist", "-p", "no:cacheprovider",
         "--collect-only", "-q", str(Path(__file__).name)],
        cwd=root / "tests", capture_output=True, text=True, timeout=600)
    assert "PluginValidationError" not in proc.stderr + proc.stdout, proc.stderr[-2000:]
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])


#: Modules the torch-free collector admits -- their imports are torch-free by
#: design -- whose test bodies once reached for torch anyway (tessera#309).
_TORCH_FREE_MODULES_WITH_TORCH_REACHING_BODIES = (
    "test_hardware_byte_grid.py",
    "test_serving_native_extensions.py",
    # The forest-body roster: the refusal lives in ``alphabet``, so this
    # module's import is torch-free, and two bodies read the other home of
    # the same fact out of ``export`` (tessera#285).
    "test_forest_grid_roster.py",
)


def test_a_collectable_module_does_not_reach_torch_inside_a_test_body():
    """The collector classifies a module by its *import* (tessera#154), so a
    body that imports torch in a module whose import is torch-free collects
    in the ``pure`` job and then fails there instead of skipping -- six did,
    and master read ``6 failed`` for three merges (tessera#309).  A box with
    torch cannot see that, so hide torch from a child the way the job's
    interpreter lacks it: ``sys.modules["torch"] = None`` makes every
    ``import torch`` raise ``ModuleNotFoundError``, which ``importorskip``
    turns into the skip the job expects and a bare import turns into the
    failure it reported.  The probe is skipped in the child (the mark) so
    this exercises the bodies, not the collector."""
    tests = Path(__file__).resolve().parent
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['torch'] = None\n"
         "import pytest\n"
         "raise SystemExit(pytest.main(sys.argv[1:]))",
         "-p", "no:xdist", "-p", "no:cacheprovider", "-q",
         *_TORCH_FREE_MODULES_WITH_TORCH_REACHING_BODIES],
        cwd=tests, capture_output=True, text=True, timeout=600,
        env={**os.environ, conftest._PROBE_MARK: "1"})
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert "No module named 'torch'" not in tail, tail
    assert proc.returncode == 0, (proc.returncode, tail)
