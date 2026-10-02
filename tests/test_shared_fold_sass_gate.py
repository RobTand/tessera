"""Exercise #799's existing-kernel SASS gate through the actual check driver."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "experiments/t8r_speed/shared_fold_check.py"
    spec = importlib.util.spec_from_file_location("shared_fold_check_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("nonidentical", [None, "multiset", "only_before", "differs"])
def test_sass_gate_requires_existing_instruction_sequences_unchanged(checker, monkeypatch,
                                                                  tmp_path, nonidentical):
    """A generic comparator's success cannot qualify a reordered existing kernel."""
    rows = {
        library: dict(identical=["token_sum_kernel"], multiset=[], differs=[], only_before=[],
                      only_after=["token_sum_shared_kernel"])
        for library in checker.SASS_LIBRARIES
    }
    if nonidentical is not None:
        rows["value"]["identical"] = []
        rows["value"][nonidentical] = ["token_sum_kernel"]

    def run(argv, **kwargs):
        if "compare" in argv:
            Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows))
        # The generic comparator accepts "multiset". Independently require
        # its per-kernel result, including when its exit status is zero.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    base = tmp_path / "base.cu"
    base.write_text("// retained base source\n")
    record = checker.sass_check(str(base), str(tmp_path))
    assert record["passed"] is (nonidentical is None)


def test_retained_sass_uses_the_same_gate_without_compiling(checker, monkeypatch, tmp_path):
    rows = {lib: dict(identical=["token_sum_kernel"], multiset=[], differs=[], only_before=[],
                      only_after=["token_sum_shared_kernel"])
            for lib in checker.SASS_LIBRARIES}

    def run(argv, **kwargs):
        assert "dump" not in argv, "a retained baseline must never be rebuilt"
        assert argv[argv.index("compare") + 1:argv.index("--json")] == ["retained-before", "retained-after"]
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(rows))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    base = tmp_path / "base.cu"
    base.write_text("// retained baseline source\n")
    result = checker.sass_check(str(base), str(tmp_path), "retained-before", "retained-after")
    assert result["passed"] is True
    assert result["mode"] == "retained_dso_sass"


def test_one_retained_sass_arm_is_refused_before_compilation(checker, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("incomplete retained inputs must not start a subprocess")

    monkeypatch.setattr(checker.subprocess, "run", unexpected)
    base = tmp_path / "base.cu"
    base.write_text("// retained baseline source\n")
    with pytest.raises(ValueError, match="both before and after"):
        checker.sass_check(str(base), str(tmp_path), "retained-before", None)
