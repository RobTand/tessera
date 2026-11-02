"""Test-to-test imports may only shrink as shared helpers move out of tests."""
import ast
from pathlib import Path
import os
import subprocess

import pytest


def _check_test_imports(root):
    tests = root / "tests"
    modules = {path.stem: path for path in tests.glob("test_*.py")}
    actual = set()
    for importer, path in sorted(modules.items()):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.module in (None, "tests"):
                    names.extend(alias.name for alias in node.names)
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "importlib"
                  and node.func.attr == "import_module"
                  and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            for name in names:
                imported = name.removeprefix("tests.")
                if imported in modules:
                    actual.add(f"tests/{importer}.py -> tests/{imported}.py")

    recorded = set((tests / "fixtures/test_to_test_imports.txt").read_text().splitlines())
    assert actual <= recorded, "Unrecorded test imports:\n" + "\n".join(sorted(actual - recorded))
    assert recorded <= actual, "Remove vanished test imports:\n" + "\n".join(sorted(recorded - actual))


def test_test_to_test_imports_only_shrink():
    _check_test_imports(Path(__file__).resolve().parents[1])


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True,
        env={"PATH": os.defpath, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
    ).stdout.strip()


def _commit(root):
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Import check", "-c", "user.email=imports@example.invalid",
         "commit", "-qm", "synthetic graph")
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def import_repo(tmp_path):
    root = tmp_path / "repo"
    fixtures = root / "tests/fixtures"
    fixtures.mkdir(parents=True)
    (root / "tests/test_a.py").write_text("import test_b\n")
    (root / "tests/test_b.py").write_text("")
    (root / "tests/test_c.py").write_text("")
    (fixtures / "test_to_test_imports_count.txt").write_text("1\n")
    # A stale legacy list reproduces the merge-batch failure before its deletion.
    (fixtures / "test_to_test_imports.txt").write_text("")
    _git(root, "init", "-q", "-b", "master")
    _commit(root)
    _git(root, "switch", "-qc", "change")
    return root


def test_edges_already_in_the_base_pass(import_repo):
    _check_test_imports(import_repo)


def test_new_head_only_edges_fail_by_name(import_repo):
    (import_repo / "tests/test_a.py").write_text("import test_b\nimport test_c\n")
    with pytest.raises(AssertionError, match="tests/test_a.py -> tests/test_c.py"):
        _check_test_imports(import_repo)


def test_removals_pass_with_a_lower_count(import_repo):
    (import_repo / "tests/test_a.py").write_text("")
    (import_repo / "tests/fixtures/test_to_test_imports_count.txt").write_text("0\n")
    _check_test_imports(import_repo)


def test_removals_require_recording_the_lower_count(import_repo):
    (import_repo / "tests/test_a.py").write_text("")
    with pytest.raises(AssertionError, match="Record test-to-test import count 0"):
        _check_test_imports(import_repo)


def test_the_count_cannot_be_raised(import_repo):
    (import_repo / "tests/fixtures/test_to_test_imports_count.txt").write_text("2\n")
    with pytest.raises(AssertionError, match="count may only fall"):
        _check_test_imports(import_repo)


def test_master_gains_do_not_spend_this_prs_count(import_repo):
    _git(import_repo, "switch", "-q", "master")
    (import_repo / "tests/test_b.py").write_text("import test_c\n")
    _commit(import_repo)
    _git(import_repo, "switch", "-q", "change")
    _git(import_repo, "rebase", "master")
    _check_test_imports(import_repo)
    (import_repo / "tests/test_a.py").write_text("")
    (import_repo / "tests/fixtures/test_to_test_imports_count.txt").write_text("0\n")
    _check_test_imports(import_repo)


def test_the_remote_master_ref_takes_precedence(import_repo):
    base = _git(import_repo, "rev-parse", "master")
    _git(import_repo, "update-ref", "refs/remotes/origin/master", base)
    _git(import_repo, "switch", "-q", "master")
    (import_repo / "tests/test_a.py").write_text("import test_b\nimport test_c\n")
    _commit(import_repo)
    _git(import_repo, "switch", "-q", "change")
    (import_repo / "tests/test_a.py").write_text("import test_b\nimport test_c\n")
    with pytest.raises(AssertionError, match="tests/test_a.py -> tests/test_c.py"):
        _check_test_imports(import_repo)


def test_the_repository_stores_no_edge_list():
    assert not (Path(__file__).resolve().parent / "fixtures/test_to_test_imports.txt").exists()
