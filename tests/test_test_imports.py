"""Test-to-test imports may only shrink as shared helpers move out of tests."""
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

import pytest


COUNT = "tests/fixtures/test_to_test_imports_count.txt"


def _test_to_test_edges(root, files=None):
    paths = (root / "tests").glob("test_*.py") if files is None else (root / name for name in files)
    modules = {path.stem: path for path in paths
               if path.parent == root / "tests" and path.match("test_*.py")
               and path.is_file() and not path.is_symlink()}
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

    return actual


def _git(root, *args, binary=False):
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=not binary, check=True,
        env={"PATH": os.defpath, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
    ).stdout
    return result if binary else result.strip()


def _merge_base(root, explicit=None):
    """Use the draft base when supplied, otherwise the merge base with master."""
    if explicit:
        return _git(root, "rev-parse", "--verify", explicit + "^{commit}")
    try:
        _git(root, "rev-parse", "--verify", "origin/master^{commit}")
        branch = "origin/master"
    except subprocess.CalledProcessError:
        branch = "master"
    try:
        return _git(root, "merge-base", "HEAD", branch)
    except subprocess.CalledProcessError:
        raise AssertionError(f"No merge base with {branch}; fetch full git history") from None


def _base_graph(root, revision):
    """Read the base's test sources and count without checking out another tree."""
    files = _git(root, "ls-tree", "-r", "--name-only", revision, "--", "tests").splitlines()
    archive = _git(root, "archive", "--format=tar", revision, "tests", binary=True)
    with tempfile.TemporaryDirectory() as temporary:
        tree = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tree, filter="data")
        edges = _test_to_test_edges(tree, files)
        count = int((tree / COUNT).read_text()) if COUNT in files else len(edges)
        return edges, count


def _check_test_imports(root, *, base_revision=None):
    base, previous_count = _base_graph(root, _merge_base(root, base_revision))
    head = _test_to_test_edges(root)
    added = head - base
    assert not added, "New test imports:\n" + "\n".join(sorted(added))

    recorded = int((root / COUNT).read_text())
    assert 0 <= recorded <= previous_count, "The test-to-test import count may only fall"
    # Credit this PR's removals even if master gained edges since recording its count.
    expected = max(0, previous_count - len(base - head))
    assert recorded == expected, f"Record test-to-test import count {expected} in {COUNT}"


def _draft_base():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path:
        pr = json.loads(Path(event_path).read_text()).get("pull_request", {})
        if pr.get("head", {}).get("ref", "").startswith("mergify/merge-queue/"):
            return pr["base"]["sha"]
    return None


def test_test_to_test_imports_only_shrink():
    _check_test_imports(Path(__file__).resolve().parents[1], base_revision=_draft_base())


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
    _commit(import_repo)
    _git(import_repo, "branch", "-f", "master", "HEAD")
    with pytest.raises(AssertionError, match="tests/test_a.py -> tests/test_c.py"):
        _check_test_imports(import_repo)


def test_the_repository_stores_no_edge_list():
    assert not (Path(__file__).resolve().parent / "fixtures/test_to_test_imports.txt").exists()


def test_a_mergify_batch_uses_its_draft_base(import_repo, tmp_path, monkeypatch):
    (import_repo / "tests/test_a.py").write_text("import test_b\nimport test_c\n")
    draft_base = _commit(import_repo)
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {
        "head": {"ref": "mergify/merge-queue/master/batch"},
        "base": {"sha": draft_base},
    }}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    _check_test_imports(import_repo, base_revision=_draft_base())


def test_missing_git_history_fails_closed(import_repo):
    _git(import_repo, "branch", "-D", "master")
    with pytest.raises(AssertionError, match="fetch full git history"):
        _check_test_imports(import_repo)


def test_an_unreachable_remote_base_does_not_fall_back_to_local_master(import_repo):
    _git(import_repo, "checkout", "-q", "--orphan", "unrelated")
    (import_repo / "tests/test_c.py").write_text("VALUE = 1\n")
    remote = _commit(import_repo)
    _git(import_repo, "update-ref", "refs/remotes/origin/master", remote)
    _git(import_repo, "switch", "-q", "change")
    with pytest.raises(AssertionError, match="fetch full git history"):
        _check_test_imports(import_repo)
