"""The impacted-test receipt describes the tree it actually classified."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "impacted_tests.py"
_SPEC = importlib.util.spec_from_file_location("impacted_tests", SCRIPT)
impacted = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(impacted)  # type: ignore[union-attr]


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Tessera test")
    (repo / "seed.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _selector(repo: Path, ref: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--ref", ref, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_parentless_snapshot_uses_a_direct_base_tree_comparison(
    tmp_path: Path,
) -> None:
    """A fetched base object is enough; synthetic snapshots have no ancestry."""

    repo, base = _repo(tmp_path)
    _git(repo, "checkout", "--orphan", "pbrun-snapshot")
    (repo / "seed.txt").write_text("branch\n", encoding="utf-8")
    (repo / ".pbrun-closure.0123456789abcdef.json").write_text(
        "{}\n", encoding="utf-8"
    )
    _git(repo, "add", "seed.txt", ".pbrun-closure.0123456789abcdef.json")
    _git(repo, "commit", "-qm", "parentless snapshot")
    head = _git(repo, "rev-parse", "HEAD")

    result = _selector(repo, f"{base}...HEAD")

    assert result["comparison"] == (
        f"direct {base}..{head} "
        f"(no merge base; requested {base}...HEAD)"
    )
    assert result["changed"] == 1
    assert result["verdict"] == "none"
    assert result["reason"] == "inert changed paths require no tests"


def test_only_the_exact_pbrun_closure_basename_is_excluded(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    names = [
        ".pbrun-closure.0123456789abcdef.json",
        ".pbrun-closure.0123456789abcde.json",
        ".pbrun-closure.0123456789abcdef0.json",
        ".pbrun-closure.0123456789abcdeF.json",
        "prefix.pbrun-closure.0123456789abcdef.json",
        ".pbrun-closure.0123456789abcdef.json.bak",
    ]
    for name in names:
        (repo / name).write_text("{}\n", encoding="utf-8")
    _git(repo, "add", *names)
    _git(repo, "commit", "-qm", "generated and lookalike names")

    changed, comparison = impacted.changed_files(f"{base}...HEAD", repo)

    assert comparison == f"{base}...HEAD"
    assert changed == sorted(names[1:])


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("notes.md", "inert changed paths require no tests"),
        (
            "settings.yaml",
            "non-Python changed paths have no text-matched tests",
        ),
        (
            "module.py",
            "Python import graph found no reverse-reachable tests",
        ),
    ],
)
def test_no_test_reason_names_why_the_path_selected_nothing(
    tmp_path: Path,
    name: str,
    reason: str,
) -> None:
    repo, base = _repo(tmp_path)
    (repo / name).write_text("changed\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", "change")

    result = _selector(repo, f"{base}...HEAD")

    assert result["verdict"] == "none"
    assert result["reason"] == reason
