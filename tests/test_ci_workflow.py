"""The hosted workflows, held to the properties a reviewer cannot see.

A workflow file is code that runs with the repository's credentials, and the
two properties that make it safe are invisible in a diff of the thing it
protects: *what* code an ``uses:`` line actually runs, and *which* commit a
publish is cut from.  Both are asserted here, from the file itself, so a
regression is a red test rather than a supply-chain incident nobody sees.

Torch-free and stdlib-only by construction -- no YAML parser either, because
the bytes-only CI job installs pytest and nothing else, and a test that
``conftest`` has to skip in the one place it could catch the regression is
not a test.  The reads below are textual and fail closed: an anchor this file
cannot find is a failure, never a silent pass.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

#: ``owner/repo@ref``, plus whatever trails it on the line (a comment, we hope).
USES = re.compile(r"^\s*-?\s*uses:\s*(?P<action>[^\s#]+)(?P<rest>.*)$")
#: A git commit object name: the only ref a third party cannot move under us.
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
#: The version the SHA was, without which the pin is unreadable and unbumpable.
VERSION_COMMENT = re.compile(r"#\s*v\d+(\.\d+)*")


def _workflow_files() -> list[Path]:
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, f"no workflow files under {WORKFLOWS}"
    return files


def test_every_action_is_pinned_to_a_commit_sha():
    """A tag or a branch in ``uses:`` is code someone else can change later.

    ``actions/checkout@v4`` runs whatever ``v4`` points at *when the job
    starts*, and the same is true of ``@release/v1``.  Both are mutable refs
    owned by another account: whoever controls them controls a step that runs
    beside this repository's OIDC token.  The rule is derived, not a roster --
    every ``uses:`` in every workflow, whatever the action.
    """
    unpinned = []
    for path in _workflow_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = USES.match(line)
            if match is None:
                continue
            action = match.group("action")
            ref = action.partition("@")[2]
            if not COMMIT_SHA.match(ref):
                unpinned.append(f"{path.name}:{number}: {action} is not a commit SHA")
            elif not VERSION_COMMENT.search(match.group("rest")):
                unpinned.append(
                    f"{path.name}:{number}: {action} has no '# vX.Y.Z' comment, "
                    "so nobody can read or bump the pin"
                )
    assert not unpinned, "unpinned or unlabelled actions:\n" + "\n".join(unpinned)


# ---------------------------------------------------------------------------
# The publish gate: which commit a tag is allowed to publish.
# ---------------------------------------------------------------------------

#: The one home of the reachability rule.  The workflow calls it; so do these
#: tests, which is the point of it being a file and not an inline `run:` block.
ANCESTRY_SCRIPT = ROOT / ".github" / "scripts" / "require_tag_on_master.sh"


def _job_block(text: str, name: str) -> str:
    """The body of one job, sliced by indentation.

    Fails closed: an anchor that is not there is an error, never an empty
    block that every ``in`` below would then pass on.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line == f"  {name}:"]
    assert len(starts) == 1, f"expected exactly one '  {name}:' line, found {len(starts)}"
    body = []
    for line in lines[starts[0] + 1 :]:
        if line.strip() and not line.startswith("    "):
            break
        body.append(line)
    assert body, f"job {name!r} has an empty body"
    return "\n".join(body)


def test_publish_gates_the_tag_before_it_builds_anything():
    """A ``v*`` tag is not a review gate, so the job needs one of its own.

    Three properties, and each is load-bearing on its own: the ancestry check
    is called at all; it runs before the build, so an unreviewable commit is
    refused rather than packaged and then refused; and the checkout is deep,
    because ``git merge-base --is-ancestor`` cannot answer reachability from
    a shallow clone.  ``needs: pure`` is asserted here too -- it is what makes
    the tag path run the same tests the branch path does.
    """
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    publish = _job_block(text, "publish")

    assert "needs: pure" in publish, "publish must not run before the pure job is green"
    assert "fetch-depth: 0" in publish, (
        "the publish checkout must be deep: reachability cannot be answered "
        "from a shallow clone"
    )
    call = ANCESTRY_SCRIPT.relative_to(ROOT).as_posix()
    assert call in publish, f"publish never calls {call}"

    lines = publish.splitlines()
    check_at = next(i for i, line in enumerate(lines) if call in line)
    build_at = next(i for i, line in enumerate(lines) if "python -m build" in line)
    assert check_at < build_at, (
        "the ancestry check runs after the build; refuse the commit before "
        "packaging it"
    )


#: A job-level key sits at four spaces; a step's own keys are deeper.  The
#: depth is the discriminator, so `environment:` under some future step cannot
#: be mistaken for the gate on the job.
JOB_ENVIRONMENT = re.compile(r"^    environment:\s*(?P<name>\S+)\s*$", re.M)


def test_publish_runs_inside_a_protected_environment():
    """The ancestry check says *which commit*; the environment says *who says so*.

    Reachability from master is a property of the commit, and a commit is on
    master because a review put it there -- but nothing in the tag path asks a
    person before the OIDC token is minted.  A GitHub ``environment`` is the
    only place that question can be asked, because it is the only gate that
    sits between the job being scheduled and its credentials existing.  The
    protection itself lives in repository settings, which no test in this tree
    can read; what is asserted here is the half that lives in the file, and
    without which the settings half has nothing to attach to.

    Job-level, by indentation: an ``environment:`` on a step gates nothing.
    """
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    publish = _job_block(text, "publish")
    found = JOB_ENVIRONMENT.search(publish)
    assert found is not None, (
        "the publish job declares no job-level `environment:`, so the OIDC "
        "token is minted with no gate between a pushed tag and PyPI"
    )
    assert found.group("name") not in {"", "~", "null"}, (
        f"publish names environment {found.group('name')!r}, which gates nothing"
    )


# --- the script's own behaviour, on real repositories -----------------------

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
}


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, env=GIT_ENV, check=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    (repo / "file").write_text(message, encoding="utf-8")
    _git(repo, "add", "file")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def origin_and_clone(tmp_path):
    """An origin with two commits on master and one on a side branch, plus a
    full clone of it -- the shape a tag build sees."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    # Name the branch here rather than inheriting whatever the installation
    # defaults to: the script under test takes the branch as an argument, so
    # the fixture has to say which one it built, and `git init -b` is a newer
    # spelling than `symbolic-ref` (the suite runs on two boxes).
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/master")
    first = _commit(origin, "one")
    tip = _commit(origin, "two")
    _git(origin, "checkout", "-q", "-b", "side")
    side = _commit(origin, "side")
    _git(origin, "checkout", "-q", "master")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True, env=GIT_ENV, capture_output=True, text=True,
    )
    return SimpleNamespace(origin=origin, clone=clone, first=first, tip=tip, side=side)


def _run_check(repo: Path, commit: str, *extra: str):
    return subprocess.run(
        ["bash", str(ANCESTRY_SCRIPT), commit, *extra],
        cwd=str(repo), capture_output=True, text=True, env=GIT_ENV,
    )


def test_a_commit_on_master_is_publishable(origin_and_clone):
    for name, commit in (("tip", origin_and_clone.tip), ("older", origin_and_clone.first)):
        done = _run_check(origin_and_clone.clone, commit)
        assert done.returncode == 0, (
            f"{name} commit on master refused: {done.stdout}{done.stderr}"
        )


def test_a_commit_off_master_is_refused(origin_and_clone):
    """The whole point: a tag pushed at an unreviewed commit publishes nothing."""
    done = _run_check(origin_and_clone.clone, origin_and_clone.side)
    assert done.returncode != 0, "a commit that is not on master was accepted"
    assert origin_and_clone.side[:12] in (done.stdout + done.stderr)


def test_a_shallow_checkout_is_refused_rather_than_guessed(origin_and_clone):
    """A shallow clone answers reachability from the history it happens to
    have.  Refuse, and name the checkout setting that fixes it."""
    shallow = origin_and_clone.clone.parent / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1",
         f"file://{origin_and_clone.origin}", str(shallow)],
        check=True, env=GIT_ENV, capture_output=True, text=True,
    )
    done = _run_check(shallow, origin_and_clone.tip)
    assert done.returncode != 0, "a shallow clone was accepted"
    assert "fetch-depth" in (done.stdout + done.stderr)


def test_a_branch_that_does_not_exist_is_refused(origin_and_clone):
    """Fail closed on the ref, too: no branch, no publish."""
    done = _run_check(origin_and_clone.clone, origin_and_clone.tip, "origin", "no-such-branch")
    assert done.returncode != 0, "a missing branch was treated as reachable"
    assert "no-such-branch" in (done.stdout + done.stderr), (
        "the refusal must name the branch it could not read; any non-zero "
        "exit passes this test otherwise, including the script being absent"
    )
