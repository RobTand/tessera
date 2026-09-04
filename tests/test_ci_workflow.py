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

import re
from pathlib import Path

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
