#!/usr/bin/env python3
"""Check rule 12 from the pull request event, without external dependencies."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

CUTOFF = datetime(2026, 10, 9, 17, tzinfo=timezone.utc)
BRANCH = re.compile(r"tessera-([1-9][0-9]*)(?:-[a-z0-9][a-z0-9-]*)?")
REFERENCE = re.compile(
    r"(?<![\w/])(?:(?P<repo>[a-z0-9_.-]+(?:/[a-z0-9_.-]+)?))?"
    r"#(?P<number>[1-9][0-9]*)(?!\w)", re.IGNORECASE,
)
ISSUE_URL = re.compile(
    r"https://github\.com/(?P<repo>[a-z0-9_.-]+/[a-z0-9_.-]+)/issues/"
    r"(?P<number>[1-9][0-9]*)(?![\w/])", re.IGNORECASE,
)


def linked_issues(body: str, repository: str) -> set[int]:
    """Read local issue references and GitHub issue URLs."""
    repository = repository.lower()
    name = repository.rsplit("/", 1)[-1]
    issues = set()
    for match in REFERENCE.finditer(body):
        qualifier = (match.group("repo") or "").lower()
        if qualifier in {"", name, repository}:
            issues.add(int(match.group("number")))
    for match in ISSUE_URL.finditer(body):
        if match.group("repo").lower() == repository:
            issues.add(int(match.group("number")))
    return issues


def require_branch(pull_request: dict, repository: str) -> None:
    """Refuse a branch that does not match a linked issue, except legacy PRs."""
    branch = pull_request["head"]["ref"]
    created_at = datetime.fromisoformat(pull_request["created_at"].replace("Z", "+00:00"))
    if branch.startswith(("ig/", "release")) or created_at < CUTOFF:
        return

    issues = linked_issues(pull_request["body"] or "", repository)
    match = BRANCH.fullmatch(branch)
    if match is not None and int(match.group(1)) in issues:
        return

    expected = ", ".join(f"tessera-{number}" for number in sorted(issues))
    if not expected:
        expected = "tessera-<issue> after you link an issue in the pull request body"
    raise ValueError(
        f"Branch rule 12 requires tessera-<issue> or tessera-<issue>-<word>. "
        f"Branch {branch!r} does not match a linked issue. Expected: {expected}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path, help="GitHub pull request event JSON")
    parser.add_argument("repository", help="GitHub repository, in owner/name form")
    args = parser.parse_args()
    event = json.loads(args.event.read_text(encoding="utf-8"))
    try:
        require_branch(event["pull_request"], args.repository)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print("Branch rule 12 passed or the pull request is exempt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
