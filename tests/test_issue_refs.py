"""Findings must not be able to go missing.

A finding in this project is born in one of four places -- a worker's report, a
handover or triage doc, a measurement receipt, or a chat -- and only one of
those four has a lifecycle: the issue tracker.  Reports die with the task, docs
are append-only and never say what happened next, and a chat is gone.  So the
rule is that **a finding not fixed in the same change becomes an issue**, and
these tests are what make breaking that rule a test failure rather than a
memory lapse.

Two checks, both offline against ``docs/issues-snapshot.json`` (refresh it with
``tools/refresh_issues.py``; a check that needs the network is a check that gets
skipped, and a skipped check is how the finding goes missing):

1. Every issue reference in ``docs/`` resolves to a real issue.  A dangling
   ``#37`` reads exactly like a tracked item and is worse than no reference,
   because it stops anyone looking further.
2. Every finding listed in a triage document carries an explicit disposition --
   ``[FIXED <sha>]``, ``[#N]`` or ``[DISMISSED]``.  A triage doc that lists a
   real bug with no disposition is the exact artifact this project keeps
   mistaking for a tracker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SNAPSHOT = DOCS / "issues-snapshot.json"

# ``#12`` or ``owner/repo#12``.  The negative lookbehind keeps us off URL
# fragments, ``file.py#L20`` anchors and hex colours; requiring a boundary after
# the digits keeps us off ``#1234abcd``-shaped things.
REF = re.compile(r"(?<![\w/])(?:([\w.-]+/[\w.-]+))?#(\d+)\b")

# A fenced code block may legitimately contain a ``#123`` that is not a
# reference (a comment, a shell fragment), so those lines are skipped.
FENCE = re.compile(r"^\s*```")

DISPOSITION = re.compile(r"\[(FIXED\b[^\]]*|DISMISSED|#\d+|[\w.-]+/[\w.-]+#\d+)\]")

# A triage document's finding lists.  Headings are matched loosely because the
# tier names carry prose ("Tier A -- real, reachable on a shipping path").
TIER_HEADING = re.compile(r"^##+\s+Tier\s+[AB]\b", re.IGNORECASE)
ANY_HEADING = re.compile(r"^##+\s")
ITEM = re.compile(r"^(?:[-*]|\d+[.)]|\d+[a-z][.)])\s+\S")


def _snapshot() -> dict:
    if not SNAPSHOT.exists():
        pytest.fail(f"{SNAPSHOT} is missing; run tools/refresh_issues.py")
    return json.loads(SNAPSHOT.read_text())


def _markdown() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if "archive" not in p.parts)


def _prose_lines(text: str):
    """Yield ``(lineno, line)`` outside fenced code blocks."""
    fenced = False
    for n, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield n, line


def test_every_issue_reference_in_the_docs_resolves():
    snap = _snapshot()
    default = snap["default_repo"]
    known = {repo: set(issues) for repo, issues in snap["repos"].items()}
    dangling = []
    for path in _markdown():
        if path == SNAPSHOT:
            continue
        for lineno, line in _prose_lines(path.read_text()):
            for repo, number in REF.findall(line):
                repo = repo or default
                if repo not in known:
                    continue  # a third-party tracker; not ours to verify
                if number not in known[repo]:
                    rel = path.relative_to(ROOT)
                    dangling.append(f"{rel}:{lineno} -> {repo}#{number}")
    assert not dangling, (
        "issue references that resolve to nothing (refresh the snapshot with "
        "tools/refresh_issues.py if these were filed after it was written):\n  "
        + "\n  ".join(dangling)
    )


def _triage_docs() -> list[Path]:
    return sorted(DOCS.rglob("*triage*.md"))


def test_a_triage_document_exists_to_check():
    """Guards the check above from silently passing on an empty set."""
    assert _triage_docs(), "no triage document found under docs/"


@pytest.mark.parametrize("path", _triage_docs(), ids=lambda p: p.name)
def test_every_triaged_finding_has_a_disposition(path: Path):
    """Every item under a Tier A/B heading says what happened to it.

    ``[FIXED <sha>]`` (it landed), ``[#N]`` (it is tracked) or ``[DISMISSED]``
    (it was checked and is not real).  Nothing else counts: "we should look at
    this" is how a finding goes missing.
    """
    undispositioned = []
    in_tier = False
    for lineno, line in _prose_lines(path.read_text()):
        if ANY_HEADING.match(line):
            in_tier = bool(TIER_HEADING.match(line))
            continue
        if not in_tier or not ITEM.match(line):
            continue
        if not DISPOSITION.search(line):
            rel = path.relative_to(ROOT)
            undispositioned.append(f"{rel}:{lineno} {line.strip()[:88]}")
    assert not undispositioned, (
        "triaged findings with no disposition -- each needs [FIXED <sha>], "
        "[#N] or [DISMISSED]:\n  " + "\n  ".join(undispositioned)
    )
