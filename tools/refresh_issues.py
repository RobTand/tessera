#!/usr/bin/env python3
"""Regenerate ``docs/issues-snapshot.json`` from GitHub issues and pull requests.

The snapshot exists so that ``tests/test_issue_refs.py`` can check every issue
reference in the docs **offline**.  A test that needs the network is a test that
gets skipped, and a reference check that gets skipped is how a finding goes
missing -- which is the whole problem this pair of files is here to solve.

    python tools/refresh_issues.py           # both repos
    python tools/refresh_issues.py --check   # exit 1 if the snapshot is stale

Issues and pull requests share a numbered namespace, and documentation may
reference either. Run it after filing, opening, merging or closing anything.
It is not run by the test suite.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOS = ("RobTand/tessera", "RobTand/prismaquant")
SNAPSHOT = Path(__file__).resolve().parent.parent / "docs" / "issues-snapshot.json"


def rows(out: str):
    """Every issue row in gh's paginated output, whatever shape it came in.

    ``gh api --paginate`` prints one JSON array per page, concatenated, which
    is not a JSON document; ``--slurp`` wraps those pages in a single array
    and exists only in gh 2.53 and later.  Requiring it made this tool
    unrunnable on a box with gh 2.45 -- ``unknown flag: --slurp`` -- and the
    snapshot the offline reference gate reads cannot be refreshed from a box
    that cannot run this.  Decoding values off the stream reads both shapes,
    so the refresher does not depend on the gh version it happens to find.
    """

    decoder = json.JSONDecoder()
    index = 0
    while True:
        while index < len(out) and out[index].isspace():
            index += 1
        if index >= len(out):
            return
        value, index = decoder.raw_decode(out, index)
        for item in value:
            if isinstance(item, list):       # a --slurp'd array of pages
                yield from item
            else:
                yield item


def fetch(repo: str) -> dict[str, dict]:
    out = subprocess.run(
        ["gh", "api", "--paginate",
         f"repos/{repo}/issues?state=all&per_page=100"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {
        str(row["number"]): {"title": row["title"], "state": row["state"].upper()}
        for row in rows(out)
    }


def build() -> dict:
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "default_repo": REPOS[0],
        "repos": {repo: fetch(repo) for repo in REPOS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed snapshot and the repository "
                         "disagree: an issue the snapshot lacks, one it still "
                         "lists that the repository no longer has, or one "
                         "whose state moved")
    a = ap.parse_args()
    fresh = build()
    if a.check:
        if not SNAPSHOT.exists():
            print("no snapshot; run tools/refresh_issues.py", file=sys.stderr)
            return 1
        old = json.loads(SNAPSHOT.read_text())
        stale = False
        for repo, issues in fresh["repos"].items():
            known = old.get("repos", {}).get(repo, {})
            missing = sorted(set(issues) - set(known), key=int)
            # The other direction, which nothing checked: an ID the snapshot
            # still lists and the repository no longer has -- deleted, or
            # transferred out. `tests/test_issue_refs.py` builds its allowed
            # set straight out of this file, so a stale ID keeps validating
            # every documentation reference to an issue that is gone (#220).
            removed = sorted(set(known) - set(issues), key=int)
            moved = sorted(
                (n for n in set(issues) & set(known)
                 if issues[n]["state"] != known[n]["state"]), key=int)
            if missing:
                print(f"{repo}: not in the snapshot: {missing}", file=sys.stderr)
                stale = True
            if removed:
                print(f"{repo}: in the snapshot but not in the repository: "
                      f"{removed}", file=sys.stderr)
                stale = True
            if moved:
                print(f"{repo}: state changed: {moved}", file=sys.stderr)
                stale = True
        return 1 if stale else 0
    SNAPSHOT.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    counts = {r: len(v) for r, v in fresh["repos"].items()}
    print(f"wrote {SNAPSHOT.relative_to(SNAPSHOT.parent.parent)}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
