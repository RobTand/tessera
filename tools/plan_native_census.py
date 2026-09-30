#!/usr/bin/env python3
"""Build an unmeasured census plan; this command never runs a GPU (#689)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.serving.census_plan import build_census_plan  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path,
                        help="JSON array of explicit rank-local census scopes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        requests = json.loads(args.requests.read_text())
        if not isinstance(requests, list):
            raise ValueError("requests must be a JSON array")
        plan = build_census_plan(requests)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    except OSError as exc:
        parser.error(str(exc))
    print(f"{len(plan['rows'])} planned scopes; not executed or qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
