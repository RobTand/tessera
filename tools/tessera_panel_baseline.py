#!/usr/bin/env python3
"""Judge #688 panel rows against the preserved #685 baseline bands.

The acceptance consumer for the third #688 line: "The four routed rows
reproduce the #685 after-run medians within their IQR".  The band rule is the
historical bench's own (see tessera.serving.panel_baseline); this CLI is the
thin entry point a reviewer runs.  CPU-only; it executes no measurement.

    # Prove the recorded rule over every cell of one preserved table:
    tools/tessera_panel_baseline.py verify-table --baseline TABLE.json \
        --sha256 9071f514...

    # Compare a validated tessera.shape_time_panel.v1 receipt against the
    # preserved after-table (the #685 comparison basis):
    tools/tessera_panel_baseline.py compare --baseline TABLE.json \
        --sha256 4d51fbe9... --panel panel.json --expected-runtime runtime.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving import panel_baseline as pb, timing_panel as tp  # noqa: E402


def _bound(path: Path, *, sha256=None):
    binding = tp.file_binding(path)
    if sha256 is not None:
        tp._sha(sha256, "sha256")
        if sha256 != binding["sha256"]:
            raise ValueError(f"pinned digest differs: pinned {sha256}, "
                             f"{path} is {binding['sha256']}")
    return binding


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    verify = sub.add_parser("verify-table", help="prove the recorded rule over every cell")
    verify.add_argument("--baseline", type=Path, required=True)
    verify.add_argument("--sha256")
    compare = sub.add_parser("compare", help="compare a validated panel with the baseline")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--sha256", help="pin the preserved table's digest")
    compare.add_argument("--panel", type=Path, required=True)
    compare.add_argument("--panel-sha256")
    compare.add_argument("--expected-runtime", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.action == "verify-table":
            raw = args.baseline.read_bytes()
            binding = _bound(args.baseline, sha256=args.sha256)
            proof = pb.verify_recorded_statistics(raw, where=str(args.baseline))
            result = {"schema": pb.SCHEMA, "mode": "verify_rule",
                      "baseline": binding, "proof": proof,
                      "bench_rule": {"formula": pb.BENCH_RULE, "source": pb.BENCH_RULE_SOURCE,
                                     "proven_reproduction": True}}
        else:
            raw = args.baseline.read_bytes()
            _bound(args.baseline, sha256=args.sha256)
            panel_raw = args.panel.read_bytes()
            if args.panel_sha256 is not None:
                tp._sha(args.panel_sha256, "panel sha256")
                if hashlib.sha256(panel_raw).hexdigest() != args.panel_sha256:
                    raise ValueError("panel bytes differ from the pinned digest")
            panel = tp.json_bytes(panel_raw)
            expected = tp.json_bytes(args.expected_runtime.read_bytes())
            tp.validate_panel(panel, expected_runtime=expected)
            rows = pb.panel_row_views(panel)
            result = pb.compare(raw, rows, baseline_sha256=args.sha256,
                                baseline_path=str(args.baseline.resolve()))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError) as exc:
        print(f"[panel-baseline] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
