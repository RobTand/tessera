"""The finite native fused T-4 serving gate.

The shared qualification harness owns numerical references, production route
intake, eager/graph comparison, and mandatory population checks. Historical
TCQ fixture receipts do not qualify the replacement WINDOW route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pytest
import torch

from experiments.t4_code import t4_fused_qualify


def run_gate(report_path=None, *, dry_run=False) -> dict:
    """Run the shared harness; preserve its exit status and structured result."""
    with tempfile.TemporaryDirectory(prefix="t4-serving-gate-") as scratch:
        target = Path(report_path) if report_path else Path(scratch) / "result.json"
        if dry_run:
            argv = ["--mode", "dry-run", "--out", str(target),
                    "--q256", *map(str, range(128, 1025, 128)),
                    "--ms", "1", "16", "2048", "4096",
                    "--experts", "2", "--top-k", "2", "--hidden", "512", "--inter", "1280",
                    "--dense-shapes", "partial:32x256", "index:128x512"]
            t4_fused_qualify.main(argv)
            report = json.loads(target.read_text())
        else:
            report = t4_fused_qualify.qualify_serving(
                q256=tuple(range(128, 1025, 128)), ms=(1, 16, 2048, 4096),
                hidden=512, inter=1280, experts=2, top_k=2,
                dense_shapes=(("partial", 32, 256), ("index", 128, 512)),
                tp_cuts=False, out=str(target))
        report["ok"] = report["complete"] is True
        if report_path:
            target.write_text(json.dumps(report, indent=1) + "\n")
        return report


def test_native_a4_serving_gate(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("native fused T-4 serving gate requires CUDA")
    assert run_gate(tmp_path / "native-t4-serving.json")["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_gate(args.report, dry_run=args.dry_run)
    raise SystemExit(0 if result["ok"] else 1)
