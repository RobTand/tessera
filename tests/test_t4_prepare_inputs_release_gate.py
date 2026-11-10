"""Regression for #1300: the editable release gate still checks declared ranges.

The defect is in the original test, not prepare_inputs. Keep this harness separate
so that migrating that test cannot silently remove its range assertions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_t4_fused_qualify as gate


def test_prepare_inputs_gate_still_checks_ranges(tmp_path, monkeypatch):
    valid = tmp_path / "valid"
    valid.mkdir()
    gate.test_prepare_inputs_declares_exact_ranges(valid)

    prepare = gate.qual.prepare_inputs
    for defect in ("phase_end", "unit_bytes", "declared_bytes"):
        scratch = tmp_path / defect
        scratch.mkdir()

        def corrupt(args, defect=defect):
            result = prepare(args)
            manifest_path = Path(result["data_manifest"])
            manifest = json.loads(manifest_path.read_text())
            unit = result["units"][0]
            if defect == "phase_end":
                manifest["annotations"]["phases"][-1]["cumulative_bytes"] -= 1
            elif defect == "unit_bytes":
                unit["bytes"] -= 1
            else:
                declared = next(entry for entry in manifest["entries"]
                                if entry["path"] == unit["path"]
                                and entry["offset"] == unit["offset"])
                declared["bytes"] -= 1
            manifest_path.write_text(json.dumps(manifest))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(gate.qual, "prepare_inputs", corrupt)
            # Direct assertions or the declared-input validator may reject corruption;
            # ImportError and skips must never masquerade as a successful gate.
            with pytest.raises((AssertionError, ValueError)):
                gate.test_prepare_inputs_declares_exact_ranges(scratch)
