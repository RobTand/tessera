"""Regression for #1300: exercise the editable release gate through public SDK APIs.

The defect is in the original test, not prepare_inputs. Keep this harness separate
so that migrating that test cannot silently remove its range assertions.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

import test_t4_fused_qualify as gate


def test_prepare_inputs_gate_uses_public_sdk_and_still_checks_ranges(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def public_sdk_only(name, globals=None, locals=None, fromlist=(), level=0):
        # Restrict this consumer, not the SDK's own internal imports. The old
        # helper's relocation is not a new public compatibility promise.
        if globals is gate.__dict__ and name.startswith("prismabuild."):
            assert name == "prismabuild.client", (
                "T4 release gate depends on a private PrismaBuild module: " + name
                + "; use the public prismabuild.client manifest APIs"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", public_sdk_only)
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
            # Direct assertions or public SDK validation may reject corruption;
            # ImportError and skips must never masquerade as a successful gate.
            with pytest.raises((AssertionError, ValueError)):
                gate.test_prepare_inputs_declares_exact_ranges(scratch)
