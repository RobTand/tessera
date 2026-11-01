"""#1310: deterministic consumer coverage with only the published PB client.

Recovered from the reviewed, unlanded 6700f0a6 change and extended to the T4
consumer reported in the issue discussion. These responses test Tessera's
boundary/refusals, not PrismaBuild's codec or native/GPU qualification. Keep
the existing optional real-SDK integration cases unchanged.
"""
from __future__ import annotations

import builtins
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace

import pytest
from _whole_phase_sdk import whole_phase_sdk

import test_value_prefetch_numeric as gate
import test_t4_prefetch_staged_qualification as t4_gate
from experiments.t4_code import prefetch_qualification as qualified


@pytest.fixture
def public_sdk(monkeypatch, whole_phase_sdk):
    # Exercise the real public parser/layout snapshot. A JSON-only fake cannot
    # establish the coverage and phase guarantees the consumers now rely on.
    state = SimpleNamespace(encoding="identity", wrong_read_order=False, reads=[],
                            client=whole_phase_sdk)
    parse = whole_phase_sdk.read_data_manifest
    read_entries = whole_phase_sdk.manifest_read_entries

    def read_data_manifest(path):
        manifest, _ = parse(path)
        return manifest, state.encoding

    def manifest_read_entries(manifest):
        state.reads.append(manifest)
        entries = read_entries(manifest)
        if state.wrong_read_order:
            entries = [dict(entry) for entry in entries]
            entries[0]["offset"] += 1
        return entries

    monkeypatch.setattr(whole_phase_sdk, "read_data_manifest", read_data_manifest)
    monkeypatch.setattr(whole_phase_sdk, "manifest_read_entries", manifest_read_entries)
    original_import = builtins.__import__

    def public_only(name, globals=None, locals=None, fromlist=(), level=0):
        source = (globals or {}).get("__file__", "")
        if source.rsplit("/", 1)[-1] in {
            "value_prefetch_numeric.py", "prefetch_qualification.py", "pb_staged_store.py"
        }:
            if name == "prismabuild":
                assert set(fromlist) <= {"client"}, (
                    "readset validation must use the public PB client, "
                    f"not private imports {fromlist!r}"
                )
            elif name.startswith("prismabuild."):
                assert name == "prismabuild.client", (
                    "readset validation imports a private PB module: " + name
                )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", public_only)
    return state


@pytest.mark.parametrize("corrupt_phase", [False, True])
def test_original_whole_bank_cases_use_only_public_sdk(
    tmp_path, public_sdk, corrupt_phase,
):
    # Execute both original release selectors without an optional-dependency skip.
    gate.test_public_pb_readset_and_whole_bank_phase_contract(tmp_path, corrupt_phase, public_sdk.client)
    if not corrupt_phase:
        assert len(public_sdk.reads) == 1, "public read-order validation was bypassed"


@pytest.mark.parametrize("fault", [None, "offset", "duplicate-path", "order"])
def test_t4_original_whole_file_cases_use_only_public_sdk(tmp_path, public_sdk, fault):
    t4_gate.test_public_manifest_validator_enforces_exact_whole_file_members(tmp_path, fault, public_sdk.client)
    if fault is None:
        assert len(public_sdk.reads) == 1, "public read-order validation was bypassed"


def _manifest(consumer, tmp_path):
    if consumer == "numeric":
        entries = [{"path": str(tmp_path / "bank.pt"), "offset": 0, "bytes": 32, "sha256": "0" * 64}]
        return gate.numeric.validate_readset, "whole-synthetic-bank", entries
    entries = [{"path": path, "offset": 0, "bytes": size, "sha256": digest}
               for path, (size, digest) in qualified.native_members().items()]
    return qualified.validate_readset, qualified.PHASE, entries


@pytest.mark.parametrize("consumer", ["numeric", "t4"])
@pytest.mark.parametrize("defect", [
    "encoding", "read_order", "phase_name", "short_frontier", "boolean_frontier",
    "missing_phases", "extra_nonempty_phase",
])
def test_public_sdk_migration_preserves_readset_refusals(tmp_path, public_sdk, consumer, defect):
    validate, phase_name, entries = _manifest(consumer, tmp_path)
    total = sum(entry["bytes"] for entry in entries)
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "total_bytes": total, "entry_count": len(entries),
        "mount_prefix": os.path.commonpath([str(Path(entry["path"]).parent) for entry in entries]), "produced_by": {"test": "public SDK boundary"},
        "entries": entries,
        "annotations": {"phases": [
            {"name": phase_name, "cumulative_bytes": total},
        ]},
    }
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    validate(target)  # Every refusal mutation starts from an accepted control.
    message = "staging phase|whole-phase|phase|frontier"
    if defect == "encoding":
        public_sdk.encoding = "zstd"
    elif defect == "read_order":
        public_sdk.wrong_read_order = True
        message = "staging read order|consumption order"
    elif defect == "phase_name":
        manifest["annotations"]["phases"][0]["name"] = "not-the-whole-bank"
    elif defect == "short_frontier":
        manifest["annotations"]["phases"][0]["cumulative_bytes"] -= 1
    elif defect == "boolean_frontier":
        manifest["annotations"]["phases"][0]["cumulative_bytes"] = True
    elif defect == "missing_phases":
        manifest["annotations"] = {}
    else:
        # An empty trailing phase is valid in the public SDK. Split the whole
        # bank into two nonempty phases to test the consumer's actual refusal.
        manifest["annotations"]["phases"][0]["cumulative_bytes"] = entries[0]["bytes"] if len(entries) > 1 else total // 2
        manifest["annotations"]["phases"].append({"name": "extra", "cumulative_bytes": total})
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validate(target)
