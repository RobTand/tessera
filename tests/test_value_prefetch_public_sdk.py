"""#1310: deterministic consumer coverage with only the published PB client.

Recovered from the reviewed, unlanded 6700f0a6 change and extended to the T4
consumer reported in the issue discussion. These responses test Tessera's
boundary/refusals, not PrismaBuild's codec or native/GPU qualification. Keep
the existing optional real-SDK integration cases unchanged.
"""
from __future__ import annotations

import builtins
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import test_value_prefetch_numeric as gate
import test_t4_prefetch_staged_qualification as t4_gate
from experiments.t4_code import prefetch_qualification as qualified


@pytest.fixture
def public_sdk(monkeypatch):
    state = SimpleNamespace(encoding="identity", wrong_read_order=False, reads=[])
    package = ModuleType("prismabuild")
    package.__path__ = []
    client = ModuleType("prismabuild.client")

    def read_data_manifest(path):
        return json.loads(path.read_text()), state.encoding

    def manifest_read_entries(manifest):
        state.reads.append(manifest)
        entries = [dict(entry) for entry in manifest["entries"]]
        if state.wrong_read_order:
            entries[0]["offset"] += 1
        return entries

    client.read_data_manifest = read_data_manifest
    client.manifest_read_entries = manifest_read_entries
    package.client = client
    monkeypatch.setitem(sys.modules, "prismabuild", package)
    monkeypatch.setitem(sys.modules, "prismabuild.client", client)
    original_import = builtins.__import__

    def public_only(name, globals=None, locals=None, fromlist=(), level=0):
        # Restrict consumers, not the SDK's own internal implementation imports.
        if globals is gate.numeric.__dict__ or globals is qualified.__dict__:
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
    gate.test_public_pb_readset_and_whole_bank_phase_contract(tmp_path, corrupt_phase)
    if not corrupt_phase:
        assert len(public_sdk.reads) == 1, "public read-order validation was bypassed"


@pytest.mark.parametrize("fault", [None, "offset", "duplicate-path", "order"])
def test_t4_original_whole_file_cases_use_only_public_sdk(tmp_path, public_sdk, fault):
    t4_gate.test_public_manifest_validator_enforces_exact_whole_file_members(tmp_path, fault)
    if fault is None:
        assert len(public_sdk.reads) == 1, "public read-order validation was bypassed"


def _manifest(consumer, tmp_path):
    if consumer == "numeric":
        entries = [{"path": str(tmp_path / "bank.pt"), "offset": 0, "bytes": 32}]
        return gate.numeric.validate_readset, "whole-synthetic-bank", entries
    entries = [{"path": path, "offset": 0, "bytes": size, "sha256": digest}
               for path, (size, digest) in qualified.native_members().items()]
    return qualified.validate_readset, qualified.PHASE, entries


@pytest.mark.parametrize("consumer", ["numeric", "t4"])
@pytest.mark.parametrize("defect", [
    "encoding", "read_order", "phase_name", "short_frontier", "boolean_frontier",
    "missing_phases", "extra_phase",
])
def test_public_sdk_migration_preserves_readset_refusals(tmp_path, public_sdk, consumer, defect):
    validate, phase_name, entries = _manifest(consumer, tmp_path)
    total = sum(entry["bytes"] for entry in entries)
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "total_bytes": total,
        "entries": entries,
        "annotations": {"phases": [
            {"name": phase_name, "cumulative_bytes": total},
        ]},
    }
    message = "staging phase|whole-phase"
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
        manifest["annotations"]["phases"].append({"name": "extra", "cumulative_bytes": total})
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validate(target)
