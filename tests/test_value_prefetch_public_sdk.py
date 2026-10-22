"""#1310: exercise the numeric readset consumer without an installed PB SDK.

These controlled public-client responses test Tessera's boundary and refusals,
not PrismaBuild's codec implementation. The original real-SDK cases remain the
integration coverage when that optional dependency is available.
"""
from __future__ import annotations

import builtins
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import test_value_prefetch_numeric as gate


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
        # Restrict the consumer, not any SDK implementation's own imports.
        if globals is gate.numeric.__dict__:
            if name == "prismabuild":
                assert set(fromlist) <= {"client"}, (
                    "numeric readset validation must use the public PB client, "
                    f"not private imports {fromlist!r}"
                )
            elif name.startswith("prismabuild."):
                assert name == "prismabuild.client", (
                    "numeric readset validation imports a private PB module: " + name
                )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", public_only)
    return state


@pytest.mark.parametrize("corrupt_phase", [False, True])
def test_original_whole_bank_cases_use_only_public_sdk(
    tmp_path, public_sdk, corrupt_phase,
):
    # Reuse both release selectors verbatim, but do not let optional SDK
    # absence turn this deterministic consumer regression into a skip.
    gate.test_public_pb_readset_and_whole_bank_phase_contract(tmp_path, corrupt_phase)
    if not corrupt_phase:
        assert len(public_sdk.reads) == 1, "public read-order validation was bypassed"


@pytest.mark.parametrize("defect", ["encoding", "read_order", "phase_name"])
def test_public_sdk_migration_preserves_readset_refusals(tmp_path, public_sdk, defect):
    path = str(tmp_path / "bank.pt")
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "total_bytes": 32,
        "entries": [{"path": path, "offset": 0, "bytes": 32}],
        "annotations": {"phases": [
            {"name": "whole-synthetic-bank", "cumulative_bytes": 32},
        ]},
    }
    message = "staging phase"
    if defect == "encoding":
        public_sdk.encoding = "zstd"
    elif defect == "read_order":
        public_sdk.wrong_read_order = True
        message = "staging read order"
    else:
        manifest["annotations"]["phases"][0]["name"] = "not-the-whole-bank"
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        gate.numeric.validate_readset(target)
