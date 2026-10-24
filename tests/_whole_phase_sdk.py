"""Pinned public manifest SDK fixture; no installed SDK or source-path injection.

Import ``whole_phase_sdk`` into a test module and request it before importing
consumers of ``prismabuild``. The fixture returns the patched public client.
Snapshot provenance and limitations: fixtures/pb_manifest_sdk/README.md.
"""
from __future__ import annotations

import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType
import uuid

import pytest


_SNAPSHOT_COMMIT = "b1168d0703f74e9bb8496a263f3f0f270d483959"
_SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "pb_manifest_sdk"


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    # Every snapshot module is loaded by explicit filename, not package search.
    package.__path__ = []
    package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    return package


def _load_snapshot_module(monkeypatch, package: ModuleType, name: str) -> ModuleType:
    path = _SNAPSHOT / f"{name}.py"
    qualified_name = f"{package.__name__}.{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load manifest SDK snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, qualified_name, module)
    spec.loader.exec_module(module)
    setattr(package, name, module)
    print(
        f"whole_phase_sdk source={path} bytes={path.stat().st_size} "
        f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
    )
    return module


@pytest.fixture
def whole_phase_sdk(monkeypatch):
    """Always use the pinned real parser/layout through the public client API.

    All module entries, including any pre-existing installed SDK entries, are
    restored by monkeypatch. No phase-range export is supplied: consumers must
    use the supported parser and read-entry API rather than an internal helper.
    """
    print(
        f"whole_phase_sdk interpreter={sys.executable} "
        f"version={sys.version!r} snapshot_commit={_SNAPSHOT_COMMIT}"
    )
    private_name = f"_tessera_test_whole_phase_pb_{uuid.uuid4().hex}"
    snapshot = _package(private_name)
    monkeypatch.setitem(sys.modules, private_name, snapshot)
    _load_snapshot_module(monkeypatch, snapshot, "core")
    parser = _load_snapshot_module(monkeypatch, snapshot, "data_manifest")
    layout = _load_snapshot_module(monkeypatch, snapshot, "manifest_layout")

    public = _package("prismabuild")
    client = ModuleType("prismabuild.client")
    client.__package__ = "prismabuild"
    client.read_data_manifest = parser.read_data_manifest
    client.manifest_read_entries = layout.manifest_read_entries
    client.__all__ = ("read_data_manifest", "manifest_read_entries")
    storage_tiers = ModuleType("prismabuild.storage_tiers")
    storage_tiers.__package__ = "prismabuild"
    public.client = client
    public.storage_tiers = storage_tiers
    monkeypatch.setitem(sys.modules, "prismabuild", public)
    monkeypatch.setitem(sys.modules, "prismabuild.client", client)
    monkeypatch.setitem(sys.modules, "prismabuild.storage_tiers", storage_tiers)
    return client


def manifest_value(entries, phase_name, schema, variant="whole"):
    """Build declarations only; the verbatim PB parser validates them in tests."""
    import copy

    entries = copy.deepcopy(entries)
    total = sum(entry["bytes"] for entry in entries)
    indices = list(range(len(entries)))
    phases = [dict(name=phase_name, bytes=total, cumulative_bytes=total)]
    if schema == "v2":
        phases[0]["entry_indices"] = indices
    if variant == "wrong":
        phases[0]["name"] = "unrelated-phase"
    elif variant == "incomplete":
        phases[0]["cumulative_bytes"] = total - 1
    elif variant in ("split", "multiple"):
        boundary = entries[0]["bytes"] - (variant == "split")
        phases = [dict(name=phase_name, bytes=boundary, cumulative_bytes=boundary),
                  dict(name="remainder", bytes=total - boundary, cumulative_bytes=total)]
        if schema == "v2":
            phases[0]["entry_indices"] = [0]
            phases[1]["entry_indices"] = indices[1:]
    elif variant in ("leading-zero", "trailing-zero"):
        empty = dict(name="empty", bytes=0,
                     cumulative_bytes=0 if variant == "leading-zero" else total)
        if schema == "v2":
            empty["entry_indices"] = []
        phases.insert(0 if variant == "leading-zero" else 1, empty)
    elif variant == "read-order":
        if schema != "v2":
            raise AssertionError("read-order fixture requires v2")
        entries.reverse()
        phases[0]["entry_indices"] = indices[::-1]
    elif variant == "resident":
        if schema != "v2":
            raise AssertionError("resident fixture requires v2")
        phases[0]["resident_before_launch"] = True
    elif variant == "v1-advisory-bytes":
        if schema != "v1":
            raise AssertionError("advisory bytes fixture requires v1")
        phases[0]["bytes"] = 1  # v1 boundaries are cumulative_bytes, not bytes.
    elif variant not in ("whole", "missing"):
        raise AssertionError(f"unknown phase fixture {variant}")
    import os
    mount = os.path.commonpath([str(Path(entry["path"]).parent) for entry in entries])
    value = dict(schema=f"prismaquant.prismabuild.data_manifest.{schema}",
                 produced_by={"test": "metadata-only"}, mount_prefix=mount,
                 entries=entries, entry_count=len(entries), total_bytes=total,
                 annotations={})
    if schema == "v1":
        if variant != "missing":
            value["annotations"]["phases"] = phases
    elif schema == "v2":
        value["read_plan"] = dict(phases=[] if variant == "missing" else phases,
                                  read_bytes=total)
    else:
        raise AssertionError(f"unknown test schema {schema}")
    return value


def write_manifest(path, value):
    import json
    path.write_text(json.dumps(value), encoding="utf-8")
    return path
