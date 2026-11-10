"""Declared-input record builders for whole-phase readset tests (metadata only, no artifact reads)."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def manifest_value(entries, phase_name, schema="v1", variant="whole"):
    """Build one ``tessera.declared_inputs.v1`` declaration; the real validator judges it in tests."""
    if schema != "v1":
        raise AssertionError(f"unknown test schema {schema}")
    entries = copy.deepcopy(entries)
    total = sum(entry["bytes"] for entry in entries)
    phases = [dict(name=phase_name, bytes=total, cumulative_bytes=total)]
    if variant == "wrong":
        phases[0]["name"] = "unrelated-phase"
    elif variant == "incomplete":
        phases[0]["cumulative_bytes"] = total - 1
    elif variant in ("split", "multiple"):
        boundary = entries[0]["bytes"] - (variant == "split")
        phases = [dict(name=phase_name, bytes=boundary, cumulative_bytes=boundary),
                  dict(name="remainder", bytes=total - boundary, cumulative_bytes=total)]
    elif variant in ("leading-zero", "trailing-zero"):
        empty = dict(name="empty", bytes=0, cumulative_bytes=0 if variant == "leading-zero" else total)
        phases.insert(0 if variant == "leading-zero" else 1, empty)
    elif variant == "v1-advisory-bytes":
        phases[0]["bytes"] = 1  # boundaries are cumulative_bytes, not bytes.
    elif variant not in ("whole", "missing"):
        raise AssertionError(f"unknown phase fixture {variant}")
    mount = os.path.commonpath([str(Path(entry["path"]).parent) for entry in entries])
    value = dict(schema="tessera.declared_inputs.v1", produced_by={"test": "metadata-only"},
                 mount_prefix=mount, entries=entries, entry_count=len(entries),
                 total_bytes=total, annotations={})
    if variant != "missing":
        value["annotations"]["phases"] = phases
    return value


def write_manifest(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path
