"""A4 serving seam: compact preparer -> native kernel, held to the parsed path.

The serve path must use the loader's compact preparer
(``compact_prep.prepare_span2_compact``: verified metadata, rank-local packed
planes, no parent expansion).  This gate holds that bundle to the
materialising path's bundle on the same real rank cut: same planes, same
kernel, same numbers.  It also exercises ``serving.native_a4``'s dense apply,
which is what the route calls.

Fixtures: ``TESSERA_A4_WIRE_DIR`` (``gate_proj_wire.bin`` etc. + config), same
bytes the kernel gate uses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

try:
    import pytest
except ImportError:  # the stock serve image runs the gate without pytest
    pytest = None

from tessera.compact_prep import parse_compact_expert, prepare_span2_compact
from tessera.errors import GrammarError
from tessera.kernel_a4 import A4Unit, a4_quantize_activation, a4_span2_gemm
from tessera.serving.native_a4 import prepare_a4_unit, a4_dense_apply

DATA = Path(os.environ.get("TESSERA_A4_WIRE_DIR", "/mnt/shared/astra-native-a4/data"))
PREFIX = "model.language_model.layers.3.mlp.experts"
LAYER = "layers_3"
TP = 2
ROLES = {"w13": ("gate_proj", "up_proj"), "w2": ("down_proj",)}


def declared():
    from tessera.serving.scheme import validate_tessera_moe_scheme

    cfg = json.loads((DATA / "a4-config.json").read_text())
    scheme = cfg["quantization_config"]["config_groups"][
        f"tessera_model_language_model_{LAYER}_mlp_experts"]["scheme"]
    return validate_tessera_moe_scheme(scheme, PREFIX)


def rank_cut(group: str, rank: int):
    """The dense route's rank-local plan for one expert group, as ranges."""
    from tessera.serving.moe_route import _packed_group_shard_plan

    return _packed_group_shard_plan(declared(), group, PREFIX, rank, TP)


def compact_unit(role: str, group: str, rank: int, device="cuda") -> A4Unit:
    # The fixture files are ``tessera.fused`` containers with one member each
    # (the artifact frames every role), so the compact reader is the expert
    # reader, and the member keeps its own name and row frame.
    members = parse_compact_expert((DATA / f"{role}_wire.bin").read_bytes(), device)
    wire = next((member for member in members if member.name == role), members[0])
    plan = rank_cut(group, rank)
    shard = plan.role(role)
    if plan.axis == "row":
        rows, cols = (shard.lo, shard.hi), None
    elif plan.axis == "column":
        rows, cols = None, (shard.lo, shard.hi)
    else:                                   # whole unit: nothing to cut
        rows = cols = None
    return prepare_a4_unit(wire, rows=rows, cols=cols)


def parsed_unit(role: str, group: str, rank: int, device="cpu") -> A4Unit:
    from tessera.lane_planes import prepare_span2_planes
    from tessera.serving.scheme import expert_role_declarations, parse_tessera_expert_blob
    from tessera.serving.sharding import shard_parsed_roles

    declared_scheme = declared()
    declarations = expert_role_declarations(declared_scheme["groups"][group])
    index = ROLES[group].index(role)
    parsed = parse_tessera_expert_blob((DATA / f"{role}_wire.bin").read_bytes(),
                                       declarations[index], f"{PREFIX} {role}",
                                       device=device)[0]
    local = shard_parsed_roles([parsed], rank_cut(group, rank))[0]
    return A4Unit.from_prepared(prepare_span2_planes(local[1], device="cuda"))


def check_compact_equals_parsed(rank: int, group: str):
    results = {}
    for role in ROLES[group]:
        compact = compact_unit(role, group, rank)
        parsed = parsed_unit(role, group, rank)
        assert (compact.rows, compact.cols) == (parsed.rows, parsed.cols)
        for field in ("select", "label", "point", "nibbles", "label_lut",
                      "subset_nibbles", "code_nibbles"):
            left = getattr(compact, field)
            right = getattr(parsed, field)
            assert left.shape == right.shape, (role, field, left.shape, right.shape)
            if left.dtype is torch.uint16 or right.dtype is torch.uint16:
                assert torch.equal(left, right), f"{role}: {field} differs"
            elif left.dtype.is_floating_point:
                assert torch.equal(left.view(torch.uint8), right.view(torch.uint8)), \
                    f"{role}: {field} bytes differ"
            else:
                assert torch.equal(left, right), f"{role}: {field} differs"
        assert compact.global_scale == parsed.global_scale
        torch.manual_seed(5)
        x = torch.randn(8, compact.cols, dtype=torch.bfloat16, device="cuda") * 0.25
        gscale = torch.tensor([448.0 * 6.0 / max(float(x.abs().max()), 1e-6)],
                              dtype=torch.float32, device="cuda")
        packed, scales = a4_quantize_activation(x, gscale)
        left = a4_span2_gemm(packed, scales, compact, compact.epilogue_for(gscale),
                             out_dtype=torch.float32)
        right = a4_span2_gemm(packed, scales, parsed, parsed.epilogue_for(gscale),
                              out_dtype=torch.float32)
        assert torch.equal(left, right), f"{role}: kernel outputs differ"
        dense = a4_dense_apply(x.reshape(1, 8, compact.cols), compact, gscale,
                               out_dtype=torch.float32)
        assert torch.equal(dense.reshape(8, compact.rows), left), \
            f"{role}: serving apply differs from the kernel"
        results[role] = float((left - right).abs().max())
    return results


def check_compact_cut_refusal():
    """A cut the compact repacker cannot serve refuses, by name."""
    members = parse_compact_expert((DATA / "gate_proj_wire.bin").read_bytes(), "cuda")
    wire = next((member for member in members if member.name == "gate_proj"), members[0])
    rows = wire.rows
    try:
        prepare_span2_compact(wire, rows=(0, rows // 3), cols=None)
    except GrammarError as exc:
        return {"refused": str(exc)[:80]}
    raise AssertionError("a row cut below the select-plane byte was accepted")


def check_route_imports():
    """The two routes and the adapter import and carry their native hooks."""
    import importlib

    route = importlib.import_module("tessera.serving.nvfp4_route")
    moe = importlib.import_module("tessera.serving.nvfp4_moe_route")
    adapter = importlib.import_module("tessera.serving.native_a4")
    for module, name in ((route, "a4_span2_gemm"), (moe, "a4_grouped_apply"),
                         (adapter, "prepare_a4_unit")):
        source = Path(module.__file__).read_text()
        assert name in source, f"{module.__name__} lost its native hook {name}"
    assert "parse_compact_expert" in Path(moe.__file__).read_text()
    return {"routes": ["nvfp4_route", "nvfp4_moe_route", "native_a4"]}


def run_gate(report_path=None) -> dict:
    import json as _json
    import traceback

    report = {"checks": [], "ok": False}
    for rank in (0, 1):
        for group, roles in ROLES.items():
            name = f"compact_equals_parsed.rank{rank}.{group}"
            entry = {"check": name}
            try:
                entry["detail"] = check_compact_equals_parsed(rank, group)
                entry["ok"] = True
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()[-2000:]
            report["checks"].append(entry)
            shown = (entry["detail"] if entry["ok"]
                     else str(entry.get("error", "")).splitlines()[0])
            print(f"[{'PASS' if entry['ok'] else 'FAIL'}] {name} {shown}", flush=True)
    entry = {"check": "route_imports"}
    try:
        entry["detail"] = check_route_imports()
        entry["ok"] = True
    except Exception as exc:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-2000:]
    report["checks"].append(entry)
    shown = (entry["detail"] if entry["ok"]
             else str(entry.get("error", "")).splitlines()[0])
    print(f"[{'PASS' if entry['ok'] else 'FAIL'}] route_imports {shown}", flush=True)

    entry = {"check": "compact_cut_refusal"}
    try:
        entry["detail"] = check_compact_cut_refusal()
        entry["ok"] = True
    except Exception as exc:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-2000:]
    report["checks"].append(entry)
    shown = (entry["detail"] if entry["ok"]
             else str(entry.get("error", "")).splitlines()[0])
    print(f"[{'PASS' if entry['ok'] else 'FAIL'}] compact_cut_refusal {shown}", flush=True)
    report["ok"] = all(c["ok"] for c in report["checks"])
    if report_path:
        with open(report_path, "w") as handle:
            _json.dump(report, handle, indent=1)
    print("REPORT " + _json.dumps(report))
    print("gate:", "PASS" if report["ok"] else "FAIL")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    outcome = run_gate(report_path=args.report)
    raise SystemExit(0 if outcome["ok"] else 1)
