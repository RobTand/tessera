"""Declared weight transforms are refused, by name, wherever a tile stops short
of them (#233).

Segment-2a diagonals and the branch rotation are transforms the encoder priced
and the wire carries; ``tessera.decode.reconstruct_unit`` undoes them AFTER
the codes-times-scale product.  A materialisation that stops at that product
-- the stock tensors, the FP8/BF16 serving routes' reference decodes, the
routed-MoE expert decode, the NVFP4 route's stock reference -- computes a
*different Linear* on a transformed unit, wrong by a rank-1 factor or a basis
change, and nothing downstream raises.  ``decode.require_untransformed`` is
the one home for the refusal; these tests drive it through every consumer,
each on real wire/plane objects through the real reader.

Before the refusal existed (measured on this tree at the parent of the fix):

* the committed fitted-diagonal NVFP4 fixture's stock dequant differed from
  ``reconstruct_unit`` by **3.2873** max abs, silently;
* a rotated E4M3 window wire built from the committed fixture passed the
  whole FP8 preparation gate (packed-window decode == ``materialize_fp8``,
  both dropping the rotation) and served with **0.1153** max abs against the
  complete reconstruction.

Everything here runs on CPU: the transforms are refused before any device
work, and the fixtures are the committed legacy wires.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tessera.decode import reconstruct_unit                          # noqa: E402
from tessera.errors import GrammarError                              # noqa: E402
from tessera.manifest import RotationState                           # noqa: E402
from tessera.unit_artifact import parse_unit_artifact                # noqa: E402

LEGACY = Path(__file__).resolve().parent / "data" / "legacy"
DIAG_FIXTURE = LEGACY / "e2m1-256-cfull-lut-diag-256c.tessera"
LUT_FIXTURE = LEGACY / "e2m1-256-cfull-lut-512c.tessera"
E4M3_FIXTURE = LEGACY / "e4m3-1024-window-channel-256c.tessera"
BF16_FIXTURE = LEGACY / "bf16-1024-window-channel-256c.tessera"


def _parse(path):
    return parse_unit_artifact(path.read_bytes(), device="cpu")


def _rotated(parsed):
    """The same parsed wire with the unit declared rotated, planes untouched."""
    parsed.unit = dataclasses.replace(
        parsed.unit, rotation=RotationState.R_IN_ONLY, rotation_block=128)
    return parsed


# --- the rule's one home ------------------------------------------------------

def test_the_rule_names_the_field_and_the_consumer():
    # Imported here, not at module top, so a tree without the rule fails these
    # tests one by one (DID NOT RAISE) instead of erroring at collection.
    from tessera.decode import require_untransformed

    diag = _parse(DIAG_FIXTURE)
    with pytest.raises(GrammarError, match=r"someplace does not undo segment-2a"):
        require_untransformed(diag.unit, "someplace")
    rot = _rotated(_parse(LUT_FIXTURE))
    with pytest.raises(GrammarError, match=r"someplace does not undo .*R_IN_ONLY"):
        require_untransformed(rot.unit, "someplace")
    # The untransformed wire passes: the gate reads the unit's declaration,
    # nothing else.
    require_untransformed(_parse(LUT_FIXTURE).unit, "someplace")


# --- direct stock materialisation --------------------------------------------

def test_materialize_stock_refuses_the_fitted_diagonal_fixture():
    """The audit's reproduced case: the fitted-diagonal NVFP4 fixture's stock
    dequant was 3.2873 max abs off ``reconstruct_unit`` and nothing raised."""
    from tessera.stock import materialize_stock

    parsed = _parse(DIAG_FIXTURE)
    assert parsed.unit.diagonals is not None, "the fixture must carry segment 2a"
    with pytest.raises(GrammarError, match=r"materialize_stock does not undo segment-2a"):
        materialize_stock(parsed.unit, parsed.forests, parsed.code)
    # The complete inverse path still serves the same bytes: the refusal is
    # about who applies the transform, not about the wire.
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    assert full.shape == (parsed.manifest.geometry.rows, parsed.manifest.geometry.columns)
    assert bool(torch.isfinite(full).all())


def test_materialize_stock_refuses_a_rotated_unit_and_serves_the_unrotated_one():
    from tessera.stock import materialize_stock, stock_kind

    clean = _parse(LUT_FIXTURE)
    tensors = materialize_stock(clean.unit, clean.forests, clean.code)
    assert stock_kind(tensors) == "nvfp4"
    rotated = _rotated(_parse(LUT_FIXTURE))
    with pytest.raises(GrammarError, match=r"materialize_stock does not undo the unit's rotation"):
        materialize_stock(rotated.unit, rotated.forests, rotated.code)


# --- the dense FP8 serving route ----------------------------------------------

def _fused_e4m3_wire(rotation):
    """A real fused E4M3 window/CHANNEL wire at q1024, rebuilt from the
    committed fixture with the given rotation state -- serialized and parsed
    back through the real reader, exactly the audit's counterexample."""
    from tessera.fused import pack_fused
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact

    p = _parse(E4M3_FIXTURE)
    unit = dataclasses.replace(p.unit, rotation=rotation,
                               rotation_block=128 if rotation is not RotationState.NONE else 1)
    manifest, _region, blob = build_unit_artifact(
        unit, "weight", p.forests, p.manifest.branch.root_q256,
        p.code or ConvCode(), fixture_id=None)
    fused = pack_fused([("weight", manifest.geometry.rows, blob)])
    scheme = {
        "family": "TESSERA_FP8", "grid": p.grid.name, "body": "WINDOW",
        "plane": "CHANNEL", "q256": 1024, "rows": manifest.geometry.rows,
        "columns": manifest.geometry.columns, "wire_bytes": len(fused),
        "roles": [["weight", manifest.geometry.rows]],
    }
    return fused, scheme


def test_the_fp8_route_refuses_a_rotated_wire_at_preparation():
    """The wire parses -- the manifest and digest are valid, and the sidecar
    carries no rotation field to disagree with -- and the refusal fires in
    ``prepare_tessera_fp8_module`` through its reference decode, before any
    residency holds a tile."""
    from tessera.serving.fp8_route import prepare_tessera_fp8_module
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    fused, scheme = _fused_e4m3_wire(RotationState.R_IN_ONLY)
    roles = parse_tessera_blob_for_scheme(fused, scheme, "t", device="cpu")
    assert roles[0][1].unit.rotation is RotationState.R_IN_ONLY
    with pytest.raises(GrammarError, match=r"does not undo the unit's rotation"):
        prepare_tessera_fp8_module(roles, device="cpu")


def test_the_fp8_route_still_prepares_the_unrotated_wire():
    """The control: same fixture, rotation NONE, the whole gate passes and the
    prepared bytes are the complete reconstruction's (no transform pending,
    ``reconstruct_unit`` and the served tile agree exactly)."""
    from tessera.serving.fp8_route import prepare_tessera_fp8_module
    from tessera.serving.scheme import parse_tessera_blob_for_scheme

    fused, scheme = _fused_e4m3_wire(RotationState.NONE)
    roles = parse_tessera_blob_for_scheme(fused, scheme, "t", device="cpu")
    prepared = prepare_tessera_fp8_module(roles, device="cpu")
    served = (prepared.decode().view(torch.float8_e4m3fn).float()
              * prepared.row_scale().reshape(-1, 1))
    parsed = roles[0][1]
    reference = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    assert torch.equal(served, reference)


# --- the dense BF16 serving route ----------------------------------------------

def test_the_bf16_route_refuses_a_rotated_unit_at_preparation():
    from tessera.serving.bf16_route import prepare_tessera_bf16_module

    rotated = _rotated(_parse(BF16_FIXTURE))
    with pytest.raises(GrammarError, match=r"does not undo the unit's rotation"):
        prepare_tessera_bf16_module([("weight", rotated)], device="cpu")
    # The control: the committed fixture itself prepares.
    prepared = prepare_tessera_bf16_module([("weight", _parse(BF16_FIXTURE))], device="cpu")
    assert prepared.rows == _parse(BF16_FIXTURE).manifest.geometry.rows


# --- the routed-MoE expert route ------------------------------------------------

def test_the_moe_route_refuses_a_rotated_expert_wire():
    """One rotated projection refuses the stack: the expert decode is
    ``materialize_fp8`` per projection, and a rotated gate served unrotated
    would route tokens through a different expert Linear."""
    from tessera.alphabet import E4M3_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.serving.moe_route import prepare_tessera_moe_experts
    from tessera.serving.scheme import validate_tessera_moe_scheme

    hidden, inter, q256 = 64, 32, 1024

    def _wire(rows, cols, name, seed, rotation=RotationState.NONE):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(rows, cols, generator=g) * 0.02
        exported, _unit, _forests = encode_linear_planes(
            w.contiguous(), grid=E4M3_GRID, q256=q256, name=name,
            rotation=rotation, verify=False)
        return pack_fused([(name, rows, exported.blob)])

    gate = _wire(inter, hidden, "gate_proj", 1, rotation=RotationState.R_IN_ONLY)
    up = _wire(inter, hidden, "up_proj", 2)
    down = _wire(hidden, inter, "down_proj", 3)
    scheme = {
        "family": "TESSERA_FP8", "structure": "routed_moe", "grid": "E4M3",
        "body": "WINDOW", "plane": "CHANNEL", "experts": 1,
        "groups": {
            "w13": {"rows": 2 * inter, "columns": hidden, "q256": q256,
                    "wire_stride": max(len(gate), len(up)),
                    "roles": [["gate_proj", inter], ["up_proj", inter]]},
            "w2": {"rows": hidden, "columns": inter, "q256": q256,
                   "wire_stride": len(down),
                   "roles": [["down_proj", hidden]]}},
    }
    declared = validate_tessera_moe_scheme(scheme, "m")
    with pytest.raises(GrammarError, match=r"does not undo the unit's rotation"):
        prepare_tessera_moe_experts(
            {"w13": [[gate, up]], "w2": [[down]]}, declared, "m", device="cpu")
