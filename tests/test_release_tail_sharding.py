"""Capability agrees with cuts of real RELEASE wires with a partial tail."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tessera.decode import reconstruct_unit
from tessera.encode import _canonical_release_order
from tessera.errors import GrammarError
from tessera.layout import can_shard, slice_unit, unsliceable_reason
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact


@pytest.fixture(scope="module", params=[384, 640])
def partial_tail(request):
    """Extend committed uniform-rate planes; neither case needs an encoder."""
    legacy = Path(__file__).parent / "data" / "legacy"
    seed = parse_unit_artifact(
        (legacy / "e2m1-768-release256-256c.tessera").read_bytes(), device="cpu"
    )
    rows, old_columns = seed.manifest.geometry.rows, seed.manifest.geometry.columns
    columns = request.param
    repeats = -(-columns // old_columns)
    changes = {
        "rates": (seed.unit.rates * repeats)[:columns],
        "release_index": torch.zeros(0, dtype=torch.long),
        "release_code": torch.zeros(0, dtype=torch.long),
    }
    for name in ("anchors", "codes", "body_bits", "completion_bits"):
        plane = getattr(seed.unit, name)
        if plane.numel():
            changes[name] = plane.repeat(1, repeats)[:, :columns].contiguous()
    for name, block in (("scale_base", seed.unit.group), ("scale_refine", seed.unit.half)):
        plane = getattr(seed.unit, name)
        if plane.numel():
            changes[name] = (
                plane.reshape(rows, old_columns // block)
                .repeat(1, repeats)[:, :columns // block].reshape(-1).contiguous()
            )
    assert seed.unit.diagonals is None
    plain = replace(seed.unit, **changes)
    decoded = reconstruct_unit(plain, seed.forests, seed.code)
    released = replace(
        plain,
        release_index=_canonical_release_order(decoded, columns, 256, 96),
        release_code=torch.arange(96) % 16,
    )
    result = []
    for unit in (plain, released):
        _, _, blob = build_unit_artifact(
            unit, "partial-tail", seed.forests, seed.manifest.branch.root_q256,
            seed.code, fixture_id=None,
        )
        parsed = parse_unit_artifact(blob, device="cpu")
        assert torch.equal(
            reconstruct_unit(unit, seed.forests, seed.code),
            reconstruct_unit(parsed.unit, parsed.forests, parsed.code),
        )
        result.append(parsed)
    return result


def _view(parsed, view):
    if view == "encoded":
        return parsed.unit, {
            "arity": parsed.grid.arity,
            "superblock": parsed.manifest.geometry.superblock_columns,
        }
    return (parsed.manifest if view == "manifest" else parsed), {}


@pytest.mark.parametrize("view", ["encoded", "parsed", "manifest"])
def test_released_partial_tail_capability_matches_the_cutter(partial_tail, view):
    """#350: a row split retains the tail; legal column cuts still work."""
    plain, released = partial_tail
    rows = released.manifest.geometry.rows
    columns = released.manifest.geometry.columns
    superblock = released.manifest.geometry.superblock_columns
    parent = reconstruct_unit(released.unit, released.forests, released.code)
    obj, options = _view(released, view)
    assert unsliceable_reason(obj, **options) is None

    # Both the complete superblocks and the final partial one remain cuttable.
    # Once the complete prefix is cut out, equal column splits are admissible.
    prefix = columns // superblock * superblock
    for bounds in ((0, superblock), (0, prefix), (prefix, columns)):
        shard = slice_unit(released, cols=bounds)
        _, _, blob = build_unit_artifact(
            shard, "column-control", released.forests,
            released.manifest.branch.root_q256, released.code, fixture_id=None,
        )
        back = parse_unit_artifact(blob, device="cpu")
        assert torch.equal(
            reconstruct_unit(back.unit, back.forests, back.code),
            parent[:, bounds[0]:bounds[1]],
        )
        if bounds == (0, prefix):
            column_obj, column_options = _view(back, view)
            tp = prefix // superblock
            assert can_shard(column_obj, tp, "column", **column_options)
            for rank in range(tp):
                part = slice_unit(back, cols=(rank * superblock, (rank + 1) * superblock))
                assert torch.equal(
                    reconstruct_unit(part, back.forests, back.code),
                    parent[:, rank * superblock:(rank + 1) * superblock],
                )

    # Removing RELEASE is sufficient to restore full-width row/identity cuts.
    plain_obj, plain_options = _view(plain, view)
    plain_weights = reconstruct_unit(plain.unit, plain.forests, plain.code)
    for tp in (1, 2, 4):
        assert can_shard(plain_obj, tp, "row", **plain_options)
        for rank in range(tp):
            bounds = (rank * rows // tp, (rank + 1) * rows // tp)
            shard = slice_unit(plain, rows=bounds)
            assert torch.equal(
                reconstruct_unit(shard, plain.forests, plain.code),
                plain_weights[bounds[0]:bounds[1]],
            )

    with pytest.raises(GrammarError, match="only on superblock boundaries"):
        slice_unit(released)
    for tp in (1, 2, 4):
        for rank in range(tp):
            with pytest.raises(GrammarError, match="only on superblock boundaries"):
                slice_unit(released, rows=(rank * rows // tp, (rank + 1) * rows // tp))
        assert not can_shard(obj, tp, "row", **options)


def test_partial_tail_serving_refusal_names_the_cut_without_an_impossible_divisor(
    partial_tail,
):
    """Both serving seams must preserve the cutter's geometry obstruction."""
    from tessera.serving.sharding import (
        RoleShard, ShardPlan, _shard_unit_for_rank, check_shard_granularity,
    )

    _plain, released = partial_tail
    rows, columns = (released.manifest.geometry.rows,
                     released.manifest.geometry.columns)
    assert unsliceable_reason(released) is None  # Column cuts still exist.
    role = RoleShard("weight", rows, 0, rows // 2, 2)
    plan = ShardPlan("partial-tail", rows, columns, rows // 2, columns,
                     0, 2, "row", (role,))
    for tp in range(1, rows + 1):
        if rows % tp == 0:
            assert not can_shard(released, tp, "row")
    with pytest.raises(GrammarError) as cutter:
        slice_unit(released, rows=(role.lo, role.hi))
    with pytest.raises(ValueError) as seam:
        _shard_unit_for_rank(released, plan, role)
    with pytest.raises(ValueError) as check:
        check_shard_granularity(plan, role, released)
    assert str(seam.value) == str(check.value)
    assert str(cutter.value) in str(seam.value)
    assert "tensor_parallel_size that divides" not in str(seam.value)
    assert "tensor_parallel_size=1" in str(seam.value)
