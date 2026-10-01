"""The compact loader's LUT-plane window unit (E2M1x2): the fused E2M1 lane's inputs.

``compact_prep.prepare_window_lut_compact`` reads an E2M1x2 window unit over
the LUT16 plane straight from the packed wire.  The oracle is the
materialising reader: ``parse_unit_artifact`` for a whole unit and
``sharding.shard_parsed_roles`` for a TP cut, packed by the documented
tile-word packer (``window_pack_reference.pack_bitstream``) over the body's
CODE rows -- one code per tuple, so a 512-row tile is 512 tuples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_pack_reference import pack_bitstream  # noqa: E402

from tessera.errors import GrammarError  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the compact window repack is a CUDA path")

ROWS, COLS = 1536, 512           # 768 tuples: a whole tile and a partial one


def _pair_grid():
    from tessera.alphabet import E2M1_GRID, tuple_grid

    return tuple_grid(E2M1_GRID, 2)


def _encoded(q256, window_bits, seed):
    from tessera.export import encode_linear_planes
    from tessera.manifest import BodyKind, ScalePlaneKind

    torch.manual_seed(seed)
    weight = (torch.randn(ROWS, COLS, device="cuda") * 0.02).contiguous()
    exported, _unit, _forests = encode_linear_planes(
        weight, grid=_pair_grid(), q256=q256, name="w", body=BodyKind.WINDOW,
        scale_plane=ScalePlaneKind.LUT, window_bits=window_bits, verify=False)
    return exported.blob


def _plan(axis, rank, tp=2):
    from tessera.serving.sharding import plan_shard

    if axis == "row":
        return plan_shard("m", roles=[("w", ROWS)], columns=COLS, out_partitions=[ROWS // tp],
                          in_size=COLS, tp_rank=rank, tp_size=tp, input_size=COLS, output_size=ROWS)
    return plan_shard("m", roles=[("w", ROWS)], columns=COLS, out_partitions=[ROWS],
                      in_size=COLS // tp, tp_rank=rank, tp_size=tp, input_size=COLS, output_size=ROWS)


def _cut(plan):
    from tessera.serving.sharding import AXIS_ROWS

    role = plan.roles[0]
    return {"rows": (role.lo, role.hi)} if plan.axis == AXIS_ROWS else {"cols": (role.lo, role.hi)}


def _assert_matches(unit, ref, where):
    """``unit`` (the compact prep) against ``ref`` (the reader's EncodedUnit)."""
    from tessera.lane_planes import lut_scale_bytes, pack_scale_nibbles

    steps, cols = ref.body_bits.shape
    rows = steps * 2
    assert (int(unit.rows), int(unit.cols), int(unit.arity)) == (rows, cols, 2), where
    rates = tuple(int(r) for r in ref.rates)
    want = pack_bitstream(ref.body_bits.detach().cpu(), rates)
    assert int(unit.rep.rows) == steps, (where, "the repack counts tuples")
    assert int(unit.rep.tile_words) == int(want.tile_words), where
    assert int(unit.rep.n_tiles) == int(want.n_tiles), where
    assert torch.equal(unit.rep.perm.cpu(), want.perm), where
    assert torch.equal(unit.rep.runs.cpu(), want.runs), where
    assert torch.equal(unit.rep.words.cpu(), want.words), (where, "words")
    assert torch.equal(unit.codes.cpu(), ref.window_codes.cpu().to(torch.uint8)), (where, "codes")
    plane = pack_scale_nibbles(ref.scale_refine.cuda(), rows, cols, int(ref.half))
    assert torch.equal(unit.scale_plane, plane), (where, "scale plane")
    assert torch.equal(unit.scale_lut, lut_scale_bytes(ref.scale_lut, "cuda")), (where, "lut")
    assert unit.global_scale == float(ref.scale_global), (where, "global")
    state = getattr(ref, "initial_state", None)
    if state is None:
        assert not bool(unit.initial_state.any()), where
        assert unit.permuted_start_state() is None, where
    else:
        assert torch.equal(unit.initial_state, state.to(unit.initial_state.device, torch.int32)), (where, "state")


@cuda
@pytest.mark.parametrize("window_bits", [12, 14])
@pytest.mark.parametrize("q256", [192, 448, 512, 704, 1024])
def test_whole_unit_matches_the_reader(q256, window_bits):
    """One-run ([4], [8]) and two-run ([1,2], [3,4], [5,6]) tables, at the
    recipe's L=12 and at L=14."""
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact
    from tessera.unit_artifact import parse_unit_artifact

    blob = _encoded(q256, window_bits, seed=q256 + window_bits)
    ref = parse_unit_artifact(blob, device="cuda").unit
    wire = parse_compact_wire(blob, device="cuda", name="w")
    unit = prepare_window_lut_compact(wire, device="cuda")
    assert int(unit.window_bits) == window_bits
    assert int(unit.row_offset) == 0
    _assert_matches(unit, ref, ("whole", q256, window_bits))


@cuda
@pytest.mark.parametrize("q256", [448, 512])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_tp2_cuts_match_the_sliced_reader(axis, q256):
    """TP2 row and column cuts: the compact prep equals the materialising
    reader's shard, and rank 1's row cut carries the sliced unit's own
    window state before its first tuple."""
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.unit_artifact import parse_unit_artifact

    blob = _encoded(q256, 14, seed=7 * q256)
    parsed = parse_unit_artifact(blob, device="cuda")
    wire = parse_compact_wire(blob, device="cuda", name="w")
    for rank in (0, 1):
        plan = _plan(axis, rank)
        shard = shard_parsed_roles([("w", parsed)], plan)[0][1]
        unit = prepare_window_lut_compact(wire, device="cuda", **_cut(plan))
        _assert_matches(unit, shard.unit, (axis, rank, q256))
        assert int(unit.row_offset) == int(getattr(shard.unit, "row_offset", 0)), (axis, rank)
        if axis == "row" and rank == 1:
            assert bool(unit.initial_state.any()), "rank 1 starts mid-stream"


@cuda
def test_refuses_what_it_does_not_read():
    """A TCQ body and a CHANNEL-plane window unit are refused by name."""
    from tessera.alphabet import E4M3_GRID
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact
    from tessera.export import encode_linear_planes
    from tessera.manifest import BodyKind, ScalePlaneKind

    torch.manual_seed(3)
    weight = (torch.randn(256, 128, device="cuda") * 0.02).contiguous()
    tcq, _u, _f = encode_linear_planes(weight, grid=_pair_grid(), q256=768, name="w",
                                       body=BodyKind.TCQ, span=2,
                                       scale_plane=ScalePlaneKind.LUT, verify=False)
    with pytest.raises(GrammarError, match="takes a window unit"):
        prepare_window_lut_compact(parse_compact_wire(tcq.blob, device="cuda"), device="cuda")
    fp8, _u, _f = encode_linear_planes(weight, grid=E4M3_GRID, q256=1024, name="w", verify=False)
    with pytest.raises(GrammarError, match="LUT scale plane"):
        prepare_window_lut_compact(parse_compact_wire(fp8.blob, device="cuda"), device="cuda")
