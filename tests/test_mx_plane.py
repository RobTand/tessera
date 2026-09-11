"""Schema minor 8: the OCP MXFP8 scale plane (tessera#443, bullets 1 and 2).

``ScalePlaneKind.MX`` is one E8M0 byte per 32 consecutive columns of one
output row on SCALE_BASE and nothing else: no refinement plane, no rank-1
pair, no global.  A weight is ``e4m3(code) * 2^(E-127)`` in fp32, exactly, on
the E4M3 grid alone.  These tests hold the two acceptance bullets the plane
was built against:

  * **Representation and encoder.**  The exact scale decoder covers every
    legal E8M0 byte; the seam round-trips from bytes alone at minor 8; the
    initial plane is chosen by measured error and the refit step is monotone;
    zero blocks take byte 0x00 and nonfinite weights are refused by name;
    boundary blocks never straddle rows; every unsupported combination (grid,
    diagonals, group, coupled metric, stock tensor, kernel lane, config
    spelling, served route) is refused where its bytes would be decided.
  * **Exact accounting and decoder parity.**  ``priced == written == served``
    across rates, shapes and padding: the plane costs exactly a quarter bit
    per weight before padding, the reported total is the physical byte count,
    and the MXFP8 pair a block-scaled kernel would read dequantises bit for
    bit to the reference decode -- with nonuniform scales, boundary blocks,
    expert selection through the fused and MoE containers, and rank slicing.

Every S6b, LUT and CHANNEL artifact keeps its bytes and its minor.  CPU only.
"""
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID
from tessera.calculator import terminal_rate
from tessera.container import HEADER_BYTES, MX_SCHEMA_MINOR, SCHEMA_MINOR, SCHEMA_MINORS_READ, parse
from tessera.decode import (
    materialize_fp8,
    materialize_mxfp8,
    mxfp8_dequantize,
    reconstruct_unit,
    unit_half_scales,
    unit_scale_field,
)
from tessera.encode import (
    _mx_po2_field,
    _pack_scales_mx,
    _refit_scales_mx,
    _rtn_sse,
    encode_unit,
    grid_vector_table,
    window_table,
)
from tessera.errors import GrammarError, ManifestError, ScaleCodecError, TesseraError
from tessera.export import WireRecipe, _require_config_spellable, encode_linear, recipe_table
from tessera.footprint import account_terminal, plane_byte_report
from tessera.fp8 import E4M3FN_NAN_BYTES, E8M0_NAN_BYTE
from tessera.fused import pack_fused, parse_fused
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import BodyKind, ScalePlane, ScalePlaneKind
from tessera.moe_layout import pack_moe_wires, unpack_moe_wires
from tessera.planes import PlaneKind
from tessera.slicing import can_shard, shard_granularity, slice_unit
from tessera.stock import materialize_stock
from tessera.unit_artifact import build_unit_artifact, encoder_profile_id, parse_unit_artifact
from tessera.wire import mx_scales_from_plane, scales_from_planes

MX = ScalePlaneKind.MX
WINDOW = BodyKind.WINDOW
BLOCK = 32


def _weights(rows=16, cols=256, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, cols) * 0.02


def _rates(cols, q256):
    return bresenham_rate_schedule(root_from_q256(q256), cols, cap=E4M3_GRID.payload_bits)


def _mx_unit(w, q256=6 * 256, window=8, refit=2, **over):
    kw = dict(body=WINDOW, window_bits=window, scale_plane=MX, scale_refit=refit,
              trellis_weighting="scale")
    kw.update(over)
    return encode_unit(w, E4M3_GRID, _rates(w.shape[1], q256), None, **kw)


def _blob(unit, q256, name="u"):
    return build_unit_artifact(unit, name, E4M3_GRID, q256, None)


def _elements(blob, kind):
    art = parse(blob)
    return art.terminal.plane_elements[art.manifest.plane_order.index(kind)]


def _side_bytes(art):
    return HEADER_BYTES + len(art.manifest.encode(art.manifest.schema_minor))


# ------------------------------------------------------ the representation


def test_the_exact_scale_decoder_covers_every_legal_e8m0_byte():
    """``2^(E-127)`` for every byte but the NaN word, bit-exact in fp32."""
    bytes_ = torch.arange(0, 255, dtype=torch.uint8)
    field = mx_scales_from_plane(bytes_)
    assert field.dtype is torch.float32 and field.shape == (255,)
    assert all(float(field[e]) == 2.0 ** (e - 127) for e in range(255))
    assert bool(torch.isfinite(field).all())
    # The same numbers S6b's reader derives at a zero refinement word: an MX
    # plane read as a T-po2 S6b plane decodes to the same scales, which is
    # exactly why the reader dispatches on the kind and not on the counts.
    as_s6b = scales_from_planes(bytes_, torch.zeros(510, dtype=torch.uint8), 32, 16)
    assert torch.equal(as_s6b, torch.repeat_interleave(field, 2))
    with pytest.raises(ScaleCodecError, match="reserved E8M0 NaN word"):
        mx_scales_from_plane(torch.tensor([3, E8M0_NAN_BYTE], dtype=torch.uint8))


def test_the_manifest_record_is_the_kind_alone():
    assert ScalePlane.mx().kind is MX
    with pytest.raises(ManifestError, match="no table or global"):
        ScalePlane(MX, table=b"\x38")
    with pytest.raises(ManifestError, match="no table or global"):
        ScalePlane(MX, global_scale=Fraction(2))


@pytest.mark.parametrize("q256,refit", [(4 * 256, 0), (6 * 256, 2), (8 * 256, 1)])
def test_wire_round_trip_of_an_mx_plane(q256, refit):
    w = _weights()
    rows, cols = w.shape
    unit = _mx_unit(w, q256=q256, refit=refit)
    assert unit.scale_plane is MX
    assert unit.scale_base.shape == (rows * cols // BLOCK,) and unit.scale_refine.numel() == 0
    manifest, region, blob = _blob(unit, q256)
    assert manifest.schema_minor == MX_SCHEMA_MINOR == 8 and blob[10] == 8
    assert manifest.scale_plane == ScalePlane.mx()
    assert manifest.geometry.group_weights == BLOCK
    assert _elements(blob, PlaneKind.SCALE_BASE) == rows * cols // BLOCK
    for kind in (PlaneKind.SCALE_REFINE, PlaneKind.DIAG_SU, PlaneKind.DIAG_SV):
        assert _elements(blob, kind) == 0
    parsed = parse_unit_artifact(blob)
    assert parsed.manifest.scale_plane.kind is MX and parsed.grid == E4M3_GRID
    assert torch.equal(parsed.unit.scale_base, unit.scale_base)
    back = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    assert torch.equal(back, reconstruct_unit(unit, E4M3_GRID, None))
    assert bool(torch.isfinite(back).all())
    # The reconstruction really is codes times exact powers of two.
    field = unit_scale_field(parsed.unit, rows, cols)
    assert torch.equal(torch.log2(field), torch.log2(field).round())


def test_the_profile_id_binds_the_mx_plane():
    rates = (6,) * 256
    ids = {kind: encoder_profile_id(None, rates, E4M3_GRID, 1, kind, WINDOW, 8)
           for kind in ScalePlaneKind}
    assert len(set(ids.values())) == len(ScalePlaneKind)


# ------------------------------------------------------------- the encoder


def test_the_initial_plane_is_chosen_by_measured_error_not_by_a_rule():
    """Per block, the lower of two candidate binades' RTN error wins.

    The candidate set is what the body can emit: here the L=8 window table
    at the plane's own block, whose entries are sparse away from the peak,
    so the finer binade wins for some blocks and the clipping-free one for
    others.  On the full E4M3 grid the measured choice coincides with
    ``ceil`` on Gaussian blocks -- a float grid quantises with the same
    relative precision in every binade, so only clipping and subnormal
    loss can separate the two -- which is a fact about that grid, not a
    rule the pack applies.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 256) * 0.02
    table = window_table(E4M3_GRID, 8, half=BLOCK)
    emit = grid_vector_table(E4M3_GRID)[table.long()].reshape(-1).float().unique()
    base, effective = _pack_scales_mx(w, BLOCK, emit)
    groups = w.reshape(-1, BLOCK)
    amax = groups.abs().amax(dim=1)
    reach = float(emit.abs().max())
    e_hi = torch.ceil(torch.log2(amax / reach))
    e_lo = e_hi - 1
    cost_hi = _rtn_sse(groups, _mx_po2_field(e_hi), emit)
    cost_lo = _rtn_sse(groups, _mx_po2_field(e_lo), emit)
    chosen = base.float() - 127.0
    assert torch.equal(chosen, torch.where(cost_lo < cost_hi, e_lo, e_hi))
    assert torch.equal(effective, _mx_po2_field(chosen))
    # Both binades are reached across this draw: neither ``floor`` nor
    # ``ceil`` alone reproduces the plane.  A draw on which one never wins
    # is a finding about the table, and this says so rather than passing.
    assert bool((chosen == e_hi).any()) and bool((chosen == e_lo).any()), (
        f"the measured choice never left one binade on this draw: "
        f"{int((chosen == e_lo).sum())} finer, {int((chosen == e_hi).sum())} clipping-free"
    )


def test_the_refit_step_is_monotone_and_lands_on_a_bracketing_power_of_two():
    torch.manual_seed(5)
    rows, cols = 8, 256
    w = torch.randn(rows, cols)
    emit = torch.tensor(E4M3_GRID.values).float().unique()
    base, effective = _pack_scales_mx(w, BLOCK, emit)
    # Codes as the trellis would leave them: nearest grid value under the plane.
    scale = torch.repeat_interleave(effective, BLOCK).reshape(rows, cols)
    idx = torch.searchsorted(emit, (w / scale).reshape(-1)).clamp(0, emit.numel() - 1)
    units = emit[idx].reshape(rows, cols)
    W, U = w.reshape(-1, BLOCK), units.reshape(-1, BLOCK)
    A, B = (U * U).sum(1), (W * U).sum(1)
    new_base, new_eff = _refit_scales_mx(w, units, BLOCK, base, effective)
    assert torch.equal(new_eff, mx_scales_from_plane(new_base))
    old_cost = A * effective ** 2 - 2 * B * effective
    new_cost = A * new_eff ** 2 - 2 * B * new_eff
    assert bool((new_cost <= old_cost).all())
    star = B / A
    moved = new_base != base
    lo = torch.floor(torch.log2(star))
    assert bool(((new_base.float() - 127 == lo) | (new_base.float() - 127 == lo + 1))[moved].all())
    # A block whose codes anti-correlate keeps its word.
    anti = -units
    kept_base, _kept = _refit_scales_mx(w, anti, BLOCK, base, effective)
    assert torch.equal(kept_base, base)


def test_a_refit_pass_never_raises_the_squared_error_of_the_unit():
    """The trailing refit is monotone in the unit's own squared error.

    Measured while writing this: on Gaussian E4M3 units the least-squares
    optimum ``B / A`` after a trellis pass sits within a few percent of the
    power of two the pass quantised against (0 of 128 blocks outside
    ``[1/sqrt 2, sqrt 2]`` at 16x256), so the po2 lattice is too coarse for
    the step to move a word and ``scale_refit`` is a measured no-op on the
    MX plane.  The property pinned is the inequality; the no-op is recorded
    in the PR as a negative result, not asserted here as a contract.
    """
    w = _weights(seed=9)
    once = _mx_unit(w, refit=0)
    refit = _mx_unit(w, refit=1)
    assert refit.sse <= once.sse


def test_zero_blocks_take_the_floor_word_and_nonfinite_weights_are_refused():
    w = _weights()
    w[3, :BLOCK] = 0.0
    w[7, 64:96] = 0.0
    unit = _mx_unit(w)
    plane = unit.scale_base.reshape(w.shape[0], -1)
    assert int(plane[3, 0]) == 0 and int(plane[7, 2]) == 0
    assert int(plane.max()) < E8M0_NAN_BYTE
    back = reconstruct_unit(unit, E4M3_GRID, None)
    # Not exactly zero: the window body's shared history can leave nonzero
    # codes in a block the target never asked for, and the floor word bounds
    # what they reconstruct to by the grid's peak at 2^-127: below any
    # weight this format prices, and finite.
    bound = 448.0 * 2.0 ** -127
    assert bool((back[3, :BLOCK].abs() <= bound).all())
    assert bool((back[7, 64:96].abs() <= bound).all())
    assert bool(torch.isfinite(back).all())
    tile, scales = materialize_mxfp8(unit, E4M3_GRID, None)
    assert int(scales[3, 0]) == 0 and torch.equal(mxfp8_dequantize(tile, scales), back)
    for bad in (float("nan"), float("inf"), float("-inf")):
        w2 = _weights()
        w2[2, 40] = bad
        with pytest.raises(GrammarError, match="nonfinite"):
            _mx_unit(w2)


def test_nonuniform_block_magnitudes_give_distinct_words_indexed_by_block():
    """Per-block scales differ by construction and index ``(row*cols+col)//32``."""
    rows, cols = 8, 128
    torch.manual_seed(1)
    w = torch.randn(rows, cols) * 0.02
    for r in range(rows):
        for b in range(cols // BLOCK):
            w[r, b * BLOCK:(b + 1) * BLOCK] *= 2.0 ** (3 * b - 2 * r)
    unit = _mx_unit(w, q256=6 * 256)
    plane = unit.scale_base.reshape(rows, cols // BLOCK).long()
    # Along a row the exponent climbs three binades per block; down a column
    # it drops two per row.  Measured, not asserted from the rule: the pack
    # may pick either bracketing binade, so the step is 3 +- 1 and 2 +- 1.
    assert bool(((plane[:, 1:] - plane[:, :-1] - 3).abs() <= 1).all())
    assert bool(((plane[:-1, :] - plane[1:, :] - 2).abs() <= 1).all())
    assert plane.unique().numel() >= 6
    field = unit_scale_field(unit, rows, cols)
    assert torch.equal(
        field, torch.repeat_interleave(mx_scales_from_plane(unit.scale_base), BLOCK).reshape(rows, cols)
    )
    # Moving one word moves exactly its 32 positions of one row.
    back = reconstruct_unit(unit, E4M3_GRID, None)
    bumped = unit.scale_base.clone()
    bumped[2 * (cols // BLOCK) + 1] += 1
    moved = reconstruct_unit(unit, E4M3_GRID, None, scale=torch.repeat_interleave(
        mx_scales_from_plane(bumped), BLOCK).reshape(rows, cols)) != back
    expect = torch.zeros(rows, cols, dtype=torch.bool)
    expect[2, BLOCK:2 * BLOCK] = back[2, BLOCK:2 * BLOCK] != 0
    assert torch.equal(moved, expect)


def test_boundary_blocks_never_straddle_rows_and_off_block_widths_are_refused():
    rows, cols = 4, 64
    torch.manual_seed(2)
    w = torch.randn(rows, cols) * 0.02
    w[1, BLOCK:] *= 64.0      # last block of row 1 loud
    w[2, :BLOCK] /= 64.0      # first block of row 2 quiet
    unit = _mx_unit(w, q256=4 * 256)
    plane = unit.scale_base.reshape(rows, 2).long()
    assert int(plane[1, 1]) > int(plane[2, 0]) + 8
    manifest, _r, blob = _blob(unit, 4 * 256)
    parsed = parse_unit_artifact(blob)
    assert torch.equal(reconstruct_unit(parsed.unit, parsed.forests, parsed.code),
                       reconstruct_unit(unit, E4M3_GRID, None))
    # 48 columns is one and a half blocks: refused by the encoder and the
    # writer with the same words, from ``grammar.require_scale_groups``.
    with pytest.raises(GrammarError, match="whole number of 32"):
        _mx_unit(torch.randn(4, 48) * 0.02, q256=4 * 256)
    narrow = _mx_unit(w, q256=4 * 256)
    narrow.body_bits = narrow.body_bits[:, :48].contiguous()
    narrow.rates = narrow.rates[:48]
    narrow.codes = narrow.codes[:, :48]
    narrow.anchors = narrow.anchors[:, :48]
    narrow.completion_bits = narrow.completion_bits[:, :48]
    with pytest.raises(GrammarError, match="whole number of 32"):
        _blob(narrow, 4 * 256)


# ------------------------------------------------------ exact accounting


@pytest.mark.parametrize("rows,cols", [(8, 256), (16, 512), (6, 96), (32, 1024)])
@pytest.mark.parametrize("q256", [4 * 256, 6 * 256, 8 * 256])
def test_the_scale_plane_is_priced_at_a_quarter_bit_and_the_total_is_the_bytes(rows, cols, q256):
    """priced == written == served: the accountant, the wire and the pair agree."""
    cap = E4M3_GRID.payload_bits
    priced = terminal_rate(q256, rows, cols, with_scale_base=True, with_scale_refine=False,
                           cap=cap, window_bits=8)
    body_only = terminal_rate(q256, rows, cols, with_scale_base=False, cap=cap, window_bits=8)
    assert priced - body_only == Fraction(1, 4)
    w = _weights(rows, cols, seed=rows + cols)
    unit = _mx_unit(w, q256=q256, refit=1)
    manifest, region, blob = _blob(unit, q256)
    art = parse(blob)
    terminal = art.terminal
    assert terminal.exact_bpp == priced
    descriptor = manifest.plane(PlaneKind.SCALE_BASE)
    assert descriptor.element_count == rows * cols // BLOCK
    assert descriptor.byte_length() == rows * cols // BLOCK
    report = plane_byte_report(manifest, terminal, _side_bytes(art))
    assert report.scale_bytes == rows * cols // BLOCK
    assert report.scale_bpp == Fraction(1, 4)
    assert report.padding_bytes == 0
    assert report.plane_region_bytes == len(art.plane_region) == terminal.exact_bytes == len(region)
    assert report.total_bytes == len(blob)
    assert report.payload_bpp == terminal.exact_bpp
    by_kind = {row.kind: row for row in report.planes}
    assert by_kind[PlaneKind.SCALE_REFINE].total_bytes == 0
    assert by_kind[PlaneKind.DIAG_SV].total_bytes == 0
    assert by_kind[PlaneKind.ALPHABET].total_bytes == 1 << 8
    assert sum(row.total_bytes for row in report.planes) == report.plane_region_bytes
    assert account_terminal(manifest, terminal, _side_bytes(art), len(art.plane_region)).agrees
    # Served: the pair a block-scaled kernel reads is one byte per weight plus
    # one per 32, which is what the plane region charged for those planes.
    tile, scales = materialize_mxfp8(unit, E4M3_GRID, None)
    assert tile.numel() == rows * cols and scales.numel() == report.scale_bytes
    assert scales.numel() * 8 == report.scale_bpp * rows * cols


def test_padding_is_charged_and_reported_where_alignment_demands_it():
    w = _weights(6, 96, seed=4)
    unit = _mx_unit(w, q256=5 * 256, refit=0)
    aligned = build_unit_artifact(unit, "u", E4M3_GRID, 5 * 256, None, alignment_bytes=64)
    art = parse(aligned[2])
    report = plane_byte_report(art.manifest, art.terminal, _side_bytes(art))
    assert report.padding_bytes > 0
    assert report.scale_bytes >= 6 * 96 // BLOCK
    assert report.plane_region_bytes == len(art.plane_region) == art.terminal.exact_bytes
    assert report.total_bytes == len(aligned[2])
    base = next(row for row in report.planes if row.kind is PlaneKind.SCALE_BASE)
    assert base.content_bytes == 6 * 96 // BLOCK and base.total_bytes % 64 == 0


# ------------------------------------------------------- decoder parity


def test_materialize_mxfp8_is_the_reference_decode_bit_for_bit():
    w = _weights(16, 256, seed=8)
    unit = _mx_unit(w)
    _m, _r, blob = _blob(unit, 6 * 256)
    parsed = parse_unit_artifact(blob)
    tile, scales = materialize_mxfp8(parsed.unit, parsed.forests, parsed.code)
    assert tile.dtype is torch.uint8 and tile.shape == (16, 256)
    assert scales.dtype is torch.uint8 and scales.shape == (16, 8)
    assert not any(int(b) in E4M3FN_NAN_BYTES for b in tile.unique())
    assert int(scales.max()) < E8M0_NAN_BYTE
    reference = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    assert torch.equal(mxfp8_dequantize(tile, scales), reference)
    assert torch.equal(scales, parsed.unit.scale_base.reshape(16, 8))
    # ``scales[r, b]`` scales ``tile[r, 32 b : 32 b + 32]`` and nothing else.
    bumped = scales.clone()
    bumped[5, 3] += 1
    moved = mxfp8_dequantize(tile, bumped) != reference
    expect = torch.zeros_like(moved)
    expect[5, 3 * BLOCK:4 * BLOCK] = reference[5, 3 * BLOCK:4 * BLOCK] != 0
    assert torch.equal(moved, expect)


@pytest.mark.parametrize("axis", ["row", "column"])
@pytest.mark.parametrize("tp", [2, 4])
def test_a_rank_slice_decodes_and_materialises_to_its_window(axis, tp):
    w = _weights(16, 512, seed=11)
    unit = _mx_unit(w)
    _m, _r, blob = _blob(unit, 6 * 256)
    parsed = parse_unit_artifact(blob)
    assert shard_granularity(parsed.unit, 256, 1) == (1, BLOCK)
    assert shard_granularity(parsed.manifest) == (1, BLOCK)
    full = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
    tile, scales = materialize_mxfp8(parsed.unit, parsed.forests, parsed.code)
    extent = 16 if axis == "row" else 512
    step = extent // tp
    for rank in range(tp):
        lo, hi = rank * step, (rank + 1) * step
        cut = dict(rows=(lo, hi)) if axis == "row" else dict(cols=(lo, hi))
        shard = slice_unit(parsed, **cut)
        window = full[lo:hi] if axis == "row" else full[:, lo:hi]
        assert torch.equal(reconstruct_unit(shard, parsed.forests, parsed.code), window)
        s_tile, s_scales = materialize_mxfp8(shard, parsed.forests, parsed.code)
        if axis == "row":
            assert torch.equal(s_tile, tile[lo:hi]) and torch.equal(s_scales, scales[lo:hi])
        else:
            assert torch.equal(s_tile, tile[:, lo:hi])
            assert torch.equal(s_scales, scales[:, lo // BLOCK:hi // BLOCK])
        # The shard is a whole artifact at minor 8 and reads back from bytes.
        sm, _sr, sblob = build_unit_artifact(shard, f"r{rank}", parsed.forests, 6 * 256, parsed.code)
        assert sm.schema_minor == 8 and sm.scale_plane.kind is MX
        reread = parse_unit_artifact(sblob)
        assert torch.equal(reconstruct_unit(reread.unit, reread.forests, reread.code), window)
    # A cut inside a block is refused, naming the block, and can_shard says no
    # before the cutter would: 512 columns shard 16 ways (32 each), not 32.
    assert can_shard(parsed, 16, "column") and not can_shard(parsed, 32, "column")
    with pytest.raises(GrammarError, match="32"):
        slice_unit(parsed, cols=(16, 48))


def test_expert_selection_through_the_fused_and_moe_containers():
    """Three experts, three roles each: the selected expert's pair is its own."""
    experts, hidden, inter = 3, 128, 64
    q256 = 6 * 256
    pairs, w13, w2 = {}, [], []
    for e in range(experts):
        member_blobs = {}
        for role, (rows, cols) in (("gate_proj", (inter, hidden)),
                                   ("up_proj", (inter, hidden)),
                                   ("down_proj", (hidden, inter))):
            w = _weights(rows, cols, seed=100 * e + rows + cols) * (1.0 + e)
            unit = _mx_unit(w, q256=q256, refit=1)
            _m, _r, blob = _blob(unit, q256, name=f"e{e}.{role}")
            member_blobs[role] = (rows, blob)
            pairs[(e, role)] = materialize_mxfp8(unit, E4M3_GRID, None)
        w13.append([pack_fused([("gate_proj", *member_blobs["gate_proj"])]),
                    pack_fused([("up_proj", *member_blobs["up_proj"])])])
        w2.append(pack_fused([("down_proj", *member_blobs["down_proj"])]))
    packed = pack_moe_wires(w13, w2)
    back13, back2 = unpack_moe_wires(packed)
    assert back13 == w13 and back2 == w2
    for e in (2, 0, 1):
        for p, role in ((0, "gate_proj"), (1, "up_proj")):
            member, = parse_fused(back13[e][p])
            assert member.name == role
            parsed = parse_unit_artifact(member.blob)
            assert parsed.manifest.scale_plane.kind is MX
            tile, scales = materialize_mxfp8(parsed.unit, parsed.forests, parsed.code)
            want_tile, want_scales = pairs[(e, role)]
            assert torch.equal(tile, want_tile) and torch.equal(scales, want_scales)
            assert torch.equal(mxfp8_dequantize(tile, scales),
                               reconstruct_unit(parsed.unit, parsed.forests, parsed.code))
        member, = parse_fused(back2[e])
        parsed = parse_unit_artifact(member.blob)
        tile, scales = materialize_mxfp8(parsed.unit, parsed.forests, parsed.code)
        assert torch.equal(tile, pairs[(e, "down_proj")][0])
        assert torch.equal(scales, pairs[(e, "down_proj")][1])
    # Different experts are different pairs: selection is not a no-op.
    assert not torch.equal(pairs[(0, "gate_proj")][1], pairs[(1, "gate_proj")][1])


# ------------------------------------------------------------ refusals


def test_unsupported_combinations_are_refused_where_bytes_are_decided():
    w = _weights()
    rates2 = bresenham_rate_schedule(root_from_q256(2 * 256), 256, cap=E2M1_GRID.payload_bits)
    with pytest.raises(GrammarError, match="MX scale plane needs a scalar 256-code"):
        encode_unit(w, E2M1_GRID, rates2, None, body=WINDOW, window_bits=6, scale_plane=MX)
    rates16 = bresenham_rate_schedule(root_from_q256(12 * 256), 256, cap=BF16_GRID.payload_bits)
    with pytest.raises(GrammarError, match="MX scale plane needs a scalar 256-code"):
        encode_unit(w, BF16_GRID, rates16, None, body=WINDOW, window_bits=12, scale_plane=MX)
    with pytest.raises(GrammarError, match="DIAG_SU/DIAG_SV"):
        _mx_unit(w, with_diagonals=True)
    with pytest.raises(GrammarError, match="K32"):
        _mx_unit(w, group=16, half=16)
    with pytest.raises(GrammarError, match="couples"):
        _mx_unit(w, refit_metric=torch.eye(256))
    with pytest.raises(GrammarError, match="couples"):
        _mx_unit(w, refit_metric_trailing=torch.eye(256))
    diagonal = _mx_unit(w, refit_metric=torch.linspace(0.5, 2.0, 256))
    assert diagonal.scale_plane is MX and diagonal.scale_base.numel() == 16 * 256 // BLOCK
    with pytest.raises(GrammarError, match="silently ignored"):
        _mx_unit(w, refit_reach_floor=True)


def test_the_writer_refuses_a_unit_that_is_not_the_plane_it_claims():
    w = _weights()
    unit = _mx_unit(w)
    wrong_refine = _mx_unit(w)
    wrong_refine.scale_refine = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(GrammarError, match="no SCALE_REFINE"):
        _blob(wrong_refine, 6 * 256)
    short = _mx_unit(w)
    short.scale_base = unit.scale_base[:-1]
    with pytest.raises(GrammarError, match="one E8M0 word per 32"):
        _blob(short, 6 * 256)
    wrong_group = _mx_unit(w)
    wrong_group.group = 16
    with pytest.raises(GrammarError, match="K32"):
        _blob(wrong_group, 6 * 256)
    with_global = _mx_unit(w)
    with_global.scale_global = 2.0
    with pytest.raises(GrammarError, match="no global"):
        _blob(with_global, 6 * 256)
    reserved = _mx_unit(w)
    reserved.scale_base = unit.scale_base.clone()
    reserved.scale_base[0] = E8M0_NAN_BYTE
    with pytest.raises(ScaleCodecError, match="reserved E8M0 NaN word"):
        _blob(reserved, 6 * 256)


def test_the_reader_fails_closed_on_a_header_too_old_to_name_the_plane():
    unit = _mx_unit(_weights())
    _m, _r, blob = _blob(unit, 6 * 256)
    assert blob[10] == 8
    older = bytearray(blob)
    older[10] = 7
    with pytest.raises(TesseraError, match="minor 8"):
        parse(bytes(older))
    assert tuple(SCHEMA_MINORS_READ) == tuple(range(MX_SCHEMA_MINOR + 1))


def test_no_consumer_falls_through_to_another_plane_s_layout():
    unit = _mx_unit(_weights())
    with pytest.raises(GrammarError, match="no per-half scales"):
        unit_half_scales(unit)
    with pytest.raises(GrammarError, match="block layout"):
        materialize_fp8(unit, E4M3_GRID, None)
    with pytest.raises(GrammarError, match="materialize_mxfp8"):
        materialize_stock(unit, E4M3_GRID, None)
    from tessera.lane_planes import _pack_window_unit

    with pytest.raises(GrammarError, match="MX plane"):
        _pack_window_unit(unit, E4M3_GRID)
    channel = encode_unit(_weights(), E4M3_GRID, _rates(256, 6 * 256), None, body=WINDOW,
                          window_bits=8, scale_plane=ScalePlaneKind.CHANNEL, scale_refit=1)
    with pytest.raises(GrammarError, match="MXFP8 pair"):
        materialize_mxfp8(channel, E4M3_GRID, None)


def test_the_exporter_writes_the_unit_and_refuses_the_checkpoint():
    """``encode_linear`` is the opt-in surface; no config can spell the plane."""
    w = _weights()
    exported = encode_linear(w, grid=E4M3_GRID, q256=6 * 256, scale_plane=MX, scale_refit=1)
    art = parse(exported.blob)
    assert art.manifest.scale_plane.kind is MX and art.manifest.schema_minor == 8
    parsed = parse_unit_artifact(exported.blob)
    tile, scales = materialize_mxfp8(parsed.unit, parsed.forests, parsed.code)
    assert torch.equal(mxfp8_dequantize(tile, scales),
                       reconstruct_unit(parsed.unit, parsed.forests, parsed.code))
    recipe = WireRecipe(body=WINDOW, span=1, scale_plane=MX, window_bits=8)
    with pytest.raises(GrammarError, match="bullet 6"):
        recipe.to_config()
    table = recipe_table(E4M3_GRID, lambda _grid, _q: recipe)
    with pytest.raises(GrammarError, match="no spelling for the MX"):
        _require_config_spellable(table)
    from tessera.serving import scheme as S

    dense = {"family": S.TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "MX",
             "rows": 16, "columns": 256, "q256": 6 * 256, "wire_bytes": len(exported.blob),
             "roles": [["weight", 16]]}
    with pytest.raises(ValueError, match="has no FP8 tile"):
        S.validate_tessera_scheme(dense, "d")


# ------------------------------------------------- the other planes' bytes


def test_every_other_plane_keeps_its_minor_and_the_recipe_minor_is_unchanged():
    assert SCHEMA_MINOR == 7 and MX_SCHEMA_MINOR == 8
    w = _weights()
    rates = _rates(256, 6 * 256)
    for kind in (ScalePlaneKind.CHANNEL, ScalePlaneKind.LUT):
        unit = encode_unit(w, E4M3_GRID, rates, None, body=WINDOW, window_bits=8,
                           scale_plane=kind, scale_refit=1)
        manifest, _r, blob = _blob(unit, 6 * 256)
        assert manifest.schema_minor == SCHEMA_MINOR and blob[10] == SCHEMA_MINOR
    exported = encode_linear(w, grid=E4M3_GRID, q256=6 * 256)
    assert parse(exported.blob).manifest.schema_minor == SCHEMA_MINOR
