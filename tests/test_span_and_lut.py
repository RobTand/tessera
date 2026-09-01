"""Schema minor 1: the span-L trellis and the LUT scale plane.

Two wire changes, one measurement behind them
(``docs/measurements/tessera-index-plane-2026-09-01.md``): a 4-bit index per
16 weights into a per-unit E4M3 table reproduces the 8-bit plane to the third
digit at half the bytes, and the freed quarter-bit spent on Wei's span-2
partition is 1.125x at the same 4.0 bpp on the GLM experts.  These tests hold
the implementation to the properties the measurement relied on:

  * the vectorised span-L Viterbi is *exact* -- its summed squared error equals
    the scalar oracle's to the digit, and its bits replay to its anchors under
    the oracle's decoder, at every rate and span;
  * span 1 over S6b is untouched: it serialises as a minor-0 artifact and the
    profile id is the pre-minor-1 digest;
  * a LUT plane's bytes decode to the encoder's scales exactly, the table is
    sixteen distinct ascending E4M3 bytes, and the refit is monotone;
  * the full seam round-trips at span 2 over a LUT plane, mixed rates included,
    and ``E2M1_K2_R896`` weighs exactly 4.0 bpp of plane region;
  * the reader fails closed when a manifest's span disagrees with the profile
    id, and the kernel lane refuses a span it cannot decode.
"""
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import E2M1_GRID, build_forest, tuple_grid
from tessera.container import parse
from tessera.decode import reconstruct_unit, replay_body
from tessera.encode import (
    _pack_scales_lut,
    _refit_scales_lut,
    encode_unit,
    viterbi_columns,
)
from tessera.errors import GrammarError, ManifestError, SchemaError, TesseraError
from tessera.export import DEFAULT_SCALE_PLANE, DEFAULT_SPAN
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import ScalePlane, ScalePlaneKind
from tessera.trellis import TCQ, ConvCode, body_bits
from tessera.unit_artifact import (
    build_unit_artifact,
    encoder_profile_id,
    read_unit_artifact,
)
from tessera.wire import field_widths, pack_body, scales_from_lut, unpack_body

CODE = ConvCode(memory=6)
SMALL = ConvCode(memory=3)
K2 = tuple_grid(E2M1_GRID, 2)
FORESTS1 = {r: build_forest(r) for r in (1, 2, 3)}
FORESTS2 = {7: build_forest(7, grid=K2)}


def _weights(rows=64, cols=512, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, cols) * 0.02


# ------------------------------------------------------------ the trellis


@pytest.mark.parametrize("rate", [1, 2, 3])
@pytest.mark.parametrize("span", [1, 2, 3, 4])
def test_span_viterbi_equals_the_scalar_oracle(rate, span):
    """The min-plus fold over Z/4 is the exhaustive label search, exactly."""
    torch.manual_seed(rate * 10 + span)
    targets = torch.randn(24, 4) * 0.54
    forest = build_forest(rate)
    anchors, bits, sse = viterbi_columns(targets, forest, SMALL, 3 - rate, span=span)
    tcq = TCQ(forest, SMALL)
    oracle = 0.0
    for column in range(4):
        _, _, sse_col = tcq.encode(targets[:, column].tolist(), 3 - rate, span=span)
        oracle += sse_col
        stream = []
        widths = field_widths(rate, span)
        for step in range(24):
            width = widths[step % span]
            field = int(bits[step, column])
            stream += [(field >> k) & 1 for k in range(width - 1, -1, -1)]
        assert len(stream) == body_bits(rate, 24, span)
        assert tcq.decode(stream, 24, span) == anchors[:, column].tolist()
    assert sse == pytest.approx(oracle, rel=1e-6)


def test_span_two_on_the_tuple_grid_equals_the_oracle():
    torch.manual_seed(0)
    targets = torch.randn(32, 3)
    forest = FORESTS2[7]
    _, _, sse = viterbi_columns(targets, forest, CODE, 0, span=2)
    oracle = sum(
        TCQ(forest, CODE).encode(targets[:, j].reshape(-1, 2).tolist(), 0, span=2)[2]
        for j in range(3)
    )
    assert sse == pytest.approx(oracle, rel=1e-6)


def test_replay_lands_on_the_encoders_anchors_at_span_two():
    w = _weights()
    unit = encode_unit(w, FORESTS2, (7,) * 512, CODE, span=2, scale_refit=0)
    assert torch.equal(replay_body(unit.body_bits, FORESTS2[7], CODE, span=2), unit.anchors)
    assert torch.equal(
        unpack_body(pack_body(unit.body_bits, unit.rates, 2), unit.rates, 32, span=2),
        unit.body_bits,
    )


def test_a_column_that_is_not_a_whole_number_of_super_symbols_refuses():
    with pytest.raises(GrammarError, match="super-symbols"):
        encode_unit(_weights(rows=66), FORESTS2, (7,) * 512, CODE, span=4)
    with pytest.raises(GrammarError, match="super-symbols"):
        encode_unit(_weights(rows=65), FORESTS1, (3,) * 512, CODE, span=2)


# ------------------------------------------------------------- the plane


def test_lut_plane_decodes_to_the_encoders_scales_exactly():
    w = _weights()
    unit = encode_unit(w, FORESTS2, (7,) * 512, CODE, scale_plane=ScalePlaneKind.LUT)
    table = unit.scale_lut
    assert table.numel() == 16
    assert bool((table[1:] > table[:-1]).all()), "distinct, ascending"
    assert 1 <= int(table.min()) and int(table.max()) <= 0x7E, "positive finite E4M3FN"
    assert unit.scale_base.numel() == 0
    assert int(unit.scale_refine.max()) < 16
    decoded = scales_from_lut(unit.scale_refine, table, unit.scale_global)
    _, _, effective, _ = _pack_scales_lut(w, 16, peak=6.0)
    # The refit moved the plane; the decoder reads what the encoder holds.
    rows, cols = w.shape
    scale = torch.repeat_interleave(decoded, 16).reshape(rows, cols)
    assert torch.equal(reconstruct_unit(unit, FORESTS2, CODE),
                       reconstruct_unit(unit, FORESTS2, CODE, scale))


def test_scale_weighted_trellis_never_ends_a_column_worse():
    """With the branch metric weighted by the position's scale squared the
    Viterbi minimises ``sum (w - c q)^2`` exactly (the cap rate has no
    completion subtree, so the metric is the leaf error), so on a fixed plane
    (refit 0) no column can end with more true error than the unweighted
    path, and the total is lower whenever the plane varies along a column."""
    w = _weights()
    plain = encode_unit(w, FORESTS2, (7,) * 512, CODE, scale_refit=0, completion=0)
    weighted = encode_unit(w, FORESTS2, (7,) * 512, CODE, scale_refit=0, completion=0,
                           trellis_weighting="scale")
    per_col = lambda u: ((reconstruct_unit(u, FORESTS2, CODE) - w) ** 2).sum(dim=0)
    a, b = per_col(plain), per_col(weighted)
    assert bool((b <= a * (1 + 1e-5) + 1e-12).all()), int((b > a).sum())
    assert float(b.sum()) < float(a.sum())
    with pytest.raises(GrammarError, match="trellis_weighting"):
        encode_unit(w, FORESTS2, (7,) * 512, CODE, trellis_weighting="hessian")


def test_lut_refit_is_monotone_and_beats_its_amax_start():
    w = _weights()
    errors = []
    for k in range(4):
        unit = encode_unit(w, FORESTS2, (7,) * 512, CODE,
                           scale_plane=ScalePlaneKind.LUT, scale_refit=k)
        errors.append(float(((reconstruct_unit(unit, FORESTS2, CODE) - w) ** 2).sum()))
    assert all(a >= b for a, b in zip(errors, errors[1:])), errors
    assert errors[3] < 0.97 * errors[0], errors


def test_lut_refit_never_accepts_a_worse_table():
    """The candidate set is {old table, new fit}; the lower cost wins."""
    w = _weights()
    unit = encode_unit(w, FORESTS2, (7,) * 512, CODE, scale_plane=ScalePlaneKind.LUT)
    from tessera.encode import grid_vector_table
    vectors = grid_vector_table(K2)
    units = vectors[unit.codes].permute(0, 2, 1).reshape(w.shape)
    effective = scales_from_lut(unit.scale_refine, unit.scale_lut, unit.scale_global)
    W = w.reshape(-1, 16); U = units.reshape(-1, 16)
    A, B = (U * U).sum(1), (W * U).sum(1)
    before = float((A * effective * effective - 2 * B * effective).sum())
    _, _, after_eff = _refit_scales_lut(
        w, units, 16, unit.scale_lut, unit.scale_refine, effective, unit.scale_global
    )
    after = float((A * after_eff * after_eff - 2 * B * after_eff).sum())
    assert after <= before + 1e-6 * abs(before)


# --------------------------------------------------------------- the seam


@pytest.mark.parametrize("q256,diagonals", [(768, False), (640, True), (512, False)])
def test_wire_round_trip_at_span_two_over_a_lut_plane(q256, diagonals):
    w = _weights()
    rates = bresenham_rate_schedule(root_from_q256(q256), 512)
    unit = encode_unit(w, FORESTS1, rates, CODE, span=2,
                       scale_plane=ScalePlaneKind.LUT, with_diagonals=diagonals)
    _, region, blob = build_unit_artifact(unit, "unit0", FORESTS1, q256, CODE)
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, FORESTS1, CODE))
    art = parse(blob)
    assert blob[10] == 1, "schema minor 1"
    assert art.manifest.span == 2
    assert art.manifest.scale_plane.kind is ScalePlaneKind.LUT
    assert len(art.manifest.scale_plane.table) == 16
    assert art.terminal.exact_bytes == len(region)


def test_e2m1_k2_r896_weighs_exactly_four_bits_per_parameter():
    """3.75 of body at span 2 plus 0.25 of LUT index: the same 4.0 bpp as the
    span-1 S6b wire, on a unit large enough that the forest planes vanish."""
    w = _weights(rows=512, cols=1024)
    unit = encode_unit(w, FORESTS2, (7,) * 1024, CODE, span=2,
                       scale_plane=ScalePlaneKind.LUT)
    _, region, blob = build_unit_artifact(unit, "unit0", FORESTS2, 7 * 256, CODE)
    from tessera.planes import CANONICAL_PLANE_ORDER, PlaneKind

    terminal = parse(blob).terminal
    forest_bytes = sum(
        terminal.plane_elements[CANONICAL_PLANE_ORDER.index(kind)]
        for kind in (PlaneKind.ALPHABET, PlaneKind.DESCENDANT)
    )
    assert forest_bytes == 512, "256 anchors, one descendant each, one byte apiece"
    assert terminal.exact_bpp - Fraction(8 * forest_bytes, w.numel()) == Fraction(4)


def test_the_exporter_defaults_are_the_new_wire():
    assert DEFAULT_SPAN == 2
    assert DEFAULT_SCALE_PLANE is ScalePlaneKind.LUT


def test_span_one_over_s6b_is_still_a_minor_zero_artifact():
    """Every artifact written before minor 1 is reproducible: same bytes,
    same header, same profile id."""
    w = _weights()
    unit = encode_unit(w, FORESTS2, (7,) * 512, CODE)
    assert unit.span == 1 and unit.scale_plane is ScalePlaneKind.S6B
    manifest, _, blob = build_unit_artifact(unit, "unit0", FORESTS2, 7 * 256, CODE)
    assert blob[10] == 0
    assert manifest.schema_minor == 0
    assert manifest.encoder_profile_id == encoder_profile_id(CODE, unit.rates, K2)
    assert manifest.encode() == manifest.encode(0)
    with pytest.raises(ManifestError, match="needs minor"):
        # a minor-1 manifest cannot be squeezed into minor 0
        parse(blob).manifest.__class__(
            **{**manifest.__dict__, "span": 2}
        ).encode(0)


def test_profile_id_binds_span_and_plane_conditionally():
    rates = (7,) * 8
    base = encoder_profile_id(CODE, rates, K2)
    assert encoder_profile_id(CODE, rates, K2, 1, ScalePlaneKind.S6B) == base
    assert encoder_profile_id(CODE, rates, K2, 2, ScalePlaneKind.S6B) != base
    assert encoder_profile_id(CODE, rates, K2, 1, ScalePlaneKind.LUT) != base
    assert encoder_profile_id(CODE, rates, K2, 2, ScalePlaneKind.LUT) not in {
        base,
        encoder_profile_id(CODE, rates, K2, 2, ScalePlaneKind.S6B),
        encoder_profile_id(CODE, rates, K2, 1, ScalePlaneKind.LUT),
    }


def test_a_manifest_whose_span_disagrees_with_the_profile_fails_closed():
    from tessera.container import serialize

    w = _weights()
    unit = encode_unit(w, FORESTS2, (7,) * 512, CODE, span=2,
                       scale_plane=ScalePlaneKind.LUT)
    manifest, region, _ = build_unit_artifact(unit, "unit0", FORESTS2, 7 * 256, CODE)
    lying = manifest.__class__(**{**manifest.__dict__, "span": 1})
    # span 1 has a different body size, so the layout itself disagrees first;
    # the digest check is the backstop for a forged manifest with a matching
    # plane region.  Either way: refused, never a silent misdecode.
    with pytest.raises(TesseraError):
        read_unit_artifact(serialize(lying, region))


def test_scale_plane_record_validates_its_table():
    with pytest.raises(ManifestError, match="ascending"):
        ScalePlane.lut(bytes([8, 8, 9]), 1.0)              # not strictly ascending
    with pytest.raises(ManifestError, match="normal"):
        ScalePlane.lut(bytes([0, 8, 9]), 1.0)              # zero is not a scale
    with pytest.raises(ManifestError, match="normal"):
        ScalePlane.lut(bytes([7, 8, 9]), 1.0)              # subnormal: the kernel
    with pytest.raises(ManifestError, match="normal"):     # misdecodes it
        ScalePlane.lut(bytes([8, 9, 0x7F]), 1.0)           # NaN byte
    with pytest.raises(ManifestError, match="no table"):
        ScalePlane(ScalePlaneKind.S6B, bytes([8, 9]))      # S6b carries no table
    ScalePlane.lut(bytes(range(8, 24)), 2.0 ** -10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel lane is a CUDA path")
def test_the_kernel_lane_packs_span_2_and_refuses_a_span_it_cannot_decode():
    """Span 2 is the shipping wire and packs to three planes; span 3 is a
    body the kernel has no decode for and is refused at the seam rather
    than decoded as if it were span 1."""
    from tessera.kernel import pack_kernel_planes

    w = _weights(rows=256, cols=512).cuda()
    unit = encode_unit(w, {3: FORESTS1[3]}, (3,) * 512, CODE, span=2, scale_refit=0)
    select, label, point = pack_kernel_planes(unit.body_bits, rate=3, span=2)
    assert point.numel() == 256 * 512 * 2 // 8
    assert label.numel() == 128 * 512 * 2 // 8
    unit3 = encode_unit(w[:192], {3: FORESTS1[3]}, (3,) * 512, CODE, span=3, scale_refit=0)
    with pytest.raises(GrammarError, match="span"):
        pack_kernel_planes(unit3.body_bits, rate=3, span=3)
