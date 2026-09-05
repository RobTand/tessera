"""The kernel lane: Tessera bits consumed by a GEMV without materialising NVFP4.

Every test here compares against ``reconstruct_unit`` rather than against a
tolerance where it can, because the kernel and the reference decoder are two
implementations of one grammar and "close" is the wrong relationship for them
to have.
"""

from __future__ import annotations

import pytest
import torch

from tessera.alphabet import build_forest
from tessera.decode import decode_codes_mixed, materialize_nvfp4, reconstruct_unit
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import RotationState
from tessera.trellis import ConvCode
from tessera.wire import pack_body

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel lane is a CUDA path"
)

CODE = ConvCode(memory=6)


@pytest.fixture(scope="module")
def unit():
    """A full-rate, release-free unit -- the configuration the kernel lane serves.

    R=3 everywhere means completion width 0, and release=0 means no
    data-dependent positions, which together are what make the body a pure
    windowed function of the bitstream.
    """
    device = "cuda"
    forests = {r: build_forest(r) for r in (1, 2, 3)}
    torch.manual_seed(0)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device=device) * 0.02).contiguous()
    rates = bresenham_rate_schedule(root_from_q256(768), cols)
    assert set(rates) == {3}
    encoded = encode_unit(
        weights, forests, rates, CODE,
        rotation=RotationState.NONE, with_diagonals=False, released_positions=0,
    )
    codes = decode_codes_mixed(encoded, forests, CODE)
    _packed, e4m3, global_scale = materialize_nvfp4(
        codes, encoded.scale_base, encoded.scale_refine, encoded.group, encoded.half
    )
    return {
        "unit": encoded, "forests": forests, "rates": rates, "codes": codes,
        "rows": rows, "cols": cols, "e4m3": e4m3, "global_scale": global_scale,
        "reference": reconstruct_unit(encoded, forests, CODE).float(),
    }


def test_dequant_kernel_is_bit_exact_against_the_reference_decoder(unit):
    from tessera.kernel import build_code_lut, tessera_dequant

    body = torch.frombuffer(
        bytearray(pack_body(unit["unit"].body_bits, unit["rates"]) + b"\x00"),
        dtype=torch.uint8,
    ).to("cuda")
    got = tessera_dequant(
        body, build_code_lut(unit["forests"][3], CODE), unit["e4m3"],
        unit["global_scale"], unit["rows"], unit["cols"],
    )
    assert torch.equal(got, unit["reference"])


def test_the_legacy_gemvs_count_every_column_exactly_once(unit):
    """A split whose width is not a whole number of ``BLOCK_K`` tiles.

    Both legacy GEMVs give program ``pid_k`` the columns ``[pid_k * span,
    pid_k * span + span)`` for ``span = cdiv(cols, SPLIT_K)``, and then step
    through them ``BLOCK_K`` at a time.  At the wrappers' own defaults over
    these 512 columns ``span`` is 32 and ``BLOCK_K`` is 64, so the *tile* runs
    32 columns past the split it belongs to -- and the tile is masked against
    ``cols``, not against the end of its split, so the neighbouring program
    accumulates those columns a second time.

    The two wrappers partition K identically, so agreeing with each other
    would prove nothing about either; each is compared against the reference
    decode instead.  A one-hot column is the sharp probe -- nothing is summed,
    so a doubled column is exactly ``2 * W[:, k]``.
    """
    from tessera.kernel import build_code_lut, nvfp4_gemv, tessera_gemv

    rows, cols = unit["rows"], unit["cols"]
    reference = unit["reference"]
    body = torch.frombuffer(
        bytearray(pack_body(unit["unit"].body_bits, unit["rates"]) + b"\x00"),
        dtype=torch.uint8,
    ).to("cuda")
    lut = build_code_lut(unit["forests"][3], CODE)
    packed, e4m3, gs = materialize_nvfp4(
        unit["codes"], unit["unit"].scale_base, unit["unit"].scale_refine,
        unit["unit"].group, unit["unit"].half,
    )
    # 32 and 96 are the first columns of splits 1 and 3 under the defaults --
    # the ones the previous split's 64-wide tile also reaches.
    for k in (0, 32, 63, 96, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        assert torch.equal(
            tessera_gemv(x, body, lut, unit["e4m3"], gs, rows, cols),
            reference[:, k],
        ), f"tessera_gemv column {k}"
        assert torch.equal(
            nvfp4_gemv(x, packed, e4m3, gs, rows, cols), reference[:, k]
        ), f"nvfp4_gemv column {k}"

    # And a whole dot product at a split width that is neither a multiple of
    # BLOCK_K nor a divisor of cols: cdiv(512, 5) = 103.
    torch.manual_seed(2)
    x = torch.randn(cols, device="cuda")
    want = reference @ x
    for split_k in (5, 16):
        got = tessera_gemv(x, body, lut, unit["e4m3"], gs, rows, cols,
                           split_k=split_k)
        assert (got - want).norm() / want.norm() < 1e-5, f"tessera split_k={split_k}"
        got = nvfp4_gemv(x, packed, e4m3, gs, rows, cols, split_k=split_k)
        assert (got - want).norm() / want.norm() < 1e-5, f"nvfp4 split_k={split_k}"


def test_sliced_planes_round_trip_through_the_wire_body(unit):
    """The resident layout is a permutation of the wire body, not new data."""
    from tessera.kernel import SELECT_PAD, pack_kernel_planes

    select, point = pack_kernel_planes(unit["unit"].body_bits)
    rows, cols = unit["rows"], unit["cols"]
    assert select.numel() == cols * (rows + SELECT_PAD) // 8
    assert point.numel() == cols * rows * 2 // 8
    # Same bits as the wire body: 3 per position, no more.
    assert (select.numel() + point.numel()) * 8 == rows * cols * 3 + cols * SELECT_PAD


def test_one_hot_gemv_decodes_each_column_bit_exactly(unit):
    """``x = e_k`` returns column k of W, so this compares the decode itself.

    Columns 0..7 matter most: their history window straddles the select plane's
    zero pad, which is the mechanism standing in for "the trellis starts in
    state 0" and the easiest thing in this layout to get wrong.
    """
    from tessera.kernel import build_value_lut, pack_kernel_planes, tessera_gemv_wide

    select, point = pack_kernel_planes(unit["unit"].body_bits)
    lut = build_value_lut(unit["forests"][3], CODE)
    rows, cols = unit["rows"], unit["cols"]
    scales = unit["e4m3"].reshape(rows, cols // 16).t().contiguous()
    for k in (0, 1, 5, 6, 7, 8, 33, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = tessera_gemv_wide(
            x, select, point, lut, scales, unit["global_scale"], rows, cols,
            lanes=32, split_k=4,
        )
        assert torch.equal(got, unit["reference"][:, k]), f"column {k}"


def test_a_strided_activation_is_the_vector_the_caller_wrote(unit):
    """A ``[cols]`` view with stride 2 is not ``cols`` contiguous elements.

    ``reshape(-1)`` on a tensor that already has the shape asked for returns
    the tensor itself, so the stride survives into the launch -- and the
    kernels read ``x_ptr + k``, which walks the *backing storage*.  A one-hot
    at column 33 of a stride-2 view therefore used to come back as column 66
    of the reference decode: a legal-looking answer to a question nobody
    asked.  The prefill GEMM has the same gap in two dimensions, where a
    column-sliced batch passes the ``[M, cols]`` guard and is then addressed
    as ``offs_m * cols + offs_k``.
    """
    from tessera.kernel import (
        build_code_lut, build_value_lut, pack_kernel_planes, tessera_gemm,
        tessera_gemv, tessera_gemv_wide,
    )

    rows, cols = unit["rows"], unit["cols"]
    reference = unit["reference"]
    select, point = pack_kernel_planes(unit["unit"].body_bits)
    values = build_value_lut(unit["forests"][3], CODE)
    scales = unit["e4m3"].reshape(rows, cols // 16).t().contiguous()
    body = torch.frombuffer(
        bytearray(pack_body(unit["unit"].body_bits, unit["rates"]) + b"\x00"),
        dtype=torch.uint8,
    ).to("cuda")

    k = 33
    backing = torch.zeros(2 * cols, device="cuda")
    backing[2 * k] = 1.0
    x = backing[::2]
    assert tuple(x.shape) == (cols,) and x.stride() == (2,)
    assert x.reshape(-1).stride() == (2,), "the flatten this lane relies on"
    got = tessera_gemv_wide(
        x, select, point, values, scales, unit["global_scale"], rows, cols,
        lanes=32, split_k=4,
    )
    assert torch.equal(got, reference[:, k]), "tessera_gemv_wide"
    got = tessera_gemv(
        x, body, build_code_lut(unit["forests"][3], CODE), unit["e4m3"],
        unit["global_scale"], rows, cols,
    )
    assert torch.equal(got, reference[:, k]), "tessera_gemv"

    torch.manual_seed(4)
    batch = torch.randn(3, 2 * cols, device="cuda")
    strided = batch[:, ::2]
    assert tuple(strided.shape) == (3, cols) and not strided.is_contiguous()
    got = tessera_gemm(
        strided, select, point, values, scales, unit["global_scale"], rows, cols
    )
    want = strided @ reference.t()
    assert torch.allclose(got, want, rtol=2e-5, atol=2e-4), "tessera_gemm"


def test_gemv_matches_the_materialised_path(unit):
    """The two lanes must agree: same artifact, same answer, different route."""
    from tessera.kernel import (
        build_value_lut, nvfp4_gemv_sliced, pack_kernel_planes,
        pack_nvfp4_column_major, tessera_gemv_wide,
    )

    rows, cols = unit["rows"], unit["cols"]
    select, point = pack_kernel_planes(unit["unit"].body_bits)
    scales = unit["e4m3"].reshape(rows, cols // 16).t().contiguous()
    torch.manual_seed(1)
    x = torch.randn(cols, device="cuda")
    tessera = tessera_gemv_wide(
        x, select, point, build_value_lut(unit["forests"][3], CODE), scales,
        unit["global_scale"], rows, cols, lanes=32, split_k=4,
    )
    stock = nvfp4_gemv_sliced(
        x, pack_nvfp4_column_major(unit["codes"]), scales,
        unit["global_scale"], rows, cols, block_n=64, split_k=4,
    )
    reference = unit["reference"] @ x
    assert (tessera - reference).norm() / reference.norm() < 1e-5
    assert (tessera - stock).norm() / stock.norm() < 1e-5


def test_the_kernel_lane_refuses_a_body_that_carries_completion_bits(unit):
    """Below R=3 a position needs completion bits, a plane this lane does not read.

    Failing closed matters more than usual here: the LUT would silently be built
    over the wrong descendant set and decode to plausible, wrong weights.
    """
    from tessera.errors import GrammarError
    from tessera.kernel import build_code_lut

    with pytest.raises(GrammarError, match="full-rate"):
        build_code_lut(unit["forests"][2], CODE)


@pytest.mark.parametrize("memory", [4, 8])
def test_the_scalar_kernels_read_the_history_their_code_declares(memory):
    """A history window is ``memory + 1`` bits wide, not seven.

    ``_sliced_gemv_kernel``, ``_wide_gemv_kernel`` and ``_gemm_kernel`` all
    masked the select-plane window to ``0x7F``, and two of them shifted it by
    a constant derived for ``memory = 6`` -- while their wrappers took any
    ``memory`` at all and ``build_history_lut``/``build_value_lut`` built the
    matching table for it without a word.  ``ConvCode(memory=8)`` has
    declared default generators and needs a nine-bit window; a shallower code
    forms indices past the end of its own smaller table.  Both decode against
    the reference here, and ``tests/test_kernel_shape_guards.py`` holds the
    named refusal past the select pad.
    """
    from tessera.kernel import (
        build_history_lut, build_value_lut, pack_kernel_planes, tessera_gemm,
        tessera_gemv_sliced, tessera_gemv_wide,
    )
    from tessera.wire import nvfp4_scale_bytes

    code = ConvCode(memory=memory)
    forest = build_forest(3)
    torch.manual_seed(19)
    rows, cols = 64, 64
    w = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    unit = encode_unit(w, {3: forest}, (3,) * cols, code,
                       rotation=RotationState.NONE, with_diagonals=False)
    reference = reconstruct_unit(unit, {3: forest}, code).float()
    select, point = pack_kernel_planes(unit.body_bits, 3, memory)
    e4m3, gs = nvfp4_scale_bytes(
        unit.scale_base, unit.scale_refine, unit.group, unit.half
    )
    scales = e4m3.reshape(rows, cols // 16).t().contiguous()
    nibbles = build_history_lut(forest, code)
    values = build_value_lut(forest, code)
    assert nibbles.numel() == (1 << (memory + 1)) * 4
    for k in (0, 1, 7, 8, 33, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = tessera_gemv_wide(
            x, select, point, values, scales, gs, rows, cols, memory=memory,
            lanes=8, split_k=2,
        )
        assert torch.equal(got, reference[:, k]), f"wide, column {k}"
        got = tessera_gemv_sliced(
            x, select, point, nibbles, scales, gs, rows, cols, memory=memory,
            block_n=64, split_k=2,
        )
        assert torch.equal(got, reference[:, k]), f"sliced, column {k}"
        probe = torch.zeros(1, cols, device="cuda")
        probe[0, k] = 1.0
        got = tessera_gemm(
            probe, select, point, values, scales, gs, rows, cols, memory=memory
        )[0]
        assert torch.equal(got, reference[:, k]), f"prefill, column {k}"


# --- the k-tuple lane -----------------------------------------------------


@pytest.fixture(scope="module")
def tuple_unit(request):
    """A k=2 body over the grid the parameter names, encoded at R = cap."""
    from tessera.alphabet import E2M1_GRID, lloyd_max_grid, tuple_grid

    base = E2M1_GRID if request.param == "E2M1" else lloyd_max_grid(16)
    grid = tuple_grid(base, 2)
    rate = grid.rate_cap
    forests = {rate: build_forest(rate, grid=grid)}
    torch.manual_seed(0)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    encoded = encode_unit(
        weights, forests, (rate,) * cols, CODE, rotation=RotationState.NONE,
        with_diagonals=False, completion=0,
    )
    from tessera.wire import nvfp4_scale_bytes

    e4m3, global_scale = nvfp4_scale_bytes(
        encoded.scale_base, encoded.scale_refine, encoded.group, encoded.half
    )
    return {
        "unit": encoded, "forests": forests, "rate": rate, "rows": rows,
        "cols": cols, "global_scale": global_scale,
        "scales": e4m3.reshape(rows, cols // 16).t().contiguous(),
        "reference": reconstruct_unit(encoded, forests, CODE, completion=0).float(),
    }


@pytest.mark.parametrize("tuple_unit", ["E2M1", "free-16"], indirect=True)
def test_tuple_one_hot_gemv_is_bit_exact(tuple_unit):
    """A code covers two rows, so an off-by-one in the fan-out is invisible
    in aggregate and total per column.  One-hot is what exposes it."""
    from tessera.kernel import (
        build_anchor_values, build_tuple_index_lut, pack_kernel_planes,
        tessera_gemv_tuple,
    )

    rate, rows, cols = tuple_unit["rate"], tuple_unit["rows"], tuple_unit["cols"]
    select, point = pack_kernel_planes(tuple_unit["unit"].body_bits, rate=rate)
    index = build_tuple_index_lut(tuple_unit["forests"][rate], CODE)
    values = build_anchor_values(tuple_unit["forests"][rate])
    for k in (0, 1, 5, 7, 8, 9, 33, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = tessera_gemv_tuple(
            x, select, point, index, values, tuple_unit["scales"],
            tuple_unit["global_scale"], rows, cols, rate=rate, arity=2,
            lanes=8, split_k=4,
        )
        assert torch.equal(got, tuple_unit["reference"][:, k]), f"column {k}"


@pytest.mark.parametrize("tuple_unit", ["free-16"], indirect=True)
def test_tuple_gemv_matches_the_reference_decode(tuple_unit):
    from tessera.kernel import (
        build_anchor_values, build_tuple_index_lut, pack_kernel_planes,
        tessera_gemv_tuple,
    )

    rate, rows, cols = tuple_unit["rate"], tuple_unit["rows"], tuple_unit["cols"]
    select, point = pack_kernel_planes(tuple_unit["unit"].body_bits, rate=rate)
    torch.manual_seed(1)
    x = torch.randn(cols, device="cuda")
    got = tessera_gemv_tuple(
        x, select, point,
        build_tuple_index_lut(tuple_unit["forests"][rate], CODE),
        build_anchor_values(tuple_unit["forests"][rate]),
        tuple_unit["scales"], tuple_unit["global_scale"], rows, cols,
        rate=rate, arity=2, lanes=8, split_k=4,
    )
    want = tuple_unit["reference"] @ x
    assert (got - want).norm() / want.norm() < 1e-5


@pytest.mark.parametrize("tuple_unit", ["E2M1"], indirect=True)
def test_tuple_kernel_refuses_shapes_its_shifts_do_not_cover(tuple_unit):
    from tessera.errors import GrammarError
    from tessera.kernel import (
        build_anchor_values, build_tuple_index_lut, pack_kernel_planes,
        tessera_gemv_tuple,
    )

    rate, rows, cols = tuple_unit["rate"], tuple_unit["rows"], tuple_unit["cols"]
    select, point = pack_kernel_planes(tuple_unit["unit"].body_bits, rate=rate)
    index = build_tuple_index_lut(tuple_unit["forests"][rate], CODE)
    values = build_anchor_values(tuple_unit["forests"][rate])
    x = torch.zeros(cols, device="cuda")
    with pytest.raises(GrammarError, match="derived for VEC=8"):
        tessera_gemv_tuple(
            x, select, point, index, values, tuple_unit["scales"],
            tuple_unit["global_scale"],
            rows, cols, rate=rate, arity=2, vec=4,
        )
    with pytest.raises(GrammarError, match="multiple of 8 codes"):
        tessera_gemv_tuple(
            x, select, point, index, values, tuple_unit["scales"],
            tuple_unit["global_scale"],
            rows=100, cols=cols, rate=rate, arity=2,
        )


def test_tuple_gemv_decodes_a_lut_scale_plane_bit_exact():
    """A LUT plane materialises to per-16 E4M3 bytes (``nvfp4_scale_bytes_lut``)
    and the kernel reads them by field arithmetic, ``2^(e-7) (1 + m/8)``.  The
    reference decoder reads the same bytes through the dtype.  They agree only
    while every table entry is a normal: a subnormal (exponent field 0) is
    ``2^-6 * m/8`` to the dtype and ``2^-7 * (1 + m/8)`` to the kernel, and the
    S6b path never produced one.  This is the test that pins the LUT plane to
    the kernel's contract, over a unit whose scales span several binades."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.kernel import (
        build_anchor_values, build_tuple_index_lut, pack_kernel_planes,
        tessera_gemv_tuple,
    )
    from tessera.manifest import ScalePlaneKind
    from tessera.wire import nvfp4_scale_bytes_lut

    grid = tuple_grid(E2M1_GRID, 2)
    rate = grid.rate_cap
    forests = {rate: build_forest(rate, grid=grid)}
    torch.manual_seed(3)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    weights[:64] *= 8.0
    weights[64:128] *= 0.125
    encoded = encode_unit(
        weights, forests, (rate,) * cols, CODE, rotation=RotationState.NONE,
        with_diagonals=False, completion=0, span=1,
        scale_plane=ScalePlaneKind.LUT,
    )
    assert encoded.scale_plane is ScalePlaneKind.LUT
    assert int(encoded.scale_lut.min()) >= 0x08, "a subnormal table entry"
    e4m3, global_scale = nvfp4_scale_bytes_lut(
        encoded.scale_refine, encoded.scale_lut, encoded.scale_global
    )
    scales = e4m3.reshape(rows, cols // 16).t().contiguous()
    reference = reconstruct_unit(encoded, forests, CODE, completion=0).float()
    select, point = pack_kernel_planes(encoded.body_bits, rate=rate)
    index = build_tuple_index_lut(forests[rate], CODE)
    values = build_anchor_values(forests[rate])
    for k in (0, 17, 255, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = tessera_gemv_tuple(
            x, select, point, index, values, scales, global_scale, rows, cols,
            rate=rate, arity=2, lanes=8, split_k=4,
        )
        assert torch.equal(got, reference[:, k]), f"column {k}"


# --- the span-2 lane: one select bit per pair, a label plane, LUT scales ---


@pytest.fixture(scope="module")
def span2_unit(request):
    """A span-2 body over the grid the parameter names, LUT scale plane, at
    R = cap.  256 rows = 128 codes = 8 lanes of 8 codes: every lane parity
    (pair index 0 or 4 mod 8) occurs, which is what the per-lane window shift
    is for."""
    from tessera.alphabet import E2M1_GRID, lloyd_max_grid, tuple_grid
    from tessera.manifest import ScalePlaneKind

    base = E2M1_GRID if request.param == "E2M1" else lloyd_max_grid(16)
    grid = tuple_grid(base, 2)
    rate = grid.rate_cap
    forests = {rate: build_forest(rate, grid=grid)}
    torch.manual_seed(5)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    weights[:32] *= 4.0
    encoded = encode_unit(
        weights, forests, (rate,) * cols, CODE, rotation=RotationState.NONE,
        with_diagonals=False, completion=0, span=2,
        scale_plane=ScalePlaneKind.LUT,
    )
    assert encoded.span == 2 and encoded.scale_plane is ScalePlaneKind.LUT
    return {
        "unit": encoded, "forest": forests[rate], "rate": rate, "rows": rows,
        "cols": cols,
        "reference": reconstruct_unit(encoded, forests, CODE, completion=0).float(),
    }


@pytest.mark.parametrize("span2_unit", ["E2M1", "free-16"], indirect=True)
def test_span2_one_hot_gemv_is_bit_exact(span2_unit):
    """Every column through the kernel equals the reference decode exactly:
    the derived label at even positions, the stored one at odd, the window
    at both lane parities, the nibble scale at both row parities."""
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    cols = span2_unit["cols"]
    packed = pack_unit_for_kernel(span2_unit["unit"], span2_unit["forest"], CODE)
    for k in (0, 1, 5, 7, 8, 15, 16, 17, 33, 255, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = gemv_from_packed(x, packed, lanes=8, split_k=4)
        assert torch.equal(got, span2_unit["reference"][:, k]), f"column {k}"


@pytest.mark.parametrize("span2_unit", ["E2M1"], indirect=True)
def test_span2_gemv_matches_the_reference_decode(span2_unit):
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    packed = pack_unit_for_kernel(span2_unit["unit"], span2_unit["forest"], CODE)
    torch.manual_seed(1)
    x = torch.randn(span2_unit["cols"], device="cuda")
    got = gemv_from_packed(x, packed)
    want = span2_unit["reference"] @ x
    assert (got - want).norm() / want.norm() < 1e-5


@pytest.mark.parametrize("span2_unit", ["E2M1"], indirect=True)
def test_span2_planes_weigh_the_wire(span2_unit):
    """The kernel's resident bytes are the wire's: 3.75 b/wt of body (one
    select bit per pair, two label bits per pair, six point bits per code
    over two weights) plus a nibble per sixteen for the scale plane -- 4.0
    b/wt, the same as the on-disk artifact, against the span-1 kernel's 3.5
    body + 0.5 of materialised E4M3 bytes."""
    from tessera.kernel import SELECT_PAD, pack_unit_for_kernel

    rows, cols = span2_unit["rows"], span2_unit["cols"]
    packed = pack_unit_for_kernel(span2_unit["unit"], span2_unit["forest"], CODE)
    steps = rows // 2
    pairs = steps // 2
    assert packed["select"].numel() == cols * (pairs + SELECT_PAD) // 8 + 8
    assert packed["label"].numel() == cols * pairs * 2 // 8
    assert packed["point"].numel() == cols * steps * 6 // 8
    assert packed["nibbles"].numel() == rows * cols // 16 // 2
    body_bits = (packed["select"].numel() - 8) * 8 - cols * SELECT_PAD
    body_bits += (packed["label"].numel() + packed["point"].numel()) * 8
    assert body_bits == rows * cols * 3.75
    assert packed["nibbles"].numel() * 8 == rows * cols * 0.25


def test_span2_luts_compose_to_the_span1_index_table():
    """``index[window, point] == subset[label[window], point]`` -- the fused
    span-1 table and its two span-2 halves are one function."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.kernel import build_span2_luts, build_tuple_index_lut

    grid = tuple_grid(E2M1_GRID, 2)
    forest = build_forest(grid.rate_cap, grid=grid)
    fused = build_tuple_index_lut(forest, CODE).long()
    label_lut, subset_lut = build_span2_luts(forest, CODE)
    points = 1 << (grid.rate_cap - 1)
    windows = torch.arange(1 << (CODE.memory + 1), device="cuda")
    composed = subset_lut.long()[
        (label_lut[windows].long()[:, None] * points + torch.arange(points, device="cuda")[None, :]).reshape(-1)
    ]
    assert torch.equal(fused, composed)


@pytest.mark.parametrize("span2_unit", ["E2M1"], indirect=True)
def test_span2_lane_refuses_what_it_does_not_decode(span2_unit):
    from tessera.errors import GrammarError
    from tessera.kernel import (
        gemv_from_packed, pack_kernel_planes, pack_unit_for_kernel,
    )
    from tessera.manifest import ScalePlaneKind

    packed = pack_unit_for_kernel(span2_unit["unit"], span2_unit["forest"], CODE)
    x = torch.zeros(span2_unit["cols"], device="cuda")
    with pytest.raises(GrammarError, match="derived for VEC=8"):
        gemv_from_packed(x, packed, vec=4)
    with pytest.raises(GrammarError, match="multiple of 16 codes"):
        gemv_from_packed(x, {**packed, "rows": 240})
    with pytest.raises(GrammarError, match="span-1 and span-2"):
        pack_kernel_planes(span2_unit["unit"].body_bits, rate=span2_unit["rate"], span=3)
    with pytest.raises(GrammarError, match="multiple of 16"):
        pack_kernel_planes(span2_unit["unit"].body_bits[:120], rate=span2_unit["rate"], span=2)
    # an S6b plane at span 2 has no kernel path: the kernel reads nibbles
    rows, cols = span2_unit["rows"], span2_unit["cols"]
    torch.manual_seed(6)
    w = torch.randn(rows, cols, device="cuda") * 0.02
    s6b = encode_unit(w, {span2_unit["rate"]: span2_unit["forest"]}, (span2_unit["rate"],) * cols,
                      CODE, completion=0, span=2, scale_plane=ScalePlaneKind.S6B)
    with pytest.raises(GrammarError, match="LUT scale plane"):
        pack_unit_for_kernel(s6b, span2_unit["forest"], CODE)
    span1 = encode_unit(w, {span2_unit["rate"]: span2_unit["forest"]}, (span2_unit["rate"],) * cols,
                        CODE, completion=0, span=1, scale_plane=ScalePlaneKind.LUT)
    with pytest.raises(GrammarError, match="span-2 path"):
        pack_unit_for_kernel(span1, span2_unit["forest"], CODE)


def test_a_wide_alphabet_code_survives_the_window_packer_and_its_gemv():
    """A 16-bit ALPHABET code is not a byte.

    ``PayloadGrid.code_bytes`` is what one stored code weighs, and it is
    derived from the grid's own code space: two bytes for BF16, whose code
    *is* the bf16 bit pattern.  Narrowed to a byte, 0x3f80 (1.0) becomes 0x80
    -- a different, smaller number that selects a different row of the grid's
    value table, with no launch error to say so.  The generic packer must
    hand the kernel the codes the grid declares; the kernel's own load
    converts to int32 either way.
    """
    from tessera.alphabet import BF16_GRID
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel
    from tessera.manifest import BodyKind, ScalePlaneKind

    torch.manual_seed(13)
    rows, cols = 64, 128
    w = (torch.randn(rows, cols, device="cuda") * 0.5).float()
    unit = encode_unit(
        w, BF16_GRID, (7,) * cols, CODE, body=BodyKind.WINDOW, window_bits=9,
        scale_plane=ScalePlaneKind.CHANNEL, scale_refit=1, completion=0,
    )
    assert BF16_GRID.code_bytes == 2
    assert int(unit.window_codes.max()) > 0xFF, "a byte-wide table proves nothing"
    # The code the narrowing loses outright, named in the finding.
    unit.window_codes = unit.window_codes.clone()
    unit.window_codes[0] = 0x3F80
    packed = pack_unit_for_kernel(unit, BF16_GRID, CODE)
    assert torch.equal(
        packed["table"].long().cpu(), unit.window_codes.long().cpu()
    ), "the packer narrowed the ALPHABET plane"
    assert int(packed["table"][0]) == 0x3F80

    reference = reconstruct_unit(unit, BF16_GRID, None).float()
    for k in (0, 1, 17, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = gemv_from_packed(x, packed, lanes=2, split_k=4)
        assert torch.equal(got, reference[:, k]), f"column {k}"


@pytest.mark.parametrize("span2_unit", ["E2M1"], indirect=True)
def test_the_span2_lane_refuses_the_transforms_its_gemv_never_applies(span2_unit):
    """What packing accepts has to be what the GEMV executes.

    The window branch refuses released positions, diagonals and a rotation by
    name, because no kernel on this lane undoes any of them.  The TCQ branch
    dispatches to ``tessera_gemv_tuple_span2``, whose accumulation applies
    none of them either -- and it used to take a unit carrying any of the
    three without a word, serving the transformed quantisation space as if it
    were ``reconstruct_unit(unit) @ x``.

    Release is undefined at arity > 1 (a k-tuple code has no per-position
    code to override), so that leg runs over a scalar grid at span 2.
    """
    from tessera.errors import GrammarError
    from tessera.kernel import pack_unit_for_kernel
    from tessera.manifest import ScalePlaneKind

    forest, rate = span2_unit["forest"], span2_unit["rate"]
    torch.manual_seed(9)
    rows, cols = 32, 64
    w = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    common = dict(completion=0, span=2, scale_plane=ScalePlaneKind.LUT)
    tuple_forests, scalar_forests = {rate: forest}, {3: build_forest(3)}

    def encode(forests, r, **kw):
        return encode_unit(w, forests, (r,) * cols, CODE, **common, **kw)

    # not over-broad: the same shape with none of the three still packs
    pack_unit_for_kernel(encode(tuple_forests, rate), forest, CODE)
    pack_unit_for_kernel(encode(scalar_forests, 3), scalar_forests[3], CODE)

    rotated = encode(tuple_forests, rate, rotation=RotationState.R_IN_ONLY)
    assert rotated.rotation is not RotationState.NONE
    with pytest.raises(GrammarError, match="rotated"):
        pack_unit_for_kernel(rotated, forest, CODE)

    diagonal = encode(tuple_forests, rate, with_diagonals=True)
    assert diagonal.diagonals is not None
    with pytest.raises(GrammarError, match="diagonals"):
        pack_unit_for_kernel(diagonal, forest, CODE)

    released = encode(scalar_forests, 3, released_positions=8)
    assert int(released.release_index.numel()) == 8
    with pytest.raises(GrammarError, match="released positions"):
        pack_unit_for_kernel(released, scalar_forests[3], CODE)


# --- prefill: the same planes, decoded into a tile instead of a vector -------


@pytest.fixture(scope="module")
def prefill_unit():
    from tessera.decode import decode_codes
    from tessera.kernel import build_value_lut, pack_kernel_planes

    torch.manual_seed(11)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).float()
    forest = {3: build_forest(3)}
    unit = encode_unit(weights, forest, (3,) * cols, ConvCode(memory=6),
                       rotation=RotationState.NONE, with_diagonals=False)
    reference = reconstruct_unit(unit, forest, ConvCode(memory=6)).float()
    select, point = pack_kernel_planes(unit.body_bits, 3, 6)
    lut = build_value_lut(forest[3], ConvCode(memory=6), "cuda")
    codes = decode_codes(unit, forest[3], ConvCode(memory=6))
    _packed, e4m3, gs = materialize_nvfp4(
        codes, unit.scale_base, unit.scale_refine, unit.group, unit.half)
    scales = e4m3.reshape(rows, cols // 16).t().contiguous()
    return unit, reference, select, point, lut, scales, gs, rows, cols


def test_prefill_gemm_one_hot_is_bit_exact(prefill_unit):
    """A one-hot row must return a column of the reference decode exactly.

    Nothing is summed, so this isolates the tile decode from the accumulate --
    and unlike the GEMV, this path reduces inside one program and stores, so
    there is no atomic ordering to blur the low bits.
    """
    from tessera.kernel import tessera_gemm
    _u, reference, select, point, lut, scales, gs, rows, cols = prefill_unit
    for column in (0, 1, 7, 63, cols - 1):
        probe = torch.zeros(1, cols, device="cuda")
        probe[0, column] = 1.0
        got = tessera_gemm(probe, select, point, lut, scales, gs, rows, cols)[0]
        assert torch.equal(got, reference[:, column]), f"column {column}"


def test_prefill_gemm_matches_the_reference_decode(prefill_unit):
    """Full GEMM against the materialised weights, over several M.

    M crosses the tile boundary in both directions so a tail-masking error in
    the M dimension cannot hide inside a full tile.
    """
    from tessera.kernel import tessera_gemm
    _u, reference, select, point, lut, scales, gs, rows, cols = prefill_unit
    for m in (1, 15, 64, 129):
        x = torch.randn(m, cols, device="cuda")
        got = tessera_gemm(x, select, point, lut, scales, gs, rows, cols)
        want = x @ reference.t()
        assert torch.allclose(got, want, rtol=2e-5, atol=2e-4), f"M={m}"


def test_prefill_gemm_is_deterministic(prefill_unit):
    """No atomics here, so two runs must agree bit for bit.

    The GEMV cannot make this claim -- its split-K lands through `atomic_add`
    and the low bits depend on arrival order.  The distinction is worth a test
    because it decides which path may be cited in a reproducibility claim.
    """
    from tessera.kernel import tessera_gemm
    _u, _r, select, point, lut, scales, gs, rows, cols = prefill_unit
    x = torch.randn(96, cols, device="cuda")
    first = tessera_gemm(x, select, point, lut, scales, gs, rows, cols)
    second = tessera_gemm(x, select, point, lut, scales, gs, rows, cols)
    assert torch.equal(first, second)


def test_prefill_gemm_refuses_a_mismatched_reduction(prefill_unit):
    """x's inner dimension is the axis the trellis does NOT run down."""
    from tessera.errors import GrammarError
    from tessera.kernel import tessera_gemm
    _u, _r, select, point, lut, scales, gs, rows, cols = prefill_unit
    with pytest.raises(GrammarError, match="reduction runs over"):
        tessera_gemm(torch.randn(8, cols - 1, device="cuda"),
                     select, point, lut, scales, gs, rows, cols)


@pytest.mark.parametrize("tuple_unit", ["E2M1", "free-16"], indirect=True)
def test_the_split_lookup_composes_back_to_the_fused_table(tuple_unit):
    """The serving path splits one 64 KB table into 16 KB shared + 2 KB per
    unit, so that a per-unit grid costs 2 KB of resident lookup instead of
    64 KB -- 37,694 units x 64 KB would be 2.4 GB, spending 1.6% of the body to
    buy back bits the format just saved.

    Two tables that must agree are two tables that can drift, so the fused form
    is now DEFINED as the composition and this pins that definition: if the
    split ever stopped reproducing the fused table exactly, every kernel result
    would move with it and no other test would say why.
    """
    from tessera.kernel import (
        build_anchor_values, build_tuple_index_lut, build_tuple_value_lut,
    )

    forest = tuple_unit["forests"][tuple_unit["rate"]]
    fused = build_tuple_value_lut(forest, CODE)
    index = build_tuple_index_lut(forest, CODE)
    values = build_anchor_values(forest)
    arity = forest.grid.arity

    assert torch.equal(
        values.reshape(-1, arity)[index.long()].reshape(-1), fused)
    # and the split is actually smaller where it counts: the per-unit half
    assert values.numel() * 4 <= fused.numel() * 4 // 16, (
        f"per-unit table {values.numel() * 4} B vs fused {fused.numel() * 4} B")
