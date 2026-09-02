"""The kernel lane over a window body (schema minor 2).

The span-2 lane in ``tests/test_kernel.py`` decodes a convolutional trellis:
a window of select bits, a stored label, a point field, three tables.  A
window body has none of that.  ``state_t = ((state_{t-1} << R) | bits_t) mod
2^L`` from ``state_{-1} = 0`` makes a state *literally* the last ``L`` bits
of the column's stream, so the kernel reads an ``L``-bit field out of the
plane and indexes the unit's own ``2^L`` table.  These tests hold that
kernel to the reader.

The comparison is against ``read_unit_artifact`` -- **bytes**, not the
encoder's tensors -- wherever a unit can be serialised, because the claim
the lane makes is that resident bytes and stored bytes decode to the same
weights.  Where an artifact cannot be built (a hand-made plane at a window
wider than the reference encoder can search) the reference is
``replay_window`` composed with the grid, which is the same grammar written
out in torch.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

import tessera.unit_artifact as unit_artifact_module
from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.decode import replay_window
from tessera.encode import encode_unit, grid_vector_table
from tessera.errors import GrammarError
from tessera.manifest import ArrangementMode, BodyKind, RotationState, ScalePlaneKind
from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, read_unit_artifact
from tessera.wire import scales_from_lut

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel lane is a CUDA path"
)

CODE = ConvCode(memory=6)
K2 = tuple_grid(E2M1_GRID, 2)
WINDOW = BodyKind.WINDOW

#: A three-rate schedule over 512 columns whose root is exactly 6 and whose
#: every 256-column superblock keeps the quota.  ``bresenham_rate_schedule``
#: mixes only the two rates bracketing the root, so a {5, 6, 7} unit -- the
#: mixed case the lane has to decode -- is reachable only as a stored
#: arrangement.
MIXED_567 = ((5,) * 64 + (6,) * 128 + (7,) * 64) * 2
assert sum(MIXED_567) == 6 * 512


@contextlib.contextmanager
def _stored_arrangement():
    """Build the next artifact with ``ArrangementMode.STORED``.

    ``build_unit_artifact`` declares BRESENHAM, and a Bresenham schedule is
    two adjacent rates by construction -- so there is no three-rate artifact
    without this.  STORED is the wire's own answer for an importance-placed
    schedule (``manifest.py``: "importance-placed; the rate vector is on the
    wire") and ``read_unit_artifact`` already reads the vector back off it, so
    the round trip below is the real reader on real bytes, not a shortcut
    around it.
    """
    real = unit_artifact_module.Manifest
    unit_artifact_module.Manifest = lambda **kw: real(
        **{**kw, "arrangement": ArrangementMode.STORED}
    )
    try:
        yield
    finally:
        unit_artifact_module.Manifest = real


# --- the plane: a shift register's stream, column-major, with an L-bit pad ---


def _read_state(plane: torch.Tensor, start: int, window: int) -> int:
    """The ``window``-bit big-endian field at bit ``start``, read a bit at a time.

    Deliberately the slowest possible reader: it shares no arithmetic with
    either the kernel's eight-byte span load or ``pack_window_planes``'s packing, so
    an error in the bit order cannot cancel between them.
    """
    value = 0
    for offset in range(window):
        bit = start + offset
        value = (value << 1) | ((int(plane[bit // 8]) >> (7 - bit % 8)) & 1)
    return value


@pytest.mark.parametrize("window", [10, 16, 20])
def test_the_window_plane_holds_replay_windows_states(window):
    """The plane's ``L``-bit field at ``offset[c] + (t+1)*R`` is ``state_t``.

    This is the whole contract between ``pack_window_planes`` and the kernel,
    tested against ``decode.replay_window`` -- the reader's own closed form --
    over a mixed schedule, at three widths up to the wire's widest.
    """
    from tessera.kernel import pack_window_planes

    torch.manual_seed(window)
    steps, cols = 9, 32
    rates = tuple(min((3, 5, 7)[c % 3], window) for c in range(cols))
    bits = torch.stack([
        torch.randint(0, 1 << r, (steps,), device="cuda") for r in rates
    ], dim=1).to(torch.uint8)
    plane, offsets, rate_t = pack_window_planes(bits, rates, window)
    assert torch.equal(rate_t.cpu(), torch.tensor(rates, dtype=torch.int32))
    host = plane.cpu()
    for col, rate in enumerate(rates):
        want = replay_window(bits[:, col : col + 1], window, rate).reshape(-1)
        base = int(offsets[col])
        assert base % 8 == 0, "a column starts on a byte"
        for step in range(steps):
            got = _read_state(host, base + (step + 1) * rate, window)
            assert got == int(want[step]), f"column {col} step {step}"


def test_the_window_plane_refuses_widths_and_rates_it_cannot_carry():
    from tessera.kernel import pack_window_planes

    bits = torch.zeros(8, 16, dtype=torch.uint8, device="cuda")
    with pytest.raises(GrammarError, match="outside 1..20"):
        pack_window_planes(bits, (3,) * 16, 21)
    with pytest.raises(GrammarError, match="does not fit"):
        pack_window_planes(bits, (9,) * 16, 8)
    with pytest.raises(GrammarError, match="rates for"):
        pack_window_planes(bits, (3,) * 15, 10)


# --- the kernel with no encoder in front of it: any width, hand-made bytes ---


@pytest.mark.parametrize("window", [12, 18, 20])
def test_window_gemv_is_bit_exact_over_a_hand_made_plane(window):
    """A random table and a random stream, decoded two ways.

    The reference encoder searches ``2^L`` states per position, so a unit at
    ``L = 20`` costs minutes to *encode* and seconds to decode -- which would
    price a correctness test at the encoder's rate, not the kernel's.  Nothing
    about the decode needs an encoder: the wire is a table, a stream and a
    scale plane, and this builds all three directly.  It is the only place the
    kernel's window read is exercised past ``L = 16``.
    """
    from tessera.kernel import (
        build_window_values, lut_scale_table, pack_scale_nibbles,
        pack_window_planes, tessera_gemv_window,
    )

    torch.manual_seed(window)
    grid, arity, half = K2, 2, 16
    steps, cols = 96, 128
    rows = steps * arity
    rates = tuple((5, 6, 7)[c % 3] for c in range(cols))
    bits = torch.stack([
        torch.randint(0, 1 << r, (steps,), device="cuda") for r in rates
    ], dim=1).to(torch.uint8)
    table = torch.randint(0, grid.size, (1 << window,), dtype=torch.uint8, device="cuda")
    lut = torch.arange(0x30, 0x40, dtype=torch.uint8, device="cuda")
    refine = torch.randint(0, 16, (rows * cols // half,), dtype=torch.uint8, device="cuda")

    # the reference: replay, table, grid, scale -- ``dequantize``'s own order
    codes = torch.zeros(steps, cols, dtype=torch.long, device="cuda")
    for rate in sorted(set(rates)):
        which = torch.tensor([c for c, r in enumerate(rates) if r == rate], device="cuda")
        codes[:, which] = table[replay_window(bits[:, which], window, rate)].long()
    vectors = grid_vector_table(grid, "cuda")[codes]              # [steps, cols, arity]
    scale = torch.repeat_interleave(
        scales_from_lut(refine, lut, 1.0), half
    ).reshape(rows, cols)
    reference = vectors.permute(0, 2, 1).reshape(rows, cols) * scale

    plane, offsets, rate_t = pack_window_planes(bits, rates, window)
    for k in (0, 1, 2, 17, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = tessera_gemv_window(
            x, plane, offsets, rate_t, table, build_window_values(grid, "cuda"),
            pack_scale_nibbles(refine, rows, cols, half), lut_scale_table(lut, "cuda"),
            1.0, rows, cols, window, arity, half=half, lanes=8, split_k=4,
        )
        assert torch.equal(got, reference[:, k]), f"window {window} column {k}"


# --- encoded units, through the wire, back out of the bytes ----------------


CASES = [
    # (id, grid, rates, window_bits, scale plane, rows, cols)
    ("k2-mixed567-L10-lut", K2, MIXED_567, 10, ScalePlaneKind.LUT, 256, 512),
    ("k2-mixed567-L12-lut", K2, MIXED_567, 12, ScalePlaneKind.LUT, 256, 512),
    ("k2-mixed567-L14-lut", K2, MIXED_567, 14, ScalePlaneKind.LUT, 256, 512),
    ("k2-mixed567-L16-lut", K2, MIXED_567, 16, ScalePlaneKind.LUT, 256, 512),
    ("k2-mixed567-L10-s6b", K2, MIXED_567, 10, ScalePlaneKind.S6B, 256, 512),
    ("k2-mixed567-L14-s6b", K2, MIXED_567, 14, ScalePlaneKind.S6B, 256, 512),
    ("e4m3-r4-L12-lut", E4M3_GRID, (4,) * 512, 12, ScalePlaneKind.LUT, 256, 512),
    ("e4m3-r4-L14-lut", E4M3_GRID, (4,) * 512, 14, ScalePlaneKind.LUT, 256, 512),
    ("e4m3-r4-L16-lut", E4M3_GRID, (4,) * 512, 16, ScalePlaneKind.LUT, 256, 512),
    ("e4m3-r5-L12-lut", E4M3_GRID, (5,) * 512, 12, ScalePlaneKind.LUT, 256, 512),
    ("e4m3-r5-L14-lut", E4M3_GRID, (5,) * 512, 14, ScalePlaneKind.LUT, 256, 512),
    ("e4m3-r5-L16-lut", E4M3_GRID, (5,) * 512, 16, ScalePlaneKind.LUT, 256, 512),
    ("e4m3-r4-L12-s6b", E4M3_GRID, (4,) * 512, 12, ScalePlaneKind.S6B, 256, 512),
    ("e4m3-r5-L14-s6b", E4M3_GRID, (5,) * 512, 14, ScalePlaneKind.S6B, 256, 512),
    # The shipping expert shape, and an odd one: 100 codes is neither a
    # multiple of the lane's VEC nor of a program's LANES*VEC, so both tail
    # masks run; 768 columns is not a power of two.
    ("k2-r7-L12-lut-2048x4096", K2, (7,) * 4096, 12, ScalePlaneKind.LUT, 2048, 4096),
    ("k2-r7-L10-lut-200x768", K2, (7,) * 768, 10, ScalePlaneKind.LUT, 200, 768),
]


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def window_unit(request):
    """One encoded window unit, its artifact, and the weights read from bytes."""
    _name, grid, rates, window_bits, plane, rows, cols = request.param
    torch.manual_seed(rows + cols + window_bits)
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    weights[: rows // 8] *= 4.0                      # several scale binades
    unit = encode_unit(
        weights, grid, rates, CODE, body=WINDOW, window_bits=window_bits,
        scale_plane=plane, scale_refit=1, rotation=RotationState.NONE,
        with_diagonals=False,
    )
    q256 = sum(rates) * 256 // len(rates)
    assert sum(rates) * 256 == q256 * len(rates), "the schedule must have an exact root"
    canonical = len(set(rates)) <= 2
    with contextlib.nullcontext() if canonical else _stored_arrangement():
        _manifest, _region, blob = build_unit_artifact(unit, "unit0", grid, q256, CODE)
    assert blob[10] == 2, "schema minor 2"
    return {
        "unit": unit, "grid": grid, "rates": rates, "rows": rows, "cols": cols,
        "window_bits": window_bits, "plane": plane, "blob": blob,
        "reference": read_unit_artifact(blob, device="cuda"),
    }


def test_the_window_kernel_decodes_the_bytes_the_reader_decodes(window_unit):
    """One-hot columns: the kernel's weights equal the artifact's, exactly.

    ``x = e_k`` sums nothing, so this is the decode itself and ``torch.equal``
    is the right relationship -- the split-K atomic that blurs the low bits of
    a real GEMV never fires.  The reference is ``read_unit_artifact``, so what
    is compared is the kernel against **the bytes**, with the encoder's
    tensors out of the loop entirely.
    """
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    cols, reference = window_unit["cols"], window_unit["reference"]
    packed = pack_unit_for_kernel(window_unit["unit"], window_unit["grid"], CODE)
    assert packed["kind"] == "window"
    probes = [0, 1, 7, 8, 15, 16, 17, 33, cols // 2, cols - 1]
    for k in dict.fromkeys(probes):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        got = gemv_from_packed(x, packed, lanes=8, split_k=4)
        assert torch.equal(got, reference[:, k]), f"column {k}"


def test_the_window_gemv_matches_the_reference_decode(window_unit):
    """The lane's tolerance contract, as ``test_kernel.py`` states it."""
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    packed = pack_unit_for_kernel(window_unit["unit"], window_unit["grid"], CODE)
    torch.manual_seed(1)
    x = torch.randn(window_unit["cols"], device="cuda")
    got = gemv_from_packed(x, packed)
    want = window_unit["reference"] @ x
    assert (got - want).norm() / want.norm() < 1e-5


def test_the_window_planes_weigh_the_wire(window_unit):
    """Resident bytes are the wire's: ``R`` bits per code and a scale nibble.

    The window body carries no label plane and no separated point plane, so at
    the E2M1x2 cap it is 3.5 b/wt of body against the span-2 trellis's 3.75 --
    the two are *not* the same bytes and the bench says so.  The only resident
    bytes the wire does not charge are the per-column ``L``-bit pad, which
    stands in for ``state_{-1} = 0`` exactly as ``SELECT_PAD`` does for the
    trellis lane, and eight bytes of slack for the last eight-byte span read.
    """
    from tessera.kernel import pack_unit_for_kernel

    rows, cols = window_unit["rows"], window_unit["cols"]
    rates, window = window_unit["rates"], window_unit["window_bits"]
    steps = rows // window_unit["grid"].arity
    packed = pack_unit_for_kernel(window_unit["unit"], window_unit["grid"], CODE)
    assert packed["plane"].numel() == sum(
        -(-(window + steps * r) // 8) for r in rates
    ) + 8
    body_bits = steps * sum(rates)
    pad_bits = window * cols
    assert packed["plane"].numel() * 8 >= body_bits + pad_bits
    assert (packed["plane"].numel() - 8) * 8 - pad_bits - body_bits < 8 * cols
    assert packed["table"].numel() == 1 << window          # the ALPHABET plane
    if window_unit["plane"] is ScalePlaneKind.LUT:
        assert packed["scale_plane"].numel() * 8 == rows * cols * 0.25
    else:
        assert packed["scale_plane"].numel() * 8 == rows * cols * 0.5


def test_the_window_lane_refuses_what_it_does_not_decode():
    """Three post-decode transforms the reader applies and no GEMV here does.

    Each of them would otherwise return weights that are wrong by a rank-1
    factor, an orthogonal transform, or a handful of released positions --
    all plausible, none detectable without the comparison this test makes
    impossible to skip.
    """
    from tessera.kernel import pack_unit_for_kernel, tessera_gemv_window

    torch.manual_seed(9)
    rows, cols = 128, 512
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    common = dict(body=WINDOW, window_bits=9, scale_plane=ScalePlaneKind.LUT,
                  scale_refit=0)
    # release is arity-1 only, so it is exercised on E4M3
    released = encode_unit(weights, E4M3_GRID, (7,) * cols, CODE,
                           released_positions=8, **common)
    with pytest.raises(GrammarError, match="released positions"):
        pack_unit_for_kernel(released, E4M3_GRID, CODE)
    diagonal = encode_unit(weights, K2, (7,) * cols, CODE, with_diagonals=True, **common)
    with pytest.raises(GrammarError, match="diagonals"):
        pack_unit_for_kernel(diagonal, K2, CODE)
    plain = encode_unit(weights, K2, (7,) * cols, CODE, **common)
    packed = pack_unit_for_kernel(plain, K2, CODE)
    plain.rotation = RotationState.R_IN_ONLY
    with pytest.raises(GrammarError, match="rotated"):
        pack_unit_for_kernel(plain, K2, CODE)

    x = torch.zeros(cols, device="cuda")
    args = (packed["plane"], packed["offsets"], packed["rates"], packed["table"],
            packed["values"], packed["scale_plane"], packed["scale_table"],
            packed["global_scale"])
    with pytest.raises(GrammarError, match="window table holds"):
        tessera_gemv_window(x, *args, rows, cols, 10, 2)
    with pytest.raises(GrammarError, match="offsets and"):
        tessera_gemv_window(x, *args, rows, cols - 16, 9, 2)
    with pytest.raises(GrammarError, match="whole codes"):
        tessera_gemv_window(x, *args, rows - 1, cols, 9, 2)
    with pytest.raises(GrammarError, match="whole number"):
        tessera_gemv_window(x[:-1], *args, rows, cols - 1, 9, 2)


def test_the_span2_trellis_lane_still_dispatches_to_itself():
    """A TCQ unit is unchanged by the window branch: same key, same kernel."""
    from tessera.alphabet import build_forest
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel
    from tessera.decode import reconstruct_unit

    torch.manual_seed(12)
    rows, cols = 256, 512
    weights = (torch.randn(rows, cols, device="cuda") * 0.02).contiguous()
    forest = build_forest(K2.rate_cap, grid=K2)
    unit = encode_unit(weights, {K2.rate_cap: forest}, (K2.rate_cap,) * cols, CODE,
                       completion=0, span=2, scale_plane=ScalePlaneKind.LUT)
    packed = pack_unit_for_kernel(unit, forest, CODE)
    assert packed["kind"] == "span2"
    reference = reconstruct_unit(unit, {K2.rate_cap: forest}, CODE, completion=0).float()
    for k in (0, 5, cols - 1):
        x = torch.zeros(cols, device="cuda")
        x[k] = 1.0
        assert torch.equal(gemv_from_packed(x, packed, lanes=8, split_k=4), reference[:, k])
