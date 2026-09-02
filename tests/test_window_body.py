"""Schema minor 2: the window body.

One wire change behind one measurement
(``docs/measurements/tessera-window-body-2026-09-02.md``): on the hardware
tile, a trellis whose reconstruction at a position is a table lookup on the
last L bits of the column's stream (Tseng et al.'s bitshift trellis) beats
the shaped convolutional trellis by 1.2-1.3x below the E2M1x2 cap and on
E4M3 under a per-channel plane, and at L=14 beats EXL3 K4 in output space
at 4.0 bpp.  These tests hold the implementation to what the measurement
relied on:

  * ``viterbi_window`` is *exact* -- its summed squared error equals the
    exhaustive search over every bit string, with and without weights, at
    arity 1 and 2 -- and it starts from state 0, as the decoder assumes;
  * ``replay_window`` lands on the encoder's states from the bits alone;
  * the seam round-trips: the table rides the ALPHABET plane, DESCENDANT and
    COMPLETION are empty, the header says minor 2, the accountant agrees
    with the bytes to the bit, and the reader needs nothing but bytes;
  * every TCQ artifact is untouched: same bytes, same minor, same profile id;
  * the profile id binds the body kind and the width, the reader fails
    closed on a manifest that disagrees with it or a table outside the grid,
    and the kernel lane decodes the body at the wire's own bytes
    (``tests/test_kernel_window.py``);
  * the exporter records the body and replays a config at its own meaning.
"""
import itertools
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import E2M1_GRID, E4M3_GRID, build_forest, tuple_grid
from tessera.calculator import terminal_rate
from tessera.container import parse, serialize
from tessera.decode import decode_codes_mixed, reconstruct_unit, replay_window
from tessera.encode import encode_unit, viterbi_window, window_table
from tessera.errors import GrammarError, ManifestError, TesseraError
from tessera.export import (
    DEFAULT_BODY,
    DEFAULT_WINDOW_BITS,
    encode_linear,
    encode_settings_from_config,
    export_checkpoint,
    read_checkpoint_config,
)
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import WINDOW_BITS_MAX, BodyKind, ScalePlaneKind
from tessera.planes import CANONICAL_PLANE_ORDER, PlaneKind
from tessera.trellis import ConvCode
from tessera.unit_artifact import (
    build_unit_artifact,
    encoder_profile_id,
    read_unit_artifact,
)

CODE = ConvCode(memory=6)
K2 = tuple_grid(E2M1_GRID, 2)
WINDOW = BodyKind.WINDOW


def _weights(rows=64, cols=512, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, cols) * 0.02


def _elements(blob: bytes, kind: PlaneKind) -> int:
    return parse(blob).terminal.plane_elements[CANONICAL_PLANE_ORDER.index(kind)]


# ------------------------------------------------------------ the trellis


def _exhaustive(targets, vectors, window, rate, weights=None):
    """Every bit string of a column, from state 0: the oracle the Viterbi must match."""
    steps = targets.shape[0] // vectors.shape[1]
    arity = vectors.shape[1]
    mask = (1 << window) - 1
    total = 0.0
    for col in range(targets.shape[1]):
        best = None
        for bits in itertools.product(range(1 << rate), repeat=steps):
            state, cost = 0, 0.0
            for step, new in enumerate(bits):
                state = ((state << rate) | new) & mask
                for k in range(arity):
                    row = step * arity + k
                    err = float(targets[row, col] - vectors[state, k]) ** 2
                    if weights is not None:
                        err *= float(weights[row, col])
                    cost += err
            best = cost if best is None else min(best, cost)
        total += best
    return total


@pytest.mark.parametrize("rate,window,arity", [(1, 3, 1), (2, 4, 1), (2, 5, 2), (3, 5, 1), (3, 3, 1)])
def test_window_viterbi_is_the_exhaustive_search(rate, window, arity):
    torch.manual_seed(rate * 10 + window + arity)
    steps, cols = 5, 3
    vectors = torch.randn(1 << window, arity)
    targets = torch.randn(steps * arity, cols)
    states, sse = viterbi_window(targets, vectors, window, rate)
    assert states.shape == (steps, cols)
    assert sse == pytest.approx(_exhaustive(targets, vectors, window, rate), rel=1e-5)
    # the states are a legal path from state 0 and cost exactly ``sse``
    bits = states & ((1 << rate) - 1)
    assert torch.equal(replay_window(bits, window, rate), states)
    hat = vectors[states].permute(0, 2, 1).reshape(steps * arity, cols)
    assert float(((hat - targets) ** 2).sum()) == pytest.approx(sse, rel=1e-5)


def test_window_viterbi_honours_per_position_weights():
    torch.manual_seed(3)
    vectors = torch.randn(16, 1)
    targets = torch.randn(4, 2)
    weights = torch.rand(4, 2) + 0.1
    _, sse = viterbi_window(targets, vectors, 4, 2, weights=weights)
    assert sse == pytest.approx(_exhaustive(targets, vectors, 4, 2, weights), rel=1e-5)


def test_window_viterbi_is_chunk_invariant():
    torch.manual_seed(4)
    vectors = torch.randn(256, 1)
    targets = torch.randn(32, 40)
    a, sa = viterbi_window(targets, vectors, 8, 3, chunk=7)
    b, sb = viterbi_window(targets, vectors, 8, 3, chunk=512)
    assert torch.equal(a, b) and sa == pytest.approx(sb)


def test_window_viterbi_refuses_a_rate_the_window_cannot_hold():
    with pytest.raises(GrammarError, match="does not fit"):
        viterbi_window(torch.zeros(4, 1), torch.zeros(8, 1), 3, 4)
    with pytest.raises(GrammarError, match="states"):
        viterbi_window(torch.zeros(4, 1), torch.zeros(8, 1), 4, 2)


def test_replay_window_is_the_shift_register():
    torch.manual_seed(5)
    for rate, window in ((1, 6), (3, 7), (4, 4), (7, 9)):
        bits = torch.randint(0, 1 << rate, (12, 3))
        states = replay_window(bits, window, rate)
        mask = (1 << window) - 1
        for col in range(3):
            state = 0
            for step in range(12):
                state = ((state << rate) | int(bits[step, col])) & mask
                assert int(states[step, col]) == state


# --------------------------------------------------------------- the table


def test_window_table_is_deterministic_snapped_and_cached():
    a = window_table(K2, 10, seed=1)
    b = window_table(K2, 10, seed=1)
    assert torch.equal(a, b) and a.numel() == 1024 and a.dtype == torch.uint8
    assert int(a.max()) < K2.size
    assert not torch.equal(a, window_table(K2, 10, seed=2))
    assert not torch.equal(a, window_table(K2, 10, seed=1, sigma=2.0))
    a[0] = 255                                   # a caller's copy, never the cache's
    assert int(window_table(K2, 10, seed=1)[0]) < K2.size
    # E4M3's duplicate slots are never chosen: ties fall to the lower, legal byte
    e = window_table(E4M3_GRID, 12)
    assert not bool(((e == 0x7F) | (e == 0xFF) | (e == 0x80)).any())
    with pytest.raises(GrammarError):
        window_table(K2, WINDOW_BITS_MAX + 1)
    # narrower than a half: order statistics of one half's worth, no crash
    for bits in (1, 2, 3):
        small = window_table(E2M1_GRID, bits)
        assert small.numel() == 1 << bits and int(small.max()) < E2M1_GRID.size
        assert window_table(K2, bits, sigma=1.0).numel() == 1 << bits
    with pytest.raises(GrammarError, match="sigma"):
        window_table(K2, 8, sigma=0.0)


# --------------------------------------------------------------- the seam


def test_replay_lands_on_the_encoders_states_over_mixed_rates():
    w = _weights()
    rates = bresenham_rate_schedule(root_from_q256(int(4.5 * 256)), 512, cap=7)
    assert set(rates) == {4, 5}
    unit = encode_unit(w, E4M3_GRID, rates, CODE, body=WINDOW, window_bits=8,
                       scale_plane=ScalePlaneKind.LUT, scale_refit=1)
    assert unit.body is WINDOW and unit.window_bits == 8 and unit.span == 1
    assert unit.completion_limit == 0 and not unit.completion_bits.any()
    for rate in (4, 5):
        which = torch.tensor([j for j, r in enumerate(rates) if r == rate])
        assert torch.equal(replay_window(unit.body_bits[:, which], 8, rate), unit.anchors[:, which])
    assert torch.equal(unit.codes, unit.window_codes.long()[unit.anchors])


@pytest.mark.parametrize(
    "grid,q256,window,plane,diagonals",
    [
        (K2, 7 * 256, 9, ScalePlaneKind.LUT, False),
        (K2, 6 * 256, 10, ScalePlaneKind.LUT, True),
        (K2, 6 * 256 + 128, 10, ScalePlaneKind.S6B, False),
        (E4M3_GRID, 4 * 256, 8, ScalePlaneKind.LUT, False),
        (E4M3_GRID, int(4.5 * 256), 7, ScalePlaneKind.S6B, True),
        (E2M1_GRID, 3 * 256, 6, ScalePlaneKind.LUT, False),
    ],
)
def test_wire_round_trip_of_a_window_body(grid, q256, window, plane, diagonals):
    w = _weights()
    rates = bresenham_rate_schedule(root_from_q256(q256), 512, cap=grid.rate_cap)
    unit = encode_unit(w, grid, rates, CODE, body=WINDOW, window_bits=window,
                       scale_plane=plane, with_diagonals=diagonals, scale_refit=1)
    manifest, region, blob = build_unit_artifact(unit, "unit0", grid, q256, CODE)
    assert torch.equal(read_unit_artifact(blob), reconstruct_unit(unit, grid, None))
    assert blob[10] == 2, "schema minor 2"
    art = parse(blob)
    assert art.manifest.body is WINDOW and art.manifest.window_bits == window
    assert art.manifest.span == 1 and art.manifest.scale_plane.kind is plane
    assert art.terminal.exact_bytes == len(region)
    assert _elements(blob, PlaneKind.ALPHABET) == 1 << window
    assert _elements(blob, PlaneKind.DESCENDANT) == 0
    assert _elements(blob, PlaneKind.COMPLETION) == 0
    assert region[: 1 << window] == bytes(unit.window_codes.tolist())
    # the accountant prices the table exactly as the wire charges it
    rows, cols = w.shape
    predicted = terminal_rate(
        q256, rows, cols, with_scale_base=plane is ScalePlaneKind.S6B,
        with_scale_refine=True, with_diagonals=diagonals, cap=grid.rate_cap,
        arity=grid.arity, window_bits=window,
    )
    assert predicted == art.terminal.exact_bpp
    assert art.terminal.exact_bpp - Fraction(8 << window, w.numel()) == Fraction(
        sum(rates) * (rows // grid.arity) + (rows * cols // 16) * (4 + (4 if plane is ScalePlaneKind.S6B else 0))
        + (16 * (rows + cols) if diagonals else 0), w.numel()
    )


def test_a_window_body_refuses_what_it_has_no_meaning_for():
    w = _weights()
    with pytest.raises(GrammarError, match="span"):
        encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=9, span=2)
    with pytest.raises(GrammarError, match="completion"):
        encode_unit(w, K2, (6,) * 512, CODE, body=WINDOW, window_bits=9, completion=1)
    with pytest.raises(GrammarError, match="cannot hold"):
        encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=6)
    with pytest.raises(GrammarError, match="only meaningful"):
        encode_unit(w, {7: build_forest(7, grid=K2)}, (7,) * 512, CODE, window_bits=9)
    with pytest.raises(GrammarError, match="forests"):
        encode_unit(w, K2, (7,) * 512, CODE)                 # TCQ needs its forests
    with pytest.raises(GrammarError, match="table"):
        unit = encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=9)
        unit.window_codes = None
        build_unit_artifact(unit, "unit0", K2, 7 * 256, CODE)


# ----------------------------------------------------- what did not change


def test_tcq_artifacts_are_byte_identical_and_keep_their_minor():
    w = _weights()
    unit = encode_unit(w, {7: build_forest(7, grid=K2)}, (7,) * 512, CODE)
    assert unit.body is BodyKind.TCQ and unit.window_bits == 0 and unit.window_codes is None
    manifest, _, blob = build_unit_artifact(unit, "unit0", {7: build_forest(7, grid=K2)}, 7 * 256, CODE)
    assert blob[10] == 0 and manifest.schema_minor == 0
    assert manifest.body is BodyKind.TCQ and manifest.window_bits == 0
    assert manifest.encode() == manifest.encode(0) == manifest.encode(1)[: len(manifest.encode(0))]
    span2 = encode_unit(w, {7: build_forest(7, grid=K2)}, (7,) * 512, CODE, span=2,
                        scale_plane=ScalePlaneKind.LUT)
    m2, _, blob2 = build_unit_artifact(span2, "unit0", {7: build_forest(7, grid=K2)}, 7 * 256, CODE)
    assert blob2[10] == 1 and m2.schema_minor == 1
    # a minor-2 manifest cannot be squeezed into minor 1
    win = encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=9)
    mw, _, _ = build_unit_artifact(win, "unit0", K2, 7 * 256, CODE)
    assert mw.schema_minor == 2
    with pytest.raises(ManifestError, match="needs minor 2"):
        mw.encode(1)
    with pytest.raises(ManifestError, match="only meaningful"):
        manifest.__class__(**{**manifest.__dict__, "window_bits": 9})
    with pytest.raises(ManifestError, match="span must be 1"):
        mw.__class__(**{**mw.__dict__, "span": 2})
    with pytest.raises(ManifestError, match="cannot hold"):
        mw.__class__(**{**mw.__dict__, "window_bits": 6})


def test_profile_id_binds_the_body_and_its_width_and_nothing_else():
    rates = (7,) * 8
    base = encoder_profile_id(CODE, rates, K2)
    assert encoder_profile_id(CODE, rates, K2, 1, ScalePlaneKind.S6B, BodyKind.TCQ, 0) == base
    w9 = encoder_profile_id(CODE, rates, K2, 1, ScalePlaneKind.S6B, WINDOW, 9)
    w10 = encoder_profile_id(CODE, rates, K2, 1, ScalePlaneKind.S6B, WINDOW, 10)
    assert len({base, w9, w10}) == 3
    # no convolutional code is involved, so none is bound
    assert encoder_profile_id(None, rates, K2, 1, ScalePlaneKind.S6B, WINDOW, 9) == w9
    assert encoder_profile_id(ConvCode(3), rates, K2, 1, ScalePlaneKind.S6B, WINDOW, 9) == w9
    assert encoder_profile_id(None, rates, K2, 1, ScalePlaneKind.LUT, WINDOW, 9) != w9
    with pytest.raises(GrammarError, match="span 1"):
        encoder_profile_id(None, rates, K2, 2, ScalePlaneKind.S6B, WINDOW, 9)
    with pytest.raises(GrammarError, match="convolutional"):
        encoder_profile_id(None, rates, K2)


def test_the_reader_fails_closed_on_a_manifest_that_disagrees_with_the_profile():
    w = _weights()
    unit = encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=9)
    manifest, region, _ = build_unit_artifact(unit, "unit0", K2, 7 * 256, CODE)
    # the same bytes, a manifest claiming a different width: refused, never
    # decoded against a table that is not there
    lying = manifest.__class__(**{**manifest.__dict__, "window_bits": 10})
    with pytest.raises(TesseraError):
        read_unit_artifact(serialize(lying, region))
    # a manifest claiming a TCQ body over a window profile: no code matches
    tcq = manifest.__class__(**{**manifest.__dict__, "body": BodyKind.TCQ, "window_bits": 0})
    with pytest.raises(TesseraError):
        read_unit_artifact(serialize(tcq, region))


def test_a_table_outside_the_grid_is_refused_before_it_indexes_anything():
    w = _weights()
    unit = encode_unit(w, E2M1_GRID, (3,) * 512, CODE, body=WINDOW, window_bits=6)
    unit.window_codes = unit.window_codes.clone()
    unit.window_codes[5] = 16                    # E2M1 has sixteen codes: 0..15
    with pytest.raises(GrammarError, match="outside"):
        decode_codes_mixed(unit, E2M1_GRID, None)
    with pytest.raises(GrammarError, match="outside"):
        build_unit_artifact(unit, "unit0", E2M1_GRID, 3 * 256, CODE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel lane is a CUDA path")
def test_the_kernel_lane_decodes_a_window_body():
    """The lane packs a window body and decodes it to the reader's weights.

    It refused one until the shift-register GEMV landed; the decode itself
    lives in ``tests/test_kernel_window.py``, which is where the widths, the
    grids, the scale planes and the shapes are swept.  This is the seam
    check: a window unit reaches the kernel lane at all, and it accepts a
    bare grid where a TCQ unit hands it a forest.
    """
    from tessera.kernel import gemv_from_packed, pack_unit_for_kernel

    w = _weights(rows=256, cols=512).cuda()
    unit = encode_unit(w, K2, (7,) * 512, CODE, body=WINDOW, window_bits=9,
                       scale_plane=ScalePlaneKind.LUT, scale_refit=0)
    packed = pack_unit_for_kernel(unit, K2, CODE)
    assert packed["kind"] == "window" and packed["window_bits"] == 9
    reference = reconstruct_unit(unit, K2, None).float()
    x = torch.zeros(512, device="cuda")
    x[3] = 1.0
    assert torch.equal(gemv_from_packed(x, packed, lanes=8, split_k=4), reference[:, 3])


# ---------------------------------------------------------------- exporter


def test_the_exporter_default_is_still_the_tcq_body():
    """The window body is measured better below the cap and on E4M3 but has
    no kernel decode and an O(2^L) reference encoder; it flips when both are
    in, not before."""
    assert DEFAULT_BODY is BodyKind.TCQ and DEFAULT_WINDOW_BITS == 0


def test_encode_linear_and_the_config_carry_the_window_body(tmp_path):
    w = _weights().bfloat16()
    exported = encode_linear(w, grid=K2, q256=6 * 128, name="w", body=WINDOW, window_bits=10,
                             window_seed=7, window_sigma=2.5)
    assert exported.blob[10] == 2
    art = parse(exported.blob)
    assert art.manifest.body is WINDOW and art.manifest.window_bits == 10
    export_checkpoint({"w": w}, {"w": 6 * 128}, tmp_path, grid=K2, body=WINDOW,
                      window_bits=10, window_seed=7, window_sigma=2.5)
    config = read_checkpoint_config(tmp_path)
    assert config["body"] == {"kind": "window", "window_bits": 10, "seed": 7, "sigma": 2.5}
    settings = encode_settings_from_config(config)
    assert (settings["body"], settings["window_bits"], settings["window_seed"],
            settings["window_sigma"]) == (WINDOW, 10, 7, 2.5)
    # a config written before the field existed (and before the per-rung
    # recipe table that now carries it too) means the TCQ body
    legacy = {k: v for k, v in config.items() if k not in ("body", "wire")}
    s = encode_settings_from_config(legacy)
    assert (s["body"], s["window_bits"], s["window_seed"], s["window_sigma"]) == (BodyKind.TCQ, 0, 0, None)
    with pytest.raises(GrammarError, match="body kind"):
        encode_settings_from_config({**config, "body": {"kind": "hash"}})
    with pytest.raises(GrammarError, match="completion"):
        encode_linear(w, grid=K2, q256=6 * 128, body=WINDOW, window_bits=10, completion=1)
    # today's default writes a TCQ body, and says so
    export_checkpoint({"w": w}, {"w": 6 * 128}, tmp_path / "tcq", grid=K2)
    assert read_checkpoint_config(tmp_path / "tcq")["body"]["kind"] == "tcq"


# ----------------------------------------------------------------- quality


def test_the_window_body_beats_the_trellis_below_the_cap():
    """The measurement the wire exists for, at test scale: E2M1x2 at rate 6
    (3.0 body bits per weight), a 2^10 table against the span-1 coset
    trellis with the same plane and refits.  Measured 1.19x at L=10 on a
    256x1024 Gaussian; the bar here is loose because the matrix is small."""
    w = _weights(rows=128, cols=512, seed=1)
    rates = (6,) * 512
    forests = {6: build_forest(6, grid=K2)}
    tcq = encode_unit(w, forests, rates, CODE, scale_plane=ScalePlaneKind.LUT,
                      scale_refit=2, completion=0, trellis_weighting="scale")
    win = encode_unit(w, K2, rates, CODE, scale_plane=ScalePlaneKind.LUT, scale_refit=2,
                      body=WINDOW, window_bits=10, trellis_weighting="scale")
    err = lambda unit, f: float(((reconstruct_unit(unit, f, CODE if f is forests else None) - w) ** 2).sum())
    assert err(win, K2) < 0.92 * err(tcq, forests)
