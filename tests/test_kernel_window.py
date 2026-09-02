"""The fused window decoder: byte-identity first, then the GEMV's tolerance.

The contract this file holds the kernel to is the one the FP8 route needs:
``kernel_window.decode_fp8_tile`` must produce the *same bytes* as
``decode.materialize_fp8`` -- not close, identical -- because the serving lane
already refuses a module whose in-forward decoder disagrees with the reference
by one byte, and because those bytes are what a stock ``float-quantized``
checkpoint carries.  The GEMV is held to a derived fp32 tolerance instead,
since split-K sums in a different order.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E4M3_GRID                                  # noqa: E402
from tessera.decode import materialize_fp8                             # noqa: E402
from tessera.errors import GrammarError                                # noqa: E402
from tessera.export import E4M3_RECIPE, encode_linear_planes           # noqa: E402
from tessera.fused import parse_fused                                  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact                  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

#: The reach checkpoint of `docs/measurements/tessera-dense-reach-fix-2026-09-02.md`:
#: 196 units, 112 modules, the E4M3 wire at 4.07 bpp, with the materialised
#: stock twin beside it.
REACH = Path("/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook")
TWIN = Path("/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-stock-twin")
checkpoint = pytest.mark.skipif(
    not (REACH / "model.safetensors").exists(),
    reason="the reach checkpoint is not on this box",
)


def _kw():
    from tessera import kernel_window

    return kernel_window


def _units(limit=None):
    """``(module, role, ParsedUnit)`` for the reach checkpoint's units."""
    from safetensors import safe_open

    with safe_open(str(REACH / "model.safetensors"), framework="pt") as handle:
        keys = sorted(k for k in handle.keys() if k.endswith(".wire_bytes"))
        for key in keys[:limit]:
            blob = bytes(handle.get_tensor(key).numpy().tobytes())
            for member in parse_fused(blob):
                yield key[: -len(".wire_bytes")], member.name, parse_unit_artifact(
                    member.blob, device="cuda"
                )


def _reference_codes(body_bits, rates, window_bits, table, initial):
    """The window body's codes by its **definition**, one step at a time.

    ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L`` from ``state_{-1} =
    initial``.  ``decode.replay_window`` is the closed form of this with the
    start pinned at zero; this walk is what an initial-state shard has to
    agree with, so the nonzero case is checked against the definition rather
    than against another closed form that could share its mistake.
    """
    rows, cols = body_bits.shape
    mask = (1 << window_bits) - 1
    state = initial.to(torch.int64).clone()
    out = torch.zeros(rows, cols, dtype=torch.uint8, device=body_bits.device)
    rate = torch.tensor(rates, dtype=torch.int64, device=body_bits.device)
    for t in range(rows):
        state = ((state << rate) | body_bits[t].to(torch.int64)) & mask
        out[t] = table[state]
    return out


# --- the tables -------------------------------------------------------------


def test_every_legal_e4m3_byte_is_exact_in_fp16():
    """The value table is fp16, and that is lossless for this grid.

    Three mantissa bits over ``2**-9 .. 448`` sits inside fp16's normals, so
    the GEMV's 32 KB table reconstructs exactly what an FP8 GEMM over the
    decoded tile would see.  Asserted, not argued.
    """
    byte = torch.arange(256, dtype=torch.uint8)
    value = byte.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(value)
    assert int(finite.sum()) == 254
    assert torch.equal(value[finite].half().float(), value[finite])


@cuda
def test_code_table_is_the_reference_byte_map():
    """``window_code_table`` composes the two lookups ``materialize_fp8`` does."""
    kw = _kw()
    codes = torch.arange(1 << 8, dtype=torch.uint8, device="cuda") % 256
    table = kw.window_code_table(codes, E4M3_GRID, "cuda")
    native = torch.tensor(E4M3_GRID.native, dtype=torch.uint8, device="cuda")
    assert torch.equal(table, native[codes.long()])
    assert torch.equal(
        kw.window_value_table(table).float(), table.view(torch.float8_e4m3fn).float()
    )


# --- byte identity on the shipped checkpoint --------------------------------


@cuda
@checkpoint
def test_every_reach_unit_decodes_byte_identically():
    """All 196 units: the fused tile equals ``materialize_fp8``, byte for byte,
    and the row scale equals the reference's fp32 expression exactly."""
    kw = _kw()
    seen = 0
    for _module, _role, parsed in _units():
        reference, scale = materialize_fp8(parsed.unit, parsed.forests or parsed.grid,
                                           parsed.code)
        prepared = kw.prepare_from_parsed(parsed, device="cuda")
        assert torch.equal(prepared.decode(), reference.cuda()), _role
        assert torch.equal(prepared.row_scale, scale.cuda().float()), _role
        seen += 1
    assert seen == 196


@cuda
@checkpoint
@pytest.mark.skipif(not (TWIN / "model.safetensors").exists(), reason="no stock twin")
def test_reach_units_match_the_stock_twin_tensors():
    """The fused tile is the *materialised checkpoint's* bytes.

    ``materialize_fp8`` is the decoder; the twin is what was written to disk
    from it and what vanilla vLLM loads.  Both, because an agreement with the
    decoder alone would not catch a checkpoint written from something else.
    """
    from safetensors import safe_open

    kw = _kw()
    with safe_open(str(TWIN / "model.safetensors"), framework="pt") as twin:
        for module, role, parsed in _units(limit=12):
            name = module.rsplit(".", 1)[0] + "." + role
            weight = twin.get_tensor(name + ".weight")
            scale = twin.get_tensor(name + ".weight_scale")
            prepared = kw.prepare_from_parsed(parsed, device="cuda")
            assert torch.equal(
                prepared.decode().cpu().view(torch.float8_e4m3fn), weight
            ), name
            assert torch.equal(prepared.row_scale.cpu(), scale.reshape(-1).float()), name


@cuda
@checkpoint
@pytest.mark.skipif(not (TWIN / "model.safetensors").exists(), reason="no stock twin")
def test_a_fused_module_stacks_into_the_twin_tensor():
    """The seam W2 calls: ``window_module_decode`` over a fused module's roles.

    A fused ``gate_up_proj`` is two units on disk and one tensor in the stock
    twin, so this is the only place the STACKING is checked -- a helper that
    decoded every unit right and concatenated them backwards passes every other
    test here and serves garbage.  ``window_module_linear`` is checked on the
    same module at both sides of the batch cap against the stacked tile; it
    returns bf16 (the lane's activation dtype), so its bar is bf16 rounding on
    the small-M side and the E4M3 activation quantisation on the large-M side.
    The fp32 GEMV itself is held to 2e-6 per unit elsewhere.
    """
    from safetensors import safe_open

    kw = _kw()
    modules = {}
    for module, role, parsed in _units():
        modules.setdefault(module, []).append((role, parsed))
        if len(modules) > 24:
            break
    fused = [(m, r) for m, r in modules.items() if len(r) > 1]
    assert fused, "the reach checkpoint carries fused modules"
    module, roles = fused[0]
    name = module.rsplit(".", 1)[0]
    with safe_open(str(TWIN / "model.safetensors"), framework="pt") as twin:
        want = torch.cat(
            [twin.get_tensor(f"{name}.{role}.weight") for role, _p in roles], 0
        )
        want_scale = torch.cat(
            [twin.get_tensor(f"{name}.{role}.weight_scale").reshape(-1) for role, _p in roles]
        )
    units = [kw.prepare_from_parsed(parsed, device="cuda") for _role, parsed in roles]
    got = kw.window_module_decode(units)
    assert torch.equal(got.cpu().view(torch.float8_e4m3fn), want), name
    assert torch.equal(kw.window_module_row_scale(units).cpu(), want_scale.float())

    tile = got.view(torch.float8_e4m3fn).float() * kw.window_module_row_scale(units)[:, None]
    for m in (1, kw.GEMV_MAX_M + 1):
        x = torch.randn(m, units[0].cols, device="cuda", dtype=torch.bfloat16)
        out = kw.window_module_linear(x, units)
        ref = x.float() @ tile.t()
        assert out.shape == ref.shape
        bar = 2.0 ** -8 if m <= kw.GEMV_MAX_M else 8e-2
        assert float((out.float() - ref).abs().max() / ref.abs().max()) < bar, m


# --- synthetic units --------------------------------------------------------


@cuda
@pytest.mark.parametrize("rows,cols", [(64, 64), (128, 256), (200, 176), (48, 3072),
                                       (1024, 112), (96, 37), (37, 1)])
def test_synthetic_units_decode_byte_identically(rows, cols):
    """Shapes off the tile: ``rows`` not a multiple of ``LANES * VEC`` (256) and
    ``cols`` not a multiple of ``BLOCK_C``, both, and neither.

    ``BLOCK_C`` is a tuned constant (16 today, 64 before), so a shape whose
    ``cols`` is off *64* stops exercising the column mask the moment it is
    lowered -- 176 and 112 are both multiples of 16.  ``(96, 37)`` and the
    single-column ``(37, 1)`` are off every block width this kernel is likely
    to take, and they stay off it if the constant moves again.
    """
    kw = _kw()
    torch.manual_seed(rows * 1000 + cols)
    weight = torch.randn(rows, cols, dtype=torch.float32) * 0.02
    _exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=1024, name="synthetic",
        body=E4M3_RECIPE.body, span=E4M3_RECIPE.span,
        scale_plane=E4M3_RECIPE.scale_plane, window_bits=E4M3_RECIPE.window_bits,
        window_seed=E4M3_RECIPE.window_seed,
    )
    reference, scale = materialize_fp8(unit, forests or E4M3_GRID, None)
    prepared = kw.prepare_window_unit(
        unit.body_bits, unit.rates, unit.window_bits, unit.window_codes, E4M3_GRID,
        unit.scale_rows.float() * float(unit.scale_global), device="cuda",
    )
    assert torch.equal(prepared.decode(), reference.cuda())
    assert torch.equal(prepared.row_scale, scale.cuda().float())


@cuda
def test_a_mixed_rate_unit_decodes_byte_identically():
    """Not every column carries the same rate.

    The reach checkpoint is uniform R=4 on all 286,720 columns, so every unit
    test above exercises one runtime rate.  ``q256=896`` on E4M3 splits the
    columns half at R=3 and half at R=4, which is both the packer's mixed
    branch (columns of different bit lengths, each still byte-aligned) and the
    kernel's per-column ``rate`` load feeding a per-column shift.
    """
    kw = _kw()
    torch.manual_seed(896)
    weight = torch.randn(200, 176, dtype=torch.float32) * 0.02
    _exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=896, name="mixed",
        body=E4M3_RECIPE.body, span=E4M3_RECIPE.span,
        scale_plane=E4M3_RECIPE.scale_plane, window_bits=E4M3_RECIPE.window_bits,
        window_seed=E4M3_RECIPE.window_seed,
    )
    assert len(set(unit.rates)) > 1, "q256=896 no longer mixes rates on E4M3"
    reference, scale = materialize_fp8(unit, forests or E4M3_GRID, None)
    prepared = kw.prepare_window_unit(
        unit.body_bits, unit.rates, unit.window_bits, unit.window_codes, E4M3_GRID,
        unit.scale_rows.float() * float(unit.scale_global), device="cuda",
    )
    assert torch.equal(prepared.decode(), reference.cuda())

    x = torch.randn(2, prepared.cols, device="cuda", dtype=torch.bfloat16)
    tile = prepared.decode().view(torch.float8_e4m3fn).float() * prepared.row_scale[:, None]
    got = prepared.gemv(x)
    want = x.float() @ tile.t()
    assert float((got - want).abs().max() / want.abs().max()) < 2e-6


@cuda
def test_gemv_refuses_a_batch_it_cannot_block():
    """``window_gemv`` is the small-M path and says so.

    The accumulator is ``[MBLK, LANES, VEC]`` fp32 in registers; M past the cap
    spills instead of running, and the decode-then-GEMM path is faster there
    anyway.  ``window_linear`` routes large M itself, so the refusal only ever
    reaches a caller who asked for the GEMV by name.
    """
    kw = _kw()
    rows, cols, window_bits, rate = 64, 48, 14, 4
    torch.manual_seed(11)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    codes = torch.randint(0, 256, (1 << window_bits,), dtype=torch.uint8, device="cuda")
    scale = torch.rand(rows, device="cuda") + 0.5
    p = kw.prepare_window_unit(body, (rate,) * cols, window_bits, codes, E4M3_GRID,
                               scale, device="cuda")
    x = torch.randn(kw.GEMV_MAX_M + 1, p.cols, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(GrammarError, match="small-M"):
        p.gemv(x)


@cuda
def test_a_nonzero_initial_window_decodes_from_the_definition():
    """A TP shard starts mid-column: the kernel must take the initial window.

    The same stream is decoded twice -- once by the kernel with a per-column
    initial window, once by the one-step-at-a-time definition seeded with the
    same windows -- and the two must agree byte for byte.  The zero case is
    checked in the same test so a kernel that ignored the argument entirely
    would fail on the nonzero one and pass on the zero one, which is the
    failure mode worth separating.
    """
    kw = _kw()
    rows, cols, window_bits, rate = 96, 80, 14, 4
    torch.manual_seed(7)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    codes = torch.randint(0, 256, (1 << window_bits,), dtype=torch.uint8, device="cuda")
    table = kw.window_code_table(codes, E4M3_GRID, "cuda")
    scale = torch.rand(rows, device="cuda") + 0.5
    for initial in (
        torch.zeros(cols, dtype=torch.int64, device="cuda"),
        torch.randint(0, 1 << window_bits, (cols,), dtype=torch.int64, device="cuda"),
    ):
        prepared = kw.prepare_window_unit(
            body, (rate,) * cols, window_bits, codes, E4M3_GRID, scale,
            initial=initial, device="cuda",
        )
        expected = _reference_codes(body, (rate,) * cols, window_bits, table, initial)
        assert torch.equal(prepared.decode(), expected)


def _poke_pad(plane: torch.Tensor, offsets, window_bits: int, initial) -> torch.Tensor:
    """Write each column's initial window into its stream's ``L`` pad bits.

    ``pack_window_planes`` leaves ``window_bits`` zero bits in front of every
    column -- which *is* ``state_{-1} = 0`` -- and a TP shard fills them with
    the window its parent's rows ended on.  MSB-first, like every plane here.
    """
    out = plane.clone().cpu()
    for c in range(len(offsets)):
        off = int(offsets[c])
        w = int(initial[c])
        for i in range(window_bits):
            if (w >> (window_bits - 1 - i)) & 1:
                out[(off + i) // 8] |= 1 << (7 - ((off + i) % 8))
    return out


@cuda
def test_the_initial_window_may_travel_in_the_stream_pad():
    """A TP shard's initial state in the wire's pad, not in a side tensor.

    The two representations must be the same object: ``pack_window_planes``
    pads every column with ``L`` zero bits, and code ``t``'s window starts at
    ``offset + (t + 1) * R``, so for ``(t + 1) * R < L`` that window *reaches
    into the pad*.  A shard therefore needs no extra plane -- it writes its
    parent's last ``L`` bits into the pad and the kernel reads them like any
    other wire bits.  This asserts the kernel does exactly that: the same
    states decoded three ways -- pad-carried with ``initial = 0``, side-tensor
    with a zero pad, and the one-step definition -- agree byte for byte, and
    the GEMV built on the pad-carried stream agrees with its own tile.

    It reaches for ``_plane_words`` because the point is a *wire* fact, and
    the wire is what the lane's packer hands the kernel; there is no public
    seam that takes a stream someone else packed.
    """
    from tessera.lane_planes import pack_window_planes

    kw = _kw()
    rows, cols, window_bits, rate = 96, 80, 14, 4
    torch.manual_seed(11)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    codes = torch.randint(0, 256, (1 << window_bits,), dtype=torch.uint8, device="cuda")
    table = kw.window_code_table(codes, E4M3_GRID, "cuda")
    scale = torch.rand(rows, device="cuda") + 0.5
    start = torch.randint(0, 1 << window_bits, (cols,), dtype=torch.int64, device="cuda")

    plane, offsets, rate_t = pack_window_planes(body, (rate,) * cols, window_bits)
    padded = _poke_pad(plane, offsets.cpu(), window_bits, start.cpu()).to("cuda")
    zero = torch.zeros(cols, dtype=torch.int64, device="cuda")
    words = kw._plane_words(padded.contiguous())
    from_pad = kw.decode_fp8_tile(words, offsets.contiguous(), rate_t.contiguous(),
                                  zero, table, rows, cols, window_bits, rate)

    side = kw.prepare_window_unit(body, (rate,) * cols, window_bits, codes, E4M3_GRID,
                                  scale, initial=start, device="cuda")
    assert torch.equal(from_pad, side.decode())
    assert torch.equal(from_pad,
                       _reference_codes(body, (rate,) * cols, window_bits, table, start))

    x = torch.randn(1, cols, device="cuda", dtype=torch.bfloat16) * 0.1
    y = kw.window_gemv(x, words, offsets.contiguous(), rate_t.contiguous(), zero,
                       kw.window_value_table(table, "cuda"), scale, rows, cols,
                       window_bits, rate)
    ref = (from_pad.view(torch.float8_e4m3fn).float() * scale[:, None]) @ x[0].float()
    assert float((y[0].float() - ref).abs().max() / ref.abs().max()) < 2.0 ** -8


@cuda
def test_an_initial_window_wider_than_the_wire_is_refused():
    kw = _kw()
    body = torch.zeros(32, 16, dtype=torch.uint8, device="cuda")
    codes = torch.zeros(1 << 10, dtype=torch.uint8, device="cuda")
    with pytest.raises(GrammarError, match="initial window"):
        kw.prepare_window_unit(
            body, (4,) * 16, 10, codes, E4M3_GRID,
            torch.ones(32, device="cuda"),
            initial=torch.full((16,), 1 << 10, dtype=torch.int64, device="cuda"),
            device="cuda",
        )


@cuda
@checkpoint
def test_a_released_or_rotated_unit_is_refused():
    """The lane applies no post-decode transform, so it refuses units carrying
    one rather than serving weights the reference would not produce."""
    kw = _kw()
    _module, _role, parsed = next(_units(limit=1))
    parsed.unit.release_index = torch.zeros(3, dtype=torch.int64, device="cuda")
    with pytest.raises(GrammarError, match="released"):
        kw.prepare_from_parsed(parsed, device="cuda")


# --- the GEMV ---------------------------------------------------------------


@cuda
@checkpoint
def test_gemv_matches_the_dequantised_reference():
    """``window_gemv`` against ``(tile.float() * scale) @ x.float()``.

    Tolerance: the kernel accumulates in fp32 and reduces over split-K by
    atomic add, so the sum order differs from torch's.  A ``K``-term fp32 sum
    carries a relative error bounded by about ``K * 2**-24`` in the worst case
    and ``sqrt(K) * 2**-24`` in practice; at ``K = 3072`` that is 1.8e-4 worst
    case, 1.0e-5 typical.  ``2e-6`` of the row's own magnitude is the bar --
    an order under the practical bound and three under the worst case -- the
    same shape of bar the span-2 receipt set at 3.4e-07 for its shorter sum.
    A one-hot probe is checked separately at exact equality, where nothing is
    summed and no tolerance is owed.
    """
    kw = _kw()
    torch.manual_seed(3)
    for _module, role, parsed in _units(limit=6):
        prepared = kw.prepare_from_parsed(parsed, device="cuda")
        weight = prepared.decode().view(torch.float8_e4m3fn).float() * prepared.row_scale[:, None]
        for m in (1, 2, 4, 8):
            x = torch.randn(m, prepared.cols, device="cuda", dtype=torch.bfloat16)
            got = prepared.gemv(x)
            want = x.float() @ weight.t()
            error = (got - want).abs().max() / want.abs().max()
            assert float(error) < 2e-6, (role, m, float(error))
        probe = torch.zeros(1, prepared.cols, device="cuda", dtype=torch.bfloat16)
        probe[0, 5] = 1.0
        assert torch.equal(prepared.gemv(probe)[0], weight[:, 5])


@cuda
@checkpoint
def test_window_linear_dispatches_on_m_inside_the_op():
    """One op, both branches: the GEMV under ``gemv_max`` and the decoded W8A8
    GEMM above it, each within its own contract's error of the fp32 product.

    ``window_linear`` returns **bf16**, so the bar is bf16's own rounding (8
    mantissa bits, ``2**-9`` relative per element) and not the fp32 GEMV's
    2e-6: an exact result rounded to bf16 already lands ~2e-3 from an fp32
    reference measured against the row's largest entry.

    Each branch is compared to *its own* contract's exact product: below
    ``gemv_max`` the bf16 activation goes in whole, above it the activation is
    quantised per token to E4M3 first.  Comparing the W8A8 branch to the bf16
    product instead would measure the A-side's 3-bit mantissa (~3e-2), which
    is the contract's error and not this kernel's.
    """
    kw = _kw()
    _module, _role, parsed = next(_units(limit=1))
    p = kw.prepare_from_parsed(parsed, device="cuda")
    weight = p.decode().view(torch.float8_e4m3fn).float() * p.row_scale[:, None]
    torch.manual_seed(11)
    for m in (1, 8, 64):
        x = torch.randn(m, p.cols, device="cuda", dtype=torch.bfloat16) * 0.1
        y = kw.window_linear(x, p.plane_words, p.offsets, p.rates, p.initial,
                             p.code_table, p.value_table, p.row_scale, p.rows, p.cols,
                             p.window_bits, p.max_rate)
        if m <= 8:
            a = x.float()
        else:
            a_q, a_scale = kw._fp8_per_token(x)
            a = a_q.float() * a_scale
        want = a @ weight.t()
        assert y.shape == (m, p.rows) and y.dtype == torch.bfloat16
        assert float((y.float() - want).abs().max() / want.abs().max()) < 5e-3, m


# --- the BF16 family --------------------------------------------------------
#
# Same WINDOW body, same CHANNEL plane, same 2^14 table -- the table holds bf16
# VALUES instead of E4M3 codes, and the tile that comes out is what a stock
# BF16 GEMM multiplies, with the row scale already folded in.  The family's
# encoder is not merged, so these are synthetic streams against the one-step
# definition; what they prove is the parametrisation, not the grid.


def _reference_states(body_bits, rates, window_bits, initial):
    """The window state at every position, by the body's definition.

    ``_reference_codes`` walks the same recurrence but returns the TABLE's
    byte; the value family needs the state itself, in int64, because its
    table is 2^14 floats and not a byte map.
    """
    rows, cols = body_bits.shape
    mask = (1 << window_bits) - 1
    state = initial.to(torch.int64).clone()
    out = torch.zeros(rows, cols, dtype=torch.int64, device=body_bits.device)
    rate = torch.tensor(rates, dtype=torch.int64, device=body_bits.device)
    for t in range(rows):
        state = ((state << rate) | body_bits[t].to(torch.int64)) & mask
        out[t] = state
    return out


def _value_reference(body_bits, rates, window_bits, table, initial, row_scale):
    """The tile by the body's definition: walk, look up, scale, round.

    The rounding is last and is the tile's own dtype, which is what the
    kernel does -- it widens the table entry to fp32, applies the fp32 row
    scale and stores one rounded value.
    """
    state = _reference_states(body_bits, rates, window_bits, initial)
    return (table[state].float() * row_scale[:, None]).to(table.dtype)


def _value_reference_f32(body_bits, rates, window_bits, table, initial, row_scale):
    """The same product WITHOUT the tile's rounding.

    The GEMV never materialises the tile, so it accumulates the unrounded
    ``table[state] * scale``; holding it to a rounded tile would be holding a
    more accurate kernel to a less accurate reference.
    """
    state = _reference_states(body_bits, rates, window_bits, initial)
    return table[state].float() * row_scale[:, None]


@cuda
@pytest.mark.parametrize("rows,cols", [(256, 128), (200, 176), (128, 1024),
                                       (72, 45)])
def test_a_bf16_table_decodes_a_scaled_bf16_tile(rows, cols):
    """``decode_value_tile``: the tile is the table's dtype, scale applied."""
    kw = _kw()
    window_bits, rate = 14, 4
    torch.manual_seed(rows + cols)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")
    assert unit.family == "value"
    got = unit.decode()
    assert got.dtype is torch.bfloat16
    want = _value_reference(body, (rate,) * cols, window_bits, table,
                            torch.zeros(cols, dtype=torch.int64, device="cuda"), scale)
    assert torch.equal(got, want)


@cuda
def test_the_bf16_family_starts_from_an_initial_window_too():
    """A TP shard of the value family is the same shard rule as the FP8 one."""
    kw = _kw()
    rows, cols, window_bits, rate = 96, 80, 14, 4
    torch.manual_seed(3)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    start = torch.randint(0, 1 << window_bits, (cols,), dtype=torch.int64, device="cuda")
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    initial=start, device="cuda")
    want = _value_reference(body, (rate,) * cols, window_bits, table, start, scale)
    assert torch.equal(unit.decode(), want)
    zero = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")
    assert not torch.equal(zero.decode(), want)


@cuda
def test_the_gemv_reads_a_bf16_table():
    """One GEMV kernel, two alphabets: the table's dtype is all that changes.

    Held to the same fp32-accumulation bar as the FP8 family, against the
    decoded tile in fp32 -- the tile already carries the scale, so the
    reference is a plain matmul.
    """
    kw = _kw()
    rows, cols, window_bits, rate = 512, 640, 14, 4
    torch.manual_seed(5)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")
    tile = _value_reference_f32(body, (rate,) * cols, window_bits, table,
                                torch.zeros(cols, dtype=torch.int64, device="cuda"),
                                scale)
    for m in (1, 4, 8):
        x = torch.randn(m, cols, device="cuda", dtype=torch.bfloat16)
        got = unit.gemv(x)
        want = x.float() @ tile.t()
        assert float((got - want).abs().max() / want.abs().max()) < 2e-6, m


@cuda
def test_the_bf16_linear_routes_both_sides_of_the_cap():
    """``window_value_linear`` at M=1 (GEMV) and M past the cap (BF16 GEMM).

    Both sides are W16A16 -- the wide side decodes a bf16 tile and runs the
    stock GEMM, so unlike the FP8 family nothing quantises the activation --
    and both are held to bf16 rounding against the fp32 product.
    """
    kw = _kw()
    rows, cols, window_bits, rate = 384, 512, 14, 4
    torch.manual_seed(9)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")
    tile = _value_reference_f32(body, (rate,) * cols, window_bits, table,
                                torch.zeros(cols, dtype=torch.int64, device="cuda"),
                                scale)
    for m in (1, kw.GEMV_MAX_M + 1, 64):
        x = torch.randn(m, cols, device="cuda", dtype=torch.bfloat16)
        got = unit.linear(x)
        assert got.dtype is torch.bfloat16
        want = x.float() @ tile.t()
        assert float((got.float() - want).abs().max() / want.abs().max()) < 2.0 ** -7, m
    got = kw.window_module_linear(torch.randn(1, cols, device="cuda",
                                              dtype=torch.bfloat16), [unit, unit])
    assert got.shape == (1, 2 * rows)


@cuda
def test_the_two_families_refuse_each_other_s_tables():
    """A value table through the FP8 op, and a code table through the value op."""
    kw = _kw()
    rows, cols, window_bits, rate = 64, 64, 14, 4
    torch.manual_seed(13)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")
    with pytest.raises(GrammarError, match="E4M3 bytes"):
        kw.decode_fp8_tile(unit.plane_words, unit.offsets, unit.rates, unit.initial,
                           unit.value_table, rows, cols, window_bits, rate)
    codes = torch.randint(0, 256, (1 << window_bits,), dtype=torch.uint8, device="cuda")
    with pytest.raises(GrammarError, match="floating-point"):
        kw.prepare_window_values(body, (rate,) * cols, window_bits, codes, scale,
                                 device="cuda")


@cuda
@checkpoint
def test_the_ops_survive_a_compiled_forward():
    """The three ops are opaque, functional and static-shaped, so Inductor
    traces through a forward that calls them without a graph break, without
    reading a data pointer and without guarding on the token dimension."""
    kw = _kw()
    _module, _role, parsed = next(_units(limit=1))
    p = kw.prepare_from_parsed(parsed, device="cuda")

    def forward(x):
        y = kw.window_linear(x, p.plane_words, p.offsets, p.rates, p.initial,
                             p.code_table, p.value_table, p.row_scale, p.rows, p.cols,
                             p.window_bits, p.max_rate)
        return y * 2

    x = torch.randn(1, p.cols, device="cuda", dtype=torch.bfloat16) * 0.1
    eager = forward(x)
    compiled = torch.compile(forward, fullgraph=True, dynamic=False)(x)
    # Not `equal`: the GEMV is split-K over atomics, so two launches of the
    # same kernel sum the partials in whatever order the blocks retire, and the
    # bf16 rounding of a differently-ordered fp32 sum can land one ulp apart.
    # What this test is for is that `fullgraph=True` holds -- the op traced,
    # nothing broke the graph -- so the values are held to bf16's own step.
    assert torch.allclose(eager, compiled, rtol=2 ** -7, atol=0.0)


@cuda
def test_the_value_ops_survive_a_compiled_forward():
    """The BF16 family's two ops trace as well as the FP8 family's.

    Same contract, and it is not implied by the FP8 test: these ops have their
    own ``register_fake``, and a fake impl that got the output dtype from a
    real tensor's ``.dtype`` would only fail here, under fake tensors, where
    the table is a meta tensor and the eager path never runs.
    """
    kw = _kw()
    rows, cols, window_bits, rate = 128, 256, 14, 4
    torch.manual_seed(11)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")

    def forward(x):
        return unit.linear(x) * 2

    x = torch.randn(1, cols, device="cuda", dtype=torch.bfloat16) * 0.1
    eager = forward(x)
    compiled = torch.compile(forward, fullgraph=True, dynamic=False)(x)
    assert compiled.dtype is torch.bfloat16
    assert torch.allclose(eager, compiled, rtol=2 ** -7, atol=0.0)

    def decode_forward():
        return unit.decode().float().sum()

    got = torch.compile(decode_forward, fullgraph=True, dynamic=False)()
    assert torch.allclose(got, decode_forward(), rtol=2 ** -12, atol=0.0)


@cuda
def test_the_value_family_has_no_row_scale_to_hand_out():
    """``window_module_row_scale`` refuses the value family.

    Its scale is already inside the decoded tile; returning it to a lane that
    would pass it to ``_scaled_mm`` beside the tile is a silent second
    multiplication, and this seam exists to stop exactly that.
    """
    kw = _kw()
    rows, cols, window_bits, rate = 64, 64, 14, 4
    torch.manual_seed(12)
    body = torch.randint(0, 1 << rate, (rows, cols), dtype=torch.uint8, device="cuda")
    table = (torch.randn(1 << window_bits, device="cuda") * 0.05).to(torch.bfloat16)
    scale = torch.rand(rows, device="cuda") + 0.25
    unit = kw.prepare_window_values(body, (rate,) * cols, window_bits, table, scale,
                                    device="cuda")
    with pytest.raises(GrammarError, match="already applied"):
        kw.window_module_row_scale([unit])
