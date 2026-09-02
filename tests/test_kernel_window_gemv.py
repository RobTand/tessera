"""The window-body GEMV (``kernel_window_gemv``): bit-exactness first, then
the GEMV's derived fp32 tolerance, then the value family and M<=8.

The repack is a bijection of the BODY plane and the kernel's state extraction
is the definition ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L`` -- so
``decode_codes`` must equal ``materialize_fp8`` byte for byte on every unit of
the reach checkpoint, and the GEMV must land within a bound derived from fp32
accumulation of ``(tile.float() * scale) @ x``.  A one-hot ``x`` is exact.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E4M3_GRID                                  # noqa: E402
from tessera.decode import materialize_fp8                             # noqa: E402
from tessera.errors import GrammarError                                # noqa: E402
from tessera.fused import parse_fused                                  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact                  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

REACH = Path("/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook")
TWIN = Path("/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-stock-twin")
checkpoint = pytest.mark.skipif(
    not (REACH / "model.safetensors").exists(),
    reason="the reach checkpoint is not on this box",
)
L = 14


def _kg():
    from tessera import kernel_window_gemv

    return kernel_window_gemv


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


def _synthetic(rows, cols, rates, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rate = torch.tensor(rates, dtype=torch.int64)
    body = (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.uint8)
    codes = torch.randint(0, 256, (1 << L,), generator=g).to(torch.uint8)
    return body.to(device), codes.to(device)


class _Parsed:
    """A ParsedUnit stand-in for synthetic bodies (only the fields prepare reads)."""

    def __init__(self, body, rates, codes, scale_rows, scale_global=1.0):
        from types import SimpleNamespace
        from tessera.manifest import BodyKind, RotationState, ScalePlaneKind

        self.grid = E4M3_GRID
        self.unit = SimpleNamespace(
            body=BodyKind.WINDOW, scale_plane=ScalePlaneKind.CHANNEL,
            release_index=torch.zeros(0, dtype=torch.int64), diagonals=None,
            rotation=RotationState.NONE, window_codes=codes, window_bits=L,
            scale_rows=scale_rows, scale_global=scale_global, body_bits=body, rates=tuple(rates),
        )


def _reference_bytes(body, rates, codes):
    kg = _kg()
    states = kg.reference_states(body, rates, L)
    native = torch.tensor(E4M3_GRID.native, dtype=torch.uint8, device=body.device)
    return native[codes[states].long()]


def _bound(tile_f32, scale, x):
    """A deterministic fp32 accumulation bound: ``2K * 2^-23 * sum_j |w_ij x_j|``
    (each of the K partial sums is rounded once, in either order; the factor 2
    covers the reference's own rounding of the same size).  Shape ``[M, rows]``."""
    K = x.shape[1]
    mag = (tile_f32 * scale[:, None]).abs().double() @ x.abs().double().t()   # [rows, M]
    return (2 * K * 2.0 ** -23) * mag.t() + 1e-30


# --- bijection and tables ------------------------------------------------------


def test_every_legal_e4m3_byte_is_exact_in_bf16():
    """The value table is bf16 -- lossless for E4M3 (3 mantissa bits, exponents inside bf16's)."""
    byte = torch.arange(256, dtype=torch.uint8)
    value = byte.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(value)
    assert int(finite.sum()) == 254
    assert torch.equal(value[finite].bfloat16().float(), value[finite])


@cuda
@pytest.mark.parametrize("rates_spec", ["4", "2", "1", "mixed24", "mixed124"])
@pytest.mark.parametrize("rows", [512, 1024, 96, 1000])
def test_synthetic_decode_is_the_definition(rates_spec, rows):
    """Every rate, mixed schedules, rows on and off the tile: the kernel's
    codes equal the step-at-a-time definition, native byte for native byte."""
    kg = _kg()
    cols = 40
    if rates_spec == "mixed24":
        rates = tuple(2 if (c % 3) else 4 for c in range(cols))
    elif rates_spec == "mixed124":
        rates = tuple((1, 2, 4)[c % 3] for c in range(cols))
    else:
        rates = (int(rates_spec),) * cols
    body, codes = _synthetic(rows, cols, rates)
    scale_rows = torch.rand(rows, dtype=torch.float16) + 0.5
    unit = kg.prepare_from_parsed(_Parsed(body, rates, codes, scale_rows, 0.75))
    got, scale = kg.decode_fp8(unit)
    assert torch.equal(got, _reference_bytes(body, rates, codes))
    assert torch.equal(scale, (scale_rows.float() * 0.75).cuda())
    # the repack is a bijection: the words carry rows_p*sum(R) bits and nothing else
    assert unit.rep.words.numel() * 32 == unit.rep.rows_p * sum(rates)
    # and the GEMV agrees with the decoded tile at M=1 and M=4 (both item caps)
    w = got.view(torch.float8_e4m3fn).float()
    for M in (1, 4):
        x = torch.randn(M, cols, device="cuda").bfloat16()
        if M == 4 and 1 in rates:
            with pytest.raises(GrammarError, match="rate-1 column"):   # refused, never silent
                kg.window_gemv(unit, x)
            continue
        y = kg.window_gemv(unit, x)
        ref = (w * scale[:, None]).double() @ x.double().t()
        assert bool(((y.double() - ref.t()).abs() <= _bound(w, scale, x)).all())


@cuda
def test_prepare_refuses_rate_three_and_names_the_fallback():
    kg = _kg()
    body, codes = _synthetic(512, 8, (3,) * 8)
    with pytest.raises(GrammarError, match="materialised FP8 path"):
        kg.prepare_from_parsed(_Parsed(body, (3,) * 8, codes, torch.ones(512, dtype=torch.float16)))


@cuda
def test_prepare_refuses_other_windows():
    kg = _kg()
    body, codes = _synthetic(512, 8, (4,) * 8)
    p = _Parsed(body, (4,) * 8, codes, torch.ones(512, dtype=torch.float16))
    p.unit.window_bits = 12
    p.unit.window_codes = codes[: 1 << 12]
    with pytest.raises(GrammarError, match="window_bits 12"):
        kg.prepare_from_parsed(p)


# --- byte identity on the shipped checkpoint ------------------------------------


@cuda
@checkpoint
def test_every_reach_unit_decodes_byte_identically():
    """All 196 units: the kernel's decode equals ``materialize_fp8`` byte for
    byte, and the row scale equals the reference's fp32 expression exactly."""
    kg = _kg()
    seen = 0
    for _module, _role, parsed in _units():
        unit = kg.prepare_from_parsed(parsed)
        got, scale = kg.decode_fp8(unit)
        want, want_scale = materialize_fp8(parsed.unit, parsed.forests, parsed.code)
        assert torch.equal(got, want.cuda()), f"{_module}/{_role}"
        assert torch.equal(scale, want_scale.cuda().float()), f"{_module}/{_role}"
        seen += 1
    assert seen == 196


@cuda
@checkpoint
def test_reach_units_gemv_within_fp32_bound_and_one_hot_exact():
    """Every unit: M=1 GEMV within the derived bound of the fp32 reference, and
    a one-hot x reproduces the scaled column exactly."""
    kg = _kg()
    g = torch.Generator(device="cuda").manual_seed(1)
    for _module, _role, parsed in _units():
        unit = kg.prepare_from_parsed(parsed)
        tile, scale = materialize_fp8(parsed.unit, parsed.forests, parsed.code)
        w = tile.cuda().view(torch.float8_e4m3fn).float()
        scale = scale.cuda().float()
        x = torch.randn(1, unit.cols, device="cuda", generator=g).bfloat16()
        y = kg.window_gemv(unit, x)
        ref = (w * scale[:, None]).double() @ x.double().t()
        err = (y.double() - ref.t()).abs()
        bound = _bound(w, scale, x)
        assert bool((err <= bound).all()), f"{_module}/{_role}: max err {err.max()} bound {bound.min()}"
        j = int(torch.randint(0, unit.cols, (1,), device="cuda", generator=g))
        onehot = torch.zeros(1, unit.cols, device="cuda", dtype=torch.bfloat16)
        onehot[0, j] = 1
        assert torch.equal(kg.window_gemv(unit, onehot)[0], w[:, j] * scale), f"{_module}/{_role}"


@cuda
@checkpoint
def test_reach_matches_stock_twin_bytes():
    """The twin checkpoint's materialised FP8 bytes are what the kernel decodes."""
    from safetensors import safe_open

    kg = _kg()
    twin = TWIN / "model.safetensors"
    if not twin.exists():
        pytest.skip("no stock twin on this box")
    with safe_open(str(twin), framework="pt") as handle:
        keys = set(handle.keys())
        checked = 0
        for module, role, parsed in _units(limit=12):
            # fused modules (qkv_proj, gate_up_proj) carry their members by role;
            # the twin stores each member under the parent's name
            parent = module.rsplit(".", 1)[0]
            candidates = [f"{parent}.{role}.weight", f"{module}.{role}.weight", f"{module}.weight"]
            name = next((n for n in candidates if n in keys), None)
            if name is None:
                continue
            unit = kg.prepare_from_parsed(parsed)
            got, _ = kg.decode_fp8(unit)
            want = handle.get_tensor(name)
            if want.dtype != torch.uint8:
                want = want.view(torch.uint8)
            assert torch.equal(got.cpu(), want), f"{module}/{role}"
            checked += 1
    if checked == 0:
        pytest.skip("twin names did not match the fused roles")


# --- the GEMV on synthetic units: M, plans, the value family ---------------------


@cuda
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 8])
def test_gemv_m_tiles_within_bound(M):
    kg = _kg()
    rows, cols = 1000, 640
    rates = (4,) * cols
    body, codes = _synthetic(rows, cols, rates, seed=M)
    scale_rows = (torch.rand(rows, dtype=torch.float16) + 0.5)
    unit = kg.prepare_from_parsed(_Parsed(body, rates, codes, scale_rows, 0.5), M=M)
    tile, scale = kg.decode_fp8(unit)
    w = tile.view(torch.float8_e4m3fn).float()
    x = torch.randn(M, cols, device="cuda").bfloat16()
    y = kg.window_gemv(unit, x)
    assert y.shape == (M, rows)
    ref = (w * scale[:, None]).double() @ x.double().t()
    err = (y.double() - ref.t()).abs()
    assert bool((err <= _bound(w, scale, x)).all())


@cuda
@pytest.mark.parametrize("rpl,warps,cpi,balanced", [(16, 16, 64, False), (16, 8, 16, False), (8, 16, 32, True),
                                                      (8, 8, 4, False), (16, 8, 1024, True), (16, 16, 300, True)])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_gemv_plans_agree(rpl, warps, cpi, balanced, table_dtype):
    """Every launch shape, both planners and both table types compute the same GEMV."""
    kg = _kg()
    rows, cols = 1536, 300
    rates = tuple(2 if c % 5 == 0 else 4 for c in range(cols))
    body, codes = _synthetic(rows, cols, rates, seed=7)
    unit = kg.prepare_from_parsed(_Parsed(body, rates, codes, torch.ones(rows, dtype=torch.float16)))
    unit = unit.with_plan(kg.Plan(rpl=rpl, warps=warps, blocks=13, cols_per_item=cpi, table_dtype=table_dtype,
                                  balanced=balanced))
    tile, scale = kg.decode_fp8(unit)
    w = tile.view(torch.float8_e4m3fn).float()
    x = torch.randn(2, cols, device="cuda").bfloat16()
    y = kg.window_gemv(unit, x)
    ref = (w * scale[:, None]).double() @ x.double().t()
    assert bool(((y.double() - ref.t()).abs() <= _bound(w, scale, x)).all())


@cuda
def test_value_family_bf16_table():
    """The bf16 value family: the same kernel over a bf16 table, no scale."""
    kg = _kg()
    rows, cols = 768, 256
    rates = (4,) * cols
    body, _ = _synthetic(rows, cols, rates, seed=3)
    values = (torch.randn(1 << L) * 0.02).bfloat16().cuda()
    unit = kg.prepare_value_unit(body, rates, L, values)
    states = kg.reference_states(body, rates, L)
    w = values.float()[states]
    x = torch.randn(1, cols, device="cuda").bfloat16()
    y = kg.window_gemv(unit, x)
    ref = w.double() @ x.double().t()
    ones = torch.ones(rows, device="cuda")
    assert bool(((y.double() - ref.t()).abs() <= _bound(w, ones, x)).all())
    j = 5
    onehot = torch.zeros(1, cols, device="cuda", dtype=torch.bfloat16)
    onehot[0, j] = 1
    assert torch.equal(kg.window_gemv(unit, onehot)[0], w[:, j])


@cuda
def test_value_family_scale_is_applied_on_the_output_not_the_tile():
    """The value family's contract: ``decode_values`` is the raw table value
    (bf16, exact), and the fp32 row scale is applied once on the accumulated
    output -- ``y_i = s_i * sum_k t_ik x_k`` -- never folded into a bf16 tile."""
    kg = _kg()
    rows, cols = 1000, 320
    rates = tuple(4 if c % 4 else 2 for c in range(cols))
    body, _ = _synthetic(rows, cols, rates, seed=11)
    values = (torch.randn(1 << L) * 0.05).bfloat16().cuda()
    scale = (torch.rand(rows, device="cuda") * 3 + 0.1)
    unit = kg.prepare_value_unit(body, rates, L, values, scale=scale)
    states = kg.reference_states(body, rates, L)
    raw = values[states]                                   # bf16 [rows, cols]
    assert torch.equal(kg.decode_values(unit), raw)        # the tile is the raw value, exactly
    x = torch.randn(3, cols, device="cuda").bfloat16()
    y = kg.window_gemv(unit, x)
    ref = scale.double()[:, None] * (raw.double() @ x.double().t())
    assert bool(((y.double() - ref.t()).abs() <= _bound(raw.float(), scale, x)).all())
    # and a bf16 fold of the scale into the tile would NOT be this number
    folded = (raw.float() * scale[:, None]).bfloat16().double() @ x.double().t()
    assert not torch.allclose(folded, ref, rtol=0, atol=0)


def _every_column_through_the_gemv(kg, unit, w, scale, M):
    """Identity slices ``x = [e_j, ..., e_{j+M-1}]`` over ALL columns: every
    weight travels through ``run_item``'s own lookback (lane 0, tile
    boundaries, the RPL=8 half tiles) and must come back exactly."""
    K = unit.cols
    for j0 in range(0, K, M):
        m = min(M, K - j0)
        x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
        for m_ in range(m):
            x[m_, j0 + m_] = 1
        y = kg.window_gemv(unit, x)
        want = (w[:, j0:j0 + m] * scale[:, None]).t()
        if not torch.equal(y[:m], want):
            bad = (y[:m] != want).nonzero()[0].tolist()
            raise AssertionError(f"column {j0 + bad[0]} row {bad[1]}: {y[bad[0], bad[1]]} != {want[bad[0], bad[1]]}")


@cuda
@pytest.mark.parametrize("M", [2, 8])
def test_gemv_every_weight_exact_synthetic(M):
    """Both lane widths (M=2 -> 16 rows per lane, M=8 -> 8): every weight of a
    mixed-rate unit with rows off the tile comes back exactly through the GEMV."""
    kg = _kg()
    rows, cols = 1000, 640
    rates = tuple(2 if c % 7 == 0 else 4 for c in range(cols))
    body, codes = _synthetic(rows, cols, rates, seed=21)
    scale_rows = torch.rand(rows, dtype=torch.float16) + 0.5
    unit = kg.prepare_from_parsed(_Parsed(body, rates, codes, scale_rows, 0.75), M=M)
    tile, scale = kg.decode_fp8(unit)
    assert torch.equal(tile, _reference_bytes(body, rates, codes))
    _every_column_through_the_gemv(kg, unit, tile.view(torch.float8_e4m3fn).float(), scale, M)


@cuda
@checkpoint
@pytest.mark.parametrize("M", [2, 8])
def test_gemv_every_weight_exact_on_a_reach_unit(M):
    """One real unit (the first q_proj), every column, both lane widths."""
    kg = _kg()
    _module, _role, parsed = next(_units(limit=1))
    unit = kg.prepare_from_parsed(parsed, M=M)
    tile, scale = materialize_fp8(parsed.unit, parsed.forests, parsed.code)
    _every_column_through_the_gemv(kg, unit, tile.cuda().view(torch.float8_e4m3fn).float(), scale.cuda().float(), M)


@cuda
def test_window_linear_seam_and_m_limit():
    kg = _kg()
    rows, cols = 512, 128
    body, codes = _synthetic(rows, cols, (4,) * cols, seed=5)
    unit = kg.prepare_from_parsed(_Parsed(body, (4,) * cols, codes, torch.ones(rows, dtype=torch.float16)))
    x = torch.randn(2, 3, cols, device="cuda").bfloat16()
    y = kg.window_linear(unit, x)
    assert y.shape == (2, 3, rows) and y.dtype == torch.bfloat16
    with pytest.raises(GrammarError, match="M=9"):
        kg.window_gemv(unit, torch.randn(9, cols, device="cuda").bfloat16())
