"""The window-body GEMV (``kernel_window_gemv``): bit-exactness first, then
the GEMV's derived fp32 tolerance, then the value family and M<=8.

The repack is a bijection of the BODY plane and the kernel's state extraction
is the definition ``state_t = ((state_{t-1} << R) | bits_t) mod 2^L`` -- so
``decode_codes`` must equal ``materialize_fp8`` byte for byte on every unit of
the reach checkpoint, and the GEMV must land within a bound derived from fp32
accumulation of ``(tile.float() * scale) @ x``.  A one-hot ``x`` is exact.
"""

import os
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


def _require_toolchain() -> None:
    """Skip only when this host holds no ``nvcc`` ANYWHERE.

    An absent toolchain is not a failing kernel, so it skips.  But the search
    has to be the same one the module itself performs, or the skip lies: the
    first version of this guard read ``cpp_extension.CUDA_HOME``, which on
    sparky points at an alternatives symlink to a toolkit with no compiler,
    and so skipped 50 tests on a box where the kernel builds and all 51 pass.
    A toolkit that IS found and then fails to compile is a real failure and is
    left to raise.
    """
    from tessera.kernel_window_gemv import cuda_home_with_nvcc

    if cuda_home_with_nvcc():
        return
    pytest.skip("no nvcc under CUDA_HOME, PATH or any /usr/local/cuda-* root; "
                "the window GEMV extension cannot be built here")


def _kg():
    from tessera import kernel_window_gemv

    _require_toolchain()
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


def test_a_code_times_a_per_token_scale_is_NOT_exact_in_bf16():
    """The half-truth above is what #110 was built on, so pin the other half.

    A code being exact in bf16 was read as licence to hand the GEMV
    ``bf16(code * a_scale)``.  It is not: the code carries four significant
    bits, an fp32 per-token scale carries twenty-four, and their product needs
    up to twenty-eight where bf16 keeps eight.  So the fold is exactly ONE
    bf16 rounding of every activation element -- which is why the lane applies
    ``a_scale`` to the fp32 OUTPUT instead
    (``fp8_gemv.streamed_apply``; the same rule ``bf16_route`` holds for the
    weight side).

    CPU, no kernel: this is arithmetic, and the point of putting it here is
    that it runs on a box with no GPU, unlike everything else that pins the
    fix.  The bounds are dtype properties (bf16 keeps 8 significand bits, so
    one rounding is at most 2^-8 relative), never tuned constants.
    """
    torch.manual_seed(0)
    codes = (torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
             .float().double())
    codes = codes[torch.isfinite(codes) & (codes != 0)]
    # A per-token scale as the route makes it: amax / 448 over a random token.
    scales = (torch.rand(4096, dtype=torch.float64) * 8.0 + 0.01) / 448.0
    exact = codes[None, :] * scales[:, None]
    folded = exact.to(torch.bfloat16).double()
    rel = ((folded - exact) / exact).abs()

    assert float(rel.max()) > 0.0, "bf16 rounded nothing: the product cannot be exact"
    assert float(rel.max()) <= 2.0 ** -8 + 1e-12, "more than one bf16 rounding appeared"
    rms = float(((folded - exact).pow(2).sum() / exact.pow(2).sum()).sqrt())
    assert 1.0e-3 < rms < 2.0 ** -9 * 1.2, f"one bf16 rounding should read ~1.6e-3, got {rms:.3e}"
    # And the operand the lane actually hands the kernel after the fix IS exact.
    assert torch.equal(codes.float().to(torch.bfloat16).double(), codes)


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


def test_m2_is_not_routed_to_the_mt4_build():
    """M=2 runs the MT=2 (16-rows-per-lane) build: no blanket M-only routing (#59).

    ``docs/measurements/tessera-window-gemv-2026-09-02.md`` §11 once recorded
    routing M=2 to the MT=4 kernel as "a free ~8%" from the contended per-token
    table. The same document's quiet-box addendum re-took M=1,2,4,8 on an idle
    box, and the M=4 column -- which is what a padded M=2 launch would cost
    (same ``mt``, same ``items_for(mt)``, same ``rpl = 8``) -- is +11.6% /
    +4.7% / +21.4% per token against MT=2 on the three lists measured. A
    per-shape variant stays open, but its threshold must be derived from
    occupancy and re-measured, so the M-only rule must not come back through
    ``_m_tile``: ``_gemv_concrete`` follows ``mt`` into the ``items_1`` /
    ``rpl=16`` build for ``mt <= 2`` and the ``items_4`` / ``rpl=8`` build
    above, so this mapping IS the routing. CPU-only by construction: the rule
    is pure Python and must hold where no toolchain exists to hide behind.
    """
    from tessera.kernel_window_gemv import _m_tile

    assert _m_tile(1) == 1
    assert _m_tile(2) == 2
    assert _m_tile(3) == 4 and _m_tile(4) == 4
    for M in (5, 6, 7, 8):
        assert _m_tile(M) == 8, M
    with pytest.raises(GrammarError, match="exceeds the GEMV"):
        _m_tile(9)


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


# ---------------------- the TP shard this lane cannot take ------------------
#
# §6 P0-shardstate of the 2026-09-02 math audit.  ``kernel_window_gemv`` never
# read ``initial_state``, so a row shard was repacked as if it started from the
# pinned zero register.  Unlike the Triton lane it cannot simply be threaded:
# ``csrc/window_gemv.cu`` supplies ``state_{-1}`` itself -- "the L-bit pad that
# opens every wire column is not stored: it is all zeros by definition
# (state_{-1} = 0) and the kernel supplies it" (window_gemv.cu:17), and lane 0
# of tile 0 takes its ``prev`` word from "the zero pad on tile 0"
# (window_gemv.cu:23).  Taking a start state is a kernel change on a path this
# pass cannot measure, so the lane fails closed instead, exactly as
# ``lane_planes.pack_unit_for_kernel`` already does for a span-2 TCQ shard.
#
# CPU-reachable on purpose: the refusal precedes every device touch.


class _ShardParsed(_Parsed):
    """``_Parsed`` carrying an INITIAL_STATE plane, as ``layout.SlicedUnit`` does."""

    def __init__(self, body, rates, codes, scale_rows, initial_state):
        super().__init__(body, rates, codes, scale_rows)
        self.unit.initial_state = initial_state


def _kg_no_build():
    """The module without ``_require_toolchain``: these refusals never build."""
    from tessera import kernel_window_gemv

    return kernel_window_gemv


def test_the_window_gemv_refuses_a_shard_start_state():
    """A shard is refused by name, not decoded from zero."""
    kg = _kg_no_build()
    rows, cols = 512, 64
    body = torch.zeros(rows, cols, dtype=torch.uint8)
    codes = torch.zeros(1 << L, dtype=torch.uint8)
    scale_rows = torch.ones(rows, dtype=torch.float16)
    start = torch.arange(cols, dtype=torch.int64) % (1 << L)
    with pytest.raises(GrammarError, match="start state"):
        kg.prepare_from_parsed(_ShardParsed(body, (4,) * cols, codes, scale_rows, start))


def test_the_value_family_refuses_a_shard_start_state():
    """The raw-bits entry point fails closed on the same state.

    ``prepare_value_unit`` takes body bits rather than a unit, so a caller
    holding a ``SlicedUnit`` would otherwise drop the state with nowhere to
    say so.  The keyword exists only to be refused.
    """
    kg = _kg_no_build()
    rows, cols = 512, 64
    body = torch.zeros(rows, cols, dtype=torch.uint8)
    values = torch.zeros(1 << L, dtype=torch.bfloat16)
    with pytest.raises(GrammarError, match="start state"):
        kg.prepare_value_unit(body, (4,) * cols, L, values,
                              initial_state=torch.zeros(cols, dtype=torch.int64))


# ---------------------- the compiled arm (RobTand/tessera#52) -------------------
#
# Every route in ``tessera.serving`` is served eager AND compiled, and this
# lane has been broken by exactly what the first ``window_gemv`` did in the
# traced region: a Python branch on the token dim, a pad and a slice on it, an
# ``lru_cache``d JIT build and a direct pybind call
# (``vllm-compiled-forward-breaks-lane-hot-paths``).  On the pre-#52 tree
# ``torch.compile(fullgraph=True)`` of the seam raised ``Unsupported:
# Attempted to call function marked as skipped`` (``posix.stat`` inside the
# build Dynamo traced into).  These tests are the arm that would have said so.


def _mixed_unit(kg, rows=1000, cols=640, seed=31, M=1):
    """A mixed-rate E4M3 unit (so the column permutation is a real gather and
    both item tables exist) with its fp32 tile and scale."""
    rates = tuple(2 if c % 7 == 0 else 4 for c in range(cols))
    body, codes = _synthetic(rows, cols, rates, seed=seed)
    scale_rows = torch.rand(rows, dtype=torch.float16) + 0.5
    unit = kg.prepare_from_parsed(_Parsed(body, rates, codes, scale_rows, 0.75), M=M)
    tile, scale = kg.decode_fp8(unit)
    return unit, tile.view(torch.float8_e4m3fn).float(), scale


def _compiled_dynamic_m(fn):
    """``fn`` compiled whole-graph, and an ``x`` factory that marks the token
    dim UNBACKED -- the strict spelling of what vLLM's compiled forward hands
    a route.

    ``mark_dynamic`` is not enough to pin the property: Dynamo's own 0/1
    specialisation adds a ``2 <= x.size()[0]`` guard the user code never
    wrote (the recompile message says so itself and names ``mark_unbacked``
    as the way out), so M=1 would recompile on any kernel.  vLLM's serve runs
    the code traced at its warmup shape with guards disabled
    (``vllm-compiled-forward-breaks-lane-hot-paths``, item 6), so the graph
    has to be valid at M=1 with no branch on M at all -- which is exactly what
    an unbacked size enforces: a Python branch on it fails the trace instead
    of guarding.
    """
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    def x_for(M, cols, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(M, cols, device="cuda", generator=g).bfloat16()
        torch._dynamo.decorators.mark_unbacked(x, 0)
        return x

    return compiled, x_for


@cuda
def test_the_gemv_survives_a_compiled_forward_with_a_dynamic_token_dim():
    """One graph serves every M in 1..8.

    ``fullgraph=True`` (nothing breaks the graph), the token dim unbacked (a
    Python branch on it fails the trace, the way a marked-dynamic one raised
    ``ConstraintViolationError`` at vLLM engine start on 2026-09-02), and
    after the trace ``error_on_recompile`` (a guard that specialised M would
    recompile at the next batch).  Values are held to the fp32 accumulation bound, not to
    ``equal``: the GEMV retires partials by ``atomicAdd`` in whatever order
    the blocks finish, so two launches of the same kernel can already differ
    in the last fp32 bit.
    """
    kg = _kg()
    unit, w, scale = _mixed_unit(kg)
    compiled, x_for = _compiled_dynamic_m(lambda x: kg.window_gemv(unit, x))
    # trace at M=2 (a warmup-shaped batch, as vLLM traces), then every other M
    # through the same graph
    for i, M in enumerate((2, 1, 3, 4, 5, 8)):
        x = x_for(M, unit.cols, seed=100 + M)
        with torch._dynamo.config.patch(error_on_recompile=(i > 0)):
            y = compiled(x)
        assert y.shape == (M, unit.rows) and y.dtype == torch.float32
        ref = ((w * scale[:, None]).double() @ x.double().t()).t()
        assert bool(((y.double() - ref).abs() <= _bound(w, scale, x)).all()), f"M={M}"


@cuda
def test_the_module_seam_survives_a_compiled_forward_with_a_dynamic_token_dim():
    """``window_linear`` -- what a route's ``apply()`` calls -- under the same
    contract, its bf16 output held to the fp32 bound plus one bf16 ulp of the
    reference (the bf16 rounding of a differently-ordered fp32 sum can land
    one step apart)."""
    kg = _kg()
    unit, w, scale = _mixed_unit(kg, seed=32)
    compiled, x_for = _compiled_dynamic_m(lambda x: kg.window_linear(unit, x) * 2)
    for i, M in enumerate((2, 1, 4, 8)):
        x = x_for(M, unit.cols, seed=200 + M)
        with torch._dynamo.config.patch(error_on_recompile=(i > 0)):
            y = compiled(x)
        assert y.shape == (M, unit.rows) and y.dtype == torch.bfloat16
        ref = 2 * ((w * scale[:, None]).double() @ x.double().t()).t()
        tol = 2 * _bound(w, scale, x) + 2.0 ** -7 * ref.abs()
        assert bool(((y.double() - ref).abs() <= tol).all()), f"M={M}"


@cuda
def test_the_refusals_survive_a_compiled_forward():
    """The two things the GEMV refuses -- 8 rows per lane over a rate-1
    column, and M past ``GEMV_MAX_M`` -- are refused BY NAME from inside the
    compiled graph too, at the batch that needs them: the op runs its concrete
    checks at call time, so a compiled route cannot serve them silently."""
    kg = _kg()
    rows, cols = 512, 96
    rates = tuple((1, 2, 4)[c % 3] for c in range(cols))
    body, codes = _synthetic(rows, cols, rates, seed=33)
    unit = kg.prepare_from_parsed(_Parsed(body, rates, codes, torch.ones(rows, dtype=torch.float16)))
    assert unit.serveable_keys() == (1,)
    compiled, x_for = _compiled_dynamic_m(lambda x: kg.window_gemv(unit, x))
    tile, scale = kg.decode_fp8(unit)
    w = tile.view(torch.float8_e4m3fn).float()
    for M in (2, 1):
        x = x_for(M, cols, seed=300 + M)
        y = compiled(x)
        ref = ((w * scale[:, None]).double() @ x.double().t()).t()
        assert bool(((y.double() - ref).abs() <= _bound(w, scale, x)).all())
    with torch._dynamo.config.patch(error_on_recompile=True):
        with pytest.raises(GrammarError, match="rate-1 column"):
            compiled(x_for(4, cols, seed=304))
        with pytest.raises(GrammarError, match="M=9"):
            compiled(x_for(9, cols, seed=309))


@cuda
def test_the_extension_is_resolved_at_preparation_not_on_the_first_call(monkeypatch):
    """``prepare_*`` resolves (builds or finds) the extension, so a route's
    first GEMV -- under a compiled forward, the trace itself -- never takes the
    build.  The op body still goes through ``_ext`` (a cache hit); what is
    pinned is that preparation already did."""
    kg = _kg()
    real = kg._ext
    calls = []

    def recording():
        calls.append("ext")
        return real()

    monkeypatch.setattr(kg, "_ext", recording)
    rows, cols = 512, 64
    body, _ = _synthetic(rows, cols, (4,) * cols, seed=34)
    values = (torch.randn(1 << L) * 0.02).bfloat16().cuda()
    kg.prepare_value_unit(body, (4,) * cols, L, values)
    assert calls == ["ext"], "prepare_value_unit must resolve the extension itself"
    calls.clear()
    body, codes = _synthetic(rows, cols, (4,) * cols, seed=35)
    kg.prepare_from_parsed(_Parsed(body, (4,) * cols, codes, torch.ones(rows, dtype=torch.float16)))
    assert calls == ["ext"], "prepare_from_parsed must resolve the extension itself"


@cuda
def test_item_tables_are_planned_at_preparation_not_on_the_call_path(monkeypatch):
    """After preparation the call path plans nothing: ``plan_items`` reads the
    run table back to Python, which a compiled forward cannot do.  Every M
    tile the unit can serve runs with the planner disabled; a rate-1 unit
    plans only the table it can run; ``with_plan`` replans for the new plan
    and shares a replica's tables."""
    kg = _kg()
    unit, w, scale = _mixed_unit(kg, rows=600, cols=320, seed=36)
    assert set(unit.items_by_mt) == {1, 4}
    assert unit.items_for(8) is unit.items_for(4) and unit.items_for(2) is unit.items_for(1)
    replan = unit.with_plan(kg.Plan(rpl=8, warps=8, blocks=7, cols_per_item=32, balanced=False))
    replica = replan.with_plan(replan.plan, share_from=replan)
    assert replica.items_by_mt is replan.items_by_mt and set(replan.items_by_mt) == {1, 4}
    rows, cols = 512, 48
    rates = tuple((1, 4)[c % 2] for c in range(cols))
    body, codes = _synthetic(rows, cols, rates, seed=37)
    rate_one = kg.prepare_from_parsed(_Parsed(body, rates, codes, torch.ones(rows, dtype=torch.float16)))
    assert set(rate_one.items_by_mt) == {1}

    def no_planning(*_a, **_k):
        raise AssertionError("plan_items ran on the call path")

    monkeypatch.setattr(kg, "plan_items", no_planning)
    monkeypatch.setattr(kg, "items_for", no_planning)
    for u in (unit, replan, replica):
        for M in (1, 2, 4, 8):
            x = torch.randn(M, u.cols, device="cuda").bfloat16()
            y = kg.window_gemv(u, x)
            ref = ((w * scale[:, None]).double() @ x.double().t()).t()
            assert bool(((y.double() - ref).abs() <= _bound(w, scale, x)).all())
    for M in (1, 2):
        kg.window_gemv(rate_one, torch.randn(M, cols, device="cuda").bfloat16())


@cuda
def test_the_out_instrument_accumulates_into_the_callers_buffer():
    """``out=`` is the bench's eager instrument: the same concrete launch into a
    caller-owned zeroed ``[M_tile, rows]`` buffer, the functional op's answer
    within the bound.  It stays so the receipt's timing arms keep running."""
    kg = _kg()
    unit, w, scale = _mixed_unit(kg, rows=600, cols=320, seed=38)
    x = torch.randn(3, unit.cols, device="cuda").bfloat16()
    scratch = torch.zeros(4, unit.rows, dtype=torch.float32, device="cuda")
    y = kg.window_gemv(unit, x, out=scratch)
    assert y.shape == (3, unit.rows) and y.data_ptr() == scratch.data_ptr()
    ref = ((w * scale[:, None]).double() @ x.double().t()).t()
    assert bool(((y.double() - ref).abs() <= _bound(w, scale, x)).all())
    assert torch.equal(scratch[3], torch.zeros(unit.rows, device="cuda"))   # the pad row stays zero
