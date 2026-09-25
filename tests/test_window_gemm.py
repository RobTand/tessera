"""The window-body prefill GEMM (``window_gemm``): the direct packed decode
against the state definition, both families, M tails, rate runs, the tile
lookback, the TP row-cut initial state, the prepared bundle and its graph
compatibility.

The oracle is the definition itself, ``reference_states`` + the raw value or
E4M3 byte table -- no decode lane is asked to certify this one.  Under the
default ``arithmetic="epilogue"`` the per-row fp32 scale is applied once on the
accumulated output; under ``arithmetic="folded"`` (the value family only, what
the BF16 route serves since tessera#614) each weight is ``bf16(value *
row_scale)`` before the dot, ``decode.materialize_bf16_folded``'s tile, and
nothing is applied after it.  The FP8 family adds the per-token scale from
vLLM's native quantizer, exactly the contract the route publishes.

The units here are built straight from ``repack_window_body`` (and, for one
rate case the official roster excludes, from the documented tile-word recipe),
so the tests do not depend on the CUDA extension building.
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import kernel_window_gemv as kg    # noqa: E402
from tessera import window_gemm as wg           # noqa: E402
from tessera.errors import GrammarError         # noqa: E402

import window_pack_reference as wpr             # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

L = 14


def _body(rows, cols, rates, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rate = torch.tensor(rates, dtype=torch.int64)
    return (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.uint8)


def _make(rows, cols, rates, *, seed=0):
    body = _body(rows, cols, rates, seed)
    values = (torch.randn(1 << L, generator=torch.Generator().manual_seed(seed + 1))
              * 0.03).bfloat16()
    scale = (torch.rand(rows, generator=torch.Generator().manual_seed(seed + 2)) * 2 + 0.25).cuda()
    rep = kg.repack_window_body(body.cuda(), tuple(rates))
    unit = kg.WindowGemvUnit(
        rep=rep, table=values.cuda(), scale=scale, window_bits=L,
        plan=kg.default_plan(rows, cols, 1), family="value",
    )
    return unit, body, values


def _make_fp8(rows, cols, rates, *, seed=0):
    unit, body, _ = _make(rows, cols, rates, seed=seed)
    g = torch.Generator().manual_seed(seed + 7)
    codes = torch.randint(0, 256, (1 << L,), generator=g).to(torch.uint8).cuda()
    native = torch.arange(256, dtype=torch.uint8)
    native[0x7F] = 0          # E4M3FN NaN bytes are not weights
    native[0xFF] = 0
    unit = dataclasses.replace(unit, family="e4m3", codes_of_state=codes,
                               native=native.cuda())
    return unit, body, codes, native.cuda()


def _states(body, rates, L, init=None):
    mask = (1 << L) - 1
    rate = torch.tensor(rates, dtype=torch.int64)
    bits = body.to(torch.int64)
    rows, cols = body.shape
    states = torch.empty(rows, cols, dtype=torch.int64)
    state = torch.zeros(cols, dtype=torch.int64) if init is None else init.to(torch.int64).clone()
    for t in range(rows):
        state = ((state << rate) | bits[t]) & mask
        states[t] = state
    return states


def _reference(body, values, rates, scale, x, init=None):
    raw = values.float().cuda()[_states(body, rates, L, init)]
    acc = raw @ x.float().t()                               # fp32 [rows, M]
    return (scale[:, None] * acc).t().bfloat16()            # [M, rows]


def _reference_folded(body, values, rates, scale, x, init=None):
    """``materialize_bf16_folded``'s tile -- one bf16 rounding of value * row
    scale per weight -- multiplied in fp32 and cast once."""
    raw = values.float().cuda()[_states(body, rates, L, init)]
    w = (raw * scale.float()[:, None]).to(torch.bfloat16).float()
    return (x.float() @ w.t()).bfloat16()                   # [M, rows]


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _reference_fp8(body, codes, native, rates, scale, x, init=None):
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
    require_native_fp8_quant("window_gemm test reference")
    states = _states(body, rates, L, init)
    byte = native[codes[states].long()]                       # [rows, cols] u8
    w = byte.view(torch.float8_e4m3fn).float()
    xq, a = native_fp8_quant(x)                              # [M,K] fp8, [M,1] fp32
    acc = xq.float() @ w.t()                                 # fp32 [M, rows]
    return ((acc * a) * scale[None, :]).bfloat16()


def _tol(ref):
    return 5e-3 + 1e-2 * float(ref.float().abs().max())


def _mixed_rates(cols):
    return tuple(1 if c % 5 == 0 else (2 if c % 3 else 4) for c in range(cols))


# --- BF16 (value family) -------------------------------------------------------


@cuda
def test_window_gemm_matches_the_state_definition_m128():
    rows, cols = 1024, 256
    rates = _mixed_rates(cols)
    unit, body, values = _make(rows, cols, rates, seed=3)
    x = torch.randn(128, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, block_m=64, block_n=64, block_k=64)
    assert y.shape == (128, rows) and y.dtype == torch.bfloat16
    ref = _reference(body, values, rates, unit.scale, x)
    assert float((y.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_window_gemm_m32_crosses_the_tile_lookback():
    """rows 768: the second tile's N blocks start at local row 0, where the
    window crosses the previous tile -- the lookback the CUDA lane 0 performs."""
    rows, cols = 768, 192
    rates = tuple(2 if c % 4 == 0 else 4 for c in range(cols))
    unit, body, values = _make(rows, cols, rates, seed=7)
    x = torch.randn(32, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, block_m=32, block_n=64, block_k=64)
    ref = _reference(body, values, rates, unit.scale, x)
    diff = (y.float() - ref.float()).abs()
    assert float(diff.max()) < _tol(ref)
    assert float(diff[:, 8 * 64:].max()) < _tol(ref)     # the cross-tile blocks


@cuda
def test_window_gemm_m_tails_are_masked_not_refused():
    """M = 0, 1, 8, 9, 15, 17: tails are masked, no multiple-of-16 rule."""
    rows, cols = 768, 192
    rates = tuple(2 if c % 4 == 0 else 4 for c in range(cols))
    unit, body, values = _make(rows, cols, rates, seed=17)
    for m in (1, 8, 9, 15, 17):
        x = torch.randn(m, cols, device="cuda").bfloat16()
        y = wg.window_gemm(unit, x, block_m=16 if m <= 16 else 32,
                           block_n=64, block_k=64)
        assert y.shape == (m, rows)
        ref = _reference(body, values, rates, unit.scale, x)
        assert float((y.float() - ref.float()).abs().max()) < _tol(ref), f"M={m}"
    empty = torch.randn(0, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, empty, block_m=16, block_n=64, block_k=64)
    assert y.shape == (0, rows)


@cuda
def test_window_gemm_tp_row_cut_consumes_initial_state():
    """A row cut's history seeds the windows from local row 0 on: the same
    body with the parent's state before row 0 must decode as the seeded
    definition, not as a zero start."""
    rows, cols = 768, 192
    rates = tuple(4 if c % 3 else 2 for c in range(cols))
    unit, body, values = _make(rows, cols, rates, seed=23)
    init = torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(99),
                         dtype=torch.int32)
    x = torch.randn(32, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, initial_state=init, block_m=32, block_n=64, block_k=64)
    ref = _reference(body, values, rates, unit.scale, x, init=init)
    assert float((y.float() - ref.float()).abs().max()) < _tol(ref)
    # and the same unit through the bundle's own field
    unit.rep.initial_state = init
    y2 = wg.window_gemm(unit, x, block_m=32, block_n=64, block_k=64)
    assert torch.equal(y2, y)
    # a zero start is NOT this answer
    zero = wg.window_gemm(unit, x, initial_state=torch.zeros(cols, dtype=torch.int32),
                          block_m=32, block_n=64, block_k=64)
    assert not torch.equal(zero, y)


@cuda
def test_window_gemm_one_hot_is_exact_at_every_position():
    rows, cols = 1024, 256
    rates = _mixed_rates(cols)
    unit, body, values = _make(rows, cols, rates, seed=11)
    raw = values.float().cuda()[_states(body, rates, L)]
    for j in (0, cols // 2, cols - 1):
        onehot = torch.zeros(32, cols, device="cuda", dtype=torch.bfloat16)
        onehot[:, j] = 1
        y = wg.window_gemm(unit, onehot, block_m=32, block_n=64, block_k=64)
        ref = (raw[:, j] * unit.scale).bfloat16()
        assert torch.equal(y, ref.unsqueeze(0).expand(32, rows).contiguous()), f"column {j}"


@cuda
def test_window_gemm_mixed_rate_columns_use_their_own_activations():
    """Regression: a later rate run must read its own x columns and its own
    history, not the first run's (the run-relative index was passed to both)."""
    rows, cols = 600, 32
    rates = tuple(1 if c % 5 == 0 else (2 if c % 3 else 4) for c in range(cols))
    body = _body(rows, cols, rates, seed=41)
    rep = kg.repack_window_body(body.cuda(), rates)
    values = (torch.arange(1 << L) & 127).to(torch.bfloat16).cuda()
    unit = kg.WindowGemvUnit(rep=rep, table=values, scale=torch.ones(rows, device="cuda"),
                             window_bits=L, plan=kg.default_plan(rows, cols, 1), family="value")
    x = torch.eye(cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, block_m=16, block_n=64, block_k=64)
    assert torch.equal(y.long().t(), (_states(body, rates, L) & 127).cuda())


# --- FP8 (e4m3 family) ---------------------------------------------------------


# --- BF16 folded arithmetic (tessera#614) ----------------------------------------


@cuda
@pytest.mark.parametrize("m", [1, 17, 64, 128])
def test_window_gemm_folded_is_the_folded_tile_and_not_the_epilogue(m):
    """``arithmetic="folded"`` IS ``x @ bf16(values * scale)^T`` -- the same
    exact bf16 products summed in fp32, one cast -- and it is not the epilogue
    arithmetic: the two differ on nontrivial scales, and each is closer to its
    own definition than to the other's."""
    rows, cols = 1024, 256
    rates = _mixed_rates(cols)
    unit, body, values = _make(rows, cols, rates, seed=61)
    folded = wg.prepare_window_gemm(unit, block_m=64, block_n=64, block_k=64,
                                    arithmetic="folded")
    epilogue = wg.prepare_window_gemm(unit, block_m=64, block_n=64, block_k=64)
    assert (folded.arithmetic, epilogue.arithmetic) == ("folded", "epilogue")
    x = torch.randn(m, cols, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(m)).bfloat16()
    y_f, y_e = folded(x), epilogue(x)
    ref_f = _reference_folded(body, values, rates, unit.scale, x)
    ref_e = _reference(body, values, rates, unit.scale, x)
    # The products are exact in fp32 on both sides, so only the summation order
    # differs: at most one bf16 ulp, never the fold's own rounding.
    gap = (y_f.float() - ref_f.float()).abs()
    bound = ref_f.float().abs() * 2 ** -7 + 1e-4 * float(ref_f.float().abs().max())
    assert bool((gap <= bound).all()), float((gap - bound).max())
    assert float((y_e.float() - ref_e.float()).abs().max()) < _tol(ref_e)
    assert _rel(y_f, ref_f) < _rel(y_f, ref_e)
    assert _rel(y_e, ref_e) < _rel(y_e, ref_f)
    assert not torch.equal(y_f, y_e), "folded and epilogue must differ on nontrivial scales"
    # The one-shot helper takes the same argument.
    assert torch.equal(wg.window_gemm(unit, x, block_m=64, block_n=64, block_k=64,
                                      arithmetic="folded"), y_f)


@cuda
def test_window_gemm_folded_tp_row_cut_consumes_initial_state():
    """The fold multiplies the decoded weight, so a row cut's history must
    reach it exactly as it reaches the epilogue form."""
    rows, cols = 768, 192
    rates = tuple(4 if c % 3 else 2 for c in range(cols))
    unit, body, values = _make(rows, cols, rates, seed=67)
    init = torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(98),
                         dtype=torch.int32)
    x = torch.randn(32, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, initial_state=init, block_m=32, block_n=64, block_k=64,
                       arithmetic="folded")
    ref = _reference_folded(body, values, rates, unit.scale, x, init=init)
    assert float((y.float() - ref.float()).abs().max()) < _tol(ref)
    zero = _reference_folded(body, values, rates, unit.scale, x)
    assert _rel(y, ref) < _rel(y, zero)


@cuda
def test_dense_folded_is_bit_identical_to_a_one_expert_grouped_folded_stack():
    """One definition of the fold, in both kernels: the dense GEMM and the
    grouped (routed) GEMM run the same register expression on the same decoded
    tile, so on one expert, one route and the same blocks they agree bit for
    bit.  This is what makes the BF16 route's dense and routed modules one
    arithmetic rather than two that happen to be close."""
    from tessera import window_gemm_grouped as wgg

    rows, cols = 768, 192
    unit, _body_, _values = _make(rows, cols, _mixed_rates(cols), seed=71)
    dense = wg.prepare_window_gemm(unit, block_m=64, block_n=64, block_k=64,
                                   arithmetic="folded")
    grouped = wgg.prepare_grouped_window_gemm([unit], block_m=64, block_n=64, block_k=64,
                                              arithmetic="folded")
    t = 48
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.zeros(t, 1, dtype=torch.int32, device="cuda")
    rw = torch.ones(t, 1, device="cuda")
    route = grouped(x, ids, rw, preserve=True)
    assert torch.equal(route.reshape(t, rows), dense(x))


@cuda
def test_folded_arithmetic_refusals_are_by_name():
    rows, cols = 256, 64
    unit, _b, _v = _make(rows, cols, _mixed_rates(cols), seed=73)
    with pytest.raises(GrammarError, match="unknown weight arithmetic"):
        wg.prepare_window_gemm(unit, arithmetic="fold")
    fp8_unit, *_ = _make_fp8(rows, cols, _mixed_rates(cols), seed=74)
    with pytest.raises(GrammarError, match="folded"):
        wg.prepare_window_gemm(fp8_unit, quantizer=None, arithmetic="folded")
    # The bundle states its arithmetic itself, so a rebuilt one (the serving
    # custom op rebuilds it per call) cannot carry a combination prepare refuses.
    bundle = wg.prepare_window_gemm(unit, arithmetic="folded")
    with pytest.raises(GrammarError, match="folded"):
        dataclasses.replace(bundle, family="e4m3")
    with pytest.raises(GrammarError, match="unknown weight arithmetic"):
        dataclasses.replace(bundle, arithmetic="both")


@cuda
def test_window_gemm_fp8_matches_scaled_reference_m128():
    rows, cols = 1024, 256
    rates = _mixed_rates(cols)
    unit, body, codes, native = _make_fp8(rows, cols, rates, seed=3)
    x = torch.randn(128, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, block_m=64, block_n=64, block_k=64)
    assert y.shape == (128, rows) and y.dtype == torch.bfloat16
    ref = _reference_fp8(body, codes, native, rates, unit.scale, x)
    assert float((y.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_window_gemm_fp8_m_tails_and_rates():
    rows, cols = 768, 192
    rates = tuple(1 if c % 7 == 0 else (2 if c % 4 else 4) for c in range(cols))
    unit, body, codes, native = _make_fp8(rows, cols, rates, seed=19)
    for m in (0, 1, 8, 9, 15, 16, 17, 32):
        x = torch.randn(m, cols, device="cuda").bfloat16()
        y = wg.window_gemm(unit, x, block_m=16 if m <= 16 else 32,
                           block_n=64, block_k=64)
        assert y.shape == (m, rows)
        if m == 0:
            continue
        ref = _reference_fp8(body, codes, native, rates, unit.scale, x)
        assert float((y.float() - ref.float()).abs().max()) < _tol(ref), f"M={m}"


@cuda
def test_window_gemm_fp8_tp_row_cut_initial_state():
    rows, cols = 768, 192
    rates = tuple(4 if c % 3 else 2 for c in range(cols))
    unit, body, codes, native = _make_fp8(rows, cols, rates, seed=29)
    init = torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(7),
                         dtype=torch.int32)
    x = torch.randn(32, cols, device="cuda").bfloat16()
    y = wg.window_gemm(unit, x, initial_state=init, block_m=32, block_n=64, block_k=64)
    ref = _reference_fp8(body, codes, native, rates, unit.scale, x, init=init)
    assert float((y.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_prepared_fp8_accepts_prequantized_activations():
    """The same answer from the already-quantized operands: the contract is
    the quantizer's, and a bare fp8 tensor without its scale is refused."""
    from tessera.serving.native_ops import native_fp8_quant
    rows, cols = 768, 192
    unit, body, codes, native = _make_fp8(rows, cols, (4,) * cols, seed=31)
    prepared = wg.prepare_window_gemm(unit, block_m=32, block_n=64, block_k=64)
    x = torch.randn(32, cols, device="cuda").bfloat16()
    y = prepared(x)
    xq, a = native_fp8_quant(x)
    y2 = prepared(xq, a_scale=a)
    assert torch.equal(y, y2)
    xq_perm = xq.index_select(1, unit.rep.perm.long()).contiguous()
    with pytest.raises(GrammarError, match="activation scale"):
        prepared(xq_perm, a_scale=None)
    compute_only = wg.prepare_window_gemm(unit, block_m=32, block_n=64, block_k=64,
                                          quantizer=None)
    assert torch.equal(compute_only(xq, a_scale=a), y)
    with pytest.raises(GrammarError, match="without a quantizer"):
        compute_only(x)


# --- prepared bundle: constants, graph capture, boundaries --------------------


@cuda
def test_prepared_call_reuses_constants_and_captures_in_a_graph():
    """The hot call must not rebuild the constant bundle, and must be
    capturable: no tensor-content validation, no host sync in the call."""
    rows, cols = 768, 192
    rates = tuple(2 if c % 4 == 0 else 4 for c in range(cols))
    unit, _, _ = _make(rows, cols, rates, seed=43)
    prepared = wg.prepare_window_gemm(unit, block_m=32, block_n=64, block_k=64)
    before = (prepared.table.data_ptr(), prepared.scale.data_ptr(),
              prepared.words.data_ptr(), prepared.init_perm.data_ptr(),
              prepared.runs.data_ptr())
    x = torch.randn(32, cols, device="cuda").bfloat16()
    y = prepared(x)
    after = (prepared.table.data_ptr(), prepared.scale.data_ptr(),
             prepared.words.data_ptr(), prepared.init_perm.data_ptr(),
             prepared.runs.data_ptr())
    assert before == after, "the constants were rebuilt across calls"

    graph = torch.cuda.CUDAGraph()
    out = torch.empty_like(y)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            prepared(x)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        prepared(x, out=out)
    graph.replay()
    assert torch.equal(out, y)


@cuda
def test_prepared_fp8_prequantized_call_captures_in_a_graph():
    from tessera.serving.native_ops import native_fp8_quant
    rows, cols = 768, 192
    unit, _, _, _ = _make_fp8(rows, cols, (4,) * cols, seed=47)
    prepared = wg.prepare_window_gemm(unit, block_m=32, block_n=64, block_k=64)
    x = torch.randn(32, cols, device="cuda").bfloat16()
    xq, a = native_fp8_quant(x)
    xq = xq.index_select(1, unit.rep.perm.long()).contiguous()
    y = prepared(xq, a_scale=a)
    graph = torch.cuda.CUDAGraph()
    out = torch.empty_like(y)
    with torch.cuda.graph(graph):
        prepared(xq, a_scale=a, out=out)
    graph.replay()
    assert torch.equal(out, y)


@cuda
def test_prepare_refusals_are_at_the_boundary():
    rows, cols = 512, 128
    unit, _, _ = _make(rows, cols, (4,) * cols, seed=13)
    x = torch.randn(16, cols, device="cuda").bfloat16()

    with pytest.raises(GrammarError, match="power of two"):
        wg.prepare_window_gemm(unit, block_n=48)
    with pytest.raises(GrammarError, match="divide"):
        wg.prepare_window_gemm(unit, block_n=1024)
    with pytest.raises(GrammarError, match="window states"):
        wg.prepare_window_gemm(unit, initial_state=torch.full((cols,), 1 << L, dtype=torch.int32))
    with pytest.raises(GrammarError, match="initial_state has"):
        wg.prepare_window_gemm(unit, initial_state=torch.zeros(cols - 1, dtype=torch.int32))
    lossy = dataclasses.replace(unit, table=torch.rand(1 << L, device="cuda"))
    with pytest.raises(GrammarError, match="bf16"):
        wg.prepare_window_gemm(lossy)

    prepared = wg.prepare_window_gemm(unit, block_m=16)
    with pytest.raises(GrammarError, match="bf16"):
        prepared(torch.randn(32, cols, device="cuda"))
    with pytest.raises(GrammarError, match="tensor on"):
        prepared(torch.randn(32, cols + 16, device="cuda").bfloat16())
    e4m3 = dataclasses.replace(
        unit, family="e4m3",
        codes_of_state=torch.zeros(1 << L, dtype=torch.uint8, device="cuda"),
        native=torch.zeros(256, dtype=torch.uint8, device="cuda"))
    with pytest.raises(GrammarError, match="activation scale"):
        wg.prepare_window_gemm(e4m3)(torch.zeros(32, cols, device="cuda").to(torch.float8_e4m3fn))


@cuda
def test_prepared_refuses_noncontiguous_out_and_scale():
    """The hot path assumes contiguous operands; a strided out or activation
    scale is refused at the API boundary, not misaddressed."""
    rows, cols = 512, 128
    unit, _, _ = _make(rows, cols, (4,) * cols, seed=57)
    prepared = wg.prepare_window_gemm(unit, block_m=16)
    x = torch.randn(16, cols, device="cuda").bfloat16()
    wide = torch.empty(16, rows * 2, dtype=torch.bfloat16, device="cuda")[:, ::2]
    assert not wide.is_contiguous()
    with pytest.raises(GrammarError, match="contiguous"):
        prepared(x, out=wide)

    e4m3 = dataclasses.replace(
        unit, family="e4m3",
        codes_of_state=torch.zeros(1 << L, dtype=torch.uint8, device="cuda"),
        native=torch.zeros(256, dtype=torch.uint8, device="cuda"))
    p8 = wg.prepare_window_gemm(e4m3, block_m=16)
    xq = torch.zeros(16, cols, device="cuda").to(torch.float8_e4m3fn)
    a_wide = torch.rand(16 * 2, device="cuda")[::2]
    assert not a_wide.is_contiguous()
    with pytest.raises(GrammarError, match="contiguous"):
        p8(xq, a_scale=a_wide)


@cuda
def test_prepare_freezes_strided_constants_contiguously():
    """The constants may arrive as strided views (they do not normally, but
    the API does not say so); preparation freezes contiguous copies and the
    result is bit-identical to the contiguous unit's."""
    rows, cols = 768, 192
    unit, _, _ = _make(rows, cols, (4,) * cols, seed=59)
    wide_scale = torch.empty(rows * 2, device="cuda")
    wide_scale[::2] = unit.scale
    wide_scale[1::2] = 0.0
    strided_scale = wide_scale[::2]
    assert not strided_scale.is_contiguous()
    wide_words = torch.empty(unit.rep.words.numel() * 2, dtype=torch.int32, device="cuda")
    wide_words[::2] = unit.rep.words
    wide_words[1::2] = 0
    strided_words = wide_words[::2]
    assert not strided_words.is_contiguous()
    rep = dataclasses.replace(unit.rep, words=strided_words)
    unit2 = dataclasses.replace(unit, rep=rep, scale=strided_scale)

    prepared = wg.prepare_window_gemm(unit2, block_m=32, block_n=64, block_k=64)
    assert prepared.words.is_contiguous() and prepared.scale.is_contiguous()
    x = torch.randn(32, cols, device="cuda").bfloat16()
    ref = wg.window_gemm(unit, x, block_m=32, block_n=64, block_k=64)
    assert torch.equal(prepared(x), ref)


# --- rate roster boundary ------------------------------------------------------


@cuda
def test_window_gemm_decodes_every_rate_through_the_bitstream_layout():
    """The layout is bit-exact for every rate 1..8 (512*R bits is always whole
    words); only the legacy ``repack_window_body`` byte step is restricted.
    The independent bitstream packer agrees with it wherever it can run, and
    the kernel decodes every rate exactly against the definition."""
    body = _body(600, 32, (4,) * 32, seed=53)
    values = (torch.arange(1 << L) & 127).to(torch.bfloat16).cuda()
    x = torch.eye(32, device="cuda").bfloat16()

    official = kg.repack_window_body(body.cuda(), (4,) * 32)
    reference4 = wpr.pack_bitstream(body, (4,) * 32)
    assert torch.equal(reference4.words, official.words.cpu())
    assert torch.equal(reference4.runs, official.runs.cpu())
    assert torch.equal(reference4.perm, official.perm.cpu())

    for bad in (3, 5, 6, 7):
        # the LEGACY implementation cannot pack these (its 8//rate byte step);
        # the layout itself is fine, so the test packs them independently
        with pytest.raises(GrammarError, match="no lane"):
            kg.repack_window_body(body.cuda(), (bad,) * 32)
    for rate in range(1, 9):
        rates = (rate,) * 32
        body_r = _body(600, 32, rates, seed=53)
        rep = wpr.pack_bitstream(body_r, rates)
        rep = dataclasses.replace(rep, runs=rep.runs.cuda(), words=rep.words.cuda(),
                                  perm=rep.perm.cuda())
        unit = kg.WindowGemvUnit(rep=rep, table=values, scale=torch.ones(600, device="cuda"),
                                 window_bits=L, plan=kg.default_plan(600, 32, 1),
                                 family="value")
        y = wg.window_gemm(unit, x, block_m=16, block_n=64, block_k=64)
        assert torch.equal(y.long().t(), (_states(body_r, rates, L) & 127).cuda()), f"rate {rate}"
