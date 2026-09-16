"""The window-body prefill GEMM (``window_gemm``): the direct packed decode
against the state definition, M tails, rate runs, the tile lookback and the
TP row-cut initial state.

The oracle is the definition itself, ``reference_states`` + the raw value
table -- no decode lane is asked to certify this one.  The per-row fp32 scale
is applied once on the accumulated output, the epilogue ``decode_values``
documents; a one-hot ``x`` is exact at every row and column.

The units here are built straight from ``repack_window_body``, exactly the
raw-bits path the existing GEMV tests use, so the test does not depend on the
CUDA extension building.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import kernel_window_gemv as kg    # noqa: E402
from tessera import window_gemm as wg           # noqa: E402
from tessera.errors import GrammarError         # noqa: E402

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


def _tol(ref):
    return 5e-3 + 1e-2 * float(ref.float().abs().max())


def _mixed_rates(cols):
    return tuple(1 if c % 5 == 0 else (2 if c % 3 else 4) for c in range(cols))


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


@cuda
def test_window_gemm_refusals():
    rows, cols = 512, 128
    unit, _, _ = _make(rows, cols, (4,) * cols, seed=13)
    x = torch.randn(16, cols, device="cuda").bfloat16()

    with pytest.raises(GrammarError, match="bf16"):
        wg.window_gemm(unit, torch.randn(32, cols, device="cuda"))
    with pytest.raises(GrammarError, match="features"):
        wg.window_gemm(unit, torch.randn(32, cols + 16, device="cuda").bfloat16())
    with pytest.raises(GrammarError, match="power of two"):
        wg.window_gemm(unit, x, block_n=48)
    with pytest.raises(GrammarError, match="divide"):
        wg.window_gemm(unit, x, block_n=1024)
    with pytest.raises(GrammarError, match="window states"):
        wg.window_gemm(unit, x, initial_state=torch.full((cols,), 1 << L, dtype=torch.int32))
    with pytest.raises(GrammarError, match="initial_state has"):
        wg.window_gemm(unit, x, initial_state=torch.zeros(cols - 1, dtype=torch.int32))

    import dataclasses
    fp8 = dataclasses.replace(unit, family="e4m3", codes_of_state=torch.zeros(1 << L, dtype=torch.uint8))
    with pytest.raises(GrammarError, match="E4M3"):
        wg.window_gemm(fp8, x)
    lossy = dataclasses.replace(unit, table=torch.rand(1 << L, device="cuda"))
    with pytest.raises(GrammarError, match="bf16"):
        wg.window_gemm(lossy, x)
