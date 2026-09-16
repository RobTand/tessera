"""The grouped window GEMM: routing parity against an independent per-expert
oracle, device-side routing, both families, empty and repeated routes, M
tails, per-expert history and rates, and graph capture.

The oracle loops experts on the host (allowed here, never in the kernel):
each expert's weights are the definition -- ``reference_states`` through its
own table, or the E4M3 bytes through its own codes/native tables -- and the
routing is applied to the per-expert reference outputs exactly as the
semantics state.  Packed streams for rates the legacy repacker cannot emit
come from the independent bitstream packer.
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import kernel_window_gemv as kg     # noqa: E402
from tessera import window_gemm_grouped as wgg   # noqa: E402
from tessera.errors import GrammarError          # noqa: E402

import window_pack_reference as wpr              # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")

L = 14


def _body(rows, cols, rates, seed):
    g = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates, dtype=torch.int64)
    return (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.uint8)


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


class Expert:
    """One stacked expert's definition-side facts and its unit."""

    def __init__(self, rows, cols, rates, seed, *, family="value", init=None, device="cuda"):
        self.rows, self.cols, self.rates = rows, cols, tuple(rates)
        self.body = _body(rows, cols, self.rates, seed)
        g = torch.Generator().manual_seed(seed + 101)
        self.values = (torch.randn(1 << L, generator=g) * 0.03).bfloat16()
        self.scale = (torch.rand(rows, generator=torch.Generator().manual_seed(seed + 202)) * 2
                      + 0.25).to(device)
        self.init = init
        rep = wpr.pack_bitstream(self.body, self.rates)
        rep = dataclasses.replace(rep, runs=rep.runs.to(device), words=rep.words.to(device),
                                  perm=rep.perm.to(device))
        if init is not None:
            rep.initial_state = init
        kwargs = {}
        if family == "e4m3":
            codes = torch.randint(0, 256, (1 << L,), generator=g).to(torch.uint8).to(device)
            native = torch.arange(256, dtype=torch.uint8)
            native[0x7F] = 0
            native[0xFF] = 0
            kwargs = {"codes_of_state": codes, "native": native.to(device)}
        self.unit = kg.WindowGemvUnit(
            rep=rep, table=self.values.to(device), scale=self.scale, window_bits=L,
            plan=kg.default_plan(rows, cols, 1), family=family, **kwargs,
        )
        self.states = _states(self.body, self.rates, L, init)

    def reference(self, x, family, folded=False):
        """The per-expert dense definition: fp32 [T, rows], no routing.

        ``folded=True`` is the research BF16 contract:
        ``bf16(values * row_scale)`` before the dot, no epilogue scale --
        exactly ``bf16_route.decode_folded``'s arithmetic.
        """
        if family == "e4m3":
            byte = self.unit.native[self.unit.codes_of_state[self.states.cuda()].long()]
            w = byte.view(torch.float8_e4m3fn).float()
        else:
            w = self.values.float().cuda()[self.states.cuda()]
            if folded:
                w = (w * self.scale[:, None]).bfloat16().float()
                acc = x.float() @ w.t()
                return acc
        acc = x.float() @ w.t()
        return acc * self.scale[None, :]


def _quant(x):
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
    require_native_fp8_quant("grouped test")
    return native_fp8_quant(x)


def _oracle(experts, x, ids, rw, family):
    """Per-expert references (host loop over E only), then the routing sum."""
    y = []
    if family == "e4m3":
        x_fp8, a = _quant(x)
        xq = x_fp8.float()
    else:
        xq, a = x, None
    for e in experts:
        y_e = e.reference(xq, family)
        if family == "e4m3":
            y_e = y_e * a
        y.append(y_e)
    stacked = torch.stack(y)                                  # [E, T, rows]
    t_tokens = ids.shape[0]
    rows_sel = torch.arange(t_tokens, device=ids.device)[:, None].expand_as(ids)
    picked = stacked[ids.long().clamp(0, len(experts) - 1), rows_sel]   # [T, K, rows]
    return (picked * rw[..., None]).sum(1).bfloat16()


def _stack(rows, cols, family, seeds, *, rates=None):
    experts = []
    for i, seed in enumerate(seeds):
        r = rates[i] if rates is not None else tuple(
            1 if c % 5 == 0 else (2 if c % 3 else 4) for c in range(cols))
        experts.append(Expert(rows, cols, r, seed, family=family))
    return experts


def _tol(ref):
    return 5e-3 + 1e-2 * float(ref.float().abs().max())


@cuda
def test_grouped_bf16_matches_the_per_expert_oracle():
    rows, cols, experts = 768, 192, 4
    rates = [
        tuple(3 if c % 4 == 0 else 4 for c in range(cols)),      # rate 3: legacy repacker cannot pack
        tuple(5 if c % 3 == 0 else 2 for c in range(cols)),
        tuple(1 if c % 2 else 7 for c in range(cols)),
        tuple(8 if c % 5 == 0 else 4 for c in range(cols)),
    ]
    stack = _stack(rows, cols, "value", [11, 12, 13, 14], rates=rates)
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    t, k = 32, 2
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    out = prepared(x, ids, rw)
    ref = _oracle(stack, x, ids, rw, "value")
    assert out.shape == (t, rows) and out.dtype == torch.bfloat16
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_grouped_fp8_matches_the_per_expert_oracle():
    rows, cols, experts = 768, 192, 3
    rates = [
        tuple(3 if c % 4 == 0 else 4 for c in range(cols)),
        tuple(2 if c % 3 else 5 for c in range(cols)),
        (4,) * cols,
    ]
    stack = _stack(rows, cols, "e4m3", [21, 22, 23], rates=rates)
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    t, k = 32, 2
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    out = prepared(x, ids, rw)
    ref = _oracle(stack, x, ids, rw, "e4m3")
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_grouped_m_tails():
    rows, cols, experts = 768, 192, 4
    stack = _stack(rows, cols, "value", [31, 32, 33, 34])
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    for t in (1, 8, 32, 128):
        x = torch.randn(t, cols, device="cuda").bfloat16()
        ids = torch.randint(0, experts, (t, 2), device="cuda", dtype=torch.int32)
        rw = torch.rand(t, 2, device="cuda")
        out = prepared(x, ids, rw)
        ref = _oracle(stack, x, ids, rw, "value")
        assert out.shape == (t, rows)
        assert float((out.float() - ref.float()).abs().max()) < _tol(ref), f"T={t}"
    empty = torch.randn(0, cols, device="cuda").bfloat16()
    out = prepared(empty, torch.zeros(0, 2, device="cuda", dtype=torch.int32),
                   torch.zeros(0, 2, device="cuda"))
    assert out.shape == (0, rows)


@cuda
def test_grouped_empty_experts_and_repeated_ids():
    rows, cols, experts = 768, 192, 4
    stack = _stack(rows, cols, "value", [41, 42, 43, 44])
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    t, k = 16, 2

    # experts 0 and 2 are never routed: computed, masked out
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.tensor([[1, 3]] * t, device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    out = prepared(x, ids, rw)
    ref = _oracle(stack, x, ids, rw, "value")
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)

    # one token routes the same expert twice; another repeats across tokens
    ids = torch.zeros(t, k, device="cuda", dtype=torch.int32)
    ids[:, 1] = 2
    ids[0] = torch.tensor([1, 1], device="cuda")
    ids[1] = torch.tensor([3, 3], device="cuda")
    rw = torch.rand(t, k, device="cuda")
    out = prepared(x, ids, rw)
    ref = _oracle(stack, x, ids, rw, "value")
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)

    # a zero route weight contributes nothing
    rw = torch.rand(t, k, device="cuda")
    rw[:, 0] = 0.0
    out = prepared(x, ids, rw)
    ref = _oracle(stack, x, ids, rw, "value")
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_grouped_per_expert_history_and_row_cut():
    """Every expert carries its own row-cut state: one zero-start, the rest
    seeded differently -- the TP2 row-cut case, oracle-seeded."""
    rows, cols, experts = 768, 192, 3
    inits = [
        None,
        torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(5),
                      dtype=torch.int32),
        torch.randint(0, 1 << L, (cols,), generator=torch.Generator().manual_seed(6),
                      dtype=torch.int32),
    ]
    stack = [Expert(rows, cols, (4,) * cols, 51 + i, init=inits[i]) for i in range(experts)]
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    t, k = 24, 2
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    out = prepared(x, ids, rw)
    ref = _oracle(stack, x, ids, rw, "value")
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)


@cuda
def test_grouped_routing_predicate_and_refusals():
    rows, cols = 512, 128
    stack = _stack(rows, cols, "value", [61, 62])
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    ok = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int32)
    assert bool(wgg.routing_ids_ok(ok, 2))
    bad = torch.tensor([[0, 2], [1, 0]], device="cuda", dtype=torch.int32)
    assert not bool(wgg.routing_ids_ok(bad, 2))

    with pytest.raises(GrammarError, match="at least one"):
        wgg.prepare_grouped_window_gemm([])
    with pytest.raises(GrammarError, match="homogeneous"):
        wgg.prepare_grouped_window_gemm(
            [e.unit for e in stack]
            + [Expert(rows, cols + 16, (4,) * (cols + 16), 99).unit])
    x = torch.randn(8, cols, device="cuda").bfloat16()
    ids = torch.zeros(8, 2, device="cuda", dtype=torch.int32)
    rw = torch.zeros(8, 2, device="cuda")
    with pytest.raises(GrammarError, match="bf16"):
        prepared(torch.randn(8, cols, device="cuda"), ids, rw)
    with pytest.raises(GrammarError, match="share"):
        prepared(x, ids, rw[:, :1])
    x_nc = torch.randn(8, cols * 2, device="cuda").bfloat16()[:, ::2]
    assert not x_nc.is_contiguous()
    with pytest.raises(GrammarError, match="contiguous"):
        prepared(x_nc, ids, rw)
    with pytest.raises(GrammarError, match="rows"):
        prepared(x[:4], ids, rw)
    with pytest.raises(GrammarError, match="route-indexed"):
        prepared(x, ids, rw, route_input=True)
    with pytest.raises(GrammarError, match="two different stages"):
        prepared(x, ids, rw, preserve=True, route_input=True)
    wide = torch.empty(8, rows * 2, dtype=torch.float32, device="cuda")[:, ::2]
    assert not wide.is_contiguous()
    with pytest.raises(GrammarError, match="contiguous"):
        prepared(x, ids, rw, out=wide)


@cuda
def test_grouped_two_stage_moe_matches_the_route_preserving_oracle():
    """gate/up preserved per route -> SwiGLU -> down reduced, for BOTH families.

    The oracle mirrors vLLM's placement, not an algebraic equivalent:
    ``fused_moe.py`` dispatches gemm1 with ``apply_router_weight_on_input``
    (each kernel multiplies the fp32 accumulator by the route weight when
    ``MUL_ROUTED_WEIGHT`` is set, before the activation and the next A-quant)
    and gemm2 with ``not apply_router_weight_on_input``;
    ``topk_weight_and_reduce.py`` weights the per-route outputs only when the
    input did not already carry the weights, then sums.
    """
    rows_h, cols_h, inter, experts = 768, 192, 96, 3
    t, k = 16, 2
    seeds = [81, 82, 83]
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    sel = torch.arange(t, device="cuda")[:, None].expand_as(ids)
    route_rows = torch.arange(t * k, device="cuda")

    for family, arithmetic in (("value", "epilogue"), ("e4m3", "epilogue"),
                               ("value", "folded")):
        folded = arithmetic == "folded"
        gu_stack = [Expert(2 * inter, cols_h, (4,) * cols_h, s, family=family)
                    for s in seeds]
        dn_stack = [Expert(rows_h, inter, (2 if i % 2 else 4,) * inter, s + 10,
                           family=family) for i, s in enumerate(seeds)]
        gu = wgg.prepare_grouped_window_gemm([e.unit for e in gu_stack],
                                             block_m=32, block_n=64, block_k=64,
                                             arithmetic=arithmetic)
        dn = wgg.prepare_grouped_window_gemm([e.unit for e in dn_stack],
                                             block_m=32, block_n=64, block_k=64,
                                             arithmetic=arithmetic)
        x = torch.randn(t, cols_h, device="cuda").bfloat16()
        if family == "e4m3":
            x_in, a1 = _quant(x)
            xq1 = x_in.float()
        else:
            xq1, a1 = x, None

        def reference(weight_input):
            picked1 = torch.stack([e.reference(xq1, family, folded=folded)
                                   for e in gu_stack])                           # [E,T,2I]
            if family == "e4m3":
                picked1 = picked1 * a1.reshape(1, t, 1)
            routes1 = picked1[ids.long(), sel]                                   # [T,K,2I]
            if weight_input:
                routes1 = routes1 * rw[..., None]      # gemm1 MUL_ROUTED_WEIGHT
            route = routes1.bfloat16()
            gate, up = route[..., :inter].float(), route[..., inter:].float()
            act = (torch.nn.functional.silu(gate) * up).bfloat16()               # [T,K,I]
            flat = act.reshape(t * k, inter)
            if family == "e4m3":
                a_in, a2 = _quant(flat)
                flat_q = a_in.float()
            else:
                flat_q, a2 = flat, None
            picked2 = torch.stack([e.reference(flat_q, family, folded=folded)
                                   for e in dn_stack])                           # [E,T*K,H]
            routes2 = picked2[ids.long().reshape(-1), route_rows]                    # [T*K, H]
            routes2 = routes2.reshape(t, k, rows_h)
            if family == "e4m3":
                routes2 = routes2 * a2.reshape(t, k, 1)
            if not weight_input:
                routes2 = (routes2 * rw[..., None]).bfloat16()   # gemm2 cast
            else:
                routes2 = routes2.bfloat16()
            return routes2.float().sum(1).bfloat16()             # moe_sum

        for weight_input in (False, True):
            route = gu(x, ids, rw, preserve=True,
                       apply_router_weight_on_input=weight_input)
            assert route.shape == (t, k, 2 * inter) and route.dtype == torch.bfloat16
            gate, up = route[..., :inter].float(), route[..., inter:].float()
            act = (torch.nn.functional.silu(gate) * up).bfloat16()
            out = dn(act.reshape(t * k, inter), ids, rw, route_input=True,
                     apply_router_weight_on_input=weight_input)
            ref = reference(weight_input)
            assert out.shape == (t, rows_h)
            assert float((out.float() - ref.float()).abs().max()) < _tol(ref), \
                f"{family}/{arithmetic} apply_router_weight_on_input={weight_input}"
            if family == "e4m3":
                # the per-route scale must be indexed by route, not by token:
                # the prequantized path and the internal-quantizer path agree
                xq2, a2 = _quant(act.reshape(t * k, inter))
                out2 = dn(xq2, ids, rw, a_scale=a2, route_input=True,
                          apply_router_weight_on_input=weight_input)
                assert torch.equal(out2, out), "route-indexed a_scale disagrees"


@cuda
def test_grouped_out_buffer_is_overwritten_not_accumulated():
    rows, cols, experts = 768, 192, 3
    stack = _stack(rows, cols, "value", [91, 92, 93])
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack],
                                               block_m=32, block_n=64, block_k=64)
    x = torch.randn(8, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (8, 2), device="cuda", dtype=torch.int32)
    rw = torch.rand(8, 2, device="cuda")
    fresh = prepared(x, ids, rw)
    dirty = torch.full((8, rows), 7.5, dtype=torch.float32, device="cuda")
    reused = prepared(x, ids, rw, out=dirty)
    assert torch.equal(reused, fresh)


@cuda
def test_grouped_folded_arithmetic_is_decode_folded_and_differs_from_epilogue():
    """``arithmetic="folded"`` is exactly ``decode_folded``'s
    ``bf16(value * row_scale)`` before the dot -- not the dense epilogue --
    and the two differ on nontrivial scales; FP8 refuses it."""
    rows, cols, experts = 768, 192, 3
    stack = _stack(rows, cols, "value", [101, 102, 103])
    folded = wgg.prepare_grouped_window_gemm([e.unit for e in stack],
                                             block_m=32, block_n=64, block_k=64,
                                             arithmetic="folded")
    dense = wgg.prepare_grouped_window_gemm([e.unit for e in stack],
                                            block_m=32, block_n=64, block_k=64)
    t, k = 16, 2
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    sel = torch.arange(t, device="cuda")[:, None].expand_as(ids)
    route = folded(x, ids, rw, preserve=True)
    y_e = []
    for e in stack:
        w = e.values.float().cuda()[e.states.cuda()]
        w = (w * e.scale[:, None]).bfloat16().float()        # decode_folded
        y_e.append(x.float() @ w.t())
    ref = torch.stack(y_e)[ids.long(), sel].bfloat16()
    assert float((route.float() - ref.float()).abs().max()) < _tol(ref)
    epi = dense(x, ids, rw, preserve=True)
    assert not torch.allclose(route.float(), epi.float(), rtol=1e-3, atol=1e-5), \
        "folded and epilogue arithmetic must differ on nontrivial scales"
    with pytest.raises(GrammarError, match="folded"):
        wgg.prepare_grouped_window_gemm(
            [e.unit for e in _stack(rows, cols, "e4m3", [111, 112])],
            arithmetic="folded")
    with pytest.raises(GrammarError, match="unknown weight arithmetic"):
        wgg.prepare_grouped_window_gemm([e.unit for e in stack], arithmetic="fold")


@cuda
def test_grouped_round_routes_is_the_stock_boundary():
    """Every route's down output is bf16 before the reduction -- vLLM's bf16
    cache13 plus moe_sum -- while the default fp32 reduction rounds once at
    the end; the two are different arithmetic and the mode is explicit."""
    rows, cols, experts = 768, 192, 3
    stack = _stack(rows, cols, "value", [201, 202, 203])
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack],
                                               block_m=32, block_n=64, block_k=64)
    t, k = 16, 2
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    out = prepared(x, ids, rw, round_routes=True)
    sel = torch.arange(t, device="cuda")[:, None].expand_as(ids)
    routes = torch.stack([e.reference(x, "value") for e in stack])[ids.long(), sel]
    ref = (routes * rw[..., None]).bfloat16().float().sum(1).bfloat16()
    assert float((out.float() - ref.float()).abs().max()) < _tol(ref)
    plain = prepared(x, ids, rw, round_routes=False)
    assert not torch.allclose(out.float(), plain.float(), rtol=0, atol=0), \
        "the round_routes boundary must be observable"


@cuda
def test_grouped_call_captures_in_a_graph():
    rows, cols, experts = 768, 192, 3
    stack = _stack(rows, cols, "value", [71, 72, 73])
    prepared = wgg.prepare_grouped_window_gemm([e.unit for e in stack], block_m=32, block_n=64, block_k=64)
    t, k = 32, 2
    x = torch.randn(t, cols, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    y = prepared(x, ids, rw)
    graph = torch.cuda.CUDAGraph()
    out = torch.zeros_like(y, dtype=torch.float32)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            prepared(x, ids, rw)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        prepared(x, ids, rw, out=out)
    graph.replay()
    assert torch.equal(out.bfloat16(), y)
