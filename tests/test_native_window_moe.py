"""The native window MoE adapter: fused and split gate/up, both families, the
folded BF16 contract, placement flags, refusal of unsupported activations,
and graph capture -- against the same independent per-expert oracle the
grouped operator uses.

The oracle loops experts on the host (allowed in tests) and mirrors vLLM's
placement, not an algebraic equivalent: weights on gemm1 iff
``apply_router_weight_on_input``, otherwise on gemm2 with a plain sum; the
activation is fp32-silu cast once to bf16; the folded contract is
``bf16(value * row_scale)`` before each dot.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import native_window_moe as nwm    # noqa: E402
from tessera.errors import GrammarError         # noqa: E402

from test_window_gemm_grouped import Expert, _quant, _tol   # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the lane is a CUDA kernel")


def _per_expert(stack, xq, family, folded, a1, t):
    picked = torch.stack([e.reference(xq, family, folded=folded) for e in stack])
    if family == "e4m3" and a1 is not None:
        picked = picked * a1.reshape(1, t, 1)
    return picked                                            # [E, T, rows]


def _routes(picked, ids, sel, rw, weight_input):
    routes = picked[ids.long(), sel]
    if weight_input:
        routes = routes * rw[..., None]                      # gemm1 MUL_ROUTED_WEIGHT
    return routes.bfloat16()


def _down(dn_stack, act, ids, rw, rows_h, family, folded, weight_input):
    t, k = ids.shape
    route_rows = torch.arange(t * k, device=ids.device)
    flat = act.reshape(t * k, dn_stack[0].cols)
    if family == "e4m3":
        a_in, a2 = _quant(flat)
        flat_q = a_in.float()
    else:
        flat_q, a2 = flat, None
    picked = _per_expert(dn_stack, flat_q, family, folded, None, t)
    routes = picked[ids.long().reshape(-1), route_rows].reshape(t, k, rows_h)
    if family == "e4m3":
        routes = routes * a2.reshape(t, k, 1)
    if not weight_input:
        routes = (routes * rw[..., None]).bfloat16()         # gemm2 bf16 cast
    else:
        routes = routes.bfloat16()
    return routes.float().sum(1).bfloat16()                  # stock moe_sum


@cuda
def test_native_window_moe_matches_the_oracle_fused_and_split():
    rows_h, cols_h, inter, experts = 768, 192, 96, 3
    t, k = 16, 2
    seeds = [131, 132, 133]
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    sel = torch.arange(t, device="cuda")[:, None].expand_as(ids)
    x = torch.randn(t, cols_h, device="cuda").bfloat16()

    for family, arithmetic in (("value", "epilogue"), ("value", "folded"),
                               ("e4m3", "epilogue")):
        folded = arithmetic == "folded"
        gu_stack = [Expert(2 * inter, cols_h, (4,) * cols_h, s, family=family)
                    for s in seeds]
        gate_stack = [Expert(inter, cols_h, (4,) * cols_h, s + 200, family=family)
                      for s in seeds]
        up_stack = [Expert(inter, cols_h, (4,) * cols_h, s + 300, family=family)
                    for s in seeds]
        dn_stack = [Expert(rows_h, inter, (2 if i % 2 else 4,) * inter, s + 10,
                           family=family) for i, s in enumerate(seeds)]
        fused = nwm.prepare_native_window_moe(
            [e.unit for e in gu_stack], [e.unit for e in dn_stack],
            arithmetic=arithmetic, activation="silu")
        split = nwm.prepare_native_window_moe(
            [e.unit for e in gate_stack], [e.unit for e in dn_stack],
            up=[e.unit for e in up_stack], arithmetic=arithmetic, activation="silu")
        if family == "e4m3":
            x_in, a1 = _quant(x)
            xq1 = x_in.float()
        else:
            xq1, a1 = x, None
        for weight_input in (False, True):
            fu = _routes(_per_expert(gu_stack, xq1, family, folded, a1, t),
                         ids, sel, rw, weight_input)
            gate, up = fu[..., :inter].float(), fu[..., inter:].float()
            act_f = (torch.nn.functional.silu(gate) * up).bfloat16()
            ref_f = _down(dn_stack, act_f, ids, rw, rows_h, family, folded, weight_input)
            sg = _routes(_per_expert(gate_stack, xq1, family, folded, a1, t),
                         ids, sel, rw, weight_input)
            su = _routes(_per_expert(up_stack, xq1, family, folded, a1, t),
                         ids, sel, rw, weight_input)
            act_s = (torch.nn.functional.silu(sg.float()) * su.float()).bfloat16()
            ref_s = _down(dn_stack, act_s, ids, rw, rows_h, family, folded, weight_input)

            out_f = fused(x, ids, rw, apply_router_weight_on_input=weight_input)
            out_s = split(x, ids, rw, apply_router_weight_on_input=weight_input)
            assert out_f.shape == (t, rows_h) and out_f.dtype == torch.bfloat16
            assert float((out_f.float() - ref_f.float()).abs().max()) < _tol(ref_f), (
                f"fused {family}/{arithmetic} weight_input={weight_input}")
            assert float((out_s.float() - ref_s.float()).abs().max()) < _tol(ref_s), (
                f"split {family}/{arithmetic} weight_input={weight_input}")


@cuda
def test_packed_window_units_prepare_matches_the_direct_adapter():
    """The loader-facing bundle: packed units in, adapter out, fused and split
    spellings, with the per-expert bytes accounted."""
    rows_h, cols_h, inter, experts = 768, 192, 96, 2
    gu = [Expert(2 * inter, cols_h, (4,) * cols_h, 900 + i) for i in range(experts)]
    gate = [Expert(inter, cols_h, (4,) * cols_h, 910 + i) for i in range(experts)]
    up = [Expert(inter, cols_h, (4,) * cols_h, 920 + i) for i in range(experts)]
    dn = [Expert(rows_h, inter, (4,) * inter, 930 + i) for i in range(experts)]
    t, k = 16, 2
    x = torch.randn(t, cols_h, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")

    fused_pack = nwm.PackedWindowUnits(gate=tuple(e.unit for e in gu), up=(),
                                       down=tuple(e.unit for e in dn), family="value")
    split_pack = nwm.PackedWindowUnits(gate=tuple(e.unit for e in gate),
                                       up=tuple(e.unit for e in up),
                                       down=tuple(e.unit for e in dn), family="value")
    for pack, direct in (
        (fused_pack, nwm.prepare_native_window_moe([e.unit for e in gu],
                                                   [e.unit for e in dn],
                                                   arithmetic="folded")),
        (split_pack, nwm.prepare_native_window_moe([e.unit for e in gate],
                                                   [e.unit for e in dn],
                                                   up=[e.unit for e in up],
                                                   arithmetic="folded")),
    ):
        assert pack.experts == experts and pack.resident_bytes() > 0
        from_pack = pack.prepare()
        assert torch.equal(from_pack(x, ids, rw), direct(x, ids, rw))
        assert from_pack.down.arithmetic == "folded", \
            "the research BF16 wire's default arithmetic is folded"
    fp8_gu = [Expert(2 * inter, cols_h, (4,) * cols_h, 940 + i, family="e4m3")
              for i in range(experts)]
    fp8_dn = [Expert(rows_h, inter, (4,) * inter, 950 + i, family="e4m3")
              for i in range(experts)]
    fp8_pack = nwm.PackedWindowUnits(gate=tuple(e.unit for e in fp8_gu), up=(),
                                     down=tuple(e.unit for e in fp8_dn), family="e4m3")
    assert fp8_pack.prepare().down.arithmetic == "epilogue"


# -- the model's SwiGLU clamp (CPU: the arithmetic, not the kernel) --------

def test_silu_and_mul_applies_the_models_swiglu_clamp_in_fp32():
    """``swiglu_limit`` saturates the gate at ``+limit`` and the up branch at
    ``+-limit``, in fp32, before the activation -- vLLM's ``gemm1_clamp_limit``
    (config.py: "backends that do not implement the clamp cannot silently
    select one and drop the clamp").

    Direction matters and is asserted branch by branch: the gate is clamped
    with a *minimum* only (``min(gate, limit)``), the up branch with a
    two-sided clamp.  A sign slip here would read as a plausible activation.
    """
    limit = 10.0
    band = torch.tensor([[-3.0, 0.0, 3.0]], dtype=torch.float32)
    # gate above the limit with the up branch inside it: silu(limit) * up
    gate_hi = torch.tensor([[12.0, 3.0, 3.0]], dtype=torch.float32)
    # gate below the limit must NOT be clamped: silu(-12) * -10
    gate_lo = torch.tensor([[-12.0, 3.0, 3.0]], dtype=torch.float32)
    up_hi = torch.tensor([[3.0, 12.0, 3.0]], dtype=torch.float32)
    up_lo = torch.tensor([[3.0, -12.0, 3.0]], dtype=torch.float32)

    want_hi = (torch.nn.functional.silu(torch.tensor([[10.0, 3.0, 3.0]]))
               * torch.tensor([[3.0, 10.0, 3.0]])).bfloat16()
    assert torch.equal(nwm._silu_and_mul(gate_hi, up_hi, clamp_limit=limit), want_hi)
    want_lo = (torch.nn.functional.silu(torch.tensor([[-12.0, 3.0, 3.0]]))
               * torch.tensor([[3.0, -10.0, 3.0]])).bfloat16()
    assert torch.equal(nwm._silu_and_mul(gate_lo, up_lo, clamp_limit=limit), want_lo)

    # inside the band the clamp is a no-op; above it, the clamp is observable
    assert torch.equal(nwm._silu_and_mul(band, band, clamp_limit=limit),
                       nwm._silu_and_mul(band, band))
    assert not torch.equal(nwm._silu_and_mul(gate_hi, up_hi, clamp_limit=limit),
                           nwm._silu_and_mul(gate_hi, up_hi))
    # and no limit is exactly the pre-clamp arithmetic
    assert torch.equal(nwm._silu_and_mul(gate_hi, up_hi, clamp_limit=None),
                       nwm._silu_and_mul(gate_hi, up_hi))


def test_the_adapter_carries_the_clamp_limit_to_the_activation():
    """The limit reaches the activation from the call, not from a constant."""
    limit = 2.0
    gate = torch.tensor([[8.0]], dtype=torch.float32)
    up = torch.tensor([[8.0]], dtype=torch.float32)
    clamped = nwm._silu_and_mul(gate, up, clamp_limit=limit)
    assert torch.equal(clamped,
                       (torch.nn.functional.silu(torch.tensor([[2.0]]))
                        * torch.tensor([[2.0]])).bfloat16())


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "ten", object()])
def test_native_window_moe_refuses_an_unusable_clamp_limit(bad):
    """Not a number, or not finite and positive: refuse, never arm a clamp that
    cannot saturate anything."""
    with pytest.raises(GrammarError, match="swiglu_limit"):
        nwm.checked_swiglu_limit(bad)


def test_native_window_moe_accepts_the_models_limit_and_still_refuses_alpha_beta():
    assert nwm.checked_swiglu_limit(None) is None
    assert nwm.checked_swiglu_limit(10.0) == 10.0
    assert nwm.checked_swiglu_limit(10) == 10.0


@cuda
def test_native_window_moe_refuses_unsupported_activations_and_families():
    rows_h, cols_h, inter, experts = 768, 192, 96, 2
    gu_stack = [Expert(2 * inter, cols_h, (4,) * cols_h, 141 + i) for i in range(experts)]
    dn_stack = [Expert(rows_h, inter, (4,) * inter, 151 + i) for i in range(experts)]
    with pytest.raises(GrammarError, match="not served"):
        nwm.prepare_native_window_moe([e.unit for e in gu_stack], [e.unit for e in dn_stack],
                                      activation="gelu")
    fp8 = [Expert(2 * inter, cols_h, (4,) * cols_h, 161 + i, family="e4m3")
           for i in range(experts)]
    with pytest.raises(GrammarError, match="folded"):
        nwm.prepare_native_window_moe([e.unit for e in fp8],
                                      [e.unit for e in dn_stack],
                                      arithmetic="folded")


@cuda
def test_native_window_moe_call_captures_in_a_graph():
    rows_h, cols_h, inter, experts = 768, 192, 96, 3
    gu_stack = [Expert(2 * inter, cols_h, (4,) * cols_h, 171 + i) for i in range(experts)]
    dn_stack = [Expert(rows_h, inter, (4,) * inter, 181 + i) for i in range(experts)]
    prepared = nwm.prepare_native_window_moe([e.unit for e in gu_stack],
                                             [e.unit for e in dn_stack])
    t, k = 32, 2
    x = torch.randn(t, cols_h, device="cuda").bfloat16()
    ids = torch.randint(0, experts, (t, k), device="cuda", dtype=torch.int32)
    rw = torch.rand(t, k, device="cuda")
    y = prepared(x, ids, rw)
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            prepared(x, ids, rw)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        out = prepared(x, ids, rw)
    graph.replay()
    assert torch.equal(out, y)
