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
