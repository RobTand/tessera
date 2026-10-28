"""Pending exact-tuple GPU evidence; these tests do not admit a serving cell."""
from types import SimpleNamespace

import pytest
import torch

from tessera import routed_fused as rf
from tessera.native_window_moe import _silu_and_mul
from tessera.t16_routed_split import SplitT16RoutedMoE

import fused_bound as fb
from test_routed_fused_window import (
    _bundles, _down_bounds, _gate_up_bounds, _routes, _stacks,
    _tp2_rank1_bundles, _tp2_rank1_stacks,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="split T16 requires CUDA")
HIGH_RATES = range(rf.RATE_MAX + 1, rf.DENSE_RATE_MAX["value"] + 1)
Q256 = [rate * 256 for rate in HIGH_RATES] + [
    rate * 256 + 128 for rate in range(rf.RATE_MAX, rf.DENSE_RATE_MAX["value"])]


def independent_down_bounds(truths, activation, activation_error, ids, weights):
    """Compose the independent activation, not the native activation.

    Propagate its admitted BF16-ulp uncertainty through absolute down weights.
    The existing operation/rounding bound evaluated at absolute input plus
    that uncertainty bounds every allowed native input, including route sums
    and BF16 binade crossings. Neither term reads a native activation value.
    """
    absolute_down = [SimpleNamespace(**(vars(truth) | {
        "values": truth.values.abs(), "scale": truth.scale.abs()})) for truth in truths[2]]
    absolute_truths = (*truths[:2], absolute_down)
    reference, _bound = _down_bounds(truths, "value", activation, ids, weights)
    uncertainty, _bound = _down_bounds(absolute_truths, "value", activation_error, ids, weights.abs())
    _reference, rounding = _down_bounds(
        absolute_truths, "value", activation.double().abs() + activation_error, ids, weights.abs())
    return reference, rounding + uncertainty


def check_stages(owner, truths, x, ids, weights):
    routing = rf._routing_tables(ids, weights, owner.down.words.shape[0], x.device, (rf.BM,))
    gate = owner._project(owner.gate, x, routing, a_row_mode=0, mul_weight=False)
    up = owner._project(owner.up, x, routing, a_row_mode=0, mul_weight=False)
    ref_gate, bound_gate, ref_up, bound_up = _gate_up_bounds(truths, "value", x, ids)
    fb.check_within(gate.reshape_as(ref_gate), ref_gate, bound_gate, "split gate real wire")
    fb.check_within(up.reshape_as(ref_up), ref_up, bound_up, "split up real wire")
    activation = torch.empty_like(gate)
    owner.module.split_swiglu(gate, up, routing.flat_sorted, 3.0, activation)
    reference_activation = _silu_and_mul(gate, up, clamp_limit=3.0)
    activation_error = fb.bf16_ulp(reference_activation.double().abs())
    sorted_routes = routing.flat_sorted.long()
    fb.check_within(activation, reference_activation[sorted_routes].double(),
                    activation_error[sorted_routes], "split activation vs independent SwiGLU")
    reference, bound = independent_down_bounds(
        truths, reference_activation, activation_error, ids, weights)
    output = owner(x, ids, weights, swiglu_limit=3.0)
    fb.check_within(output, reference, bound, "split down real wire")
    assert torch.equal(output, owner(x, ids, weights, swiglu_limit=3.0))
    return output


@pytest.mark.parametrize("q256", Q256)
def test_high_rate_mixed_tables_eager_and_graph(q256):
    stacks = _stacks("value", hidden=256, inter=256, q256=q256, cap=rf.DENSE_RATE_MAX["value"])
    bundles = _bundles("value", stacks)
    owner = SplitT16RoutedMoE.from_bundles(bundles.gate, bundles.up, bundles.down)
    x = torch.randn(71, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    ids, weights = _routes(71, 3, 802)
    eager = check_stages(owner, stacks, x, ids, weights)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = owner(x, ids, weights, swiglu_limit=3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)
    x.mul_(0.5)
    ids.copy_(ids.flip(0))
    weights.mul_(0.75)
    changed = owner(x, ids, weights, swiglu_limit=3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, changed)


@pytest.mark.parametrize("q256", Q256)
def test_tp2_rank1_encoded_wire_and_axis_intake(q256):
    units, truths, _rates = _tp2_rank1_stacks("value", q256, rate_max=rf.DENSE_RATE_MAX["value"])
    bundles = _tp2_rank1_bundles("value", units, "axis")
    owner = SplitT16RoutedMoE.from_bundles(bundles.gate, bundles.up, bundles.down)
    experts = owner.down.words.shape[0]
    x = torch.eye(owner.gate.cols, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(experts, device="cuda", dtype=torch.int32).expand(x.shape[0], experts).contiguous()
    weights = torch.ones_like(ids, dtype=torch.float32)
    check_stages(owner, truths, x, ids, weights)
