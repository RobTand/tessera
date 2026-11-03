"""The REAL vLLM window MoE method: create_weights -> every loader callback ->
process_weights_after_loading -> apply through the mandatory native class path.

Runs in the serving image with real vLLM and CUDA. Native-valid TP2 fixtures
exercise row-cut gate/up and column-cut down on one GPU; this is not a two-node
result.  The two-node run is a separate milestone.

What is checked here:
* FP8 ordinary route (no research config), TP1;
* FP8 ordinary route at TP2, rank 0 and rank 1, row-cut w13 and column-cut w2;
* BF16 through the research route and the production stack at TP1 and both TP2 ranks;
* actual routing and the nonlinear activation against a stock-arithmetic
  reference built from the same reference tensors the wire decodes to;
* shared-expert ownership: the method returns ROUTED output only and never
  calls the shared-experts seam (the runner owns NO_OVERLAP combination);
* refusals: a missing projection, a duplicated projection, and a wrong
  sidecar/rung.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.serving import moe_route                    # noqa: E402
from serving_moe_route_support import EXPERTS              # noqa: E402

from native_window_moe_method_support import (
    HIDDEN,
    INTER,
    _stack,
    _native_layer,
    _load_all,
    _fp8_reference,
    _SharedSpy,
    _native_method,
    bf16_wires_native,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the native method runs CUDA kernels")


@cuda
def test_fp8_native_method_tp1_loads_compacts_and_matches_the_reference():
    w13_blobs, w2_blobs, scheme, reference = _stack()
    layer = _native_layer()
    method = _native_method(scheme, layer)
    _load_all(method, layer, w13_blobs, w2_blobs)
    method.process_weights_after_loading(layer)
    assert method._native is not None
    assert method.moe_kernel is None, "no internal MK kernel for the runner's overlap logic"
    assert not dict(layer.named_parameters()), "only packed constants stay resident"

    x = (torch.randn(8, HIDDEN) * 0.5).bfloat16().cuda()
    ids = torch.randint(0, EXPERTS, (8, 2), dtype=torch.int32, device='cuda')
    weights = torch.rand(8, 2, device='cuda')
    shared = _SharedSpy()
    out = method.apply(layer, x, weights, ids, shared, None)
    assert shared.calls == 0, "shared experts are the runner's (NO_OVERLAP)"
    ref = _fp8_reference(reference, x, ids, weights)
    diff = (out.float() - ref.float()).abs()
    if float(diff.max()) >= 5e-2 + 2e-2 * float(ref.float().abs().max()):
        i, j = (diff == diff.max()).nonzero()[0].tolist()
        print("DIAG max-diff element", i, j, "out", float(out[i, j]), "ref", float(ref[i, j]),
              "ids", ids[i].tolist(), "weights", weights[i].tolist(),
              "row diff count", int((diff[i] > 5e-2).sum()))
        print("DIAG per-row max", [round(float(v), 4) for v in diff.max(1).values])
    assert float(diff.max()) < 5e-2 + 2e-2 * float(ref.float().abs().max()), \
        f"max abs diff {float(diff.max())}"


@cuda
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_fp8_native_method_tp2_shard_shapes_are_rank_local(tp_rank):
    """Both shard axes on one GPU: a component check of the rank-local
    geometry/arithmetic, not a two-node result."""
    w13_blobs, w2_blobs, scheme, reference = _stack()
    layer = _native_layer(tp_rank=tp_rank, tp_size=2)
    method = _native_method(scheme, layer)
    _load_all(method, layer, w13_blobs, w2_blobs)
    method.process_weights_after_loading(layer)
    assert method._native is not None

    x = (torch.randn(8, HIDDEN) * 0.5).bfloat16().cuda()
    ids = torch.randint(0, EXPERTS, (8, 2), dtype=torch.int32, device='cuda')
    weights = torch.rand(8, 2, device='cuda')
    out = method.apply(layer, x, weights, ids, _SharedSpy(), None)
    ref = _fp8_reference(reference, x, ids, weights, tp_rank=tp_rank, tp_size=2)
    diff = (out.float() - ref.float()).abs()
    assert float(diff.max()) < 5e-2 + 2e-2 * float(ref.float().abs().max()), \
        f"rank {tp_rank}: max abs diff {float(diff.max())}"


@cuda
def test_bf16_native_research_route_matches_the_canonical_reference(bf16_wires_native):
    w13_blobs, w2_blobs, scheme, expected = bf16_wires_native
    layer = _native_layer(tp_rank=0, tp_size=2)
    layer.global_num_experts = 2
    config = moe_route.ResearchSelectedMoeConfig(
        max_experts_per_chunk=2, expected_tensor_parallel_size=2)
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(
            types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True))):
        method = moe_route.build_tessera_moe_method(
            scheme, 'm', 'resident', layer, research_selected=config)
    method.create_weights(layer, 2, HIDDEN, INTER // 2, torch.bfloat16)
    _load_all(method, layer, [[pair[0], pair[1]] for pair in w13_blobs],
              [pair[0] for pair in w2_blobs])
    method.process_weights_after_loading(layer)
    assert method._native is not None

    lo, hi = 0, INTER // 2
    x = (torch.randn(8, HIDDEN) * 0.5).bfloat16().cuda()
    ids = torch.randint(0, 2, (8, 2), dtype=torch.int32, device='cuda')
    weights = torch.rand(8, 2, device='cuda')
    out = method.apply(layer, x, weights, ids, _SharedSpy(), None)
    assert out.dtype == torch.bfloat16 and out.shape == (8, HIDDEN)

    # The canonical effective weights have no per-weight BF16 conversion.
    ref = torch.zeros(8, HIDDEN, dtype=torch.float32, device='cuda')
    for token in range(8):
        for choice in range(2):
            e = int(ids[token, choice])
            full13, full2 = expected[e]
            gate = full13[lo:hi].float().to(x.device)
            up = full13[INTER + lo:INTER + hi].float().to(x.device)
            down = full2[:, lo:hi].float().to(x.device)
            g = gate @ x[token].float()
            u = up @ x[token].float()
            act = (torch.nn.functional.silu(g) * u).bfloat16().float()
            ref[token] += weights[token, choice] * (down @ act)
    ref = ref.bfloat16()
    diff = (out.float() - ref.float()).abs()
    assert float(diff.max()) < 5e-2 + 2e-2 * float(ref.float().abs().max()), \
        f"max abs diff {float(diff.max())}"


@cuda
@pytest.mark.parametrize('tp_rank,tp_size', [(0, 1), (0, 2), (1, 2)])
def test_bf16_production_route_matches_the_canonical_reference(
        bf16_wires_native, tp_rank, tp_size):
    """Exercise the real native owner on canonical BF16 weights at each TP cut."""

    w13_blobs, w2_blobs, scheme, expected = bf16_wires_native
    layer = _native_layer(tp_rank=tp_rank, tp_size=tp_size)
    layer.global_num_experts = 2
    method = moe_route.build_tessera_moe_method(scheme, 'm', 'resident', layer)
    method.create_weights(layer, 2, HIDDEN, INTER // tp_size, torch.bfloat16)
    registered = dict(layer.named_parameters())
    assert not {'w13_weight', 'w2_weight', 'w13_weight_scale',
                'w2_weight_scale'} & set(registered), sorted(registered)
    _load_all(method, layer, [[pair[0], pair[1]] for pair in w13_blobs],
              [pair[0] for pair in w2_blobs])
    method.process_weights_after_loading(layer)
    assert method._native is not None and method.moe_kernel is None
    assert not dict(layer.named_parameters()), "only packed constants stay resident"


    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    x = (torch.randn(8, HIDDEN, generator=torch.Generator().manual_seed(41)) * 0.5
         ).bfloat16().cuda()
    ids = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0], [1, 1], [0, 0]],
                       dtype=torch.int32, device='cuda')
    weights = torch.rand(8, 2, generator=torch.Generator().manual_seed(42)).cuda()
    shared = _SharedSpy()
    out = method.apply(layer, x, weights, ids, shared, None)
    assert shared.calls == 0
    assert out.dtype == torch.bfloat16 and out.shape == (8, HIDDEN)


    ref = torch.zeros(8, HIDDEN, dtype=torch.float32, device='cuda')
    for token in range(8):
        for choice in range(2):
            e = int(ids[token, choice])
            full13, full2 = expected[e]
            gate = full13[lo:hi].float().cuda()
            up = full13[INTER + lo:INTER + hi].float().cuda()
            down = full2[:, lo:hi].float().cuda()
            g = gate @ x[token].float()
            u = up @ x[token].float()
            act = (torch.nn.functional.silu(g) * u).bfloat16().float()
            ref[token] += weights[token, choice] * (down @ act)
    ref = ref.bfloat16()
    diff = (out.float() - ref.float()).abs()
    assert float(diff.max()) < 5e-2 + 2e-2 * float(ref.float().abs().max()), \
        f"TP{tp_size} rank {tp_rank}: max abs diff {float(diff.max())}"


@cuda
def test_router_weight_on_input_matches_actual_stock_placement():
    """At topk=1, compare to the actual STOCK MODULAR prepare/kernel.

    Its prepare multiplies bf16 x by bf16 routing weights BEFORE per-token
    FP8 quantization. Legacy ``fused_experts``' accumulator placement is not
    the served contract and cannot be this test's oracle.
    """
    from native_window_moe_stock_crosscheck import _stock_modular_reference
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    w13_blobs, w2_blobs, scheme, reference = _stack()
    layer = _native_layer()
    layer.moe_config.experts_per_token = 1
    method = _native_method(scheme, layer)
    _load_all(method, layer, w13_blobs, w2_blobs)
    method.process_weights_after_loading(layer)
    x = (torch.randn(8, HIDDEN) * 0.5).bfloat16().cuda()
    ids = torch.randint(0, EXPERTS, (8, 1), dtype=torch.int32, device="cuda")
    weights = torch.rand(8, 1, device="cuda")
    layer.apply_router_weight_on_input = True
    native = method.apply(layer, x, weights, ids, _SharedSpy(), None)

    w1 = torch.stack([torch.cat([r["gate"]["weight"], r["up"]["weight"]])
                      for r in reference]).cuda().contiguous()
    s1 = torch.stack([torch.cat([r["gate"]["weight_scale"], r["up"]["weight_scale"]])
                      for r in reference]).cuda().contiguous()
    w2 = torch.stack([r["down"]["weight"] for r in reference]).cuda().contiguous()
    s2 = torch.stack([r["down"]["weight_scale"] for r in reference]).cuda().contiguous()
    stock = _stock_modular_reference(
        x, w1, w2, weights, ids, clamp=None,
        experts=EXPERTS, apply_router_weight_on_input=True,
        w1_scale=s1, w2_scale=s2, moe_config=method.moe)
    diff = (native.float() - stock.float()).abs()
    assert float(diff.max()) < 5e-3 + 1e-2 * float(stock.float().abs().max()), \
        f"max abs diff {float(diff.max())}"


@cuda
def test_native_method_refuses_missing_duplicate_and_wrong_rung():
    w13_blobs, w2_blobs, scheme, _reference = _stack()

    # missing projection: one expert's up never arrives
    layer = _native_layer()
    method = _native_method(scheme, layer)
    for expert in range(EXPERTS):
        for shard, blob in (('w1', w13_blobs[expert][0]),
                            ('w3', w13_blobs[expert][1] if expert else None),
                            ('w2', w2_blobs[expert])):
            if blob is None:
                continue
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            param.weight_loader(param, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
                                'wire', shard, expert, return_success=True)
    # The wire-length validator reads what the loader recorded; record the
    # skipped wire as declared-but-absent so the AXIS is what refuses it.
    layer.tessera_w13_wire_len[0, 1] = len(w13_blobs[0][1])
    with pytest.raises(Exception, match="missing experts"):
        method.process_weights_after_loading(layer)

    # duplicate projection: the same callback twice
    layer = _native_layer()
    method = _native_method(scheme, layer)
    blob = w13_blobs[0][0]
    layer.w13_wire.weight_loader(
        layer.w13_wire, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
        'wire', 'w1', 0, return_success=True)
    with pytest.raises(ValueError):
        layer.w13_wire.weight_loader(
            layer.w13_wire, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
            'wire', 'w1', 0, return_success=True)

    # wrong sidecar/rung: blobs encoded at one rung, sidecar declares another
    import copy
    bad = copy.deepcopy(scheme)
    bad['groups']['w13']['q256'] = 256
    layer = _native_layer()
    with pytest.raises(Exception):
        method = moe_route.build_tessera_moe_method(bad, 'm', 'resident', layer)
        method.create_weights(layer, EXPERTS, HIDDEN, INTER, torch.bfloat16)
        _load_all(method, layer, w13_blobs, w2_blobs)
        method.process_weights_after_loading(layer)
