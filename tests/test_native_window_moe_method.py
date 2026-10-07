"""The REAL vLLM window MoE method: create_weights -> every loader callback ->
process_weights_after_loading -> apply through the mandatory native class path.

Runs in the serving image with real vLLM and CUDA. Native-valid TP2 fixtures
exercise row-cut gate/up and column-cut down on one GPU; this is not a two-node
result.  The two-node run is a separate milestone.

What is checked here:
* FP8 and folded BF16 arithmetic, TP1 and both TP2 ranks;
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
from test_serving_moe_route import _stack as _reference_stack, EXPERTS  # noqa: E402
from test_serving_moe_selected import _layer             # noqa: E402

# The fused class decoder needs native-valid K and rank-local N geometry.
HIDDEN, INTER = 128, 512


def _stack():
    return _reference_stack(experts=EXPERTS, hidden=HIDDEN, inter=INTER)

class _Lax(types.SimpleNamespace):
    """A stub config that answers the properties the real dataclasses
    compute (``is_*`` predicates and similar) instead of enumerating them."""

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return False if item.startswith("is_") else None


def _native_layer(tp_rank: int = 0, tp_size: int = 1, *, hidden: int = None,
                  inter: int = None, experts: int = None):
    """Complete the actual framework method and stock-reference configuration."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    import types as _types

    layer = _layer()
    layer.moe_config = _Lax(**vars(layer.moe_config))
    layer.moe_config.moe_parallel_config = _Lax(**vars(layer.moe_config.moe_parallel_config))
    mc, mp = layer.moe_config, layer.moe_config.moe_parallel_config
    # The expert count is a parameter too: the canonical fixture decodes E=2
    # and a stock factory asked for a config whose expert count disagrees with
    # the weights it is handed is being asked a different question.
    mc.num_experts = int(experts or EXPERTS)
    mc.num_local_experts = int(experts or EXPERTS)
    mc.num_logical_experts = int(experts or EXPERTS)
    mc.experts_per_token = 2
    # Geometry is a parameter: the canonical fixture is H=4096 / I=2048, not
    # this helper's small synthetic tile, and a harness that ran the real
    # containers at the synthetic geometry would be a false pass.
    mc.hidden_dim = int(hidden or HIDDEN)
    mc.intermediate_size = int(inter or INTER)
    mc.intermediate_size_per_partition = int(inter or INTER) // tp_size
    mc.intermediate_size_per_partition_unpadded = int(inter or INTER) // tp_size
    mc.has_bias = False
    mc.is_lora_enabled = False
    mc.activation = MoEActivation.SILU
    mc.device = torch.device("cuda")
    mc.in_dtype = torch.bfloat16
    mc.routing_method = None
    mc.moe_backend = "auto"
    mc.skip_final_all_reduce = False
    mc.defer_moe_finalize = False
    mc.max_capture_size = 0
    mc.rocm_aiter_fmoe_enabled = False
    mp.tp_size = tp_size
    mp.tp_rank = tp_rank
    mp.ep_size = 1
    mp.dp_size = 1
    mp.pcp_size = 1
    mp.sp_size = 1
    mp.use_ep = False
    mp.enable_eplb = False
    mp.use_deepep_v = False
    mp.use_batched_activation_format = False
    mp.is_sequence_parallel = False
    layer.activation = MoEActivation.SILU
    return layer

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the native method runs CUDA kernels")


def _load_all(method, layer, w13_blobs, w2_blobs):
    for expert in range(len(w13_blobs)):
        for shard, blob in (('w1', w13_blobs[expert][0]), ('w3', w13_blobs[expert][1]),
                            ('w2', w2_blobs[expert])):
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            assert param.weight_loader(
                param, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
                'wire', shard, expert, return_success=True)


def _fp8_reference(reference, x, ids, weights, tp_rank=0, tp_size=1):
    """Stock-arithmetic reference for one rank's routed output.

    Both stages quantize their activation with vLLM's per-token dynamic FP8 op
    (``fused_moe.py`` dispatches gemm1 and ``moe_kernel_quantize_input`` for
    gemm2): stage 1 scales x by its own scale *before* the nonlinearity, and
    the down projection consumes the quantized activation per route.  Written
    out so the test does not read the implementation back to itself.
    """
    from vllm import _custom_ops  # noqa: F401  (registers torch.ops._C)

    local_inter = INTER // tp_size
    lo, hi = tp_rank * local_inter, (tp_rank + 1) * local_inter

    def _quant(row):
        q = torch.empty((1, row.numel()), dtype=torch.float8_e4m3fn, device=row.device)
        s = torch.empty((1, 1), dtype=torch.float32, device=row.device)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(q, row.reshape(1, -1), s, None)
        return q.reshape(-1).float() * s.reshape(-1)[0]

    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        xg = _quant(x[token].float())
        for choice in range(ids.shape[1]):
            e = int(ids[token, choice])
            ref = reference[e]
            gate = ref["gate"]["weight"][lo:hi].float().to(x.device) \
                * ref["gate"]["weight_scale"][lo:hi].to(x.device)
            up = ref["up"]["weight"][lo:hi].float().to(x.device) \
                * ref["up"]["weight_scale"][lo:hi].to(x.device)
            down = ref["down"]["weight"][:, lo:hi].float().to(x.device) \
                * ref["down"]["weight_scale"].to(x.device)
            g = gate @ xg
            u = up @ xg
            act_q = _quant(torch.nn.functional.silu(g) * u)
            out[token] += weights[token, choice] * (down @ act_q)
    return out.bfloat16()


class _SharedSpy:
    """The shared-experts coordinator seam: any call is a failure here."""

    def __init__(self):
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("the native method must not compute shared experts")


def _native_method(scheme, layer):
    method = moe_route.build_tessera_moe_method(scheme, 'm', 'resident', layer)
    method.create_weights(layer, EXPERTS, HIDDEN,
                          INTER // layer.moe_config.moe_parallel_config.tp_size,
                          torch.bfloat16)
    return method


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


def bf16_wires_native_data():
    """Folded BF16 reference wires at the native WINDOW width."""
    from tessera.alphabet import BF16_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.unit_artifact import read_unit_artifact

    w13_blobs, w2_blobs, expected = [], [], []
    for expert in range(2):
        parts = {}
        for projection, rows, cols in (('gate_proj', INTER, HIDDEN),
                                       ('up_proj', INTER, HIDDEN),
                                       ('down_proj', HIDDEN, INTER)):
            weight = torch.randn(rows, cols, generator=torch.Generator().manual_seed(
                700 + expert * 3 + len(parts)))
            written, _unit, _forests = encode_linear_planes(
                weight, grid=BF16_GRID, q256=512, name=projection,
                window_bits=14, verify=False)
            parts[projection] = (pack_fused([(projection, rows, written.blob)]),
                                 read_unit_artifact(written.blob).to(torch.bfloat16))
        w13_blobs.append([parts['gate_proj'][0], parts['up_proj'][0]])
        w2_blobs.append([parts['down_proj'][0]])
        expected.append((torch.cat([parts['gate_proj'][1], parts['up_proj'][1]]),
                         parts['down_proj'][1]))
    scheme = {
        'family': 'TESSERA_BF16', 'structure': 'routed_moe', 'grid': 'BF16',
        'body': 'WINDOW', 'plane': 'CHANNEL', 'experts': 2,
        'expert_ids': [0, 1],
        'expert_classes': [{'start': 0, 'end': 2,
                            'q256': {'w13': [512, 512], 'w2': [512]}}],
        'groups': {
            'w13': {'rows': 2 * INTER, 'columns': HIDDEN, 'q256': 512,
                    'wire_stride': max(len(blob) for pair in w13_blobs for blob in pair),
                    'roles': [['gate_proj', INTER], ['up_proj', INTER]]},
            'w2': {'rows': HIDDEN, 'columns': INTER, 'q256': 512,
                   'wire_stride': max(len(b[0]) for b in w2_blobs),
                   'roles': [['down_proj', HIDDEN]]}}}
    return w13_blobs, w2_blobs, scheme, expected


@pytest.fixture(scope="module")
def bf16_wires_native():
    return bf16_wires_native_data()


@cuda
def test_bf16_folded_native_route_matches_folded_reference(bf16_wires_native):
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

    # folded reference: bf16(values * scale) tiles, exactly decode_folded
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
def test_bf16_native_route_matches_the_folded_reference(
        bf16_wires_native, tp_rank, tp_size):
    """Routed-only folded BF16 math over actual TP1/TP2 loaded wire slices."""

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
        x, w1, w2, weights, ids, family="TESSERA_FP8", clamp=None,
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
