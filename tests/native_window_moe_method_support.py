"""Shared native MoE layers, loader controls and reference wires."""
from __future__ import annotations

import types

import pytest
import torch

from tessera.serving import moe_route
from serving_moe_route_support import _stack as _reference_stack, EXPERTS
from serving_moe_selected_support import _layer


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


def bf16_wires_native_data():
    """Canonical BF16 reference wires at the native WINDOW width."""
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
                                 read_unit_artifact(written.blob))
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
