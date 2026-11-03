"""Shared routed-constructor layer configuration."""
import types

import pytest

torch = pytest.importorskip('torch')

from serving_moe_route_support import EXPERTS


def _layer():
    layer = torch.nn.Module()
    layer.moe_config = types.SimpleNamespace(is_act_and_mul=True,
        moe_parallel_config=types.SimpleNamespace(tp_size=1, tp_rank=0, ep_size=1, dp_size=1,
            pcp_size=1, sp_size=1, use_ep=False, enable_eplb=False),
        experts_per_token=2)
    layer.activation = 'silu'
    layer.global_num_experts = EXPERTS
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer._expert_routing_tables = lambda: None
    return layer
