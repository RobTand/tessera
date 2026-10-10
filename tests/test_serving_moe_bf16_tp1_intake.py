"""Routed class construction must not allocate a decoded checkpoint tile.

This is allocation coverage only; actual eager/graph numerics are exercised by
the CUDA class packet and plugin tests. There is no reader-selection fallback.
"""

import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.model_executor.layers.fused_moe.activation")
# tessera#1031: a stub ``vllm`` satisfies a top-package guard while the
# activation submodule stays unimportable. Guard the submodule instead.

import torch

from tessera.serving import moe_route
from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4
# Sibling test modules by their own names: ``tests/conftest.py`` puts this
# directory on ``sys.path``.
from test_native_window_moe_method import _native_layer
from test_serving_dispatch import _moe_scheme

# The routed scheme this file builds: ``test_serving_dispatch._moe_scheme``'s
# defaults, and the layer stub must carry the same numbers.
EXPERTS, HIDDEN, INTER = 4, 128, 64


def _eager():
    from vllm.config import set_current_vllm_config

    return set_current_vllm_config(
        types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True)))


def _research(tp_size):
    return moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2,
                                              expected_tensor_parallel_size=tp_size)


def _build(*, family, tp_rank=0, tp_size=1, research=True, with_layer=False):
    """Construct the routed method and take its intake, without loading bytes."""
    scheme = (_moe_scheme() if family == TESSERA_FP8
              else _moe_scheme(family=TESSERA_BF16, grid="BF16"))
    layer = _native_layer(tp_rank=tp_rank, tp_size=tp_size, hidden=HIDDEN,
                          inter=INTER, experts=EXPERTS)
    with _eager():
        method = moe_route.build_tessera_moe_method(
            scheme, "m", "resident", layer,
            research_selected=(_research(tp_size) if research else None))
        method.create_weights(layer, EXPERTS, HIDDEN, INTER // tp_size, torch.bfloat16)
    return (method, layer) if with_layer else method


@pytest.mark.parametrize("family", [TESSERA_FP8, TESSERA_BF16])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_the_compact_intake_registers_no_stock_expert_tile(family, tp_size):
    """vLLM constructs every layer before it loads any weight, so a stock tile
    registered here is held for EVERY routed layer at once -- and the compact
    lane drops it unused.  Neither family registers one on this lane."""
    method, layer = _build(family=family, tp_size=tp_size, research=False,
                           with_layer=True)
    assert sum(parameter.untyped_storage().nbytes()
               for parameter in layer.parameters()) == 0


