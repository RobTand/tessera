"""A16 routed experts: one predicate decides the intake, at TP1 as well as TP2.

The routed window lane asks the same question twice -- once for the
construction-time identity (``native_route``) and once for loader ownership
(``incremental``).  At A16 the answer is the research-selected config: the FP8
stack takes the compact lane at every world size, and compressed BF16 takes it
only under ``research_selected``, whose folded arithmetic is that config's
contract.  TP1 was the hole: the two questions were spelled once as
``family == FP8 or tp_size == 2`` and once as ``family == FP8 or
(research is not None and tp_size == 2)``, so an A16 research-selected TP1
stack was built non-native and loaded through the padded wire bank.

CONSTRUCTION ONLY.  The compact lane's kernels are CUDA-only and are exercised
on a device by ``test_serving_moe_tp2.py``; what is pinned here is which intake
a construction selects, that both questions give the same answer, and that an
unsupported BF16 stack refuses by name instead of reaching a materialising
branch with no backend selected.  No claim about served numerics is made: a
native A16 TP1 serve still owes its device result.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch

from tessera.serving import moe_route
from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8
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


def _build(*, family, tp_rank=0, tp_size=1, research=True):
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
    return method


@pytest.mark.parametrize("family,compact_ready,tp_size,research,expected", [
    # FP8 takes the compact lane whenever the shared reader exists, at any world.
    (TESSERA_FP8, True, 1, False, True),
    (TESSERA_FP8, True, 2, False, True),
    # BF16 takes it only under the research-selected config, at TP1 and TP2.
    (TESSERA_BF16, True, 1, True, True),
    (TESSERA_BF16, True, 2, True, True),
    # An unsupported BF16 stack is not silently routed to a materialiser.
    (TESSERA_BF16, True, 1, False, False),
    (TESSERA_BF16, True, 2, False, False),
    # A world size the research contract cannot check is not this predicate's
    # business to admit.
    (TESSERA_BF16, True, 4, True, False),
    # No shared reader published: nothing takes the compact lane.
    (TESSERA_FP8, False, 2, False, False),
])
def test_one_predicate_answers_both_questions(family, compact_ready, tp_size, research,
                                              expected):
    selected = moe_route.compact_window_lane(
        family, compact_ready, tp_size=tp_size,
        research_selected=(object() if research else None))
    assert selected is expected


@pytest.mark.parametrize("tp_rank,tp_size", [(0, 1), (0, 2), (1, 2)])
def test_bf16_research_selected_takes_the_compact_intake(tp_rank, tp_size):
    method = _build(family=TESSERA_BF16, tp_rank=tp_rank, tp_size=tp_size)
    assert method._native_mode is True
    assert method._rank_local_intake is not None
    # The compact lane owns the compute: no stock kernel and no experts class
    # may be selected beside it.
    assert method.fp8_backend is None and method.bf16_backend is None
    assert method.experts_cls is None


def test_fp8_still_takes_the_compact_intake_without_research_selected():
    method = _build(family=TESSERA_FP8, tp_size=2, research=False)
    assert method._native_mode is True
    assert method._rank_local_intake is not None


@pytest.mark.parametrize("tp_size", [1, 2])
def test_bf16_without_research_selected_refuses_by_name(tp_size):
    """Ordinary compressed BF16 is refused at the builder's front door.

    ``refuse_a_family_with_no_expert_route`` (``scheme.py:860-880``) is asked
    before any vLLM fused-MoE import, and the builder's own carve-out admits a
    BF16 stack only with an explicit research-selected config
    (``moe_route.py:663-665``).  So no ordinary BF16 stack reaches a
    materialising branch -- at either world size.
    """
    with pytest.raises(ValueError, match="has no expert route in this build"):
        _build(family=TESSERA_BF16, tp_size=tp_size, research=False)
