"""A16 routed experts: one predicate decides the intake, at TP1 as well as TP2.

UPDATED FOR tessera#609.  Compressed BF16 is now a production expert family
(``scheme.MOE_BUILDERS``): it takes the compact lane at every world size
exactly as FP8 does, with or without a research-selected config, and a build
without the compact reader refuses a production BF16 stack by name rather
than reaching a materialising branch.  The history below is kept because the
TP1 hole it describes is why one predicate answers both questions.

The routed window lane asks the same question twice -- once for the
construction-time identity (``native_route``) and once for loader ownership
(``incremental``).  At A16 the answer is the research-selected config: the FP8
stack takes the compact lane at every world size, and compressed BF16 takes it
only under ``research_selected``, whose folded arithmetic is that config's
contract.  TP1 was the hole, and it was not a disagreement between the two
spellings: both said ``tp_size == 2`` for BF16, so an A16 research-selected TP1
stack was *consistently* built non-native and loaded through the padded wire
bank.  The one input they answered differently -- ordinary BF16 at TP2, where
the identity arm said native and the intake arm did not -- is refused at the
builder's front door and was never reachable.

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


@pytest.mark.parametrize("family,compact_ready,tp_size,research,expected", [
    # FP8 takes the compact lane whenever the shared reader exists, at any world.
    (TESSERA_FP8, True, 1, False, True),
    (TESSERA_FP8, True, 2, False, True),
    # BF16 takes it too (tessera#609), research-selected or not, at any world:
    # the research contract's TP1/TP2 bound is that route's own refusal.
    (TESSERA_BF16, True, 1, True, True),
    (TESSERA_BF16, True, 2, True, True),
    (TESSERA_BF16, True, 1, False, True),
    (TESSERA_BF16, True, 2, False, True),
    (TESSERA_BF16, True, 4, False, True),
    # No shared reader published: nothing takes the compact lane.
    (TESSERA_FP8, False, 2, False, False),
    (TESSERA_BF16, False, 2, False, False),
    # A family this builder does not serve is not admitted through the lane --
    # at any world size, with or without a research-selected config.  NVFP4 has
    # its own builder (``scheme.MOE_BUILDERS``).
    (TESSERA_NVFP4, True, 1, False, False),
    (TESSERA_NVFP4, True, 2, False, False),
    (TESSERA_NVFP4, True, 2, True, False),
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


@pytest.mark.parametrize("tp_rank,tp_size", [(0, 1), (0, 2), (1, 2)])
def test_bf16_production_takes_the_compact_intake(tp_rank, tp_size):
    """An ordinary BF16 stack is a production stack now (tessera#609)."""
    method = _build(family=TESSERA_BF16, tp_rank=tp_rank, tp_size=tp_size, research=False)
    assert method._native_mode is True
    assert method._rank_local_intake is not None
    assert method.fp8_backend is None and method.bf16_backend is None
    assert method.experts_cls is None


@pytest.mark.parametrize("family", [TESSERA_FP8, TESSERA_BF16])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_the_compact_intake_registers_no_stock_expert_tile(family, tp_size):
    """vLLM constructs every layer before it loads any weight, so a stock tile
    registered here is held for EVERY routed layer at once -- and the compact
    lane drops it unused.  Neither family registers one on this lane."""
    method, layer = _build(family=family, tp_size=tp_size, research=False,
                           with_layer=True)
    assert method._rank_local_intake is not None
    registered = set(dict(layer.named_parameters()))
    assert registered == {"w13_wire", "w2_wire"}, sorted(registered)
    assert layer.w13_wire.numel() == 0 and layer.w2_wire.numel() == 0


def test_bf16_production_without_the_compact_reader_refuses_by_name(monkeypatch):
    """No compact reader: a production BF16 stack has nowhere to go.  There is
    no materialising BF16 expert path, and the refusal says so rather than
    reaching vLLM's unquantized backend oracle."""
    from tessera.serving import scheme as _scheme

    monkeypatch.delattr(_scheme, "parse_compact_tessera_expert_blob")
    with pytest.raises(ValueError, match="compact native window lane"):
        _build(family=TESSERA_BF16, tp_size=1, research=False)
