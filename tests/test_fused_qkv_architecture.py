"""The fused q/k/v rule is scoped by architecture (tessera#706).

``fused_module`` is a rule over tensor NAMES; ``Glm5NextForConditionalGeneration``
keeps ``q_proj``/``k_proj``/``v_proj`` as separate Linears (no ``qkv_proj``
module is ever built; contract v44 census), so the name rule alone declares a
module vLLM never constructs.  The architecture is explicit data, threaded
from ``config.json``'s ``architectures[0]``, not a name heuristic.
"""
import importlib

import pytest

from tessera.serving import dense_ownership

export = importlib.import_module("tessera.export_serving")

K = "model.layers.0.self_attn.k_proj.weight"
Q = "model.layers.0.self_attn.q_proj.weight"
V = "model.layers.0.self_attn.v_proj.weight"
QKV = ("model.layers.0.self_attn.qkv_proj",
       (Q, K, V))
GATE = "model.layers.0.mlp.gate_proj.weight"
GLM5 = "Glm5NextForConditionalGeneration"


def test_the_separate_qkv_set_is_explicit_data():
    assert isinstance(dense_ownership.SEPARATE_QKV_ARCHITECTURES, frozenset)
    assert GLM5 in dense_ownership.SEPARATE_QKV_ARCHITECTURES


@pytest.mark.parametrize("architecture", [None, "Qwen3ForCausalLM"])
def test_default_and_fusing_architectures_still_name_qkv_proj(architecture):
    kwargs = {} if architecture is None else {"architecture": architecture}
    got = dense_ownership.fused_module(K, **kwargs)
    assert got == QKV
    assert got is not None
    members = got[1]
    assert len(members) == 3
    assert [m.rsplit(".", 2)[-2] for m in members] == ["q_proj", "k_proj", "v_proj"]


def test_the_no_argument_call_is_the_default():
    assert dense_ownership.fused_module(K) == dense_ownership.fused_module(K, architecture=None)


@pytest.mark.parametrize("name", [Q, K, V])
def test_a_separate_qkv_architecture_does_not_fuse_attention(name):
    assert dense_ownership.fused_module(name, architecture=GLM5) is None


def test_a_separate_qkv_architecture_still_fuses_gate_up():
    got = dense_ownership.fused_module(GATE, architecture=GLM5)
    assert got is not None
    assert got[0] == "model.layers.0.mlp.gate_up_proj"
    assert got[1] == (GATE, "model.layers.0.mlp.up_proj.weight")
    # Identical to the default for that row: only the qkv row is scoped.
    assert got == dense_ownership.fused_module(GATE)


def test_export_shim_passes_the_architecture_through():
    assert export.fused_module(K) == QKV
    assert export.fused_module(K, architecture="Qwen3ForCausalLM") == QKV
    assert export.fused_module(K, architecture=GLM5) is None
    assert export.fused_module(GATE, architecture=GLM5)[0] == "model.layers.0.mlp.gate_up_proj"


def test_export_ignored_modules_agrees_with_fused_module():
    assert export.ignored_modules(Q, (8, 8)) == ("model.layers.0.self_attn.qkv_proj",)
    assert export.ignored_modules(Q, (8, 8), architecture="Qwen3ForCausalLM") == \
        ("model.layers.0.self_attn.qkv_proj",)
    assert export.ignored_modules(Q, (8, 8), architecture=GLM5) == \
        ("model.layers.0.self_attn.q_proj",)
    # The gate/up row is untouched by the architecture.
    assert export.ignored_modules(GATE, (8, 8), architecture=GLM5) == \
        ("model.layers.0.mlp.gate_up_proj",)
