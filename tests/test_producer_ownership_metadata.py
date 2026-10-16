"""Source naming is public producer metadata and needs no serving runtime."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("architecture", [None, "Qwen3ForCausalLM"])
def test_ordinary_members_and_roles(architecture):
    from tessera.metadata import fused_module, role_name

    prefix = "model.layers.0.self_attn."
    members = tuple(prefix + leaf + ".weight" for leaf in ("q_proj", "k_proj", "v_proj"))
    for member in members:
        assert fused_module(member, architecture) == (prefix + "qkv_proj", members)
        assert role_name(member) == member.removesuffix(".weight").rsplit(".", 1)[-1]
    assert fused_module("model.layers.0.self_attn.o_proj.weight", architecture) is None
    assert role_name("bare") == "bare"
    assert role_name("model.proj") == "proj"


def test_glm_groups_are_the_naming_owners_in_declared_order():
    from tessera.metadata import fused_module, glm_fused_groups

    groups = glm_fused_groups()
    assert isinstance(groups, tuple)
    assert tuple(target for target, _members in groups) == (
        "in_proj_qkvbfg_a", "fused_qkv_a_proj", "wk_weights_proj")
    for target, leaves in groups:
        assert isinstance(leaves, tuple)
        prefix = "model.layers.0.self_attn." + ("indexer." if target == "wk_weights_proj" else "")
        members = tuple(prefix + leaf + ".weight" for leaf in leaves)
        for member in members:
            assert fused_module(member, "Glm5NextForConditionalGeneration",
                                config={"layer_types": ["linear_attention"]}) == (prefix + target, members)


@pytest.mark.parametrize("config,names,fused", [
    (None, None, False),
    (None, ("model.layers.0.self_attn.b_proj.weight",), True),
    ({"layer_types": ["linear_attention"]}, None, True),
    ({"layer_types": ["full_attention"]}, ("model.layers.0.self_attn.b_proj.weight",), False),
    ({"text_config": {"layer_types": ["linear_attention"]}}, None, True),
])
def test_glm_config_precedes_source_name_fallback(config, names, fused):
    from tessera.metadata import fused_module

    got = fused_module("model.layers.0.self_attn.q_proj.weight",
                       "Glm5NextForConditionalGeneration", config=config, tensor_names=names)
    assert (got is not None) is fused


@pytest.mark.parametrize("config,allowed", [
    (None, True), ({}, True), ({"q_lora_rank": None}, False),
    ({"text_config": {"q_lora_rank": None}}, False), ({"q_lora_rank": 16}, True),
])
def test_glm_mla_query_rank(config, allowed):
    from tessera.metadata import fused_module

    got = fused_module("model.layers.0.self_attn.q_a_proj.weight",
                       "Glm5NextForConditionalGeneration", config=config)
    assert (got is not None) is allowed


def test_cold_naming_metadata_import_blocks_serving_and_numerical_runtimes():
    program = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tessera.serving" or fullname.startswith("tessera.serving.") or fullname.split(".")[0] in {"torch", "transformers", "triton", "vllm"}:
            raise AssertionError("blocked dependency " + fullname)
sys.meta_path.insert(0, Block())
from tessera.metadata import fused_module, role_name, glm_fused_groups, __all__
assert {"fused_module", "role_name", "glm_fused_groups"} <= set(__all__)
assert fused_module("model.layers.0.self_attn.q_proj.weight") is not None
assert role_name("model.proj.weight") == "proj"
assert glm_fused_groups()
'''
    result = subprocess.run([sys.executable, "-c", program], env=os.environ.copy(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
