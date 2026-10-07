"""Dense declarations must name the modules the consumer constructs."""
from __future__ import annotations

import json

import pytest

from tessera import export_serving as exporter


def _source(tmp_path, tensors, text=None):
    from safetensors.torch import save_file
    src = tmp_path / "src"
    src.mkdir()
    save_file(tensors, str(src / "model.safetensors"))
    config = {"architectures": ["Glm5NextForConditionalGeneration"],
              "text_config": {"hidden_size": 32, "moe_intermediate_size": 32,
                              "layer_types": ["linear_attention"], **(text or {})}}
    (src / "config.json").write_text(json.dumps(config))
    return src


def _run(tmp_path, monkeypatch, tensors, plan, **kw):
    src = _source(tmp_path, tensors, kw.get("text"))
    out = tmp_path / "out"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
                                    "--q256", "1024", "--device", "cpu", "--no-verify",
                                    "--plan-json", str(plan_path)])
    exporter.main()
    return out


def test_bf16_kda_members_have_one_consumer_ignore_and_exact_bytes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    prefix = "model.language_model.layers.0."
    roles = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
    tensors = {prefix + "self_attn." + role + ".weight":
               torch.arange(1024).reshape(32, 32).bfloat16() for role in roles}
    down = prefix + "mlp.down_proj.weight"
    tensors[down] = torch.zeros(32, 32, dtype=torch.bfloat16)
    out = _run(tmp_path, monkeypatch, tensors,
               {name: "PASSTHROUGH" for name in tensors if name != down})
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    assert prefix + "self_attn.in_proj_qkvbfg_a" in config["ignore"]
    assert not any(prefix + "self_attn." + role in config["ignore"] for role in roles)
    written = load_file(str(out / "model.safetensors"))
    for name in tensors:
        if name != down:
            assert written[name].dtype == tensors[name].dtype
            assert torch.equal(written[name], tensors[name])


def test_explicit_vision_projection_is_in_the_producer_roster(tmp_path):
    torch = pytest.importorskip("torch")
    vision = "model.visual.blocks.0.attn.qkv.weight"
    src = _source(tmp_path, {vision: torch.zeros(96, 32, dtype=torch.bfloat16)})
    _shards, shapes, _experts, _routed = exporter.quantizable(src, selected={vision})
    assert shapes == {vision: (96, 32)}


def test_mla_query_is_not_a_kda_or_generic_qkv_member():
    from tessera.serving.dense_ownership import fused_module
    config = {"layer_types": ["deepseek_sparse_attention"], "q_lora_rank": None}
    name = "model.layers.0.self_attn.q_proj.weight"
    assert fused_module(name, "Glm5NextForConditionalGeneration", config=config) is None


def test_nope_padding_applies_only_to_the_kv_input_member():
    from tessera.serving.dense_ownership import partition_members
    module = "model.layers.0.self_attn.fused_qkv_a_proj"
    members = ["model.layers.0.self_attn.q_a_proj.weight",
               "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]
    rows = dict(zip(members, [32, 32]))
    parts = partition_members(module, members, rows, [32, 64],
                              padding_rows={members[1]: 32})
    assert [(part.rows, part.source_rows) for part in parts] == [(32, 32), (64, 32)]
    with pytest.raises(ValueError, match="rows"):
        partition_members(module, members, rows, [64, 32], padding_rows={members[1]: 32})
