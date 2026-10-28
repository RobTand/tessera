"""Dense declarations must name the modules the consumer constructs.

CPU vision fixtures cover export, reader validation and rank ownership only.
They do not qualify a loaded vLLM forward, CUDA graph, quality or serving cell.
"""
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
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", kw.get("grid", "E4M3"),
                                    "--q256", "1024", "--device", "cpu", "--no-verify",
                                    "--plan-json", str(plan_path), *kw.get("extra", ())])
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
    import copy
    from tessera.serving.contract import construction_entry

    # This fixture declares a small MLP output partition, not the production model.
    small_runtime = copy.deepcopy(construction_entry(["Glm5NextForConditionalGeneration"]))
    small_runtime["output_sizes"]["language_model.model.layers.*.mlp.down_proj"] = [32]
    monkeypatch.setattr(exporter, "construction_entry", lambda _architectures: small_runtime)
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


def test_mla_input_refuses_an_undeclared_total_only_partition():
    from tessera.serving.dense_ownership import partition_members
    module = "model.layers.0.self_attn.fused_qkv_a_proj"
    members = ["model.layers.0.self_attn.q_a_proj.weight",
               "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]
    rows = dict(zip(members, [32, 32]))
    with pytest.raises(ValueError, match="cannot be paired"):
        partition_members(module, members, rows, [96], padding_rows={members[1]: 32})


def test_bf16_vision_control_keeps_the_prefix_bias_and_numerical_output(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    prefix = "model.visual.blocks.0.attn.qkv"
    weight = torch.arange(96 * 32).reshape(96, 32).bfloat16()
    bias = torch.arange(96).bfloat16()
    src = _source(tmp_path, {prefix + ".weight": weight, prefix + ".bias": bias})
    out = tmp_path / "out"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({prefix + ".weight": "PASSTHROUGH"}))
    monkeypatch.setattr("sys.argv", ["export", str(src), str(out), "--grid", "E4M3",
        "--q256", "1024", "--device", "cpu", "--layers", "0", "--plan-json", str(plan)])
    exporter.main()
    written = load_file(str(out / "model.safetensors"))
    assert set(written) == {prefix + ".weight", prefix + ".bias"}
    assert written[prefix + ".weight"].dtype == written[prefix + ".bias"].dtype == torch.bfloat16
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    assert torch.equal(x @ written[prefix + ".weight"].t() + written[prefix + ".bias"], x @ weight.t() + bias)
    assert "quantization_config" not in json.loads((out / "config.json").read_text())


def _vision_partitions():
    from tessera.serving.contract import construction_entry

    entry = construction_entry(["Glm5NextForConditionalGeneration"])
    return {name: sizes for name, sizes in entry["output_sizes"].items()
            if name.startswith("visual.")}


def _vision_scalar_grids():
    from tessera.serving.scheme import ROUTES

    return [grid for route in ROUTES.values() if route["plane"] == "CHANNEL"
            for grid in route["grids"]]


def _vision_fixture(monkeypatch, pattern):
    import copy
    import torch
    from tessera.serving.contract import construction_entry, normalise_module

    entry = copy.deepcopy(construction_entry(["Glm5NextForConditionalGeneration"]))
    sizes = [32 for _size in entry["output_sizes"][pattern]]
    entry["output_sizes"][pattern] = sizes
    monkeypatch.setattr(exporter, "construction_entry", lambda _architectures: entry)
    module = "model." + pattern.replace("*", "0")
    assert normalise_module(module.removeprefix("model.")) == pattern
    if module.endswith(".gate_up_proj"):
        members = [(module.replace(".gate_up_proj", ".gate_proj") + ".weight", sizes[0]),
                   (module.replace(".gate_up_proj", ".up_proj") + ".weight", sizes[1])]
        expected_roles = ["gate_proj", "up_proj"]
    else:
        members = [(module + ".weight", sum(sizes))]
        leaf = module.rsplit(".", 1)[-1]
        expected_roles = [f"{leaf}.{index}" for index in range(len(sizes))] if len(sizes) > 1 else [leaf]
    tensors = {}
    for index, (name, rows) in enumerate(members):
        values = torch.arange(rows * 32, dtype=torch.float32).reshape(rows, 32)
        tensors[name] = (values.sin() + index + 1).bfloat16()
    tensors[module + ".bias"] = torch.arange(sum(sizes)).bfloat16()
    stock = "model.visual.blocks.1.attn.proj"
    tensors[stock + ".weight"] = torch.arange(1024).reshape(32, 32).bfloat16()
    tensors[stock + ".bias"] = torch.arange(32).bfloat16()
    tensors["model.visual.patch_embed.proj.weight"] = torch.ones(32, 3, 2, 2, 2).bfloat16()
    return module, tensors, [name for name, _rows in members], sizes, expected_roles, entry


@pytest.mark.parametrize("pattern", _vision_partitions())
@pytest.mark.parametrize("grid", _vision_scalar_grids())
def test_explicit_vision_bytes_roles_and_rank_slices(tmp_path, monkeypatch, pattern, grid):
    import torch
    from safetensors.torch import load_file
    from tessera.decode import reconstruct_unit
    from tessera.fused import parse_fused
    from tessera.serving.scheme import parse_compact_blob_for_scheme, parse_tessera_blob_for_scheme
    from tessera.serving.sharding import plan_shard, shard_parsed_roles

    module, tensors, members, sizes, expected_roles, entry = _vision_fixture(monkeypatch, pattern)
    expected_partitions = {}
    row_offset = 0
    for index, (role, size) in enumerate(zip(expected_roles, sizes)):
        member = members[0] if len(members) == 1 else members[index]
        start = row_offset if len(members) == 1 else 0
        stop = start + size
        unit_name = member if size == tensors[member].shape[0] else f"{member}[{start}:{stop}]"
        expected_partitions[role] = (unit_name, tensors[member][start:stop].float())
        row_offset += size
    encoded_units = {}
    original_encode = exporter.encode_linear_planes

    def observe_encode(weight, **kwargs):
        unit_name = kwargs["name"]
        assert unit_name not in encoded_units
        result = original_encode(weight, **kwargs)
        encoded_units[unit_name] = (weight.clone(), result[0].blob)
        return result

    monkeypatch.setattr(exporter, "encode_linear_planes", observe_encode)
    plan = {name: {"grid": grid, "q256": 1024} for name in members}
    out = _run(tmp_path, monkeypatch, tensors, plan, grid=grid)
    written = load_file(str(out / "model.safetensors"))
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    groups = list(config["config_groups"].values())
    assert len(groups) == 1
    assert groups[0]["targets"] == [module]
    scheme = groups[0]["scheme"]
    assert scheme["roles"] == [[role, size] for role, size in zip(expected_roles, sizes)]
    assert set(written) == (set(tensors) - set(members)) | {module + ".wire_bytes"}
    for name in set(tensors) - set(members):
        assert written[name].dtype == tensors[name].dtype
        assert torch.equal(written[name], tensors[name]), name
    assert "model.visual.blocks.1.attn.proj" in config["ignore"]
    assert module not in config["ignore"]
    assert all(name.removesuffix(".weight") not in config["ignore"] for name in members)
    assert set(encoded_units) == {name for name, _weight in expected_partitions.values()}
    for unit_name, expected in expected_partitions.values():
        assert torch.equal(encoded_units[unit_name][0], expected), unit_name
    blob = written[module + ".wire_bytes"].numpy().tobytes()
    assert len(blob) == scheme["wire_bytes"] == manifest["modules"][module]["container_bytes"]
    framed = parse_fused(blob)
    assert [member.name for member in framed] == expected_roles
    for member in framed:
        unit_name, _weight = expected_partitions[member.name]
        assert member.blob == encoded_units[unit_name][1], (module, member.name, unit_name)
    compact = parse_compact_blob_for_scheme(blob, scheme, module, device="cpu")
    parsed = parse_tessera_blob_for_scheme(blob, scheme, module, device="cpu")
    assert [name for name, _wire in compact] == expected_roles
    assert [name for name, _unit in parsed] == expected_roles
    geometry = [(wire.metadata.rows, wire.metadata.columns) for _name, wire in compact]
    assert geometry == [(size, 32) for size in sizes]
    for reader in (parse_compact_blob_for_scheme, parse_tessera_blob_for_scheme):
        with pytest.raises(ValueError):
            reader(blob[:-1], scheme, module, device="cpu")
        if len(expected_roles) > 1:
            wrong_roles = {**scheme, "roles": list(reversed(scheme["roles"]))}
            with pytest.raises(ValueError, match="role"):
                reader(blob, wrong_roles, module, device="cpu")
    for axis in ("row", "column"):
        for rank in range(2):
            shard_plan = plan_shard(module, roles=scheme["roles"], columns=32,
                                    out_partitions=[size // 2 for size in sizes] if axis == "row" else sizes,
                                    in_size=32 if axis == "row" else 16, tp_rank=rank, tp_size=2,
                                    input_size=32, output_size=sum(sizes))
            assert shard_plan.axis == axis
            sharded = shard_parsed_roles(parsed, shard_plan)
            assert [role for role, _shard in sharded] == expected_roles, (module, axis, rank)
            for (role, parent), (shard_role, shard) in zip(parsed, sharded):
                assert shard_role == role
                decoded = reconstruct_unit(parent.unit, parent.forests, parent.code)
                actual = reconstruct_unit(shard.unit, shard.forests, shard.code)
                if axis == "row":
                    half_rows = decoded.shape[0] // 2
                    expected = decoded[rank * half_rows:(rank + 1) * half_rows]
                else:
                    expected = decoded[:, rank * 16:(rank + 1) * 16]
                assert torch.equal(actual, expected), (module, role, axis, rank)
    gate = manifest["serving_gate"]
    assert gate["construction_census"] == {key: entry[key] for key in ("architecture", "runtime", "model", "receipt")}
    assert gate["unrouted"] == []
    assert gate["geometry"]["unattested_modules"] == []


@pytest.mark.parametrize("selected", [False, True])
def test_vision_roster_requires_exact_explicit_weight_selection(tmp_path, selected):
    import torch

    weight = "model.visual.blocks.0.attn.qkv.weight"
    other = "model.visual.blocks.1.attn.qkv.weight"
    tensors = {weight: torch.ones(96, 32).bfloat16(), other: torch.ones(96, 32).bfloat16(),
               "model.visual.patch_embed.proj.weight": torch.ones(32, 3, 2, 2, 2).bfloat16()}
    src = _source(tmp_path, tensors)
    selection = {weight} if selected else {weight.removesuffix(".weight")}
    _shards, shapes, experts, routed = exporter.quantizable(src, selected=selection)
    assert shapes == ({weight: (96, 32)} if selected else {})
    assert experts == routed == {}


def test_selected_vision_census_is_not_loaded_forward_evidence():
    from pathlib import Path
    from tessera.serving.contract import construction_entry

    entry = construction_entry(["Glm5NextForConditionalGeneration"])
    receipt = json.loads((Path(__file__).resolve().parents[1] / entry["receipt"]).read_text())
    assert receipt["weights_loaded"] is False
    assert receipt["forward_executed"] is False
    assert _vision_partitions().keys() <= set(entry["offered"])


@pytest.mark.parametrize("call_style", ["keyword", "positional"])
def test_vision_constructor_hook_selects_only_named_prefixes(monkeypatch, call_style):
    import sys
    import types
    import torch
    from tessera.serving import projection_routes

    class LinearBase:
        def __init__(self, input_size, output_size, bias=True, params_dtype=None,
                     quant_config=None, prefix=""):
            self.arguments = (input_size, output_size, bias, params_dtype, quant_config, prefix)

    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.linear"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.model_executor.layers.linear"].LinearBase = LinearBase
    selected_prefix = "visual.blocks.0.attn.qkv"
    selected_config, supplied_config = object(), object()
    lookups = []

    def selected_config_for(prefix):
        lookups.append(prefix)
        return selected_config if prefix == selected_prefix else None

    monkeypatch.setattr(projection_routes, "_selected_config", selected_config_for)
    projection_routes.install()
    constructor = LinearBase.__init__
    projection_routes.install()
    assert LinearBase.__init__ is constructor
    for prefix, supplied, expected in ((selected_prefix, None, selected_config),
                                       ("visual.blocks.1.attn.qkv", None, None),
                                       (selected_prefix, supplied_config, supplied_config),
                                       ("visual.blocks.1.attn.qkv", supplied_config, supplied_config)):
        if call_style == "positional":
            layer = LinearBase(32, 96, False, torch.bfloat16, supplied, prefix)
        else:
            layer = LinearBase(32, 96, bias=False, params_dtype=torch.bfloat16,
                               quant_config=supplied, prefix=prefix)
        assert layer.arguments == (32, 96, False, torch.bfloat16, expected, prefix)
    assert lookups == [selected_prefix, "visual.blocks.1.attn.qkv"]
