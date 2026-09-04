"""A serving part is incomplete by construction; only its checked union loads."""
import json
import stat
import struct
from pathlib import Path

import pytest

from tessera import serving_parts as parts


def _tensor_file(path, names):
    header = {name: {"dtype": "BF16", "shape": [1],
                     "data_offsets": [2 * i, 2 * i + 2]}
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0" * len(names))


def _fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = ["model.layers.0.norm.weight", "model.layers.1.norm.weight", "lm_head.weight"]
    _tensor_file(source / "model.safetensors", names)
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    identity = {"source": parts.source_identity(source), "code_sha256": "a" * 64,
                "runtime_image": "test/image@sha256:" + "b" * 64,
                "options": {"plan": {}}}
    paths = []
    for rank in range(2):
        path = tmp_path / f"part{rank}"
        path.mkdir()
        owned = [name for name in names if parts.partition_owner(name, 2) == rank]
        _tensor_file(path / "model.safetensors", owned)
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in owned}}))
        config = {"architectures": ["Example"], "quantization_config": {
            "quant_method": "tessera", "format": "mixed-precision", "config_groups": {},
            "ignore": [name.removesuffix(".weight") for name in owned]}}
        (path / "tessera_part_config.json").write_text(json.dumps(config))
        manifest = {"source": str(source), "git": "abc", "modules": {},
                    "totals": {"passthrough_bytes": len(owned) * 2},
                    "routed_moe": {"quantized_stacks": [], "modules": [],
                                   "packed_source_tensors": 0, "unpacked_source_tensors": 0,
                                   "quantized_source_tensors": 0, "quantized_logical_units": 0},
                    "export_partition": {"schema": parts.SCHEMA, "index": rank, "count": 2,
                        "identity": identity, "source_tensors": owned,
                        "output_sha256": {"model.safetensors": parts.sha256_file(path / "model.safetensors")}}}
        (path / "tessera_serving_manifest.json").write_text(json.dumps(manifest))
        paths.append(path)
    return source, paths


def _change(path, mutate):
    manifest_path = path / "tessera_serving_manifest.json"
    value = json.loads(manifest_path.read_text())
    mutate(value)
    manifest_path.write_text(json.dumps(value))


def test_partitions_keep_whole_layers_and_balance_lfm_routed_stack_range():
    assert [sum(parts.partition_owner(f"model.layers.{n}.feed_forward.experts.0.w1.weight", 2)
                == rank for n in range(2, 24)) for rank in range(2)] == [11, 11]
    assert parts.partition_owner("model.language_model.layers.3.mlp.experts.1.up_proj.weight", 2) == 1
    assert parts.partition_owner("lm_head.weight", 2) == 0
    assert parts.parse_partition("1/2") == (1, 2)


@pytest.mark.parametrize("value", ["2/2", "0/0", "-1/2", "0", "0/1/2"])
def test_invalid_partition_refuses(value):
    with pytest.raises(ValueError, match="partition"):
        parts.parse_partition(value)


def test_checked_union_writes_one_complete_checkpoint_without_reencoding(tmp_path):
    source, paths = _fixture(tmp_path)
    out = tmp_path / "merged"
    parts.merge_serving_parts(paths[::-1], out, source)
    config = json.loads((out / "config.json").read_text())
    index = json.loads((out / "model.safetensors.index.json").read_text())
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert set(index["weight_map"]) == set(parts.source_identity(source)["tensors"])
    assert len(set(index["weight_map"].values())) == 2
    assert config["quantization_config"]["ignore"] == ["lm_head", "model.layers.0.norm", "model.layers.1.norm"]
    assert manifest["totals"]["passthrough_bytes"] == 6
    assert manifest["totals"]["checkpoint_bytes"] == sum((out / n).stat().st_size for n in set(index["weight_map"].values()))
    assert "export_partition" not in manifest
    assert not (paths[0] / "config.json").exists()
    for rank, path in enumerate(paths):
        assert (out / f"part-{rank:05d}-model.safetensors").read_bytes() == (path / "model.safetensors").read_bytes()


@pytest.mark.parametrize("mode", [0o600, 0o400, 0o640])
def test_merged_shards_add_read_bits_without_changing_private_parts(tmp_path, mode):
    source, paths = _fixture(tmp_path)
    originals = {}
    for path in paths:
        shard = path / "model.safetensors"
        shard.chmod(mode)
        originals[shard] = shard.read_bytes()
    out = tmp_path / "merged"
    parts.merge_serving_parts(paths, out, source)
    read_bits = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for rank, path in enumerate(paths):
        shard = path / "model.safetensors"
        served = out / f"part-{rank:05d}-model.safetensors"
        assert stat.S_IMODE(served.stat().st_mode) == mode | read_bits
        assert served.read_bytes() == originals[shard]
        assert shard.read_bytes() == originals[shard]
        assert stat.S_IMODE(shard.stat().st_mode) == mode


@pytest.mark.parametrize("field", ["code_sha256", "runtime_image", "options", "source"])
def test_identity_drift_refuses_before_output_exists(tmp_path, field):
    source, paths = _fixture(tmp_path)
    _change(paths[1], lambda m: m["export_partition"]["identity"].update({field: "different"}))
    with pytest.raises(ValueError, match="identity"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert not (tmp_path / "merged").exists()


def test_missing_part_and_duplicate_part_refuse(tmp_path):
    source, paths = _fixture(tmp_path)
    for chosen in (paths[:1], [paths[0], paths[0]]):
        with pytest.raises(ValueError, match="partition"):
            parts.merge_serving_parts(chosen, tmp_path / "merged", source)


def test_wrong_source_coverage_refuses(tmp_path):
    source, paths = _fixture(tmp_path)
    _change(paths[1], lambda m: m["export_partition"].update({"source_tensors": []}))
    with pytest.raises(ValueError, match="source.*coverage"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_output_corruption_refuses(tmp_path):
    source, paths = _fixture(tmp_path)
    with (paths[0] / "model.safetensors").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_source_replaced_after_export_refuses(tmp_path):
    source, paths = _fixture(tmp_path)
    (source / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="source identity"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_existing_output_refuses(tmp_path):
    source, paths = _fixture(tmp_path)
    out = tmp_path / "merged"
    out.mkdir()
    with pytest.raises(ValueError, match="exists"):
        parts.merge_serving_parts(paths, out, source)


def test_exporter_writes_only_owned_tensors_and_withholds_loadable_config(tmp_path, monkeypatch):
    import importlib.util
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    script = Path(__file__).resolve().parents[1] / "experiments/export_tessera_serving.py"
    spec = importlib.util.spec_from_file_location("serving_export_partition_test", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    source = tmp_path / "source"
    source.mkdir()
    tensors = {f"model.layers.{layer}.mlp.down_proj.weight": torch.ones(32, 16)
               for layer in range(2)}
    tensors["lm_head.weight"] = torch.ones(32, 16)
    safetensors.save_file(tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    paths = []
    for rank in range(2):
        out = tmp_path / f"export{rank}"
        monkeypatch.setattr("sys.argv", ["export", str(source), str(out), "--grid", "E4M3",
            "--q256", "1024", "--layers", "0", "--device", "cpu", "--partition", f"{rank}/2",
            "--partition-runtime-image", "test/image@sha256:" + "b" * 64])
        exporter.main()
        assert not (out / "config.json").exists()
        manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
        owned = {name for name in tensors if parts.partition_owner(name, 2) == rank}
        assert set(manifest["export_partition"]["source_tensors"]) == owned
        assert parts.tensor_names(out / "model.safetensors") == owned
        paths.append(out)
    parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_partitioned_expert_wires_equal_one_process_export(tmp_path, monkeypatch):
    """Actual CPU encode, then compare every emitted tensor and declared scheme."""
    import importlib.util
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    script = Path(__file__).resolve().parents[1] / "experiments/export_tessera_serving.py"
    spec = importlib.util.spec_from_file_location("expert_partition_export", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    source = tmp_path / "source"
    source.mkdir()
    generator = torch.Generator().manual_seed(5)
    stacks = [f"model.language_model.layers.{layer}.mlp.experts" for layer in range(2)]
    tensors = {f"{stack}.0.{projection}.weight": torch.randn(32, 32, generator=generator) * 0.02
               for stack in stacks for projection in exporter.EXPERT_PROJECTIONS}
    tensors["lm_head.weight"] = torch.randn(32, 32, generator=generator)
    safetensors.save_file(tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps({"architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": 32, "moe_intermediate_size": 32, "n_routed_experts": 1}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({stack: {"grid": "E4M3", "q256": 1024} for stack in stacks}))
    common = ["--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--plan-json", str(plan)]
    paths = []
    for rank in range(2):
        out = tmp_path / f"export{rank}"
        monkeypatch.setattr("sys.argv", ["export", str(source), str(out), *common,
            "--partition", f"{rank}/2", "--partition-runtime-image", "test/image@sha256:" + "b" * 64])
        exporter.main()
        paths.append(out)
    merged = tmp_path / "merged"
    manifest = parts.merge_serving_parts(paths, merged, source)
    whole = tmp_path / "whole"
    monkeypatch.setattr("sys.argv", ["export", str(source), str(whole), *common])
    exporter.main()
    index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
    actual = {}
    for filename in set(index.values()):
        actual.update(safetensors.load_file(str(merged / filename)))
    expected = safetensors.load_file(str(whole / "model.safetensors"))
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[name], expected[name]) for name in actual)
    whole_manifest = json.loads((whole / "tessera_serving_manifest.json").read_text())
    assert manifest["modules"] == whole_manifest["modules"]
    assert manifest["routed_moe"] == whole_manifest["routed_moe"]
    for field in ("wire_bytes", "on_disk_bytes", "quantized_params", "by_family", "passthrough_bytes"):
        assert manifest["totals"][field] == whole_manifest["totals"][field]
    assert json.loads((merged / "config.json").read_text()) == json.loads((whole / "config.json").read_text())


def _moe_plan_parts(tmp_path, encoded=None, count=2):
    """Two source stacks; an omitted encode is internally consistent BF16."""
    source = tmp_path / "source"
    source.mkdir()
    encoded = range(count) if encoded is None else encoded
    stacks = [f"model.layers.{i}.feed_forward.experts" for i in range(count)]
    source_names = [f"{stack}.0.{shard}.weight" for stack in stacks for shard in ("w1", "w3", "w2")]
    _tensor_file(source / "model.safetensors", source_names)
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    plan = {stack: {"grid": "E4M3", "q256": 1024} for stack in stacks}
    identity = {"source": parts.source_identity(source), "code_sha256": "a" * 64,
                "runtime_image": "test/image@sha256:" + "b" * 64, "options": {"plan": plan}}
    paths = []
    for rank, stack in enumerate(stacks):
        path = tmp_path / f"part{rank}"
        path.mkdir()
        owned = [n for n in source_names if n.startswith(stack + ".")]
        modules, groups, ignore = {}, {}, []
        if rank in encoded:
            names = [n.removesuffix(".weight") + ".wire" for n in owned]
            roles = [{"tensor": tensor, "source_tensor": tensor, "expert": 0,
                      "role": role, "group": "w2" if role == "down_proj" else "w13",
                      "grid": "E4M3", "q256": 1024, "rows": 32, "cols": 32}
                     for tensor, role in zip(owned, ("gate_proj", "up_proj", "down_proj"))]
            modules[stack] = {"structure": "routed_moe", "family": "TESSERA_FP8",
                "grid": "E4M3", "q256": 1024, "experts": 1, "roles": roles,
                "wire_bytes": 6, "container_bytes": 6, "resident_bytes_resident_mode": 3072}
            groups[f"stack{rank}"] = {"targets": [stack], "format": "TESSERA", "scheme": {
                "structure": "routed_moe", "family": "TESSERA_FP8", "grid": "E4M3",
                "body": "WINDOW", "plane": "CHANNEL", "experts": 1, "groups": {
                    "w13": {"q256": 1024, "rows": 64, "columns": 32, "wire_stride": 2,
                            "roles": [["gate_proj", 32], ["up_proj", 32]]},
                    "w2": {"q256": 1024, "rows": 32, "columns": 32, "wire_stride": 2,
                           "roles": [["down_proj", 32]]}}}}
        else:
            names, ignore = owned, [stack]
        _tensor_file(path / "model.safetensors", names)
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in names}}))
        (path / "tessera_part_config.json").write_text(json.dumps({"architectures": ["Example"],
            "quantization_config": {"quant_method": "tessera", "format": "mixed-precision",
                                    "config_groups": groups, "ignore": ignore}}))
        manifest = {"modules": modules, "totals": {"passthrough_bytes": 0 if modules else 6},
                    "routed_moe": {"quantized_stacks": list(modules), "modules": ignore,
                        "packed_source_tensors": 0, "unpacked_source_tensors": 3,
                        "quantized_source_tensors": 3 if modules else 0,
                        "quantized_logical_units": 3 if modules else 0},
                    "export_partition": {"schema": parts.SCHEMA, "index": rank, "count": count,
                        "identity": identity, "source_tensors": owned,
                        "output_sha256": {"model.safetensors": parts.sha256_file(path / "model.safetensors")}}}
        (path / "tessera_serving_manifest.json").write_text(json.dumps(manifest))
        paths.append(path)
    return source, paths, plan


def test_explicit_plan_cannot_lose_a_whole_stack_to_bf16(tmp_path):
    source, paths, _plan = _moe_plan_parts(tmp_path, encoded=range(21), count=22)
    with pytest.raises(ValueError, match="plan.*stack"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert not (tmp_path / "merged").exists()


def test_complete_explicit_stack_plan_still_merges_partial_matrix(tmp_path):
    source, paths, plan = _moe_plan_parts(tmp_path)
    manifest = parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert set(manifest["modules"]) == set(plan)
    assert manifest["totals"]["units"] == 6


@pytest.mark.parametrize("field,value", [("grid", "BF16"), ("q256", 896)])
def test_explicit_plan_checks_each_emitted_role(tmp_path, field, value):
    source, paths, _plan = _moe_plan_parts(tmp_path)
    def mutate(manifest):
        next(iter(manifest["modules"].values()))["roles"][0][field] = value
    _change(paths[0], mutate)
    with pytest.raises(ValueError, match="plan"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_explicit_plan_checks_declared_group_rungs(tmp_path):
    source, paths, _plan = _moe_plan_parts(tmp_path)
    path = paths[0] / "tessera_part_config.json"
    config = json.loads(path.read_text())
    next(iter(config["quantization_config"]["config_groups"].values()))["scheme"]["groups"]["w13"]["q256"] = 896
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="plan"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_explicit_plan_requires_every_source_expert(tmp_path):
    source, paths, _plan = _moe_plan_parts(tmp_path)
    # Omit a role from the source ownership recorded by the module while the
    # corresponding source weight passes through and all file/index checks agree.
    manifest_path = paths[0] / "tessera_serving_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(iter(manifest["modules"].values()))
    role = record["roles"].pop()
    index_path = paths[0] / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop(role["tensor"].removesuffix(".weight") + ".wire")
    index["weight_map"][role["tensor"]] = "model.safetensors"
    index_path.write_text(json.dumps(index))
    _tensor_file(paths[0] / "model.safetensors", list(index["weight_map"]))
    manifest["export_partition"]["output_sha256"]["model.safetensors"] = parts.sha256_file(paths[0] / "model.safetensors")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="plan.*coverage"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
