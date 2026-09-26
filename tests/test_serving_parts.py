"""A serving part is incomplete by construction; only its checked union loads."""
import json
import hashlib
import stat
import struct
from pathlib import Path

import pytest

from tessera import serving_parts as parts


@pytest.mark.parametrize("carrier", ["config", "manifest", "identity"])
def test_research_execution_cannot_be_added_to_only_one_carrier(tmp_path, carrier):
    source, paths = _fixture(tmp_path)
    block = {"schema": "tessera.research_selected_moe.v1", "max_experts_per_chunk": 3,
             "decode_backend": "triton", "expected_tensor_parallel_size": 2}
    text = json.dumps(block)
    record = {"input_utf8": text, "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "config": block}
    for path in paths:
        if carrier == "config":
            config_path = path / "tessera_part_config.json"
            config = json.loads(config_path.read_text())
            config["quantization_config"]["research_selected_moe"] = block
            config_path.write_text(json.dumps(config))
        elif carrier == "manifest":
            _change(path, lambda m: m.update(research_selected_moe=record))
        else:
            _change(path, lambda m: m["export_partition"]["identity"]["options"].update(
                research_selected_moe=record))
    out = tmp_path / "merged"
    with pytest.raises(ValueError, match="research_selected_moe"):
        parts.merge_serving_parts(paths, out, source)
    assert not out.exists()


def _tensor_file(path, names):
    header = {name: {"dtype": "BF16", "shape": [1],
                     "data_offsets": [2 * i, 2 * i + 2]}
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0" * len(names))


SHARD_A, SHARD_B = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"


def _fixture(tmp_path, two_shards=False):
    """Two parts; with ``two_shards`` each part reads a different source shard."""
    source = tmp_path / "source"
    source.mkdir()
    names = ["model.layers.0.norm.weight", "model.layers.1.norm.weight", "lm_head.weight"]
    if two_shards:
        layout = {SHARD_A: [names[0], names[2]], SHARD_B: [names[1]]}
        for shard, held in layout.items():
            _tensor_file(source / shard, held)
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: shard for shard, held in layout.items() for name in held}}))
    else:
        _tensor_file(source / "model.safetensors", names)
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    inventory = parts.source_inventory(source)
    shared = {"code_sha256": "a" * 64, "runtime_image": "test/image@sha256:" + "b" * 64,
              "options": {"plan": {}}}
    paths = []
    for rank in range(2):
        path = tmp_path / f"part{rank}"
        path.mkdir()
        owned = [name for name in names if parts.partition_owner(name, 2) == rank]
        identity = {"source": parts.source_part_identity(source, {inventory[n] for n in owned}),
                    **shared}
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
    # q256 896: the routed E4M3 cells' rung (contract v38).
    plan.write_text(json.dumps({stack: {"grid": "E4M3", "q256": 896} for stack in stacks}))
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


def _count_hashes(monkeypatch, source):
    """Names of the ``source`` files ``serving_parts`` digests, once per digest.

    Part outputs carry the source shard filenames, so only files whose parent
    is the source directory count.
    """
    hashed = []
    original = parts.sha256_file

    def counting(path):
        if Path(path).resolve().parent == Path(source).resolve():
            hashed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(parts, "sha256_file", counting)
    return hashed


def test_a_partition_part_stamps_only_the_shards_it_reads(tmp_path, monkeypatch):
    """tessera#495: a part hashes its own input, never the whole checkpoint."""
    import importlib.util
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    script = Path(__file__).resolve().parents[1] / "experiments/export_tessera_serving.py"
    spec = importlib.util.spec_from_file_location("serving_export_stamp_test", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    source = tmp_path / "source"
    source.mkdir()
    layout = {SHARD_A: {"model.layers.0.mlp.down_proj.weight": torch.ones(32, 16),
                        "lm_head.weight": torch.ones(32, 16)},
              SHARD_B: {"model.layers.1.mlp.down_proj.weight": torch.ones(32, 16)}}
    for shard, tensors in layout.items():
        safetensors.save_file(tensors, str(source / shard))
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for shard, tensors in layout.items() for name in tensors}}))
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    paths = []
    for rank, reads in ((0, SHARD_A), (1, SHARD_B)):
        hashed = _count_hashes(monkeypatch, source)
        out = tmp_path / f"export{rank}"
        monkeypatch.setattr("sys.argv", ["export", str(source), str(out), "--grid", "E4M3",
            "--q256", "1024", "--layers", "0", "--device", "cpu", "--partition", f"{rank}/2",
            "--partition-runtime-image", "test/image@sha256:" + "b" * 64])
        exporter.main()
        stamp = json.loads((out / "tessera_serving_manifest.json").read_text())[
            "export_partition"]["identity"]["source"]
        assert stamp["schema"] == parts.SOURCE_PART_SCHEMA
        assert set(stamp["files"]) == {reads}
        assert stamp["tensors"] == parts.source_inventory(source)
        other = SHARD_B if reads == SHARD_A else SHARD_A
        assert reads in hashed and other not in hashed, hashed
        monkeypatch.undo()
        paths.append(out)
    hashed = _count_hashes(monkeypatch, source)
    manifest = parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert sorted(n for n in hashed if n in layout) == [SHARD_A, SHARD_B], hashed
    assert set(manifest["export_identity"]["source"]["files"]) == {SHARD_A, SHARD_B}


def test_the_merge_accepts_parts_with_one_pass_over_the_source(tmp_path, monkeypatch):
    source, paths = _fixture(tmp_path, two_shards=True)
    stamps = [json.loads((p / "tessera_serving_manifest.json").read_text())[
        "export_partition"]["identity"]["source"] for p in paths]
    assert [set(s["files"]) for s in stamps] == [{SHARD_A}, {SHARD_B}]
    hashed = _count_hashes(monkeypatch, source)
    manifest = parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert sorted(n for n in hashed if n.startswith("model-")) == [SHARD_A, SHARD_B], hashed
    assert manifest["export_identity"]["source"] == parts.source_part_identity(source)


def test_a_part_whose_shard_differs_from_the_source_refuses(tmp_path):
    source, paths = _fixture(tmp_path, two_shards=True)
    shard = source / SHARD_B
    raw = shard.read_bytes()
    shard.write_bytes(raw[:-2] + b"\1\1")  # same header and size, other bytes
    with pytest.raises(ValueError, match=f"partition 1: source identity changed.*{SHARD_B}"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert not (tmp_path / "merged").exists()


def test_a_part_naming_a_shard_the_source_lacks_refuses(tmp_path):
    source, paths = _fixture(tmp_path, two_shards=True)
    _change(paths[1], lambda m: m["export_partition"]["identity"]["source"]["files"].update(
        {"model-00003-of-00002.safetensors": "c" * 64}))
    with pytest.raises(ValueError, match="partition 1: .*model-00003-of-00002.safetensors, "
                                         "which the source does not hold"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("field", ["tensors", "config_sha256", "auxiliary_sha256"])
def test_a_part_whose_source_fields_differ_refuses(tmp_path, field):
    source, paths = _fixture(tmp_path, two_shards=True)
    changed = {"tensors": {"model.layers.1.norm.weight": SHARD_B},
               "config_sha256": "d" * 64,
               "auxiliary_sha256": {"config.json": "d" * 64}}[field]
    _change(paths[1], lambda m: m["export_partition"]["identity"]["source"].update({field: changed}))
    with pytest.raises(ValueError, match="partition 1: source identity changed since partition export"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert not (tmp_path / "merged").exists()


def test_a_part_stamping_other_shards_than_it_read_refuses(tmp_path):
    source, paths = _fixture(tmp_path, two_shards=True)
    digest = parts.sha256_file(source / SHARD_A)
    _change(paths[1], lambda m: m["export_partition"]["identity"]["source"]["files"].update(
        {SHARD_A: digest}))
    with pytest.raises(ValueError, match="partition 1: source stamp coverage"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_a_part_with_a_whole_source_identity_from_before_495_refuses_by_name(tmp_path):
    source, paths = _fixture(tmp_path, two_shards=True)
    legacy = parts.source_identity(source)
    for path in paths:
        _change(path, lambda m: m["export_partition"]["identity"].update({"source": legacy}))
    with pytest.raises(ValueError, match="partition 0: source identity is not a tessera.source-part.v1"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def _moe_plan_parts(tmp_path, encoded=None, count=2, input_scales=False):
    """Two source stacks; an omitted encode is internally consistent BF16.

    ``input_scales`` writes the NVFP4 routed shape instead of the FP8 one:
    each role declares an ``input_global_scale`` and the part writes one
    beside each wire, which is what ``nvfp4_moe_route`` reads.
    """
    source = tmp_path / "source"
    source.mkdir()
    encoded = range(count) if encoded is None else encoded
    stacks = [f"model.layers.{i}.feed_forward.experts" for i in range(count)]
    source_names = [f"{stack}.0.{shard}.weight" for stack in stacks for shard in ("w1", "w3", "w2")]
    _tensor_file(source / "model.safetensors", source_names)
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    grid, q256 = ("E2M1x2", 896) if input_scales else ("E4M3", 1024)
    family = "TESSERA_NVFP4" if input_scales else "TESSERA_FP8"
    # The body/plane pair each route decodes (``scheme.ROUTES``): the NVFP4
    # route reads a TCQ body over the LUT plane its group-16 ue4m3 block
    # scales come from, the FP8 route a WINDOW body over the CHANNEL plane.
    body, plane = ("TCQ", "LUT") if input_scales else ("WINDOW", "CHANNEL")
    plan = {stack: {"grid": grid, "q256": q256} for stack in stacks}
    identity = {"source": parts.source_part_identity(source), "code_sha256": "a" * 64,
                "runtime_image": "test/image@sha256:" + "b" * 64, "options": {"plan": plan}}
    paths = []
    for rank, stack in enumerate(stacks):
        path = tmp_path / f"part{rank}"
        path.mkdir()
        owned = [n for n in source_names if n.startswith(stack + ".")]
        modules, groups, ignore = {}, {}, []
        if rank in encoded:
            names = [n.removesuffix(".weight") + ".wire" for n in owned]
            if input_scales:
                names += [n.removesuffix(".weight") + ".input_global_scale" for n in owned]
            roles = [{"tensor": tensor, "source_tensor": tensor, "expert": 0,
                      "role": role, "group": "w2" if role == "down_proj" else "w13",
                      "grid": grid, "q256": q256, "rows": 32, "cols": 32,
                      **({"input_global_scale": 2.5} if input_scales else {})}
                     for tensor, role in zip(owned, ("gate_proj", "up_proj", "down_proj"))]
            modules[stack] = {"structure": "routed_moe", "family": family,
                "grid": grid, "q256": q256, "experts": 1, "roles": roles,
                "wire_bytes": 6, "container_bytes": 6, "resident_bytes_resident_mode": 3072}
            groups[f"stack{rank}"] = {"targets": [stack], "format": "TESSERA", "scheme": {
                "structure": "routed_moe", "family": family, "grid": grid,
                "body": body, "plane": plane, "experts": 1, "groups": {
                    "w13": {"q256": q256, "rows": 64, "columns": 32, "wire_stride": 2,
                            "roles": [["gate_proj", 32], ["up_proj", 32]]},
                    "w2": {"q256": q256, "rows": 32, "columns": 32, "wire_stride": 2,
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


def test_routed_nvfp4_input_scales_are_expected_outputs(tmp_path):
    """A routed NVFP4 part writes one A-side scale per role, and merges.

    ``nvfp4_moe_route`` reads ``experts.{e}.{proj}.input_global_scale`` beside
    each wire -- the quantity the dense route reads as
    ``trellis_input_global_scale`` -- and the exporter writes it.  The merge
    once expected only the wires, so every routed NVFP4 part failed its
    written-coverage check on tensors the loader requires (tessera#526, found
    by the GLM-5.3 A4 merge: 864 scales per MoE layer, 43 layers).
    """
    source, paths, plan = _moe_plan_parts(tmp_path, input_scales=True)
    manifest = parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert set(manifest["modules"]) == set(plan)
    index = json.loads((tmp_path / "merged" / "model.safetensors.index.json").read_text())
    assert sum(name.endswith(".input_global_scale") for name in index["weight_map"]) == 6


def test_routed_nvfp4_part_that_drops_one_input_scale_refuses(tmp_path):
    """The expectation bites: a declared scale that was never written refuses."""
    source, paths, _plan = _moe_plan_parts(tmp_path, input_scales=True)
    index_path = paths[0] / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    del weight_map[sorted(n for n in weight_map if n.endswith(".input_global_scale"))[0]]
    index_path.write_text(json.dumps({"weight_map": weight_map}))
    _tensor_file(paths[0] / "model.safetensors", sorted(weight_map))
    _change(paths[0], lambda m: m["export_partition"]["output_sha256"].update(
        {"model.safetensors": parts.sha256_file(paths[0] / "model.safetensors")}))
    with pytest.raises(ValueError, match="partition 0: written tensor coverage"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert not (tmp_path / "merged").exists()


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


def test_merged_manifest_is_compact_json(tmp_path):
    """The merged manifest ships in the artifact; whitespace is paid in bytes."""
    source, paths = _fixture(tmp_path)
    out = tmp_path / "merged"
    parts.merge_serving_parts(paths, out, source)
    text = (out / "tessera_serving_manifest.json").read_text()
    assert "\n" not in text
    assert text == json.dumps(json.loads(text), separators=(",", ":"))
    assert text == parts.manifest_json(json.loads(text))
