"""A merge hashes every part's output shards on a bounded pool, with the serial refusals (tessera#499)."""
import json
import struct
import threading

import pytest

from tessera import serving_parts as parts

NAMES = [f"model.layers.{n}.norm.weight" for n in range(4)] + ["lm_head.weight"]


def _tensor_file(path, names):
    header = {name: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0" * len(names))


def _fixture(tmp_path):
    """Two parts, each writing one output shard per owned tensor."""
    source = tmp_path / "source"
    source.mkdir()
    layout = {f"model-0000{i + 1}-of-00002.safetensors": held
              for i, held in enumerate((NAMES[:3], NAMES[3:]))}
    for shard, held in layout.items():
        _tensor_file(source / shard, held)
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for shard, held in layout.items() for name in held}}))
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    inventory = parts.source_inventory(source)
    shared = {"code_sha256": "a" * 64, "runtime_image": "test/image@sha256:" + "b" * 64,
              "options": {"plan": {}}}
    paths = []
    for rank in range(2):
        path = tmp_path / f"part{rank}"
        path.mkdir()
        owned = [name for name in NAMES if parts.partition_owner(name, 2) == rank]
        files = {name: f"out-{i}.safetensors" for i, name in enumerate(owned)}
        for name, filename in files.items():
            _tensor_file(path / filename, [name])
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": files}))
        config = {"architectures": ["Example"], "quantization_config": {
            "quant_method": "tessera", "format": "mixed-precision", "config_groups": {},
            "ignore": [name.removesuffix(".weight") for name in owned]}}
        (path / "tessera_part_config.json").write_text(json.dumps(config))
        identity = {"source": parts.source_part_identity(source, {inventory[n] for n in owned}),
                    **shared}
        manifest = {"source": str(source), "git": "abc", "modules": {},
                    "totals": {"passthrough_bytes": 2 * len(owned)},
                    "routed_moe": {"quantized_stacks": [], "modules": [],
                                   "packed_source_tensors": 0, "unpacked_source_tensors": 0,
                                   "quantized_source_tensors": 0, "quantized_logical_units": 0},
                    "export_partition": {"schema": parts.SCHEMA, "index": rank, "count": 2,
                        "identity": identity, "source_tensors": owned,
                        "output_sha256": {f: parts.sha256_file(path / f) for f in files.values()}}}
        (path / "tessera_serving_manifest.json").write_text(json.dumps(manifest))
        paths.append(path)
    return source, paths


def _record_threads(monkeypatch):
    seen = []
    real = parts.sha256_file

    def recorded(path):
        seen.append((threading.current_thread().name, path.name))
        return real(path)

    monkeypatch.setattr(parts, "sha256_file", recorded)
    return seen


def _merged(out):
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())
            if p.name != "tessera_serving_manifest.json"}


def test_parallel_merge_writes_the_serial_merge(tmp_path, monkeypatch):
    source, paths = _fixture(tmp_path)
    monkeypatch.setattr(parts, "_affinity_cpus", lambda: 1)
    parts.merge_serving_parts(paths, tmp_path / "serial", source)
    monkeypatch.setattr(parts, "_affinity_cpus", lambda: 4)
    seen = _record_threads(monkeypatch)
    parts.merge_serving_parts(paths, tmp_path / "parallel", source)
    assert _merged(tmp_path / "parallel") == _merged(tmp_path / "serial")
    output_hashes = sorted(name for thread, name in seen if thread.startswith("output-sha256"))
    assert output_hashes == sorted(["out-0.safetensors", "out-1.safetensors", "out-2.safetensors",
                                    "out-0.safetensors", "out-1.safetensors"])


@pytest.mark.parametrize("workers", [1, 4])
def test_mismatching_output_shard_is_refused_by_name(tmp_path, monkeypatch, workers):
    source, paths = _fixture(tmp_path)
    monkeypatch.setattr(parts, "_affinity_cpus", lambda: workers)
    seen = _record_threads(monkeypatch)
    # Part 1 holds out-0 and out-1; corrupt both, the later one first.
    for filename in ("out-1.safetensors", "out-0.safetensors"):
        with (paths[1] / filename).open("ab") as handle:
            handle.write(b"changed")
    out = tmp_path / "merged"
    with pytest.raises(ValueError) as refused:
        parts.merge_serving_parts(paths, out, source)
    # The first mismatching shard in sorted order, in the first failing part.
    assert str(refused.value) == "partition 1: output sha256 mismatch: out-0.safetensors"
    assert not out.exists()
    if workers > 1:
        assert any(thread.startswith("output-sha256") for thread, _ in seen)


def test_earlier_part_check_refuses_before_a_later_part_mismatch(tmp_path, monkeypatch):
    source, paths = _fixture(tmp_path)
    monkeypatch.setattr(parts, "_affinity_cpus", lambda: 4)
    with (paths[1] / "out-0.safetensors").open("ab") as handle:
        handle.write(b"changed")
    manifest_path = paths[0] / "tessera_serving_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["export_partition"]["output_sha256"]["out-2.safetensors"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError) as refused:
        parts.merge_serving_parts(paths, tmp_path / "merged", source)
    assert str(refused.value) == "partition 0: output sha256 mismatch: out-2.safetensors"


def test_a_later_unreadable_shard_does_not_preempt_an_earlier_refusal(tmp_path, monkeypatch):
    source, paths = _fixture(tmp_path)
    monkeypatch.setattr(parts, "_affinity_cpus", lambda: 4)
    (paths[1] / "out-1.safetensors").unlink()
    manifest_path = paths[0] / "tessera_serving_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["export_partition"]["output_sha256"]["out-9.safetensors"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    # Serially, part 0's coverage check refuses before part 1's shard is opened.
    with pytest.raises(ValueError, match="^partition 0: output sha256 coverage disagrees with index$"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source)


def test_missing_output_shard_raises_the_serial_exception(tmp_path, monkeypatch):
    source, paths = _fixture(tmp_path)
    (paths[1] / "out-1.safetensors").unlink()
    raised = {}
    for workers in (1, 4):
        monkeypatch.setattr(parts, "_affinity_cpus", lambda: workers)
        with pytest.raises(OSError) as refused:
            parts.merge_serving_parts(paths, tmp_path / f"merged-{workers}", source)
        raised[workers] = (type(refused.value), refused.value.filename)
    assert raised[4] == raised[1] == (FileNotFoundError, str(paths[1] / "out-1.safetensors"))
