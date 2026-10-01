"""Producer CLI cache wiring; geometry is a stub, source seals use real toy bytes."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import ModuleType

import pytest

from tessera import serving_parts
from tessera import source_digest_cache

ROOT = Path(__file__).resolve().parents[1]


def _producer_stubs() -> tuple[ModuleType, ModuleType]:
    """Separate geometry and plan reading from real source-seal/cache behavior."""
    geometry = ModuleType("export_tessera_serving")
    geometry.__dict__.update(
        quantizable=lambda src: ([], {}, {}, {}),
        project_expert_plan=lambda *args: {"projection": "control"},
    )
    manifest = ModuleType("tessera.cached_unit")
    manifest.__dict__["read_manifest"] = lambda path: {}
    return geometry, manifest


@pytest.fixture
def producer(monkeypatch):
    # Both mocked dependencies must be isolated before any CLI import executes.
    for dependency in _producer_stubs():
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    spec = importlib.util.spec_from_file_location(
        "producer_plan_cache_cli", ROOT / "experiments" / "tessera_producer_plan.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    header = json.dumps({"model.layers.0.norm.weight": {
        "dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    (root / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    (root / "tokenizer_config.json").write_text("{}")
    return root


@pytest.fixture
def cache_directory(tmp_path):
    directory = tmp_path / "digests"
    directory.mkdir(mode=0o700)
    return directory


@pytest.fixture
def quiescent_clock(source, monkeypatch):
    """Advance the fixture clock, not the library's production quiescence rule."""
    def now():
        st = (source / "model.safetensors").stat()
        return max(st.st_mtime_ns, st.st_ctime_ns) + int(
            source_digest_cache.DEFAULT_QUIESCENT_SECONDS * 1e9)
    monkeypatch.setattr(source_digest_cache.time, "time_ns", now)


@pytest.fixture
def hash_reads(monkeypatch):
    reads = []
    original = serving_parts.sha256_file

    def counted(path):
        reads.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(serving_parts, "sha256_file", counted)
    return reads


def _call(producer, source, tmp_path, cache=None):
    output = tmp_path / "projection.json"
    argv = [str(source), "--stack-plan", str(tmp_path / "stack-plan.json"), "--out", str(output)]
    if cache is not None:
        argv += ["--source-digest-cache", str(cache)]
    assert producer.main(argv) == 0
    return json.loads(output.read_text())


def test_cli_without_cache_keeps_legacy_output_bytes(producer, source, tmp_path):
    expected = {"projection": "control", "source": serving_parts.source_identity(source)}
    _call(producer, source, tmp_path)
    assert (tmp_path / "projection.json").read_bytes() == (json.dumps(expected, indent=2) + "\n").encode()


def test_cli_cache_reuses_seal_and_keeps_config_and_auxiliary_reads(
        producer, source, tmp_path, cache_directory, quiescent_clock, hash_reads):
    first = _call(producer, source, tmp_path, cache_directory)
    assert hash_reads.count("model.safetensors") == 1
    assert first["source_digest_cache"]["hashed_shards"] == 1
    assert first["source_digest_cache"]["shards"][0]["recorded"] is True
    hash_reads.clear()
    second = _call(producer, source, tmp_path, cache_directory)
    assert second["source"] == first["source"]
    assert "model.safetensors" not in hash_reads
    assert {"config.json", "tokenizer_config.json"} <= set(hash_reads)
    receipt = second["source_digest_cache"]
    assert receipt["schema"] == source_digest_cache.RECEIPT_SCHEMA
    assert receipt["mode"] == "stat-bound"
    assert receipt["cached_shards"] == 1 and receipt["hashed_shards"] == 0
    assert receipt["shards"][0]["writer"]["quiescent_seconds"] == source_digest_cache.DEFAULT_QUIESCENT_SECONDS


def test_cli_cache_rehashes_same_size_changed_shard(
        producer, source, tmp_path, cache_directory, quiescent_clock, hash_reads):
    first = _call(producer, source, tmp_path, cache_directory)
    shard = source / "model.safetensors"
    st = shard.stat()
    shard.write_bytes(shard.read_bytes()[:-2] + b"\0\1")
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    hash_reads.clear()
    changed = _call(producer, source, tmp_path, cache_directory)
    assert changed["source"]["files"] != first["source"]["files"]
    assert hash_reads.count(shard.name) == 1
    assert changed["source_digest_cache"]["hashed_shards"] == 1


def test_cli_cache_refuses_corrupt_entry(
        producer, source, tmp_path, cache_directory, quiescent_clock):
    _call(producer, source, tmp_path, cache_directory)
    entry, = cache_directory.glob("*.json")
    entry.write_text("{}")
    with pytest.raises(ValueError, match="source digest cache entry is corrupt"):
        _call(producer, source, tmp_path, cache_directory)


@pytest.mark.parametrize("kind,message", [
    ("missing", "source digest cache is not a directory"),
    ("world-writable", "source digest cache is world-writable"),
    ("inside-source", "source digest cache lies inside the source it seals"),
])
def test_cli_cache_keeps_directory_refusals(producer, source, tmp_path, kind, message):
    directory = source / "digests" if kind == "inside-source" else tmp_path / "bad-digests"
    if kind != "missing":
        directory.mkdir(mode=0o700)
    if kind == "world-writable":
        directory.chmod(0o777)
    with pytest.raises(ValueError, match=message):
        _call(producer, source, tmp_path, directory)
