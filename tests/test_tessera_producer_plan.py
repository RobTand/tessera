"""Producer CLI cache wiring; geometry is a stub, source seals use real toy bytes."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType

import pytest

from tessera import serving_parts
from tessera import source_digest_cache

ROOT = Path(__file__).resolve().parents[1]


def _producer_stubs() -> tuple[ModuleType, ModuleType]:
    """Separate geometry and plan reading from real source-seal/cache behavior."""
    geometry = ModuleType("tessera.export_serving")
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
        "producer_plan_cache_cli", ROOT / "src" / "tessera" / "producer_plan.py")
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


def _producer_child(mode, source, cache, output, report):
    for dependency in _producer_stubs():
        sys.modules[dependency.__name__] = dependency
    from tessera import producer_plan

    source = Path(source)
    output = Path(output)
    timestamps = [timestamp for shard in source.glob("*.safetensors")
                  for timestamp in (shard.stat().st_mtime_ns, shard.stat().st_ctime_ns)]
    source_digest_cache.time.time_ns = lambda: max(timestamps) + int(
        source_digest_cache.DEFAULT_QUIESCENT_SECONDS * 1e9)
    reads = []
    original = serving_parts.sha256_file

    def counted(path):
        reads.append(Path(path).name)
        return original(path)

    serving_parts.sha256_file = counted
    plan = source.parent / "stack-plan.json"
    receipt = None
    if mode == "cli":
        argv = [str(source), "--stack-plan", str(plan), "--out", str(output)]
        if cache:
            argv += ["--source-digest-cache", cache]
        status = producer_plan.main(argv)
        if status != 0:
            raise SystemExit(status)
        receipt = json.loads(output.read_text()).get("source_digest_cache")
    else:
        digest_cache = (source_digest_cache.SourceDigestCache(Path(cache), source=source)
                        if cache else None)
        projected = producer_plan.producer_projection(source, plan, digest_cache=digest_cache)
        output.write_text(json.dumps(projected, indent=2) + "\n")
        if digest_cache is not None:
            receipt = digest_cache.receipt()
    Path(report).write_text(json.dumps({"reads": reads, "receipt": receipt, "pid": os.getpid()}))


def _fresh_producer(mode, source, tmp_path, label, cache=None):
    output = tmp_path / f"{label}.json"
    report = tmp_path / f"{label}-report.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["TMPDIR"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c",
         "import runpy, sys; runpy.run_path(sys.argv[1])['_producer_child'](*sys.argv[2:])",
         str(Path(__file__).resolve()), mode, str(source), str(cache) if cache else "",
         str(output), str(report)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output.read_bytes(), json.loads(report.read_text())


@pytest.mark.parametrize("mode", ["projection", "cli"])
def test_producer_cache_survives_fresh_interpreters(mode, source, tmp_path, cache_directory):
    header = json.dumps({"model.layers.1.norm.weight": {
        "dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    changed_shard = source / "second.safetensors"
    changed_shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    shard_names = {shard.name for shard in source.glob("*.safetensors")}
    expected_bytes, _ = _fresh_producer("projection", source, tmp_path, "uncached")
    cold_bytes, cold = _fresh_producer(mode, source, tmp_path, "cold", cache_directory)
    warm_bytes, warm = _fresh_producer(mode, source, tmp_path, "warm", cache_directory)

    previous_stat = changed_shard.stat()
    changed_shard.write_bytes(changed_shard.read_bytes()[:-2] + b"\0\1")
    os.utime(changed_shard, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1))
    changed_bytes, changed = _fresh_producer(mode, source, tmp_path, "changed", cache_directory)
    changed_expected, _ = _fresh_producer("projection", source, tmp_path, "changed-uncached")
    assert json.loads(changed_expected)["source"] != json.loads(expected_bytes)["source"]

    for actual, report, expected, hashed in (
            (cold_bytes, cold, expected_bytes, shard_names),
            (warm_bytes, warm, expected_bytes, set()),
            (changed_bytes, changed, changed_expected, {changed_shard.name})):
        receipt = report["receipt"]
        if mode == "cli":
            expected_projection = json.loads(expected)
            expected_projection["source_digest_cache"] = receipt
            expected = (json.dumps(expected_projection, indent=2) + "\n").encode()
        assert actual == expected
        reads = report["reads"]
        assert {name for name in reads if name in shard_names} == hashed
        assert all(reads.count(name) == 1 for name in hashed)
        assert {"config.json", "tokenizer_config.json"} <= set(reads)
        assert receipt["schema"] == source_digest_cache.RECEIPT_SCHEMA
        assert receipt["cache"] == str(cache_directory)
        assert receipt["hashed_shards"] == len(hashed)
        assert receipt["cached_shards"] == len(shard_names - hashed)
        assert receipt["mode"] == ("hashed" if hashed == shard_names else "stat-bound")
        rows = {row["shard"]: row for row in receipt["shards"]}
        assert set(rows) == shard_names
        assert len(receipt["shards"]) == len(shard_names)
        for name, row in rows.items():
            assert row["how"] == ("hashed" if name in hashed else "cached")
            if name in hashed:
                assert row["recorded"] is True
            else:
                assert row["writer"]["pid"] == cold["pid"]
                assert row["writer"]["quiescent_seconds"] == (
                    source_digest_cache.DEFAULT_QUIESCENT_SECONDS)


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


def test_public_projection_cache_hit_is_byte_identical_to_uncached_projection(
        producer, source, tmp_path, cache_directory, quiescent_clock, hash_reads):
    plan = tmp_path / "stack-plan.json"
    uncached = producer.producer_projection(source, plan)
    expected_bytes = (json.dumps(uncached, indent=2) + "\n").encode()
    hash_reads.clear()
    cold_cache = source_digest_cache.SourceDigestCache(cache_directory, source=source)
    cold = producer.producer_projection(source, plan, digest_cache=cold_cache)
    assert (json.dumps(cold, indent=2) + "\n").encode() == expected_bytes
    assert hash_reads.count("model.safetensors") == 1
    assert cold_cache.receipt()["hashed_shards"] == 1
    hash_reads.clear()
    # A new instance checks on-disk reuse within this interpreter.
    warm_cache = source_digest_cache.SourceDigestCache(cache_directory, source=source)
    warm = producer.producer_projection(source, plan, digest_cache=warm_cache)
    assert (json.dumps(warm, indent=2) + "\n").encode() == expected_bytes
    assert "model.safetensors" not in hash_reads
    assert {"config.json", "tokenizer_config.json"} <= set(hash_reads)
    assert warm_cache.receipt()["cached_shards"] == 1
    assert warm_cache.receipt()["hashed_shards"] == 0


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


