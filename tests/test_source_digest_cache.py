"""Stat-bound source shard digests: reuse, refusal of changed shards, merge skip (tessera#499)."""
import json
import os
import struct
import time

import pytest

from tessera import serving_parts as parts
from tessera.source_digest_cache import SourceDigestCache

NAMES = [f"model.layers.{n}.norm.weight" for n in range(4)] + ["lm_head.weight"]
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def _tensor_file(path, names, fill=b"\0\0"):
    header = {name: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + fill * len(names))


def _cache(tmp_path, **kwargs):
    directory = tmp_path / "digests"
    directory.mkdir(mode=0o755, exist_ok=True)
    return SourceDigestCache(directory, quiescent_seconds=kwargs.pop("quiescent_seconds", 0), **kwargs)


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for shard, held in zip(SHARDS, (NAMES[:3], NAMES[3:])):
        _tensor_file(source / shard, held)
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard for shard, held in zip(SHARDS, (NAMES[:3], NAMES[3:]))
                       for name in held}}))
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    return source


def _parts(tmp_path, source, cache):
    """Two serving parts whose source stamps went through ``cache``."""
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
        (path / "tessera_part_config.json").write_text(json.dumps({
            "architectures": ["Example"], "quantization_config": {
                "quant_method": "tessera", "format": "mixed-precision", "config_groups": {},
                "ignore": [name.removesuffix(".weight") for name in owned]}}))
        identity = {"source": parts.source_part_identity(
            source, {inventory[n] for n in owned}, digest_cache=cache), **shared}
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
    return paths


def _count_hashes(monkeypatch):
    seen = []
    real = parts.sha256_file

    def counted(path):
        seen.append(path.name)
        return real(path)

    monkeypatch.setattr(parts, "sha256_file", counted)
    return seen


def _rewrite_same_size(path):
    """Same size, different bytes, mtime and atime restored: only ctime can tell."""
    before = os.stat(path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    time.sleep(0.02)
    with path.open("r+b") as handle:
        handle.write(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_ino, after.st_mtime_ns) == (before.st_size, before.st_ino,
                                                                 before.st_mtime_ns)
    assert after.st_ctime_ns != before.st_ctime_ns


def test_warm_cache_serves_the_fresh_document_without_reading_shards(tmp_path, monkeypatch):
    source = _source(tmp_path)
    fresh = parts.source_part_identity(source)
    cache = _cache(tmp_path)
    assert parts.source_part_identity(source, digest_cache=cache) == fresh
    assert cache.receipt()["mode"] == "hashed"
    assert all(row["recorded"] for row in cache.receipt()["shards"])
    seen = _count_hashes(monkeypatch)
    warm = _cache(tmp_path)
    document = parts.source_part_identity(source, digest_cache=warm)
    assert json.dumps(document) == json.dumps(fresh)
    assert not set(seen) & set(SHARDS)  # config and auxiliary files are still hashed
    assert "config.json" in seen
    receipt = warm.receipt()
    assert (receipt["mode"], receipt["cached_shards"], receipt["hashed_shards"]) == ("stat-bound", 2, 0)


@pytest.mark.parametrize("workers", [1, 4])
def test_same_size_rewrite_is_rehashed_and_refused_by_the_merge(tmp_path, monkeypatch, workers):
    source = _source(tmp_path)
    paths = _parts(tmp_path, source, _cache(tmp_path))
    parts.source_part_identity(source, digest_cache=_cache(tmp_path))  # every shard recorded
    _rewrite_same_size(source / SHARDS[1])
    monkeypatch.setattr(parts, "_affinity_cpus", lambda: workers)
    seen = _count_hashes(monkeypatch)
    cache = _cache(tmp_path)
    # Both parts stamped the rewritten shard; the first part in rank order refuses.
    with pytest.raises(ValueError, match=f"^partition 0: .*{SHARDS[1]}"):
        parts.merge_serving_parts(paths, tmp_path / "merged", source, source_digest_cache=cache)
    assert seen.count(SHARDS[1]) == 1 and SHARDS[0] not in seen
    assert {row["shard"]: row["how"] for row in cache.receipt()["shards"]} == {
        SHARDS[0]: "cached", SHARDS[1]: "hashed"}


def test_warm_merge_skips_the_source_pass_and_still_hashes_outputs(tmp_path, monkeypatch):
    source = _source(tmp_path)
    paths = _parts(tmp_path, source, _cache(tmp_path))
    cold = parts.merge_serving_parts(paths, tmp_path / "cold", source)
    seen = _count_hashes(monkeypatch)
    cache = _cache(tmp_path)
    warm = parts.merge_serving_parts(paths, tmp_path / "warm", source, source_digest_cache=cache)
    assert not set(seen) & set(SHARDS)
    assert sorted(n for n in seen if n.startswith("out-")) == sorted(
        ["out-0.safetensors", "out-1.safetensors", "out-2.safetensors",
         "out-0.safetensors", "out-1.safetensors"])
    assert warm["export_identity"] == cold["export_identity"]
    assert "source_proof" not in cold
    assert (warm["source_proof"]["mode"], warm["source_proof"]["cached_shards"]) == ("stat-bound", 2)


def test_a_shard_changing_while_hashed_is_refused(tmp_path, monkeypatch):
    source = _source(tmp_path)
    real = parts.sha256_file

    def mutating(path):
        digest = real(path)
        if path.name == SHARDS[0]:
            _rewrite_same_size(path)
        return digest

    monkeypatch.setattr(parts, "sha256_file", mutating)
    cache = _cache(tmp_path)
    with pytest.raises(ValueError, match=f"source shard changed while hashing: .*{SHARDS[0]}"):
        parts.source_part_identity(source, workers=1, digest_cache=cache)
    assert not list(cache.directory.glob("*.json"))


def test_a_recently_changed_shard_is_hashed_but_not_recorded(tmp_path):
    source = _source(tmp_path)
    cache = _cache(tmp_path, quiescent_seconds=300)
    parts.source_part_identity(source, digest_cache=cache)
    assert [row["recorded"] for row in cache.receipt()["shards"]] == [False, False]
    assert not list(cache.directory.glob("*.json"))


def test_a_corrupt_entry_is_refused_by_name(tmp_path):
    source = _source(tmp_path)
    parts.source_part_identity(source, digest_cache=_cache(tmp_path))
    cache = _cache(tmp_path)
    key = cache._key(cache.fingerprint(source / SHARDS[0]))
    cache._entry_path(key).write_text("{not json")
    with pytest.raises(ValueError, match=f"corrupt .*{SHARDS[0]}"):
        parts.source_part_identity(source, digest_cache=cache)


def test_two_reads_disagreeing_about_one_stat_identity_are_refused(tmp_path):
    source = _source(tmp_path)
    cache = _cache(tmp_path)
    parts.source_part_identity(source, digest_cache=cache)
    key = cache._key(cache.fingerprint(source / SHARDS[0]))
    with pytest.raises(ValueError, match="disagree"):
        cache._record(cache._entry_path(key), key, "0" * 64, {})
    assert not [p for p in cache.directory.iterdir() if p.name.startswith(".entry-")]


def test_a_symlinked_source_path_reuses_the_digest(tmp_path, monkeypatch):
    source = _source(tmp_path)
    parts.source_part_identity(source, digest_cache=_cache(tmp_path))
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    seen = _count_hashes(monkeypatch)
    parts.source_part_identity(alias, digest_cache=_cache(tmp_path))
    assert not set(seen) & set(SHARDS)


def test_cache_location_is_preflighted(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="not a directory"):
        SourceDigestCache(tmp_path / "missing")
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    with pytest.raises(ValueError, match="world-writable"):
        SourceDigestCache(open_dir)
    inside = source / "digests"
    inside.mkdir()
    with pytest.raises(ValueError, match="inside the source"):
        SourceDigestCache(inside, source=source)


def test_whole_source_identity_reuses_the_same_stat_bound_cache(tmp_path, monkeypatch):
    from pathlib import Path
    from tessera.serving_parts import source_identity
    source=tmp_path/'whole'; source.mkdir()
    (source/'config.json').write_text('{}')
    # The shard is written by this module's own byte builder, not through
    # torch/safetensors: every other test here is collected by the bytes-only
    # CI job, and a torch import inside a test body is collected there and
    # then fails at run time rather than being ignored.
    _tensor_file(source/'model.safetensors', ['x.weight'])
    cache_dir=tmp_path/'cache';cache_dir.mkdir()
    cache=SourceDigestCache(cache_dir,source=source,quiescent_seconds=0)
    expected=source_identity(source)
    assert source_identity(source,digest_cache=cache)==expected
    assert cache.receipt()['hashed_shards']==1
    import tessera.serving_parts as parts
    original=parts.sha256_file
    def refuse_shard(path):
        if Path(path).suffix=='.safetensors':raise AssertionError('valid source digest was re-read')
        return original(path)
    monkeypatch.setattr(parts,'sha256_file',refuse_shard)
    reused=SourceDigestCache(cache_dir,source=source,quiescent_seconds=0)
    assert source_identity(source,digest_cache=reused)==expected
    assert reused.receipt()['cached_shards']==1
    shard=source/'model.safetensors';raw=shard.read_bytes();shard.write_bytes(raw)
    with pytest.raises(AssertionError,match='re-read'):
        source_identity(source,digest_cache=reused)
