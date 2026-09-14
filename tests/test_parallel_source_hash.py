"""A part's source shards hash in parallel with the serial pass's result (tessera#499)."""
import hashlib
import json
import os
import struct
import threading

import pytest

from tessera import encoder_identity as ei
from tessera import serving_parts as parts

SHARDS = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]


def _source(tmp_path):
    """Four shards of distinct bytes, so a digest in the wrong slot cannot pass."""
    source = tmp_path / "source"
    source.mkdir()
    weight_map = {}
    for i, shard in enumerate(SHARDS):
        name = f"model.layers.{i}.norm.weight"
        header = json.dumps({name: {"dtype": "BF16", "shape": [i + 1],
                                    "data_offsets": [0, 2 * (i + 1)]}}).encode()
        payload = bytes([i + 1]) * (2 * (i + 1))
        (source / shard).write_bytes(struct.pack("<Q", len(header)) + header + payload)
        weight_map[name] = shard
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (source / "config.json").write_text(json.dumps({"architectures": ["Example"]}))
    return source


def _expected_files(source, chosen):
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in chosen}


@pytest.mark.parametrize("shards", [None, [SHARDS[3], SHARDS[0], SHARDS[2]]])
def test_parallel_part_identity_is_the_serial_document(tmp_path, shards):
    source = _source(tmp_path)
    serial = parts.source_part_identity(source, shards, workers=1)
    chosen = sorted(shards) if shards is not None else SHARDS
    # Independent reference: each name's own bytes, in sorted-name order.
    assert serial["files"] == _expected_files(source, chosen)
    for workers in (None, 2, 4, 16):
        parallel = parts.source_part_identity(source, shards, workers=workers)
        assert list(parallel["files"]) == chosen
        assert parallel["files"] == _expected_files(source, chosen)
        # The manifest is json.dumps'd: key order is part of the bytes.
        assert json.dumps(parallel) == json.dumps(serial)


def test_sha256_files_returns_each_digest_in_its_path_slot(tmp_path):
    source = _source(tmp_path)
    paths = [source / name for name in SHARDS]
    expected = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert parts.sha256_files(paths, workers=4) == expected
    assert parts.sha256_files(list(reversed(paths)), workers=4) == list(reversed(expected))
    assert parts.sha256_files([], workers=4) == []


def test_default_workers_hash_concurrently_up_to_the_affinity(tmp_path, monkeypatch):
    source = _source(tmp_path)
    paths = [source / name for name in SHARDS]
    monkeypatch.setattr(parts.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    barrier = threading.Barrier(len(paths), timeout=30)
    real = parts.sha256_file

    def gated(path):
        barrier.wait()  # breaks, and raises, unless all four run at once
        return real(path)

    monkeypatch.setattr(parts, "sha256_file", gated)
    assert parts.sha256_files(paths) == [real(p) for p in paths]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root reads a mode-000 file")
@pytest.mark.parametrize("workers", [1, 4])
def test_unreadable_shard_refuses_with_the_serial_exception(tmp_path, monkeypatch, workers):
    source = _source(tmp_path)
    inventory = parts.source_inventory(source)
    # The inventory reads shard headers first; hold it so the refusal is the hash's.
    monkeypatch.setattr(parts, "source_inventory", lambda _source: inventory)
    for name in (SHARDS[3], SHARDS[1]):
        (source / name).chmod(0)
    try:
        with pytest.raises(PermissionError) as refused:
            parts.source_part_identity(source, workers=workers)
    finally:
        for name in SHARDS:
            (source / name).chmod(0o644)
    # The first unreadable shard in sorted order, as the serial pass raised it.
    assert refused.value.filename == str(source / SHARDS[1])


def test_encoder_fixture_id_built_on_a_helper_thread_is_the_same_id(monkeypatch):
    first = ei.encoder_fixture_id()
    monkeypatch.setattr(ei, "_MEMO", [])
    seen = []
    helper = threading.Thread(target=lambda: seen.append(ei.encoder_fixture_id()))
    helper.start()
    helper.join()
    assert seen == [first]
    # The caller's thread now takes the helper's memo.
    assert ei.encoder_fixture_id() is seen[0]
