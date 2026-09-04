"""Read actual wire names and shard headers before spending a serve."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
STACK = "model.layers.2.feed_forward.experts"


def _checker():
    spec = importlib.util.spec_from_file_location(
        "ts5_sidecar_check", ROOT / "experiments" / "ts5_sidecar_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path, *, shard, aliases, omit=None):
    groups = {
        "w13": {"wire_stride": 8, "roles": [["gate_proj", 32], ["up_proj", 32]]},
        "w2": {"wire_stride": 8, "roles": [["down_proj", 64]]},
    }
    config = {"quantization_config": {"quant_method": "tessera", "ignore": [],
        "config_groups": {"experts": {"targets": [STACK], "scheme": {
            "structure": "routed_moe", "experts": 2, "groups": groups}}}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    header = {}
    offset = 0
    for expert in range(2):
        for role in aliases:
            if (expert, role) == omit:
                continue
            header[f"{STACK}.{expert}.{role}.wire"] = {
                "dtype": "U8", "shape": [8], "data_offsets": [offset, offset + 8]}
            offset += 8
    encoded = json.dumps(header).encode()
    (tmp_path / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


@pytest.mark.parametrize("shard", ["model.safetensors", "model-00001-of-00001.safetensors"])
@pytest.mark.parametrize("aliases", [("w1", "w3", "w2"),
                                    ("gate_proj", "up_proj", "down_proj")])
def test_exact_wires_cover_canonical_and_lfm_source_names(tmp_path, shard, aliases):
    _checkpoint(tmp_path, shard=shard, aliases=aliases)
    assert _checker().main(tmp_path) == 0


def test_maximum_stride_does_not_hide_a_missing_expert_projection(tmp_path):
    _checkpoint(tmp_path, shard="model-00001-of-00001.safetensors",
                aliases=("gate_proj", "up_proj", "down_proj"), omit=(1, "up_proj"))
    assert _checker().main(tmp_path) != 0


@pytest.mark.parametrize("indexed", [False, True])
def test_duplicate_tensor_names_across_shards_are_refused(tmp_path, indexed):
    aliases = ("gate_proj", "up_proj", "down_proj")
    first, second = "model-a.safetensors", "model-b.safetensors"
    _checkpoint(tmp_path, shard=first, aliases=aliases)
    _checkpoint(tmp_path, shard=second, aliases=aliases)
    if indexed:
        # Both shards are indexed, even though the index gives each logical
        # tensor just one owner. Reading those shards must still detect that
        # each contains a second physical copy of the same wire.
        weight_map = {f"{STACK}.{expert}.{role}.wire":
                      (first if expert == 0 else second)
                      for expert in range(2) for role in aliases}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="duplicate tensor name") as failure:
        _checker().main(tmp_path)
    assert f"{STACK}.0.gate_proj.wire" in str(failure.value)
    assert first in str(failure.value) and second in str(failure.value)


def test_embedded_plan_cannot_claim_an_unwritten_stack(tmp_path):
    _checkpoint(tmp_path, shard="model.safetensors", aliases=("w1", "w3", "w2"))
    other = "model.layers.3.feed_forward.experts"
    manifest = {"modules": {STACK: {"structure": "routed_moe"}},
                "routed_moe": {"quantized_stacks": [STACK]},
                "export_identity": {"options": {"plan": {
                    STACK: {"grid": "E4M3", "q256": 1024},
                    other: {"grid": "E4M3", "q256": 1024}}}}}
    (tmp_path / "tessera_serving_manifest.json").write_text(json.dumps(manifest))
    assert _checker().main(tmp_path) != 0


def test_explicit_plan_requires_manifest(tmp_path):
    _checkpoint(tmp_path, shard="model.safetensors", aliases=("w1", "w3", "w2"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({STACK: {"grid": "E4M3", "q256": 1024}}))
    assert _checker().main(tmp_path, plan_json=plan) != 0
