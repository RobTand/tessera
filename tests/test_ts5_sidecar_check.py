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
