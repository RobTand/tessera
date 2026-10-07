"""The portable census preflight reports real config shapes, not module offers."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import tessera_construction_census as census

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/results/glm53_u1_stub_d_config.json"


@pytest.fixture
def config_dir(tmp_path):
    value = json.loads(CONFIG.read_text())
    (tmp_path / "config.json").write_text(json.dumps(value))
    return tmp_path


def test_preflight_keeps_global_and_local_kda_roles(config_dir):
    result = census.preflight(str(config_dir), "declared-image")
    assert result["construction_performed"] is False
    assert result["construction_performed"] is False
    assert result["model_construction_performed"] is False
    assert result["cpu_linear_construction_performed"] is False
    assert result["cuda_initialized"] is False
    assert "linears" not in result
    one, two = result["KDA_geometry"]
    assert one["columns"] == two["columns"] == 4096
    assert one["global_rows"] == two["global_rows"] == 24896
    assert one["local_rows"] == 24896
    assert two["local_rows"] == 12576
    assert two["local_output_sizes"] == [4096, 4096, 4096, 32, 128, 128]
    assert two["replicated_shard_ids"] == [4, 5]
    assert result["text_dimensions"]["index_n_heads"] == 32
    assert result["config"]["bytes"] == (config_dir / "config.json").stat().st_size


@pytest.mark.parametrize("defect", ["layer_count", "hidden_size", "head_count", "head_dim"])
def test_preflight_refuses_shapes_that_a_device_load_cannot_use(config_dir, defect):
    path = config_dir / "config.json"
    value = json.loads(path.read_text())
    text = value["text_config"]
    if defect == "layer_count":
        text["num_hidden_layers"] += 1
    elif defect == "hidden_size":
        text["hidden_size"] = 0
    elif defect == "head_count":
        text["linear_attn_config"]["num_heads"] = 63
    else:
        text["linear_attn_config"]["head_dim"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="layer count|hidden size|head count|head dimension"):
        census.preflight(str(config_dir), "declared-image")
