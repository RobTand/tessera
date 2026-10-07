"""Check the measurement inputs and the paired statistic without CUDA."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def harness():
    path = Path(__file__).resolve().parents[1] / "experiments/t8r_speed/bench_t16_decode_once.py"
    spec = importlib.util.spec_from_file_location("t16_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kda_shape_keeps_replicated_rows(harness):
    config = {"text_config": {"hidden_size": 4096,
              "linear_attn_config": {"num_heads": 64, "head_dim": 128}}}
    roles, cols = harness.shape(config, 2)
    assert roles == [("q", 4096), ("k", 4096), ("v", 4096),
                     ("b", 32), ("f_a", 128), ("g_a", 128)]
    assert sum(n for _, n in roles) == 12576 and cols == 4096
    with pytest.raises(ValueError):
        harness.require_shape(config, 1)


def test_paired_statistic_uses_each_order_median(harness):
    passes = {"F": {"samples_ms": [1.0, 2.0, 90.0]},
              "R": {"samples_ms": [3.0, 4.0, 5.0]}}
    got = harness.paired_stats(passes)
    assert got == {"forward_median_ms": 2.0, "reverse_median_ms": 4.0,
                   "mean_ms": 3.0, "spread_fraction": 2.0 / 3.0}
    with pytest.raises(ValueError):
        harness.paired_stats({"F": {"samples_ms": [1.0]}})


def test_fixed_screen_threshold_boundaries(harness):
    assert harness.screen_ratio(1.15) == "GO"
    assert harness.screen_ratio(1.5) == "INCONCLUSIVE"
    assert harness.screen_ratio(1.50001) == "KILL"


def test_reference_reads_history_and_original_column_order(harness):
    # The first 14-bit field has six history bits and eight BODY bits.
    words = torch.zeros(16 * 8 * 2, dtype=torch.int32)
    words[0] = 0x12000000
    words[16 * 8] = 0x34000000
    table = torch.arange(1 << 14, dtype=torch.float32).bfloat16().view(torch.int16)
    wire = {"words": words[None], "table": table[None],
            "init": torch.tensor([[3, 5]], dtype=torch.int32),
            "scale": torch.tensor([[0.25]], dtype=torch.float32),
            "perm": torch.tensor([[1, 0]], dtype=torch.int32),
            "runs": torch.tensor([[8, 0, 2, 0, 0, 2, 0, 256]], dtype=torch.int32),
            "tile_words": 256, "window_bits": 14}
    got = harness.reference_weight(wire, 1, 2, "cpu")
    states = torch.tensor([(3 << 8) | 0x12, (5 << 8) | 0x34])
    packed = (table.view(torch.bfloat16)[states].float() * 0.25).bfloat16()
    assert torch.equal(got, packed.flip(0)[None])


def test_real_cli_rejects_a_changed_population(harness, tmp_path):
    with pytest.raises(ValueError, match="M=16,2048,4096"):
        harness.parse_args(["--out", str(tmp_path), "--ms", "16,2048"])
    with pytest.raises(ValueError, match="protocol"):
        harness.parse_args(["--out", str(tmp_path), "--iters", "1"])
