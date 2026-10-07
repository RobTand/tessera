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


@pytest.fixture
def action():
    path = Path(__file__).resolve().parents[1] / "experiments/t8r_speed/t16_decode_once_action.py"
    spec = importlib.util.spec_from_file_location("t16_action", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flag", ["--cpu-preflight", "--collect-only"])
def test_test_mode_collects_on_cpu(action, tmp_path, flag):
    out, args, cpu, tests = action.parse_action_args(
        [str(tmp_path), "--tests", flag, "--", "-n", "1",
         "--durations=10", "tests/test_bf16_prefill.py"])
    assert out == tmp_path.resolve()
    assert args == ["-n", "1", "--durations=10", "tests/test_bf16_prefill.py"]
    assert cpu and tests


def test_test_mode_requires_an_explicit_selection(action, tmp_path):
    with pytest.raises(ValueError, match="test file"):
        action.parse_action_args([str(tmp_path), "--tests", "--", "-q"])
    with pytest.raises(ValueError, match="separator"):
        action.parse_action_args([str(tmp_path), "--tests", "tests/test_bf16_prefill.py"])


def test_cpu_test_preflight_reads_real_inputs(action, tmp_path):
    test = tmp_path / "test_small.py"
    test.write_text("def test_small():\n    assert 1 + 1 == 2\n")
    runner = tmp_path / "runner"
    for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "xdist", "execnet"):
        package = runner / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n")
    (runner / "py.py").write_text("VALUE = 2\n")
    evidence = action.test_preflight([str(test), "-n", "1"], str(runner))
    assert evidence["test_reads"][0]["bytes_read"] == test.stat().st_size
    assert len(evidence["runner_reads"]) == 8
    assert all(item["bytes_read"] > 0 for item in evidence["runner_reads"])
    (runner / "py.py").unlink()
    with pytest.raises(FileNotFoundError):
        action.test_preflight([str(test)], str(runner))
