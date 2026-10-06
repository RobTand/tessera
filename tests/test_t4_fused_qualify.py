"""Narrow CPU tests for the finite fused T-4 qualification harness.

No CUDA, no compact packer, no full model. Small synthetic weights only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "t4_code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import t4_fused_qualify as qual


def test_window_recipe_fields_are_fixed():
    kw, source = qual.resolve_window_kwargs(896)
    assert int(kw["q256"]) == 896
    assert int(kw["window_bits"]) == 14
    assert int(kw["span"]) == 1
    assert source in ("served-recipe", "explicit-window-pending-served-recipe")
    assert str(kw["body"]).endswith("WINDOW")
    assert str(kw["scale_plane"]).endswith("LUT")


def test_pure_width_map_covers_all_pairs():
    assert qual.PURE_Q256 == [128, 256, 384, 512, 640, 768, 896, 1024]
    assert qual.Q_TO_RATE[896] == 7


def test_routing_parse_shapes():
    parsed = qual.parse_routing(16, 2, 2, 7)
    assert parsed["tokens"] == 16
    assert parsed["routes"] == 32
    assert parsed["ids_shape"] == [16, 2]


def test_numeric_oracle_helpers_are_exact():
    from experiments.t4_code.fused_e2m1_check import compare

    good = compare(torch.ones(1, dtype=torch.bfloat16),
                   torch.ones(1, dtype=torch.float64),
                   torch.zeros(1, dtype=torch.float64), True)
    assert good["ok"]
    bad = compare(torch.ones(1, dtype=torch.bfloat16),
                  torch.full((1,), float("inf"), dtype=torch.float64),
                  torch.zeros(1, dtype=torch.float64), True)
    assert not bad["ok"]


def test_fractional_rung_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        qual.main(["--mode", "dry-run", "--out", str(tmp_path / "q.json"),
                   "--q256", "900"])


def test_dry_run_reads_real_bytes(tmp_path):
    out = tmp_path / "dry.json"
    rc = qual.main(["--mode", "dry-run", "--out", str(out),
                    "--q256", "128", "1024",
                    "--ms", "1",
                    "--projections", "gate", "dense",
                    "--experts", "2", "--top-k", "1"])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["mode"] == "dry-run"
    assert report["population"]["cells"] == 4
    assert not report["skips"]
    for cell in report["cells"]:
        assert cell["wire_bytes"] > 0
        assert cell["pure"] is True
        assert cell["cuda_repack"] == "not-called"
        assert cell["serving_owner"] is False
        assert cell["decoded_shape"] == [cell["reader_rows"], cell["reader_cols"]]
        assert len(cell["routing"]) == 1


def test_quality_loader_prefers_small_tiles():
    assert qual.TABLE_BYTES == 16384
    assert qual.SCHEMA == "tessera.t4_fused_qualify.v1"
