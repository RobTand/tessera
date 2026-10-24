"""The fallback sweep reports the launch schedule, not just prepared blocks."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


SWEEP = Path(__file__).resolve().parents[1] / "experiments/sweep_window_gemm_decode.py"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the sweep measures a CUDA kernel")
def test_wrapper_records_report_effective_schedule_and_grid(tmp_path, monkeypatch, capsys):
    # Exercise the emitted packet on a small packed unit, including the decode
    # boundary and a multi-M-block launch. Timing values are not performance gates.
    spec = importlib.util.spec_from_file_location("sweep_window_gemm_decode", SWEEP)
    sweep = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(sweep)
    finally:
        sys.path[:] = old_path
    monkeypatch.setattr(sweep, "SHAPES", ((768, 64),))
    monkeypatch.setattr(sweep, "MS", (1, 8, 16, 17, 65))
    monkeypatch.setattr(sweep, "BMS", (64,))
    monkeypatch.setattr(sweep, "BNS", (64,))
    # Telemetry is irrelevant to this metadata regression; don't start nvidia-smi.
    sampler = sweep.PowerSampler()
    sampler.available = False
    monkeypatch.setattr(sweep, "PowerSampler", lambda: sampler)
    out = tmp_path / "sweep.json"
    assert sweep.main(["--out", str(out), "--iterations", "1", "--warmup", "1",
                       "--power-seconds", "0"]) == 0
    records = json.loads(out.read_text())["records"]
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == records
    assert len(records) == 2 * len(sweep.MS)
    for record in records:
        m = record["m"]
        if record["path"] == "raw" or m > 16:
            blocks = (64, 64, 64)
        else:
            blocks = (16, 32, 64)
        assert tuple(record[f"block_{axis}"] for axis in "mnk") == blocks
        if record["path"] == "served":
            assert tuple(record[f"prepared_block_{axis}"] for axis in "mnk") == (64, 64, 64)
        bm, bn, _ = blocks
        grid = [(768 + bn - 1) // bn, (m + bm - 1) // bm]
        assert record["grid"] == grid
        assert record["ctas"] == grid[0] * grid[1]
