"""The synthetic-rung forward bench builds the wire's run table and reads its own ratios (tessera#694).

``experiments/t8r_speed/bench_rung_forward.py`` times the routed E4M3 owner at any
q256 rung.  Its numbers are only as good as the wire it builds, so the run
split and the column order are held to the independent bitstream packer, and
every run table the lane reads is built on the CPU and put through the lane's own
wire checks.  The summary's arithmetic is held to hand-built cells.
"""
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/t8r_speed"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(HERE))


def _load():
    spec = importlib.util.spec_from_file_location("bench_rung_forward", HERE / "bench_rung_forward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()

#: Every rung the packer can realise over 256 columns: one column is 1/256 of a
#: bit, so every integer q256 is exact there.  The step is a cost choice.
Q256_OVER_256_COLUMNS = range(256, 256 * bench.RATE_CAP + 1, 32)


@pytest.mark.parametrize("q256", Q256_OVER_256_COLUMNS)
def test_rung_geometry_is_the_packers_run_table_and_column_order(q256):
    from tessera.grammar import rate_set

    import window_pack_reference as wpr

    cols = 256
    geometry = bench.rung_geometry(q256, cols)
    rates = geometry["rates"]
    assert tuple(sorted(set(rates))) == rate_set(Fraction(q256, 256), cap=bench.RATE_CAP)
    generator = torch.Generator().manual_seed(q256)
    rate = torch.tensor(rates, dtype=torch.int64)
    body = (torch.randint(0, 1 << 16, (512, cols), generator=generator) & ((1 << rate) - 1)).to(torch.int32)
    rep = wpr.pack_bitstream(body, rates)
    assert geometry["perm"] == rep.perm.tolist()
    n_lo = cols - geometry["n_hi"]
    runs = [[geometry["r_lo"], 0, n_lo, 0]]
    if geometry["n_hi"]:
        runs.append([geometry["r_lo"] + 1, n_lo, geometry["n_hi"], 16 * n_lo * geometry["r_lo"]])
    assert rep.runs.tolist() == runs


def test_a_rung_the_columns_cannot_realise_refuses_by_name():
    with pytest.raises(Exception, match="not realisable"):
        bench.rung_geometry(257, 128)


def test_the_cpu_preflight_builds_every_run_table_the_lane_reads():
    pytest.importorskip("safetensors")
    rungs = [256 * r for r in range(1, bench.RATE_CAP + 1)]
    rungs += [256 * r + f for r in range(1, bench.RATE_CAP) for f in (64, 128, 192)]
    result = bench.cpu_preflight(sorted(rungs))
    assert result["status"] == "cpu_preflight_passed"
    assert result["rungs"] == sorted(rungs)
