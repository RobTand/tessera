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


def _cell(f_wall, r_wall, *, kernel=None, per_joule=None, watts=None):
    cell = {"F": {"wall": {"median_ms": f_wall}}, "R": {"wall": {"median_ms": r_wall}}}
    if kernel is not None:
        cell["F"]["profile"] = {"kernel_us_per_call": kernel}
    if per_joule is not None:
        cell["F"]["power"] = {"calls_per_j": per_joule, "mean_w": watts}
    return cell


def _doc(table):
    return {"meta": {"ms": sorted({int(m) for cells in table.values() for m in cells})},
            "rungs": {str(q): {"cells": {str(m): cell for m, cell in cells.items()}}
                      for q, cells in table.items()}}


def test_a_cell_time_is_the_mean_of_the_two_passes_and_their_spread():
    got = bench.cell_time(_cell(1.0, 1.2), "wall")
    assert got["ms"] == pytest.approx(1.1)
    assert got["spread"] == pytest.approx(0.2 / 1.1)
    assert bench.cell_time({"F": {"wall": {"median_ms": 1.0}}}, "wall") is None
    assert bench.cell_time(_cell(1.0, 1.2), "graph") is None


def test_the_summary_divides_each_rung_by_the_reference_at_the_same_m():
    doc = _doc({
        1024: {1: _cell(1.0, 1.0, kernel=800.0, per_joule=10.0, watts=50.0),
               512: _cell(10.0, 10.0, kernel=9000.0, per_joule=1.0, watts=70.0)},
        1152: {1: _cell(1.5, 1.7, kernel=1300.0, per_joule=6.0, watts=52.0),
               512: _cell(14.0, 14.0, kernel=13000.0, per_joule=0.7, watts=72.0)},
    })
    summary = bench.summarize_rungs(doc, 1024)
    assert summary["1"]["1152"]["wall"]["ratio"] == pytest.approx(1.6)
    assert summary["512"]["1152"]["wall"]["ratio"] == pytest.approx(1.4)
    assert summary["1"]["1152"]["kernel"]["ratio"] == pytest.approx(1300.0 / 800.0)
    assert summary["1"]["1024"]["wall"]["ratio"] == pytest.approx(1.0)
    assert summary["1"]["1152"]["wall"]["spread"] == pytest.approx(0.2 / 1.6)
    assert summary["1"]["1024"]["rank"] == 1 and summary["1"]["1152"]["rank"] == 2
    assert summary["512"]["1152"]["fwd_per_j"] == pytest.approx(0.7)


def test_the_bar_names_every_cell_above_it_and_none_at_or_below_it():
    doc = _doc({
        1024: {1: _cell(1.0, 1.0), 512: _cell(10.0, 10.0)},
        1152: {1: _cell(1.5, 1.5), 512: _cell(15.1, 15.1)},
        1088: {1: _cell(1.4, 1.4), 512: _cell(14.0, 14.0)},
    })
    summary = bench.summarize_rungs(doc, 1024)
    assert bench.rungs_over_the_bar(summary, 1024) == [(1152, 512, pytest.approx(1.51))]
    assert bench.BAR == 1.5
