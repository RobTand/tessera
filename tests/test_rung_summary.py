"""The reduction of the synthetic-rung bench's cells: each rung over the reference at the same M (tessera#694)."""
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/t8r_speed"


def _load():
    spec = importlib.util.spec_from_file_location("rung_summary", HERE / "rung_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rung_summary = _load()


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
    got = rung_summary.cell_time(_cell(1.0, 1.2), "wall")
    assert got["ms"] == pytest.approx(1.1)
    assert got["spread"] == pytest.approx(0.2 / 1.1)
    assert rung_summary.cell_time({"F": {"wall": {"median_ms": 1.0}}}, "wall") is None
    assert rung_summary.cell_time(_cell(1.0, 1.2), "graph") is None


def test_the_summary_divides_each_rung_by_the_reference_at_the_same_m():
    doc = _doc({
        1024: {1: _cell(1.0, 1.0, kernel=800.0, per_joule=10.0, watts=50.0),
               512: _cell(10.0, 10.0, kernel=9000.0, per_joule=1.0, watts=70.0)},
        1152: {1: _cell(1.5, 1.7, kernel=1300.0, per_joule=6.0, watts=52.0),
               512: _cell(14.0, 14.0, kernel=13000.0, per_joule=0.7, watts=72.0)},
    })
    summary = rung_summary.summarize_rungs(doc, 1024)
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
    summary = rung_summary.summarize_rungs(doc, 1024)
    assert rung_summary.rungs_over_the_bar(summary, 1024) == [(1152, 512, pytest.approx(1.51))]
    assert rung_summary.BAR == 1.5
