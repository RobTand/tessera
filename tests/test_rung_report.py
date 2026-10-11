"""The rung report prints the receipts' ratios, and marks the cells above the bar (tessera#694)."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "experiments/t8r_speed"
sys.path.insert(0, str(HERE))


def _load():
    spec = importlib.util.spec_from_file_location("rung_report", HERE / "rung_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load()

GATE_UP = "void (anonymous namespace)::routed_fused_kernel<true, 0, false, false, {r}, {two}, 64, false, false>(int)"
DOWN = "void (anonymous namespace)::routed_fused_kernel<true, 2, false, false, {r}, {two}, 64, false, false>(int)"


def _cell(wall, graph, gate_up, down, token_sum, *, r, two, per_joule, watts):
    prof = {"kernel_us_per_call": gate_up + down + token_sum,
            "top": {GATE_UP.format(r=r, two=str(two).lower()): {"us_per_call": gate_up},
                    DOWN.format(r=r, two=str(two).lower()): {"us_per_call": down},
                    "(anonymous namespace)::token_sum_kernel(unsigned short const*)": {"us_per_call": token_sum}}}
    forward = {"wall": {"median_ms": wall}, "profile": prof,
               "power": {"calls_per_j": per_joule, "mean_w": watts}}
    reverse = {"wall": {"median_ms": wall}}
    if graph:
        forward["graph"], reverse["graph"] = {"median_ms": graph}, {"median_ms": graph}
    return {"F": forward, "R": reverse}


def _doc():
    """R1024 at 1.0 ms (M 1) and 10 ms (M 512); R832 at 1.3 and 13; R1088 at 1.4 and 15.1."""
    return {"meta": {"ms": [1, 512], "host": "h", "kernel_sha": "8a1536df00", "pb_action": "fd310391aa"},
            "rungs": {
                "1024": {"cells": {"1": _cell(1.0, 0.8, 300.0, 200.0, 10.0, r=4, two=False, per_joule=40.0, watts=70.0),
                                   "512": _cell(10.0, None, 6000.0, 3000.0, 100.0, r=4, two=False, per_joule=1.3, watts=72.0)}},
                "832": {"cells": {"1": _cell(1.3, 1.0, 400.0, 250.0, 10.0, r=3, two=True, per_joule=30.0, watts=75.0),
                                  "512": _cell(13.0, None, 8000.0, 4000.0, 100.0, r=3, two=True, per_joule=0.9, watts=76.0)}},
                "1088": {"cells": {"1": _cell(1.4, 1.2, 450.0, 290.0, 10.0, r=4, two=True, per_joule=28.0, watts=75.0),
                                   "512": _cell(15.1, None, 9000.0, 4500.0, 100.0, r=4, two=True, per_joule=0.8, watts=76.0)}}}}


def _real():
    def group(name, wall, graph, kernel_ms):
        return {"group": name, "cells": {
            "1": {"wall": {"median_ms": wall}, "graph": {"median_ms": graph},
                  "profile": {"kernel_us_per_call": kernel_ms * 1000.0}},
            "512": {"wall": {"median_ms": wall * 10}, "profile": {"kernel_us_per_call": kernel_ms * 9900.0}}}}
    return {"meta": {"pb_action": "fd310391aa"},
            "results": [group("experts.R1024.L10", 1.0, 0.8, 0.9), group("experts.R1088.L11", 1.45, 1.2, 1.3),
                        group("experts.R832.L42", 1.3, 1.1, 1.2)]}


def _run(tmp_path, capsys, *args, real=False):
    run = tmp_path / "run.json"
    run.write_text(json.dumps(_doc()))
    argv = ["--run", f"A={run}", "--splits", "1", *args]
    if real:
        path = tmp_path / "real.json"
        path.write_text(json.dumps(_real()))
        argv += ["--real", str(path)]
    assert report.main(argv) == 0
    return capsys.readouterr().out


def test_the_ratio_table_marks_only_the_cells_above_the_bar(tmp_path, capsys):
    out = _run(tmp_path, capsys)
    wall = out.split("eager forward wall:")[1].split("Time over")[0]
    assert "| 1 | 1 | 1.300 | 1.400 |" in wall
    assert "| 512 | 1 | 1.300 | **1.510** |" in wall
    assert "| max | 1 | 1.300 | **1.510** |" in wall
    graph = out.split("graph replay (M <= 8):")[1].split("Time over")[0]
    assert "| 1 | 1 | 1.250 | 1.500 |" in graph
    assert "**" not in graph
    assert "| 512 |" not in graph


def test_the_launch_split_and_the_joule_ranking_come_from_the_forward_pass(tmp_path, capsys):
    out = _run(tmp_path, capsys)
    split = out.split("Launch split at M = 1")[1].split("Work per joule")[0]
    assert "| R1088 | 450.0 | 1.500 | 290.0 | 1.450 | 750.0 | 1.471 |" in split
    assert "| R1024 | 300.0 | 1.000 | 200.0 | 1.000 | 510.0 | 1.000 |" in split
    ranking = out.split("Work per joule at M = 1:")[1]
    assert ranking.index("R1024") < ranking.index("R832") < ranking.index("R1088")
    assert "| 1 | R1024 | 70.0 | 0.50 | 40.000 |" in ranking


def test_the_real_wire_table_sets_both_ratios_and_the_harness_bias_side_by_side(tmp_path, capsys):
    out = _run(tmp_path, capsys, real=True)
    wall = out.split("Real wires against synthetic, eager forward wall")[1].split("Real wires")[0]
    assert "| 1 | 1.450 | 1.400 | 1.300 | 1.300 | 1.000 | 0.966 | 1.000 |" in wall
    assert "| 512 | 1.450 | **1.510** | 1.300 | 1.300 | 1.000 | 1.041 | 1.000 |" in wall
    graph = out.split("Real wires against synthetic, graph replay")[1].split("Real wires")[0]
    assert "| 1 |" in graph and "| 512 |" not in graph


def test_the_maxima_table_names_each_runs_worst_cell_per_rung(tmp_path, capsys):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps(_doc()))
    worse = _doc()
    worse["rungs"]["1088"]["cells"]["1"] = _cell(1.6, 1.2, 450.0, 290.0, 10.0, r=4, two=True, per_joule=28.0, watts=75.0)
    second.write_text(json.dumps(worse))
    assert report.main(["--run", f"A={first}", "--run", f"B={second}", "--maxima-only"]) == 0
    out = capsys.readouterr().out
    assert "Run A" not in out
    wall = out.split("eager forward wall:")[1].split("graph replay")[0]
    assert "| R1088 | **1.510** at M 512 | **1.600** at M 1 |" in wall
    assert "| R832 | 1.300 at M 512 | 1.300 at M 512 |" in wall
