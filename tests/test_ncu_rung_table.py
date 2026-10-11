"""The NCU table of the synthetic-rung bench reads a raw report and splits it over the bench's calls (tessera#694)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/t8r_speed"


def _load():
    spec = importlib.util.spec_from_file_location("ncu_rung_table", HERE / "ncu_rung_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table = _load()

GATE_UP = "void (anonymous namespace)::routed_fused_kernel<true, 0, false, false, {r}, {two}, 64, false, false>(int)"
DOWN = "void (anonymous namespace)::routed_fused_kernel<true, 2, false, false, {r}, {two}, 64, false, false>(int)"
TOKEN_SUM = "(anonymous namespace)::token_sum_kernel(unsigned short const*, unsigned short*, long, int, long)"


def _csv(calls):
    """A raw-page CSV: one header row, one units row, one row per launch."""
    duration = table.METRICS["duration_us"]
    stall = table.METRICS["stall_barrier"]
    lines = ['==PROF== Connected to process 1',
             f'"ID","Kernel Name","{duration}","{stall}","launch__grid_size"',
             '"","","nsecond","","block"']
    ident = 0
    for r, two, ns in calls:
        for name, scale in ((GATE_UP, 3.0), (DOWN, 1.0), (TOKEN_SUM, 0.01)):
            kernel = name.format(r=r, two=str(bool(two)).lower())
            lines.append(f'"{ident}","{kernel}","{int(ns * scale):,}","{2.5 + ident / 10}","48"')
            ident += 1
    return "\n".join(lines) + "\n"


def _cells(rungs, ms):
    return {"rungs": {str(q): {"cells": {str(m): {"ncu": True} for m in ms}} for q in rungs}}


def test_a_routed_launch_is_named_by_its_template():
    got = table.parse_template(GATE_UP.format(r=4, two="true"))
    assert got == {"fp8": True, "mode": 0, "dense": False, "split": False, "r_lo": 4, "two": True,
                   "bm": 64, "piece_major": False, "paired": False}
    assert table.parse_template(TOKEN_SUM) is None
    with pytest.raises(ValueError, match="template arguments"):
        table.parse_template("routed_fused_kernel<true, 0>")


def test_the_launches_split_evenly_over_the_calls_in_the_bench_order():
    cells = _cells([832, 1088], [1, 512])
    calls = table.cell_order(cells)
    assert calls == [(832, 1), (832, 512), (1088, 1), (1088, 512)]
    launches, missing = table.read_launches(_csv([(3, 1, 1000), (3, 1, 2000), (4, 1, 1500), (4, 1, 2500)]))
    groups = table.split_by_call(launches, calls)
    table.check_pattern(groups)
    assert [(g["rung"], g["M"]) for g in groups] == calls
    assert [l["template"]["r_lo"] for l in groups[2]["launches"] if l["template"]] == [4, 4]
    assert groups[0]["launches"][0]["duration_us"] == pytest.approx(3.0)
    assert groups[1]["launches"][0]["stall_barrier"] == pytest.approx(2.5 + 3 / 10)
    assert "duration_us" not in missing and "stall_long_scoreboard" in missing


def test_a_split_that_does_not_divide_refuses_by_count():
    launches, _ = table.read_launches(_csv([(4, 0, 1000), (4, 0, 2000)]))
    with pytest.raises(ValueError, match="do not split"):
        table.split_by_call(launches, [(1024, 1), (1024, 2), (1024, 3), (1024, 4)])


def test_calls_with_different_kernel_patterns_refuse():
    launches, _ = table.read_launches(_csv([(4, 0, 1000), (4, 0, 2000)]))
    groups = table.split_by_call(launches, [(1024, 1), (1024, 512)])
    groups[1]["launches"] = groups[1]["launches"][:2]
    with pytest.raises(ValueError, match="different kernel patterns"):
        table.check_pattern(groups)


def test_the_cli_writes_the_table_and_prints_every_launch(tmp_path, capsys):
    csv_path, cells_path, out = tmp_path / "raw.csv", tmp_path / "cells.json", tmp_path / "table.json"
    csv_path.write_text(_csv([(3, 0, 1000), (4, 1, 2000)]))
    cells_path.write_text(json.dumps(_cells([768, 1088], [1])))
    assert table.main(["--csv", str(csv_path), "--cells", str(cells_path), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert [(c["rung"], c["M"]) for c in written["calls"]] == [(768, 1), (1088, 1)]
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert sum("launch" in p for p in printed) == 6
    assert any(p.get("missing_metrics") for p in printed)


def test_the_markdown_table_lists_the_routed_launches_and_dashes_what_the_report_lacks():
    launches, _ = table.read_launches(_csv([(3, 1, 1000), (4, 0, 2000)]))
    groups = table.split_by_call(launches, [(832, 1), (1024, 1)])
    out = table.markdown(groups)
    assert out.startswith("M = 1:")
    lines = [l for l in out.splitlines() if l.startswith("| R") and l[3].isdigit()]
    assert len(lines) == 4
    assert lines[0].startswith("| R832 | gate/up | 3/4 | 3.0 | - | - | - | 2.50 |")
    assert lines[3].startswith("| R1024 | down | 4 | 2.0 | - | - | - | 2.90 |")
