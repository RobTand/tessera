"""The registered reading is code, not prose (issue #18).

``experiments/bf16_matched_reach_run.sh`` registers, before the run exists,
both what the recovered fraction means and where it may be read.  A threshold
that lives only in a comment is a threshold the report can quietly ignore, so
``matched_reach_report.read_split`` carries it and these tests pin it against
the numbers the landed grids actually hold.

Pure: the reader parses JSON, so this runs in the no-torch lane.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

_spec = importlib.util.spec_from_file_location(
    "matched_reach_report", ROOT / "experiments" / "matched_reach_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def test_the_floor_is_one_percent_and_it_is_a_log():
    assert report.LOG_A_FLOOR == math.log(1.01)


def test_the_one_landed_cell_the_floor_excludes():
    """GLM R=4 ``L=16`` lands at 1.0029x -- the header names it by number."""
    said = report.read_split(1.0029, 0.99)
    assert said.startswith("NOT READ"), said
    assert "0.29%" in said, said
    # and its neighbour at 1.0134x clears it, so the floor is not blanket.
    assert not report.read_split(1.0134, 0.99).startswith("NOT READ")


def test_every_landed_dense_cell_is_readable():
    for A in (0.8916, 1.1882, 0.9003, 1.1720, 0.9565, 1.1655):
        assert not report.read_split(A, 0.97).startswith("NOT READ"), A


def test_the_three_verdicts_and_the_opposite_sign_case():
    # A = 0.9332 (the landed GLM R=8 L=16 bundle).  B chosen to sit in each band.
    A = 0.9332
    assert "SPREAD" in report.read_split(A, A ** 0.8)
    assert "ENTRY COUNT" in report.read_split(A, A ** 0.05)
    assert "BOTH" in report.read_split(A, A ** 0.3)
    # A win that the spread alone reverses is reported as the bundle it is,
    # and never as a fraction: log B / log A would be negative and read as
    # "under 15% recovered", which is the opposite of what happened.
    hurts = report.read_split(A, 1.04)
    assert "HURTS" in hurts and "%" not in hurts


def test_a_non_positive_ratio_is_refused_not_charted():
    assert "unreadable" in report.read_split(0.0, 0.9)
    assert "unreadable" in report.read_split(0.9, 0.0)
