"""The register-direct D41 table producer: correctness coverage, the D32 source seals and the rate range.

Review rev-1007-125515-85fd of PR 1029: a timed cell needs a passing correctness check at the
same (mode, profile, M); the compile receipt's source identity is a seal (``seal_check``).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "regdirect_stage1"))
import d41_table  # noqa: E402


def _run(checks, cells=(("0", "R768", 1), ("0", "R768", 16))):
    return {"meta": {}, "check": [{"mode": m, "profile": p, "M": M, "pass": ok} for m, p, M, ok in checks],
            "cells": {f"{arm}.mode{m}.{p}.M{M}": {} for m, p, M in cells for arm in ("baseline", "regdirect")}}


def test_every_timed_cell_needs_a_passing_check_at_its_own_mode_profile_and_m():
    d41_table.require_checked_cells(_run([(0, "R768", 1, True), (0, "R768", 16, True)]), "run")
    with pytest.raises(ValueError, match=r"mode0\.R768\.M16"):
        d41_table.require_checked_cells(_run([(0, "R768", 1, True)]), "run")            # M16 unchecked
    with pytest.raises(ValueError, match=r"mode0\.R768\.M16"):
        d41_table.require_checked_cells(_run([(0, "R768", 1, True), (0, "R768", 16, False)]), "run")
    with pytest.raises(ValueError, match="check"):
        d41_table.require_checked_cells(_run([]), "run")                                 # --skip-check


def test_the_compile_receipt_source_is_a_seal(monkeypatch):
    assert d41_table.compile_receipt_matches({"source_sha256": "a"}, "a")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(ValueError, match="measured kernel source"):
        d41_table.compile_receipt_matches({"source_sha256": "a"}, "b")
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    assert d41_table.compile_receipt_matches({"source_sha256": "a"}, "b") is False


def test_the_rung_range_spans_the_rates_the_kernel_serves():
    from tessera import regdirect_routed as rr
    assert (d41_table.RUNG_MIN, d41_table.RUNG_MAX) == (256 * min(rr.SERVED_RATES), 256 * max(rr.SERVED_RATES))


def test_each_timing_run_is_of_the_measured_kernel_source(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    assert d41_table.require_run_source({"meta": {"kernel_source_sha256": "a"}}, "run", "a")
    with pytest.raises(ValueError, match="measured kernel source"):
        d41_table.require_run_source({"meta": {"kernel_source_sha256": "a"}}, "run", "b")
    with pytest.raises(ValueError, match="records no kernel source"):
        d41_table.require_run_source({"meta": {}}, "run", "a")


def test_the_prefetch_depth_is_the_one_the_run_recorded_for_each_rate():
    cell = {"meta": {"decode_depth": {"5": 3, "6": 2}}}
    assert d41_table.prefetch_depth(cell, [5, 6], "run") == {"5": 3, "6": 2}
    with pytest.raises(ValueError, match="decode depth"):
        d41_table.prefetch_depth({"meta": {}}, [5], "run")
    with pytest.raises(ValueError, match="decode depth"):
        d41_table.prefetch_depth({"meta": {"decode_depth": {"5": 3}}}, [5, 6], "run")


def test_quality_rows_come_from_the_table_then_each_rung_quality_file_once(tmp_path):
    import json
    table, extra = tmp_path / "table.json", tmp_path / "extra.json"
    table.write_text(json.dumps({"rungs": [{"rung": 768, "quality": {"measurement_status": "measured"}}]}))
    extra.write_text(json.dumps({"schema": "tessera.rung_quality.v1",
                                 "rungs": {"1280": {"measurement_status": "measured"}}}))
    src = d41_table.quality_sources(str(table), [str(extra)])
    assert src == {768: ({"measurement_status": "measured"}, str(table)),
                   1280: ({"measurement_status": "measured"}, str(extra))}
    with pytest.raises(ValueError, match="rung 1280 already has quality"):
        d41_table.quality_sources(str(table), [str(extra), str(extra)])
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": "other", "rungs": {}}))
    with pytest.raises(ValueError, match="tessera.rung_quality.v1"):
        d41_table.quality_sources(str(table), [str(wrong)])
