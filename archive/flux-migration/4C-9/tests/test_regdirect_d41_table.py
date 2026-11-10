"""The register-direct D41 table producer: correctness coverage and the D32 source seal.

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
