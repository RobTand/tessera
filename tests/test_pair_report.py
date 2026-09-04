"""What the pair reader must not let past it.

``experiments/pair_report.py`` is the thing the #18 receipt is read off, so
the properties worth pinning are the ones that decide whether a table can be
believed: every ratio is rebuilt from the raw arms and only against a
reference at the *same bytes*, and a file that lost cells or lost a control
is reported as such rather than summarised.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# ``pair_report`` imports ``pair_grid_audit`` as a sibling, so the directory
# has to be importable -- but the module itself is loaded by path and NOT with
# ``importorskip``. An ``importorskip`` here turns a broken reader into a
# skipped module: a green suite over code nobody ran, which is the failure
# mode #97 is about. If it cannot import, that is the finding.
sys.path.insert(0, str(ROOT / "experiments"))
_spec = importlib.util.spec_from_file_location(
    "pair_report", ROOT / "experiments" / "pair_report.py")
pair_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pair_report)


TABLE_BPP = {12: 0.02, 14: 0.08, 16: 0.32}


def _arm(q256, L, ratio, *, rung, err, bpp):
    return {"bpp": bpp, "q256": q256, "rung": rung, "L": L, "ratio": ratio,
            "wt": err, "h": err, "out": err, "over": 0.1, "reach_rms": 4.0,
            "sha": f"{q256}-{L}-{ratio}", "tsha": f"t{q256}-{L}-{ratio}",
            "secs": 1.0}


def _doc(*, drop_cell=None, break_control=False, unmatched_bytes=False,
         shipped_L=14, requested=None, cross_group=False):
    """A two-unit, one-rung pair document with a known answer.

    Every candidate is built so that the shipped-pair arm at its own bytes
    exists, and ``L=16`` is made 10% better than the reference it is matched
    against -- so a reader that rebuilt ratios correctly must report 2/2 wins
    at 0.9x, and one that matched the wrong reference cannot.
    """
    bits, ratios, q = [12, 14, 16], [1.0, 1.25], 1024
    units = {}
    for u, base in (("u1", 0.10), ("u2", 0.20)):
        res = {"rows": 8, "cols": 8, "numel": 64, "gate": "h",
               "default_L": shipped_L}
        for L in bits:
            bpp = 4.0 + TABLE_BPP[L]
            # the byte-matched shipped-pair reference at this width's bytes
            ref_q = q + int(round((TABLE_BPP[L] - TABLE_BPP[shipped_L]) * 256))
            key = (f"R{ref_q} L={shipped_L} r=1"
                   + ("" if L == shipped_L else f" [bytematch L={L}]"))
            # ``cross_group`` moves the L=16 reference two rungs away while
            # keeping its bytes: an exact bpp match that is not the same
            # comparison, which is the collision ``same_rung_group`` exists
            # to refuse.
            ref_rung = q + 512 if (cross_group and L == 16) else q
            res[key] = _arm(ref_q, shipped_L, 1.0, rung=ref_rung, err=base,
                            bpp=bpp)
            for r in ratios:
                if L == shipped_L and r == 1.0:
                    continue
                factor = 0.9 if L == 16 else 1.1
                arm_bpp = bpp + (0.5 if unmatched_bytes and L == 12 else 0.0)
                res[f"R{q} L={L} r={r:g}"] = _arm(
                    q, L, r, rung=q, err=base * factor, bpp=arm_bpp)
        if drop_cell:
            res.pop(drop_cell, None)
        res[f"R{q} L={shipped_L} r=1 [repeat]"] = _arm(
            q, shipped_L, 1.0, rung=q, err=base,
            bpp=4.0 + TABLE_BPP[shipped_L])
        res[f"R{q}_control"] = {
            "arm": "R1024 shipped pair", "ran": not break_control,
            "bytes_identical": True, "tensor_identical": True,
            "secs_first": 10.0, "secs_repeat": 10.0}
        if break_control:
            res[f"R{q}_control"] = {"arm": "x", "ran": False, "reason": "gone"}
        # A stored comparison that is present, complete and WRONG: every
        # ratio 1.0.  The reader must ignore it and rebuild from the arms --
        # that is the property under test, and a file whose stored block was
        # merely absent would not distinguish "rebuilt" from "read".
        res[f"R{q}_vs_shipped"] = {
            k: {"ref": "whatever", "bpp_gap": 0.0, "bytes_matched": True,
                "wt_ratio": 1.0, "h_ratio": 1.0, "out_ratio": 1.0}
            for k in list(res) if k.startswith(f"R{q} L=") and "[" not in k}
        if drop_cell:
            res[f"R{q}_vs_shipped"].pop(drop_cell, None)
        units[u] = res
    args = {"pair_bits": bits, "pair_ratios": ratios, "rungs": [q],
            "eval_rows": 8}
    if requested is not None:
        args["stage"], args["units"] = "pair-dense", list(requested)
    return {"args": args, "grid": "bf16", "population": "test", "gate": "h",
            "units": units}


def _run(tmp_path, doc, name="pair.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return pair_report.report(str(p))


def test_ratios_come_from_the_raw_arms_at_matched_bytes(tmp_path, capsys):
    """The reader must not read the stored comparison, which here is wrong.

    The fixture's ``R{q}_vs_shipped`` is complete and says every arm is level
    with the shipped pair. The arms say ``L=16`` is 10% better than the
    reference at its own bytes, on both units, and that is what a reader that
    rebuilds must print.
    """
    assert _run(tmp_path, _doc()) is True
    out = capsys.readouterr().out
    assert "L=16 r=1" in out
    line = [l for l in out.splitlines() if l.startswith("L=16 r=1 ")][0]
    assert line.split()[2:4] == ["2", "2"], line          # wins, wins@1%
    assert "0.9000" in line, line
    assert "every candidate found an exact-bpp reference" in out


def test_a_candidate_with_no_reference_at_its_bytes_is_refused(tmp_path, capsys):
    """A ratio is only a ratio if the two arms provably weigh the same."""
    assert _run(tmp_path, _doc(unmatched_bytes=True)) is False
    assert "UNMATCHED" in capsys.readouterr().out


def test_a_missing_cell_is_named_not_silently_dropped(tmp_path, capsys):
    """#93: an absent cell must not read like a cell that was measured."""
    assert _run(tmp_path, _doc(drop_cell="R1024 L=16 r=1.25")) is False
    out = capsys.readouterr().out
    assert "GRID INCOMPLETE" in out and "L=16 r=1.25" in out


def test_a_control_that_did_not_run_is_not_a_control_that_passed(tmp_path,
                                                                 capsys):
    """#96: three outcomes, and only one of them is 'checked and clean'."""
    assert _run(tmp_path, _doc(break_control=True)) is False
    assert "CONTROL MISSING" in capsys.readouterr().out


def test_a_partial_unit_set_is_named_not_summarised(tmp_path, capsys):
    """A sweep in flight looks exactly like a finished one.

    The units are written as they complete, into one fixed path, so a re-run
    of a sweep truncates the artifact it is reproducing -- which happened to
    ``pair_dense.json`` while the #18 receipt was being merged, and a reader
    that took its denominator from the units present would have reported the
    partial file as a whole population.
    """
    doc = _doc(requested=["u1", "u2", "u3"])
    assert _run(tmp_path, doc) is False
    out = capsys.readouterr().out
    assert "UNIT SET INCOMPLETE: 2 of 3" in out
    assert "missing: u3" in out


def test_a_complete_unit_set_says_so(tmp_path, capsys):
    """Reporting completeness only by silence cannot be told from not checking."""
    assert _run(tmp_path, _doc(requested=["u1", "u2"])) is True
    assert "unit set: 2 of 2 requested units present" in capsys.readouterr().out


def test_an_undeclared_unit_set_is_reported_as_unchecked(tmp_path, capsys):
    assert _run(tmp_path, _doc()) is True
    assert "args declares no unit set -- NOT checked" in capsys.readouterr().out


def test_the_shipped_width_comes_from_the_file_not_the_constant(tmp_path,
                                                                capsys):
    """The reference width is the run's ``default_L``, whatever the recipe is.

    A reader holding ``L=14`` after the recipe moved to 12 would find no
    reference for the shipped arm and drop it, and would score every other arm
    against a width this run never ran.
    """
    assert _run(tmp_path, _doc(shipped_L=12)) is True
    out = capsys.readouterr().out
    assert "shipped L=12" in out
    assert "vs byte-matched shipped pair L=12 r=1" in out
    assert "L=14 r=1 " in out                     # now a candidate, not the ref


def test_a_reference_at_the_wrong_rung_group_is_refused(tmp_path, capsys):
    """Equal bytes are necessary and not sufficient.

    Byte matching deliberately shifts the reference's rung, so the rungs must
    not be required to agree -- but a match more than a table's bytes away is
    two arms that weigh the same and are not the same comparison.
    """
    assert _run(tmp_path, _doc(cross_group=True)) is False
    out = capsys.readouterr().out
    assert "matched across rung groups" in out
