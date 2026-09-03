"""The invariant that tells a trustworthy pair-sweep file from a broken one.

Issue #93 states it exactly, and this is where a test can read it: at every
rung ``q256``, the count of ``R{q}_vs_shipped`` entries must equal

    |{L in pair_bits : L * 256 >= q256}| x |pair_ratios|

-- 12 entries per rung on the shipped ``{12, 14, 16} x {1.0, 1.25, 1.4142,
1.75}`` grid, where the file written by the broken revision had 4.

Two properties are worth separating.  The first is that the *denominator*
comes from the run's own recorded ``args`` and not from the rows that reached
the comparison; a count taken from the survivors is self-fulfilling and reads
``N of N`` over any drop at all.  The second is that an absent cell is
*named*, because "eight of twelve" tells a reader that something is missing
and nothing about what to re-run.

This module needs no torch and nothing from ``src/``, so the rule stays
checkable in the pure interpreter -- which matters, because the failure it
guards is a bookkeeping one and would otherwise only ever be exercised on a
box with a GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pair_grid_audit", ROOT / "experiments" / "pair_grid_audit.py")
PGA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PGA)

BITS = [12, 14, 16]
RATIOS = [1.0, 1.25, 1.4142135623730951, 1.75]


def _cmp(q256, bits=BITS, ratios=RATIOS, *, drop=()):
    """A comparison block over the grid, minus the cells in ``drop``."""
    return {PGA.pair_arm_key(q256, L, r): {"bytes_matched": True}
            for L in bits for r in ratios
            if PGA.cell_label(L, r) not in drop and L * 256 >= q256}


def _doc(units, *, rungs=(1024,), bits=BITS, ratios=RATIOS):
    return {"args": {"pair_bits": list(bits), "pair_ratios": list(ratios),
                     "rungs": list(rungs)},
            "gate": "h", "units": units}


# ------------------------------------------------------------ the invariant

@pytest.mark.parametrize("q256,expect", [
    (1024, 12),      # every width is legal at R=4
    (2048, 12),      # ... and at R=8: 12 * 256 = 3072 >= 2048
    (3300, 8),       # L=12 costs a table wider than the rung's budget
    (3800, 4),       # only L=16 survives
])
def test_the_grid_is_the_widths_the_rung_can_carry(q256, expect):
    cells = PGA.requested_cells(q256, BITS, RATIOS)
    assert len(cells) == expect
    assert len(cells) == len(PGA.legal_widths(q256, BITS)) * len(RATIOS)
    assert all(L * 256 >= q256 for L, _ in cells)


def test_a_full_grid_is_complete():
    audit = PGA.audit_rung(1024, BITS, RATIOS, _cmp(1024))
    assert (audit["expected"], audit["present"]) == (12, 12)
    assert audit["complete"] and not audit["missing"]


def test_the_denominator_survives_a_drop():
    """The whole of #93 in one assertion: 12 does not become 8."""
    drop = {PGA.cell_label(12, r) for r in RATIOS}
    audit = PGA.audit_rung(1024, BITS, RATIOS, _cmp(1024, drop=drop))
    assert audit["expected"] == 12 and audit["present"] == 8
    assert not audit["complete"]
    assert set(audit["missing"]) == drop


def test_a_missing_cell_carries_a_reason():
    audit = PGA.audit_rung(
        1024, BITS, RATIOS, _cmp(1024, drop={"L=16 r=1.75"}),
        reasons={"L=16 r=1.75": PGA.NO_BYTE_MATCH})
    assert audit["missing"] == {"L=16 r=1.75": PGA.NO_BYTE_MATCH}
    # A caller that knows nothing still gets an honest answer, not a guess.
    blind = PGA.audit_rung(1024, BITS, RATIOS, _cmp(1024, drop={"L=16 r=1.75"}))
    assert blind["missing"] == {"L=16 r=1.75": PGA.CANDIDATE_MISSING}


def test_an_unmatched_arm_is_present_but_flagged():
    """Encoded at the wrong bpp is a different fault from never encoded."""
    cmp_ = _cmp(1024)
    cmp_[PGA.pair_arm_key(1024, 12, 1.25)]["bytes_matched"] = False
    audit = PGA.audit_rung(1024, BITS, RATIOS, cmp_)
    assert audit["complete"] and audit["present"] == 12
    assert audit["unmatched"] == ["L=12 r=1.25"]


def test_a_key_outside_the_grid_is_reported():
    """Writer/reader label drift looks exactly like this, and must not be quiet."""
    cmp_ = _cmp(1024)
    cmp_["R1024 L=13 r=1"] = {"bytes_matched": True}
    assert PGA.audit_rung(1024, BITS, RATIOS, cmp_)["unexpected"] == \
        ["R1024 L=13 r=1"]


def test_the_log_names_every_absent_cell():
    drop = {PGA.cell_label(12, r) for r in RATIOS}
    lines = "\n".join(PGA.audit_lines(
        PGA.audit_rung(1024, BITS, RATIOS, _cmp(1024, drop=drop))))
    assert "8 of 12 arms" in lines and "GRID INCOMPLETE at R1024" in lines
    assert all(label in lines for label in drop)


# ----------------------------------------------------------- the reader side

def test_reader_passes_a_complete_file(capsys):
    doc = _doc({"unit": {"R1024_vs_shipped": _cmp(1024)}})
    assert PGA.audit_doc(doc)["complete"]


def test_reader_fails_the_shape_of_the_broken_artifact(tmp_path, capsys):
    """The exact reading #93 describes: four cells where twelve were asked for."""
    keep = {PGA.cell_label(14, r) for r in RATIOS}
    drop = {PGA.cell_label(L, r) for L in (12, 16) for r in RATIOS}
    doc = _doc({"unit": {"R1024_vs_shipped": _cmp(1024, drop=drop)}})
    assert len(doc["units"]["unit"]["R1024_vs_shipped"]) == len(keep) == 4

    result = PGA.audit_doc(doc)
    assert not result["complete"]
    audit = result["units"]["unit"]["R1024"]
    assert (audit["expected"], audit["present"]) == (12, 4)

    path = tmp_path / "pair.json"
    path.write_text(json.dumps(doc))
    assert PGA.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "GRID INCOMPLETE" in out and "VERDICT" in out
    assert all(label in out for label in drop)
    # A complete file exits 0, so the CLI is usable as a gate in a chain.
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_doc({"unit": {"R1024_vs_shipped": _cmp(1024)}})))
    assert PGA.main([str(good)]) == 0


def test_reader_treats_a_missing_rung_block_as_a_whole_absent_rung():
    doc = _doc({"unit": {"R1024_vs_shipped": _cmp(1024)}}, rungs=(1024, 2048))
    audit = PGA.audit_doc(doc)["units"]["unit"]["R2048"]
    assert audit["present"] == 0 and audit["expected"] == 12


def test_reader_refuses_a_file_that_does_not_record_its_grid():
    """A file that never says what it asked for cannot be cleared, only refused."""
    doc = _doc({"unit": {"R1024_vs_shipped": _cmp(1024)}})
    del doc["args"]["pair_ratios"]
    with pytest.raises(SystemExit, match="pair_ratios"):
        PGA.audit_doc(doc)
