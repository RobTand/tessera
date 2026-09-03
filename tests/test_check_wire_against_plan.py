"""Three outcomes, not two: agreement, disagreement, and a plan that is silent.

``check_wire_against_plan`` answers "do the exported bytes weigh what the
allocator charged for them?".  It used to fold *"the plan does not price this
unit"* into *"the plan prices it differently"*, so a plan built without the
allocator attached reported every one of its units under ``PER-UNIT MISMATCH``
and printed ``DISAGREEMENT`` -- naming the wire as the offender when the wire
was the only side that had spoken (#49).  Worse, the two totals ran over
different row sets, so the headline equality was false for a reason unrelated
to pricing.

The unpriced rows are real: ``plan_from_layer_config.charged_bits`` returns
``None`` when the PrismaQuant tree is absent or its accounting import fails,
and the sidecar then carries ``prismaquant_charged_bits_exact: null``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_wire_against_plan", ROOT / "experiments" / "check_wire_against_plan.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def _unit(tensor, *, charged, rows=64, cols=512, q256=896):
    """A plan row.  ``charged`` is an exact [num, den] pair or None."""
    return {"tensor": tensor, "prismaquant_charged_bits_exact": charged,
            "q256": q256, "grid": "E2M1x2", "rows": rows, "columns": cols,
            "params": rows * cols}


def _role(tensor, *, wire_bytes, rows=64, cols=512, q256=896):
    return {"tensor": tensor, "wire_bytes": wire_bytes, "q256": q256,
            "grid": "E2M1x2", "rows": rows, "cols": cols}


def _write(tmp_path, units, roles, wire_bpp=4.0):
    prov = tmp_path / "plan.json.provenance.json"
    prov.write_text(json.dumps({"units": units}))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {"m": {"roles": roles}}, "totals": {"wire_bpp": wire_bpp}}))
    return [str(prov), str(ckpt)]


AGREE = _unit("a.q_proj", charged=[16384, 1])
AGREE_ROLE = _role("a.q_proj", wire_bytes=2048)          # 2048 * 8 == 16384


def test_an_unpriced_unit_is_not_reported_as_a_mispriced_one(tmp_path, capsys):
    """The whole issue in one case: one unit agrees, one is unpriced."""
    argv = _write(
        tmp_path,
        [AGREE, _unit("a.k_proj", charged=None)],
        [AGREE_ROLE, _role("a.k_proj", wire_bytes=1048576)])

    code = CHECK.main(argv)
    out = capsys.readouterr().out

    assert code == 1, "identity is not shown for the unpriced unit"
    assert "UNPRICED IN THE PLAN (1)" in out, out
    assert "a.k_proj" in out.split("UNPRICED IN THE PLAN")[1], out
    # The detection path it must NOT take.
    assert "PER-UNIT MISMATCH" not in out, out
    assert "VERDICT: DISAGREEMENT" not in out, out
    assert "the priced units agree" in out, out


def test_the_totals_are_over_the_same_rows_when_a_unit_is_unpriced(tmp_path, capsys):
    """A subset of the bits divided by all of the params is low by construction.

    Both totals and the denominator now run over the priced rows, and the lines
    say so, so nobody quotes a partial number as a whole-model bpp.
    """
    argv = _write(
        tmp_path,
        [AGREE, _unit("a.k_proj", charged=None)],
        [AGREE_ROLE, _role("a.k_proj", wire_bytes=1048576)])

    CHECK.main(argv)
    out = capsys.readouterr().out

    charged = next(l for l in out.splitlines() if l.startswith("charged"))
    emitted = next(l for l in out.splitlines() if l.startswith("emitted"))
    # 16384 bits over 64*512 params -- the priced unit alone, on both lines.
    assert "16384 bits" in charged and "16384 bits" in emitted, out
    assert "0.500000000 bpp" in charged and "0.500000000 bpp" in emitted, out
    assert "(priced units only)" in charged and "(priced units only)" in emitted, out
    assert "units compared        2 (1 priced, 1 unpriced)" in out, out
    assert "(whole checkpoint)" in out, "the manifest bpp is over every unit"


def test_a_fully_priced_agreeing_plan_still_passes(tmp_path, capsys):
    """The over-correction guard: do not fail everything to fix one case."""
    argv = _write(tmp_path, [AGREE], [AGREE_ROLE])

    code = CHECK.main(argv)
    out = capsys.readouterr().out

    assert code == 0, out
    assert "VERDICT: the bytes served are the bytes priced" in out, out
    assert "UNPRICED" not in out, out
    assert "(priced units only)" not in out, "no scope caveat when nothing is unpriced"


def test_a_genuinely_mispriced_unit_still_disagrees(tmp_path, capsys):
    """The original detection path is intact -- both sides spoke and differ."""
    argv = _write(
        tmp_path,
        [AGREE, _unit("a.v_proj", charged=[16384, 1])],
        [AGREE_ROLE, _role("a.v_proj", wire_bytes=4096)])   # 32768 != 16384

    code = CHECK.main(argv)
    out = capsys.readouterr().out

    assert code == 1
    assert "PER-UNIT MISMATCH (1)" in out, out
    assert "a.v_proj: charged 16384 R896 vs wire 32768 R896" in out, out
    assert "VERDICT: DISAGREEMENT" in out, out
    assert "UNPRICED IN THE PLAN" not in out, out


def test_a_zero_charge_is_priced_and_not_read_as_silence(tmp_path, capsys):
    """``is None``, not truthiness -- pinning a contract, not fixing a live bug.

    Be precise about what changed.  The old guard tested the *list*, and
    ``[0, 1]`` is truthy, so this case was already handled correctly; the
    truthiness bug was latent, waiting on a writer that emits ``0`` or ``[]``
    for a zero charge.  This test does not reproduce on the old code and is not
    claimed to.  It states the contract the new ``is None`` makes explicit: a
    charge of zero is a PRICE, and disagreeing with the wire about it is a
    mispricing, never silence.
    """
    argv = _write(
        tmp_path,
        [_unit("a.q_proj", charged=[0, 1])],
        [AGREE_ROLE])

    code = CHECK.main(argv)
    out = capsys.readouterr().out

    assert code == 1
    assert "PER-UNIT MISMATCH (1)" in out, out
    assert "charged 0 R896 vs wire 16384 R896" in out, out
    assert "UNPRICED" not in out, out


def test_missing_and_extra_stay_disagreements(tmp_path, capsys):
    """They are not the same shape as unpriced, and the verdict says so.

    An unpriced unit is the plan declining to make a claim.  A unit in the plan
    and not in the export -- or the reverse -- is the two sides making
    contradictory claims about which units exist.  That is a disagreement.
    """
    argv = _write(tmp_path, [AGREE, _unit("a.gone", charged=[8, 1])], [AGREE_ROLE])

    code = CHECK.main(argv)
    out = capsys.readouterr().out

    assert code == 1
    assert "IN THE PLAN, NOT EXPORTED (1)" in out, out
    assert "VERDICT: DISAGREEMENT" in out, out


def test_a_wholly_unpriced_plan_does_not_divide_by_zero(tmp_path, capsys):
    """The no-PrismaQuant path prices nothing at all, so params is 0."""
    argv = _write(tmp_path, [_unit("a.q_proj", charged=None)], [AGREE_ROLE])

    code = CHECK.main(argv)
    out = capsys.readouterr().out

    assert code == 1
    assert "no priced unit to total" in out, out
    assert "UNPRICED IN THE PLAN (1)" in out, out
