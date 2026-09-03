"""A cell the pair stage did not run must be named, not subtracted (#93).

The ``pair-dense``/``pair-glm`` stage compares each ``(L, ratio)`` candidate
against a byte-matched reference and skips a cell whose reference it could
not build or whose encode failed.  The skip was correct; the accounting was
not.  Its audit line

    byte match: N of M arms sit at their reference's exact bpp

took ``M`` from the comparison it had just assembled, so ``M`` shrank in step
with every drop and the ratio read clean no matter how much was missing.  The
one artifact written that way reported ``2 of 2`` over a grid of four, and the
``best on <gate>`` line beneath it named an arm that was not the best -- by
28% on the gate metric.

These tests drive ``run_pair_unit`` itself, the function both revisions have,
with the encode stubbed out; only the bookkeeping is under test.  Each one
removes a known set of cells by a *different* mechanism, because the three
drop paths are separate lines of code:

* no integral byte-matched rung exists on this shape (``bytematched_rung``
  returns ``None``);
* the byte-matched **reference** arm failed to encode;
* a **candidate** arm failed to encode.

All three fail against the pre-fix stage, which reports the surviving count
as the denominator and prints no label for what it dropped.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

#: 1024 x 1024 is the shape the byte match is exact on: a BF16 table costs
#: ``2^L * 16 / numel`` bpp, so L=12 is 0.0625 and L=14 is 0.25, a delta of
#: exactly -48 q256 steps.  A shape where the delta is fractional would drop
#: cells for a reason this test is not about.
ROWS = COLS = 1024
NUMEL = ROWS * COLS
RUNG = 1024
RATIOS = [1.0, 1.25, 1.4142135623730951, 1.75]
BITS = [12, 14, 16]


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location(
        "bf16_l_sigma_sweep", ROOT / "experiments" / "bf16_l_sigma_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub(sweep, monkeypatch, *, fail=(), no_match=()):
    """Replace every arm's real work with arithmetic, and choose the drops.

    ``fail`` is a set of arm keys ``try_arm`` reports as failed -- the same
    ``None`` the real one returns when an encode raises.  ``no_match`` is a
    set of widths for which no integral byte-matched rung exists.
    """
    monkeypatch.setattr(sweep, "reach_stats",
                        lambda *a, **k: {"reach_row_rms": 1.0,
                                         "rows_over_reach": 0.0})
    monkeypatch.setattr(sweep, "tensor_sha", lambda t: "tsha")
    # An error that is a strictly increasing function of neither axis alone,
    # so a subset argmin and the grid's argmin can differ.
    monkeypatch.setattr(sweep, "score",
                        lambda w, hat, **k: {"wt": 0.5, "h": 0.5})
    monkeypatch.setattr(
        sweep, "check_repeat_tensor",
        lambda b, first, last, label: {"arm": label, "bytes_identical": True,
                                       "tensor_identical": True})
    real_rung = sweep.bytematched_rung
    monkeypatch.setattr(
        sweep, "bytematched_rung",
        lambda q, L, dL, numel: None if L in no_match else real_rung(q, L, dL, numel))

    def try_arm(b, label, fn):
        if label in fail:
            b.log(f"    {label:<30} !! FAILED: RuntimeError: stubbed")
            return None
        return fn()

    monkeypatch.setattr(sweep, "try_arm", try_arm)

    def encode_arm(w, grid, q256, name, *, window_bits, window_sigma,
                   channel_sigma):
        return w, q256 / 256 + sweep.table_bpp(window_bits, w.numel()), "sha", 0.0

    monkeypatch.setattr(sweep, "encode_arm", encode_arm)


def _run(sweep, tmp_path, **kw):
    b = sweep.Bench(str(tmp_path / "pair.json"))
    a = SimpleNamespace(window_bits=14, gate="h", rungs=[RUNG],
                        pair_bits=list(BITS), pair_ratios=list(RATIOS))
    w = torch.arange(NUMEL, dtype=torch.float32).reshape(ROWS, COLS) % 7 - 3
    res = sweep.run_pair_unit(b, a, "unit", w, "unit", h=torch.ones(COLS))
    return b, res


def _audit_line(b):
    hits = [line for line in b.lines if "byte match:" in line]
    assert len(hits) == 1, b.lines
    return hits[0]


def test_complete_grid_reports_the_whole_grid(sweep, tmp_path, monkeypatch):
    """The control: nothing dropped, and the denominator is still the grid."""
    _stub(sweep, monkeypatch)
    b, res = _run(sweep, tmp_path)
    assert len(res[f"R{RUNG}_vs_shipped"]) == len(BITS) * len(RATIOS) == 12
    audit = res[f"R{RUNG}_grid_audit"]
    assert (audit["expected"], audit["present"], audit["complete"]) == (12, 12, True)
    assert "of 12 arms" in _audit_line(b)
    assert not any("GRID INCOMPLETE" in line for line in b.lines)


def test_a_width_with_no_byte_match_is_named(sweep, tmp_path, monkeypatch):
    """``L=12`` has no integral byte-matched rung: four cells, all named."""
    _stub(sweep, monkeypatch, no_match=[12])
    b, res = _run(sweep, tmp_path)
    audit = res[f"R{RUNG}_grid_audit"]
    assert audit["expected"] == 12 and audit["present"] == 8
    # The denominator is the grid.  Pre-fix this line read "8 of 8".
    assert "8 of 12 arms" in _audit_line(b)
    assert set(audit["missing"]) == {f"L=12 r={r:g}" for r in RATIOS}
    assert all(why == sweep.NO_BYTE_MATCH for why in audit["missing"].values())
    log = "\n".join(b.lines)
    assert "GRID INCOMPLETE" in log
    for r in RATIOS:                       # every absent cell, by name
        assert f"L=12 r={r:g}" in log


def test_a_failed_reference_encode_is_named(sweep, tmp_path, monkeypatch):
    """The reference arm dies, so its whole row does -- and says which row."""
    _stub(sweep, monkeypatch,
          fail={sweep.pair_arm_key(1216, 14, 1.0) + " [bytematch L=16]"})
    b, res = _run(sweep, tmp_path)
    audit = res[f"R{RUNG}_grid_audit"]
    assert audit["expected"] == 12 and audit["present"] == 8
    assert "8 of 12 arms" in _audit_line(b)
    assert set(audit["missing"]) == {f"L=16 r={r:g}" for r in RATIOS}
    assert all(why == sweep.REFERENCE_MISSING
               for why in audit["missing"].values())


def test_a_failed_candidate_encode_is_named(sweep, tmp_path, monkeypatch):
    """One cell, not a row: the count moves by one and the label is exact."""
    _stub(sweep, monkeypatch, fail={sweep.pair_arm_key(RUNG, 14, 1.25)})
    b, res = _run(sweep, tmp_path)
    audit = res[f"R{RUNG}_grid_audit"]
    assert audit["expected"] == 12 and audit["present"] == 11
    assert "11 of 12 arms" in _audit_line(b)
    assert audit["missing"] == {"L=14 r=1.25": sweep.CANDIDATE_MISSING}


def test_an_argmin_over_a_subset_says_so(sweep, tmp_path, monkeypatch):
    """The line that read wrong by 28% now carries its own scope."""
    _stub(sweep, monkeypatch, no_match=[12])
    b, _ = _run(sweep, tmp_path)
    best = [line for line in b.lines if "best on h at matched bytes" in line]
    assert len(best) == 1 and "OVER 8 OF 12 CELLS" in best[0]


def test_summarise_pair_flags_an_incomplete_rung(sweep, tmp_path, monkeypatch):
    """The summary refuses to read a partial grid as the grid."""
    _stub(sweep, monkeypatch, no_match=[12])
    b, res = _run(sweep, tmp_path)
    b.doc = {"args": {"pair_bits": list(BITS), "pair_ratios": list(RATIOS),
                      "rungs": [RUNG]},
             "gate": "h", "units": {"unit": res}}
    b.lines.clear()
    sweep.summarise_pair(b)
    log = "\n".join(b.lines)
    assert "REFUSING TO READ" in log and "GRID INCOMPLETE" in log
    assert b.doc["summary_grid_audit"][f"R{RUNG}"]["complete"] is False


# ------------------------------------------------------- the repeat control
#
# #96.  The shipped arm is run first and repeated last, and the repeat is the
# stage's only evidence that no arm leaked state into a later one.  It was
# recorded ``if last in res``, so when the repeat died -- the arm most likely
# to, since it runs last, after every wide table has churned the allocator --
# no ``R{q}_control`` key was written, nothing was logged, and the summary
# never looked.  A rung with no contamination check read exactly like a rung
# whose contamination check passed.

def test_a_control_that_ran_is_recorded_as_having_run(sweep, tmp_path,
                                                      monkeypatch):
    """The control: the happy path still records a passing control."""
    _stub(sweep, monkeypatch)
    b, res = _run(sweep, tmp_path)
    control = res[f"R{RUNG}_control"]
    assert control["ran"] is True and control["tensor_identical"] is True
    assert not any("CONTROL MISSING" in line for line in b.lines)


def test_a_failed_repeat_leaves_a_control_that_says_it_did_not_run(
        sweep, tmp_path, monkeypatch):
    """The absence is written down, and said out loud."""
    _stub(sweep, monkeypatch,
          fail={sweep.pair_arm_key(RUNG, 14, 1.0) + " [repeat]"})
    b, res = _run(sweep, tmp_path)
    # Pre-fix this key does not exist at all.
    control = res[f"R{RUNG}_control"]
    assert control["ran"] is False
    assert control["reason"] == sweep.CONTROL_REPEAT_MISSING
    log = "\n".join(b.lines)
    assert f"CONTROL MISSING at R{RUNG}" in log and "#96" in log
    # ... and the grid is untouched by it: the repeat is not a grid cell.
    assert res[f"R{RUNG}_grid_audit"]["complete"]


def test_a_failed_first_arm_names_the_baseline_not_the_repeat(
        sweep, tmp_path, monkeypatch):
    """Two ways to have no control, told apart."""
    _stub(sweep, monkeypatch, fail={sweep.pair_arm_key(RUNG, 14, 1.0)})
    _, res = _run(sweep, tmp_path)
    assert res[f"R{RUNG}_control"]["reason"] == sweep.CONTROL_BASELINE_MISSING


def test_summarise_pair_counts_controls_over_the_units_it_has(
        sweep, tmp_path, monkeypatch):
    """A count over the reporters is the #93 mistake; count over the units."""
    _stub(sweep, monkeypatch,
          fail={sweep.pair_arm_key(RUNG, 14, 1.0) + " [repeat]"})
    b, res = _run(sweep, tmp_path)
    b.doc = {"args": {"pair_bits": list(BITS), "pair_ratios": list(RATIOS),
                      "rungs": [RUNG]},
             "gate": "h", "units": {"unit": res}}
    b.lines.clear()
    sweep.summarise_pair(b)
    log = "\n".join(b.lines)
    assert f"CONTROL MISSING at R{RUNG}" in log
    assert "contamination controls: 0 of 1 unit(s)" in log
    assert b.doc["summary_grid_audit"][f"R{RUNG}"]["controls_passed"] == 0


def test_a_cell_missing_both_ways_says_both(sweep, tmp_path, monkeypatch):
    """Which to re-run depends on knowing both, so both are named."""
    _stub(sweep, monkeypatch,
          fail={sweep.pair_arm_key(RUNG, 16, 1.25),
                sweep.pair_arm_key(1216, 14, 1.0) + " [bytematch L=16]"})
    _, res = _run(sweep, tmp_path)
    why = res[f"R{RUNG}_grid_audit"]["missing"]["L=16 r=1.25"]
    assert sweep.CANDIDATE_MISSING in why and sweep.REFERENCE_MISSING in why
    # ... while its siblings, whose own encodes survived, name only the ref.
    assert res[f"R{RUNG}_grid_audit"]["missing"]["L=16 r=1"] == \
        sweep.REFERENCE_MISSING
