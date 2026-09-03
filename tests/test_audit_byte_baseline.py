"""The byte-proof harness must cover every grid a reader can decode.

``experiments/audit_byte_baseline.py`` is how a change proves its byte claim
instead of asserting it: hash the encode matrix before and after, and
``--diff``.  A proof is only as wide as its matrix, and the matrix is a
hand-written list.  It covered E2M1, E2M1x2 and E4M3 and not ``BF16``, which
joined ``SERIALISABLE_GRIDS`` after the list was written -- so a change to
``BF16_WINDOW_BITS`` or ``BF16_CHANNEL_SIGMA`` moved real bytes and the harness
reported ``0 changed``.  That is worse than no proof, because it reads like
one.

These tests are the guard that makes the next hole noisy rather than silent.
Two holes are pinned here, because the matrix had both:

* A **grid** a reader will accept but the baseline never encodes (the BF16 hole
  above).
* A **condition** the baseline never reaches (issue #39).  The shape matrix is
  ``randn`` weights with no refit metric, no LDL factor and no completion, so it
  proves shape arithmetic and nothing else: reverting the CHANNEL refit's
  ``B > 0`` hold leaves all eighteen of its digests where they were.  The value
  matrix encodes a real weight slice against a real Hessian, and the same revert
  moves it.  ``test_the_value_matrix_catches_the_channel_refit_mutation`` runs
  that mutation rather than describing it, so the corpus cannot shrink back to
  a blind one while still printing a total.
"""
from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import pytest

import tessera.scale_channel
from tessera.alphabet import BF16_GRID, SERIALISABLE_GRIDS
from tessera.export import BF16_RECIPE, DEFAULT_LDLQ_SIGMA, wire_recipe
from tessera.manifest import ScalePlaneKind

HARNESS = Path(__file__).resolve().parents[1] / "experiments" / "audit_byte_baseline.py"

#: The hold whose removal the value matrix has to notice.  Quoted from the
#: source rather than reimplemented: ``_scale_channel_at`` compiles the real
#: module text, so a rewrite of this line fails the test loudly instead of
#: leaving a stale copy of it silently passing.
CHANNEL_HOLD = "valid = (A > 0) & (B > 0)"


def _load():
    """Import the harness without leaving its CPU pin on this process.

    The module does ``os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")`` at
    import so it can run beside a GPU measurement.  Importing it from a test
    session would pin the *whole* session to CPU and quietly skip every
    ``cuda`` test that follows, so the variable is restored to whatever it was.
    """
    had = "CUDA_VISIBLE_DEVICES" in os.environ
    prior = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        spec = importlib.util.spec_from_file_location("audit_byte_baseline", HARNESS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if had:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def test_the_encode_matrix_covers_every_serialisable_grid():
    module = _load()
    covered = {grid.name for _label, grid, _q, _r, _c in module._cases()}
    expected = {grid.name for grid in SERIALISABLE_GRIDS.values()}
    missing = expected - covered
    assert not missing, (
        f"{sorted(missing)} can be decoded by a reader but is never encoded by "
        f"the byte-proof matrix, so a change to its recipe constants moves real "
        f"bytes while the harness reports '0 changed'. Add a case to _cases()."
    )


def test_the_bf16_rows_reach_off_the_reference_rung():
    """Issue #48: the reach term resolves to the pinned wire's value at the
    reference rung, so a BF16 matrix that only encodes there is blind to it.
    Two rows are required: one whose spread differs from the reference (a
    change to ``_window_sigma_for`` moves it) and one at an exact half rate
    (a change to ``_reach_rate_for``'s rounding moves it)."""
    bf16 = [(label, q) for label, grid, q, _r, _c in _load()._cases()
            if grid.name == BF16_GRID.name]
    off_reference = [label for label, q in bf16
                     if wire_recipe(BF16_GRID, q).window_sigma != BF16_RECIPE.window_sigma]
    assert off_reference, (
        "every BF16 case sits on the reach term's reference rung, where the "
        "recipe resolves to the pinned wire's spread -- a change to "
        "_window_sigma_for moves every other rung and this matrix reports "
        "'0 changed'. Add a BF16 case away from BF16_REACH_REFERENCE_RATE.")
    half_rate = [label for label, q in bf16 if (q * BF16_GRID.arity) % 256 == 128]
    assert half_rate, (
        "no BF16 case sits at an exact half rate, so the rounding rule of "
        "_reach_rate_for (half-up, not banker's) moves no digest here.")


def test_every_case_names_a_grid_a_reader_would_accept():
    """The converse: a case on a grid no reader accepts proves nothing."""
    module = _load()
    accepted = {grid.name for grid in SERIALISABLE_GRIDS.values()}
    for label, grid, _q, _r, _c in module._cases():
        assert grid.name in accepted, (
            f"case {label!r} encodes grid {grid.name}, which is not in "
            f"SERIALISABLE_GRIDS -- its digest is not a byte anyone can decode"
        )


def _scale_channel_at(hold: str) -> types.ModuleType:
    """A second ``tessera.scale_channel`` with the CHANNEL refit's hold rewritten.

    Compiled from the installed module's own text, not hand-copied: a copy of a
    fifty-line numerical function is a copy that drifts, and a drifted copy
    would keep passing while testing something else.  Replacing the exact
    ``CHANNEL_HOLD`` line also means a rewrite of that line breaks this test
    rather than silently turning the mutation into a no-op.
    """
    source = Path(tessera.scale_channel.__file__).read_text()
    assert source.count(CHANNEL_HOLD) == 1, (
        f"{CHANNEL_HOLD!r} appears {source.count(CHANNEL_HOLD)} times in "
        f"scale_channel.py; this test mutates that one line and needs to know "
        f"which one it is"
    )
    module = types.ModuleType("tessera._scale_channel_under_test")
    module.__file__ = tessera.scale_channel.__file__
    module.__package__ = "tessera"          # the file's relative imports need it
    exec(compile(source.replace(CHANNEL_HOLD, hold), module.__file__, "exec"),
         module.__dict__)
    return module


def _channel_value_case(module):
    """The cheapest value case on the CHANNEL plane -- what the mutation moves."""
    for case in module._value_cases():
        if wire_recipe(case.grid, case.q256).scale_plane is ScalePlaneKind.CHANNEL:
            if "ldlq" not in case.label:    # LDLQ's block loop costs ~3x
                return case
    raise AssertionError("no CHANNEL-plane value case to mutate")


def test_the_value_matrix_catches_the_channel_refit_mutation(monkeypatch):
    """Remove the ``B > 0`` hold; a value digest must move.

    Three arms, and each one carries its own weight.  The **kept** arm is the
    real function.  The **copy** arm is the recompiled source unchanged, and it
    must reproduce the kept digest exactly -- that is what makes the third arm
    trustworthy, because a copy that does not reproduce the original is not
    testing the original.  The **mutant** arm removes the hold, and its digest
    must differ: without that, the corpus is back to proving shape arithmetic
    only.
    """
    module = _load()
    payload = module.load_value_slice()
    case = _channel_value_case(module)

    kept = module.encode_value_case(case, payload)

    copy = _scale_channel_at(CHANNEL_HOLD)
    # ``encode_unit`` imports ``refit_channel_scale`` inside its refit loop, so
    # the module attribute is what a call resolves.
    monkeypatch.setattr(tessera.scale_channel, "refit_channel_scale",
                        copy.refit_channel_scale)
    assert module.encode_value_case(case, payload) == kept, (
        "the recompiled copy of scale_channel does not reproduce the real "
        "function's bytes, so the mutation arm below proves nothing"
    )

    mutant = _scale_channel_at("valid = (A > 0)")
    monkeypatch.setattr(tessera.scale_channel, "refit_channel_scale",
                        mutant.refit_channel_scale)
    assert module.encode_value_case(case, payload) != kept, (
        f"value case {case.label!r} encodes the same bytes with and without the "
        f"CHANNEL refit's {CHANNEL_HOLD!r} hold, so the byte proof would report "
        f"'0 changed' for a change that collapses a row's scale to 2^-14. The "
        f"case has stopped reaching the condition -- check the slice, the rung "
        f"and DEFAULT_REFIT_OBJECTIVE, not this test"
    )


def test_the_shape_matrix_alone_is_blind_to_that_mutation(monkeypatch):
    """The measurement behind issue #39, kept in the tree rather than in prose.

    This is not a property anyone wants to preserve -- it is the reason the
    value matrix exists.  Pinning it keeps the claim honest: if a later change
    makes the ``randn`` cases reach the hold after all, this test fails and the
    docstrings that say they cannot must be rewritten.
    """
    module = _load()
    label, grid, q256, rows, cols = next(
        c for c in module._cases() if c[0] == "e4m3-1024-256c")   # a CHANNEL case
    kept = module.encode_shape_case(label, grid, q256, rows, cols, "scale")

    mutant = _scale_channel_at("valid = (A > 0)")
    monkeypatch.setattr(tessera.scale_channel, "refit_channel_scale",
                        mutant.refit_channel_scale)
    assert module.encode_shape_case(label, grid, q256, rows, cols, "scale") == kept, (
        "a randn shape case now moves under the CHANNEL hold mutation. That is "
        "better coverage, not a regression -- but the harness docstring and "
        "issue #39 both say it cannot, so fix the prose"
    )


def test_the_value_matrix_reaches_every_condition_the_shape_matrix_cannot():
    """The cheap leg: the arms are still in the list, without encoding anything.

    Each entry is a condition the shape matrix leaves unreachable, and each was
    unreachable when the CHANNEL fixes were proved against ``0 changed of 36``.
    A case removed from ``_value_cases`` fails here in a second, rather than
    fifteen minutes later in the mutation test.
    """
    module = _load()
    cases = module._value_cases()
    planes = {c.label: wire_recipe(c.grid, c.q256).scale_plane for c in cases}

    channel_grids = {
        c.grid.name for c in cases if planes[c.label] is ScalePlaneKind.CHANNEL
    }
    # Both grids whose shipping recipe is the CHANNEL plane, so a digest that
    # moves on one and not the other says which half changed.
    assert {"E4M3", "BF16"} <= channel_grids, (
        f"the CHANNEL plane is encoded on {sorted(channel_grids)}; the exact "
        f"full-H refit is that plane's shipping objective and every grid that "
        f"ships it needs a case"
    )
    assert any(planes[c.label] is ScalePlaneKind.LUT for c in cases), (
        "no value case on the LUT plane -- its metric refit is a different "
        "function than CHANNEL's and takes a 1-D metric, not the full H"
    )
    # ``.get(..., DEFAULT_LDLQ_SIGMA)``: a case that names no ``ldlq_sigma`` is
    # taking the exporter's default, which is the arm worth having.  Reading the
    # constant rather than the literal keeps this honest if the default moves to
    # ``None`` -- then no case runs LDLQ and this must fail, not pass on a key
    # that happens to be absent.
    assert any(c.source.get("ldlq_sigma", DEFAULT_LDLQ_SIGMA) is not None
               for c in cases), (
        "no value case runs LDLQ, so a change to block_ldl or to the block "
        "schedule moves real exported bytes and this harness reports 0 changed"
    )
    assert any(c.source.get("refit_reach_floor") for c in cases), (
        "no value case sets refit_reach_floor, which is land_at_least's only "
        "caller"
    )
    assert any(c.encode.get("completion") for c in cases), (
        "no value case spends the completion axis, so the completion argmin "
        "has one descendant and cannot choose wrongly"
    )


def test_the_value_slice_refuses_rather_than_skipping(monkeypatch, tmp_path):
    """A missing fixture stops the run; it never shrinks the corpus quietly."""
    module = _load()
    monkeypatch.setattr(module, "VALUE_SLICE", tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="make_audit_value_slice"):
        module.load_value_slice()
