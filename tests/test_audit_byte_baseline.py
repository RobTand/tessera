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
from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.export import DEFAULT_LDLQ_SIGMA, wire_recipe
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


def test_the_value_matrix_catches_the_initial_reach_floor(monkeypatch):
    """Removing #87's upward landing must move a baseline digest.

    The condition is selected from the recipe-owning data: a CHANNEL value
    case whose activation source enables LDLQ.  The mutation disables only the
    one-ulp correction after the ordinary round-to-nearest landing.  If the
    digest holds, the byte audit no longer exercises the change it is meant to
    price.
    """
    module = _load()
    payload = module.load_value_slice()
    case = next(
        c for c in module._value_cases()
        if wire_recipe(c.grid, c.q256).scale_plane is ScalePlaneKind.CHANNEL
        and c.source.get("ldlq_sigma", DEFAULT_LDLQ_SIGMA) is not None
    )
    kept = module.encode_value_case(case, payload)

    def no_bump(stored, effective, _floor, _global_scale, where=None):
        return stored, effective

    monkeypatch.setattr(
        tessera.scale_channel, "_bump_below_floor", no_bump, raising=False
    )
    assert module.encode_value_case(case, payload) != kept, (
        f"value case {case.label!r} keeps the same bytes when the initial "
        "reach floor is landed to nearest, so the byte audit no longer "
        "covers issue #87's condition"
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
        "no value case sets refit_reach_floor, so the refit caller of "
        "land_at_least is uncovered"
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



# ------------------------------------------------------------ plane coverage
#
# The third hole, and it is the first one again in another coordinate: the
# matrix covered every *grid*, but only the planes ``wire_recipe`` selects and
# only on whole units.  Three of the ten planes a reader will read were written
# by no row at all -- SCALE_BASE, DIAG_SU and INITIAL_STATE -- so a change to
# the S6b base packing, to ``diagonals.fit_diagonals``, or to anything schema
# minor 4 added for shards moved real bytes while the harness reported
# "0 changed" (issue #143).  Both guards derive their expected set from the
# module that owns it, so the next plane is noisy rather than silent.


def _effective_scale_plane(grid, q256, overrides):
    """The plane a case actually encodes on: its override, else its recipe."""
    from tessera.export import tcq_cap_q256

    if q256 is None:                      # the release rows spell the cap this way
        q256 = tcq_cap_q256(grid)
    override = overrides.get("scale_plane")
    # ``is None``, never ``or``: ``ScalePlaneKind.S6B`` is 0, and an ``or``
    # here would silently fall through to the recipe for the one plane this
    # guard exists to catch.
    return wire_recipe(grid, q256).scale_plane if override is None else override


def test_the_matrix_encodes_every_scale_plane_a_reader_accepts():
    """The grid guard's sibling on the other axis of the same blind spot.

    ``unit_artifact._read_scale_planes`` dispatches on ``ScalePlaneKind``, so
    every member of that enum is a plane some artifact can carry and some
    reader will decode -- whether or not ``wire_recipe`` ever selects it.  S6b
    is exactly that case: ``encode_linear_planes(scale_plane=...)`` is a
    caller-facing override, ``_read_scale_planes`` accepts what it writes, and
    no row encoded one.
    """
    module = _load()
    covered = {
        _effective_scale_plane(grid, q256, {})
        for _label, grid, q256, _r, _c in module._cases()
    }
    covered |= {
        _effective_scale_plane(case.grid, case.q256, {})
        for case in module._value_cases()
    }
    covered |= {
        _effective_scale_plane(grid, q256, {})
        for _label, grid, q256, _r, _c, _rel, _cut in module._release_cases()
    }
    covered |= {
        _effective_scale_plane(case.grid, case.q256, case.encode)
        for case in module._layout_cases()
    }
    missing = set(ScalePlaneKind) - covered
    assert not missing, (
        f"{sorted(p.name for p in missing)} is a scale plane a reader decodes "
        f"-- _read_scale_planes dispatches on ScalePlaneKind -- and no case in "
        f"the byte-proof matrix encodes it, so a change to its packing or its "
        f"refit moves real bytes while the harness reports '0 changed'."
    )


@pytest.fixture(scope="module")
def layout_blobs():
    """Every layout case, encoded once for the two tests that read them."""
    module = _load()
    return module, {
        case.label: module.encode_layout_case(case)
        for case in module._layout_cases()
    }


def test_the_layout_matrix_writes_every_plane_the_others_cannot(layout_blobs):
    """Every ``PlaneKind`` on the wire is non-empty in some row's real bytes.

    The expected set is ``planes.SHARD_PLANE_ORDER`` -- the full wire order,
    the one with INITIAL_STATE in it -- read off the module that owns the order
    rather than restated here, so a plane a future schema minor adds fails this
    until a row writes it.

    Two planes are subtracted, and neither is a free pass: RELEASE belongs to
    the release matrix and COMPLETION to the value matrix's completion arm, and
    the subtraction is *conditional on those rows still existing*, asserted
    first.  Everything else the layout rows must write themselves.

    Occupancy is measured, not declared: ``written_planes`` reads each
    artifact's own ``plane_order`` against its terminal's ``plane_elements``,
    which is the pair every reader indexes.
    """
    module, blobs = layout_blobs
    from tessera.planes import PlaneKind, SHARD_PLANE_ORDER

    assert any(rel for *_head, rel, _cut in module._release_cases()), (
        "no release row releases a position, so the RELEASE plane is written "
        "by nothing and this test may not subtract it"
    )
    assert any(case.encode.get("completion") for case in module._value_cases()), (
        "no value case spends the completion axis, so the COMPLETION plane is "
        "written by nothing and this test may not subtract it"
    )
    elsewhere = {PlaneKind.RELEASE, PlaneKind.COMPLETION}

    written = {}                                   # plane -> the row that wrote it
    for label, parts in blobs.items():
        for key, payload in parts.items():
            if key == "state":                     # a raw tensor, not an artifact
                continue
            for kind in module.written_planes(payload):
                written.setdefault(kind, f"{label}/{key}")

    missing = [k for k in SHARD_PLANE_ORDER if k not in written and k not in elsewhere]
    assert not missing, (
        f"{[k.name for k in missing]} is written by no row of the byte-proof "
        f"matrix, so a change to how it is built or packed moves real bytes "
        f"and the harness reports '0 changed'. Written here: "
        f"{ {k.name: v for k, v in sorted(written.items())} }"
    )


def test_a_shard_row_decodes_to_its_parent_sliced(layout_blobs):
    """The served half of the shard rows: the bytes mean the parent's window.

    A digest proves a shard's bytes are stable, not that they are *right*.
    This is the claim schema minor 4 exists to make -- a rank's shard decodes
    bit for bit to its window of the artifact the exporter wrote -- and here it
    runs on CPU.  ``tests/test_slice_unit.py`` makes the same claim under
    ``skipif(not torch.cuda.is_available(), reason="the encoder is a CUDA
    path")`` (``:161-163``), and the encoder is not a CUDA path: this harness
    and ``encoder_identity``'s CPU-by-construction fixtures both encode with no
    device at all.
    """
    import torch
    from tessera.unit_artifact import read_unit_artifact

    module, blobs = layout_blobs
    cuts = {case.label: case.cut for case in module._layout_cases() if case.cut}
    assert cuts, "no layout case cuts a shard, so INITIAL_STATE is unwritten"
    for label, (rows, cols) in cuts.items():
        parts = blobs[label]
        whole = read_unit_artifact(parts["parent"])
        shard = read_unit_artifact(parts["bytes"])
        assert torch.equal(shard, whole[rows[0]:rows[1], cols[0]:cols[1]]), (
            f"layout row {label!r} decodes to something other than its parent's "
            f"rows {rows} x columns {cols}: its bytes are stable and wrong, "
            f"which is what a digest alone cannot tell you"
        )
