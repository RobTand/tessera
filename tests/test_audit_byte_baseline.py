"""The byte-proof harness must cover every grid a reader can decode.

``experiments/audit_byte_baseline.py`` is how a change proves its byte claim
instead of asserting it: hash the encode matrix before and after, and
``--diff``.  A proof is only as wide as its matrix, and the matrix is a
hand-written list.  It covered E2M1, E2M1x2 and E4M3 and not ``BF16``, which
joined ``SERIALISABLE_GRIDS`` after the list was written -- so a change to
``BF16_WINDOW_BITS`` or ``BF16_CHANNEL_SIGMA`` moved real bytes and the harness
reported ``0 changed``.  That is worse than no proof, because it reads like
one.

This test is the guard that makes the next grid noisy rather than silent: a
grid that a reader will accept but the baseline never encodes fails here.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from tessera.alphabet import SERIALISABLE_GRIDS

HARNESS = Path(__file__).resolve().parents[1] / "experiments" / "audit_byte_baseline.py"


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
