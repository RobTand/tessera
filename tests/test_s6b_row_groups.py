"""Issue #57: an S6b scale group must come from one output row.

``_pack_scales`` cut groups from the flattened tensor, so at ``cols % 32 != 0``
a group spanned the tail of one output row and the head of the next.  S6b's own
semantics are what make that matter: a group's two halves share one base
exponent with ``d <= 1`` -- one octave -- and two unrelated rows' magnitudes
have no reason to be within an octave of each other.

A per-row plane cannot tile such a width at all: ``cols % 32 == 16`` is an odd
number of halves per row, and the leftover half has no within-row partner.  So
those widths are refused instead of mis-grouped -- the smaller change the issue
names, matching the ``cols % half`` refusal ``kernel._require_column_groups``
already performs.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.encode import _pack_scales, _refit_scales  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402


def _row_banded(rows=4, cols=48):
    """Adjacent rows eight octaves apart: any straddling group's amax mixes
    magnitudes no one-octave pair can honestly cover."""
    w = torch.zeros(rows, cols)
    for r in range(rows):
        w[r] = (2.0 ** (8 * (r % 2))) * (0.5 + 0.01 * torch.arange(cols))
    return w


def test_pack_scales_refuses_a_width_that_is_not_a_whole_number_of_groups():
    w = _row_banded()
    assert w.shape[1] % 32 != 0
    with pytest.raises(GrammarError, match="whole number of 32"):
        _pack_scales(w, 32, 16)


def test_refit_scales_refuses_the_same_width():
    """The S6b refit groups halves the way the pack does; it must refuse the
    same widths rather than refitting straddled groups."""
    w = _row_banded()
    units = torch.ones_like(w)
    base = torch.zeros(6, dtype=torch.uint8)
    refine = torch.zeros(12, dtype=torch.uint8)
    effective = torch.ones(12)
    with pytest.raises(GrammarError, match="whole number of 32"):
        _refit_scales(w, units, 32, 16, base, refine, effective)
