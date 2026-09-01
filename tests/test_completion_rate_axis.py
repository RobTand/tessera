"""The COMPLETION plane must cost what the encoder spent, not what the rate allows.

``encode_unit`` truncates to ``level = min(completion, cap - R)``.  For most of
this format's life the serialiser sized and packed that plane from ``cap - R``
alone, so a unit encoded at ``completion=0`` wrote a full-width plane of zeros
and paid for every bit of it.  The consequence was not a rounding error, it was
a lie about the format's rate: ``sum(R) + sum(cap - R) = columns * cap`` is
**constant**, so as the body rate fell the all-zero completion plane grew to
match and every rung of a family serialised to the same size.  The rate ladder
existed in the encoder and was invisible on the wire.

These tests fix the property in both directions: an artifact pays for the depth
it used, and a reader recovers that depth from the artifact itself rather than
assuming the ceiling.  The last one is the guard that matters most -- the depth
is not a new wire field, it is solved from the COMPLETION plane's already-
recorded element count, so a stale reader assumption would mis-slice the plane
silently.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E2M1_GRID, tuple_grid  # noqa: E402
from tessera.decode import reconstruct_unit  # noqa: E402
from tessera.encode import encode_unit  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402
from tessera.export import _plan_for, encode_linear  # noqa: E402
from tessera.grammar import (  # noqa: E402
    completion_capacity,
    completion_limit_from_elements,
    completion_widths,
)
from tessera.manifest import RotationState  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

CODE = ConvCode(memory=6)
K2 = tuple_grid(E2M1_GRID, 2, partition="coset")
FAMILIES = [("E2M1_K1", E2M1_GRID), ("E2M1_K2", K2)]


def _weight(rows=64, cols=512, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(rows, cols) * 0.05).to(torch.bfloat16)


def _build(grid, q256, completion, name="u"):
    return encode_linear(
        _weight(), grid=grid, q256=q256, name=name, code=CODE,
        rotation=RotationState.NONE, with_diagonals=False,
        completion=completion, verify=True,
    )


# --- the widths themselves -------------------------------------------------

def test_capacity_is_a_ceiling_not_the_spend():
    rates = (3, 2, 1)                      # rate 0 is outside the shaped domain
    assert completion_widths(rates, 3, None) == (0, 1, 2)
    assert completion_widths(rates, 3, 0) == (0, 0, 0)
    assert completion_widths(rates, 3, 1) == (0, 1, 1)
    assert completion_widths(rates, 3, 99) == completion_widths(rates, 3, None)


def test_a_full_depth_unit_is_unchanged_by_the_limit_machinery():
    """The regression guard for every artifact already written.

    At the top rung every column sits at ``R == cap``, so capacity is zero and
    the plane is empty under both the old sizing and the new one.  That is why
    this fix cannot move a shipped byte.
    """
    for _, grid in FAMILIES:
        top = 256 * grid.rate_cap // grid.arity
        rates, _ = _plan_for(grid, top, 512)
        assert set(completion_capacity(r, grid.rate_cap) for r in rates) == {0}
        assert _build(grid, top, 0).blob == _build(grid, top, None).blob


# --- the size axis ---------------------------------------------------------

@pytest.mark.parametrize("name,grid", FAMILIES)
def test_deeper_completion_costs_more_bytes(name, grid):
    """Below the top rung the plane has room, so the depth must show up in the
    size.  Before the fix these were all equal."""
    q256 = 256 * grid.rate_cap // grid.arity // 2      # mid-ladder, capacity > 0
    sizes = [_build(grid, q256, c, name).exact_bytes for c in (0, 1, 2, None)]
    assert sizes == sorted(sizes), sizes
    assert sizes[0] < sizes[-1], f"{name}: completion is still free at q{q256}"


@pytest.mark.parametrize("name,grid", FAMILIES)
def test_the_body_rate_moves_the_size_at_fixed_completion(name, grid):
    """The bug's signature: with completion written at full width, sum(R) +
    sum(cap - R) is constant and the ladder is flat.  At a fixed spend it must
    be strictly increasing in the rung."""
    cap_q = 256 * grid.rate_cap // grid.arity
    floor_q = 256 // grid.arity                 # every column at rate 1
    rungs = sorted({floor_q, cap_q // 2, (3 * cap_q) // 4, cap_q})
    sizes = [_build(grid, q, 0, name).exact_bytes for q in rungs]
    assert sizes == sorted(sizes), dict(zip(rungs, sizes))
    assert len(set(sizes)) == len(sizes), f"{name}: flat ladder {sizes}"


# --- the reader ------------------------------------------------------------

@pytest.mark.parametrize("name,grid", FAMILIES)
@pytest.mark.parametrize("completion", [0, 1, None])
def test_the_reader_recovers_the_depth_it_was_written_at(name, grid, completion):
    q256 = 256 * grid.rate_cap // grid.arity // 2
    w = _weight()
    rates, forests = _plan_for(grid, q256, w.shape[1])
    unit = encode_unit(w.float(), forests, rates, CODE,
                       rotation=RotationState.NONE, completion=completion,
                       group=32, half=16)
    reference = reconstruct_unit(unit, forests, CODE)
    blob = _build(grid, q256, completion, name).blob      # verify=True already
    assert torch.equal(read_unit_artifact(blob, device=w.device), reference)


def test_the_depth_is_solved_from_the_recorded_element_count():
    """A limit at or above the deepest column is the same artifact as no limit,
    and the reader reports it as ``None`` -- there is nothing on the wire to
    distinguish them, and inventing a distinction would make two byte-identical
    artifacts compare unequal."""
    rates = (3, 2, 1)                              # capacities 0, 1, 2
    for limit in (0, 1):
        elements = sum(completion_widths(rates, 3, limit)) * 8
        assert completion_limit_from_elements(elements, rates, 8, 3) == limit
    for saturating in (2, None):
        elements = sum(completion_widths(rates, 3, saturating)) * 8
        assert completion_limit_from_elements(elements, rates, 8, 3) is None


def test_an_impossible_element_count_refuses_rather_than_guessing():
    with pytest.raises(GrammarError):
        completion_limit_from_elements(7, (3, 2, 1), 8, 3)


# --- the depth must survive the round trip, not just the plane -------------

@pytest.mark.parametrize("name,grid", FAMILIES)
def test_error_falls_monotonically_with_the_completion_depth(name, grid):
    """A completion bit refines within the anchor's own descendant tree, and
    the Viterbi metric at depth ``c+1`` scores every path by its best depth-
    ``c+1`` descendant.  The depth-``c`` optimum is therefore feasible at
    ``c+1``, so squared error is non-increasing in the depth -- an encoder
    property, checkable without reference to any baseline.

    It did not hold.  ``decode_codes_mixed`` defaulted to the full capacity
    whatever depth the encoder spent, and a level-``c`` index read at a deeper
    level addresses a different subtree, so deeper completion decoded to
    *worse* weights.  ``encode_linear``'s round-trip check could not see it:
    it compares two decodes that shared the assumption.
    """
    torch.manual_seed(1)
    w = (torch.randn(64, 256) * 0.05).float()
    q256 = 256 * grid.rate_cap // grid.arity // 2         # mid-ladder
    rates, forests = _plan_for(grid, q256, w.shape[1])
    errors = []
    for completion in (0, 1, 2, None):
        unit = encode_unit(w, forests, rates, CODE, rotation=RotationState.NONE,
                           completion=completion, group=32, half=16)
        recon = reconstruct_unit(unit, forests, CODE)
        errors.append(float(((recon - w) ** 2).sum()))
    assert errors == sorted(errors, reverse=True), dict(
        zip(("c0", "c1", "c2", "cF"), errors))


@pytest.mark.parametrize("name,grid", FAMILIES)
def test_bytes_alone_decode_to_the_encoders_reconstruction_at_every_depth(name, grid):
    """The reader gets the depth from the artifact, never from the caller."""
    q256 = 256 * grid.rate_cap // grid.arity // 2
    w = _weight()
    rates, forests = _plan_for(grid, q256, w.shape[1])
    for completion in (0, 1, 2, None):
        unit = encode_unit(w.float(), forests, rates, CODE,
                           rotation=RotationState.NONE, completion=completion,
                           group=32, half=16)
        blob = _build(grid, q256, completion, name).blob
        assert torch.equal(read_unit_artifact(blob, device=w.device),
                           reconstruct_unit(unit, forests, CODE))
