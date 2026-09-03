"""The LDLQ block size is priced from the Hessian, not chosen by a sweep.

tessera#60.  ``block_penalty`` is a closed form, so these are identities and
limits, not tolerances against a fitted curve.  The one measured anchor is the
Qwen/GLM validation recorded in ``block_penalty``'s docstring; it is not
re-run here (it needs the capture), and nothing below stands on it.
"""
import inspect
import math

import pytest
import torch

from tessera.compensate import (block_ldl, block_penalty, choose_ldl_block,
                                regularize_hessian)
from tessera.errors import GrammarError


def _correlated(n: int, rho: float, seed: int = 0) -> torch.Tensor:
    """A positive-definite H whose off-diagonal mass is set by ``rho``."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(4 * n, n, generator=g)
    A = A + rho * A[:, :1]                       # a shared component per column
    return (A.T @ A) / A.shape[0]


def test_block_one_skips_nothing_so_the_penalty_is_exactly_one():
    H = regularize_hessian(_correlated(64, 0.8))
    assert block_penalty(H, 1) == pytest.approx(1.0, abs=1e-12)


def test_the_whole_matrix_as_one_block_prices_the_uncompensated_encode():
    """At block == n nothing is compensated, so the loss is tr(H), not tr(D)."""
    H = regularize_hessian(_correlated(64, 0.8))
    C = torch.linalg.cholesky(H)
    trD = float((torch.diagonal(C) ** 2).sum())
    assert block_penalty(H, 64) == pytest.approx(float(torch.diagonal(H).sum()) / trD,
                                                 rel=1e-5)


def test_the_penalty_is_monotone_in_the_block_size():
    H = regularize_hessian(_correlated(128, 0.6))
    vals = [block_penalty(H, b) for b in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert vals == sorted(vals), vals


def test_an_uncorrelated_hessian_costs_almost_nothing_and_a_correlated_one_costs():
    """The formula is what separates the two populations of #60."""
    flat = regularize_hessian(_correlated(128, 0.0, seed=1))
    corr = regularize_hessian(_correlated(128, 3.0, seed=1))
    # the penalty is a ratio, so what scales with correlation is its excess
    assert (block_penalty(corr, 32) - 1) > 20 * (block_penalty(flat, 32) - 1)


# --------------------------------------------------------------------------
# The floor is the caller's path's, not the method's (tessera#95).  Every test
# below states a floor as the thing under test rather than inheriting one, so
# what is pinned is "the caller says" and not any particular plane's number.
# --------------------------------------------------------------------------


def test_the_floor_has_no_default_because_the_two_callers_disagree_about_it():
    """``compensated_targets`` stitches independently-encoded slices, so it
    floors at the encoder's scale group and rotation block;
    ``encode.encode_unit(ldl=...)`` reads one plane per pass across every
    block, so it floors at 1.  A default is one of those two answers handed
    silently to the other caller: floored at the stitching path's 16, this
    chooser could not return the b8 and b4 arms its own commit validated
    ``block_penalty`` against, and nothing raised.  So there is no default,
    and omitting it is a TypeError rather than a quiet number."""
    assert (inspect.signature(choose_ldl_block).parameters["floor"].default
            is inspect.Parameter.empty)
    with pytest.raises(TypeError):
        choose_ldl_block(regularize_hessian(_correlated(32, 0.5)), max_penalty=2.0)


@pytest.mark.parametrize("floor", [1, 2, 8, 16, 32])
def test_the_block_the_chooser_returns_honours_whatever_floor_it_was_given(floor):
    H = regularize_hessian(_correlated(128, 1.0))
    b = choose_ldl_block(H, max_penalty=block_penalty(H, floor), floor=floor)
    assert b >= floor and b % floor == 0 and 128 % b == 0
    block_ldl(H, b)                              # raises if the block is illegal


@pytest.mark.parametrize("target", [2, 4, 8])
def test_the_floor_alone_decides_whether_a_small_block_is_reachable(target):
    """The rule the wrong default broke, stated without a plane-to-number table.

    One Hessian, one budget, two callers.  A path whose floor is above the
    block that budget wants cannot serve it and says so; a path with no
    scale-group constraint reaches exactly that block.  Nothing about the
    Hessian or the budget changes between the two -- only whose floor it is.
    """
    H = regularize_hessian(_correlated(128, 2.0))
    budget = block_penalty(H, target)
    assert budget < block_penalty(H, 2 * target)      # the premise of the test
    with pytest.raises(GrammarError, match="no legal block meets a budget"):
        choose_ldl_block(H, max_penalty=budget, floor=2 * target)
    assert choose_ldl_block(H, max_penalty=budget, floor=1) == target


@pytest.mark.parametrize("floor", [1, 16])
def test_a_tighter_budget_never_returns_a_larger_block(floor):
    H = regularize_hessian(_correlated(256, 1.5))
    loose = choose_ldl_block(H, max_penalty=block_penalty(H, 256), floor=floor)
    tight = choose_ldl_block(H, max_penalty=block_penalty(H, floor), floor=floor)
    assert tight <= loose
    # the two ends, not a shrug: the whole axis, and the caller's own floor
    assert loose == 256 and tight == floor


@pytest.mark.parametrize("floor", [1, 16])
def test_the_chosen_block_is_within_the_budget_and_the_next_one_up_is_not(floor):
    H = regularize_hessian(_correlated(256, 1.5))
    budget = block_penalty(H, 64)                 # a budget exactly one rung buys
    b = choose_ldl_block(H, max_penalty=budget, floor=floor)
    assert b == 64
    assert block_penalty(H, b) <= budget < block_penalty(H, 2 * b)


@pytest.mark.parametrize("floor", [2, 16])
def test_a_budget_the_floor_cannot_meet_is_refused_rather_than_served(floor):
    """Any floor above 1 can be asked for more than it can give; a floor of 1
    cannot, since a block of one skips nothing and costs exactly 1.0."""
    H = regularize_hessian(_correlated(256, 3.0))
    floor_cost = block_penalty(H, floor)
    assert floor_cost > 1.0001                      # the premise of the test
    with pytest.raises(GrammarError, match="no legal block meets a budget"):
        choose_ldl_block(H, max_penalty=1.0, floor=floor)
    # and the message names what the floor does cost, so a budget can be set
    with pytest.raises(GrammarError, match=f"{floor_cost:.6f}"):
        choose_ldl_block(H, max_penalty=1.0, floor=floor)
    # a floor of 1 is never the refused one: it is exactly full feedback
    assert block_penalty(H, 1) == pytest.approx(1.0, abs=1e-12)
    assert choose_ldl_block(H, max_penalty=1.0, floor=1) == 1


def test_a_budget_below_one_is_refused_because_the_ratio_cannot_be_below_one():
    H = regularize_hessian(_correlated(32, 0.5))
    with pytest.raises(GrammarError, match="at least 1.0"):
        choose_ldl_block(H, max_penalty=0.9, floor=1)


def test_a_block_that_does_not_divide_the_input_axis_is_refused():
    H = regularize_hessian(_correlated(48, 0.5))
    with pytest.raises(GrammarError, match="not a multiple"):
        block_penalty(H, 32)


def test_a_non_square_hessian_is_refused():
    with pytest.raises(GrammarError, match="must be square"):
        block_penalty(torch.eye(8)[:, :4], 2)


def test_the_penalty_matches_a_direct_sum_over_the_skipped_pairs():
    """Independent of the implementation's blocking, from the definition."""
    n, b = 48, 8
    H = regularize_hessian(_correlated(n, 1.2))
    C = torch.linalg.cholesky(H)
    d = torch.diagonal(C)
    L, D = C / d.unsqueeze(0), d ** 2
    want = sum(float(L[i, j] ** 2 * D[j])
               for s in range(0, n, b)
               for i in range(s, s + b) for j in range(s, i))
    assert block_penalty(H, b) == pytest.approx(1.0 + want / float(D.sum()), rel=1e-5)
