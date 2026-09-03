"""The LDLQ block size is priced from the Hessian, not chosen by a sweep.

tessera#60.  ``block_penalty`` is a closed form, so these are identities and
limits, not tolerances against a fitted curve.  The one measured anchor is the
Qwen/GLM validation recorded in ``block_penalty``'s docstring; it is not
re-run here (it needs the capture), and nothing below stands on it.
"""
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


def test_the_block_the_chooser_returns_is_one_block_ldl_accepts():
    H = regularize_hessian(_correlated(128, 1.0))
    b = choose_ldl_block(H, max_penalty=block_penalty(H, 32), floor=16)
    assert 128 % b == 0 and b >= 16
    block_ldl(H, b)                              # raises if the block is illegal


def test_a_tighter_budget_never_returns_a_larger_block():
    H = regularize_hessian(_correlated(256, 1.5))
    loose = choose_ldl_block(H, max_penalty=block_penalty(H, 256), floor=16)
    tight = choose_ldl_block(H, max_penalty=block_penalty(H, 16), floor=16)
    assert tight <= loose
    assert loose == 256 and tight == 16           # the two ends, not a shrug


def test_the_chosen_block_is_within_the_budget_and_the_next_one_up_is_not():
    H = regularize_hessian(_correlated(256, 1.5))
    budget = block_penalty(H, 64)                 # a budget exactly one rung buys
    b = choose_ldl_block(H, max_penalty=budget, floor=16)
    assert b == 64
    assert block_penalty(H, b) <= budget < block_penalty(H, 2 * b)


def test_a_budget_the_floor_cannot_meet_is_refused_rather_than_served():
    H = regularize_hessian(_correlated(256, 3.0))
    floor_cost = block_penalty(H, 16)
    assert floor_cost > 1.0001                      # the premise of the test
    with pytest.raises(GrammarError, match="no legal block meets a budget"):
        choose_ldl_block(H, max_penalty=1.0, floor=16)
    # and the message names what the floor does cost, so a budget can be set
    with pytest.raises(GrammarError, match=f"{floor_cost:.6f}"):
        choose_ldl_block(H, max_penalty=1.0, floor=16)


def test_a_budget_below_one_is_refused_because_the_ratio_cannot_be_below_one():
    H = regularize_hessian(_correlated(32, 0.5))
    with pytest.raises(GrammarError, match="at least 1.0"):
        choose_ldl_block(H, max_penalty=0.9)


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
