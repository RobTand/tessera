"""The paired trellis step returns the reference's bytes, not close ones.

Every assertion here is bit-level: states compared exactly, ``sse`` compared
as the identical float, and the intermediate fronts compared through their
int32 bit patterns so a sign of zero or a NaN payload cannot pass as equal.

Three families, because the pair has three ways to be wrong and only the
first is caught by random data:

  * random targets -- the ordinary case;
  * a table with duplicated rows, which makes two predecessors carry the
    *identical* float, so the run pins the tie rule rather than the arithmetic;
  * step 0, where the front is ``0`` at state 0 and ``+inf`` everywhere else,
    so nearly every class minimum is a scan over ties at ``inf``.  This one
    needs no construction: it is the first step of every run below, and the
    dedicated test reads the front after it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ``conftest`` puts ``src`` and ``tests`` on the path; the checkout root, where
# ``experiments`` lives, is only there when pytest was started as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessera.encode import viterbi_window  # noqa: E402
from experiments.window_viterbi_two_step import (  # noqa: E402
    branch_costs,
    pair_is_applicable,
    step_best_form,
    step_pair_nested,
    step_sequential,
    viterbi_window_best_form,
    viterbi_window_paired,
)

# (window_bits, rate, arity, rows, cols); rows must be a multiple of arity.
SHAPES = [
    (8, 2, 2, 12, 7),      # even steps, small
    (8, 2, 2, 10, 33),     # steps not a multiple of the chunk
    (8, 4, 2, 8, 9),       # L = 2R exactly, the applicability edge
    (9, 3, 3, 15, 5),      # odd steps -> a trailing sequential step
    (10, 2, 4, 16, 11),    # arity 4
    (12, 3, 2, 14, 6),     # a wider window
]


def _bits(t):
    return t.contiguous().view(torch.int32)


def _case(window_bits, rate, arity, rows, cols, seed, table_mode="random"):
    g = torch.Generator().manual_seed(seed)
    size = 1 << window_bits
    targets = torch.randn(rows, cols, generator=g)
    vectors = torch.randn(size, arity, generator=g)
    if table_mode == "duplicated":
        # Halve the distinct rows so predecessor pairs reconstruct identically
        # and their branch costs are the same float, not merely a close one.
        vectors = vectors[: size // 2].repeat(2, 1).contiguous()
    weights = torch.rand(rows, cols, generator=g) + 0.5
    return targets, vectors, weights


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
@pytest.mark.parametrize("table_mode", ["random", "duplicated"])
@pytest.mark.parametrize("weighted", [False, True])
def test_the_pair_returns_the_reference_bytes(window_bits, rate, arity, rows,
                                              cols, table_mode, weighted):
    targets, vectors, weights = _case(window_bits, rate, arity, rows, cols,
                                      seed=window_bits * 100 + rate,
                                      table_mode=table_mode)
    w = weights if weighted else None
    want_states, want_sse = viterbi_window(targets, vectors, window_bits, rate,
                                           weights=w, chunk=512,
                                           impl="reference")
    got_states, got_sse = viterbi_window_paired(targets, vectors, window_bits,
                                                rate, weights=w, chunk=512)
    assert torch.equal(got_states, want_states)
    # The identical float: not approx, and not merely equal-valued, since a
    # partial sum that differed by an ulp would still print the same.
    assert got_sse.hex() == want_sse.hex()


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
def test_the_pair_matches_step_by_step_including_the_all_inf_start(
        window_bits, rate, arity, rows, cols):
    """Compare the FRONT after every pair, not only the final answer.

    A final-answer test can pass while an intermediate front differs, because
    the traceback only reads the argmin planes.  This walks both spellings in
    lockstep and compares the raw bits each pair produces, starting from the
    pinned all-``inf`` front that step 0 actually sees.
    """
    targets, vectors, weights = _case(window_bits, rate, arity, rows, cols,
                                      seed=7 + window_bits)
    size, _ = vectors.shape
    fan = 1 << rate
    low = size >> rate
    steps = rows // arity
    assert pair_is_applicable(window_bits, rate)

    n = cols
    x = targets.float().reshape(steps, arity, cols)
    wr = weights.float().reshape(steps, arity, cols)
    table = vectors.float()

    seq = torch.full((size, n), float("inf"))
    seq[0] = 0.0
    par = seq.clone()
    # The first pair reads the pinned front: state 0 is 0.0 and every other
    # state is +inf, so nearly every class minimum ties at inf.
    assert torch.isinf(seq[1:]).all() and seq[0].eq(0.0).all()

    checked_inf_start = False
    step = 0
    while step + 1 < steps:
        b1 = branch_costs(x[step], table, wr[step])
        b2 = branch_costs(x[step + 1], table, wr[step + 1])
        mid, p1_seq = step_sequential(seq, b1, fan, low)
        seq, p2_seq = step_sequential(mid, b2, fan, low)
        par, p1, p2 = step_pair_nested(par, b1, b2, fan, low)
        if step == 0:
            # The pinned start's own front, compared bit for bit.
            assert torch.equal(_bits(par), _bits(seq))
            assert torch.equal(p1, p1_seq)
            checked_inf_start = True
        assert torch.equal(_bits(par), _bits(seq)), f"front differs after pair at step {step}"
        assert torch.equal(p1, p1_seq), f"argmin_f differs at step {step}"
        assert torch.equal(p2, p2_seq), f"argmin_g differs at step {step + 1}"
        step += 2
    assert checked_inf_start


def test_the_pair_refuses_a_rate_it_cannot_split():
    """``L < 2R`` has no ``LOW2``; the pair says so instead of indexing past it."""
    assert not pair_is_applicable(6, 4)
    targets = torch.randn(4, 3)
    vectors = torch.randn(1 << 6, 2)
    with pytest.raises(ValueError, match="L >= 2R"):
        viterbi_window_paired(targets, vectors, 6, 4)


def test_duplicated_table_rows_really_do_tie():
    """The tie family is only a tie family if the floats actually collide.

    Without this the duplicated-table run could be passing because it never
    produced a tie, which would make it a second copy of the random case.
    """
    targets, vectors, weights = _case(8, 2, 2, 12, 7, seed=3,
                                      table_mode="duplicated")
    size, arity = vectors.shape
    fan, low = 1 << 2, size >> 2
    x = targets.float().reshape(6, arity, 7)
    front = torch.full((size, 7), float("inf"))
    front[0] = 0.0
    b1 = branch_costs(x[0], vectors.float(), None)
    front, _ = step_sequential(front, b1, fan, low)
    b2 = branch_costs(x[1], vectors.float(), None)
    front, _ = step_sequential(front, b2, fan, low)
    classes = front.view(fan, low, 7)
    ties = (classes == classes.min(dim=0, keepdim=True).values).sum(dim=0) > 1
    assert bool(ties.any()), "the duplicated table produced no ties to resolve"


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
@pytest.mark.parametrize("table_mode", ["random", "duplicated"])
@pytest.mark.parametrize("weighted", [False, True])
def test_the_best_form_returns_the_reference_bytes(window_bits, rate, arity,
                                                   rows, cols, table_mode,
                                                   weighted):
    """Carrying the class minimum instead of the front changes no byte.

    This arm needs no new tie argument: its scan compares the same sums the
    reference's scan compares, in the same order, under the same strict
    ``<``.  The test is here to hold that claim to bytes anyway.
    """
    targets, vectors, weights = _case(window_bits, rate, arity, rows, cols,
                                      seed=window_bits * 100 + rate,
                                      table_mode=table_mode)
    w = weights if weighted else None
    want_states, want_sse = viterbi_window(targets, vectors, window_bits, rate,
                                           weights=w, chunk=512,
                                           impl="reference")
    got_states, got_sse = viterbi_window_best_form(targets, vectors,
                                                   window_bits, rate,
                                                   weights=w, chunk=512)
    assert torch.equal(got_states, want_states)
    assert got_sse.hex() == want_sse.hex()


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
def test_the_best_form_reproduces_the_pinned_start_in_closed_form(
        window_bits, rate, arity, rows, cols):
    """``best_0`` is asserted, not assumed.

    The best-form skips step 0's scan and writes ``best_0 = [0, inf, ...]``
    with ``back[0]`` all zeros from the closed form.  That is a claim about
    what the reference's step 0 produces, so it is checked against it rather
    than reasoned about in a comment.
    """
    targets, vectors, _ = _case(window_bits, rate, arity, rows, cols, seed=11)
    size, _ = vectors.shape
    fan, low = 1 << rate, size >> rate
    n = cols
    pinned = torch.full((size, n), float("inf"))
    pinned[0] = 0.0
    best_ref, pred_ref = pinned.view(fan, low, n).min(dim=0)
    best_closed = torch.full((low, n), float("inf"))
    best_closed[0] = 0.0
    assert torch.equal(_bits(best_closed), _bits(best_ref))
    assert torch.equal(pred_ref, torch.zeros_like(pred_ref))


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
def test_the_best_form_carries_only_a_class_wide_state(window_bits, rate,
                                                       arity, rows, cols):
    """The saving is a shape, so the shape is what the test asserts.

    A byte claim that only lives in a docstring is a byte claim nothing
    checks.  The step returns ``[2^(L-R), n]``, not ``[2^L, n]``, and that
    ratio is the whole of the lever.
    """
    targets, vectors, _ = _case(window_bits, rate, arity, rows, cols, seed=13)
    size, _ = vectors.shape
    fan, low = 1 << rate, size >> rate
    steps = rows // arity
    x = targets.float().reshape(steps, arity, cols)
    best = torch.full((low, cols), float("inf"))
    best[0] = 0.0
    b = branch_costs(x[0], vectors.float(), None)
    out, pred = step_best_form(best, b, fan, low, size)
    assert out.shape == (low, cols)
    assert pred.shape == (low, cols)
    assert low * fan == size
