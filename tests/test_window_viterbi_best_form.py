"""The best-form step returns the front-form's bytes on the board.

``TESSERA_WINDOW_BEST_FORM`` substitutes the ``2^L`` front out of the step
loop: the recurrence closes in the class minimum, so the front the reference
writes every step exists only to be minimised away by the next one.  It is a
machine knob and never the answer, which is a claim about bytes and is
therefore tested as one -- states compared exactly, ``sse`` compared as the
identical float, against BOTH the front-form fused path and the torch
reference, at every rate the two spellings share.

These are CUDA tests.  The best-form has no CPU path; ``encode.viterbi_window``
routes CPU inputs to the reference either way.
"""
from __future__ import annotations

import pytest
import torch

from tessera import window_viterbi as wv
from tessera.encode import viterbi_window

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the best-form step is a CUDA path")

# (window_bits, rate, arity, rows, cols)
SHAPES = [
    (10, 2, 1, 64, 96),
    (10, 3, 2, 64, 96),
    (12, 3, 1, 128, 64),
    (12, 4, 2, 64, 130),     # a width the batch loop must split
    (14, 3, 1, 64, 48),      # the production rate
    (14, 4, 1, 64, 48),      # the other production rate
    (8, 5, 1, 32, 33),       # R > L - R: the shift is not a class stride
]


def _inputs(window_bits, rate, arity, rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    size = 1 << window_bits
    targets = torch.randn(rows, cols, generator=g).cuda()
    vectors = torch.randn(size, arity, generator=g).cuda()
    weights = (torch.rand(rows, cols, generator=g) + 0.5).cuda()
    return targets, vectors, weights


def _run(monkeypatch, best_form, targets, vectors, window_bits, rate, weights,
         chunk, graph):
    # The knob is read per call, but the plan cache is keyed on it, so a
    # stale plan from the other spelling can never be replayed here.
    monkeypatch.setenv(wv._BEST_FORM_ENV, "1" if best_form else "0")
    monkeypatch.setenv(wv._GRAPH_ENV, "1" if graph else "0")
    wv.window_plan_cache_clear()
    return viterbi_window(targets, vectors, window_bits, rate,
                          weights=weights, chunk=chunk, impl="fused")


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_the_best_form_returns_the_front_forms_bytes(monkeypatch, window_bits,
                                                     rate, arity, rows, cols,
                                                     weighted, graph):
    targets, vectors, weights = _inputs(window_bits, rate, arity, rows, cols,
                                        seed=window_bits * 10 + rate)
    w = weights if weighted else None
    want_states, want_sse = _run(monkeypatch, False, targets, vectors,
                                 window_bits, rate, w, 512, graph)
    got_states, got_sse = _run(monkeypatch, True, targets, vectors,
                               window_bits, rate, w, 512, graph)
    assert torch.equal(got_states, want_states)
    assert got_sse.hex() == want_sse.hex()


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
def test_the_best_form_returns_the_reference_bytes(monkeypatch, window_bits,
                                                   rate, arity, rows, cols):
    """The front-form is already pinned to the reference; pin this one too.

    Comparing only against the fused path would let a shared defect pass, so
    the definition gets its own assertion.
    """
    targets, vectors, weights = _inputs(window_bits, rate, arity, rows, cols,
                                        seed=99 + window_bits)
    want_states, want_sse = viterbi_window(targets, vectors, window_bits, rate,
                                           weights=weights, chunk=512,
                                           impl="reference")
    got_states, got_sse = _run(monkeypatch, True, targets, vectors,
                               window_bits, rate, weights, 512, False)
    assert torch.equal(got_states, want_states)
    assert got_sse.hex() == want_sse.hex()


@pytest.mark.parametrize("window_bits,rate", [(12, 3), (14, 3), (14, 4)])
def test_the_best_form_holds_more_columns_in_the_same_budget(monkeypatch,
                                                             window_bits, rate):
    """The width is the lever, so the width is what the test asserts.

    The byte saving per step is arithmetic; what it buys is that the same L2
    budget admits ``FAN`` times as many columns, which is more blocks over the
    same serial chain.  A speed claim needs a measurement, but the shape claim
    does not, and it is the one that would silently regress.
    """
    size = 1 << window_bits
    low = size >> rate
    dev = torch.device("cuda")
    _, front_width, _ = wv._layout(dev, size, 4096, 512, size)
    _, best_width, _ = wv._layout(dev, size, 4096, 512, low)
    assert best_width >= front_width
    # Capped by ``nmax`` = min(chunk, cols), never by the budget alone.
    assert best_width == min(512, front_width * (1 << rate))


@pytest.mark.parametrize("raw", ["2", "yes", "true", "-1"])
def test_the_knob_refuses_a_value_it_cannot_mean(monkeypatch, raw):
    monkeypatch.setenv(wv._BEST_FORM_ENV, raw)
    with pytest.raises(ValueError, match=wv._BEST_FORM_ENV):
        wv._resolve_best_form()


def _tied_inputs(window_bits, rate, arity, rows, cols, seed):
    """Tables built so predecessors really do arrive carrying the same float.

    Two ways for a tie to appear, and the tie rule -- first minimal index --
    has to survive both.  Duplicated rows make two predecessors carry the
    *identical* cost by construction.  A coarse table makes distinct rows
    produce sums that ROUND to the same float, which is the case a reviewer
    cannot enumerate and the case a reordered scan would break silently.
    """
    g = torch.Generator().manual_seed(seed)
    size = 1 << window_bits
    targets = (torch.randn(rows, cols, generator=g) * 0.5).round() * 0.5
    half = torch.randn(size // 2, arity, generator=g)
    vectors = (half.repeat(2, 1) * 2.0).round() * 0.5
    weights = torch.ones(rows, cols)
    return targets.cuda(), vectors.cuda(), weights.cuda()


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
@pytest.mark.parametrize("weighted", [False, True])
def test_the_best_form_breaks_ties_where_the_front_form_breaks_them(
        monkeypatch, window_bits, rate, arity, rows, cols, weighted):
    targets, vectors, weights = _tied_inputs(window_bits, rate, arity, rows,
                                             cols, seed=7 + window_bits)
    w = weights if weighted else None
    want_states, want_sse = _run(monkeypatch, False, targets, vectors,
                                 window_bits, rate, w, 512, False)
    got_states, got_sse = _run(monkeypatch, True, targets, vectors,
                               window_bits, rate, w, 512, False)
    assert torch.equal(got_states, want_states)
    assert got_sse.hex() == want_sse.hex()
    ref_states, ref_sse = viterbi_window(targets, vectors, window_bits, rate,
                                         weights=w, chunk=512, impl="reference")
    assert torch.equal(got_states, ref_states)
    assert got_sse.hex() == ref_sse.hex()


@pytest.mark.parametrize("window_bits,rate,arity,rows,cols", SHAPES)
def test_the_tied_tables_really_do_tie(window_bits, rate, arity, rows, cols):
    """A tie family that stopped tying would pass the test above and mean nothing."""
    targets, vectors, _ = _tied_inputs(window_bits, rate, arity, rows, cols,
                                       seed=7 + window_bits)
    size = 1 << window_bits
    low = size >> rate
    fan = 1 << rate
    # The first step's costs, class by class: the same shape the scan reduces.
    d = targets[0][:, None, None] - vectors[None, :, :]
    cost = (d * d).sum(-1)                                   # [cols, size]
    tiles = cost.view(cols, fan, low)
    assert bool((tiles.min(dim=1).values[:, None, :] == tiles).sum(1).gt(1).any())


@pytest.mark.parametrize("chunk", [40, 96, 130])
def test_the_best_form_holds_on_a_batch_the_chunk_does_not_divide(monkeypatch,
                                                                  chunk):
    """Partial column tiles: ``m`` is read from ``ctl``, so the tail is a mask.

    ``cols % chunk`` and ``m % BC`` are different edges and both are here --
    a captured plan replays one graph for every batch, so a tail batch that
    read a stale width would show up as bytes, not as a crash.
    """
    window_bits, rate, arity, rows, cols = 12, 3, 2, 64, 205
    targets, vectors, weights = _inputs(window_bits, rate, arity, rows, cols,
                                        seed=chunk)
    want_states, want_sse = viterbi_window(targets, vectors, window_bits, rate,
                                           weights=weights, chunk=chunk,
                                           impl="reference")
    got_states, got_sse = _run(monkeypatch, True, targets, vectors,
                               window_bits, rate, weights, chunk, True)
    assert torch.equal(got_states, want_states)
    assert got_sse.hex() == want_sse.hex()
