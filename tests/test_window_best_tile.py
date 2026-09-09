"""The screened tile is a machine knob: it moves launches, never bytes.

``TESSERA_WINDOW_BEST_TILE`` overrides ``_tile_best``'s ``(BL, BC, WARPS)``
so the default can be measured rather than assumed.  That makes two claims a
test has to hold it to.

The first is about bytes.  ``_step_best`` masks both axes (``li < low``,
``ci < m``), so every legal tile writes the identical states and the identical
``sse`` float -- and a screen whose arms do not agree on bytes has measured
something other than the tile.  Compared here against the reference AND
against the tile the default picks, at the production rates.

The second is about refusal.  Triton rejects a non-power-of-two block size and
warp count late and obscurely, from inside a launch, where the name of the
setting is long gone.  The parse refuses first, and says which field.

The knob is also in the plan-cache key: a plan built under one tile would
replay the wrong graph for another, and the screen puts several tiles on one
tensor seconds apart in one process.  That is what
``test_the_cache_key_separates_tiles`` pins, and it does not need a board.
"""
from __future__ import annotations

import pytest
import torch

from tessera import window_viterbi as wv
from tessera.encode import viterbi_window

# (BL, BC, WARPS) -- the incumbent, a wider column tile, a taller class tile,
# and one that overshoots ``low`` on the 12-bit shape so ``mask_l`` is
# exercised rather than assumed.
TILES = [(128, 2, 4), (128, 8, 4), (256, 4, 8), (64, 4, 2), (1024, 4, 8)]


def test_an_unset_knob_is_the_derived_tile():
    for low, n in ((2048, 192), (1024, 64), (256, 8)):
        assert wv._resolve_tile_best(low, n) == wv._tile_best(low, n)


def test_an_empty_knob_is_the_derived_tile(monkeypatch):
    monkeypatch.setenv(wv._BEST_TILE_ENV, "")
    assert wv._resolve_tile_best(2048, 192) == wv._tile_best(2048, 192)


def test_a_set_knob_is_taken_verbatim(monkeypatch):
    monkeypatch.setenv(wv._BEST_TILE_ENV, "256,4,8")
    assert wv._resolve_tile_best(2048, 192) == (256, 4, 8)
    # Not clamped to the problem.  The kernel masks, and clamping would
    # silently collapse two arms of a screen onto one point.
    monkeypatch.setenv(wv._BEST_TILE_ENV, "4096,16,8")
    assert wv._resolve_tile_best(2048, 192) == (4096, 16, 8)


@pytest.mark.parametrize("raw,says", [
    ("128,4", "BL,BC,WARPS"),
    ("128,4,8,2", "BL,BC,WARPS"),
    ("128,x,8", "three integers"),
    ("96,4,8", "BL=96"),
    ("128,3,8", "BC=3"),
    ("128,4,5", "WARPS=5"),
    ("128,4,64", "WARPS=64"),
    ("0,4,8", "BL=0"),
    ("-128,4,8", "BL=-128"),
])
def test_a_bad_knob_is_refused_by_name(monkeypatch, raw, says):
    monkeypatch.setenv(wv._BEST_TILE_ENV, raw)
    with pytest.raises(ValueError) as exc:
        wv._resolve_tile_best(2048, 192)
    assert wv._BEST_TILE_ENV in str(exc.value)
    assert says in str(exc.value)


def test_the_cache_key_separates_tiles(monkeypatch):
    """Two tiles must not share a plan: the graph is captured per tile."""
    import inspect

    src = inspect.getsource(wv._plan_for_call)
    # After the key is opened, so a mention in the docstring above it does not
    # pass for the key carrying it.
    assert "_BEST_TILE_ENV" in src[src.index("    key = ("):], src


cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the best-form step is a CUDA path")


def _inputs(window_bits, rate, rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(rows, cols, generator=g).cuda()
    vectors = torch.randn(1 << window_bits, 1, generator=g).cuda()
    weights = (torch.rand(rows, cols, generator=g) + 0.5).cuda()
    return targets, vectors, weights


@cuda
@pytest.mark.parametrize("window_bits,rate,rows,cols", [
    (14, 3, 64, 192),        # the production rate, at the screened width
    (14, 4, 64, 64),         # the other production rate
    (12, 3, 64, 130),        # a width the batch loop must split
])
@pytest.mark.parametrize("graph", ["0", "1"])
def test_every_tile_returns_the_same_bytes(monkeypatch, window_bits, rate,
                                           rows, cols, graph):
    targets, vectors, weights = _inputs(window_bits, rate, rows, cols, seed=7)
    monkeypatch.setenv(wv._GRAPH_ENV, graph)
    wv.window_plan_cache_clear()
    ref_states, ref_sse = viterbi_window(targets, vectors, window_bits, rate,
                                         weights=weights, impl="reference")
    monkeypatch.setenv(wv._BEST_FORM_ENV, "1")
    seen = {}
    for tile in TILES:
        monkeypatch.setenv(wv._BEST_TILE_ENV, ",".join(str(x) for x in tile))
        states, sse = viterbi_window(targets, vectors, window_bits, rate,
                                     weights=weights, impl="fused")
        assert torch.equal(states, ref_states), tile
        # sse as the identical float, not a tolerance: a tile that changes the
        # summation order has changed the thing the screen is comparing.
        assert sse == ref_sse, (tile, sse.hex(), ref_sse.hex())
        seen[tile] = sse.hex()
    # And the derived default is one of the points, not a different answer.
    monkeypatch.delenv(wv._BEST_TILE_ENV, raising=False)
    states, sse = viterbi_window(targets, vectors, window_bits, rate,
                                 weights=weights, impl="fused")
    assert torch.equal(states, ref_states)
    assert sse == ref_sse, (sse.hex(), ref_sse.hex())
    assert len(set(seen.values())) == 1, seen
