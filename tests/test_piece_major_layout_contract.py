"""Source-contract tests for the piece-major resident word layout (tessera#739).

The layout is decided once at intake and recorded on the per-stack signature;
a reader that knows only the legacy order must refuse a re-laid stack rather
than re-stride it.  These tests cover the tag home (``Repacked``), the re-lay
bijection, eligibility, and the compact Triton reader's refusal.
"""
from __future__ import annotations

import pytest
import torch

from tessera.errors import GrammarError
from tessera.kernel_window_gemv import (
    PICECES_PER_TILE,
    WORD_LAYOUT_LEGACY,
    WORD_LAYOUT_PIECE_MAJOR,
    Repacked,
    piece_major_eligible,
    require_legacy_word_layout,
)


def _repacked(rates, cols=3):
    rate = rates[0]
    per_col = 16 * rate
    tile_words = cols * per_col
    words = torch.arange(tile_words, dtype=torch.int32)
    runs = torch.tensor([[int(rates[0]), 0, cols, 0]], dtype=torch.int32)
    return Repacked(
        words=words, tile_words=tile_words, n_tiles=1, rows=512, cols=cols, rows_p=512,
        perm=torch.arange(cols, dtype=torch.int32), runs=runs,
        rates=tuple(int(r) for r in rates), word_layout=WORD_LAYOUT_LEGACY)


def test_default_layout_is_legacy():
    assert _repacked((4,)).word_layout == WORD_LAYOUT_LEGACY


def test_piece_major_eligible_only_for_one_rate_four_run():
    assert piece_major_eligible(_repacked((4,)))
    assert not piece_major_eligible(_repacked((4, 5)))
    assert not piece_major_eligible(_repacked((5,)))


def test_layout_property_carries_the_word_layout():
    legacy = _repacked((4,)).layout
    relaid = _repacked((4,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR).layout
    assert ("word_layout", WORD_LAYOUT_LEGACY) in legacy
    assert ("word_layout", WORD_LAYOUT_PIECE_MAJOR) in relaid
    assert legacy != relaid


def test_piece_major_relay_is_a_bijection_and_keeps_the_counts():
    rep = _repacked((4,), cols=3)
    relaid = rep.with_word_layout(WORD_LAYOUT_PIECE_MAJOR)
    assert relaid.word_layout == WORD_LAYOUT_PIECE_MAJOR
    assert relaid.tile_words == rep.tile_words
    assert relaid.n_tiles == rep.n_tiles
    assert torch.equal(relaid.perm, rep.perm)
    assert torch.equal(relaid.runs, rep.runs)
    assert relaid.rates == rep.rates
    assert sorted(relaid.words.tolist()) == sorted(rep.words.tolist())
    per_piece = 2 * 4
    old = rep.words.reshape(3, PICECES_PER_TILE, per_piece)
    new = relaid.words.reshape(PICECES_PER_TILE, 3, per_piece)
    for t64 in range(PICECES_PER_TILE):
        for col in range(3):
            assert torch.equal(new[t64, col], old[col, t64])


def test_piece_major_relay_refuses_a_two_run_body():
    with pytest.raises(GrammarError):
        _repacked((4, 5)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)


def test_piece_major_relay_refuses_a_non_rate_four_body():
    with pytest.raises(GrammarError):
        _repacked((5,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)


def test_legacy_only_reader_refuses_a_non_legacy_stack():
    # the compact Triton reader and every other unlearned reader calls this
    require_legacy_word_layout(WORD_LAYOUT_LEGACY, "reader")          # allowed
    with pytest.raises(GrammarError):
        require_legacy_word_layout(WORD_LAYOUT_PIECE_MAJOR, "reader")
