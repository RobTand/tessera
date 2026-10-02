"""Contract tests for the piece-major resident word layout (tessera#739).

These call the PRODUCTION predicates and refusal helpers -- Repacked's re-lay,
piece_major_eligible, require_legacy_word_layout, and the readers that use
them -- not local re-implementations.  The refusal must happen BEFORE any
allocation or extension build, proven by monkeypatching the allocation/build
seams with bombs.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tessera.errors import GrammarError
from tessera import kernel_window_gemv as kw
from tessera.kernel_window_gemv import (
    PIECES_PER_TILE,
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


def _unit(rep, family="e4m3"):
    """A WindowGemvUnit around a repack, with the fields a reader reads."""
    return kw.WindowGemvUnit(
        rep=rep, table=torch.zeros(1 << 14, dtype=torch.bfloat16),
        scale=torch.ones(rep.rows, dtype=torch.float32), window_bits=14,
        plan=kw.default_plan(rep.rows, rep.cols, M=1), family=family)


# --- the production re-lay predicate and bijection ------------------------
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
    old = rep.words.reshape(3, PIECES_PER_TILE, per_piece)
    new = relaid.words.reshape(PIECES_PER_TILE, 3, per_piece)
    for t64 in range(PIECES_PER_TILE):
        for col in range(3):
            assert torch.equal(new[t64, col], old[col, t64])


def test_piece_major_relay_refuses_a_two_run_body():
    with pytest.raises(GrammarError):
        _repacked((4, 5)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)


def test_piece_major_relay_refuses_a_non_rate_four_body():
    with pytest.raises(GrammarError):
        _repacked((5,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)


def test_unknown_word_layout_is_refused():
    rep = _repacked((4,))
    with pytest.raises(GrammarError):
        rep.with_word_layout("f16_piece_major")


# --- the production readers refuse, before any allocation/build -----------
def _bomb(*a, **k):
    raise AssertionError("allocation/extension build must not run before the refusal")


def test_window_gemv_refuses_before_allocation(monkeypatch):
    unit = _unit(_repacked((4,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR))
    # _op_args / _gemv_op / _gemv_concrete are the allocation/extension seams.
    monkeypatch.setattr(kw, "_op_args", _bomb)
    monkeypatch.setattr(kw, "_gemv_op", _bomb)
    monkeypatch.setattr(kw, "_gemv_concrete", _bomb)
    with pytest.raises(GrammarError):
        kw.window_gemv(unit, torch.zeros(1, unit.cols, dtype=torch.bfloat16))


def test_decode_codes_refuses_before_extension(monkeypatch):
    unit = dataclasses.replace(_unit(_repacked((4,)).with_word_layout(WORD_LAYOUT_PIECE_MAJOR)),
                               codes_of_state=torch.zeros(1 << 14, dtype=torch.uint8))
    monkeypatch.setattr(kw, "_ext", _bomb)
    # _ext is the extension loader used only AFTER the layout gate.
    with pytest.raises(GrammarError):
        kw.decode_codes(unit)


def test_require_legacy_word_layout_is_the_production_helper():
    require_legacy_word_layout(WORD_LAYOUT_LEGACY, "reader")
    with pytest.raises(GrammarError):
        require_legacy_word_layout(WORD_LAYOUT_PIECE_MAJOR, "reader")


# --- production bundle + supported-check tags -----------------------------
def _bundle(word_layout, family="e4m3"):
    import types
    return types.SimpleNamespace(
        family=family, word_layout=word_layout, arithmetic="epilogue", device=torch.device("cpu"),
        window_bits=14, experts=2, cols=64, rows=64, quantizer="native",
        # one run per expert: (rate, col0, ncols, word0) rows, flattened [E, R, 4]
        runs_all=torch.tensor([[[4, 0, 64, 0]]] * 2, dtype=torch.int32),
        scale_all=torch.ones(2, 64), perm_all=torch.zeros(2, 64, dtype=torch.int32),
        init_all=torch.zeros(2, 64, dtype=torch.int32), has_init=torch.zeros(2, dtype=torch.int32),
        tile_words=torch.full((2,), 64 * 16 * 4, dtype=torch.int32),
        run_off=torch.tensor([0, 1, 2], dtype=torch.int32), library="threshold")


def test_supported_check_refuses_piece_major_for_a_non_e4m3_family():
    from tessera.routed_fused import fused_routed_window_supported
    g = _bundle(WORD_LAYOUT_PIECE_MAJOR, family="value")
    why = fused_routed_window_supported(g, g, g)
    # value family is refused earlier for its arithmetic; the layout must not
    # be the thing that lets a non-E4M3 stack through.
    assert why is not None


def test_has_one_rate_four_run_matches_the_bounded_shape():
    from tessera.routed_fused import has_one_rate_four_run
    assert has_one_rate_four_run(_bundle(WORD_LAYOUT_PIECE_MAJOR), 2) is True
    two_run = _bundle(WORD_LAYOUT_PIECE_MAJOR)
    two_run.runs_all = torch.tensor([[[4, 0, 32, 0], [5, 32, 32, 2048]]] * 2, dtype=torch.int32)
    two_run.run_off = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert has_one_rate_four_run(two_run, 2) is False
