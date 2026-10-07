"""``kernel_window_gemv.unpack_tile_words`` is the exact inverse of the tile-order repack.

A consumer that holds only the resident words (the routed class build's bundles) reads the
codes back in original column order: through the documented layout for every rate 1..14,
through ``repack_window_body`` and through the piece-major regrouping.
"""
from __future__ import annotations

import pytest
import torch

from tessera import kernel_window_gemv as kg
from window_pack_reference import pack_bitstream


def _body(rows, rates, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randint(0, 1 << r, (rows,), generator=g) for r in rates], 1)


@pytest.mark.parametrize("rows", [512, 700, 1024])
@pytest.mark.parametrize("rates", [(3,) * 8, (4, 3) * 6, tuple(range(1, 15)) * 2, (7, 5, 6, 3, 14)])
def test_unpack_inverts_the_documented_layout(rows, rates):
    body = _body(rows, rates, rows + len(rates))
    rep = pack_bitstream(body, rates)
    assert torch.equal(kg.unpack_tile_words(rep), body.to(torch.int32))


def test_unpack_inverts_repack_window_body_and_piece_major():
    rates = (4,) * 64
    body = _body(1024, rates, 9)
    rep = kg.repack_window_body(body, rates)
    assert torch.equal(kg.unpack_tile_words(rep), body.to(torch.int32))
    relaid = rep.with_word_layout(kg.WORD_LAYOUT_PIECE_MAJOR)
    assert not torch.equal(relaid.words, rep.words)
    assert torch.equal(kg.unpack_tile_words(relaid), body.to(torch.int32))
    mixed = (1, 2, 4) * 8
    body = _body(512, mixed, 10)
    assert torch.equal(kg.unpack_tile_words(kg.repack_window_body(body, mixed)), body.to(torch.int32))
