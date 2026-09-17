"""The documented state recipe behind ``window_gemm``'s in-kernel decode,
checked on CPU against the definition -- a test like any other, run through
PB (CPU pool), not a local screen.

``window_gemm`` decodes a state as a windowed bit gather: the last ``L`` bits
of the MSB-first code stream ending at ``q = (row + 1) * rate``, where the
first tile's opening is zero-padded and later tiles read the previous tile's
last word of the same run.  This file is that recipe transcribed for torch
int64 and held against ``reference_states`` -- the defined state machine --
for the rates the pinned packer admits and across the 512-row tile boundary.
It does not import the Triton module: the GPU tests own the kernel itself.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import kernel_window_gemv as kg    # noqa: E402

L = 14
TILE = kg.TILE_ROWS


def _body(rows, cols, rates, seed):
    g = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates, dtype=torch.int64)
    return (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.uint8)


def extract(rep, rows, cols):
    """The kernel's recipe, one step at a time, in int64."""
    states = torch.zeros(rows, cols, dtype=torch.int64)
    words = rep.words.to(torch.int64) & 0xFFFFFFFF
    total = words.numel()
    for rate, col0, ncols, word0 in rep.runs.tolist():
        rate, col0, ncols, word0 = int(rate), int(col0), int(ncols), int(word0)
        chunk = 16 * rate
        base = torch.arange(ncols, dtype=torch.int64) * chunk + word0
        for g in range(rep.n_tiles):
            t = torch.arange(TILE, dtype=torch.int64)
            q = (t + 1) * rate
            length = torch.minimum(q + g * TILE * rate, torch.tensor(L, dtype=torch.int64))
            qb = q - L
            neg = qb < 0
            wi = torch.where(neg, torch.full_like(qb, -1), qb >> 5)
            d0 = torch.where(neg, -rep.tile_words + chunk - 1, wi)
            d1 = wi + 1
            shift = 64 - q + 32 * wi
            tb = g * rep.tile_words + base
            idx0 = tb[:, None] + d0[None, :]           # [ncols, TILE]
            idx1 = tb[:, None] + d1[None, :]
            zero = torch.zeros((), dtype=torch.int64)
            ok0 = (idx0 >= 0) & (idx0 < total)
            ok1 = (idx1 >= 0) & (idx1 < total)
            w0 = torch.where(ok0, words[idx0.clamp(0, total - 1)], zero)
            w1 = torch.where(ok1, words[idx1.clamp(0, total - 1)], zero)
            combined = (w0 << 32) | w1
            st = (combined >> shift[None, :]) & ((1 << length) - 1)[None, :]
            rows_g = min(TILE, rows - g * TILE)
            orig = rep.perm[col0:col0 + ncols].to(torch.int64)
            states[g * TILE:g * TILE + rows_g][:, orig] = st[:, :rows_g].t()
    return states


def _check(rows, cols, rates, seed):
    body = _body(rows, cols, rates, seed)
    ref = kg.reference_states(body, rates, L)
    rep = kg.repack_window_body(body, rates)
    got = extract(rep, rows, cols)
    assert torch.equal(ref, got), f"rows={rows} rates={sorted(set(rates))} seed={seed}"


def test_recipe_matches_the_definition_single_rate():
    for rate in (1, 2, 4):
        _check(600, 24, (rate,) * 24, seed=10 + rate)


def test_recipe_matches_the_definition_mixed_rates():
    rates = tuple(1 if c % 5 == 0 else (2 if c % 3 else 4) for c in range(32))
    _check(1024, 32, rates, seed=5)


def test_recipe_crosses_the_tile_lookback():
    """Rows past 512 read the previous tile's last word of the same run."""
    rates = tuple(2 if c % 4 == 0 else 4 for c in range(16))
    _check(768, 16, rates, seed=7)
    _check(1536, 16, rates, seed=8)
