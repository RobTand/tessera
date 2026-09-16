"""The documented tile-word layout, packed from the bitstream for ANY rate.

The layout is bit-exact by construction: a column's 512 codes at R bits are
``512 * R`` bits, always a whole number of 32-bit words (``16 * R``), stored
MSB-first with the first stream bit as bit 31 of the column's first word;
columns of one rate run are contiguous per tile and tiles carry every run.
That recipe works for every rate 1..8.

``kernel_window_gemv.repack_window_body`` implements the same layout through
an 8//rate byte-packing step, which is why it only admits rates dividing 8;
that is an implementation restriction of that function, not a property of the
layout or of the compute kernel.  Tests that need rate 3/5/6/7 streams pack
them here (and check this packer agrees with ``repack_window_body`` wherever
that implementation can run).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import kernel_window_gemv as kg    # noqa: E402


def pack_bitstream(body, rates, tile_rows=kg.TILE_ROWS):
    """``body [rows, cols]`` codes + per-column rates -> ``kg.Repacked``.

    Pure bitstream packing: no byte grouping, no 8//rate step.  Rows are
    padded to the tile with zero codes (appending a zero code changes no
    earlier state), exactly the layout the kernel reads.
    """
    rows, cols = body.shape
    rates = tuple(int(r) for r in rates)
    if len(rates) != cols:
        raise ValueError(f"{len(rates)} rates for {cols} columns")
    if any(r < 1 or r > 8 for r in rates):
        raise ValueError(f"rates outside 1..8: {sorted(set(rates))}")
    order = sorted(range(cols), key=lambda c: (rates[c], c))
    perm = torch.tensor(order, dtype=torch.int32)
    rows_p = -(-rows // tile_rows) * tile_rows
    n_tiles = rows_p // tile_rows
    weights = (1 << torch.arange(31, -1, -1)).to(torch.int64)

    tile_parts = []
    runs = []
    col0 = 0
    word0 = 0
    for rate in sorted(set(rates)):
        which = [c for c in order if rates[c] == rate]
        n = len(which)
        idx = torch.tensor(which, dtype=torch.long)
        codes = body[:, idx].to(torch.int64)                       # [rows, n]
        if codes.numel() and int(codes.max()) >= (1 << rate):
            raise ValueError(f"a code exceeds {rate} bits (rate {rate})")
        bits = ((codes[:, :, None] >> torch.arange(rate - 1, -1, -1)) & 1)  # MSB-first
        bits = bits.permute(1, 0, 2)                               # [n, rows, rate]
        if rows_p != rows:
            bits = torch.cat([bits, torch.zeros(n, rows_p - rows, rate, dtype=torch.int64)], 1)
        bits = bits.reshape(n, n_tiles, tile_rows * rate)
        words_r = (bits.reshape(n, n_tiles, tile_rows * rate // 32, 32) * weights).sum(-1)
        tile_parts.append(words_r.permute(1, 0, 2).reshape(n_tiles, n * 16 * rate))
        runs.append((rate, col0, n, word0))
        col0 += n
        word0 += n * 16 * rate

    per_tile = torch.cat(tile_parts, dim=1)                        # [n_tiles, tile_words]
    flat = per_tile.reshape(-1)
    words = ((flat + (1 << 31)) % (1 << 32) - (1 << 31)).to(torch.int32)
    runs_t = torch.tensor(runs, dtype=torch.int32).reshape(-1, 4)
    return kg.Repacked(words=words, tile_words=word0, n_tiles=n_tiles, rows=rows, cols=cols,
                       rows_p=rows_p, perm=perm, runs=runs_t, rates=rates)
