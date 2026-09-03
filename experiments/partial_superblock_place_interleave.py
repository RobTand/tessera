"""Does ONE extra (partial) superblock cost a full block's worth at load?

The ladder in ``partial_superblock_loadcost.py`` could not resolve this.  Its
4864-column placement arm differed by 29% between the forward and reverse
passes (158.65 vs 205.27 ms) while the other three widths agreed to 0.6%, and
the same call re-measured later in the same process read 428 ms -- so the
19-block arm drifts by more than the ~9 ms/block the block sweep predicts.  A
ladder cannot separate a 9 ms effect from a 200 ms drift.

So: interleave.  A = 4864 columns (19 blocks, none partial), B = 4896 columns
(20 blocks, the last one 32/256 full), and ABABAB... in one process so any
drift lands on both arms.  Both tensors are prepared before the first
measurement, nothing is allocated between the two calls of a pair, and the
statistic is the per-pair difference -- which is drift-immune in a way a
per-arm minimum is not.

The columns differ by 32, so 32 columns of extra work is inside the B arm too;
the 5088/5120 pair from the ladder (+32 columns, SAME block count) prices that
at +1.1 to +2.2 ms, which is the correction to subtract.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tessera.alphabet import build_forest
from tessera.decode import decode_codes_mixed, unit_scale_field
from tessera.encode import _canonical_release_order, e2m1_value_table, encode_unit
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import RotationState, ScalePlaneKind
from tessera.trellis import ConvCode

Q256, SUPERBLOCK, GROUP, HALF = 768, 256, 32, 16
ROWS = int(os.environ.get("PPI_ROWS", "17408"))
PAIRS = int(os.environ.get("PPI_PAIRS", "9"))
DEV = os.environ.get("PPI_DEVICE", "cuda")
SOURCE = "/mnt/shared/tessera-ts44/gate_proj.pt"
CC = ConvCode(memory=6)
FRAC = 0.125
A_COLS, B_COLS = 4864, 4896


def prepare(W, cols):
    rates = bresenham_rate_schedule(root_from_q256(Q256), cols)
    forests = {r: build_forest(r) for r in sorted(set(rates))}
    n_rel = int(FRAC * ROWS * cols)
    unit = encode_unit(
        W[:, :cols].contiguous(), forests, rates, CC, rotation=RotationState.NONE,
        with_diagonals=False, released_positions=n_rel, group=GROUP, half=HALF,
        superblock=SUPERBLOCK, scale_plane=ScalePlaneKind.S6B,
    )
    pre = decode_codes_mixed(unit, forests, CC, apply_release=False)
    scale = unit_scale_field(unit, ROWS, cols)
    decoded = e2m1_value_table(pre.device)[pre.int()] * scale
    del pre, scale
    return {"cols": cols, "decoded": decoded, "n_rel": n_rel}


def one(arm):
    t = time.perf_counter()
    _canonical_release_order(arm["decoded"], arm["cols"], SUPERBLOCK, arm["n_rel"])
    if DEV == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    print(f"host {socket.gethostname()}  device {DEV}  rows {ROWS}  pairs {PAIRS}  "
          f"HEAD {os.environ.get('PPI_HEAD', '?')}")
    W = torch.load(SOURCE, map_location="cpu")[:ROWS].to(DEV).float().contiguous()
    A, B = prepare(W, A_COLS), prepare(W, B_COLS)
    del W
    print(f"A {A_COLS} cols = 19 blocks, none partial, {A['n_rel']} releases")
    print(f"B {B_COLS} cols = 20 blocks, last 32/256, {B['n_rel']} releases")
    one(A); one(B)                                     # warm both, then measure
    rows = []
    print(f"\n  {'pair':>4} {'A ms':>9} {'B ms':>9} {'B-A ms':>9}")
    for i in range(PAIRS):
        a, b = one(A), one(B)
        rows.append({"pair": i, "a": a, "b": b})
        print(f"  {i:>4} {a*1e3:>9.2f} {b*1e3:>9.2f} {(b-a)*1e3:>9.2f}")
    diffs = [(r["b"] - r["a"]) * 1e3 for r in rows]
    aa = [r["a"] * 1e3 for r in rows]
    bb = [r["b"] * 1e3 for r in rows]
    print(f"\n  A: min {min(aa):7.2f}  median {statistics.median(aa):7.2f}  max {max(aa):7.2f}"
          f"   (spread {max(aa)/min(aa):.2f}x)")
    print(f"  B: min {min(bb):7.2f}  median {statistics.median(bb):7.2f}  max {max(bb):7.2f}"
          f"   (spread {max(bb)/min(bb):.2f}x)")
    print(f"  paired B-A: median {statistics.median(diffs):+7.2f} ms   "
          f"mean {statistics.mean(diffs):+7.2f}   sd {statistics.stdev(diffs):6.2f}   "
          f"range [{min(diffs):+.2f}, {max(diffs):+.2f}]")
    sign = sum(1 for x in diffs if x > 0)
    print(f"  sign test: {sign}/{len(diffs)} pairs have B slower")
    report = {"host": socket.gethostname(), "device": DEV, "rows": ROWS,
              "head": os.environ.get("PPI_HEAD", ""), "a_cols": A_COLS, "b_cols": B_COLS,
              "pairs": rows, "median_diff_ms": statistics.median(diffs),
              "sign_positive": sign,
              "window": {"start": None, "end": time.time()}}
    dest = os.environ.get("PPI_JSON")
    if dest:
        with open(dest, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
