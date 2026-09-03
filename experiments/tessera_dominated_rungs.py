"""Count the dominated rungs of Tessera's rate axis, per grid and per shape (#43).

A rung is **dominated** when some other rung of the same unit costs no more
bytes and is no worse.  Offering one to an allocator is offering a choice that
cannot be right, so the question this harness answers is arithmetic before it
is a matter of taste: *how many are there, on which shapes, and by how much*.

Three legs, all measurable, `--all` runs them in order:

``--count`` (default)
    Prices every rung the grammar admits at each shape through
    :func:`tessera.control.unit_wire_bits` -- the wire's own accountant -- and
    counts the rungs some higher rung matches or beats on bytes.

``--verify``
    Proves the accountant by encoding real units and comparing
    ``encode_linear(...).exact_bytes * 8`` against it.  The count is only worth
    reading if this leg is exact: before 2026-09-02 it was not, and the
    accountant was 512 B light on every E2M1x2 unit at the coset cap, which
    inflated every drop in the table by ``4096 / params`` bpp.

``--quality``
    The other half of "dominated".  Bytes are exact arithmetic; *no worse* is
    an inference across a recipe change (window L=12 below the cap, coset
    trellis at it), so it is measured: encode a Gaussian unit at rungs spanning
    the dominated region, decode, and report relative SSE.  Gaussian source,
    one unit, weight space -- the scope of the claim.

    CPU only, minutes, by construction: it runs while a GPU measurement is in
    flight.

    python experiments/tessera_dominated_rungs.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from tessera.control import GRID_NAMES, grid_for_name, unit_wire_bits
from tessera.errors import GrammarError
from tessera.export import rung_ceiling

#: Small enough that a per-unit table is a large share of the unit, up to the
#: production shapes the issue reports as clean.  (96, 320) and (64, 640) carry
#: a partial trailing superblock, which is where the granule arithmetic floors.
SHAPES = ((96, 320), (64, 512), (64, 640), (96, 768), (512, 2048), (1024, 3072))


def priced_axis(grid_name: str, rows: int, columns: int):
    """``[(q256, bits)]`` for every rung the grammar admits at this shape."""
    grid = grid_for_name(grid_name)
    out = []
    for q in range(1, int(rung_ceiling(grid)) + 1):
        try:
            out.append((q, unit_wire_bits(grid, q, rows, columns)))
        except (GrammarError, ValueError, ZeroDivisionError):
            continue
    return out


def dominated(priced):
    """``{q256: dominating q256}`` -- rungs a higher rung matches or beats."""
    out, best_q, best_bits = {}, None, None
    for q, bits in reversed(priced):
        if best_bits is not None and best_bits <= bits:
            out[q] = best_q
        if best_bits is None or bits < best_bits:
            best_q, best_bits = q, bits
    return out


def count_table(grids, shapes) -> int:
    print("| grid | shape | legal rungs | dominated | worst gap (bpp) | drops |")
    print("|---|---|---:|---:|---:|---|")
    total = 0
    for name in grids:
        for rows, columns in shapes:
            priced = priced_axis(name, rows, columns)
            dom = dominated(priced)
            total += len(dom)
            bits = dict(priced)
            worst = max(
                (float((bits[q] - bits[dom[q]]) / (rows * columns)) for q in dom),
                default=0.0,
            )
            drops = [
                f"R{priced[i][0]}->R{priced[i + 1][0]} "
                f"{float((priced[i][1] - priced[i + 1][1]) / (rows * columns)):.4f}"
                for i in range(len(priced) - 1)
                if priced[i + 1][1] < priced[i][1]
            ]
            print(
                f"| {name} | {rows}x{columns} | {len(priced)} | {len(dom)} | "
                f"{worst:.4f} | {', '.join(drops) or '-'} |"
            )
    return total


#: (grid, q256, rows, columns) the accountant is checked against a real encode
#: on: both bodies, both arities, both sides of the coset cap, and the two
#: arity-1 rungs whose schedule carries two distinct rates (and therefore two
#: forests) against the uniform rung above them.
VERIFY_CASES = (
    ("E2M1", 256, 64, 512),
    ("E2M1", 511, 64, 512),
    ("E2M1", 512, 64, 512),
    ("E2M1x2", 512, 64, 512),
    ("E2M1x2", 895, 64, 512),
    ("E2M1x2", 896, 64, 512),
    ("E2M1x2", 896, 32, 384),
    ("E4M3", 1024, 32, 320),
    ("BF16", 1024, 32, 256),
)


def verify_bytes() -> int:
    import torch

    from tessera.export import encode_linear

    bad = 0
    print("\naccountant against the exporter (exact_bytes * 8):")
    for name, q256, rows, columns in VERIFY_CASES:
        grid = grid_for_name(name)
        torch.manual_seed(11)
        weight = torch.randn(rows, columns)
        unit = encode_linear(weight, grid=grid, q256=q256)
        accounted = int(unit_wire_bits(grid, q256, rows, columns))
        delta = unit.exact_bytes * 8 - accounted
        bad += delta != 0
        print(
            f"  {name:7s} R{q256:<5d} {rows}x{columns:<5d} exported="
            f"{unit.exact_bytes * 8:<8d} accounted={accounted:<8d} delta={delta}"
        )
    print(f"  {len(VERIFY_CASES) - bad}/{len(VERIFY_CASES)} exact")
    return bad


#: The quality leg's rungs: the cheapest dominated rung, three inside the
#: region, the last rung below the cap, and the cap itself.
QUALITY_ARMS = (
    ("E2M1x2", 64, 512, (736, 800, 860, 894, 895, 896)),
    ("E2M1", 64, 512, (510, 511, 512)),
)


def quality() -> None:
    import torch

    from tessera.export import encode_linear
    from tessera.unit_artifact import read_unit_artifact

    print("\nrelative SSE of the decoded unit, Gaussian source, weight space:")
    for name, rows, columns, rungs in QUALITY_ARMS:
        grid = grid_for_name(name)
        torch.manual_seed(7)
        weight = torch.randn(rows, columns)
        denom = float((weight.double() ** 2).sum())
        print(f"  {name} {rows}x{columns}")
        for q256 in rungs:
            start = time.time()
            unit = encode_linear(weight, grid=grid, q256=q256)
            decoded = read_unit_artifact(unit.blob).to(torch.float64)
            sse = float(((decoded - weight.double()) ** 2).sum())
            print(
                f"    R{q256:<5d} bytes={unit.exact_bytes:<7d} "
                f"bpp={float(unit.bpp):.4f}  relSSE={sse / denom:.6f}"
                f"   ({time.time() - start:.1f}s)"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=list(GRID_NAMES))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    total = count_table(args.grids, SHAPES)
    print(f"\n{total} dominated rungs over {len(args.grids)} grids x {len(SHAPES)} shapes")
    bad = 0
    if args.verify or args.all:
        bad = verify_bytes()
    if args.quality or args.all:
        quality()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
