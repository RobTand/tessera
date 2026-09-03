"""Which rungs the 16-bit route's decoder actually reads, by taking them there.

``runtime_contract.json``'s ``reader_rate_range_q256`` is a claim about what
this build's DECODER accepts, and principle 14 says a claim about a runtime is
derived from that runtime or refused.  So it is derived the way the E4M3 range
was (contract v2's changelog): every candidate rung is **encoded**, packed into
a fused container, and taken through the route's own load path --
``parse_tessera_blob_for_scheme`` -> ``shard_parsed_roles`` ->
``prepare_tessera_bf16_module``, including the element-for-element cross-check
against ``tessera.decode.materialize_bf16`` -- and the accepted set is what the
contract may state.  Nothing is inferred from the grammar; the grammar's bound
is printed beside the measurement so the two can be compared.

The rung gate itself is bypassed for the sweep (it reads the very range this
script produces, so leaving it in would make the answer circular): the sweep
asserts the *decoder*, and the gate then refuses anything outside what the
sweep found.

Rate and rung: on an arity-1 grid the rung ``q256`` is the body bits per 256
weights, so the per-position rate is ``q256 / 256``.  BF16 carries 16 payload
bits, so the grammar's shaped domain is rate 1..16, i.e. q256 256..4096 -- and
``export._window_bits_for`` widens the window table to ``max(14, rate)`` above
R = 14 rather than refusing, which is one of the things this sweep checks
actually survives a load.

Needs CUDA: the encode is the fused window Viterbi (the reference at L = 14 on
CPU is minutes a rung).

Run::

    TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      PYTHONPATH=src python experiments/bf16_reader_rate_range.py
    ... --rungs 256,512,...     # default: every integer rate plus the
                                #   between-rungs a Bresenham schedule makes
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from tessera import fused
from tessera.alphabet import BF16_GRID
from tessera.export import encode_linear_planes, wire_recipe
from tessera.serving import scheme as sch
from tessera.serving.bf16_route import prepare_tessera_bf16_module


def _default_rungs():
    """Every integer rate 1..16, plus five non-integer rungs between them.

    The rate axis is continuous (a Bresenham schedule mixes two adjacent
    rates down a column), so a range that only ever saw the round numbers
    would be a range nobody had tested at the rungs an allocator picks.
    """
    integers = [256 * r for r in range(1, 17)]
    between = [300, 777, 1000, 1500, 2500]
    return sorted(set(integers + between))


def try_rung(q256: int, rows: int, cols: int, device) -> tuple[bool, str]:
    torch.manual_seed(q256)
    w = (torch.randn(rows, cols, device=device)
         * torch.linspace(0.2, 3.0, rows, device=device)[:, None]).float()
    try:
        exported, _unit, _forests = encode_linear_planes(
            w.contiguous(), grid=BF16_GRID, q256=q256, name="weight", verify=False)
    except Exception as exc:                                     # noqa: BLE001
        return False, f"encode: {type(exc).__name__}: {exc}"
    blob = fused.pack_fused([("weight", rows, exported.blob)])
    scheme = {
        "family": sch.TESSERA_BF16, "structure": "dense", "grid": "BF16", "body": "WINDOW",
        "plane": "CHANNEL", "q256": q256, "rows": rows, "columns": cols,
        "wire_bytes": len(blob), "roles": [["weight", rows]],
    }
    try:
        # The rung gate reads the range this script is producing, so it is
        # bypassed here and only here.  Everything else in the load path runs.
        real = sch._refuse_an_unreadable_rung
        sch._refuse_an_unreadable_rung = lambda *a, **k: None
        try:
            roles = sch.parse_tessera_blob_for_scheme(blob, scheme, "sweep")
        finally:
            sch._refuse_an_unreadable_rung = real
        prepared = prepare_tessera_bf16_module(roles, device=device)
        tile = prepared.decode()
        if tuple(tile.shape) != (rows, cols) or tile.dtype != torch.bfloat16:
            return False, f"decoded {tuple(tile.shape)} {tile.dtype}"
    except Exception as exc:                                     # noqa: BLE001
        return False, f"load: {type(exc).__name__}: {exc}"
    return True, f"L={wire_recipe(BF16_GRID, q256).window_bits} wire={len(blob)}B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA: the encode is the fused window Viterbi")
        return 2
    rungs = ([int(r) for r in args.rungs.split(",")] if args.rungs else _default_rungs())
    device = torch.device("cuda")

    print(f"grid BF16 (arity 1, {BF16_GRID.payload_bits} payload bits), "
          f"{args.rows}x{args.cols}")
    print(f"grammar's shaped domain: rate 1..{BF16_GRID.payload_bits} "
          f"=> q256 [256, {256 * BF16_GRID.payload_bits}]")
    print()
    head = f"{'q256':>6s} {'rate':>6s} {'read':>6s}  detail"
    print(head)
    print("-" * (len(head) + 20))
    results = {}
    for q256 in rungs:
        ok, detail = try_rung(q256, args.rows, args.cols, device)
        results[q256] = {"read": ok, "detail": detail}
        print(f"{q256:6d} {q256 / 256:6.2f} {str(ok):>6s}  {detail}")
    accepted = sorted(q for q, r in results.items() if r["read"])
    print()
    if accepted:
        print(f"ACCEPTED q256 in [{min(accepted)}, {max(accepted)}], "
              f"{len(accepted)} of {len(rungs)} candidates")
        gaps = [q for q in rungs if min(accepted) <= q <= max(accepted)
                and not results[q]["read"]]
        print(f"gaps inside that span: {gaps or 'none -- the span is continuous'}")
    else:
        print("ACCEPTED: nothing")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": args.rows, "cols": args.cols,
                       "results": {str(k): v for k, v in results.items()}}, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
