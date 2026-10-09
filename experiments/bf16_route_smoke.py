"""Exercise one T16 unit from encode through the wire and paired decode.

The smoke reports canonical FP32 effective weights and the derived BF16 stock
tile separately. The serving operand is the raw BF16 value with its row scale.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import BF16_GRID, E4M3_GRID  # noqa: E402
from tessera.bf16_route import prepare_bf16_unit, stream_bf16  # noqa: E402
from tessera.decode import materialize_bf16, reconstruct_unit  # noqa: E402
from tessera.export import BF16_WINDOW_BITS, encode_linear_planes, wire_recipe  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--q256", type=int, nargs="+", default=[1536])
    ap.add_argument("--window-bits", type=int, default=BF16_WINDOW_BITS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    weight = (
        torch.randn(args.rows, args.cols, device=args.device)
        * torch.linspace(0.2, 3.0, args.rows, device=args.device)[:, None]
    ).float()
    out = {"args": vars(args), "recipe": str(wire_recipe(BF16_GRID, args.q256[0])), "rungs": {}}
    for q256 in args.q256:
        t0 = time.time()
        exported, unit, forests = encode_linear_planes(
            weight, grid=BF16_GRID, q256=q256, name="smoke",
            window_bits=args.window_bits,
        )
        encode_secs = time.time() - t0
        recovered = read_unit_artifact(exported.blob, device=args.device)
        reference = reconstruct_unit(unit, forests, None)
        parsed = parse_unit_artifact(exported.blob, device=args.device)
        values, scale = materialize_bf16(parsed.unit, parsed.grid, parsed.code)
        streamed_values, streamed_scale = stream_bf16(prepare_bf16_unit(parsed.unit))
        effective = values.float() * scale[:, None]
        stock_tile = effective.to(torch.bfloat16)
        row = {
            "bpp": float(exported.bpp),
            "bytes": exported.exact_bytes,
            "encode_secs": encode_secs,
            "wire_equals_encoder": bool(torch.equal(recovered, reference)),
            "streamed_equals_values": bool(torch.equal(streamed_values, values)),
            "streamed_equals_scale": bool(torch.equal(streamed_scale, scale)),
            "paired_equals_canonical": bool(torch.equal(effective, recovered)),
            "rel_err_fp32": float((reference - weight).norm() / weight.norm()),
            "rel_err_derived_stock_bf16": float((stock_tile.float() - weight).norm() / weight.norm()),
        }
        out["rungs"][q256] = row
        print(f"q256={q256} " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
