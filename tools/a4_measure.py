"""Bounded A4 measurement: resident bytes, prep time, kernel time.

One real expert's wires (the same TP2 rank shards the gate uses), measured on
the serve image.  This is a development screen on L2-resident tiles: the
numbers are per-expert kernel costs and per-rank residency projections, not a
serving throughput claim (that needs a rotating working set and grouped
execution, and the acceptance contract makes speed a later objective than
correctness).

Usage (inside the serve image):
    python3 tools/a4_measure.py --report /tmp/a4-measure.json [--iters 50]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get("PYTHONPATH", "src"))

DATA = Path(os.environ.get("TESSERA_A4_WIRE_DIR", "/mnt/shared/astra-native-a4/data"))
PREFIX = "model.language_model.layers.3.mlp.experts"
LAYER = "layers_3"
TP = 2
MOE_LAYERS = 42
EXPERTS = 288


def declared():
    from tessera.serving.scheme import validate_tessera_moe_scheme

    cfg = json.loads((DATA / "a4-config.json").read_text())
    scheme = cfg["quantization_config"]["config_groups"][
        f"tessera_model_language_model_{LAYER}_mlp_experts"]["scheme"]
    return validate_tessera_moe_scheme(scheme, PREFIX)


def time_call(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()

    from tessera.kernel_a4 import (A4Unit, a4_decode_span2_tile, a4_quantize_activation,
                                   a4_span2_gemm, require_native_fp4_mma)
    from tessera.lane_planes import prepare_span2_planes
    from tessera.serving.moe_route import _packed_group_shard_plan
    from tessera.serving.scheme import expert_role_declarations, parse_tessera_expert_blob
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.stock import materialize_stock

    require_native_fp4_mma("a4_measure")
    report = {"rank": args.rank, "iters": args.iters, "roles": {}}
    blobs = {name: (DATA / f"{name}_wire.bin").read_bytes()
             for name in ("gate_proj", "up_proj", "down_proj")}
    scheme = declared()
    roles13 = expert_role_declarations(scheme["groups"]["w13"])
    roles2 = expert_role_declarations(scheme["groups"]["w2"])
    plan13 = _packed_group_shard_plan(scheme, "w13", PREFIX, args.rank, TP)
    plan2 = _packed_group_shard_plan(scheme, "w2", PREFIX, args.rank, TP)
    parsed13 = [parse_tessera_expert_blob(blobs["gate_proj"], roles13[0], f"{PREFIX} gate",
                                          device="cpu")[0],
                parse_tessera_expert_blob(blobs["up_proj"], roles13[1], f"{PREFIX} up",
                                          device="cpu")[0]]
    parsed2 = [parse_tessera_expert_blob(blobs["down_proj"], roles2[0], f"{PREFIX} down",
                                         device="cpu")[0]]

    plane_names = ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
                   "subset_nibbles", "code_nibbles")
    total_native = 0
    total_stock = 0
    for group, parsed_roles, plan in (("w13", parsed13, plan13), ("w2", parsed2, plan2)):
        sharded = shard_parsed_roles(parsed_roles, plan)
        for name, parsed in sharded:
            t0 = time.perf_counter()
            prepared = prepare_span2_planes(parsed, device="cuda")
            unit = A4Unit.from_prepared(prepared)
            torch.cuda.synchronize()
            prep_ms = (time.perf_counter() - t0) * 1e3
            native_bytes = sum(getattr(unit, plane).numel() * getattr(unit, plane).element_size()
                               for plane in plane_names)
            stock = materialize_stock(parsed.unit, parsed.forests, parsed.code)
            stock_bytes = (stock["weight_packed"].numel() * stock["weight_packed"].element_size()
                           + stock["weight_scale"].numel() * stock["weight_scale"].element_size())
            total_native += native_bytes
            total_stock += stock_bytes
            entry = {"rows": unit.rows, "cols": unit.cols,
                     "native_plane_bytes": int(native_bytes),
                     "stock_tile_bytes": int(stock_bytes),
                     "prep_ms": prep_ms,
                     "gemm_ms": {}}
            torch.manual_seed(0)
            for m in (1, 8, 32, 128):
                x = torch.randn(m, unit.cols, dtype=torch.bfloat16, device="cuda") * 0.25
                gscale = torch.tensor([448.0 * 6.0 / max(float(x.abs().max()), 1e-6)],
                                      dtype=torch.float32, device="cuda")
                packed, scales = a4_quantize_activation(x, gscale)
                entry["gemm_ms"][str(m)] = time_call(
                    lambda packed=packed, scales=scales, unit=unit, gscale=gscale:
                    a4_span2_gemm(packed, scales, unit, gscale, out_dtype=torch.float32),
                    args.iters)
            # decode-time bandwidth: one expert's planes are read once per forward
            entry["wire_read_bytes_per_call"] = int(native_bytes)
            report["roles"][f"{group}.{name}"] = entry

    report["per_rank_projection"] = {
        "native_bytes": int(total_native * EXPERTS * MOE_LAYERS),
        "stock_tile_bytes": int(total_stock * EXPERTS * MOE_LAYERS),
        "note": ("extrapolated over 288 experts x 42 routed-MoE layers per rank "
                 "from one real expert; the artifact's experts share this geometry"),
    }
    print("MEASURE " + json.dumps(report))
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
