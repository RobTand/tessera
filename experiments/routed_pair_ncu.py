#!/usr/bin/env python3
"""Profile only native grouped GEMMs, after their warmup, using ncu's CUDA API gate.

Run via routed_pair_action.sh under PrismaBuild with ORACLE_NCU=1. The numerical
oracle remains routed_pair_oracle.py; this instrument makes no accuracy claim.
"""
import argparse
import gc
import json
from pathlib import Path

import routed_pair_oracle as oracle
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wire-root", default=oracle.WIRE_ROOT)
    parser.add_argument("--scales", default=oracle.SCALES)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--families", default="e4m3,bf16,e2m1")
    parser.add_argument("--m", default="1,64,512")
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--clamp", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=604)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rung", default=None, help="wire rung for every selected family (tessera#694)")
    args = parser.parse_args()
    if args.rung:
        for key in args.families.split(","):
            oracle.FAMILIES[key] = dict(oracle.FAMILIES[key], rung=str(args.rung))
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"device": torch.cuda.get_device_name(), "cases": []}
    experts = list(range(args.experts))
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts"
    for key in args.families.split(","):
        family = oracle.FAMILIES[key]
        blobs, _ = oracle.load_wires(args, family, experts)
        scales = None
        if family["family"] == "TESSERA_NVFP4":
            scales, _ = oracle.load_input_scales(args, experts)
        scheme, _ = oracle.scheme_for(family, blobs, len(experts))
        config = oracle.moe_config(len(experts), args.clamp)
        layer, method, info = oracle.build_after(
            family, scheme, blobs, scales, len(experts), config, args.clamp, prefix)
        del blobs
        for m in map(int, args.m.split(",")):
            x, ids, weights = oracle.make_inputs(m, len(experts), args.seed + m, args.sigma)
            for _ in range(args.warmup):
                method.apply(layer, x, weights, ids, None, None)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{key}_M{m}")
            torch.cuda.profiler.start()
            try:
                method.apply(layer, x, weights, ids, None, None)
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
                torch.cuda.nvtx.range_pop()
            report["cases"].append({"family": key, "M": m, "build": info})
            (out / "ncu-cases.json").write_text(json.dumps(report, indent=2))
            del x, ids, weights
        del layer, method
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
