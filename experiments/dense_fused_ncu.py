#!/usr/bin/env python3
"""Profile only the dense window launches, after their warmup, under ncu's
CUDA API gate: the fused identity (``routed_fused_kernel`` +
``dense_reduce_kernel``) and the Triton window GEMM (``_window_gemm_kernel``)
over the same resident bundles of the same stub-B modules.

Run via dense_fused_oracle.sh under PrismaBuild with ORACLE_NCU=1.  The
numerical oracle remains dense_fused_oracle.py; this instrument makes no
accuracy claim.
"""
import argparse
import gc
import json
from pathlib import Path

import dense_fused_oracle as dense
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stub", default=dense.STUB)
    parser.add_argument("--modules", default=",".join(dense.MODULES))
    parser.add_argument("--m", default="1,64,512,2048")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=643)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"device": torch.cuda.get_device_name(), "cases": []}
    store = dense.Store(args.stub)
    with dense.init_vllm_world1():
        for module in args.modules.split(","):
            short = f"{module.split('.')[-3]}.{module.split('.')[-1]}"
            for leg, fused in (("fused", True), ("triton", False)):
                layer, method, info = dense.build_served(store, module, mode="resident", tp_rank=0,
                                                         tp_size=1, fused=fused)
                for m in map(int, args.m.split(",")):
                    x = dense.make_x(m, int(layer.tessera_columns), args.seed + m, args.sigma)
                    for _ in range(args.warmup):
                        method.apply(layer, x)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(f"{short}_{leg}_M{m}")
                    torch.cuda.profiler.start()
                    try:
                        method.apply(layer, x)
                        torch.cuda.synchronize()
                    finally:
                        torch.cuda.profiler.stop()
                        torch.cuda.nvtx.range_pop()
                    report["cases"].append({"module": module, "leg": leg, "M": m, "build": info})
                    (out / "ncu-cases.json").write_text(json.dumps(report, indent=2, default=str))
                    del x
                del layer, method
                gc.collect()
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
