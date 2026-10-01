"""Cost of a DYNAMIC per-token A-side global against the served STATIC one.

The served NVFP4 A side (``e2m1_group16_ue4m3_static``) quantises with one
calibrated scalar global per GEMM input.  A per-token global needs one extra
reduction -- the row amax -- before the block scales exist, and a per-row
epilogue multiply after the GEMM.  This times, per (M, K), in CUDA-graph
replay so launch overhead is excluded:

* ``quant``   vLLM's ``scaled_fp4_quant`` at a scalar global (the served op);
* ``amax``    the extra row-amax reduction alone, as a separate kernel
              (``torch.amax`` over |x|): an UPPER bound on the added cost,
              since a fused quantiser reads the row once for both;
* ``both``    the two back to back, the unfused dynamic A side.

    python3 experiments/t4_code/aside_cost.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from tessera.kernel_a4 import a4_quantize_activation


def graph_time(fn, iters=200, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    best = float("inf")
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best * 1000.0   # us per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default="1,16,512,2048,8192")
    ap.add_argument("--ks", default="4096,1024,8192,12288")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rows = []
    for k in [int(v) for v in a.ks.split(",")]:
        for m in [int(v) for v in a.ms.split(",")]:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            gs = torch.tensor([2688.0 / float(x.abs().max())], device="cuda", dtype=torch.float32)
            amax = torch.empty(m, device="cuda", dtype=torch.bfloat16)

            def quant():
                a4_quantize_activation(x, gs)

            def red():
                torch.amax(x.abs(), dim=-1, out=amax)

            def both():
                torch.amax(x.abs(), dim=-1, out=amax)
                a4_quantize_activation(x, gs)

            r = {"m": m, "k": k, "quant_us": graph_time(quant), "amax_us": graph_time(red),
                 "both_us": graph_time(both)}
            r["dynamic_overhead_frac"] = (r["both_us"] - r["quant_us"]) / r["quant_us"]
            r["x_bytes"] = m * k * 2
            rows.append(r)
            print(json.dumps(r), flush=True)
    meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "image": os.environ.get("ORACLE_IMAGE"), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "host": os.environ.get("HOST_NAME"), "end_unix": time.time()}
    (out / "aside_cost.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
