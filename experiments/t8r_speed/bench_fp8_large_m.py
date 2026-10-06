"""Why the E4M3 scaled GEMM slows 4-6x at M*K >= 2^25 on GB10 (tessera#931 note).

At the two shapes that slowed (kda_in 12448x4096, MLA o_proj 4096x8192) and
M = 2048, 4096, 8192, time:

* ``rowwise``: ``torch._scaled_mm`` with row-wise scales (the lane), plus the
  kernels it launched;
* ``tensorwise``: the same GEMM with scalar scales (another cuBLASLt path);
* ``chunked``: row-wise in 2048-row slices into one output (the activation
  stays L2-sized per call);
* ``quant``: vLLM's per-token quantiser alone;
* ``bf16``: ``torch.mm`` reference.

Each cell is run forward, then the whole list again in reverse, so a thermal
drift shows as a forward/reverse disagreement.  Seeded random E4M3 operands.

Usage: bench_fp8_large_m.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

SHAPES = {"kda_in": (12448, 4096), "mla_o_proj": (4096, 8192)}   # (N, K)
MS = (2048, 4096, 8192)
CHUNK = 2048


def time_ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return sorted({e.name[:120] for e in prof.events() if e.device_type.name == "CUDA"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact", default=None, help="accepted for bench_t8r.sh; unused")
    args = ap.parse_args()
    import vllm._custom_ops  # noqa: F401 -- registers torch.ops._C's quantiser
    from tessera.serving.native_ops import native_fp8_quant

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    cells = []
    data = {}
    for name, (n, k) in SHAPES.items():
        w8 = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        wbf = torch.randn(n, k, device="cuda").bfloat16()
        sw = torch.rand(1, n, device="cuda") + 0.5
        for m in MS:
            x = torch.randn(m, k, device="cuda").bfloat16()
            xq, sa = native_fp8_quant(x)
            sa = sa.reshape(m, 1).float().contiguous()
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            one = torch.ones((), device="cuda")
            xq_t = xq.contiguous()
            data[(name, m)] = dict(x=x, xq=xq_t, sa=sa, out=out)

            def rowwise(xq=xq_t, sa=sa, w8=w8, sw=sw):
                return torch._scaled_mm(xq, w8.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)

            def tensorwise(xq=xq_t, w8=w8, one=one):
                return torch._scaled_mm(xq, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

            def chunked(xq=xq_t, sa=sa, w8=w8, sw=sw, out=out, m=m):
                for r0 in range(0, m, CHUNK):
                    r1 = min(m, r0 + CHUNK)
                    out[r0:r1] = torch._scaled_mm(xq[r0:r1], w8.t(), scale_a=sa[r0:r1], scale_b=sw,
                                                  out_dtype=torch.bfloat16)

            def quant(x=x):
                return native_fp8_quant(x)

            def bf16(x=x, wbf=wbf):
                return torch.mm(x, wbf.t())

            for arm, fn in (("rowwise", rowwise), ("tensorwise", tensorwise), ("chunked", chunked),
                            ("quant", quant), ("bf16", bf16)):
                cells.append({"shape": name, "n": n, "k": k, "m": m, "arm": arm, "fn": fn})
    for cell in cells:
        cell["fwd_ms"] = time_ms(cell["fn"])
    for cell in reversed(cells):
        cell["rev_ms"] = time_ms(cell["fn"])
        cell["ms"] = (cell["fwd_ms"] + cell["rev_ms"]) / 2
        if cell["arm"] in ("rowwise", "tensorwise"):
            cell["kernels"] = kernels(cell["fn"])
    rows = []
    for cell in cells:
        flops = 2.0 * cell["m"] * cell["n"] * cell["k"]
        row = {k: v for k, v in cell.items() if k != "fn"}
        if cell["arm"] != "quant":
            row["tflops"] = flops / (cell["ms"] * 1e-3) / 1e12
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    meta = {"host": os.environ.get("HOST_NAME"), "head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "pb_action": os.environ.get("PB_ACTION_KEY"),
            "torch": torch.__version__, "end_unix": time.time()}
    with open(os.path.join(args.out, "fp8_large_m.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
