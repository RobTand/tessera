"""MLA absorbed BMMs of GLM-5.3-Flash prefill: candidate kernels against the served call.

Stock vLLM runs, per MLA layer and 2048-token chunk (TP2 rank, 32 heads):

* ``uk``: ``torch.bmm(q_nope[h,T,256], W_UK_T[h,256,512], out=ql_nope.transpose(0,1))``
  (``mla_attention.py`` ``forward_impl``) -- cuBLAS picks a 32x32 WMMA kernel,
  about 18 TFLOP/s on GB10 (bench_proj_gemm.py, screen3);
* ``uv``: ``torch.bmm(x[h,T,512], W_UV[h,512,256], out=out.transpose(0,1))``
  (``_v_up_proj``) -- about 32 TFLOP/s.

Arms per product, each checked bitwise against the served call:

* ``served``: the stock call on the served strides (inputs and outputs are
  head-major views of token-major tensors; the weight's batch stride is 2*K*N);
* ``w_colmajor``: the same call with the weight stored column-major per head
  (a load-time layout change only);
* ``triton``: a strided batched Triton GEMM (below) reading the same views and
  writing the token-major output in place, over a small config sweep.  It walks
  K in order with one fp32 accumulator per output, as every non-split-K cuBLAS
  kernel on sm_121 does, which is why it can be bitwise equal to the stock call.

Usage: bench_mla_bmm.py --out DIR [--tokens 2048] [--reps 7 --iters 40]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import triton
import triton.language as tl

HEADS = 32
PRODUCTS = {"uk": (256, 512), "uv": (512, 256)}  # (K, N)


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
CONFIGS = [
    (128, 128, 32, 4, 4), (128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3),
    (256, 128, 32, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 256, 64, 8, 2),
    (64, 256, 32, 4, 4), (128, 128, 32, 8, 4),
]


@triton.jit
def _strided_bmm_kernel(x, w, out, M, N, K,
                        sxb, sxm, sxk, swb, swk, swn, sob, som, son,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    npn = tl.cdiv(N, BN)
    rm = (pid // npn) * BM + tl.arange(0, BM)
    rn = (pid % npn) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x + b * sxb + rm[:, None] * sxm + rk[None, :] * sxk
    wp = w + b * swb + rk[:, None] * swk + rn[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(xp, mask=(rm[:, None] < M) & (rk[None, :] < K - k0), other=0.0)
        bt = tl.load(wp, mask=(rk[:, None] < K - k0) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, bt, acc)
        xp += BK * sxk
        wp += BK * swk
    op = out + b * sob + rm[:, None] * som + rn[None, :] * son
    tl.store(op, acc.to(out.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def strided_bmm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, config=CONFIGS[0]) -> torch.Tensor:
    """``out[b] = x[b] @ w[b]`` for 3-D bf16 tensors of any strides; returns ``out``."""
    batch, m, k = x.shape
    n = w.shape[2]
    if w.shape != (batch, k, n) or out.shape != (batch, m, n):
        raise ValueError(f"strided_bmm shapes x{tuple(x.shape)} w{tuple(w.shape)} out{tuple(out.shape)}")
    bm, bn, bk, warps, stages = config
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), batch)
    _strided_bmm_kernel[grid](x, w, out, m, n, k, *x.stride(), *w.stride(), *out.stride(),
                              BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def time_calls(fn, reps, iters):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        per.append(s.elapsed_time(e) / iters)
    out = {"ms": statistics.median(per), "ms_min": min(per), "ms_max": max(per)}
    try:
        out["power_w"] = torch.cuda.power_draw() / 1000.0
        out["sm_clock_mhz"] = torch.cuda.clock_rate()
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        out["device_err"] = repr(exc)
    return out


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})


def operands(kind, tokens):
    k, n = PRODUCTS[kind]
    x_tok = torch.randn(tokens, HEADS, k, device="cuda", dtype=torch.bfloat16)
    out_tok = torch.empty(tokens, HEADS, n, device="cuda", dtype=torch.bfloat16)
    if kind == "uk":
        w = (torch.randn(HEADS, 2 * k, n, device="cuda", dtype=torch.bfloat16) * k**-0.5)[:, :k]
    else:
        w = (torch.randn(HEADS, 2 * n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5)[:, :n].transpose(1, 2)
    return x_tok, w, out_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--artifact", default=None, help="accepted for bench_t8r.sh; unused")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    meta = {"host": os.environ.get("HOST_NAME"), "head": os.environ.get("TESSERA_HEAD"),
            "state": os.environ.get("TESSERA_STATE"), "image": os.environ.get("ORACLE_IMAGE"),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(),
            "tokens": args.tokens, "heads": HEADS, "start_unix": time.time()}
    print(json.dumps({"meta": meta}), flush=True)
    res = {}
    for kind, (k, n) in PRODUCTS.items():
        flops = 2.0 * HEADS * args.tokens * k * n
        x_tok, w, out_tok = operands(kind, args.tokens)
        xv, ov = x_tok.transpose(0, 1), out_tok.transpose(0, 1)
        r = {"K": k, "N": n, "strides": {"x": list(xv.stride()), "w": list(w.stride()), "out": list(ov.stride())}}

        def served():
            torch.bmm(xv, w, out=ov)

        r["served"] = time_calls(served, args.reps, args.iters)
        r["served"]["kernels"] = kernel_names(served)
        served()
        ref = out_tok.clone()

        w_cm = w.transpose(1, 2).contiguous().transpose(1, 2)
        o2 = torch.empty_like(out_tok)

        def colmajor():
            torch.bmm(xv, w_cm, out=o2.transpose(0, 1))

        colmajor()
        r["w_colmajor"] = time_calls(colmajor, args.reps, args.iters)
        r["w_colmajor"]["kernels"] = kernel_names(colmajor)
        r["w_colmajor"]["bitwise_equal_served"] = bool(torch.equal(o2, ref))

        rows = []
        for cfg in CONFIGS:
            o3 = torch.full_like(out_tok, float("nan"))
            try:
                strided_bmm(xv, w, o3.transpose(0, 1), config=cfg)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 - e.g. shared memory over sm_121's limit
                rows.append({"cfg": cfg, "error": repr(exc)[:200]})
                continue
            row = {"cfg": cfg, "bitwise_equal_served": bool(torch.equal(o3, ref)),
                   "max_abs_vs_served": float((o3.float() - ref.float()).abs().max())}
            row.update(time_calls(lambda: strided_bmm(xv, w, o3.transpose(0, 1), config=cfg), args.reps, args.iters))
            rows.append(row)
        r["triton_rows"] = rows
        best = min((x for x in rows if "ms" in x), key=lambda x: x["ms"], default=None)
        r["triton_best"] = best
        for key in ("served", "w_colmajor"):
            r[key]["tflops"] = flops / (r[key]["ms"] * 1e-3) / 1e12
        if best:
            best["tflops"] = flops / (best["ms"] * 1e-3) / 1e12
        res[kind] = r
        print(json.dumps({"product": kind, "served_ms": r["served"]["ms"], "served_kernels": r["served"]["kernels"],
                          "colmajor_ms": r["w_colmajor"]["ms"], "colmajor_bitwise": r["w_colmajor"]["bitwise_equal_served"],
                          "colmajor_kernels": r["w_colmajor"]["kernels"],
                          "triton_best": best}), flush=True)
    meta["end_unix"] = time.time()
    with open(os.path.join(args.out, "mla_bmm.json"), "w") as f:
        json.dump({"meta": meta, "products": res}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
