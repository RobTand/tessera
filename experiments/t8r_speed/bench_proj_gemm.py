"""BF16 projection GEMMs of GLM-5.3-Flash prefill at the served shapes, on one GB10.

The A8SE752VB serve (L8192 c1, TP2, 2048-token chunks) spends about 155-159 ms
per chunk per rank in BF16 projection GEMMs, all on cuBLASLt's default
heuristic pick (``nvjet_sm121_tst_mma_*`` two-stage kernels for the large
shapes, ``cutlass_80_*`` for some others).  This script prices the levers:

* ``default``: ``torch.mm(x, w.t())``, exactly what the serve runs (F.linear);
* ``lt``: every algorithm cuBLASLt's heuristic returns for the same TN problem
  (up to ``--lt-algos``), each timed, checked against ``default`` (bitwise and
  max abs), with its tile, stages and split-K read back from the algo config;
* ``triton``: a bf16 Triton matmul over a small config sweep (no autotune
  cache; each config timed);
* ``fp8``: ``torch._scaled_mm`` E4M3 x E4M3 -> bf16, row-wise scales (the
  W8A8 speed ceiling of a quantized projection), plus the per-token activation
  quantiser it needs, timed separately;
* ``concat``: same-input projections fused into one GEMM (MLA ``q_a+kv_a`` with
  the two indexer-K projections; ``q_b`` with indexer-Q) against the sum of
  their separate default calls.

Shapes are the per-rank M = 2048 shapes from the A8SE752VB rank-0 trace, with
the calls per chunk the trace recorded.  Each timed cell rotates over enough
weight copies to exceed L2, runs back-to-back (the serve's regime), and
reports the median per-call time of ``--reps`` windows; GPU power and SM clock
are sampled after each cell.

Usage: bench_proj_gemm.py --out DIR [--shapes kda_in,...] [--lt-algos 32]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

M = 2048
# name: (K, N, calls per 2048-token chunk per rank, role)
SHAPES = {
    "kda_in": (4096, 12576, 34, "KDA fused input proj (q,k,v,gates)"),
    "kda_o": (4096, 4096, 34, "KDA o_proj"),
    "kda_aux": (128, 4096, 68, "KDA low-rank b-proj pair (K=128)"),
    "mla_qa_kva": (4096, 2048, 11, "MLA fused q_a+kv_a"),
    "mla_qb": (1536, 8192, 11, "MLA q_b"),
    "mla_idx_q": (1536, 4096, 11, "MLA indexer q"),
    "mla_idx_k160": (4096, 160, 11, "MLA indexer k (160)"),
    "mla_idx_k128": (4096, 128, 11, "MLA indexer k (128)"),
    "mla_o": (8192, 4096, 11, "MLA o_proj"),
    "shared_gate_up": (4096, 2048, 33, "shared-expert gate_up (bf16 layers)"),
    "shared_down": (1024, 4096, 40, "shared-expert down (bf16 layers)"),
}
CONCATS = {
    "mla_a_side": ["mla_qa_kva", "mla_idx_k160", "mla_idx_k128"],
    "mla_qb_side": ["mla_qb", "mla_idx_q"],
}
BF16_PEAK = 123.1e12  # mma.sync m16n8k16 bf16, measured (2026-09-30-fp8-prefill-roofline.md)
E4M3_PEAK = 246.5e12
L2_BYTES = 64 << 20  # rotate past this so weights stream from DRAM as in the serve

LT_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>

#define CK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x, " -> ", (int)s_); } while (0)

struct Problem {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
};
static std::vector<Problem> problems;
static void* workspace = nullptr;
static size_t workspace_bytes = 0;

// Row-major C[M,N] = X[M,K] @ W[N,K]^T, posed column-major as C^T = W * X^T:
// A = W (K x N col-major, transposed), B = X (K x M col-major), C (N x M).
int64_t lt_setup(int64_t m, int64_t n, int64_t k, int64_t max_algos, int64_t ws_bytes) {
  if ((size_t)ws_bytes > workspace_bytes) {
    if (workspace) cudaFree(workspace);
    TORCH_CHECK(cudaMalloc(&workspace, ws_bytes) == cudaSuccess);
    workspace_bytes = ws_bytes;
  }
  Problem p;
  CK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  CK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, k, n, k));
  CK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k));
  CK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n));
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = ws_bytes;
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  p.algos.resize(max_algos);
  int found = 0;
  auto h = at::cuda::getCurrentCUDABlasLtHandle();
  CK(cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, p.b, p.c, p.c, pref, max_algos, p.algos.data(), &found));
  p.algos.resize(found);
  cublasLtMatmulPreferenceDestroy(pref);
  problems.push_back(p);
  return (int64_t)problems.size() - 1;
}

int64_t lt_count(int64_t pid) { return (int64_t)problems.at(pid).algos.size(); }

std::vector<int64_t> lt_config(int64_t pid, int64_t i) {
  const auto& r = problems.at(pid).algos.at(i);
  auto get = [&](cublasLtMatmulAlgoConfigAttributes_t a) -> int64_t {
    uint64_t v = 0; size_t w = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&r.algo, a, &v, sizeof(v), &w) != CUBLAS_STATUS_SUCCESS) return -1;
    if (w == 4) return (int64_t)(uint32_t)v;
    if (w == 2) return (int64_t)(uint16_t)v;
    return (int64_t)v;
  };
  return {get(CUBLASLT_ALGO_CONFIG_ID), get(CUBLASLT_ALGO_CONFIG_TILE_ID), get(CUBLASLT_ALGO_CONFIG_STAGES_ID),
          get(CUBLASLT_ALGO_CONFIG_SPLITK_NUM), get(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
          get(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING), get(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
          (int64_t)r.workspaceSize, (int64_t)(r.wavesCount * 1000)};
}

void lt_run(int64_t pid, int64_t i, torch::Tensor x, torch::Tensor w, torch::Tensor out) {
  auto& p = problems.at(pid);
  float alpha = 1.f, beta = 0.f;
  auto h = at::cuda::getCurrentCUDABlasLtHandle();
  CK(cublasLtMatmul(h, p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, out.data_ptr(), p.c,
                    out.data_ptr(), p.c, &p.algos.at(i).algo, workspace, workspace_bytes,
                    at::cuda::getCurrentCUDAStream()));
}
"""


def build_lt():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="proj_gemm_lt",
        cpp_sources="",
        cuda_sources=LT_SRC,
        functions=["lt_setup", "lt_count", "lt_config", "lt_run"],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )


def sample_device():
    out = {}
    try:
        out["power_w"] = torch.cuda.power_draw() / 1000.0
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        out["power_err"] = repr(exc)
    try:
        out["sm_clock_mhz"] = torch.cuda.clock_rate()
    except Exception as exc:  # noqa: BLE001
        out["clock_err"] = repr(exc)
    return out


def time_calls(fn, n_copies, reps, iters):
    """Median per-call ms over ``reps`` windows of ``iters`` back-to-back calls."""
    for i in range(max(8, n_copies)):
        fn(i % n_copies)
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(iters):
            fn(i % n_copies)
        e.record()
        e.synchronize()
        per.append(s.elapsed_time(e) / iters)
    dev = sample_device()
    return {"ms": statistics.median(per), "ms_min": min(per), "ms_max": max(per), **dev}


def copies_for(nbytes):
    return max(2, -(-L2_BYTES // max(nbytes, 1)) + 1)


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})


def tflops(k, n, ms):
    return 2.0 * M * k * n / (ms * 1e-3) / 1e12


def bench_default(k, n, args):
    nc = copies_for(n * k * 2)
    xs = [torch.randn(M, k, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5 for _ in range(nc)]
    t = time_calls(lambda i: torch.mm(xs[i % 2], ws[i].t()), nc, args.reps, args.iters)
    t["kernels"] = kernel_names(lambda: torch.mm(xs[0], ws[0].t()))
    return t, xs, ws


def bench_lt(lt, k, n, xs, ws, args):
    ref = torch.mm(xs[0], ws[0].t())
    pid = lt.lt_setup(M, n, k, args.lt_algos, args.lt_workspace_mb << 20)
    rows = []
    for i in range(lt.lt_count(pid)):
        cfg = lt.lt_config(pid, i)
        row = dict(zip(["algo_id", "tile", "stages", "splitk", "reduction", "swizzle", "custom",
                        "workspace", "waves_x1000"], cfg))
        out = torch.empty(M, n, device="cuda", dtype=torch.bfloat16)
        try:
            lt.lt_run(pid, i, xs[0], ws[0], out)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - an algo the device refuses is a row, not a crash
            row["error"] = repr(exc)
            rows.append(row)
            continue
        row["bitwise_equal_default"] = bool(torch.equal(out, ref))
        row["max_abs_vs_default"] = float((out.float() - ref.float()).abs().max())
        row.update(time_calls(lambda j: lt.lt_run(pid, i, xs[j % 2], ws[j], out), len(ws), args.reps, args.iters))
        row["tflops"] = tflops(k, n, row["ms"])
        rows.append(row)
    best = min((r for r in rows if "ms" in r), key=lambda r: r["ms"], default=None)
    if best is not None:
        i = rows.index(best)
        out = torch.empty(M, n, device="cuda", dtype=torch.bfloat16)
        best["kernels"] = kernel_names(lambda: lt.lt_run(pid, i, xs[0], ws[0], out))
    return rows, best


def triton_matmul():
    import triton
    import triton.language as tl

    @triton.jit
    def kern(a, b, c, m, n, k, sam, sak, sbn, sbk, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
        pid = tl.program_id(0)
        npm, npn = tl.cdiv(m, BM), tl.cdiv(n, BN)
        group = GM * npn
        gid = pid // group
        first = gid * GM
        gsz = min(npm - first, GM)
        pm = first + ((pid % group) % gsz)
        pn = (pid % group) // gsz
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        ap = a + rm[:, None] * sam + rk[None, :] * sak
        bp = b + rn[None, :] * sbn + rk[:, None] * sbk
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in range(0, tl.cdiv(k, BK)):
            am = (rm[:, None] < m) & (rk[None, :] < k - kk * BK)
            bm = (rn[None, :] < n) & (rk[:, None] < k - kk * BK)
            acc += tl.dot(tl.load(ap, mask=am, other=0.0), tl.load(bp, mask=bm, other=0.0))
            ap += BK * sak
            bp += BK * sbk
        cp = c + rm[:, None] * scm + rn[None, :] * scn
        tl.store(cp, acc.to(tl.bfloat16), mask=(rm[:, None] < m) & (rn[None, :] < n))

    def run(x, w, out, cfg):
        bm, bn, bk, warps, stages = cfg
        k, n = x.shape[1], w.shape[0]
        grid = (triton.cdiv(M, bm) * triton.cdiv(n, bn),)
        kern[grid](x, w, out, M, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                   out.stride(0), out.stride(1), BM=bm, BN=bn, BK=bk, GM=8,
                   num_warps=warps, num_stages=stages)

    return run


TRITON_CFGS = [
    (128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 256, 64, 8, 3), (256, 128, 64, 8, 3),
    (128, 128, 32, 4, 4), (128, 256, 32, 8, 4), (256, 128, 32, 8, 4), (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4), (128, 128, 64, 4, 4), (128, 192, 64, 8, 3), (192, 128, 64, 8, 3),
]


def bench_triton(k, n, xs, ws, args):
    run = triton_matmul()
    ref = torch.mm(xs[0], ws[0].t())
    rows = []
    for cfg in TRITON_CFGS:
        row = {"cfg": list(cfg)}
        out = torch.empty(M, n, device="cuda", dtype=torch.bfloat16)
        try:
            run(xs[0], ws[0], out, cfg)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - e.g. shared memory over sm_121's limit
            row["error"] = repr(exc)[:200]
            rows.append(row)
            continue
        row["max_abs_vs_default"] = float((out.float() - ref.float()).abs().max())
        row.update(time_calls(lambda j: run(xs[j % 2], ws[j], out, cfg), len(ws), args.reps, args.iters))
        row["tflops"] = tflops(k, n, row["ms"])
        rows.append(row)
    best = min((r for r in rows if "ms" in r), key=lambda r: r["ms"], default=None)
    return rows, best


def quant_rowwise(x):
    amax = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / 448.0
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def bench_fp8(k, n, xs, args):
    if n % 16 or k % 16:
        return {"skipped": "scaled_mm needs K and N multiples of 16"}
    nc = copies_for(n * k)
    wq = [torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn) for _ in range(nc)]
    ws = torch.ones(1, n, device="cuda", dtype=torch.float32)
    xq, xs_scale = quant_rowwise(xs[0])
    out = {}
    try:
        t = time_calls(lambda i: torch._scaled_mm(xq, wq[i].t(), scale_a=xs_scale, scale_b=ws,
                                                  out_dtype=torch.bfloat16), nc, args.reps, args.iters)
        t["tflops"] = tflops(k, n, t["ms"])
        t["kernels"] = kernel_names(lambda: torch._scaled_mm(xq, wq[0].t(), scale_a=xs_scale, scale_b=ws,
                                                             out_dtype=torch.bfloat16))
        out["gemm_rowwise"] = t
    except Exception as exc:  # noqa: BLE001
        out["gemm_rowwise"] = {"error": repr(exc)[:300]}
    try:
        from vllm import _custom_ops as ops

        q = lambda i: ops.scaled_fp8_quant(xs[i % 2], use_per_token_if_dynamic=True)  # noqa: E731
        out["quant_vllm_per_token"] = time_calls(q, 2, args.reps, args.iters)
    except Exception as exc:  # noqa: BLE001
        out["quant_vllm_per_token"] = {"error": repr(exc)[:300]}
    out["quant_torch_ops"] = time_calls(lambda i: quant_rowwise(xs[i % 2]), 2, args.reps, max(5, args.iters // 4))
    return out


def bench_concat(names, results, args):
    k = SHAPES[names[0]][0]
    assert all(SHAPES[s][0] == k for s in names), names
    n = sum(SHAPES[s][1] for s in names)
    t, _, _ = bench_default(k, n, args)
    t["tflops"] = tflops(k, n, t["ms"])
    separate = sum(results[s]["default"]["ms"] for s in names)
    calls = SHAPES[names[0]][2]
    return {"members": names, "K": k, "N": n, "fused": t, "separate_ms": separate,
            "saved_ms_per_call": separate - t["ms"], "saved_ms_per_chunk": (separate - t["ms"]) * calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lt-algos", type=int, default=32)
    ap.add_argument("--lt-workspace-mb", type=int, default=32)
    ap.add_argument("--skip", default="", help="comma list of: lt,triton,fp8,concat")
    ap.add_argument("--artifact", default=None, help="accepted for bench_t8r.sh; unused")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    meta = {"host": os.environ.get("HOST_NAME"), "head": os.environ.get("TESSERA_HEAD"),
            "state": os.environ.get("TESSERA_STATE"), "image": os.environ.get("ORACLE_IMAGE"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()),
            "sms": torch.cuda.get_device_properties(0).multi_processor_count,
            "start_unix": time.time(), "args": vars(args)}
    print(json.dumps({"meta": meta}), flush=True)
    lt = None if "lt" in skip else build_lt()
    results = {}
    for name in args.shapes.split(","):
        k, n, calls, role = SHAPES[name]
        r = {"K": k, "N": n, "calls_per_chunk": calls, "role": role}
        d, xs, ws = bench_default(k, n, args)
        d["tflops"] = tflops(k, n, d["ms"])
        r["default"] = d
        if lt is not None:
            rows, best = bench_lt(lt, k, n, xs, ws, args)
            r["lt_rows"], r["lt_best"] = rows, best
        if "triton" not in skip:
            rows, best = bench_triton(k, n, xs, ws, args)
            r["triton_rows"], r["triton_best"] = rows, best
        if "fp8" not in skip:
            r["fp8"] = bench_fp8(k, n, xs, args)
        del xs, ws
        torch.cuda.empty_cache()
        results[name] = r
        summ = {"shape": name, "default_ms": round(d["ms"], 4), "default_tflops": round(d["tflops"], 1),
                "default_kernels": d["kernels"], "power_w": d.get("power_w"), "clock": d.get("sm_clock_mhz")}
        if r.get("lt_best"):
            b = r["lt_best"]
            summ.update(lt_best_ms=round(b["ms"], 4), lt_best_tflops=round(b["tflops"], 1),
                        lt_best_bitwise=b["bitwise_equal_default"], lt_best_kernels=b.get("kernels"),
                        lt_n=len(r["lt_rows"]))
        if r.get("triton_best"):
            summ.update(triton_best_ms=round(r["triton_best"]["ms"], 4), triton_best_cfg=r["triton_best"]["cfg"])
        g = r.get("fp8", {}).get("gemm_rowwise", {})
        if "ms" in g:
            summ.update(fp8_ms=round(g["ms"], 4), fp8_tflops=round(g["tflops"], 1))
        print(json.dumps(summ), flush=True)
    concats = {}
    if "concat" not in skip:
        for cname, members in CONCATS.items():
            if all(m in results for m in members):
                concats[cname] = bench_concat(members, results, args)
                print(json.dumps({"concat": cname, **{k2: v for k2, v in concats[cname].items() if k2 != "fused"},
                                  "fused_ms": concats[cname]["fused"]["ms"]}), flush=True)
    meta["end_unix"] = time.time()
    with open(os.path.join(args.out, "proj_gemm.json"), "w") as f:
        json.dump({"meta": meta, "shapes": results, "concats": concats,
                   "peaks": {"bf16_mma_sync": BF16_PEAK, "e4m3_mma_sync": E4M3_PEAK}}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
