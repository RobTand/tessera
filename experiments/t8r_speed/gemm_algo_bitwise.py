"""Which cuBLASLt algorithms reproduce the served BF16 projection GEMMs bit for bit, and how fast are they.

The A8SESHMN prefill chunk (2048 tokens, TP2 rank shapes) spends 171 ms in
BF16 projection GEMMs that run at 75-93% of a 99 TFLOPS estimate. A faster
algorithm is a bitwise lever only if every output element is accumulated in the
same order: the same MMA instruction over K in the same sequence, with no
split-K or stream-K reduction. That property is checked here, not assumed.

Per shape (``out[M, N] = x[M, K] @ w[N, K]^T``, bf16 in and out, fp32 accumulate,
exactly as ``torch.nn.functional.linear`` calls it):

1. ``torch.nn.functional.linear`` gives the reference output, twice (determinism)
   and once under torch.profiler (its kernel names).
2. Candidates: cuBLASLt heuristics (up to 64, 64 MiB workspace), plus an exhaustive
   sweep: every algorithm id x tile x stages x CTA swizzle, with split-K 1, no
   reduction scheme and custom option 0, kept when ``cublasLtMatmulAlgoCheck``
   accepts it.
3. Each candidate runs twice. It is BITWISE when both outputs equal the
   reference bit for bit (int16 view). Then it is timed: 10 back-to-back calls
   between CUDA events, after 3 warm-up calls.
4. Interleaved A/B (reference, best bitwise, best bitwise, reference; 3 rounds)
   on the fastest bitwise candidate, plus board power over a 2 s loop of each.

Inputs are random (x ~ N(0, 1), w ~ N(0, 0.02^2)): accumulation order does not
depend on the values. Writes ``<out>/gemm_algo_bitwise.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

SHAPES = [  # name, N, K, calls per chunk (A8SESHMN rank 0), served ms per chunk (r0)
    ("kda_in_proj", 12576, 4096, 34, 90.08),
    ("kda_o_proj", 4096, 4096, 34, 27.66),
    ("mla_o_proj", 4096, 8192, 11, 20.18),
    ("qa_kva_and_shared_gate_up", 2048, 4096, 44, 18.58),
    ("shared_down", 4096, 1024, 40, 8.80),
    ("q_b", 8192, 1536, 11, 6.13),
]
M = 2048
WS_BYTES = 64 << 20

CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#define LT_CHECK(x) do { cublasStatus_t s_ = (x); \
  TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x " failed: status ", (int)s_); } while (0)

namespace {
cublasLtHandle_t lt() {
  static cublasLtHandle_t h = [] { cublasLtHandle_t x; LT_CHECK(cublasLtCreate(&x)); return x; }();
  return h;
}
struct Descs { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; };
// torch linear: out[M,N] = x[M,K] w[N,K]^T. Column-major view: C(N x M) = op_T(W)(N x K) * X(K x M).
Descs& descs(int64_t M, int64_t N, int64_t K) {
  static std::map<std::tuple<int64_t, int64_t, int64_t>, Descs> cache;
  auto key = std::make_tuple(M, N, K);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Descs d;
  LT_CHECK(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_16BF, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_16BF, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_16BF, N, M, N));
  return cache.emplace(key, d).first->second;
}
torch::Tensor pack(const cublasLtMatmulAlgo_t& algo, size_t ws, float waves, int state) {
  static_assert(sizeof(cublasLtMatmulAlgo_t) == 64, "algo blob is 8 x uint64");
  auto t = torch::zeros({11}, torch::kInt64);
  std::memcpy(t.data_ptr<int64_t>(), &algo, sizeof(algo));
  t[8] = (int64_t)ws; t[9] = (int64_t)(waves * 1000.0f); t[10] = state;
  return t;
}
cublasLtMatmulAlgo_t unpack(const torch::Tensor& t) {
  cublasLtMatmulAlgo_t a;
  std::memcpy(&a, t.contiguous().data_ptr<int64_t>(), sizeof(a));
  return a;
}
std::vector<uint32_t> cap_u32(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
  size_t need = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &need) != CUBLAS_STATUS_SUCCESS || need == 0)
    return {};
  std::vector<uint32_t> v((need + 3) / 4);
  size_t got = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, v.data(), v.size() * 4, &got) != CUBLAS_STATUS_SUCCESS)
    return {};
  v.resize(got / 4);
  return v;
}
}  // namespace

std::vector<torch::Tensor> heuristics(int64_t M, int64_t N, int64_t K, int64_t ws, int64_t max_n) {
  auto& d = descs(M, N, K);
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t wsb = (uint64_t)ws;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb)));
  std::vector<cublasLtMatmulHeuristicResult_t> res((size_t)max_n);
  int got = 0;
  LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt(), d.op, d.a, d.b, d.c, d.c, pref, (int)max_n, res.data(), &got));
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<torch::Tensor> out;
  for (int i = 0; i < got; ++i)
    if (res[i].state == CUBLAS_STATUS_SUCCESS)
      out.push_back(pack(res[i].algo, res[i].workspaceSize, res[i].wavesCount, (int)res[i].state));
  return out;
}

std::vector<torch::Tensor> exhaustive(int64_t M, int64_t N, int64_t K, int64_t ws, int64_t max_n) {
  auto& d = descs(M, N, K);
  std::vector<int> ids(512);
  int n_ids = 0;
  LT_CHECK(cublasLtMatmulAlgoGetIds(lt(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                    CUDA_R_16BF, CUDA_R_16BF, (int)ids.size(), ids.data(), &n_ids));
  std::vector<torch::Tensor> out;
  for (int i = 0; i < n_ids && (int64_t)out.size() < max_n; ++i) {
    cublasLtMatmulAlgo_t base;
    if (cublasLtMatmulAlgoInit(lt(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, ids[i], &base) != CUBLAS_STATUS_SUCCESS)
      continue;
    auto tiles = cap_u32(base, CUBLASLT_ALGO_CAP_TILE_IDS);
    if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    auto stages = cap_u32(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
    if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
    for (uint32_t tile : tiles) {
      for (uint32_t stage : stages) {
        for (uint32_t swz = 0; swz < 2; ++swz) {
          if ((int64_t)out.size() >= max_n) break;
          cublasLtMatmulAlgo_t a = base;
          uint32_t splitk = 1, red = CUBLASLT_REDUCTION_SCHEME_NONE, custom = 0;
          if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage, sizeof(stage)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz)) ||
              cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom)))
            continue;
          cublasLtMatmulHeuristicResult_t r;
          if (cublasLtMatmulAlgoCheck(lt(), d.op, d.a, d.b, d.c, d.c, &a, &r) != CUBLAS_STATUS_SUCCESS) continue;
          if (r.workspaceSize > (size_t)ws) continue;
          out.push_back(pack(a, r.workspaceSize, r.wavesCount, (int)r.state));
        }
      }
    }
  }
  return out;
}

std::vector<int64_t> describe(torch::Tensor t) {
  auto a = unpack(t);
  const cublasLtMatmulAlgoConfigAttributes_t attrs[] = {
      CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
  std::vector<int64_t> v;
  for (auto attr : attrs) {
    size_t need = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&a, attr, nullptr, 0, &need) != CUBLAS_STATUS_SUCCESS ||
        need == 0 || need > 8) { v.push_back(-1); continue; }
    uint64_t buf = 0;
    size_t got = 0;
    v.push_back(cublasLtMatmulAlgoConfigGetAttribute(&a, attr, &buf, need, &got) == CUBLAS_STATUS_SUCCESS
                    ? (int64_t)buf : -1);
  }
  return v;
}

void run(torch::Tensor t, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws) {
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto& d = descs(M, N, K);
  auto a = unpack(t);
  const float alpha = 1.f, beta = 0.f;
  LT_CHECK(cublasLtMatmul(lt(), d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta, out.data_ptr(), d.c,
                          out.data_ptr(), d.c, &a, ws.data_ptr(), (size_t)ws.numel(),
                          at::cuda::getCurrentCUDAStream()));
}
"""

CONFIG_FIELDS = ["algo_id", "tile_id", "splitk_num", "reduction_scheme", "cta_swizzling", "custom_option",
                 "stages_id", "inner_shape_id", "cluster_shape_id"]


def kernel_names(call):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    names = []
    for ev in prof.events():
        if ev.device_type is not None and str(ev.device_type).endswith("CUDA") and ev.name not in names:
            names.append(ev.name[:140])
    return names


def time_calls(call, reps=10, warm=3):
    for _ in range(warm):
        call()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        call()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def power_of(call, seconds, sampler):
    if sampler is None:
        return None
    try:
        return sampler.sample_during(call, seconds)
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-exhaustive", type=int, default=1500)
    ap.add_argument("--shapes", default="all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    ext = load_inline(name="tessera_lt_enum", cpp_sources=[CPP],
                      functions=["heuristics", "exhaustive", "describe", "run"],
                      extra_ldflags=["-lcublasLt"], with_cuda=True, verbose=False)
    sampler = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bench_t8r import PowerSampler
        sampler = PowerSampler()
    except Exception as exc:  # noqa: BLE001
        print("power sampler unavailable:", repr(exc), flush=True)
    meta = dict(host=os.environ.get("HOST_NAME"), image=os.environ.get("ORACLE_IMAGE"),
                pb_action=os.environ.get("PB_ACTION_KEY"), tessera_head=os.environ.get("TESSERA_HEAD"),
                torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                cublaslt_version=None, build_s=round(time.time() - t0, 1), m=M, workspace_bytes=WS_BYTES,
                power_source=getattr(sampler, "source", None),
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        import ctypes
        lib = ctypes.CDLL("libcublasLt.so.13")
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        meta["cublaslt_version"] = int(lib.cublasLtGetVersion())
    except Exception as exc:  # noqa: BLE001
        meta["cublaslt_version"] = repr(exc)
    dev = torch.device("cuda")
    ws = torch.empty(WS_BYTES, dtype=torch.uint8, device=dev)
    results = []
    wanted = None if args.shapes == "all" else set(args.shapes.split(","))
    for name, n, k, calls, served_ms in SHAPES:
        if wanted and name not in wanted:
            continue
        ts = time.time()
        torch.manual_seed(n * 7 + k)
        x = torch.randn(M, k, device=dev).to(torch.bfloat16)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        ref2 = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        ref_call = lambda: torch.nn.functional.linear(x, w)  # noqa: E731
        rec = dict(shape=name, m=M, n=n, k=k, calls_per_chunk=calls, served_ms_per_chunk_r0=served_ms,
                   gflop=2 * M * n * k / 1e9,
                   reference=dict(kernels=kernel_names(ref_call), ms=time_calls(ref_call),
                                  deterministic=bool(torch.equal(ref.view(torch.int16), ref2.view(torch.int16)))))
        cands, seen = [], set()
        for source, algos in (("heuristic", ext.heuristics(M, n, k, WS_BYTES, 64)),
                              ("exhaustive", ext.exhaustive(M, n, k, WS_BYTES, args.max_exhaustive))):
            for rank, t in enumerate(algos):
                key = bytes(t[:8].numpy().tobytes())
                if key in seen:
                    continue
                seen.add(key)
                cands.append((source, rank, t))
        rows = []
        out = torch.empty(M, n, device=dev, dtype=torch.bfloat16)
        for source, rank, t in cands:
            row = dict(source=source, rank=rank, config=dict(zip(CONFIG_FIELDS, ext.describe(t))),
                       workspace=int(t[8]), waves=int(t[9]) / 1000.0)
            try:
                out.fill_(0)
                ext.run(t, x, w, out, ws)
                o1 = out.clone()
                ext.run(t, x, w, out, ws)
                torch.cuda.synchronize()
                row["bitwise"] = bool(torch.equal(o1.view(torch.int16), ref.view(torch.int16))
                                      and torch.equal(out.view(torch.int16), ref.view(torch.int16)))
                row["self_deterministic"] = bool(torch.equal(o1.view(torch.int16), out.view(torch.int16)))
                if not row["bitwise"]:
                    diff = (o1.float() - ref.float()).abs()
                    row["max_abs_diff"] = float(diff.max())
                    row["frac_elems_differ"] = float((o1.view(torch.int16) != ref.view(torch.int16)).float().mean())
                once = time_calls(lambda: ext.run(t, x, w, out, ws), reps=1, warm=0)
                if once > 3 * rec["reference"]["ms"]:
                    row["ms"], row["timed_once"] = once, True  # far slower: not worth 13 calls
                else:
                    row["ms"] = time_calls(lambda: ext.run(t, x, w, out, ws))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["_t"] = t
            rows.append(row)
        ok = [r for r in rows if "ms" in r]
        bitwise = sorted((r for r in ok if r["bitwise"]), key=lambda r: r["ms"])
        anyfast = sorted(ok, key=lambda r: r["ms"])
        # heuristic rank 0 is cuBLASLt's own pick: if it names the reference's kernel
        # yet differs bitwise, the layout here (not the algorithm) is wrong.
        first = [r for r in ok if r["source"] == "heuristic" and r["rank"] == 0]
        for r in first + bitwise[:5] + anyfast[:3]:
            if "kernels" not in r:
                r["kernels"] = kernel_names(lambda t=r["_t"]: ext.run(t, x, w, out, ws))
        rec["candidates"] = len(rows)
        rec["ran"] = len(ok)
        rec["bitwise_count"] = len(bitwise)
        rec["errors"] = sum(1 for r in rows if "error" in r)
        best = bitwise[0] if bitwise else None
        if best is not None:
            ab = {"reference": [], "best_bitwise": []}
            bt = best["_t"]
            for _ in range(3):
                for arm in ("reference", "best_bitwise", "best_bitwise", "reference"):
                    call = ref_call if arm == "reference" else (lambda: ext.run(bt, x, w, out, ws))
                    ab[arm].append(time_calls(call))
            rec["interleaved_ms"] = ab
            rec["power"] = {"reference": power_of(ref_call, 2.0, sampler),
                            "best_bitwise": power_of(lambda: ext.run(bt, x, w, out, ws), 2.0, sampler)}
        for r in rows:
            r.pop("_t", None)
        rec["heuristic_rank0"] = first[0] if first else None
        rec["best_bitwise"] = best
        rec["fastest_any"] = anyfast[0] if anyfast else None
        rec["top_bitwise"] = bitwise[:10]
        rec["top_any"] = anyfast[:10]
        rec["all_rows_summary"] = [dict(source=r["source"], rank=r["rank"], bitwise=r.get("bitwise"),
                                        ms=r.get("ms"), config=r["config"], error=r.get("error"))
                                   for r in rows]
        rec["seconds"] = round(time.time() - ts, 1)
        results.append(rec)
        ref_ms = rec["reference"]["ms"]
        print(json.dumps(dict(shape=name, ref_ms=round(ref_ms, 4), ref_kernels=rec["reference"]["kernels"][:2],
                              candidates=len(rows), bitwise=len(bitwise),
                              best_bitwise_ms=best and round(best["ms"], 4),
                              best_bitwise_kernels=best and best.get("kernels", [])[:2],
                              fastest_any_ms=anyfast and round(anyfast[0]["ms"], 4),
                              fastest_any_bitwise=anyfast and anyfast[0]["bitwise"])), flush=True)
        del x, w, ref, ref2, out
        torch.cuda.empty_cache()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "gemm_algo_bitwise.json"), "w") as fh:
        json.dump(dict(meta=meta, shapes=results), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
