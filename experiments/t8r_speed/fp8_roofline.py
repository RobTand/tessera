"""FP8 against f16 tensor-core rates, their accumulation, and memory bandwidth on one GB10.

The fused routed window kernel widens its E4M3 operands to f16 and runs
``mma.sync.m16n8k16.f32.f16.f16.f32``.  An FP8 path would run
``mma.sync.m16n8k32.f32.e4m3.e4m3.f32`` on the same bytes.  This script
measures, on the device it runs on:

* ``peak``: each instruction's throughput with register-resident operands
  (every SM, independent accumulator chains), the ceiling any kernel built
  on it can reach; plus bf16 m16n8k16 for reference;
* ``gemm``: cuBLAS rates at routed-like shapes, ``torch._scaled_mm`` (e4m3 x
  e4m3 -> bf16) and ``torch.mm`` (bf16), dense;
* ``bandwidth``: device read and copy bandwidth, the memory side of the roofline;
* ``probes``: how each instruction accumulates -- whether a small addend
  survives next to a large product inside one instruction (``inner``), and
  whether the fp32 accumulator input is added exactly (``chain``) and rounded
  to nearest or toward zero (``round``);
* ``error``: both instructions on the same e4m3 operands (random, per-row
  scaled to the e4m3 range) against the exact fp64 dot product, at K from
  one expert's intermediate slice to the hidden size, chained through fp32
  accumulators the way the fused kernel chains them.

Usage: fp8_roofline.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

// KIND 0: f16 m16n8k16, 1: bf16 m16n8k16, 2: e4m3 m16n8k32; f32 accumulate.
template <int KIND>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    if constexpr (KIND == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (KIND == 1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

template <int KIND, int CHAINS>
__global__ void peak_kernel(const uint32_t* __restrict__ seed, float* __restrict__ out, int iters) {
    uint32_t a[4], b[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = seed[(threadIdx.x + i) & 63];
    b[0] = seed[(threadIdx.x + 7) & 63];
    b[1] = seed[(threadIdx.x + 11) & 63];
    float d[CHAINS][4];
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c)
        #pragma unroll
        for (int i = 0; i < 4; ++i) d[c][i] = 0.f;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) mma<KIND>(d[c], a, b);
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c)
        #pragma unroll
        for (int i = 0; i < 4; ++i) s += d[c][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// D[M, N] = C[M, N] + A[M, K] x B[K, N], one warp per 16 x 8 tile, K in
// steps of the instruction's k, fp32 accumulators chained across steps (the
// fused kernel's order).  A is row-major e4m3 bytes, B is [N, K] e4m3 bytes
// (column-major B); KIND 0 widens both to f16 first (exact), KIND 2 feeds
// the bytes to the e4m3 instruction.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
template <int KIND>
__global__ void tile_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                            const float* __restrict__ C, float* __restrict__ D, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int tm = blockIdx.y, tn = blockIdx.x;
    const int r = lane >> 2, q = lane & 3;
    const int row0 = tm * 16 + r, row1 = row0 + 8, col = tn * 8 + r;   // B's n for the fragment
    float d[4];
    const int ncol0 = tn * 8 + 2 * q;
    d[0] = C[(long)row0 * N + ncol0]; d[1] = C[(long)row0 * N + ncol0 + 1];
    d[2] = C[(long)row1 * N + ncol0]; d[3] = C[(long)row1 * N + ncol0 + 1];
    constexpr int KS = (KIND == 2) ? 32 : 16;
    for (int k0 = 0; k0 < K; k0 += KS) {
        uint32_t a[4], b[2];
        if constexpr (KIND == 2) {
            // a0 (r, 4q..4q+3), a1 (r+8, ..), a2 (r, 16+4q..), a3 (r+8, 16+4q..); b0 (4q.., n=r), b1 (16+4q.., n=r)
            a[0] = *reinterpret_cast<const uint32_t*>(A + (long)row0 * K + k0 + 4 * q);
            a[1] = *reinterpret_cast<const uint32_t*>(A + (long)row1 * K + k0 + 4 * q);
            a[2] = *reinterpret_cast<const uint32_t*>(A + (long)row0 * K + k0 + 16 + 4 * q);
            a[3] = *reinterpret_cast<const uint32_t*>(A + (long)row1 * K + k0 + 16 + 4 * q);
            b[0] = *reinterpret_cast<const uint32_t*>(B + (long)col * K + k0 + 4 * q);
            b[1] = *reinterpret_cast<const uint32_t*>(B + (long)col * K + k0 + 16 + 4 * q);
        } else {
            // a0 (r, 2q..2q+1), a1 (r+8, ..), a2 (r, 8+2q..), a3 (r+8, 8+2q..); b0 (2q.., n=r), b1 (8+2q.., n=r)
            auto h = [](const uint8_t* p) { return e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(p)); };
            a[0] = h(A + (long)row0 * K + k0 + 2 * q);
            a[1] = h(A + (long)row1 * K + k0 + 2 * q);
            a[2] = h(A + (long)row0 * K + k0 + 8 + 2 * q);
            a[3] = h(A + (long)row1 * K + k0 + 8 + 2 * q);
            b[0] = h(B + (long)col * K + k0 + 2 * q);
            b[1] = h(B + (long)col * K + k0 + 8 + 2 * q);
        }
        mma<KIND>(d, a, b);
    }
    D[(long)row0 * N + ncol0] = d[0]; D[(long)row0 * N + ncol0 + 1] = d[1];
    D[(long)row1 * N + ncol0] = d[2]; D[(long)row1 * N + ncol0 + 1] = d[3];
}

double peak(int64_t kind, int64_t chains, int64_t blocks, int64_t threads, int64_t iters, torch::Tensor seed, torch::Tensor out) {
    auto s = reinterpret_cast<const uint32_t*>(seed.data_ptr<int32_t>());
    float* o = out.data_ptr<float>();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto run = [&]() {
#define K_(KI, CH) if (kind == KI && chains == CH) peak_kernel<KI, CH><<<blocks, threads>>>(s, o, (int)iters);
        K_(0, 2) K_(0, 4) K_(0, 8) K_(1, 2) K_(1, 4) K_(1, 8) K_(2, 2) K_(2, 4) K_(2, 8)
#undef K_
    };
    run(); cudaDeviceSynchronize();
    cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "peak kernel failed");
    return ms;
}

void tile(int64_t kind, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D) {
    const int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK(M % 16 == 0 && N % 8 == 0 && K % 32 == 0, "M % 16, N % 8, K % 32");
    dim3 grid(N / 8, M / 16);
    auto a = A.data_ptr<uint8_t>(); auto b = B.data_ptr<uint8_t>();
    if (kind == 0) tile_kernel<0><<<grid, 32>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>(), M, N, K);
    else tile_kernel<2><<<grid, 32>>>(a, b, C.data_ptr<float>(), D.data_ptr<float>(), M, N, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tile kernel failed");
}
"""
CPP_SRC = """
double peak(int64_t kind, int64_t chains, int64_t blocks, int64_t threads, int64_t iters, torch::Tensor seed, torch::Tensor out);
void tile(int64_t kind, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D);
"""
KINDS = {0: ("f16.m16n8k16", 16 * 8 * 16), 1: ("bf16.m16n8k16", 16 * 8 * 16), 2: ("e4m3.m16n8k32", 16 * 8 * 32)}


def build():
    from torch.utils.cpp_extension import load_inline
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}"
    return load_inline(name=f"fp8_roofline_sm{arch}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                       functions=["peak", "tile"], verbose=False,
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"])


def peaks(lib):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    seed = torch.randint(0x3C003C00 - 5, 0x3C003C00 + 5, (64,), dtype=torch.int32, device="cuda")
    res = {}
    for kind, (name, macs) in KINDS.items():
        best = None
        for chains in (2, 4, 8):
            for per_sm, threads in ((1, 256), (2, 256), (4, 128), (2, 512)):
                blocks = sms * per_sm
                out = torch.empty(blocks * threads, dtype=torch.float32, device="cuda")
                iters = 8192 // chains
                ms = lib.peak(kind, chains, blocks, threads, iters, seed, out)
                warps = blocks * threads // 32
                tflops = 2.0 * macs * warps * iters * chains / (ms * 1e-3) / 1e12
                cfg = {"chains": chains, "blocks": blocks, "threads": threads, "ms": ms, "TFLOPS": tflops}
                if best is None or tflops > best["TFLOPS"]:
                    best = cfg
        res[name] = best
        print(json.dumps({"peak": name, **best}), flush=True)
    return res


def gemms():
    res = {}
    shapes = [(2048, 4096, 2048), (2048, 1024, 4096), (8192, 4096, 2048), (8192, 1024, 4096), (4096, 4096, 4096)]
    for m, k, n in shapes:
        a16 = torch.randn(m, k, device="cuda").bfloat16()
        b16 = torch.randn(n, k, device="cuda").bfloat16()
        rec = {}
        t = _events(lambda: torch.mm(a16, b16.t()))
        rec["bf16_mm_TFLOPS"] = 2.0 * m * k * n / (t * 1e-3) / 1e12
        a8 = a16.to(torch.float8_e4m3fn)
        b8 = b16.to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")
        try:
            t = _events(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
            rec["e4m3_scaled_mm_TFLOPS"] = 2.0 * m * k * n / (t * 1e-3) / 1e12
        except Exception as exc:  # noqa: BLE001
            rec["e4m3_scaled_mm_error"] = repr(exc)[:300]
        res[f"{m}x{k}x{n}"] = rec
        print(json.dumps({"gemm": f"{m}x{k}x{n}", **rec}), flush=True)
    return res


def _events(call, warmup=3, iters=10):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); call(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b))
    out.sort()
    return out[len(out) // 2]


def bandwidth():
    n = 4 << 30
    x = torch.empty(n // 4, dtype=torch.int32, device="cuda")
    x.random_(0, 1 << 20)
    y = torch.empty_like(x)
    t_read = _events(lambda: torch.sum(x, dtype=torch.int64))
    t_copy = _events(lambda: y.copy_(x))
    del x, y
    torch.cuda.empty_cache()
    res = {"bytes": n, "read_GBps": n / (t_read * 1e-3) / 1e9, "copy_GBps_rw": 2 * n / (t_copy * 1e-3) / 1e9}
    print(json.dumps({"bandwidth": res}), flush=True)
    return res


E4M3_ONE = 0x38   # 1.0: sign 0, exponent 7 (bias 7), mantissa 0


def _e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.to(torch.float8_e4m3fn).view(torch.uint8)


def _run(lib, kind, A, B, C):
    D = torch.empty_like(C)
    lib.tile(kind, A, B, C, D)
    torch.cuda.synchronize()
    return D


def probes(lib):
    """16 x K by K x 8 tiles; row 0 / column 0 carries the probe, the rest zero."""
    res = {}
    for kind in (0, 2):
        name = KINDS[kind][0]
        K = 32
        out = {}

        def one(prods, c0=0.0):
            # A[0, i] = a_i, B[i, 0] (stored as B[0, i]) = b_i with a_i * b_i = prods[i]
            A = torch.zeros(16, K, device="cuda")
            Bm = torch.zeros(8, K, device="cuda")
            for i, (a, b) in enumerate(prods):
                A[0, i], Bm[0, i] = a, b
            C = torch.zeros(16, 8, device="cuda")
            C[0, 0] = c0
            return float(_run(lib, kind, _e4m3(A), _e4m3(Bm), C)[0, 0])

        def pair(e):
            # two normal e4m3 factors (exponents in [-6, 8]) whose product is 2^e
            e1 = max(-6, min(8, e // 2))
            return 2.0 ** e1, 2.0 ** (e - e1)

        big = 2.0 ** 16
        # inner: products 2^16 and 2^(16-t) in one instruction; 1.0 = the small
        # product survived exactly (exact fp32 keeps t <= 23; t = 24 is a tie)
        inner = {}
        for t in range(1, 29):
            got = one([(256.0, 256.0), pair(16 - t)])
            inner[t] = (got - big) * 2.0 ** (t - 16)
        out["inner_2^16_plus_2^(16-t)_scaled"] = inner
        # chain: accumulator input 2^16 plus one product 2^(16-t)
        chain = {}
        for t in range(1, 29):
            got = one([pair(16 - t)], c0=big)
            chain[t] = (got - big) * 2.0 ** (t - 16)
        out["chain_2^16_plus_2^(16-t)_scaled"] = chain
        # rounding: 2^24 + 3 -> RN 2^24 + 4, RZ 2^24 + 2
        out["round_2^24_plus_3_minus_2^24"] = one([(1.0, 1.0), (1.0, 2.0)], c0=2.0 ** 24) - 2.0 ** 24
        # negative side: -(2^24) - 3 -> RN -4, RZ -2
        out["round_neg_2^24_minus_3_plus_2^24"] = one([(-1.0, 1.0), (-1.0, 2.0)], c0=-(2.0 ** 24)) + 2.0 ** 24
        # cancellation inside one instruction: 2^16 - 2^16 + 2^-8 (exact = 2^-8,
        # as is any sequential fp32 sum in this order; 0 = the products were
        # aligned to the largest exponent and the small one fell off)
        out["cancel_2^16_minus_2^16_plus_2^-8"] = one([(256.0, 256.0), (-256.0, 256.0), (2.0 ** -4, 2.0 ** -4)])
        # subnormal operands: 2^-9 x 2^-9 = 2^-18 alone
        out["subnormal_2^-9_squared_times_2^18"] = one([(2.0 ** -9, 2.0 ** -9)]) * 2.0 ** 18
        res[name] = out
        print(json.dumps({"probe": name, **{k: v for k, v in out.items()}}), flush=True)
    return res


def errors(lib):
    """Random e4m3 operands: per-row activations scaled to the e4m3 range, weights
    ~N(0, 1) in e4m3; both instructions against fp64, normalised by sum |a_i b_i|."""
    res = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    for K in (1024, 2048, 4096, 8192):
        M, N = 256, 256
        x = torch.randn(M, K, device="cuda", generator=g) * torch.exp(torch.randn(M, 1, device="cuda", generator=g))
        x = x * (448.0 / x.abs().amax(1, keepdim=True))
        w = torch.randn(N, K, device="cuda", generator=g)
        A, B = _e4m3(x), _e4m3(w)
        a64 = A.view(torch.float8_e4m3fn).double()
        b64 = B.view(torch.float8_e4m3fn).double()
        exact = a64 @ b64.t()
        mag = a64.abs() @ b64.abs().t()
        C = torch.zeros(M, N, device="cuda")
        rec = {}
        outs = {}
        for kind in (0, 2):
            name = KINDS[kind][0]
            d = _run(lib, kind, A, B, C).double()
            outs[kind] = d
            err = (d - exact).abs() / mag.clamp_min(1e-300)
            ulp = (d - exact).abs() / exact.abs().clamp_min(1e-300)
            rec[name] = {"max_err_over_mag": float(err.max()), "mean_err_over_mag": float(err.mean()),
                         "max_rel": float(ulp.max()), "median_rel": float(ulp.median()),
                         "exact_fraction": float(((d - exact) == 0).double().mean())}
        diff = (outs[2] - outs[0]).abs() / mag.clamp_min(1e-300)
        rec["e4m3_vs_f16"] = {"bitwise_fraction": float((outs[2] == outs[0]).double().mean()),
                              "max_diff_over_mag": float(diff.max())}
        res[str(K)] = rec
        print(json.dumps({"error_K": K, **rec}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "cuda": torch.version.cuda, "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "pb_action": os.environ.get("PB_ACTION_KEY"),
            "host": os.environ.get("HOST_NAME"), "start_unix": time.time()}
    lib = build()
    result = {"meta": meta}
    for name, fn in (("probes", lambda: probes(lib)), ("errors", lambda: errors(lib)),
                     ("peak", lambda: peaks(lib)), ("gemm", gemms), ("bandwidth", bandwidth)):
        try:
            result[name] = fn()
        except Exception as exc:  # noqa: BLE001
            result[name] = {"error": repr(exc)[:500]}
            print(json.dumps({name: "error", "detail": repr(exc)[:500]}), flush=True)
    meta["end_unix"] = time.time()
    with open(os.path.join(args.out, "fp8_roofline.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main()
