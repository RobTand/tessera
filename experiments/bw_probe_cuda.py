"""The bandwidth denominator, measured: a plain CUDA streaming reader and copier.

The previous receipt left 103 GB/s (three torch/Triton probes) against 205 GB/s
(``torch._scaled_mm`` on a cold tile) unresolved on the same box.  This probe is
a CUDA kernel with the shape a bandwidth-bound GEMV has: many blocks, 16-byte
vector loads, a grid-stride loop, one value retired per block.  Working set
>= 256 MB so nothing is L2-resident.  Min of rounds, power sampled beside it.

Usage: python experiments/bw_probe_cuda.py [--out json]
"""
import argparse
import json
import os
import subprocess
import threading
import time

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/rob/tmp/torch-ext-gemv")

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void read_kernel(const uint4* __restrict__ src, long n_vec, float* __restrict__ out) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n_vec; i += stride) {
        uint4 v = __ldg(src + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEF) out[blockIdx.x] = 1.0f;   // never true; keeps the loads alive
}

template <int UNROLL>
__global__ void read_kernel_unrolled(const uint4* __restrict__ src, long n_vec, float* __restrict__ out) {
    long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    unsigned acc = 0;
    long i = tid;
    for (; i + (UNROLL - 1) * stride < n_vec; i += UNROLL * stride) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = __ldg(src + i + u * stride);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n_vec; i += stride) { uint4 v = __ldg(src + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0xDEADBEEF) out[blockIdx.x] = 1.0f;
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, long n_vec) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n_vec; i += stride) dst[i] = __ldg(src + i);
}

void read_bytes(torch::Tensor src, torch::Tensor out, int blocks, int threads) {
    long n_vec = src.numel() / 16;
    read_kernel<<<blocks, threads>>>((const uint4*)src.data_ptr(), n_vec, out.data_ptr<float>());
}

void read_bytes_unrolled(torch::Tensor src, torch::Tensor out, int blocks, int threads, int unroll) {
    long n_vec = src.numel() / 16;
    if (unroll == 4) read_kernel_unrolled<4><<<blocks, threads>>>((const uint4*)src.data_ptr(), n_vec, out.data_ptr<float>());
    else read_kernel_unrolled<8><<<blocks, threads>>>((const uint4*)src.data_ptr(), n_vec, out.data_ptr<float>());
}

void copy_bytes(torch::Tensor src, torch::Tensor dst, int blocks, int threads) {
    long n_vec = src.numel() / 16;
    copy_kernel<<<blocks, threads>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), n_vec);
}
"""


class Power:
    def __init__(self, period=0.1):
        self.samples = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout
                p, c = o.strip().split(",")
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                       "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=5).stdout
                n = len([l for l in apps.splitlines() if l.strip()])
                self.samples.append((time.time(), float(p), int(c), n))
            except Exception:
                pass
            self.stop.wait(0.1)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(2)

    def window(self, t0, t1):
        rows = [s for s in self.samples if t0 <= s[0] <= t1]
        if not rows:
            return {}
        return {"mean_w": round(sum(r[1] for r in rows) / len(rows), 1),
                "max_w": max(r[1] for r in rows), "sm_mhz_mean": round(sum(r[2] for r in rows) / len(rows)),
                "cuda_procs_max": max(r[3] for r in rows), "cuda_procs_min": min(r[3] for r in rows)}


def timeit(fn, iters=5, rounds=7):
    fn(); torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        best.append(s.elapsed_time(e) / iters * 1e3)
    best.sort()
    return {"us": round(best[0], 1), "median_us": round(best[len(best) // 2], 1), "max_us": round(best[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--mb", type=int, default=256)
    args = ap.parse_args()
    ext = load_inline("bw_probe",
                      cpp_sources="#include <torch/extension.h>\nvoid read_bytes(torch::Tensor, torch::Tensor, int, int);\nvoid copy_bytes(torch::Tensor, torch::Tensor, int, int);\nvoid read_bytes_unrolled(torch::Tensor, torch::Tensor, int, int, int);\n",
                      cuda_sources=SRC,
                      functions=["read_bytes", "copy_bytes", "read_bytes_unrolled"],
                      extra_cuda_cflags=["-O3", "-gencode", "arch=compute_121,code=sm_121"],
                      verbose=False)
    res = {"spec_gb_s": 273.0, "rows": {}}
    out = torch.zeros(8192, device="cuda")
    with Power() as pw:
      for mb in [12, 26, 64, args.mb]:
        n = mb << 20
        nbuf = max(1, int(256 // mb))
        bufs = [torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda") for _ in range(nbuf)]
        dst = torch.empty_like(bufs[0])
        k = [0]
        def nxt():
            b = bufs[k[0] % nbuf]; k[0] += 1
            return b
        configs = [(48 * 2, 256, 1), (48 * 4, 256, 1), (48 * 8, 256, 1), (48 * 16, 256, 1), (48 * 4, 512, 1),
                   (48 * 4, 256, 4), (48 * 8, 256, 4), (48 * 4, 512, 4), (48 * 2, 1024, 4), (48 * 4, 256, 8), (48 * 8, 256, 8)]
        for blocks, threads, unroll in configs:
            t0 = time.time()
            if unroll == 1:
                r = timeit(lambda: ext.read_bytes(nxt(), out, blocks, threads), iters=nbuf, rounds=9)
            else:
                r = timeit(lambda: ext.read_bytes_unrolled(nxt(), out, blocks, threads, unroll), iters=nbuf, rounds=9)
            t1 = time.time()
            r["gb_s"] = round(n / (r["us"] * 1e-6) / 1e9, 1)
            key = f"read {mb}MB x{nbuf} {blocks}x{threads} u{unroll}"
            res["rows"][key] = {**r, **pw.window(t0, t1)}
            print(f"{key:40s}: {r['us']:8.1f} us  {r['gb_s']:6.1f} GB/s  {pw.window(t0, t1)}", flush=True)
        t0 = time.time()
        c = timeit(lambda: ext.copy_bytes(nxt(), dst, 48 * 8, 256), iters=nbuf, rounds=9)
        c["gb_s_moved"] = round(2 * n / (c["us"] * 1e-6) / 1e9, 1)
        key = f"copy {mb}MB x{nbuf} 384x256"
        res["rows"][key] = {**c, **pw.window(t0, time.time())}
        print(f"{key:40s}: {c['us']:8.1f} us  {c['gb_s_moved']:6.1f} GB/s moved", flush=True)
      src = bufs[0]; n = src.numel()
      if True:
        # torch reference arms
        t0 = time.time()
        cl = timeit(lambda: src.clone())
        cl["gb_s_moved"] = round(2 * n / (cl["us"] * 1e-6) / 1e9, 1)
        res["rows"]["torch clone"] = {**cl, **pw.window(t0, time.time())}
        print("torch clone", cl)
        # _scaled_mm on a cold 5120x5120 fp8 tile, M=1 (the number the last receipt saw at 205 GB/s)
        tiles = []
        for _ in range(8):
            w = torch.randint(0, 200, (5120, 5120), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
            ws = torch.rand(5120, 1, device="cuda") * 0.01
            tiles.append((w, ws))
        x = torch.randn(1, 5120, device="cuda").to(torch.float8_e4m3fn)
        xs = torch.ones(1, 1, device="cuda")
        i = [0]

        def mm():
            w, ws = tiles[i[0] % 8]; i[0] += 1
            return torch._scaled_mm(x, w.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
        t0 = time.time()
        r = timeit(mm, iters=8, rounds=9)
        r["gb_s"] = round(5120 * 5120 / (r["us"] * 1e-6) / 1e9, 1)
        res["rows"]["scaled_mm 5120x5120 M=1 cold"] = {**r, **pw.window(t0, time.time())}
        print("scaled_mm 5120x5120 M=1 cold", r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
