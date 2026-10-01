#!/usr/bin/env python3
"""D1 bench: FlashInfer's SM120 sparse MLA prefill at the GLM-5.3 served shape.

Calls FlashInfer exactly as vLLM's FLASHINFER_MLA_SPARSE_SM120 backend does
(``vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:205-242`` in
image 5be13705): query ``[T, 1, 32, 512]`` bf16, packed ``fp8_ds_mla`` cache
``[blocks, 1, 64, 656]`` uint8, physical indices ``[T, 1, 2176]`` int32,
``seq_lens=None``, ``bmm1_scale=1/16``, ``kv_scale_format="arbitrary_fp32"``.

One chunk of an L8192 request is T=2048 queries at context end E in
{2048, 4096, 6144, 8192}. Index rows follow the kpool indexer
(``vllm/models/glm5next/nvidia/ops/kpool_compress.py:846``): 16 pools of 128
contiguous tokens, then the trailing incomplete pool, padded with -1 to 2176.

Modes:
  default  CUDA-event timing, kernel-only profiler times, power samples,
           stock determinism checks; writes ``d1_timing.json``.
  --ncu    one profiled launch per context between cudaProfilerStart/Stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time

import torch

T = 2048
HEADS = 32
D_LATENT = 512
ROW_BYTES = 656
PAGE = 64
POOL = 128
N_POOLS = 16
TOPK = N_POOLS * POOL  # 2048
WIDTH = 2176  # TOPK + POOL - 1 = 2175, rounded to whole 64-entry tiles
SM_SCALE = 1.0 / 16.0  # (qk_nope 256 + qk_rope 0) ** -0.5
CONTEXTS = (2048, 4096, 6144, 8192)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_cache(n_tokens: int, gen: torch.Generator, device) -> torch.Tensor:
    """Packed fp8_ds_mla rows: 512 e4m3 + 4 fp32 scales (amax/448 per 128) + 128 zero bytes."""
    blocks = (n_tokens + PAGE - 1) // PAGE + 1
    rows = blocks * PAGE
    x = torch.randn(n_tokens, 4, 128, generator=gen, device=device, dtype=torch.float32)
    scale = x.abs().amax(-1).clamp_min(1e-4) / 448.0
    q = (x / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    cache = torch.zeros(rows, ROW_BYTES, dtype=torch.uint8, device=device)
    cache[:n_tokens, :512] = q.reshape(n_tokens, 512).view(torch.uint8)
    cache[:n_tokens, 512:528] = scale.contiguous().view(torch.uint8).reshape(n_tokens, 16)
    return cache.view(blocks, PAGE, ROW_BYTES)


def make_indices(context_end: int, gen: torch.Generator, device, mode: str) -> torch.Tensor:
    """[T, 2176] int32 flat row ids (identity block table) for one prefill chunk."""
    seq = torch.arange(context_end - T + 1, context_end + 1, device=device, dtype=torch.int64)
    out = torch.full((T, WIDTH), -1, dtype=torch.int64, device=device)
    if mode == "random":
        # Uniform keys within each query's causal context, sorted, -1 past it.
        r = torch.rand(T, WIDTH, generator=gen, device=device)
        keys = (r * seq.unsqueeze(1).to(torch.float32)).floor().to(torch.int64)
        keys, _ = keys.sort(dim=1)
        valid = torch.arange(WIDTH, device=device).unsqueeze(0) < seq.unsqueeze(1)
        return torch.where(valid, keys, out).to(torch.int32)
    pool_len = seq // POOL
    tail_start = pool_len * POOL
    tail_count = seq - tail_start
    # Pool choice: all complete pools when there are at most 16; otherwise the
    # 8 most recent plus 8 distinct older pools, ascending ("mostly recent").
    max_pools = int(pool_len.max().item())
    pools = torch.full((T, N_POOLS), -1, dtype=torch.int64, device=device)
    g = torch.arange(N_POOLS, device=device).unsqueeze(0)
    few = pool_len <= N_POOLS
    pools = torch.where(few.unsqueeze(1) & (g < pool_len.unsqueeze(1)), g.expand(T, -1), pools)
    many = ~few
    if bool(many.any()):
        older = torch.rand(T, max(max_pools, 1), generator=gen, device=device)
        col = torch.arange(max(max_pools, 1), device=device).unsqueeze(0)
        older = torch.where(col < (pool_len - 8).unsqueeze(1), older, torch.full_like(older, 2.0))
        pick = older.argsort(dim=1)[:, :8]
        recent = (pool_len - 8).unsqueeze(1) + torch.arange(8, device=device).unsqueeze(0)
        chosen, _ = torch.cat([pick, recent], dim=1).sort(dim=1)
        pools = torch.where(many.unsqueeze(1), chosen, pools)
    hist = pools.unsqueeze(2) * POOL + torch.arange(POOL, device=device).view(1, 1, POOL)
    hist = torch.where(pools.unsqueeze(2) >= 0, hist, torch.full_like(hist, -1)).reshape(T, TOPK)
    out[:, :TOPK] = hist
    toff = torch.arange(WIDTH - TOPK, device=device).unsqueeze(0)
    tail = torch.where(toff < tail_count.unsqueeze(1), tail_start.unsqueeze(1) + toff,
                       torch.full_like(toff, -1))
    out[:, TOPK:] = tail
    return out.to(torch.int32)


def call(fi_mla, q, kv, idx, out, ws):
    return fi_mla(
        query=q.unsqueeze(1),
        kv_cache=kv.view(torch.uint8).unsqueeze(1),
        workspace_buffer=ws,
        qk_nope_head_dim=256,
        kv_lora_rank=D_LATENT,
        qk_rope_head_dim=0,
        block_tables=idx.unsqueeze(1),
        seq_lens=None,
        max_seq_len=idx.shape[1],
        out=out.unsqueeze(1),
        bmm1_scale=SM_SCALE,
        bmm2_scale=1.0,
        sparse_mla_top_k=idx.shape[1],
        kv_scale_format="arbitrary_fp32",
    )


class PowerSampler:
    """nvidia-smi power and SM clock every 100 ms while active."""

    def __init__(self):
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                p, c = (v.strip() for v in r.stdout.strip().split(","))
                self.samples.append((time.time(), float(p), float(c)))
            except Exception:  # noqa: BLE001 - a missing sample is recorded as absent
                pass
            self._stop.wait(0.1)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self):
        if not self.samples:
            return {"n": 0}
        ps = sorted(s[1] for s in self.samples)
        cs = sorted(s[2] for s in self.samples)
        return {"n": len(ps), "power_w_median": ps[len(ps) // 2], "power_w_max": ps[-1],
                "sm_clock_mhz_median": cs[len(cs) // 2], "sm_clock_mhz_min": cs[0]}


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def kernel_times(fn, n: int):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    rows = {}
    for e in prof.events():
        if getattr(e.device_type, "name", "") != "CUDA":
            continue
        rows.setdefault(e.name, []).append(float(e.time_range.elapsed_us()))
    return {k: {"count": len(v), "us_mean": sum(v) / len(v), "us_min": min(v)} for k, v in rows.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--index-mode", default="pools", choices=["pools", "random"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sustain-s", type=float, default=8.0)
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    import flashinfer
    # The exact import vLLM uses (vllm/utils/flashinfer.py:303-307).
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as fi_mla

    contexts = [int(c) for c in args.contexts.split(",")]
    rec = {"schema": "mla_d1_bench.v1", "start_utc": utc(), "host": os.environ.get("HOST_NAME"),
           "flashinfer": flashinfer.__version__, "torch": torch.__version__,
           "device": torch.cuda.get_device_name(), "index_mode": args.index_mode,
           "ncu": args.ncu, "shapes": []}
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    for ctx in contexts:
        gen = torch.Generator(device=dev)
        gen.manual_seed(1000 + ctx)
        kv = make_cache(ctx, gen, dev)
        idx = make_indices(ctx, gen, dev, args.index_mode)
        q = (torch.randn(T, HEADS, D_LATENT, generator=gen, device=dev) * 2.0).to(torch.bfloat16)
        out = torch.empty(T, HEADS, D_LATENT, dtype=torch.bfloat16, device=dev)
        valid = (idx >= 0).sum().item()
        fn = lambda: call(fi_mla, q, kv, idx, out, ws)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        shape = {"context_end": ctx, "valid_index_fraction": valid / idx.numel(),
                 "distinct_rows": int(torch.unique(idx[idx >= 0]).numel())}
        if args.ncu:
            torch.cuda.cudart().cudaProfilerStart()
            fn()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            shape["ncu_profiled_utc"] = utc()
            rec["shapes"].append(shape)
            continue
        # Determinism: three calls, bitwise equal.
        outs = []
        for _ in range(3):
            out.fill_(0)
            fn()
            torch.cuda.synchronize()
            outs.append(out.clone())
        shape["determinism_bitwise"] = all(torch.equal(outs[0].view(torch.int16), o.view(torch.int16)) for o in outs[1:])
        shape["out_sha256"] = sha(outs[0])
        # Split invariance: rows [0,1024) alone against the full batch.
        half = torch.empty(T // 2, HEADS, D_LATENT, dtype=torch.bfloat16, device=dev)
        call(fi_mla, q[: T // 2], kv, idx[: T // 2], half, ws)
        torch.cuda.synchronize()
        shape["split_1024_bitwise"] = bool(torch.equal(half.view(torch.int16), outs[0][: T // 2].view(torch.int16)))
        # CUDA-event timing, one event pair per call.
        ts = []
        for _ in range(args.iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        ts.sort()
        shape["call_ms_median"] = ts[len(ts) // 2]
        shape["call_ms_min"] = ts[0]
        shape["kernels"] = kernel_times(fn, 10)
        # Sustained window for power (and Netdata), back-to-back calls.
        n = 0
        with PowerSampler() as ps:
            shape["sustain_start_utc"] = utc()
            t0 = time.time()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            while time.time() - t0 < args.sustain_s:
                for _ in range(50):
                    fn()
                n += 50
                torch.cuda.synchronize()
            b.record()
            torch.cuda.synchronize()
            shape["sustain_end_utc"] = utc()
        shape["sustain_calls"] = n
        shape["sustain_ms_per_call"] = a.elapsed_time(b) / n
        shape["power"] = ps.summary()
        rec["shapes"].append(shape)
        print(json.dumps({k: shape[k] for k in ("context_end", "call_ms_median", "sustain_ms_per_call",
                                                 "determinism_bitwise", "split_1024_bitwise")}),
              flush=True)
    rec["end_utc"] = utc()
    name = "d1_ncu_marks.json" if args.ncu else f"d1_timing_{args.index_mode}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
