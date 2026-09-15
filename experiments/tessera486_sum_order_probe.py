#!/usr/bin/env python3
"""Probe: is torch's CUDA float32 full ``sum`` the order this replica claims?

tessera#486 stage 2 needs ``_lut_cost``'s ``(w * g * g).sum()`` bit for bit
without launching torch's reduction per trial.  This replica follows
``ATen/native/cuda/Reduce.cuh`` (torch 2.11) for a contiguous, 16-byte-aligned
1-D float32 input of ``n >= 128`` elements -- ``setReduceConfig`` for the
launch configuration, then the per-thread vectorised reduce, the lane tree,
and the CTA (global) reduce when the configuration splits the input across
thread blocks:

  * ``q = n // 4`` vectors; ``bw = min(last_pow2(q), 512)`` lanes; the step is
    ``bw`` vectors, or ``bw * ctas`` when the input is split across ``ctas``
    thread blocks (``values_per_thread >= 256`` and a nonzero target grid);
  * thread ``t = lane + bw * block`` has four accumulators, one per position of
    a four-element vector, fed the vectors ``t, t + step, t + 2 step, ...`` in
    order while the vector is whole;
  * the tail ``n % 4`` elements go to accumulator 0 of lanes ``0 .. n%4-1`` of
    block 0;
  * thread value ``((a0 + a1) + a2) + a3``; a halving tree over the lanes of
    each block ``v[x] = v[x] + v[x + off]``, ``off = bw/2 .. 1``;
  * with blocks, lane ``x`` of the final block sums the staged block values
    ``x, x + bw, ...`` (below ``ctas``) in order from zero, and the same halving
    tree reduces the lanes.

The first probe (PB d96114991c11) confirmed the order below the split
(n <= 130560) and showed the split changes it above.  This one tests the
split.  It changes nothing in the tree and writes nothing but stdout.
"""
import json
import math
import struct
import sys

import torch


def last_pow2(v: int) -> int:
    return 1 << (v.bit_length() - 1)


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def reduce_config(n: int, props) -> dict:
    """``setReduceConfig`` for a full 1-D float32 reduction of ``n`` elements."""
    assert n >= 128
    mnt = 512                                   # mnt_wrapper<float>::MAX_NUM_THREADS
    warp = 32                                    # at::cuda::warp_size() on CUDA
    dim0, dim1 = n // 4, 1                       # vectorize_input: dim0 /= 4
    d0p = last_pow2(dim0) if dim0 < mnt else mnt
    d1p = last_pow2(dim1) if dim1 < mnt else mnt
    bw = min(d0p, warp)
    bh = min(d1p, mnt // bw)
    bw = min(d0p, mnt // bh)
    num_threads = bw * bh
    step_input, step_output = 1, 1
    input_mult = [0, 0, 0]
    input_mult[0] = step_input
    step_input *= bw
    warp_split_threshold = min(bh * 16, 256)
    if div_up(n, step_input) >= warp_split_threshold:
        input_mult[1] = step_input
        step_input *= bh
    else:
        step_output *= bh
    blocks_per_sm = int(props.max_threads_per_multi_processor) // num_threads
    target_grid_size = int(props.multi_processor_count) * blocks_per_sm
    grid = div_up(1, step_output)
    ctas = 1
    vpt = div_up(n, step_input)
    if input_mult[1] != 0 and vpt >= 256 and grid <= target_grid_size:
        ctas = max(min(div_up(target_grid_size, grid), div_up(vpt, 16)), div_up(vpt, 256))
        if ctas > 1:
            input_mult[2] = step_input
            step_input *= ctas
        else:
            ctas = 1
    return {"n": n, "bw": bw, "bh": bh, "ctas": ctas, "step": step_input,
            "input_mult": input_mult, "target_grid_size": target_grid_size}


def halving_tree(v: torch.Tensor) -> torch.Tensor:
    """``v[..., x] = v[..., x] + v[..., x + off]`` for ``x < off``, ``off = w/2 .. 1``."""
    off = v.shape[-1] // 2
    while off >= 1:
        v = torch.cat([v[..., :off] + v[..., off:2 * off], v[..., off:]], dim=-1)
        off //= 2
    return v[..., 0]


def replica_sum(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    n = x.numel()
    bw, ctas, step = cfg["bw"], cfg["ctas"], cfg["step"]
    q = n // 4
    threads = bw * ctas
    acc = torch.zeros(threads, 4, dtype=x.dtype, device=x.device)
    t = torch.arange(threads, device=x.device)
    four = torch.arange(4, device=x.device)
    for k in range(div_up(q, step)):
        idx = t + k * step
        valid = idx < q
        vals = x[4 * idx[valid, None] + four[None, :]]
        acc[valid] = acc[valid] + vals
    for lane in range(n % 4):
        acc[lane, 0] = acc[lane, 0] + x[4 * q + lane]
    v = ((acc[:, 0] + acc[:, 1]) + acc[:, 2]) + acc[:, 3]
    blocks = halving_tree(v.view(ctas, bw))                  # [ctas]
    if ctas == 1:
        return blocks[0]
    rows = div_up(ctas, bw)
    staged = torch.zeros(rows * bw, dtype=x.dtype, device=x.device)
    staged[:ctas] = blocks
    lanes = torch.zeros(bw, dtype=x.dtype, device=x.device)
    for r in range(rows):
        lanes = lanes + staged.view(rows, bw)[r]
    return halving_tree(lanes)


def bits(t: torch.Tensor) -> int:
    return struct.unpack("<I", struct.pack("<f", float(t)))[0]


def sample(n: int, dist: str, gen: torch.Generator, dev) -> torch.Tensor:
    if dist == "lognormal":
        return torch.exp(torch.randn(n, device=dev, generator=gen) * 6.0).float()
    if dist == "uniform":
        return torch.rand(n, device=dev, generator=gen)
    s = torch.exp(torch.randn(n, device=dev, generator=gen) * 1.5)
    w = torch.rand(n, device=dev, generator=gen) * 100
    table = torch.sort(torch.exp(torch.randn(16, device=dev, generator=gen) * 1.5)).values
    g = (s[:, None] - table[None, :]).abs().amin(dim=1)
    return w * g * g


def main() -> int:
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    report = {"torch": torch.__version__, "device": props.name,
              "multi_processor_count": int(props.multi_processor_count),
              "max_threads_per_multi_processor": int(props.max_threads_per_multi_processor),
              "warp_size": int(props.warp_size), "configs": {}, "cases": []}
    assert int(props.warp_size) == 32
    sizes = [128, 129, 131, 1000, 2051, 7681, 100003, 130560, 130561, 131072, 150001,
             200000, 229375, 229376, 229377, 262143, 262144, 300007, 524287, 524288,
             524289, 700001, 1000003, 1048576, 2097151, 2097152, 3000001, 4000037,
             8000009, 16777216, 33333331]
    gen = torch.Generator(device=dev).manual_seed(20260915)
    extra = sorted({int(math.exp(v)) for v in
                    (torch.rand(120, generator=torch.Generator().manual_seed(7))
                     * (math.log(6_000_000) - math.log(128)) + math.log(128)).tolist()})
    sizes = sorted(set(sizes) | set(extra))
    totals = {"cases": 0, "replica_mismatch": 0, "order_sensitive": 0}
    for n in sizes:
        cfg = reduce_config(n, props)
        report["configs"][str(n)] = cfg
        for dist in ("lognormal", "lut", "uniform"):
            for seed in range(2):
                gen.manual_seed(n * 10 + seed)
                x = sample(n, dist, gen, dev).contiguous()
                assert x.data_ptr() % 16 == 0
                want = x.sum()
                got = replica_sum(x, cfg)
                seq = torch.cumsum(x, 0)[-1]
                ok = bits(want) == bits(got)
                sensitive = bits(want) != bits(seq)
                totals["cases"] += 1
                totals["replica_mismatch"] += 0 if ok else 1
                totals["order_sensitive"] += 1 if sensitive else 0
                if not ok:
                    report["cases"].append({"n": n, "dist": dist, "seed": seed, "cfg": cfg,
                                            "torch": float(want), "replica": float(got)})
    report["totals"] = totals
    report["sizes"] = len(sizes)
    print(json.dumps(report, indent=1))
    print(f"SUMMARY cases={totals['cases']} replica_mismatch={totals['replica_mismatch']} "
          f"order_sensitive={totals['order_sensitive']} sizes={len(sizes)}")
    return 0 if totals["replica_mismatch"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
