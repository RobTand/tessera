"""Why do >= 40 MB buffers read at half rate inside the GEMV bench?

In chain v2 every arm on the 27B-assumed shapes (44.56 MB wire, 89 MB fp8,
178 MB bf16) -- the fused GEMV, ``_scaled_mm``, cuBLAS bf16 and the plain
streaming reader alike -- ran at ~120 GB/s, while ``bw_probe_cuda.py`` reads
64-256 MB buffers at 234-246 GB/s with the same reader kernel.  This isolates
the buffer: same reader, same launch, buffers made the probe's way (randint
uint8), the bench's way (synth -> repack -> clone) and in between.

    PYTHONPATH=src python experiments/bw_size_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/home/rob/tmp/torch-ext-gemv")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import bench_kernel_window_gemv as B  # noqa: E402


def read_rate(bufs, blocks=192, threads=256, rounds=40, iters=int(os.environ.get("ITERS", "8"))):
    reader = B._reader()
    sink = torch.zeros(8192, device="cuda")
    n = len(bufs)
    for i in range(n * 2):
        reader.read_bytes(bufs[i % n], sink, blocks, threads)
    torch.cuda.synchronize()
    best = None
    k = 0
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            reader.read_bytes(bufs[k % n], sink, blocks, threads)
            k += 1
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / iters * 1000
        best = us if best is None or us < best else best
    nbytes = bufs[0].numel() * bufs[0].element_size()
    return {"us": round(best, 1), "GB_per_s": round(nbytes / (best * 1e-6) / 1e9, 1), "MB": round(nbytes / 1e6, 2), "replicas": n}


def main():
    out = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    n32 = 5120 * 17408 // 8                # 44.56 MB of int32 words (the 5120x17408 wire at 4 bits/weight)
    # probe's way: uint8 randint, 64 MB x4 and 44.56 MB x3
    out["uint8 randint 64MB x4"] = read_rate([torch.randint(0, 255, (64 << 20,), dtype=torch.uint8, device="cuda") for _ in range(4)])
    out["uint8 randint 44.56MB x3"] = read_rate([torch.randint(0, 255, (n32 * 4,), dtype=torch.uint8, device="cuda") for _ in range(3)])
    print("iters per round:", int(os.environ.get("ITERS", "8")), flush=True)
    # int32 randint of the same bytes
    out["int32 randint 44.56MB x3"] = read_rate([torch.randint(0, 1 << 30, (n32,), dtype=torch.int32, device="cuda", generator=g) for _ in range(3)])
    # the bench's way: synth -> repack -> clone (what the kernel and the wire_read arm read)
    unit = B.synth(5120, 17408)
    words = unit.rep.words
    print("bench words:", words.dtype, words.shape, words.is_contiguous(), words.numel() * 4 / 1e6, "MB", flush=True)
    out["bench repacked 44.56MB x3 (clone)"] = read_rate([words] + [words.clone() for _ in range(2)])
    out["bench repacked 44.56MB x1"] = read_rate([words])
    out["bench repacked 44.56MB x3 (contiguous copy)"] = read_rate([words.contiguous().clone() for _ in range(3)])
    # the same path at 12.45 MB (the 9728x2560 wire) for reference
    u2 = B.synth(9728, 2560)
    out["bench repacked 12.45MB x8"] = read_rate([u2.rep.words] + [u2.rep.words.clone() for _ in range(7)])
    del unit, u2
    torch.cuda.empty_cache()
    # fp8 comparator buffers at the two sizes
    out["fp8 randint 49.8MB x2 (19456x2560)"] = read_rate([torch.randint(0, 200, (19456, 2560), dtype=torch.uint8, device="cuda") for _ in range(2)])
    out["fp8 randint 24.9MB x4 (9728x2560)"] = read_rate([torch.randint(0, 200, (9728, 2560), dtype=torch.uint8, device="cuda") for _ in range(4)])
    out["fp8 randint 89MB x2 (5120x17408)"] = read_rate([torch.randint(0, 200, (5120, 17408), dtype=torch.uint8, device="cuda") for _ in range(2)])
    for k, v in out.items():
        print(f"  {k:48s} {v}", flush=True)
    mem = torch.cuda.mem_get_info()
    out["mem_get_info_free_total_GB"] = [round(mem[0] / 1e9, 1), round(mem[1] / 1e9, 1)]
    print("free/total GB:", out["mem_get_info_free_total_GB"])
    p = Path(os.environ.get("OUT", "/mnt/shared/tessera-runs/gemv")) / f"bw_size_probe_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
