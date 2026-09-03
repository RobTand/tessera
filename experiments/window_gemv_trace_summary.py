#!/usr/bin/env python3
"""Aggregate a vLLM torch profile by CUDA kernel: which kernels launched, and
for how long.

This is the half of #83's evidence that a route record cannot carry.  Under a
compiled forward the record is written from the trace and stamps the combined
``(symbol, decoder)`` pair whatever runs underneath it, so the compiled census
proves DISPATCH, not LAUNCH.  A kernel name in a chrome trace is the launch.

usage: window_gemv_trace_summary.py <trace-dir-or-file> [--top 25] [--out x.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re

#: Kernel-name fragments worth naming in the summary, and what they mean here.
#: A pattern, not an allow-list: everything else is still counted, just rolled
#: into ``other``.
BUCKETS = (
    ("window_gemv", re.compile(r"window_gemv", re.I)),
    ("window_decode", re.compile(r"window_decode|decode_window", re.I)),
    ("scaled_mm/cutlass", re.compile(r"cutlass|scaled_mm|gemm|Kernel_.*sm.*", re.I)),
    ("fp8_quant", re.compile(r"quant", re.I)),
    ("attention", re.compile(r"flash|attn|paged", re.I)),
    ("elementwise/triton", re.compile(r"triton|elementwise|vectorized_elementwise", re.I)),
)


def _load(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)


def summarise(path: str) -> dict:
    doc = _load(path)
    events = doc.get("traceEvents", doc if isinstance(doc, list) else [])
    per_kernel = collections.Counter()
    per_kernel_us = collections.Counter()
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", ""))
        if cat not in ("kernel", "Kernel", "gpu_user_annotation", "cuda_kernel"):
            continue
        if cat.startswith("gpu_user"):
            continue
        name = str(e.get("name", ""))
        per_kernel[name] += 1
        per_kernel_us[name] += float(e.get("dur", 0.0))
    buckets = collections.Counter()
    bucket_us = collections.Counter()
    for name, n in per_kernel.items():
        label = "other"
        for tag, pat in BUCKETS:
            if pat.search(name):
                label = tag
                break
        buckets[label] += n
        bucket_us[label] += per_kernel_us[name]
    return {"file": path,
            "kernel_launches": sum(per_kernel.values()),
            "kernel_us_total": round(sum(per_kernel_us.values()), 1),
            "by_bucket": {k: {"launches": buckets[k], "us": round(bucket_us[k], 1)}
                          for k in sorted(buckets, key=lambda b: -bucket_us[b])},
            "by_kernel": [{"name": n, "launches": per_kernel[n],
                           "us": round(per_kernel_us[n], 1)}
                          for n in sorted(per_kernel, key=lambda k: -per_kernel_us[k])]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "*.json*"))))
    if not files:
        raise SystemExit(f"no trace files under {args.path}")
    out = [summarise(f) for f in files]
    for s in out:
        print(f"== {os.path.basename(s['file'])}: {s['kernel_launches']} kernel launches, "
              f"{s['kernel_us_total']/1000:.1f} ms on device")
        for k, v in s["by_bucket"].items():
            print(f"   {k:22s} {v['launches']:8d} launches  {v['us']/1000:9.2f} ms")
        print("   -- top kernels --")
        for k in s["by_kernel"][:args.top]:
            print(f"   {k['us']/1000:9.3f} ms  x{k['launches']:<7d} {k['name'][:100]}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
