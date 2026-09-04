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
#:
#: ORDER IS LOAD-BEARING, AND GETTING IT WRONG INFLATED THE ONE NUMBER THIS
#: SCRIPT EXISTS TO REPORT.  ``attention`` MUST be tested before
#: ``scaled_mm/cutlass``: flash-attention kernels are templated on CUTLASS types
#: and carry ``cutlass::bfloat16_t`` inside their own mangled name, so a
#: ``cutlass`` pattern tested first swallows them whole.  On the #83 armA/compiled
#: trace that mis-bucketing reported 3,136 "GEMM" launches where there are 448 --
#: 2,688 of them were flash attention -- and 3,136 is large enough next to the
#: 9,016 window-GEMV launches to read as "the compiled forward is partly falling
#: back at M=1", which is the exact question the trace was taken to answer.  The
#: corrected split says the opposite, and says it cleanly (see ``--phases``).
BUCKETS = (
    ("window_gemv", re.compile(r"window_gemv", re.I)),
    ("window_decode", re.compile(r"window_decode|decode_window", re.I)),
    # cuBLAS's tall-skinny GEMV.  Called out rather than left in ``other``
    # because on the #83 arms it is the single largest kernel in a decode step
    # -- 242.0 of 503.1 ms of device time, 48.1% -- and a summary that hides it
    # makes the lane's own share look larger than it is.  On those arms all 50
    # launches resolved through their correlation ids to ``aten::mm`` inside
    # ``logits_processor._apply_head``, i.e. ``lm_head``; the bucket is named
    # for the kernel rather than for that attribution, because the attribution
    # is a fact about one model's trace and the kernel is not.
    ("cublas_gemv", re.compile(r"gemvx|gemv_bf16", re.I)),
    ("attention", re.compile(r"flash|attn|paged", re.I)),
    ("scaled_mm/cutlass", re.compile(r"cutlass|scaled_mm|gemm|Kernel_.*sm.*", re.I)),
    ("fp8_quant", re.compile(r"quant", re.I)),
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


def phase_split(path: str, nbins: int = 24) -> dict:
    """Bucket launches along the trace's time axis, and say which buckets ran
    while the window GEMV was running.

    WHY THIS AND NOT A TOTAL.  A total cannot distinguish "these GEMMs are the
    prefill" from "the compiled forward is falling back to a GEMM at M=1", and
    those two readings of the same number are opposite conclusions about the
    lane.  The time axis separates them for free: the GEMV only exists at M=1,
    so any bin holding window-GEMV launches is a decode bin, and a GEMM launch
    in such a bin is a candidate fallback while one outside it is prefill or
    weight materialisation.  ``gemv_cooccurring`` is that count -- for an
    engaged lane with no fallback it is 0 for ``scaled_mm/cutlass``.
    """
    doc = _load(path)
    events = doc.get("traceEvents", doc if isinstance(doc, list) else [])
    rows = []
    for e in events:
        if e.get("ph") != "X" or str(e.get("cat", "")) not in (
                "kernel", "Kernel", "cuda_kernel"):
            continue
        name = str(e.get("name", ""))
        label = "other"
        for tag, pat in BUCKETS:
            if pat.search(name):
                label = tag
                break
        rows.append((float(e.get("ts", 0.0)), label))
    if not rows:
        return {"file": path, "bins": [], "note": "no kernel events"}
    rows.sort()
    t0, t1 = rows[0][0], rows[-1][0]
    width = (t1 - t0) / nbins + 1e-9
    bins = [collections.Counter() for _ in range(nbins)]
    for ts, label in rows:
        bins[min(nbins - 1, int((ts - t0) / width))][label] += 1
    co, apart = collections.Counter(), collections.Counter()
    for b in bins:
        tgt = co if b.get("window_gemv", 0) else apart
        for k, v in b.items():
            tgt[k] += v
    return {"file": path,
            "span_s": round((t1 - t0) / 1e6, 3),
            "bin_width_s": round(width / 1e6, 4),
            "bins": [dict(b) for b in bins],
            "gemv_cooccurring": dict(co),
            "gemv_absent": dict(apart)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--phases", type=int, default=0,
                    help="also split launches into N time bins and report which "
                         "buckets co-occur with the window GEMV (0 = off)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # RECURSIVE.  vLLM 0.28 does not always drop traces at the top of the
    # configured directory -- config/profiler.py documents a `capture_traces`
    # subdirectory for the capture mode -- and a summariser that globs one level
    # reports "no trace files" for a run that actually produced them.  Given the
    # trace is the only evidence a compiled census cannot supply, failing to FIND
    # one is as costly here as failing to take one.
    files = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "**", "*.json*"),
                                   recursive=True)))
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
    if args.phases:
        for f in files:
            ph = phase_split(f, args.phases)
            if not ph.get("bins"):
                continue
            keys = ["window_gemv", "window_decode", "scaled_mm/cutlass", "attention"]
            print(f"== phases {os.path.basename(f)}: {ph['span_s']} s in "
                  f"{args.phases} bins of {ph['bin_width_s']} s")
            print("   bin  " + "".join(f"{k:>19s}" for k in keys))
            for i, b in enumerate(ph["bins"]):
                print(f"   {i:>3}  " + "".join(f"{b.get(k, 0):>19d}" for k in keys))
            co, ab = ph["gemv_cooccurring"], ph["gemv_absent"]
            print("   -- launches in bins WITH window-GEMV activity (decode) --")
            for k in keys:
                print(f"      {k:22s} {co.get(k, 0):>8d}")
            print("   -- launches in bins WITHOUT it (prefill / materialisation) --")
            for k in keys:
                print(f"      {k:22s} {ab.get(k, 0):>8d}")
            for s_ in out:
                if s_["file"] == f:
                    s_["phases"] = ph
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
