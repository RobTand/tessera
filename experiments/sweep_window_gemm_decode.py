#!/usr/bin/env python3
"""Sweep the dense window GEMM's launch schedule at decode M (tessera#617).

For each of the issue's shapes, at M = 1 and M = 8, this runs the raw
baseline prefill schedule ``(BM, BN, BK) = (64, 64, 64)`` and every raw
candidate ``BM in {16, 64} x BN in {32, 64, 128, 256}`` at ``BK = 64``,
then the served path (``PreparedWindowGemm.__call__`` on the dispatch under
test), which must match the raw schedule the dispatch selects, and reports
per point:

* the kernel's device time per call from ``torch.profiler`` (CUDA activity,
  the ``_window_gemm_kernel`` events only);
* the wall time per call from CUDA events;
* the mean board power over a timed loop from ``nvidia-smi``;
* the packed weight bytes and the effective read rate (bytes / kernel time);
* the launch grid (CTAs) against the device's SM count;
* the oracle diff of each candidate's output against the baseline schedule's.

Usage: ``sweep_window_gemm_decode.py --out PATH``.  Stdout carries one JSON
record per point; ``--out`` holds the full document.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from tessera import kernel_window_gemv as kg    # noqa: E402
from tessera import window_gemm as wg           # noqa: E402
import window_pack_reference as wpr             # noqa: E402

L = 14
KERNEL = "_window_gemm_kernel"

#: The issue's three shapes (rows = out, cols = in).
SHAPES = ((12288, 4096), (4096, 16384), (2048, 4096))
Q256 = 1024
MS = (1, 8)
BMS = (16, 64)
BNS = (32, 64, 128, 256)
BK = 64


def rung_rates(q256: int, cols: int) -> tuple:
    """Per-column integer rates whose mean is ``q256 / 256`` bits."""
    mean = q256 / 256
    base = math.floor(mean)
    high = round((mean - base) * cols)
    return tuple(base + 1 if c < high else base for c in range(cols))


def make_unit(rows: int, cols: int, q256: int, seed: int):
    rates = rung_rates(q256, cols)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rate = torch.tensor(rates, dtype=torch.int64)
    body = (torch.randint(0, 1 << 16, (rows, cols), generator=g)
            & ((1 << rate) - 1)).to(torch.uint8)
    values = (torch.randn(1 << L, generator=torch.Generator().manual_seed(seed + 1))
              * 0.03).bfloat16()
    scale = (torch.rand(rows, generator=torch.Generator().manual_seed(seed + 2)) * 2
             + 0.25).cuda()
    # The bitstream packer admits every rate 1..8; ``repack_window_body``
    # only admits rates dividing 8, and the rungs here mix rate 3 or 5 in.
    rep = wpr.pack_bitstream(body, rates)
    rep = dataclasses.replace(rep, runs=rep.runs.cuda(), words=rep.words.cuda(),
                              perm=rep.perm.cuda())
    unit = kg.WindowGemvUnit(rep=rep, table=values.cuda(), scale=scale, window_bits=L,
                             plan=kg.default_plan(rows, cols, 1), family="value")
    return unit, body, values


class PowerSampler:
    """Board power from ``nvidia-smi``, sampled on a side thread."""

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.available = shutil.which("nvidia-smi") is not None

    def _read(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        try:
            return float(out.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    def _loop(self):
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.samples = []
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()

    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else None


def kernel_us(call, iterations: int) -> float:
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
    total = 0.0
    count = 0
    for event in prof.events():
        if KERNEL in event.name and event.device_type.name == "CUDA":
            total += event.device_time
            count += 1
    if count != iterations:
        raise RuntimeError(f"profiled {count} {KERNEL} launches for {iterations} calls")
    return total / iterations


def wall_us(call, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def power_loop(call, seconds: float, sampler: PowerSampler):
    torch.cuda.synchronize()
    calls = 0
    with sampler:
        began = time.perf_counter()
        while time.perf_counter() - began < seconds:
            for _ in range(20):
                call()
            calls += 20
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
    return calls, elapsed, sampler.mean(), len(sampler.samples)


def _tol(ref):
    return 5e-3 + 1e-2 * float(ref.float().abs().max())


def raw_call(bundle, x, out, bm, bn, bk):
    """The value-family launch with explicit blocks, bypassing the schedule.

    This mirrors ``PreparedWindowGemm.__call__`` for the value family so the
    sweep measures the raw ``(BM, BN)`` grid -- including the pre-fix
    ``(64, 64)`` baseline the served path no longer launches at decode M.
    It refuses any other family rather than half-mirroring the epilogue.
    """
    import triton
    from tessera.window_geometry import TILE_ROWS
    if bundle.family != "value" or x.dtype != torch.bfloat16:
        raise SystemExit("the raw launch mirrors the value family only")
    x_perm = x.index_select(1, bundle.perm).contiguous()
    a = bundle.scale.new_zeros(1)
    m = int(x.shape[0])
    grid = (triton.cdiv(bundle.rows, bn), triton.cdiv(m, bm))
    wg._window_gemm_kernel[grid](
        bundle.words, bundle.table, bundle.codes, bundle.native, x_perm, out,
        bundle.scale, a, bundle.runs, bundle.init_perm,
        int(bundle.runs.shape[0]), bundle.tile_words, bundle.total_words,
        m, bundle.rows, bundle.cols,
        L=bundle.window_bits, TILE=TILE_ROWS,
        BM=bm, BN=bn, BK=bk,
        HAS_INIT=bundle.has_init, FP8=False, num_warps=8)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--power-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this sweep measures a CUDA kernel")
    import triton
    sampler = PowerSampler()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []
    for shape_index, (rows, cols) in enumerate(SHAPES):
        unit, _, _ = make_unit(rows, cols, Q256, seed=7000 + shape_index)
        packed_bytes = int(unit.rep.words.numel()) * 4
        bundles = {}
        for bm in BMS:
            for bn in BNS:
                bundles[(bm, bn)] = wg.prepare_window_gemm(
                    unit, quantizer=None, block_m=bm, block_n=bn, block_k=BK)
        # The baseline schedule runs first: every candidate diffs against it.
        first = (64, 64)
        bundles = {first: bundles[first],
                   **{k: v for k, v in bundles.items() if k != first}}
        for m in MS:
            x = (torch.randn(m, cols, generator=torch.Generator().manual_seed(m)) * 0.5
                 ).bfloat16().cuda()
            baseline_out = None
            for (bm, bn), bundle in bundles.items():
                out = torch.empty(m, rows, dtype=torch.bfloat16, device="cuda")

                def call(bundle=bundle, x=x, out=out, bm=bm, bn=bn):
                    return raw_call(bundle, x, out, bm, bn, BK)

                for _ in range(args.warmup):
                    call()
                torch.cuda.synchronize()
                k_us = kernel_us(call, args.iterations)
                w_us = wall_us(call, args.iterations)
                calls, elapsed, watts, n_samples = power_loop(call, args.power_seconds,
                                                               sampler)
                torch.cuda.synchronize()
                y = call().float()
                if (bm, bn) == (64, 64):
                    baseline_out = y.clone()
                    max_abs, rel = 0.0, 0.0
                else:
                    diff = (y - baseline_out).abs()
                    max_abs = float(diff.max())
                    denom = float(baseline_out.abs().max())
                    rel = max_abs / denom if denom else 0.0
                grid = [int((rows + bn - 1) // bn), int((m + bm - 1) // bm)]
                records.append({
                    "rows": rows, "cols": cols, "q256": Q256, "m": m,
                    "block_m": bm, "block_n": bn, "block_k": BK,
                    "path": "raw",
                    "baseline": (bm, bn) == (64, 64),
                    "kernel_us": k_us, "wall_us": w_us,
                    "packed_bytes": packed_bytes,
                    "effective_gb_s": packed_bytes / (k_us * 1e-6) / 1e9,
                    "grid": grid, "ctas": grid[0] * grid[1],
                    "power_loop_calls": calls, "power_loop_seconds": elapsed,
                    "power_w_mean": watts, "power_samples": n_samples,
                    "oracle_max_abs_vs_baseline": max_abs,
                    "oracle_rel_vs_baseline": rel,
                    "oracle_tol": _tol(baseline_out),
                })
                print(json.dumps(records[-1]), flush=True)
            # The served path on the dispatch under test: it must match the
            # raw schedule the dispatch selects.
            served = bundles[(64, 64)]
            out = torch.empty(m, rows, dtype=torch.bfloat16, device="cuda")

            def served_call(served=served, x=x, out=out):
                return served(x, out=out)

            for _ in range(args.warmup):
                served_call()
            torch.cuda.synchronize()
            k_us = kernel_us(served_call, args.iterations)
            w_us = wall_us(served_call, args.iterations)
            calls, elapsed, watts, n_samples = power_loop(served_call, args.power_seconds,
                                                           sampler)
            torch.cuda.synchronize()
            y = served_call().float()
            diff = (y - baseline_out).abs()
            max_abs = float(diff.max())
            denom = float(baseline_out.abs().max())
            records.append({
                "rows": rows, "cols": cols, "q256": Q256, "m": m,
                "block_m": 64, "block_n": 64, "block_k": BK,
                "path": "served",
                "baseline": False,
                "kernel_us": k_us, "wall_us": w_us,
                "packed_bytes": packed_bytes,
                "effective_gb_s": packed_bytes / (k_us * 1e-6) / 1e9,
                "grid": None, "ctas": None,
                "power_loop_calls": calls, "power_loop_seconds": elapsed,
                "power_w_mean": watts, "power_samples": n_samples,
                "oracle_max_abs_vs_baseline": max_abs,
                "oracle_rel_vs_baseline": max_abs / denom if denom else 0.0,
                "oracle_tol": _tol(baseline_out),
            })
            print(json.dumps(records[-1]), flush=True)
        del unit, bundles
        torch.cuda.empty_cache()
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    props = torch.cuda.get_device_properties(0)
    document = {
        "schema": "tessera.window_gemm_decode_sweep.v1",
        "commit": args.commit,
        "started_utc": started, "finished_utc": finished,
        "host": platform.node(),
        "device": {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
                   "sm_count": props.multi_processor_count},
        "torch": torch.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "power_source": "nvidia-smi power.draw" if sampler.available else None,
        "iterations": args.iterations, "warmup": args.warmup,
        "power_seconds": args.power_seconds,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
