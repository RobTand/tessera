#!/usr/bin/env python3
"""Profile the dense window GEMM's BF16 weight arithmetic, epilogue vs folded.

The value family can apply the per-row scale in two places:

* ``epilogue`` -- ``tl.dot`` on the raw bf16 table values, the fp32
  accumulator multiplied by the row scale once per output;
* ``folded`` -- ``bf16(value * row_scale)`` per weight in registers before
  ``tl.dot``, no epilogue scale (tessera#614).

Folding adds a multiply and a cast per decoded weight to the mainloop, so the
question this answers is what that costs at the dense shapes and rungs a GLM-5.3
allocation picks.  It runs on whichever build it is given: on a build without
the ``arithmetic`` argument it measures the epilogue kernel only, which is the
BEFORE side of the change.

Per (shape, rung, M, arithmetic) it records:

* the kernel's device time per call from ``torch.profiler`` (CUDA activity,
  the ``_window_gemm_kernel`` events only);
* the wall time per call from CUDA events;
* the mean board power over a timed loop of at least ``--power-seconds``,
  sampled by ``nvidia-smi`` in a side thread, and the useful work per joule
  (``2 * M * rows * cols`` multiply-adds counted as FLOPs).

The start and end UTC times of the whole run are recorded so the box-level
series (Netdata) can be read over the same window.

The units are synthetic: random codes at the rung's per-column rates, a random
bf16 table and a random positive row scale.  The kernel's cost depends on the
geometry and the rates, not on the values.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import inspect
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

#: GLM-5.3 dense Linear shapes (rows = out, cols = in): the dense MLP's
#: gate/up and down, the attention's q_b and o projections, and the shared
#: expert's gate/up.
DEFAULT_SHAPES = ((12288, 4096), (4096, 12288), (16384, 1536), (4096, 16384), (2048, 4096))
DEFAULT_RUNGS = (832, 1024, 1088)
DEFAULT_MS = (1, 8, 64, 512, 2048)


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
    return kg.WindowGemvUnit(rep=rep, table=values.cuda(), scale=scale, window_bits=L,
                             plan=kg.default_plan(rows, cols, 1), family="value")


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="the tessera commit under test")
    parser.add_argument("--label", required=True, help="before | after")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--power-seconds", type=float, default=1.5)
    parser.add_argument("--ms", type=int, nargs="+", default=list(DEFAULT_MS))
    parser.add_argument("--rungs", type=int, nargs="+", default=list(DEFAULT_RUNGS))
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this profile measures a CUDA kernel")
    import triton
    has_arithmetic = "arithmetic" in inspect.signature(wg.prepare_window_gemm).parameters
    arithmetics = ("epilogue", "folded") if has_arithmetic else ("epilogue",)
    sampler = PowerSampler()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []
    for shape_index, (rows, cols) in enumerate(DEFAULT_SHAPES):
        for rung in args.rungs:
            unit = make_unit(rows, cols, rung, seed=1000 * shape_index + rung)
            bundles = {}
            for arithmetic in arithmetics:
                kwargs = {"arithmetic": arithmetic} if has_arithmetic else {}
                bundles[arithmetic] = wg.prepare_window_gemm(unit, quantizer=None, **kwargs)
            for m in args.ms:
                x = (torch.randn(m, cols, generator=torch.Generator().manual_seed(m)) * 0.5
                     ).bfloat16().cuda()
                for arithmetic, bundle in bundles.items():
                    out = torch.empty(m, rows, dtype=torch.bfloat16, device="cuda")

                    def call(bundle=bundle, x=x, out=out):
                        return bundle(x, out=out)

                    for _ in range(args.warmup):
                        call()
                    k_us = kernel_us(call, args.iterations)
                    w_us = wall_us(call, args.iterations)
                    calls, elapsed, watts, n_samples = power_loop(call, args.power_seconds,
                                                                   sampler)
                    flops = 2.0 * m * rows * cols
                    joules_per_call = (watts * elapsed / calls) if watts is not None else None
                    records.append({
                        "rows": rows, "cols": cols, "q256": rung, "m": m,
                        "arithmetic": arithmetic,
                        "kernel_us": k_us, "wall_us": w_us,
                        "kernel_tflops": flops / (k_us * 1e-6) / 1e12,
                        "power_loop_calls": calls, "power_loop_seconds": elapsed,
                        "power_w_mean": watts, "power_samples": n_samples,
                        "gflop_per_joule": (flops / joules_per_call / 1e9
                                            if joules_per_call else None),
                    })
                    print(json.dumps(records[-1]), flush=True)
            del unit, bundles
            torch.cuda.empty_cache()
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    props = torch.cuda.get_device_properties(0)
    document = {
        "schema": "tessera.window_gemm_arithmetic_profile.v1",
        "label": args.label, "commit": args.commit,
        "has_arithmetic_argument": has_arithmetic,
        "started_utc": started, "finished_utc": finished,
        "host": platform.node(),
        "device": {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
                   "sm_count": props.multi_processor_count},
        "torch": torch.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "power_source": "nvidia-smi power.draw" if sampler.available else None,
        "iterations": args.iterations, "warmup": args.warmup,
        "power_seconds": args.power_seconds,
        "shapes": [list(s) for s in DEFAULT_SHAPES], "rungs": args.rungs, "ms": args.ms,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
