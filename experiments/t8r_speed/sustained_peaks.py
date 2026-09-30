"""Sustained compute and bandwidth peaks on one GB10, with SM clock and power sampled.

``fp8_roofline.py`` measured the mma.sync peaks from a 0.42 ms burst, with no
clock reading. The A8SE-SH serve ran at an SM clock of 2.31-2.46 GHz against a
3.003 GHz maximum. This script separates the burst rate from the sustained
rate, and records the clock and power behind each:

* ``mma``: the fp8_roofline peak kernel (bf16 m16n8k16, e4m3 m16n8k32), best
  burst configuration, then relaunched back to back for SUSTAIN_S seconds with
  a longer inner loop (~20 ms per launch);
* ``gemm``: ``torch.mm`` BF16 at the serve's projection shapes (A8SE-SH trace)
  and ``torch._scaled_mm`` e4m3 at two routed-like shapes, each looped for
  GEMM_S seconds;
* ``read``: the fp8_roofline 16-byte read kernel, looped for GEMM_S seconds.

A pynvml sampler thread reads the SM clock, power, temperature and the active
clock-event reasons every 50 ms. Each phase reports the burst rate (first
timed launch), the sustained rate (median over the second half of the phase)
and the sampler's medians over that same half.

Usage: sustained_peaks.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_roofline as fr  # noqa: E402

SUSTAIN_S = float(os.environ.get("SUSTAIN_S", "6"))
GEMM_S = float(os.environ.get("GEMM_S", "4"))
# (M, K, N) and the role in the A8SE-SH rank-0 trace, per 2048-token chunk
SERVE_SHAPES = [
    (2048, 4096, 12576, "KDA in-proj, 34 calls"),
    (2048, 4096, 4096, "KDA o_proj, 34 calls"),
    (2048, 8192, 4096, "MLA o_proj, 11 calls"),
    (2048, 4096, 2048, "shared gate_up + MLA q_a/kv_a, 44 calls"),
    (2048, 1024, 4096, "shared down, 40 calls"),
    (2048, 1536, 8192, "MLA q_b, 11 calls"),
    (2048, 4096, 12288, "dense gate_up, 3 calls"),
]
E4M3_SHAPES = [(2048, 4096, 2048), (8192, 4096, 2048)]


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []
        self.stop_evt = threading.Event()
        self.errors = {}

    def _q(self, name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  record once, keep sampling
            self.errors.setdefault(name, repr(e)[:200])
            return None

    def run(self):
        nv, h = self.nv, self.h
        while not self.stop_evt.is_set():
            self.samples.append({
                "t": time.time(),
                "sm_mhz": self._q("sm", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                "power_w": self._q("power", lambda: nv.nvmlDeviceGetPowerUsage(h) / 1e3),
                "temp_c": self._q("temp", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                "reasons": self._q("reasons", lambda: int(nv.nvmlDeviceGetCurrentClocksEventReasons(h))),
            })
            time.sleep(0.05)

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x["t"] <= t1]

        def med(k):
            v = [x[k] for x in s if x[k] is not None]
            return statistics.median(v) if v else None
        reasons = 0
        for x in s:
            reasons |= x["reasons"] or 0
        return {"n": len(s), "sm_mhz_med": med("sm_mhz"), "power_w_med": med("power_w"),
                "temp_c_max": max((x["temp_c"] for x in s if x["temp_c"] is not None), default=None),
                "clock_event_reasons_or": hex(reasons)}


def _phase(name, launch, flops_or_bytes, unit, seconds, sampler):
    """launch() runs one timed unit and returns its ms. Loop for `seconds`."""
    torch.cuda.synchronize()
    series = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ms = launch()
        series.append((time.time(), flops_or_bytes / (ms * 1e-3) / (1e12 if unit == "TFLOPS" else 1e9)))
    t1 = time.time()
    half = [r for t, r in series if t >= t0 + (t1 - t0) / 2]
    rec = {"phase": name, "unit": unit, "launches": len(series), "burst": series[0][1],
           "sustained_med": statistics.median(half), "sustained_min": min(half),
           "sampler_second_half": sampler.window(t0 + (t1 - t0) / 2, t1),
           "sampler_first_1s": sampler.window(t0, t0 + 1.0), "t_unix": [t0, t1]}
    print(json.dumps(rec), flush=True)
    return rec


def _ev_ms(call, reps):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        call()
    b.record(); b.synchronize()
    return a.elapsed_time(b) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lib = fr.build()
    sampler = Sampler()
    sampler.start()
    time.sleep(1.0)
    idle = sampler.window(time.time() - 1.0, time.time())
    res = {"schema": "prismaquant.sustained_peaks/1", "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cublas": torch.backends.cuda.preferred_blas_library().name
           if hasattr(torch.backends.cuda, "preferred_blas_library") else None,
           "sustain_s": SUSTAIN_S, "gemm_s": GEMM_S, "idle": idle, "mma": {}, "gemm": {}, "read": {}}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    seed = torch.randint(0x3C003C00 - 5, 0x3C003C00 + 5, (64,), dtype=torch.int32, device="cuda")
    for kind in (1, 2):
        name, macs = fr.KINDS[kind]
        best = None
        for chains in (2, 4, 8):
            for per_sm, threads in ((1, 256), (2, 256), (4, 128), (2, 512)):
                blocks = sms * per_sm
                out = torch.empty(blocks * threads, dtype=torch.float32, device="cuda")
                iters = 8192 // chains
                ms = lib.peak(kind, chains, blocks, threads, iters, seed, out)
                tf = 2.0 * macs * (blocks * threads // 32) * iters * chains / (ms * 1e-3) / 1e12
                if best is None or tf > best[0]:
                    best = (tf, chains, blocks, threads)
        tf_burst, chains, blocks, threads = best
        out = torch.empty(blocks * threads, dtype=torch.float32, device="cuda")
        iters = 8192 // chains * 48          # ~20 ms per launch
        flops = 2.0 * macs * (blocks * threads // 32) * iters * chains
        rec = _phase(f"mma.{name}", lambda: lib.peak(kind, chains, blocks, threads, iters, seed, out) / 1.0,
                     flops, "TFLOPS", SUSTAIN_S, sampler)
        rec["burst_search_TFLOPS_0.4ms"] = tf_burst
        rec["config"] = {"chains": chains, "blocks": blocks, "threads": threads, "iters": iters}
        res["mma"][name] = rec
    for m, k, n, role in SERVE_SHAPES:
        A = torch.randn(m, k, device="cuda").bfloat16()
        B = torch.randn(n, k, device="cuda").bfloat16()
        C = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            torch.mm(A, B.t(), out=C)
        rec = _phase(f"bf16_mm {m}x{k}x{n}", lambda: _ev_ms(lambda: torch.mm(A, B.t(), out=C), 10),
                     2.0 * m * k * n, "TFLOPS", GEMM_S, sampler)
        rec["role"] = role
        res["gemm"][f"bf16 {m}x{k}x{n}"] = rec
        del A, B, C
    one = torch.ones((), device="cuda")
    for m, k, n in E4M3_SHAPES:
        A = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        call = lambda: torch._scaled_mm(A, B.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: E731
        for _ in range(3):
            call()
        res["gemm"][f"e4m3 {m}x{k}x{n}"] = _phase(f"e4m3_scaled_mm {m}x{k}x{n}", lambda: _ev_ms(call, 10),
                                                   2.0 * m * k * n, "TFLOPS", GEMM_S, sampler)
        del A, B
    n = 4 << 30
    x = torch.empty(n // 4, dtype=torch.int32, device="cuda").random_()
    blocks, threads = sms * 4, 256
    out = torch.empty(blocks * threads, dtype=torch.int32, device="cuda")
    res["read"] = _phase("read 4 GiB", lambda: lib.read_bw(x, out, blocks, threads), float(n), "GB/s",
                         GEMM_S, sampler)
    sampler.stop_evt.set()
    sampler.join(timeout=2)
    res["sampler_errors"] = sampler.errors
    with open(os.path.join(a.out, "sustained_peaks.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "sampler.json"), "w") as f:
        json.dump(sampler.samples, f)
    print("wrote", os.path.join(a.out, "sustained_peaks.json"), flush=True)


if __name__ == "__main__":
    main()
