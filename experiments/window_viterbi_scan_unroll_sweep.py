"""The class scan's unroll factor, swept, so the default is a measurement.

``_scan_unroll`` picks the spelling of the window Viterbi's class-minimum
scan: the flat ``tl.static_range`` below a live-register threshold, and a
runtime ``tl.range`` with an unroll factor above it.  The threshold is derived
(``(FAN - 1) * elements_per_thread`` against 255 architectural registers); the
FACTOR is not derivable and has to be measured, which is what this does.

For each rate it times the whole fused encode at every candidate factor and
prints the compiled kernel's ``n_regs``/``n_spills`` beside it, so a factor
that wins by spilling is visible rather than merely fast on one shape.  The
reference is timed once per rate as the thing the fused path has to beat --
that is the comparison issue #11 is about.

Needs a quiet box: this is a wall-clock arm.  Check
``nvidia-smi --query-compute-apps=pid --format=csv`` first.

Run::

    TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      PYTHONPATH=src python experiments/window_viterbi_scan_unroll_sweep.py
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from tessera import window_viterbi as wv
from tessera.encode import viterbi_window


def timed(targets, vectors, L, rate, impl, warm):
    viterbi_window(warm, vectors, L, rate, impl=impl)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    states, sse = viterbi_window(targets, vectors, L, rate, impl=impl)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, float(sse), states


def compiled(targets, vectors, L, rate, warm):
    """``(n_regs, n_spills)`` of the ``_step`` kernel this configuration runs."""
    from window_viterbi_r8_diagnosis import _Spy

    step, tb, init, copy = wv._kernels()
    spy = _Spy(step)
    wv._CACHE["k"] = (spy, tb, init, copy)
    try:
        viterbi_window(warm, vectors, L, rate, impl="fused")
    finally:
        wv._CACHE["k"] = (step, tb, init, copy)
    ck = spy.compiled[-1][1]
    return getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="6,7,8,9,10")
    ap.add_argument("--factors", default="0,1,2,4,8,16")
    ap.add_argument("-L", "--window-bits", type=int, default=14)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--arity", type=int, default=1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 2
    rates = [int(r) for r in args.rates.split(",")]
    factors = [int(f) for f in args.factors.split(",")]
    L = args.window_bits
    dev = torch.device("cuda")

    torch.manual_seed(0)
    targets = torch.randn(args.rows, args.cols, device=dev)
    vectors = torch.randn(1 << L, args.arity, device=dev)
    warm = targets[:64, :64]

    print(f"device: {torch.cuda.get_device_name(0)}   L={L}  {args.rows}x{args.cols}"
          f"  arity={args.arity}")
    print("factor 0 = the flat tl.static_range; above R=8 it is skipped, because"
          " asking ptxas")
    print("to unroll 1023 iterations costs minutes of COMPILE before it costs anything else.")
    print()
    head = f"{'R':>2s} {'factor':>7s} {'secs':>9s} {'vs ref':>8s} {'n_regs':>7s} {'n_spills':>9s} {'exact':>6s}"
    print(head)
    print("-" * len(head))
    for rate in rates:
        ref_s, ref_sse, ref_states = timed(targets, vectors, L, rate, "reference", warm)
        print(f"{rate:2d} {'reference':>7s} {ref_s:9.3f} {1.0:8.2f}x {'-':>7s} {'-':>9s} {'-':>6s}")
        for factor in factors:
            if factor <= 0 and rate > 8:
                continue
            os.environ[wv._SCAN_UNROLL_ENV] = str(factor)
            try:
                s, sse, states = timed(targets, vectors, L, rate, "fused", warm)
                regs, spills = compiled(targets, vectors, L, rate, warm)
            except Exception as exc:                                # noqa: BLE001
                print(f"{rate:2d} {factor:7d} {'raised':>9s}  {type(exc).__name__}: {exc}")
                continue
            finally:
                os.environ.pop(wv._SCAN_UNROLL_ENV, None)
            regs, spills = str(regs), str(spills)
            exact = bool(torch.equal(states, ref_states)) and sse == ref_sse
            print(f"{rate:2d} {factor:7d} {s:9.3f} {ref_s / s:8.2f}x {regs:>7s} {spills:>9s} "
                  f"{str(exact):>6s}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
