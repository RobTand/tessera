"""Fused vs reference at R = 6, 7, 8 in one process: whose cliff is it?

The window encoder gets ~2x slower per rate up to R = 7 and then 40-80x
slower at R = 8.  Per step the trellis evaluates ``low * FAN = 2^L``
transitions whatever the rate, and ``_tile`` pins the launch geometry
(``grid``, ``warps``, ``BL*FAN*BC``) across R = 4..8 -- so nothing about the
*algorithm* scales with the rate, and a correct implementation should be flat
in it.  This times both implementations of the identical computation on the
identical tensor in the identical process, so contention on a shared box
cannot favour either arm.  ``sse`` is printed because the two must be one
answer computed two ways, not two answers.

Measured 2026-09-02 on sparklina (GB10, L = 14, 1024x1024, sse identical
across impls at every rate):

    impl        R=6      R=7      R=8     R7->R8
    reference   6.548 s  6.544 s  6.631 s   1.01x   <- flat, as the algebra says
    fused       0.753 s  1.474 s 65.004 s  44.09x   <- and 9.8x SLOWER at R=8
                                                       than the path it replaces

So the cliff is the Triton step kernel's, not the encoder's.  See
``docs/measurements/tessera-bf16-route-2026-09-02.md`` section 11 for the two
fixes (a dispatch rule in ``viterbi_window``; ``_tile`` holding ``BL*BC`` at a
warp) -- neither applied here, because ``encode.py`` is shared with other
branches mid-measurement.

Run::

    PYTHONPATH=src python experiments/bf16_window_rate_cliff.py
"""
import sys, time, torch
sys.path.insert(0, "src")
from tessera.encode import viterbi_window

torch.manual_seed(0)
dev = "cuda"
L = 14
rows, cols = 1024, 1024
targets = torch.randn(rows, cols, device=dev)
vectors = torch.randn(1 << L, 1, device=dev)

def timed(impl, rate, reps=1):
    torch.cuda.synchronize()
    viterbi_window(targets[:64], vectors, L, rate, impl=impl)   # warm/compile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        st, sse = viterbi_window(targets, vectors, L, rate, impl=impl)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, float(sse)

print(f"{'impl':10s} {'R':>2s} {'secs':>9s}  sse")
res = {}
for impl in ("fused", "reference"):
    for rate in (6, 7, 8):
        try:
            s, sse = timed(impl, rate)
        except Exception as e:
            print(f"{impl:10s} {rate:2d}  FAILED {type(e).__name__}: {e}")
            continue
        res[(impl, rate)] = s
        print(f"{impl:10s} {rate:2d} {s:9.3f}  {sse:.6f}")

print()
for impl in ("fused", "reference"):
    a, b, c = res.get((impl,6)), res.get((impl,7)), res.get((impl,8))
    if a and b and c:
        print(f"{impl:10s}  R6->R7 {b/a:6.2f}x   R7->R8 {c/b:6.2f}x")
