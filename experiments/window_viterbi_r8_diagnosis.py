"""Why the fused window Viterbi collapses at R = 8: registers, not arithmetic.

``experiments/bf16_window_rate_cliff.py`` established the SHAPE of the cliff --
the reference is flat in the rate (the algebra says it must be: a step
evaluates ``low * FAN = 2^L`` transitions whatever R) while the fused Triton
step goes 0.48 s / 0.89 s / 36.8 s at R = 6 / 7 / 8 on 1024x1024 at L = 14.
It did not establish the MECHANISM, and issue #11 asks for one before anyone
proposes a fix.

This script reads the mechanism off the compiled kernel instead of guessing at
it.  For each rate it:

  * prints ``window_viterbi._tile``'s launch geometry -- the class tile
    ``BL``, the column tile ``BC``, the warp count, and the resulting grid --
    so the "nothing about the algorithm scales with the rate" claim can be
    checked rather than repeated;
  * derives the two numbers the class-minimum loop actually turns on: the
    unrolled iteration count ``FAN - 1`` and the *lane occupancy* of one
    iteration's load, ``BL * BC / (32 * warps)`` -- the fraction of the
    program's threads that hold an element of the ``[BC, BL]`` tile;
  * launches one step and reads ``n_regs`` and ``n_spills`` straight off the
    ``triton.compiler.CompiledKernel`` the launch returns, plus the PTX size.

The discriminating prediction, and why it is worth a script.  The class
minimum is ``tl.static_range(1, FAN)`` over ``FAN - 1`` *independent* masked
loads feeding a dependent select chain.  A scheduler that hoists the loads to
cover their latency needs one live register per hoisted load per element a
thread holds.  At L = 14 that is 63 loads over 256 threads holding 32 elements
(R = 6), 127 over 128 threads holding 16 (R = 7) and **255 over 128 threads
holding 8** (R = 8) -- and 255 live values is exactly where a thread runs out
of the 255 architectural registers.  So:

    n_spills == 0 at R <= 7 and n_spills > 0 at R = 8  =>  the cliff is a
    register spill inside a serial chain, and issue #11's candidate fix
    (widen ``BL`` so the tile stops collapsing) makes it WORSE, because a
    wider tile is more elements per hoisted load.

    n_spills == 0 everywhere  =>  it is not spills; look at issue rate and
    instruction cache next, and the ``BL`` widening is back on the table.

Reading registers costs one tiny launch per rate and no timing, so it is
valid on a contended box -- which is the point: the timing arm is not, and is
opt-in behind ``--time``.

Run::

    TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache \
      PYTHONPATH=src python experiments/window_viterbi_r8_diagnosis.py
    ... --time            # adds the wall-clock arm; needs a quiet box
    ... --rates 4,6,7,8   # default 4,5,6,7,8
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from tessera import window_viterbi as wv
from tessera.encode import viterbi_window


class _Spy:
    """Wrap a ``triton.JITFunction`` so the launch's ``CompiledKernel`` is kept.

    ``jit[grid](*args)`` returns the compiled kernel in Triton 3.x, and
    ``n_regs``/``n_spills`` are populated once it has been launched.  Wrapping
    rather than reaching into a private cache means this reads exactly the
    kernel the encoder ran, at exactly the constexprs it ran it with.
    """

    def __init__(self, jit):
        self.jit = jit
        self.compiled = []

    def __getitem__(self, grid):
        inner = self.jit[grid]

        def call(*args, **kwargs):
            out = inner(*args, **kwargs)
            self.compiled.append((kwargs, out))
            return out

        return call


def geometry(rate: int, window_bits: int, width: int):
    fan = 1 << rate
    low = (1 << window_bits) >> rate
    bl, bc, warps = wv._tile(fan, low, width)
    threads = 32 * warps
    return dict(fan=fan, low=low, bl=bl, bc=bc, warps=warps, threads=threads,
                iters=fan - 1, tile=bl * bc, occupancy=bl * bc / threads,
                front_out=bl * fan * bc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="4,5,6,7,8")
    ap.add_argument("-L", "--window-bits", type=int, default=14)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--time", action="store_true", help="add the wall-clock arm (quiet box only)")
    args = ap.parse_args()
    rates = [int(r) for r in args.rates.split(",")]
    L = args.window_bits

    if not torch.cuda.is_available():
        print("no CUDA device: this diagnosis reads a compiled Triton kernel")
        return 2
    dev = torch.device("cuda")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"L           : {L}   rows x cols: {args.rows} x {args.cols}")
    print()

    # -- the launch geometry, from the encoder's own tiler ------------------
    # ``width`` is what viterbi_window_fused computes from the L2 budget; for
    # the tile it only caps ``bc``, and every rate here saturates that cap.
    budget = getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 24 << 20) // 6
    width = max(1, min(budget // (2 * (1 << L) * 4), min(512, args.cols)))
    print(f"column batch width (L2 budget {budget} B): {width}")
    print()
    head = (f"{'R':>2s} {'FAN':>4s} {'low':>5s} {'BL':>4s} {'BC':>3s} {'warps':>5s} "
            f"{'thr':>4s} {'iters':>6s} {'tile':>5s} {'lanes/thr':>9s} {'front_out':>9s}")
    print(head)
    print("-" * len(head))
    geo = {}
    for rate in rates:
        g = geometry(rate, L, width)
        geo[rate] = g
        print(f"{rate:2d} {g['fan']:4d} {g['low']:5d} {g['bl']:4d} {g['bc']:3d} {g['warps']:5d} "
              f"{g['threads']:4d} {g['iters']:6d} {g['tile']:5d} {g['occupancy']:9.4f} "
              f"{g['front_out']:9d}")
    print()

    # -- registers and spills, off the launch itself ------------------------
    torch.manual_seed(0)
    targets = torch.randn(64, 64, device=dev)
    vectors = torch.randn(1 << L, 1, device=dev)

    head = (f"{'R':>2s} {'scan':>10s} {'n_regs':>7s} {'n_spills':>9s} {'shared':>7s} "
            f"{'ptx lines':>10s} {'hoist estimate':>15s}")
    print(head)
    print("-" * len(head))
    rows = []
    for rate in rates:
        step, tb, init, copy = wv._kernels()
        spy = _Spy(step)
        wv._CACHE["k"] = (spy, tb, init, copy)
        try:
            viterbi_window(targets, vectors, L, rate, impl="fused")
        finally:
            wv._CACHE["k"] = (step, tb, init, copy)
        ck = spy.compiled[-1][1]
        n_regs = getattr(ck, "n_regs", None)
        n_spills = getattr(ck, "n_spills", None)
        shared = getattr(getattr(ck, "metadata", None), "shared", None)
        ptx = getattr(ck, "asm", {}).get("ptx", "")
        g = geo[rate]
        # one live register per hoisted load per element a thread holds
        hoist = g["iters"] * max(1, round(g["occupancy"]))
        unroll = wv._resolve_scan_unroll(g["fan"], g["bl"], g["bc"], g["warps"])
        spelling = "flat" if unroll <= 0 else f"loop x{unroll}"
        rows.append((rate, n_regs, n_spills, unroll))
        print(f"{rate:2d} {spelling:>10s} {str(n_regs):>7s} {str(n_spills):>9s} "
              f"{str(shared):>7s} {len(ptx.splitlines()):10d} {hoist:15d}")
    print()

    flat_spilled = [r for r, _, s, u in rows if s and u <= 0]
    loop_spilled = [r for r, _, s, u in rows if s and u > 0]
    flat = [r for r, _, s, u in rows if u <= 0]
    looped = [r for r, _, s, u in rows if u > 0]
    if flat_spilled:
        print(f"VERDICT: the FLAT-unrolled scan SPILLS at R = {flat_spilled}.")
        print("         The class-minimum scan is a serial dependent chain, so every spilled")
        print("         value is a local-memory round trip the chain waits on.  Widening BL")
        print("         (issue #11's candidate) puts MORE elements under each hoisted load and")
        print("         is predicted to spill harder, not less.")
    elif looped:
        print(f"VERDICT: R = {looped} run the RUNTIME loop and spill "
              f"{'nothing' if not loop_spilled else str(loop_spilled)}; R = {flat} keep the flat")
        print("         unroll and are byte-for-byte the kernel they always were.  Re-run with")
        print(f"         {wv._SCAN_UNROLL_ENV}=0 to force the flat scan everywhere and see the")
        print("         spill the policy exists to avoid.")
    else:
        print("VERDICT: no spills at any rate measured, and every rate took the flat scan --")
        print("         the register hypothesis is DEAD on this geometry.  Look at issue rate")
        print("         and instruction cache next.")
    print()

    if args.time:
        print("timing (contended boxes make this an upper bound; see principle 15)")
        t = torch.randn(args.rows, args.cols, device=dev)
        head = f"{'R':>2s} {'fused s':>10s} {'reference s':>12s} {'fused/ref':>10s} {'sse match':>10s}"
        print(head)
        print("-" * len(head))
        for rate in rates:
            out = {}
            for impl in ("fused", "reference"):
                viterbi_window(t[:64], vectors, L, rate, impl=impl)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                st, sse = viterbi_window(t, vectors, L, rate, impl=impl)
                torch.cuda.synchronize()
                out[impl] = (time.perf_counter() - t0, float(sse), st)
            same = bool(torch.equal(out["fused"][2], out["reference"][2])) and \
                out["fused"][1] == out["reference"][1]
            print(f"{rate:2d} {out['fused'][0]:10.3f} {out['reference'][0]:12.3f} "
                  f"{out['fused'][0] / out['reference'][0]:10.2f}x {str(same):>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
