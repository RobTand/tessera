"""Time the window body's Viterbi, reference against fused, with the board's
power beside the seconds.

One tensor's shape, the encoder's own: a 2048x4096 target, an E4M3-shaped
table at arity 1 (256 distinct values, the per-channel Tessera-8 protocol)
and an E2M1x2 pair table at arity 2 (16 grid values, 256 distinct pairs).

Usage::

    python experiments/window_viterbi_bench.py --impl reference \
        --out experiments/results/window_viterbi_bench.jsonl

Each record carries the seconds, the sse, the identity check against the
reference when ``--check`` is passed, and the power drawn while it ran, since
on GB10 ``gpu_utilization`` reads 96% for a stalled kernel and a saturated
one alike -- power against the ~140 W envelope is the load that is real.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time

import torch

from tessera.encode import viterbi_window

E2M1 = [-6., -4., -3., -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 3., 4., 6., 0.]


def table_for(L: int, arity: int, device) -> torch.Tensor:
    """A table the encoder would really carry: grid codes, one per state."""
    size = 1 << L
    if arity == 1:                                   # E4M3: 256 distinct bytes
        vals = torch.linspace(-448.0, 448.0, 256, device=device)
    else:                                            # E2M1x2: pairs off the grid
        vals = torch.tensor(E2M1, device=device)
    idx = torch.randint(0, vals.numel(), (size, arity), device=device)
    return vals[idx].contiguous()


class _Power:
    """nvidia-smi at 1 Hz for the length of one run."""

    def __init__(self):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=3)


def neighbours() -> list[str]:
    """Other jobs on this board -- a contended number says so.

    Any ``experiments/`` python, not just ``tessera_bitshift``: the first
    baselines here were taken beside a job this filter did not name, and the
    reference ran 2.6x slower for it (428 s against 166 s at L=16).
    """
    out = subprocess.run(["pgrep", "-af", "experiments/"], capture_output=True,
                         text=True).stdout.splitlines()
    mine = str(__import__("os").getpid())
    return [l.split()[0] + " " + " ".join(l.split()[1:])[:60] for l in out
            if "/bin/bash" not in l and not l.startswith(mine + " ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="fused",
                    choices=("fused", "reference", "both"),
                    help="'both' times the two back to back on the same tensor, "
                         "which is the only way a shared board compares them")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--windows", type=int, nargs="+", default=[12, 14, 16])
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--arities", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--warm", action="store_true",
                    help="run once untimed first, so the seconds are not a "
                         "Triton compile (the reference has nothing to compile)")
    ap.add_argument("--check", action="store_true",
                    help="also run the reference and compare states and sse")
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default=None,
                    help="write a torch.profiler kernel table per config here")
    a = ap.parse_args()

    dev = "cuda"
    impls = ("reference", "fused") if a.impl == "both" else (a.impl,)
    for L in a.windows:
        for R in a.rates:
            for arity in a.arities:
                for impl in impls:
                    torch.manual_seed(0)
                    vectors = table_for(L, arity, dev)
                    scale = vectors.abs().max() / 4
                    targets = torch.randn(a.rows, a.cols, device=dev) * scale
                    torch.cuda.synchronize()
                    cold = None
                    if a.warm and impl == "fused":   # the reference compiles nothing
                        t0 = time.time()
                        viterbi_window(targets, vectors, L, R, chunk=a.chunk, impl=impl)
                        torch.cuda.synchronize()
                        cold = round(time.time() - t0, 3)
                    prof = None
                    if a.profile:
                        from torch.profiler import ProfilerActivity, profile
                        prof = profile(activities=[ProfilerActivity.CPU,
                                                   ProfilerActivity.CUDA])
                        prof.__enter__()
                    with _Power() as p:
                        t0 = time.time()
                        states, sse = viterbi_window(targets, vectors, L, R,
                                                     chunk=a.chunk, impl=impl)
                        torch.cuda.synchronize()
                        dt = time.time() - t0
                    if prof is not None:
                        prof.__exit__(None, None, None)
                        with open(a.profile, "a") as f:
                            f.write(f"\n# {impl} L={L} R={R} arity={arity} "
                                    f"rows={a.rows} cols={a.cols} chunk={a.chunk} "
                                    f"wall(profiled)={dt:.3f}s\n")
                            f.write(prof.key_averages().table(
                                sort_by="cuda_time_total", row_limit=10,
                                max_name_column_width=64) + "\n")
                    rec = dict(impl=impl, L=L, R=R, arity=arity, rows=a.rows,
                               cols=a.cols, chunk=a.chunk, seconds=round(dt, 3),
                               seconds_cold=cold,
                               sse=sse, start_ts=t0, end_ts=t0 + dt,
                               power_w=dict(
                                   n=len(p.samples),
                                   mean=round(statistics.fmean(p.samples), 2) if p.samples else None,
                                   max=max(p.samples) if p.samples else None,
                                   min=min(p.samples) if p.samples else None),
                               neighbours=neighbours())
                    if a.check:
                        sr, er = viterbi_window(targets, vectors, L, R, chunk=a.chunk,
                                                impl="reference")
                        rec["states_identical"] = bool(torch.equal(states, sr))
                        rec["sse_identical"] = sse == er
                    del states, targets, vectors
                    torch.cuda.empty_cache()
                    print(json.dumps(rec), flush=True)
                    if a.out:
                        with open(a.out, "a") as f:
                            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
