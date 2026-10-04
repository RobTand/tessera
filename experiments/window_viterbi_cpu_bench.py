"""Time the CPU window Viterbi at the shapes that bound the CPU test suite.

tessera#795.  Run it on one core, on the same box, either side of a change:

    OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/window_viterbi_cpu_bench.py

Each case prints the best of ``--repeat`` wall times per call, and a digest of
the states and the ``sse`` float, so the two runs also show the same answer.
The problems are seeded; nothing is written to disk.
"""
from __future__ import annotations

import argparse
import hashlib
import time

import torch

from tessera.encode import viterbi_window

# (name, window_bits, rate, arity, rows, cols, weighted)
CASES = [
    ("L14-R1-256x256", 14, 1, 1, 256, 256, False),
    ("L14-R1-256x256-weighted", 14, 1, 1, 256, 256, True),
    ("L14-R4-256x256", 14, 4, 1, 256, 256, False),
    ("L14-R8-256x256", 14, 8, 1, 256, 256, False),
    ("E4M3-L14-R4-64x512", 14, 4, 1, 64, 512, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--only", default=None, help="run one case by name")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name, bits, rate, arity, rows, cols, weighted in CASES:
        if args.only and name != args.only:
            continue
        g = torch.Generator().manual_seed(rows * 31 + cols + rate)
        vectors = torch.randn(1 << bits, arity, generator=g)
        targets = torch.randn(rows, cols, generator=g) * 1.5
        weights = torch.rand(rows, cols, generator=g) + 0.5 if weighted else None
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            states, sse = viterbi_window(targets, vectors, bits, rate,
                                         weights=weights, impl="reference")
            times.append(time.perf_counter() - t0)
        digest = hashlib.sha256(states.numpy().tobytes()).hexdigest()[:16]
        print(f"{name}: best {min(times):.3f} s/call over {args.repeat} "
              f"(all {', '.join(f'{t:.3f}' for t in times)}) "
              f"states {digest} sse {sse!r}", flush=True)


if __name__ == "__main__":
    main()
