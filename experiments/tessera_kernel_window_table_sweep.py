"""Does the window GEMV slow down with ``L``, and is the window body slower
than the span-2 trellis at all?  Both, without an encoder in the way.

The main bench (``tessera_kernel_window_bench.py``) times each arm right after
its own 9--370 s encode, so on a shared box the arms are minutes apart and any
contention that drifts over minutes lands on one arm and not another -- which
is exactly what happened on 2026-09-02 (three other GPU jobs, two of which
started mid-run).  This sweep removes the encoder: every unit is *synthetic*
(real forests, real LUTs, real plane layouts, random body bits) so all of them
are resident before anything is timed, and the whole comparison happens inside
one short window where contention is closer to common mode.  Two passes say
whether it drifted anyway.

Nothing here decodes anything meaningful -- the numbers are memory traffic and
launch shape and nothing else, which is the question being asked.  The
hypothesis under test: the per-unit ALPHABET table is ``2^L`` bytes, indexed
randomly by every block, so it crosses out of a small working set somewhere
between L=14 (16 KB) and L=16 (64 KB).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tessera_kernel_span_bench import profiled, timed  # noqa: E402

from tessera.alphabet import E2M1_GRID, E4M3_GRID, build_forest, tuple_grid  # noqa: E402
from tessera.trellis import ConvCode  # noqa: E402

CODE = ConvCode(memory=6)


def synthetic_scale(rows, cols, half, seed):
    """A LUT scale plane of random nibbles: the traffic, none of the meaning."""
    from tessera.kernel import lut_scale_table, pack_scale_nibbles

    torch.manual_seed(seed)
    refine = torch.randint(0, 16, (rows, cols // half), device="cuda")
    lut = torch.arange(16, dtype=torch.uint8, device="cuda") + 100
    return pack_scale_nibbles(refine, rows, cols, half), lut_scale_table(lut, "cuda")


def synthetic_span2(grid, rows, cols, rate, half=16, seed=0):
    """A packed span-2 trellis unit with random body bits.

    The forest and its LUTs are real -- cheap to build, and their sizes are
    what the kernel reads -- and only the Viterbi's output is replaced by
    random bits, which costs the same bytes to walk.
    """
    from tessera.kernel import (build_span2_luts, build_subset_values,
                                pack_kernel_planes)

    forest = build_forest(rate, grid=grid)
    torch.manual_seed(seed)
    steps = rows // grid.arity
    bits = torch.randint(0, 1 << rate, (steps, cols), device="cuda", dtype=torch.uint8)
    select, label, point = pack_kernel_planes(bits, rate=rate, memory=CODE.memory, span=2)
    label_lut, _ = build_span2_luts(forest, CODE, "cuda")
    nibbles, table = synthetic_scale(rows, cols, half, seed)
    return {
        "kind": "span2",
        "select": select, "label": label, "point": point,
        "nibbles": nibbles, "table": table, "label_lut": label_lut,
        "values": build_subset_values(forest, CODE, "cuda"),
        "global_scale": 1.0,
        "rows": rows, "cols": cols, "rate": rate, "arity": grid.arity,
        "memory": CODE.memory, "half": half,
    }


def synthetic_window(grid, rows, cols, rate, window, half=16, seed=0):
    """A packed window unit with random bytes: same traffic, no meaning."""
    from tessera.kernel import build_window_values, pack_window_planes

    torch.manual_seed(seed)
    steps = rows // grid.arity
    bits = torch.randint(0, 1 << rate, (steps, cols), device="cuda", dtype=torch.uint8)
    plane, offsets, rates = pack_window_planes(bits, (rate,) * cols, window)
    scale_plane, scale_table = synthetic_scale(rows, cols, half, seed)
    return {
        "kind": "window",
        "plane": plane, "offsets": offsets, "rates": rates,
        "table": torch.randint(0, len(grid.values), (1 << window,),
                               device="cuda").to(torch.uint8),
        "values": build_window_values(grid, "cuda"),
        "scale_plane": scale_plane, "scale_table": scale_table,
        "global_scale": 1.0,
        "rows": rows, "cols": cols, "window_bits": window,
        "arity": grid.arity, "half": half, "max_rate": rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--windows", type=int, nargs="+", default=[10, 12, 14, 16, 18])
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--lanes", type=int, default=64)
    ap.add_argument("--split-k", type=int, default=128)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="experiments/results/tessera_kernel_window_table_sweep.json")
    a = ap.parse_args()
    from tessera.kernel import gemv_from_packed

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    x = torch.randn(a.cols, device="cuda")
    records = []
    print(f"synthetic units {a.rows}x{a.cols}  commit {commit}  label {a.label!r}",
          flush=True)
    print("  concurrent GPU jobs: " + subprocess.run(
        ["bash", "-lc", "ps -eo args --no-headers | grep -c "
                        "'[p]ython experiments/tessera_\\|[p]ython experiments/window_'"],
        capture_output=True, text=True).stdout.strip(), flush=True)

    arms = [("span2 E2M1x2 R=7",
             synthetic_span2(tuple_grid(E2M1_GRID, 2), a.rows, a.cols, 7), None)]
    for name, grid, rate in (("E2M1x2 R=7", tuple_grid(E2M1_GRID, 2), 7),
                             ("E4M3   R=4", E4M3_GRID, 4)):
        for window in a.windows:
            arms.append((f"window {name} L={window}",
                         synthetic_window(grid, a.rows, a.cols, rate, window), window))

    for pass_no in range(a.passes):
        print(f"  --- pass {pass_no}", flush=True)
        for arm, packed, window in arms:
            fn = lambda: gemv_from_packed(x, packed, lanes=a.lanes, split_k=a.split_k)
            fn()
            torch.cuda.synchronize()
            ms, n, t0, t1 = timed(fn, a.seconds)
            prof = profiled(fn)
            kernel = "_window_gemv_kernel" if window else "_tuple_gemv_span2_kernel"
            self_ms = next((p["self_cuda_ms_per_call"] for p in prof
                            if p["name"] == kernel), None)
            rec = {"arm": arm, "pass": pass_no, "window_bits": window,
                   "ms_per_call": ms, "calls": n, "kernel_self_cuda_ms": self_ms,
                   "table_bytes": (1 << window) if window else 0,
                   "epoch": [t0, t1]}
            records.append(rec)
            print(f"  {arm:<26} {ms:8.4f} ms/call   {self_ms:8.4f} ms self CUDA"
                  f"   table {rec['table_bytes'] // 1024} KB", flush=True)

    out = Path(a.out)
    prior = json.load(open(out)) if out.exists() else []
    prior.append({"commit": commit, "label": a.label, "shape": [a.rows, a.cols],
                  "seconds": a.seconds, "passes": a.passes, "records": records,
                  "time": time.time()})
    json.dump(prior, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
