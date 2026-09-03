"""Where capturing the TCQ Viterbi pays for itself -- the constant's receipt.

``encode.TCQ_GRAPH_BREAKEVEN_CALLS`` is ``ceil(capture / (eager - replay))``:
the one-off cost of warming and capturing a shape, over what each replay
saves against the eager loop.  This measures the three terms on the shape the
LDLQ schedule actually replays -- the receipt's unit at its segment width --
and prints the constant they imply.  Re-run it when the trellis or the box
changes; do not edit the constant by hand.

usage:
  tcq_graph_breakeven.py --model DIR --unit NAME [--grid E2M1x2] [--q256 896]
                         [--block 32] [--reps 5]
"""
import argparse
import math
import statistics
import time

import torch
from safetensors import safe_open

from tessera import encode as enc
from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.export import DEFAULT_CODE, DEFAULT_SPAN, _plan_for
from tessera.manifest import BodyKind


def grid_by_name(name: str):
    for g in SERIALISABLE_GRIDS.values():
        if g.name == name:
            return g
    raise SystemExit(f"unknown grid {name!r}")


def clock(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    dev = "cuda"
    grid = grid_by_name(a.grid)
    with safe_open(f"{a.model}/model.safetensors", framework="pt") as f:
        W = f.get_tensor(a.unit + ".weight").to(dev, torch.float32).contiguous()
    rows, cols = W.shape
    rates, forests = _plan_for(grid, a.q256, cols, BodyKind.TCQ, None)
    present = max(set(rates), key=rates.count)
    forest = forests[present]
    tables = enc.TcqTables(forest, DEFAULT_CODE, 0, dev)
    targets = (W[:, :a.block] / W[:, :a.block].abs().amax(dim=1, keepdim=True)).contiguous()
    weights = torch.rand_like(targets) + 0.5

    def eager():
        enc._viterbi_core(targets, tables, DEFAULT_SPAN, weights)

    eager()                                              # first-launch cost, charged to nobody
    t_eager = clock(eager, a.reps)

    captures = []
    for _ in range(a.reps):
        g = enc.TcqGraph(rows, a.block, tables, DEFAULT_SPAN, True)
        g.targets.copy_(targets); g.weights.copy_(weights)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g._capture()
        torch.cuda.synchronize()
        captures.append(time.perf_counter() - t0)
    t_capture = statistics.median(captures)
    t_replay = clock(lambda: g(targets, weights), a.reps)

    a_e, b_e, s_e = enc._viterbi_core(targets, tables, DEFAULT_SPAN, weights)
    a_g, b_g, s_g = g(targets, weights)
    same = bool(torch.equal(a_e, a_g) and torch.equal(b_e, b_g) and torch.equal(s_e, s_g))
    saving = t_eager - t_replay
    breakeven = math.ceil(t_capture / saving) if saving > 0 else None
    print(f"unit {a.unit} shape [{rows}, {a.block}] rate {present} span {DEFAULT_SPAN}")
    print(f"eager   {t_eager*1e3:8.1f} ms")
    print(f"capture {t_capture*1e3:8.1f} ms  (warm-up run + capture)")
    print(f"replay  {t_replay*1e3:8.1f} ms")
    print(f"saving per call {saving*1e3:8.1f} ms -> break-even after "
          f"ceil({t_capture:.3f}/{saving:.3f}) = {breakeven} calls; "
          f"constant in tree: {enc.TCQ_GRAPH_BREAKEVEN_CALLS}")
    print(f"replay == eager (anchors, bits, sse): {same}")


if __name__ == "__main__":
    main()
