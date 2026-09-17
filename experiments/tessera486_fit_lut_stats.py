#!/usr/bin/env python3
"""tessera#486 stage 2: what ``_fit_lut`` does inside a real E2M1_K2 encode.

Wraps ``tessera.encode._fit_lut`` (and counts ``_lut_cost`` calls inside it),
then runs ``tessera385_bench.py``'s batched arm unchanged.  Per call it records
the caller, the live count ``n``, the candidate bracket, the number of swap
passes (every pass costs ``1 + entries * (bracket - entries)`` cost calls, so
the pass count is exact), the wall time and the part of it before the first
cost call (bracket plus greedy elimination).  The wrapper synchronises the
device around every call, so its walls are attributable and the run's total
wall is not a throughput number.  The encoder's answer is untouched.

    python experiments/tessera486_fit_lut_stats.py --stats-out STATS.json -- <bench args>
"""
import inspect
import json
import runpy
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import tessera  # noqa: E402
import tessera.encode as enc  # noqa: E402

CALLS: list = []
_LOCAL = threading.local()
_REAL_FIT = enc._fit_lut
_REAL_COST = enc._lut_cost
_SIG = inspect.signature(_REAL_FIT)


def _cost(targets, weights, table):
    rec = getattr(_LOCAL, "rec", None)
    if rec is not None:
        if rec["cost_calls"] == 0:
            torch.cuda.synchronize()
            rec["_t_first_cost"] = time.perf_counter()
        rec["cost_calls"] += 1
    return _REAL_COST(targets, weights, table)


def _fit(*args, **kwargs):
    bound = _SIG.bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    caller = sys._getframe(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    targets, weights = a["targets"], a["weights"]
    live = weights > 0
    n = int(live.sum())
    rec = {"caller": caller.f_code.co_name, "line": caller.f_lineno,
           "numel": int(targets.numel()), "dtype": str(targets.dtype), "n": n,
           "entries": int(a["entries"]), "swaps": int(a["swaps"]), "exact": bool(a["exact"]),
           "cost_calls": 0}
    if n and not a["exact"]:
        grid = enc.e4m3_positive_values(targets.device) * a["global_scale"]
        s = targets[live]
        lo, hi = float(s.min()), float(s.max())
        first = max(int((grid < lo).sum()) - 1, 0)
        last = min(int((grid <= hi).sum()) + 1, grid.numel())
        while last - first < rec["entries"]:
            if first > 0:
                first -= 1
            if last - first < rec["entries"] and last < grid.numel():
                last += 1
        rec["bracket"] = last - first
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    _LOCAL.rec = rec
    try:
        out = _REAL_FIT(*args, **kwargs)
    finally:
        _LOCAL.rec = None
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    rec["wall_s"] = t1 - t_pre
    rec["stats_overhead_s"] = t_pre - t0
    rec["before_first_cost_s"] = rec.pop("_t_first_cost", t1) - t_pre
    if "bracket" in rec and rec["cost_calls"]:
        per_pass = 1 + rec["entries"] * (rec["bracket"] - rec["entries"])
        rec["passes"] = rec["cost_calls"] / per_pass
        rec["trials_per_pass"] = per_pass - 1
    CALLS.append(rec)
    return out


def summarise(calls, total_wall):
    by_caller = defaultdict(lambda: {"calls": 0, "wall_s": 0.0, "before_first_cost_s": 0.0,
                                     "cost_calls": 0})
    for c in calls:
        b = by_caller[f"{c['caller']}:{c['line']}"]
        b["calls"] += 1
        b["wall_s"] += c["wall_s"]
        b["before_first_cost_s"] += c["before_first_cost_s"]
        b["cost_calls"] += c["cost_calls"]
    fit_wall = sum(c["wall_s"] for c in calls)
    return {
        "calls": len(calls),
        "fit_wall_s": fit_wall,
        "run_wall_s": total_wall,
        "fit_share_of_run": fit_wall / total_wall if total_wall else None,
        "by_caller": dict(by_caller),
        "n_hist": dict(Counter(c["n"] for c in calls).most_common(12)),
        "bracket_hist": dict(Counter(c.get("bracket") for c in calls).most_common(12)),
        "passes_hist": dict(Counter(c.get("passes") for c in calls).most_common(12)),
        "cost_calls_total": sum(c["cost_calls"] for c in calls),
    }


def main() -> int:
    argv = sys.argv[1:]
    if "--stats-out" not in argv or "--" not in argv:
        print(__doc__)
        return 2
    stats_out = Path(argv[argv.index("--stats-out") + 1])
    bench_args = argv[argv.index("--") + 1:]
    enc._fit_lut = _fit
    enc._lut_cost = _cost
    print(f"[stats] tessera {tessera.__file__}", flush=True)
    sys.argv = [str(ROOT / "experiments" / "tessera385_bench.py"), *bench_args]
    t0 = time.perf_counter()
    rc = 0
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as exc:
        rc = int(exc.code or 0)
    total = time.perf_counter() - t0
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise(CALLS, total)
    stats_out.write_text(json.dumps({"tessera_file": tessera.__file__, "bench_rc": rc,
                                     "summary": summary, "calls": CALLS}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
