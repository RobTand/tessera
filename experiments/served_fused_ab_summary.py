"""Shared timing-summary convention for the #545 served A/B.

One definition of the per-phase summary, used by the driver, the analyzer,
and the CPU-only control tests, so the three cannot drift:

- the median is the real median (``statistics.median``: for an even count the
  mean of the two central order statistics, never ``sorted(x)[len(x)//2]``);
- the rep the in-engine profiler ran under is EXCLUDED from the latency and
  energy summaries (profiler overhead is not the route's cost) and its index
  is recorded beside the summary;
- the raw per-rep population is never replaced by the summary; callers keep
  the rows and the summary references their count.
"""
from __future__ import annotations

import statistics


def phase_summary(rows: list[dict], profiled_rep: int | None, *, ms_key: str = "engine_call_ms") -> dict:
    """Summarize one timed phase.

    ``rows`` are the driver's per-rep records (each with ``rep``,
    ``engine_call_ms``, ``wall_s``, ``gen_tokens``, ``prompt_tokens``).
    ``profiled_rep`` is the 1-based rep the in-engine profiler ran under, or
    None when the phase was never profiled.
    """
    if not rows:
        return {"n_total": 0, "n_included": 0, "included": []}
    excluded = profiled_rep if profiled_rep is not None else None
    included = [r for r in rows if r.get("rep") != excluded]
    if not included:
        # A phase must never summarize to nothing: every rep was profiled
        # only if the caller profiled all of them, which no arm does; refuse
        # loudly rather than widening the population back.
        raise ValueError("phase_summary: the profiled rep excluded every rep")
    call = [float(r[ms_key]) for r in included]
    wall = [float(r["wall_s"]) for r in included]
    gen = sum(int(r.get("gen_tokens") or 0) for r in included)
    pre = sum(int(r.get("prompt_tokens") or 0) for r in included)
    wall_total = float(sum(wall))
    summary = {
        "n_total": len(rows),
        "n_included": len(included),
        "excluded_profiled_rep": excluded,
        "engine_call_ms_median": statistics.median(call),
        "engine_call_ms_min": min(call),
        "engine_call_ms_max": max(call),
        "wall_s_total_included": wall_total,
        "gen_tokens_included": gen,
        "prompt_tokens_included": pre,
        "gen_tok_per_s_included": (gen / wall_total) if wall_total else None,
        "pre_tok_per_s_included": (pre / wall_total) if wall_total else None,
        "included_reps": [int(r["rep"]) for r in included],
        "timing_locus": ("CUDA-event bracket around the engine call as seen by the "
                         "driver process; kernel-time claims come only from the "
                         "in-engine profiler traces, never from this number"),
    }
    return summary


def median(values: list[float]) -> float:
    """The real median; exported so callers cannot re-derive the bug."""
    return statistics.median(values)
