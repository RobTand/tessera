#!/usr/bin/env python3
"""What does pinning a checkpoint's column rates to the GEMV lane's set cost?

Issue #104's decision was to build a rate-constrained checkpoint rather than
widen ``kernel_window_gemv.SUPPORTED_RATES``.  That is not free, and this
script prices it in the two forms the price actually takes.

**Leg 1 -- the reachable rung set, exact and code-derived.**  A rung is a root
rate; ``grammar.bresenham_rate_schedule`` realises it by mixing only the two
rates bracketing it.  So a unit's rate set is ``{floor(root)}`` for an integral
root and ``{floor(root), floor(root)+1}`` otherwise, and the constraint
"every column rate in (1, 2, 4)" admits root in ``[1, 2]`` and root ``== 4``
and nothing else.  Enumerated over every integer q256 in the family's own
published reader range, so it is a fact about this build and not an argument.

**Leg 2 -- a byte-matched screen on the published RD table.**  It reads
``experiments/rung_rd_curves_2026-09-03/results/rd_table.json``: per-unit,
per-rung measured Delta-KL against the uniform-R1006 arm, and the wire bytes of
each.  Two multi-choice knapsacks at one byte budget -- one over every measured
rung, one over the readable subset -- and the ratio between their measured
sums is what the constraint costs at that budget.

IT IS A SCREEN, AND THE RECEIPT SAYS SO IN THREE PLACES.  (a) The table is
another campaign's measurement, on seven layer-0 Linears of Qwen3-0.6B with the
other body Linears BF16 -- not this artifact.  (b) Its own receipt records the
per-unit costs summing to 84.8% of the jointly measured damage, so a knapsack
over sums is more super-additive than a serve.  (c) The readable rung nearest
the 4-bpp knee, **q256 1024, was never measured**: the table stops at 1006 and
resumes at 1044, so 1024 is INTERPOLATED here and every arm that uses it is
labelled.  A served number is in the receipt; this is the shape of the cost
away from the one budget a serve covered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tessera.grammar import rate_set, root_from_q256          # noqa: E402
from tessera.serving.contract import lane_requirements, reader_rate_grid  # noqa: E402

LANE = "tessera_window_gemv"
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "rung_rd_curves_2026-09-03", "results", "rd_table.json")
#: The fused serving units the allocator solves over: q/k/v share one rung and
#: gate/up share one, because vLLM builds one quant method per module.
GROUPS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
          ("self_attn.o_proj",),
          ("mlp.gate_proj", "mlp.up_proj"),
          ("mlp.down_proj",))


def readable_rungs(rate_cap: int, low: int, high: int):
    """Every integer rung in ``[low, high]`` whose rate set the lane can read."""
    supported = set(lane_requirements(LANE)["column_rates"])
    out = []
    for q in range(low, high + 1):
        try:
            rates = rate_set(root_from_q256(q), cap=rate_cap)
        except Exception:                        # a root outside the family's cap
            continue
        if set(rates) <= supported:
            out.append(q)
    return out


def runs_of(values):
    """``[(first, last)]`` contiguous runs, for printing a set as intervals."""
    out = []
    for v in values:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return [tuple(r) for r in out]


def _interpolate(table, unit, q, lo, hi):
    """Linear in the rung, between the two measured rungs bracketing it."""
    t = (q - lo) / (hi - lo)
    def at(field, r):
        return table[field][f"{unit}|{r}"]
    return (at("dkl", lo) + t * (at("dkl", hi) - at("dkl", lo)),
            at("wire_bytes", lo) + t * (at("wire_bytes", hi) - at("wire_bytes", lo)))


def knapsack(cells, budget):
    """Minimise summed dkl subject to summed bytes <= budget.

    ``cells`` is ``[[(dkl, bytes, label)]]`` -- one list of choices per group.
    Exact by DP over a byte grid coarse enough to be exact at this scale: the
    byte values are multiples of 1024 on these shapes, so the grid is 1024.
    """
    step = 1024
    cap = budget // step
    best = {0: (0.0, ())}
    for choices in cells:
        nxt = {}
        for used, (cost, picks) in best.items():
            for dkl, nbytes, label in choices:
                u = used + int(round(nbytes / step))
                if u > cap:
                    continue
                cand = (cost + dkl, picks + (label,))
                if u not in nxt or cand[0] < nxt[u][0]:
                    nxt[u] = cand
        best = nxt
        if not best:
            return None
    used, (cost, picks) = min(best.items(), key=lambda kv: kv[1][0])
    return {"dkl_sum": cost, "bytes": used * step, "picks": list(picks)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="512,749,1006,1024,1262",
                    help="uniform rungs whose byte totals are the budgets to solve at; a rung "
                         "the table did not measure (1024, the readable rung at the knee) is "
                         "interpolated and marked with a star")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    report: dict = {"schema": "tessera.rate_constraint_cost/1", "lane": LANE}

    # ---- leg 1: the reachable rung set -----------------------------------
    found = reader_rate_grid("TESSERA_FP8", "E4M3")
    family, low, high, _step = found
    supported = lane_requirements(LANE)["column_rates"]
    rungs = readable_rungs(rate_cap=7, low=low, high=high)
    intervals = runs_of(rungs)
    print(f"=== leg 1: which rungs the {LANE} lane can read")
    print(f"    lane column_rates {supported}; {family} reader range [{low}, {high}] (cap 7)")
    print(f"    readable q256: {len(rungs)} of {high - low + 1} integers -> "
          f"{[f'{a}..{b}' if a != b else str(a) for a, b in intervals]}")
    print(f"    i.e. root rate in {[(float(Fraction(a, 256)), float(Fraction(b, 256))) for a, b in intervals]}")
    print("    the whole open interval (2, 4) and everything above 4 is UNREACHABLE: an integral")
    print("    root 3 is uniform rate 3, and any fractional root there mixes a 3 or a 5.")
    report["leg1"] = {"family": family, "reader_range": [low, high],
                      "supported_rates": supported, "readable_rungs": len(rungs),
                      "readable_intervals": [list(i) for i in intervals],
                      "unreachable_uniform_bpp_band": "root in (2, 4) and root > 4"}

    # ---- leg 2: the byte-matched screen ----------------------------------
    if not os.path.isfile(TABLE):
        print(f"\n(no RD table at {TABLE}; leg 2 skipped)")
        return 0
    with open(TABLE) as fh:
        raw = json.load(fh)
    # ``dkl(unit, rung)`` is not stored: it is the sweep arm's KL minus the
    # uniform anchor's, which is how its own receipt states the table.  Built
    # here rather than transcribed, so a re-measured campaign flows through.
    base = next(a for a in raw["arms"] if a["arm"] == f"anchor:uniform_R{raw['uniform_rung']}")
    table = {"wire_bytes": raw["wire_bytes"], "dkl": {}}
    for unit in [u for g in GROUPS for u in g]:
        table["dkl"][f"{unit}|{raw['uniform_rung']}"] = 0.0
    for arm in raw["arms"]:
        if arm["kind"] != "sweep":
            continue
        unit, rung = arm["arm"].split(":", 1)[1].split("@R")
        table["dkl"][f"{unit}|{int(rung)}"] = arm["kl_full"] - base["kl_full"]
    measured = [int(r) for r in raw["rungs"]]
    lane_ok = {q: (set(rate_set(root_from_q256(q), cap=7)) <= set(supported)) for q in measured}
    # q256 1024 is the readable rung at the knee and the table does not have it.
    lo = max(q for q in measured if q < 1024)
    hi = min(q for q in measured if q > 1024)
    print(f"\n=== leg 2: byte-matched knapsack on the published RD table (A SCREEN)")
    print(f"    measured rungs {measured}")
    print(f"    readable of those: {[q for q in measured if lane_ok[q]]}"
          f"  -- q256 1024 INTERPOLATED between {lo} and {hi}")

    units = [u for g in GROUPS for u in g]
    def cells(allowed, use_1024):
        out = []
        for group in GROUPS:
            choices = []
            for q in allowed:
                dkl = sum(table["dkl"][f"{u}|{q}"] for u in group)
                nb = sum(table["wire_bytes"][f"{u}|{q}"] for u in group)
                choices.append((dkl, nb, f"{'/'.join(u.split('.')[-1] for u in group)}@R{q}"))
            if use_1024:
                dkl = nb = 0.0
                for u in group:
                    d, b = _interpolate(table, u, 1024, lo, hi)
                    dkl += d
                    nb += b
                choices.append((dkl, nb, f"{'/'.join(u.split('.')[-1] for u in group)}@R1024*"))
            out.append(choices)
        return out

    def uniform_arm(rung):
        """``(dkl sum, bytes)`` of the all-units-at-``rung`` arm, interpolating 1024."""
        if rung in measured:
            return (sum(table["dkl"][f"{u}|{rung}"] for u in units),
                    sum(table["wire_bytes"][f"{u}|{rung}"] for u in units))
        dkl = nb = 0.0
        for u in units:
            d, b = _interpolate(table, u, rung, lo, hi)
            dkl += d
            nb += b
        return dkl, nb

    ref = base["kl_full"]
    rows = []
    for budget_rung in [int(b) for b in args.budgets.split(",")]:
        uniform, ubytes = uniform_arm(budget_rung)
        budget = int(ubytes)
        star = "" if budget_rung in measured else "*"
        free = knapsack(cells(measured, use_1024=False), budget)
        constrained = knapsack(cells([q for q in measured if lane_ok[q]], use_1024=True), budget)
        row = {"budget_rung": budget_rung, "budget_bytes": budget,
               "interpolated_budget": bool(star),
               "uniform_dkl_sum": uniform, "uniform_kl_est": ref + uniform,
               "unconstrained": free, "constrained": constrained}
        if free and constrained:
            row["constrained_over_unconstrained"] = ((ref + constrained["dkl_sum"])
                                                     / (ref + free["dkl_sum"]))
            row["constrained_over_uniform"] = (ref + constrained["dkl_sum"]) / (ref + uniform)
        rows.append(row)
        print(f"\n    budget = uniform R{budget_rung}{star} ({budget} B), its own dkl sum "
              f"{uniform:+.6g} (KL est {ref + uniform:.6f})")
        for name, arm in (("unconstrained oracle", free), ("rate-constrained oracle", constrained)):
            if arm is None:
                print(f"      {name:24s}: NO FEASIBLE ASSIGNMENT at this budget")
                continue
            print(f"      {name:24s}: dkl sum {arm['dkl_sum']:+.6g}  KL est "
                  f"{ref + arm['dkl_sum']:.6f}  bytes {arm['bytes']}  picks {arm['picks']}")
        if free and constrained:
            print(f"      the constraint costs {row['constrained_over_unconstrained']:.3f}x the "
                  f"unconstrained oracle, {row['constrained_over_uniform']:.3f}x the uniform arm")
    report["leg2"] = {"note": "SCREEN: another campaign's table, 7 layer-0 units, sums are "
                              "84.8% additive vs the serve, R1024 interpolated",
                      "interpolated_from": [lo, hi], "rows": rows}

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
