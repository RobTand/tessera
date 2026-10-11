"""The reduction of ``bench_rung_forward.py``'s cells: each rung's time over the reference rung's (tessera#694).

Pure Python over the bench's JSON, so the report and the tests read it without
torch.  A cell holds the forward pass (``F``) and the reverse pass (``R``) of
one (rung, M); the time of a cell is the mean of the two passes' medians.
"""
from __future__ import annotations

#: The bar #694 sets: a mixed rung's time over R1024's at the same M.
BAR = 1.5


def cell_time(cell, key):
    """The mean of the forward and reverse passes' medians, and their spread."""
    f, r = cell.get("F", {}).get(key), cell.get("R", {}).get(key)
    if not f or not r:
        return None
    mean = 0.5 * (f["median_ms"] + r["median_ms"])
    return {"ms": mean, "spread": abs(f["median_ms"] - r["median_ms"]) / mean}


def summarize_rungs(doc, reference):
    """``{M: {rung: {...}}}``: each rung's time over the reference rung's, per M.

    ``wall`` is the eager CUDA-event time, ``graph`` the replayed call (M <= 8
    only), ``kernel`` the profiler's device time per call.  ``fwd_per_j`` is the
    forward pass's power loop: forwards per joule.  ``rank`` orders the rungs of
    one M by it, best first.
    """
    out = {}
    records = doc["rungs"]
    ref = records[str(reference)]["cells"]
    for m in doc["meta"]["ms"]:
        row = {}
        for rung, record in records.items():
            cell = record["cells"].get(str(m))
            ref_cell = ref.get(str(m))
            if not cell or not ref_cell:
                continue
            entry = {}
            for key in ("wall", "graph"):
                t, t0 = cell_time(cell, key), cell_time(ref_cell, key)
                if t and t0:
                    entry[key] = {"ms": t["ms"], "ratio": t["ms"] / t0["ms"],
                                  "spread": max(t["spread"], t0["spread"])}
            k, k0 = cell.get("F", {}).get("profile"), ref_cell.get("F", {}).get("profile")
            if k and k0 and k0["kernel_us_per_call"]:
                entry["kernel"] = {"us": k["kernel_us_per_call"],
                                   "ratio": k["kernel_us_per_call"] / k0["kernel_us_per_call"]}
            power = cell.get("F", {}).get("power") or {}
            if power.get("calls_per_j"):
                entry["fwd_per_j"] = power["calls_per_j"]
                entry["mean_w"] = power.get("mean_w")
            row[rung] = entry
        ranked = sorted((r for r in row if "fwd_per_j" in row[r]), key=lambda r: -row[r]["fwd_per_j"])
        for place, rung in enumerate(ranked, 1):
            row[rung]["rank"] = place
        out[str(m)] = row
    return out


def rungs_over_the_bar(summary, reference, key="wall"):
    """``[(rung, M, ratio)]`` of every cell whose ``key`` ratio exceeds the bar."""
    over = []
    for m, row in summary.items():
        for rung, entry in row.items():
            if int(rung) == int(reference) or key not in entry:
                continue
            if entry[key]["ratio"] > BAR:
                over.append((int(rung), int(m), entry[key]["ratio"]))
    return sorted(over)
