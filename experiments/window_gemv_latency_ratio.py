#!/usr/bin/env python3
"""The two-arm ratio for the window-GEMV decode path: engaged against the route
it replaces, from two served-latency receipts and, optionally, the two traces.

WHAT THE RATIO IS.  ``fallback / engaged`` on a latency, so a number greater
than 1 means the lane is faster by that factor.  Both arms are one checkpoint
(hardlinked, one inode) and one image, differing only in whether the window-GEMV
extension can be built, so what the ratio prices is the lane.

WHY THIS TOOL REFUSES AS OFTEN AS IT REPORTS.  #83's latency half produced two
arms 8x apart where the kernel difference cannot exceed ~2x, because the box
moved underneath them -- and the load average pointed the WRONG WAY, the slower
arm having the lower run queue.  Reporting that ratio would have been the worst
error available.  So the arms' contention fields are read first and a ratio from
a contended pair is printed with its verdict attached and marked not-evidence,
never as a headline.  A contaminated measurement that says it is contaminated is
useful; one that does not is worse than none.

WHY A SERVED TPOT RATIO IS NOT THE WHOLE ANSWER.  A decode step runs the whole
model, and the lane owns only the Tessera Linears in it.  armA's trace has a
cuBLAS bf16 GEMV taking more device time than the entire window-GEMV bucket, so
an end-to-end TPOT ratio is diluted by work the lane does not touch and will sit
closer to 1 than the kernel does, however good the kernel is.  Both readings are
printed: the served TPOT ratio (what a user feels) and, from the traces, the
device time in the profiled decode window by bucket (what the lane moved).
Neither is the other's headline.

usage:
  window_gemv_latency_ratio.py --engaged latency-armA-streamed-eager.json \
      --fallback latency-armB-streamed-eager.json \
      [--engaged-trace prof-armA-streamed-eager] [--fallback-trace prof-armB-streamed-eager] \
      [--out ratio-streamed-eager.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from window_gemv_trace_summary import BUCKETS  # noqa: E402  -- after the path fix

#: The two histograms the ratio is taken on, and the window each belongs to.
#: TPOT is the decode-regime number -- the per-output-token cost of the steps
#: the GEMV serves -- and TTFT is the prefill one, which the GEMV refuses by
#: name and the materialised path serves in BOTH arms.  A TTFT ratio far from 1
#: is therefore a statement about the box, not the lane, and is reported for
#: exactly that reason: it is this pair's own control.
METRICS = (("decode", "tpot"), ("decode", "itl"), ("decode", "ttft"),
           ("prefill", "ttft"), ("prefill", "tpot"))


#: metric key -> the vLLM histogram stem behind it, so a receipt whose named
#: field is ``None`` can still be read out of ``series_moved``.
#: THIS IS WHY ``moved()`` EXISTS.  The 2026-09-03 eager receipts carry
#: ``tpot: None`` and ``itl: None`` -- the driver was reading a stem this vLLM
#: release does not publish -- and the per-output-token numbers those two arms
#: are quoted for in ``docs/measurements/tessera-window-gemv-served-2026-09-03.md``
#: are recoverable ONLY from the every-series-that-moved dump.  A receipt that
#: recorded just the two stems it believed in would have cost a second serve.
STEMS = {"ttft": "vllm:time_to_first_token_seconds",
         "tpot": "vllm:request_time_per_output_token_seconds",
         "itl": "vllm:inter_token_latency_seconds"}


def _metric(window: dict, metric: str):
    """One window's metric, from the named field or from ``series_moved``.

    The source is carried on the value, because "the engine published this
    histogram" and "I reconstructed it from the deltas" are different provenance
    and a reader should not have to know which receipts had the stem right.
    """
    got = window.get(metric)
    if got and got.get("mean_s"):
        return dict(got, source="histogram")
    moved = window.get("series_moved") or {}
    stem = STEMS.get(metric)
    if not stem:
        return None
    s, n = moved.get(stem + "_sum"), moved.get(stem + "_count")
    if not s or not n:
        return None
    return {"mean_s": s / n, "sum_s": s, "count": n, "source": "series_moved"}


def _ratio(fallback, engaged):
    if not fallback or not engaged:
        return None
    fe, en = fallback.get("mean_s"), engaged.get("mean_s")
    if not fe or not en:
        return None
    return {"engaged_ms": round(1000 * en, 4), "fallback_ms": round(1000 * fe, 4),
            "ratio_fallback_over_engaged": round(fe / en, 4),
            "n_engaged": engaged.get("count"), "n_fallback": fallback.get("count"),
            "source": f"{engaged.get('source')}/{fallback.get('source')}"}


def _contention(receipt: dict) -> dict:
    """Every contention fact a receipt carries, in one place.

    ``swap_moved`` is the /2 field and is absent from the 2026-09-03 receipts;
    ``None`` there means "this receipt cannot say", which is not the same as
    "nothing moved" and is not reported as it.
    """
    io = receipt.get("swap_io") or {}
    moved = None
    if io:
        moved = any(bool(io.get(w, {}).get("moved")) for w in ("decode", "prefill"))
    return {"contended_label": receipt.get("contended"),
            "max_load1_per_cpu": receipt.get("max_load1_per_cpu"),
            "swap_resident_gib": receipt.get("max_swap_used_gib"),
            "min_mem_available_gib": receipt.get("min_mem_available_gib"),
            "swap_moved_during_windows": moved,
            "swap_io": io or None}


def _trace_file(path: str) -> str | None:
    if not path:
        return None
    if os.path.isfile(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "**", "*.pt.trace.json*"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "**", "*.json*"), recursive=True))
    # rank0 is the engine core's, and it is the one that runs the model.
    ranked = [f for f in files if "rank0" in os.path.basename(f)]
    pool = ranked or files
    return max(pool, key=os.path.getsize) if pool else None


def _bucket(name: str) -> str:
    for tag, pat in BUCKETS:
        if pat.search(name):
            return tag
    return "other"


def trace_window(path: str, t0: float, t1: float) -> dict:
    """Kernel launches and device time inside an absolute UTC window.

    ONE CUT RULE FOR BOTH ARMS.  ``--phases`` in the trace summariser finds
    decode bins by the presence of a window-GEMV launch; the fallback arm has
    no such launch, so that rule cannot identify the same window in both arms.
    The driver's ``marks_unix`` do, because they are the client's clock and the
    client drove both arms the same way.  A chrome trace's absolute clock is
    ``baseTimeNanoseconds + ts``.
    """
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        doc = json.load(fh)
    base_s = float(doc.get("baseTimeNanoseconds", 0)) / 1e9
    events = doc.get("traceEvents", [])
    launches, us = collections.Counter(), collections.Counter()
    per_kernel_us = collections.Counter()
    seen, inside = 0, 0
    span = [None, None]
    for e in events:
        if e.get("ph") != "X" or str(e.get("cat", "")) not in (
                "kernel", "Kernel", "cuda_kernel"):
            continue
        seen += 1
        t = base_s + float(e.get("ts", 0.0)) / 1e6
        span[0] = t if span[0] is None else min(span[0], t)
        span[1] = t if span[1] is None else max(span[1], t)
        if not (t0 <= t <= t1):
            continue
        inside += 1
        name = str(e.get("name", ""))
        b = _bucket(name)
        launches[b] += 1
        us[b] += float(e.get("dur", 0.0))
        per_kernel_us[name] += float(e.get("dur", 0.0))
    return {"file": path, "base_time_s": base_s,
            "trace_span_utc": span,
            "kernels_in_trace": seen, "kernels_in_window": inside,
            "window_unix": [t0, t1],
            "by_bucket": {k: {"launches": launches[k], "ms": round(us[k] / 1000, 3)}
                          for k in sorted(launches, key=lambda b: -us[b])},
            "device_ms_in_window": round(sum(us.values()) / 1000, 3),
            "top_kernels": [{"name": n, "ms": round(v / 1000, 3)}
                            for n, v in per_kernel_us.most_common(8)]}


def headroom_if_lane_were_free(lane_share: float | None) -> float | None:
    """How much faster THIS arm would be with a free lane: ``1/(1 - s)``.

    READ THE ARM THIS COMES FROM.  A decode step is ``T = R + L``: work the
    lane cannot touch, plus the lane bucket.  Applied to the **fallback** arm
    this is the A/B's Amdahl ceiling, because ``T_f/T_e < T_f/R = 1/(1 - s_f)``
    -- driving the engaged lane to zero is the best the ratio can do.  Applied
    to the **engaged** arm it is only that arm's own remaining headroom and
    bounds nothing about the other arm: ``T_f/T_e = (R + L_f)/(R + L_e)`` is
    unbounded in ``L_f``.

    An earlier spelling of this function called the engaged arm's value "the
    ceiling the ratio can reach", which is that inversion.  On the #83 engaged
    trace it is 1.424x; the A/B ceiling is not knowable from that trace at all,
    because no fallback trace exists (those receipts carry a 404 where the
    profile should be).  ``lane_speedup_implied_by`` is what the engaged arm
    CAN say.
    """
    if not lane_share or not 0 < lane_share < 1:
        return None
    return round(1.0 / (1.0 - lane_share), 4)


def lane_speedup_implied_by(served_ratio: float | None,
                            engaged_lane_share: float | None) -> float | None:
    """What a served step ratio demands of the lane bucket: ``1 + (X-1)/s_e``.

    This is the exchange rate between the number #109 asks for and the kernel's
    own worth, and it is knowable from the engaged arm alone.  On the #83 arms
    (``s_e = 0.2979``) a served 1.15x means the built kernel beat the torch
    fallback 1.50x on the work it owns -- not that it underperformed -- and the
    2026-09-03 pair's 8.024x would demand 24.6x, which is why that pair is
    refused as a lane result.
    """
    if not served_ratio or not engaged_lane_share:
        return None
    if not 0 < engaged_lane_share < 1:
        return None
    return round(1.0 + (served_ratio - 1.0) / engaged_lane_share, 4)


def _decode_steps(receipt: dict) -> int | None:
    prof = receipt.get("profiled_load") or {}
    dec = prof.get("decode") or {}
    if "requests" in dec and "max_tokens" in dec:
        return int(dec["requests"]) * int(dec["max_tokens"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engaged", required=True, help="the arm whose lane is built (armA)")
    ap.add_argument("--fallback", required=True, help="the arm without it (armB)")
    ap.add_argument("--engaged-trace", default=None)
    ap.add_argument("--fallback-trace", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = json.load(open(args.engaged))
    B = json.load(open(args.fallback))
    for name, r in (("engaged", A), ("fallback", B)):
        if r.get("serve_mode") != A.get("serve_mode") or r.get("forward") != A.get("forward"):
            raise SystemExit(
                f"the two receipts are not the same configuration: engaged is "
                f"{A.get('serve_mode')}/{A.get('forward')}, {name} is "
                f"{r.get('serve_mode')}/{r.get('forward')}")

    ratios = {}
    for window, metric in METRICS:
        wa = (A.get("windows") or {}).get(window) or {}
        wb = (B.get("windows") or {}).get(window) or {}
        got = _ratio(_metric(wb, metric), _metric(wa, metric))
        if got:
            ratios[f"{window}.{metric}"] = got

    cont = {"engaged": _contention(A), "fallback": _contention(B)}
    reasons = []
    for name, c in cont.items():
        if c["contended_label"]:
            reasons.append(f"{name} receipt labels itself contended "
                           f"(load1/cpu {c['max_load1_per_cpu']}, "
                           f"swap resident {c['swap_resident_gib']} GiB)")
        if c["swap_moved_during_windows"]:
            reasons.append(f"{name} swapped pages during its timed windows")
        if c["swap_moved_during_windows"] is None:
            reasons.append(f"{name} receipt predates the swap-activity field and "
                           "cannot say whether the box was paging")
    evidence = not reasons

    traces = {}
    for name, path, receipt in (("engaged", args.engaged_trace, A),
                                ("fallback", args.fallback_trace, B)):
        f = _trace_file(path) if path else None
        if not f:
            traces[name] = {"note": "no trace given or found"}
            continue
        mu = receipt.get("marks_unix") or {}
        t0, t1 = mu.get("profile_decode_start"), mu.get("profile_decode_end")
        if t0 is None or t1 is None:
            traces[name] = {"file": f, "note": "receipt carries no profiled-decode "
                            "marks (schema /1); the trace cannot be cut by wall clock"}
            continue
        w = trace_window(f, float(t0), float(t1))
        steps = _decode_steps(receipt)
        if steps:
            w["decode_steps"] = steps
            w["device_ms_per_step"] = round(w["device_ms_in_window"] / steps, 5)
            lane = sum(w["by_bucket"].get(k, {}).get("ms", 0.0)
                       for k in ("window_gemv", "window_decode", "scaled_mm/cutlass"))
            w["lane_bucket_ms"] = round(lane, 3)
            share = (lane / w["device_ms_in_window"]
                     if w["device_ms_in_window"] else None)
            w["lane_share_of_window"] = round(share, 4) if share else None
            w["headroom_if_lane_were_free"] = headroom_if_lane_were_free(share)
            if name == "fallback":
                # Only from THIS arm is it the A/B's ceiling; see the docstring.
                w["ab_ceiling_from_this_arm"] = w["headroom_if_lane_were_free"]
        traces[name] = w

    # THE EXCHANGE RATE, beside the ratio rather than after it.  A served X on
    # a step whose lane is share s_e implies the lane itself moved
    # 1 + (X-1)/s_e.  Reported for every served ratio taken in the decode
    # window, because "1.15x served" and "1.50x on the kernel's own work" are
    # the same measurement and a reader should not have to do the algebra to
    # see it -- nor discover it only once the served number disappoints.
    implied = {}
    s_e = (traces.get("engaged") or {}).get("lane_share_of_window")
    for k, v in ratios.items():
        if not k.startswith("decode."):
            continue
        got = lane_speedup_implied_by(v["ratio_fallback_over_engaged"], s_e)
        if got is not None:
            implied[k] = got

    lane_ratio = None
    ea, fb = traces.get("engaged", {}), traces.get("fallback", {})
    if ea.get("lane_bucket_ms") and fb.get("lane_bucket_ms"):
        pa, pb = ea.get("decode_steps"), fb.get("decode_steps")
        if pa and pb:
            lane_ratio = {
                "engaged_lane_ms_per_step": round(ea["lane_bucket_ms"] / pa, 5),
                "fallback_lane_ms_per_step": round(fb["lane_bucket_ms"] / pb, 5),
                "ratio_fallback_over_engaged": round(
                    (fb["lane_bucket_ms"] / pb) / (ea["lane_bucket_ms"] / pa), 4)}

    out = {"schema": "tessera.window_gemv.latency_ratio/1",
           "configuration": {"serve_mode": A.get("serve_mode"),
                             "forward": A.get("forward")},
           "arms": {"engaged": {"file": args.engaged, "arm": A.get("arm"),
                                "marks_utc": A.get("marks_utc")},
                    "fallback": {"file": args.fallback, "arm": B.get("arm"),
                                 "marks_utc": B.get("marks_utc")}},
           "served_ratios": ratios,
           "contention": cont,
           "ratio_is_evidence": evidence,
           "not_evidence_because": reasons,
           "traces": traces,
           "engaged_lane_share_of_decode_window": s_e,
           "lane_speedup_implied_by_served_ratio": implied,
           "ab_ceiling": (traces.get("fallback") or {}).get("ab_ceiling_from_this_arm"),
           "lane_device_time_ratio": lane_ratio}

    print(f"== window GEMV latency A/B  {A.get('serve_mode')}/{A.get('forward')}")
    print("   engaged  =", os.path.basename(args.engaged))
    print("   fallback =", os.path.basename(args.fallback))
    print("   -- served, from the engine's own histograms --")
    for k, v in ratios.items():
        print(f"   {k:16s} engaged {v['engaged_ms']:>10.3f} ms   fallback "
              f"{v['fallback_ms']:>10.3f} ms   x{v['ratio_fallback_over_engaged']:.3f}"
              f"   (n {v['n_engaged']}/{v['n_fallback']}, {v['source']})")
    if implied:
        print(f"   -- what that demands of the lane, at s_e = {s_e} --")
        for k, v in implied.items():
            print(f"   {k:16s} implies the lane bucket moved x{v:.3f}")
    if out["ab_ceiling"]:
        print(f"   -- the A/B's Amdahl ceiling, from the FALLBACK arm's lane "
              f"share: x{out['ab_ceiling']:.3f}")
    if lane_ratio:
        print("   -- lane device time in the profiled decode window, per step --")
        print(f"   {'lane buckets':16s} engaged {lane_ratio['engaged_lane_ms_per_step']:>10.5f} ms"
              f"   fallback {lane_ratio['fallback_lane_ms_per_step']:>10.5f} ms"
              f"   x{lane_ratio['ratio_fallback_over_engaged']:.3f}")
    for name, t in traces.items():
        if "by_bucket" not in t:
            print(f"   trace {name}: {t.get('note')}")
            continue
        print(f"   trace {name}: {t['kernels_in_window']} of {t['kernels_in_trace']} "
              f"kernels inside the cut, {t['device_ms_in_window']} ms device")
        for k, v in t["by_bucket"].items():
            print(f"      {k:22s} {v['launches']:>7d} launches {v['ms']:>10.3f} ms")
    print("   -- contention --")
    for name, c in cont.items():
        print(f"   {name:9s} contended={c['contended_label']} "
              f"load1/cpu {c['max_load1_per_cpu']} "
              f"swap resident {c['swap_resident_gib']} GiB "
              f"moved={c['swap_moved_during_windows']}")
    if evidence:
        print("   VERDICT: both arms quiet by their own receipts; the ratio stands "
              "as a lane result.")
    else:
        print("   VERDICT: NOT EVIDENCE about the lane --")
        for r in reasons:
            print("     -", r)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
