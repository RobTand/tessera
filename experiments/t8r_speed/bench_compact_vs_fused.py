"""The 16-bit routed experts at rates 6-8: the compact adapter against the fused launch.

Until contract v48 the value family's fused gate/up launch reached rates 1..6,
so a Tessera-16 expert stack at a rate-7 or rate-8 run table was served by the
compact adapter (``native_window_moe.NativeWindowMoE``: two grouped window
GEMMs around the activation).  Since v48 the fused launch takes those tables
at two word stages.  This bench times both adapters on the SAME stacks and the
same routing, so the route change is measured, not inferred:

* ``compact``: ``native_window_moe_from_bundles(down, gate=gate, up=up)``, the
  adapter ``PackedWindowMoeBundles.adapter`` substitutes when the fused lane
  refuses a stack;
* ``fused``: ``routed_fused.FusedRoutedWindowMoE.from_bundles``.

Shapes are the GLM-5.3-Flash TP2 rank's: 288 experts, top-8, hidden 4096, 1024
intermediate columns.  The stacks hold ``--distinct`` encoded experts, repeated
to 288 (each copy is its own stacked words, so the memory traffic is 288
experts'; the values do not change the kernels' work).  Routing is balanced
(token t picks experts (8t + j) mod 288).  Bundles are prepared at the serving
loader's blocks (64 x 64 x 64, folded arithmetic).

Each (case, arm, M) cell is timed in a forward and a reverse pass (graph
replay), and the forward pass records torch.profiler device time per kernel
and, at ``--power-ms``, NVML board power over a back-to-back loop.

Usage: bench_compact_vs_fused.py --out DIR [--cases 1536,1792,2048]
       [--ms 1,64,512,8192] [--distinct 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from fractions import Fraction

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))

try:
    import pytest  # noqa: F401  (the test module imports it; the image may not carry it)
except ImportError:
    import types

    class _Mark:
        def __getattr__(self, name):
            def deco(*a, **k):
                return a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f)
            return deco

    _stub = types.ModuleType("pytest")
    _stub.mark = _Mark()
    _stub.fixture = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
    _stub.param = lambda *a, **k: a
    sys.modules["pytest"] = _stub

from bench_geometry import time_call                                       # noqa: E402
from bench_t8r import ENVELOPE_W, PowerSampler, balanced_routing, kernel_profile  # noqa: E402
import test_window_gemm_grouped as twgg                                    # noqa: E402

EXPERTS, TOP_K, HIDDEN, INTER, SWIGLU_LIMIT = 288, 8, 4096, 1024, 10.0


def emit(obj):
    print(json.dumps(obj), flush=True)


def sched(cols, q256):
    from tessera.grammar import bresenham_rate_schedule
    return bresenham_rate_schedule(Fraction(q256, 256), cols, cap=8)


def build(q256, distinct, seed=500):
    """The value-family bundles of one q256 rung, and both adapters."""
    from tessera import routed_fused as rf
    from tessera import window_gemm_grouped as wgg
    from tessera.native_window_moe import PackedWindowMoeBundles, native_window_moe_from_bundles

    # The reference states are the tests' oracle; the bench does not read them.
    twgg._states = lambda *a, **k: None
    r_h, r_i = sched(HIDDEN, q256), sched(INTER, q256)
    units = {
        "gate": [twgg.Expert(INTER, HIDDEN, r_h, seed + i, family="value").unit for i in range(distinct)],
        "up": [twgg.Expert(INTER, HIDDEN, r_h, seed + 100 + i, family="value").unit for i in range(distinct)],
        "down": [twgg.Expert(HIDDEN, INTER, r_i, seed + 200 + i, family="value").unit for i in range(distinct)],
    }
    bundles = {name: wgg.prepare_grouped_window_gemm([u[e % distinct] for e in range(EXPERTS)],
                                                     block_m=64, block_n=64, block_k=64)
               for name, u in units.items()}
    classes = [{"start": 0, "end": EXPERTS,
                "q256": {"w13": [q256, q256], "w2": [q256]}}]
    packed = PackedWindowMoeBundles(gate=bundles["gate"], up=bundles["up"], down=bundles["down"],
                                    family="value", expert_classes=classes)
    compact = native_window_moe_from_bundles(bundles["down"], gate=bundles["gate"], up=bundles["up"])
    reason = rf.fused_routed_window_supported(bundles["gate"], bundles["up"], bundles["down"])
    fused = None if reason else rf.FusedRoutedWindowMoE.from_bundles(bundles["gate"], bundles["up"],
                                                                     bundles["down"], expert_classes=classes)
    head = {"q256": q256, "rates_hidden": sorted(set(r_h)), "rates_inter": sorted(set(r_i)),
            "fused_refused": reason, "resident_bytes": int(packed.resident_bytes())}
    return head, {"compact": compact, "fused": fused}, (units, bundles, packed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="1536,1792,2048")
    ap.add_argument("--ms", default="1,64,512,8192")
    ap.add_argument("--distinct", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prof-reps", type=int, default=3)
    ap.add_argument("--power-ms", default="64,512,8192")
    ap.add_argument("--power-s", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    cases = [int(c) for c in args.cases.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    power_ms = {int(m) for m in args.power_ms.split(",") if m}
    power = PowerSampler()
    doc = {"meta": {"device": torch.cuda.get_device_name(), "experts": EXPERTS, "top_k": TOP_K,
                    "hidden": HIDDEN, "inter": INTER, "distinct": args.distinct, "family": "value",
                    "cases": cases, "ms": ms, "torch": torch.__version__,
                    "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
                    "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
                    "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": power.source,
                    "envelope_w": ENVELOPE_W, "start_unix": time.time(),
                    "statistic": "mean of the forward and reverse passes' medians (graph replay); "
                                 "spread = |F - R| / mean"},
           "cells": {}}
    path = os.path.join(args.out, "bench_compact_vs_fused.json")

    def save():
        with open(path + ".tmp", "w") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(path + ".tmp", path)

    xs = {m: (torch.randn(m, HIDDEN, generator=torch.Generator().manual_seed(m)) * 0.5)
          .to(dev, torch.bfloat16) for m in ms}
    routes = {m: balanced_routing(m, dev) for m in ms}
    for pas in ("F", "R"):
        for q in (cases if pas == "F" else list(reversed(cases))):
            head, arms, keep = build(q, args.distinct)
            if pas == "F":
                emit({"case": q, **head})
            for arm in (("compact", "fused") if pas == "F" else ("fused", "compact")):
                adapter = arms[arm]
                if adapter is None:
                    continue
                rec = doc["cells"].setdefault(f"{q}:{arm}", dict(head, arm=arm, cells={}))
                for m in (ms if pas == "F" else list(reversed(ms))):
                    x = xs[m]
                    ids, w = routes[m]
                    cell = rec["cells"].setdefault(str(m), {})
                    try:
                        def call(adapter=adapter, x=x, ids=ids, w=w):
                            return adapter(x, ids, w, swiglu_limit=SWIGLU_LIMIT)
                        timer, samples = time_call(call, args.warmup, args.iters)
                        cell[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                     "timer": timer, "unix": time.time()}
                        if pas == "F":
                            out = call()
                            torch.cuda.synchronize()
                            cell["out_sha256"] = hashlib.sha256(
                                out.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                            cell["profile"] = kernel_profile(call, reps=args.prof_reps)
                            if m in power_ms:
                                cell["power"] = power.sample_during(call, args.power_s)
                        else:
                            f, r = cell.get("F", {}).get("median_ms"), cell["R"]["median_ms"]
                            if f:
                                cell["ms"] = 0.5 * (f + r)
                                cell["spread"] = abs(f - r) / cell["ms"]
                    except Exception as exc:  # noqa: BLE001
                        cell[pas] = {"error": repr(exc)[:500]}
                        emit({"case": q, "arm": arm, "M": m, "pass": pas, "error": repr(exc)[:300]})
                        continue
                    line = {"case": q, "arm": arm, "M": m, "pass": pas, "ms": round(cell[pas]["median_ms"], 4)}
                    if "ms" in cell and pas == "R":
                        line.update(mean_ms=round(cell["ms"], 4), spread=round(cell["spread"], 4))
                    emit(line)
                save()
            del arms, keep
            torch.cuda.empty_cache()
    doc["meta"]["end_unix"] = time.time()
    save()
    for q in cases:
        for m in ms:
            c = doc["cells"].get(f"{q}:compact", {}).get("cells", {}).get(str(m), {}).get("ms")
            f = doc["cells"].get(f"{q}:fused", {}).get("cells", {}).get(str(m), {}).get("ms")
            if c and f:
                emit({"summary": q, "M": m, "compact_ms": round(c, 4), "fused_ms": round(f, 4),
                      "fused_over_compact": round(f / c, 3)})


if __name__ == "__main__":
    main()
