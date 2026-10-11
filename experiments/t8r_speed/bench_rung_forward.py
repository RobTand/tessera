"""The routed E4M3 forward at any q256 rung, on synthetic wires (tessera#694).

``bench_t8r.py`` times the three routed rungs the T8R release artifact carries
(R832, R1024 and R1088).  The allocator may pick any rung between and beyond
them, and #694 asks for R832 to R1088, plus R768 and R1152, within 1.5x of
R1024.  This bench builds the production routed owner at every rung it is
given and times the forward the serve calls, so each rung is measured and none
is read across from a neighbour.

Shapes are the TP2 rank's: 288 experts, hidden 4096, 1024 intermediate
columns, top 8, balanced routing (token t picks experts (8t + j) mod 288).  A
rung's column placement is the grammar's own Bresenham schedule
(``grammar.bresenham_rate_schedule``, the call the encoder makes), so its run
table, column order, block descriptors, tile stride and word count are the
wire's.  Words, tables, start states and scales are random: the decode's
arithmetic and memory traffic do not depend on the values.  R1024 takes the
piece-major word order wherever the intake gives it
(``moe_route._piece_major_admissible``); every other rung keeps legacy words,
as the loader does.

The owner is ``PackedWindowMoeBundles(...).adapter()`` and the call is the
one ``bench_t8r.build_routed`` returns (the storage-id map, the SwiGLU limit,
no router weight on the input), so a cell here and a cell of ``bench_t8r.py``
time the same code.  Every owner stays resident, and one process runs every
rung, so the rungs share the device state.

Per (rung, M) cell, a forward pass records the output's sha256, the eager
CUDA-event wall time (median of ``--iters`` calls), the call replayed from a
captured CUDA graph (M <= 8, the FULL_DECODE_ONLY proxy), the torch.profiler
kernel table, and NVML board power over a ``--power-s`` loop of back-to-back
calls.  A reverse pass walks the rungs and the Ms the other way and records
wall and graph time again.  A cell's time is the mean of the two passes'
medians, and its spread is |F - R| / mean, so a drift in clock or temperature
over the job moves both halves of a comparison equally.  The summary divides
each rung's time by the reference rung's at the same M.

``--ncu`` runs one call per (rung, M) between cudaProfilerStart and
cudaProfilerStop and times nothing, for ``bench_t8r.sh`` with BENCH_NCU=1.
``--cpu-preflight`` builds a tiny owner on the CPU through the same code and
checks its run tables against the lane's own wire checks.

Usage: bench_rung_forward.py --out DIR [--rungs 768,832,896,928,960,1024,1088,1152]
       [--reference 1024] [--ms 1,2,3,4,5,6,7,8,512,2048,8192] [--ncu | --cpu-preflight]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from fractions import Fraction

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rung_summary import BAR, rungs_over_the_bar, summarize_rungs  # noqa: E402

EXPERTS, TOP_K, HIDDEN, INTER, SWIGLU_LIMIT = 288, 8, 4096, 1024, 10.0
DEFAULT_RUNGS = (768, 832, 896, 928, 960, 1024, 1088, 1152)
DEFAULT_MS = (1, 2, 3, 4, 5, 6, 7, 8, 512, 2048, 8192)
GRAPH_M_MAX = 8
#: The rate bound of the routed launches: every rate 1..8 is a case.
RATE_CAP = 8


def rung_geometry(q256, cols):
    """The run split and column order of one rung over ``cols`` columns.

    ``rates`` is the grammar's Bresenham schedule, in original column order.
    ``perm`` is the packer's column order: the original columns sorted by
    (rate, column), the low-rate run first.  ``n_hi`` is the count of columns
    at ``r_lo + 1`` (0 for a whole rate).  The schedule must realise the rung
    exactly, so the stack carries the bytes the rung prices.
    """
    from tessera.grammar import bresenham_rate_schedule

    rates = tuple(bresenham_rate_schedule(Fraction(q256, 256), cols, cap=RATE_CAP))
    r_lo = min(rates)
    n_hi = sum(1 for rate in rates if rate != r_lo)
    if set(rates) - {r_lo, r_lo + 1}:
        raise ValueError(f"q256 {q256} over {cols} columns mixes more than two adjacent rates: {sorted(set(rates))}")
    if sum(rates) * 256 != q256 * cols:
        raise ValueError(f"q256 {q256} is not exact over {cols} columns")
    perm = sorted(range(cols), key=lambda c: (rates[c], c))
    return {"q256": q256, "r_lo": r_lo, "n_hi": n_hi, "rates": rates, "perm": perm}


def piece_major_for(q256):
    """Whether the intake lays this rung's words piece-major.

    The intake's own gate (``moe_route._piece_major_admissible``) over the one
    stack rung it admits: uniform R4, rung 1024.
    """
    if q256 != 1024:
        return False
    from tessera.serving.moe_route import _piece_major_admissible

    return bool(_piece_major_admissible("e4m3"))


def build_owner(q256, *, experts, hidden, inter, seed, device, piece_major):
    """The routed owner of one rung: three grouped bundles on random words."""
    from bench_rates import build_projection
    from tessera import routed_fused as rf
    from tessera.native_window_moe import PackedWindowMoeBundles
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    bundles, per_expert_bytes = [], 0
    for role, rows, cols in ((0, inter, hidden), (1, inter, hidden), (2, hidden, inter)):
        geometry = rung_geometry(q256, cols)
        two = geometry["n_hi"] > 0
        p = build_projection(rf, experts, rows, cols, geometry["r_lo"], geometry["n_hi"],
                             seed + 100 * role, device, True)
        if p["window_bits"] != rf.WINDOW_BITS:
            raise ValueError(f"the recipe at q256={q256} does not fit the native window owner")
        perm = torch.tensor(geometry["perm"], dtype=torch.int32, device=device).expand(experts, cols).contiguous()
        runs = p["runs"].reshape(experts * 2, 4) if two else p["runs"][:, :4].contiguous()
        widths = torch.full((experts,), p["words"].shape[1], dtype=torch.int32, device=device)
        offsets = torch.arange(experts, dtype=torch.int32, device=device) * widths[0]
        native = torch.arange(256, dtype=torch.int32, device=device)
        native[(native & 127) == 127] -= 1
        native = native.to(torch.uint8).expand(experts, 256).contiguous()
        bundles.append(prepare_grouped_window_gemm_from_soa(
            words_all=p["words"].flatten(), table_all=torch.empty(0, dtype=torch.uint8, device=device),
            codes_all=p["table"], native_all=native, scale_all=p["scale"], runs_all=runs,
            init_all=p["init"], has_init=p["has_init"], word_off=offsets,
            tile_words=torch.full_like(widths, p["tile_words"]), total_words=widths,
            run_off=torch.arange(experts + 1, dtype=torch.int32, device=device) * (2 if two else 1),
            perm_all=perm, rows=rows, cols=cols, experts=experts, window_bits=p["window_bits"],
            family="e4m3", word_layout="piece_major" if piece_major else "legacy"))
        per_expert_bytes += p["bytes_per_expert"]
        del p
    classes = [{"start": 0, "end": experts, "q256": {"w13": [q256, q256], "w2": [q256]}}]
    return PackedWindowMoeBundles(*bundles, family="e4m3", expert_classes=classes), per_expert_bytes


def cpu_preflight(rungs):
    """Tiny owners on the CPU: the bundles pass the lane's own run-table checks.

    The device refusal comes first in the lane's predicate, so this calls the
    wire checks every fused lane makes (``routed_fused._run_stack_reason``)
    directly.  Nothing here launches a kernel.
    """
    from tessera import routed_fused as rf

    experts, hidden, inter = 24, 256, 128
    done = []
    for q in rungs:
        packed, _bytes = build_owner(q, experts=experts, hidden=hidden, inter=inter, seed=1,
                                     device=torch.device("cpu"), piece_major=False)
        for name in ("gate", "up", "down"):
            why = rf._run_stack_reason(name, getattr(packed, name), experts)
            if why is not None:
                raise AssertionError(f"q256 {q}: {why}")
        done.append(q)
    return {"status": "cpu_preflight_passed", "rungs": done, "tiny_geometry": [experts, hidden, inter]}


def emit(record):
    print(json.dumps(record), flush=True)


def graph_samples(call, warmup, iters, time_events):
    """The call replayed from a captured CUDA graph, or the error that refused it."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        samples = time_events(graph.replay, warmup, iters)
        replay = graph.replay
        return samples, None, replay
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, repr(exc)[:400], call


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default=",".join(str(q) for q in DEFAULT_RUNGS))
    ap.add_argument("--reference", type=int, default=1024)
    ap.add_argument("--ms", default=",".join(str(m) for m in DEFAULT_MS))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--prof-reps", type=int, default=5)
    ap.add_argument("--power-s", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=694)
    ap.add_argument("--ncu", action="store_true",
                    help="one call per (rung, M) between cudaProfilerStart/Stop; no timing")
    ap.add_argument("--cpu-preflight", action="store_true")
    args = ap.parse_args()
    rungs = [int(q) for q in args.rungs.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    if args.reference not in rungs and not args.ncu:
        ap.error("--reference must be one of --rungs")
    os.makedirs(args.out, exist_ok=True)
    if args.cpu_preflight:
        result = cpu_preflight(rungs)
        print(json.dumps(result), flush=True)
        return 0

    from bench_rates import Clock, sha
    from bench_t8r import (ENVELOPE_W, PowerSampler, _install_vllm_stubs, balanced_routing, kernel_profile,
                           summarize, time_events)
    from tessera import routed_fused as rf
    from tessera.expert_classes import storage_expert_ids

    vllm_stubbed = _install_vllm_stubs()
    dev = torch.device("cuda")
    power, clock = PowerSampler(), Clock()
    meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "experts": EXPERTS,
            "top_k": TOP_K, "hidden": HIDDEN, "inter": INTER, "ms": ms, "rungs": rungs,
            "reference": args.reference, "warmup": args.warmup, "iters": args.iters, "power_s": args.power_s,
            "prof_reps": args.prof_reps, "power_source": power.source, "envelope_w": ENVELOPE_W,
            "vllm_stubbed": vllm_stubbed, "kernel_sha": os.environ.get("KERNEL_SHA"),
            "tessera_head": os.environ.get("TESSERA_HEAD"), "tessera_state": os.environ.get("TESSERA_STATE"),
            "image": os.environ.get("ORACLE_IMAGE"), "pb_action": os.environ.get("PB_ACTION_KEY"),
            "host": os.environ.get("HOST_NAME"), "e4m3_mma": os.environ.get("TESSERA_FUSED_E4M3_MMA"),
            "statistic": "mean of the forward and reverse passes' medians; spread = |F - R| / mean",
            "bar": BAR, "start_unix": time.time()}
    doc = {"meta": meta, "rungs": {}}
    path = os.path.join(args.out, "bench_rung_forward.json")

    def save():
        with open(path + ".tmp", "w") as handle:
            json.dump(doc, handle, indent=1)
        os.replace(path + ".tmp", path)

    owners = {}
    with torch.inference_mode():
        for q in rungs:
            t0 = time.time()
            layout = piece_major_for(q)
            packed, per_expert_bytes = build_owner(q, experts=EXPERTS, hidden=HIDDEN, inter=INTER,
                                                   seed=args.seed + q, device=dev, piece_major=layout)
            adapter = packed.adapter()
            if type(adapter).__name__ != "FusedRoutedWindowMoE":
                raise RuntimeError(f"q256 {q}: the owner did not select the fused routed adapter")
            library_file = os.path.realpath(rf._ext(adapter.library).__file__)
            with open(library_file, "rb") as handle:
                library_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            owners[q] = (adapter, packed, per_expert_bytes)
            geometry = rung_geometry(q, HIDDEN)
            doc["rungs"][str(q)] = {
                "q256": q, "r_lo": geometry["r_lo"], "n_hi_hidden": geometry["n_hi"],
                "n_hi_inter": rung_geometry(q, INTER)["n_hi"], "adapter": type(adapter).__name__,
                "library": adapter.library, "library_sha256": library_sha, "launch_pair": list(adapter.launch_pair),
                "word_layouts": {role: getattr(packed, role).word_layout for role in ("gate", "up", "down")},
                "resident_bytes": int(packed.resident_bytes()), "wire_bytes_per_expert_rank": per_expert_bytes,
                "build_s": time.time() - t0, "cells": {}}
            emit({"built": q, "layout": layout, "build_s": round(time.time() - t0, 1),
                  "resident_bytes": doc["rungs"][str(q)]["resident_bytes"]})
        inverse = torch.arange(EXPERTS, dtype=torch.int32, device=dev)

        def inputs(m):
            torch.manual_seed(zlib.crc32(f"x:{m}".encode()))
            x = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16)
            ids, w = balanced_routing(m, dev)
            return x, ids, w

        def caller(adapter, x, ids, w):
            def call():
                return adapter(x, storage_expert_ids(inverse, ids), w, swiglu_limit=SWIGLU_LIMIT,
                               apply_router_weight_on_input=False)
            return call

        if args.ncu:
            for q in rungs:
                adapter = owners[q][0]
                for m in ms:
                    x, ids, w = inputs(m)
                    call = caller(adapter, x, ids, w)
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    call()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    doc["rungs"][str(q)]["cells"][str(m)] = {"ncu": True}
                    emit({"rung": q, "M": m, "ncu": True})
            meta["end_unix"] = time.time()
            save()
            return 0

        for order in ("F", "R"):
            for q in (rungs if order == "F" else list(reversed(rungs))):
                adapter, packed, per_expert_bytes = owners[q]
                for m in (ms if order == "F" else list(reversed(ms))):
                    x, ids, w = inputs(m)
                    call = caller(adapter, x, ids, w)
                    cell = doc["rungs"][str(q)]["cells"].setdefault(str(m), {})
                    rec = {"clock_start": clock.read()}
                    y = call()
                    torch.cuda.synchronize()
                    if order == "F":
                        rec["out_sha256"] = sha(y)
                        rec["routing"] = {"experts_touched": min(EXPERTS, m * TOP_K), "routes": m * TOP_K}
                        rec["bytes"] = min(EXPERTS, m * TOP_K) * per_expert_bytes
                    del y
                    start = time.time()
                    rec["wall"] = summarize(time_events(call, args.warmup, args.iters))
                    rec["wall_window_unix"] = [start, time.time()]
                    replay = call
                    if m <= GRAPH_M_MAX:
                        samples, error, replay = graph_samples(call, args.warmup, args.iters, time_events)
                        if samples is not None:
                            rec["graph"] = summarize(samples)
                        else:
                            rec["graph_error"] = error
                    if order == "F":
                        rec["profile"] = kernel_profile(call, reps=args.prof_reps)
                        rec["power"] = power.sample_during(replay, args.power_s)
                    rec["clock_end"] = clock.read()
                    cell[order] = rec
                    emit({"pass": order, "rung": q, "M": m, "wall_ms": round(rec["wall"]["median_ms"], 4),
                          "graph_ms": round(rec["graph"]["median_ms"], 4) if "graph" in rec else None,
                          "W": round(rec.get("power", {}).get("mean_w") or 0, 1)})
                    del x, ids, w
                save()
        doc["summary"] = summarize_rungs(doc, args.reference)
        doc["over_the_bar"] = {key: rungs_over_the_bar(doc["summary"], args.reference, key)
                               for key in ("wall", "graph", "kernel")}
        meta["end_unix"] = time.time()
        meta["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
        for m, row in doc["summary"].items():
            emit({"summary": int(m), **{rung: {k: round(v["ratio"], 3) for k, v in entry.items()
                                               if isinstance(v, dict) and "ratio" in v}
                                        for rung, entry in row.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
