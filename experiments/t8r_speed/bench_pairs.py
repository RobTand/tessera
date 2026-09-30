"""Synthetic run pairs through the fused window kernel, at GLM-5.3-Flash TP2 shapes.

The T8R artifact's routed stacks carry three pairs (R1024 = rate 4, one run;
R1088 = rates 4 and 5, a quarter of the columns at 5; R832 = rates 3 and 4, a
quarter at 4), so the artifact alone cannot say whether a two-run stack costs
more than a one-run one because of its second run or because of its odd rate.
This bench builds random stacks at any pair and times the raw launch
(``routed_fused_forward``) of each mode on the same balanced routing:

* ``r4``      rate 4, one run (R1024's pair);
* ``r4+1b``   rates 4 and 5, one 32-column block's worth of columns at 5,
              spread over the blocks (the two-run code path at nearly R1024's
              bytes);
* ``r4+q``    rates 4 and 5, a quarter at 5 (R1088's pair);
* ``r5-1b``   rates 4 and 5, all but 32 columns at 5;
* ``r5``      rate 5, one run;
* ``r3``, ``r3+q`` rates 3 and 4 (R832's pair), ``r4`` again as its top.

Words, tables, start states and scales are random: the decode's arithmetic and
memory traffic do not depend on the values, except the table gathers' bank
conflicts, which random 14-bit windows make uniform.  Outputs are hashed so two
kernel arms can be compared bitwise on the same inputs.

The library is the E4M3 family's in this process (``TESSERA_FUSED_E4M3_MMA``)
unless ``--library`` names one; ``--bm 128`` runs the wide superblock where the
library has it (tessera#741).

Usage: bench_pairs.py --out DIR [--cases r4,r4+q] [--modes 0,2] [--ms 1,8,512,2048]
                      [--library e4m3|e4m3mma] [--bm 64|128] [--ncu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_t8r import (EXPERTS, HIDDEN, TOP_K, PowerSampler, balanced_routing,  # noqa: E402
                       summarize, time_events)

INTER = 1024          # the TP2 rank's intermediate columns
BK = 32

# case -> (r_lo, fraction of columns at r_lo + 1: None = one run; "1b" = 32
# columns; "-1b" = all but 32)
CASES = {
    "r4": (4, None), "r4+1b": (4, "1b"), "r4+q": (4, 0.25), "r5-1b": (4, "-1b"), "r5": (5, None),
    "r3": (3, None), "r3+q": (3, 0.25), "r3+1b": (3, "1b"),
}


def n_hi_of(spec, cols):
    if spec is None:
        return 0
    if spec == "1b":
        return BK
    if spec == "-1b":
        return cols - BK
    return int(cols * spec)


def build_projection(rf, e, rows, cols, r_lo, n_hi, gen, dev, mma8=False):
    """Random words/tables/init/scales for one projection, and its run tables."""
    n_lo = cols - n_hi
    two = n_hi > 0
    pair = torch.tensor((r_lo, 0, n_lo, 0, r_lo + 1 if two else 0, n_lo, n_hi, 16 * n_lo * r_lo),
                        dtype=torch.int32)
    tile_words = rf.pair_tile_words(pair)
    words = torch.randint(-2**31, 2**31 - 1, (e, (rows // 512) * tile_words), generator=gen,
                          dtype=torch.int64).to(torch.int32)
    if mma8:
        # E4M3 bytes; the two NaN codes move to the largest finite magnitude
        table = torch.randint(0, 256, (e, 1 << 14), generator=gen, dtype=torch.int32)
        table = torch.where((table & 0x7F) == 0x7F, table - 1, table).to(torch.uint8)
    else:
        table = torch.randint(-2**15, 2**15 - 1, (e, 1 << 14), generator=gen, dtype=torch.int32).to(torch.int16)
    init = torch.randint(-2**31, 2**31 - 1, (e, cols), generator=gen, dtype=torch.int64).to(torch.int32)
    has_init = torch.ones(e, dtype=torch.int32)
    scale = torch.rand(e, rows, generator=gen) * 1e-2 + 1e-3
    # per expert: n_hi random columns at the high rate; the packer's stable sort
    perm = torch.empty(e, cols, dtype=torch.int64)
    for i in range(e):
        hi = torch.zeros(cols, dtype=torch.bool)
        if n_hi:
            hi[torch.randperm(cols, generator=gen)[:n_hi]] = True
        cidx = torch.arange(cols)
        perm[i] = torch.cat([cidx[~hi], cidx[hi]])
    bdesc = rf.block_desc(perm, n_lo, cols)
    runs = pair.reshape(1, 8).expand(e, 8).contiguous()
    return {"words": words.to(dev), "table": table.to(dev), "init": init.to(dev), "has_init": has_init.to(dev),
            "scale": scale.to(dev), "runs": runs.to(dev), "bdesc": bdesc.to(dev), "tile_words": tile_words,
            "slot_words": rf.slot_words_for_pair(pair), "bytes_per_expert": 4 * words.shape[1]}


def routing_tables(ids, w, e, bm):
    tokens, top_k = ids.shape
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(e, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = torch.zeros(e + 1, dtype=torch.int32, device=ids.device)
    offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    order = torch.argsort(flat, stable=True)
    item_off = torch.zeros(e + 1, dtype=torch.int32, device=ids.device)
    item_off[1:] = torch.cumsum((counts + bm - 1) // bm, 0, dtype=torch.int32)
    return offsets, order.to(torch.int32).contiguous(), w.reshape(-1)[order].contiguous(), item_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--modes", default="0,2")
    ap.add_argument("--ms", default="1,8,512,2048")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--power-s", type=float, default=2.0)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--library", default=None, help="a routed_fused.LIBRARIES key of the E4M3 family")
    ap.add_argument("--bm", type=int, default=64, help="routes per superblock: 64, or 128 where the launch has it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from tessera import routed_fused as rf
    library = args.library or rf.library_for("e4m3")
    mma8 = rf.library_mma8(library)
    lib = rf._ext(library)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    power = PowerSampler()
    ms = [int(v) for v in args.ms.split(",")]
    modes = [int(v) for v in args.modes.split(",")]
    meta = {"device": torch.cuda.get_device_name(), "sms": sms, "experts": EXPERTS, "hidden": HIDDEN,
            "inter": INTER, "top_k": TOP_K, "ms": ms, "modes": modes, "kernel_sha": os.environ.get("KERNEL_SHA"),
            "tessera_head": os.environ.get("TESSERA_HEAD"), "image": os.environ.get("ORACLE_IMAGE"),
            "host": os.environ.get("HOST_NAME"), "power_source": power.source, "start_unix": time.time(),
            "library": library, "bm": args.bm}
    results = []
    for case in args.cases.split(","):
        r_lo, spec = CASES[case]
        for mode in modes:
            rows, cols = (INTER, HIDDEN) if mode == 0 else (HIDDEN, INTER)
            gen = torch.Generator().manual_seed(zlib.crc32(f"{case}:{mode}".encode()))
            n_hi = n_hi_of(spec, cols)
            bm = args.bm if rf.has_width(library, mode, args.bm) else rf.BM
            projs = [build_projection(rf, EXPERTS, rows, cols, r_lo, n_hi, gen, dev, mma8)
                     for _ in range(2 if mode == 0 else 1)]
            p0, p1 = projs[0], projs[-1]
            rec = {"case": case, "mode": mode, "r_lo": r_lo, "n_hi": n_hi, "cols": cols, "rows": rows,
                   "tile_words": p0["tile_words"], "slot_words": p0["slot_words"],
                   "rate_mean": p0["tile_words"] / 16 / cols, "bm": bm, "cells": {}}
            for m in ms:
                ids, w = balanced_routing(m, dev)
                offsets, flat_sorted, rw_sorted, item_off = routing_tables(ids, w, EXPERTS, bm)
                routes = m * TOP_K
                g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"x:{case}:{mode}:{m}".encode()))
                xrows = m if mode == 0 else routes
                x = (torch.randn(xrows, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
                a_scale = torch.rand(xrows, device=dev, generator=g) * 0.1 + 0.01
                out = torch.empty((routes, rows), dtype=torch.bfloat16, device=dev)
                counter = torch.zeros(1, dtype=torch.int32, device=dev)
                slot_words = max(p["slot_words"] for p in projs)

                def call():
                    counter.zero_()
                    lib.routed_fused_forward(
                        mode, True, x, a_scale, p0["words"], p1["words"], p0["table"], p1["table"],
                        p0["init"], p1["init"], p0["has_init"], p1["has_init"], p0["scale"], p1["scale"],
                        p0["runs"], p1["runs"], p0["bdesc"], p1["bdesc"], p0["tile_words"], slot_words,
                        offsets, flat_sorted, rw_sorted, item_off, counter, TOP_K,
                        0 if mode == 0 else 1, mode == 2, float("inf"), out, sms, bm)
                if args.ncu:
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    call()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    rec["cells"][str(m)] = {"ncu": True}
                    print(json.dumps({"case": case, "mode": mode, "M": m, "ncu": True}), flush=True)
                    continue
                call()
                torch.cuda.synchronize()
                cell = {"out_sha256": hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
                cell["wall"] = summarize(time_events(call, args.warmup, args.iters))
                cell["power"] = power.sample_during(call, args.power_s)
                touched = min(EXPERTS, routes)
                cell["weight_bytes"] = touched * sum(p["bytes_per_expert"] for p in projs)
                cell["gbps"] = cell["weight_bytes"] / (cell["wall"]["median_ms"] * 1e-3) / 1e9
                rec["cells"][str(m)] = cell
                print(json.dumps({"case": case, "mode": mode, "M": m, "ms": round(cell["wall"]["median_ms"], 4),
                                  "W": round(cell["power"].get("mean_w") or 0, 1),
                                  "GBps": round(cell["gbps"], 1)}), flush=True)
            results.append(rec)
            del projs, p0, p1
            torch.cuda.empty_cache()
            json.dump({"meta": meta, "results": results}, open(os.path.join(args.out, "bench_pairs.json"), "w"),
                      indent=1)
    meta["end_unix"] = time.time()
    json.dump({"meta": meta, "results": results}, open(os.path.join(args.out, "bench_pairs.json"), "w"), indent=1)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
