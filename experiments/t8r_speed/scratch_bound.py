"""Lower bound on a decode-once-to-scratch + grouped-GEMM routed path (TP2 rank, one layer).

A two-phase routed path decodes every touched expert's wire into a global scratch,
then runs a tensor-core grouped GEMM over that scratch.  Whatever its kernels, it
must at least read the wire once, write the decoded weights once, read them back
once, and do the GEMM's arithmetic.  This script measures the two rates that bound
it on one GB10:

* ``copy``: device-to-device copy bandwidth (bytes read + bytes written per second)
  at the scratch's size, the best rate a decode-write or a scratch-read can reach;
* ``gemm``: BF16 tensor-core GEMM throughput at the routed shapes, dense
  (every route of the step against one weight: the arithmetic with no grouping
  loss) and grouped over recorded per-expert route counts (per-expert ``mm``
  replayed from a CUDA graph, so launch gaps do not count).

and reports, per M, ``T_lower = max(traffic / copy_rate, flops / dense_rate)`` with
``traffic = wire + 2 x decoded`` bytes: memory and arithmetic perfectly overlapped,
nothing else charged.  The decoded scratch is taken at 1 byte per weight (the
E4M3 codes the priced contract upcasts exactly); a bf16 scratch doubles that term.

Usage: scratch_bound.py --out DIR [--routing DIR] [--ms 512,2048,4096,8192]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

EXPERTS, TOP_K, HIDDEN, INTER_RANK = 288, 8, 4096, 1024
# (name, K, N) per rank: gate/up is column-parallel (2 x 1024 rows), down row-parallel.
SHAPES = [("gate_up", HIDDEN, 2 * INTER_RANK), ("down", INTER_RANK, HIDDEN)]
DECODED_PER_EXPERT = sum(k * n for _, k, n in SHAPES)  # weights per expert per rank


def events(call, warmup=3, iters=10):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); call(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b))
    out.sort()
    return out[len(out) // 2]


def copy_rate(nbytes):
    a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    a.random_(0, 255)
    b = torch.empty_like(a)
    ms = events(lambda: b.copy_(a))
    del a, b
    torch.cuda.empty_cache()
    return {"bytes": nbytes, "ms": ms, "GBps_rw": 2 * nbytes / (ms * 1e-3) / 1e9}


def dense_rate(rows):
    out = {}
    for name, k, n in SHAPES:
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02
        ms = events(lambda: torch.mm(x, w))
        out[name] = {"rows": rows, "ms": ms, "TFLOPS": 2 * rows * k * n / (ms * 1e-3) / 1e12}
        del x, w
    return out


def grouped_ms(counts):
    """Per-expert mm over recorded route counts, replayed from one CUDA graph."""
    res = {}
    for name, k, n in SHAPES:
        w = torch.randn(EXPERTS, k, n, device="cuda", dtype=torch.bfloat16) * 0.02
        total = int(sum(counts))
        x = torch.randn(max(total, 1), k, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(max(total, 1), n, device="cuda", dtype=torch.bfloat16)
        offs = [0]
        for c in counts:
            offs.append(offs[-1] + int(c))

        def run():
            for e in range(EXPERTS):
                if counts[e]:
                    torch.mm(x[offs[e]:offs[e + 1]], w[e], out=y[offs[e]:offs[e + 1]])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        res[name] = events(g.replay)
        del w, x, y, g
        torch.cuda.empty_cache()
    return res


def load_counts(path):
    if path.endswith(".json"):
        root = os.path.dirname(os.path.dirname(path))
        ids = torch.cat([torch.load(os.path.join(root, q), map_location="cpu", weights_only=False)["ids"]
                         for q in json.load(open(path))], 0)
    else:
        ids = torch.load(path, map_location="cpu", weights_only=False)["ids"]
    return torch.bincount(ids.flatten().long(), minlength=EXPERTS).tolist(), int(ids.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--routing", default=None)
    ap.add_argument("--ms", default="512,2048,4096,8192")
    ap.add_argument("--wire-bytes", type=float, default=1.82e9,
                    help="wire bytes of all 288 experts on one rank (R1024 layer: 1.82e9, bench_t8r)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ms = [int(v) for v in args.ms.split(",")]
    decoded = EXPERTS * DECODED_PER_EXPERT  # bytes at 1 per weight
    rec = {"meta": {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "decoded_bytes_fp8": decoded, "wire_bytes": args.wire_bytes,
                    "start_unix": time.time(), "host": os.environ.get("HOST_NAME"),
                    "pb_action": os.environ.get("PB_ACTION_KEY")}}
    rec["copy"] = [copy_rate(int(b)) for b in (int(args.wire_bytes), decoded)]
    bw = max(c["GBps_rw"] for c in rec["copy"]) * 1e9
    print(json.dumps({"copy": rec["copy"]}), flush=True)
    rec["cells"] = {}
    for m in ms:
        routes = m * TOP_K
        dense = dense_rate(routes)
        flops = sum(2 * routes * k * n for _, k, n in SHAPES)
        tf = sum(d["ms"] for d in dense.values())
        cell = {"dense": dense, "dense_ms": tf, "flops": flops}
        files = []
        if args.routing and os.path.isdir(os.path.join(args.routing, f"m{m}")):
            d = os.path.join(args.routing, f"m{m}")
            files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith((".pt", ".json")))[:9]
        grouped = []
        for f in files:
            counts, mm = load_counts(f)
            assert mm == m, (f, mm, m)
            g = grouped_ms(counts)
            grouped.append({"file": os.path.basename(f), "ms": g, "sum_ms": sum(g.values()),
                            "experts": sum(1 for c in counts if c)})
        cell["grouped"] = grouped
        traffic = args.wire_bytes + 2 * decoded
        mem_ms = traffic / bw * 1e3
        cell["traffic_bytes"] = traffic
        cell["mem_ms"] = mem_ms
        cell["t_lower_ms"] = max(mem_ms, tf)
        gm = sorted(x["sum_ms"] for x in grouped)
        cell["t_lower_grouped_ms"] = max(mem_ms, gm[len(gm) // 2]) if gm else None
        rec["cells"][str(m)] = cell
        print(json.dumps({"M": m, "mem_ms": round(mem_ms, 2), "dense_gemm_ms": round(tf, 2),
                          "grouped_gemm_ms_median": round(gm[len(gm) // 2], 2) if gm else None,
                          "t_lower_ms": round(cell["t_lower_ms"], 2)}), flush=True)
        json.dump(rec, open(os.path.join(args.out, "scratch_bound.json"), "w"), indent=1)
    rec["meta"]["end_unix"] = time.time()
    json.dump(rec, open(os.path.join(args.out, "scratch_bound.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
