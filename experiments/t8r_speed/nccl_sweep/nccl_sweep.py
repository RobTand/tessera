#!/usr/bin/env python3
"""Two-box NCCL protocol and channel sweep on vLLM's own communicator.

Runs inside the serving image, one container per box, with the serve's NCCL
environment. Every configuration gets a fresh child process, because NCCL caches
some parameters (the channel counts) on first read.

Modes:
  box    one per box; walks the configuration list in lockstep with the peer
         and launches one child per configuration.
  child  one configuration: rendezvous, PyNcclCommunicator.from_unique_id_bytes,
         timed collectives, and an overlap probe; writes one JSON file.
  merge  stdlib only; joins both ranks' JSON into OUT/nccl-sweep.json with the
         pairwise-minimum time per collective (the transfer estimate).

Timing: each isolated sample starts after a tiny all-reduce plus a host
synchronize on both ranks, so the sample holds the transfer plus the host
skew. The merge takes the minimum across ranks per sample index. The streamed
figure is 20 back-to-back calls over one event pair.

Overlap probe: a BF16 GEMM [2048,4096]x[4096,4096] loop on the main stream,
with 16.8 MB all-reduces on a side stream. It reports each side's slowdown
against running alone. This is the prerequisite measurement for chunked
collective overlap.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Ordered by value, so a deadline cut drops the least useful rows.
CONFIGS = [
    ("auto-c64", {}),
    ("Simple-c64", {"NCCL_PROTO": "Simple"}),
    ("LL-c64", {"NCCL_PROTO": "LL"}),
    ("LL128-c64", {"NCCL_PROTO": "LL128"}),
    ("Simple-c8", {"NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"}),
    ("LL-c8", {"NCCL_PROTO": "LL", "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"}),
    ("Simple-c4", {"NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}),
    ("Simple-c16", {"NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"}),
    ("LL-c16", {"NCCL_PROTO": "LL", "NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"}),
    ("LL-c4", {"NCCL_PROTO": "LL", "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}),
    ("Simple-Tree-c64", {"NCCL_PROTO": "Simple", "NCCL_ALGO": "Tree"}),
    ("auto-c16", {"NCCL_MIN_NCHANNELS": "16", "NCCL_MAX_NCHANNELS": "16"}),
    ("auto-c8", {"NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8"}),
]
HIDDEN = 4096
TOKENS = 2048          # MNBT 2048: one prefill chunk
ITERS, WARMUP, STREAM_N = 20, 5, 20


def _pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def _summ(v):
    return {"n": len(v), "min": min(v), "p10": _pct(v, 0.1), "med": statistics.median(v),
            "p90": _pct(v, 0.9), "mean": statistics.mean(v)}


# ---------------------------------------------------------------- child
def child(a):
    t_start = time.time()
    cfg_name = os.environ["SWEEP_CFG_NAME"]
    import torch
    from datetime import timedelta
    store = torch.distributed.TCPStore(a.master, a.port, 2, is_master=(a.rank == 0),
                                       timeout=timedelta(seconds=a.rdzv_timeout),
                                       wait_for_workers=False)
    if os.environ.get("NCCL_SWEEP_FAKE"):   # logic smoke only: no CUDA, no NCCL
        return _fake_child(a, store, cfg_name, t_start)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
    from vllm.utils.nccl import find_nccl_library
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    if a.rank == 0:
        uid = NCCLLibrary().ncclGetUniqueId()
        store.set("uid", bytes(uid.internal))
    uid_bytes = store.get("uid")
    t_init0 = time.time()
    comm = PyNcclCommunicator.from_unique_id_bytes(uid_bytes, a.rank, 2, dev)
    torch.cuda.synchronize()
    init_s = time.time() - t_init0

    tiny = torch.zeros(1, device=dev)

    def barrier():
        comm.all_reduce(tiny)
        torch.cuda.synchronize()

    def ev():
        return torch.cuda.Event(enable_timing=True)

    ar_big = torch.randn(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    ar_big_out = torch.empty_like(ar_big)
    rs_out = torch.empty(TOKENS // 2, HIDDEN, device=dev, dtype=torch.bfloat16)
    ag_in = torch.randn(TOKENS // 2, HIDDEN, device=dev, dtype=torch.bfloat16)
    ag_out = torch.empty(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    dec1 = torch.randn(1, HIDDEN, device=dev, dtype=torch.bfloat16)
    dec8 = torch.randn(8, HIDDEN, device=dev, dtype=torch.bfloat16)
    ops = {
        "allreduce_16.8MB": (lambda: comm.all_reduce(ar_big, ar_big_out), ar_big.numel() * 2),
        "reducescatter_16.8MB_in": (lambda: comm.reduce_scatter(rs_out, ar_big), ar_big.numel() * 2),
        "allgather_8.4MB_in": (lambda: comm.all_gather(ag_out, ag_in), ag_in.numel() * 2),
        "allreduce_8KB_decode_m1": (lambda: comm.all_reduce(dec1), dec1.numel() * 2),
        "allreduce_64KB_decode_m8": (lambda: comm.all_reduce(dec8), dec8.numel() * 2),
    }
    res = {}
    for name, (fn, nbytes) in ops.items():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        iso = []
        for _ in range(ITERS):
            barrier()
            s, e = ev(), ev()
            s.record(); fn(); e.record(); e.synchronize()
            iso.append(s.elapsed_time(e) * 1e3)
        barrier()
        s, e = ev(), ev()
        s.record()
        for _ in range(STREAM_N):
            fn()
        e.record(); e.synchronize()
        res[name] = {"bytes": nbytes, "iso_us": iso, "iso": _summ(iso),
                     "stream_us_per_op": s.elapsed_time(e) * 1e3 / STREAM_N}

    # Overlap probe: GEMM on main, all-reduce on a side stream.
    A = torch.randn(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    W = torch.randn(HIDDEN, HIDDEN, device=dev, dtype=torch.bfloat16)
    C = torch.empty(TOKENS, HIDDEN, device=dev, dtype=torch.bfloat16)
    NG, NA = 10, 8
    for _ in range(3):
        torch.matmul(A, W, out=C)
    torch.cuda.synchronize()
    reps = []
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    for _ in range(3):
        barrier()
        s, e = ev(), ev(); s.record()
        for _ in range(NG):
            torch.matmul(A, W, out=C)
        e.record(); e.synchronize(); g_alone = s.elapsed_time(e) * 1e3 / NG
        barrier()
        s, e = ev(), ev(); s.record()
        for _ in range(NA):
            comm.all_reduce(ar_big, ar_big_out)
        e.record(); e.synchronize(); a_alone = s.elapsed_time(e) * 1e3 / NA
        barrier()
        w0, w1, g0, g1, a0, a1 = (ev() for _ in range(6))
        w0.record(main)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            a0.record(side)
            for _ in range(NA):
                comm.all_reduce(ar_big, ar_big_out, stream=side)
            a1.record(side)
        g0.record(main)
        for _ in range(NG):
            torch.matmul(A, W, out=C)
        g1.record(main)
        main.wait_stream(side)
        w1.record(main)
        torch.cuda.synchronize()
        g_c = g0.elapsed_time(g1) * 1e3 / NG
        a_c = a0.elapsed_time(a1) * 1e3 / NA
        wall = w0.elapsed_time(w1) * 1e3
        serial = g_alone * NG + a_alone * NA
        reps.append({"gemm_alone_us": g_alone, "ar_alone_us": a_alone,
                     "gemm_concurrent_us": g_c, "ar_concurrent_us": a_c,
                     "wall_concurrent_us": wall, "serial_sum_us": serial,
                     "hidden_fraction_of_ar": (serial - wall) / (a_alone * NA)})
    overlap = {"gemm": "bf16 [2048,4096]x[4096,4096]", "n_gemm": NG, "n_allreduce": NA,
               "reps": reps,
               "median": {k: statistics.median(r[k] for r in reps) for k in reps[0]}}
    barrier()
    out = {"schema": "prismaquant.nccl_sweep.child/1", "config": cfg_name,
           "config_env": json.loads(os.environ.get("SWEEP_CFG_ENV", "{}")),
           "rank": a.rank, "host": socket.gethostname(),
           "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
           "nccl_version": comm.nccl_version, "nccl_library": find_nccl_library(),
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "init_s": init_s, "t_start_unix": t_start, "t_end_unix": time.time(),
           "collectives": res, "overlap": overlap}
    comm.destroy()
    path = os.path.join(a.out, f"rank{a.rank}", f"{a.index:02d}-{cfg_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    ar = res["allreduce_16.8MB"]["iso"]["med"]
    print(f"[nccl-sweep] rank{a.rank} {cfg_name}: AR16.8MB iso med {ar:.0f} us, "
          f"init {init_s:.1f} s, wall {time.time() - t_start:.1f} s", flush=True)


def _fake_child(a, store, cfg_name, t_start):
    import random
    store.set(f"hello{a.rank}", "1")
    store.get(f"hello{1 - a.rank}")
    mk = lambda base: [base * (1 + 0.05 * random.random()) for _ in range(ITERS)]
    ops = {"allreduce_16.8MB": 754.0, "reducescatter_16.8MB_in": 380.0, "allgather_8.4MB_in": 380.0,
           "allreduce_8KB_decode_m1": 28.0, "allreduce_64KB_decode_m8": 35.0}
    res = {k: {"bytes": 16777216 if "16.8" in k else 8388608 if "8.4" in k else 8192 if "8KB" in k else 65536,
               "iso_us": mk(v), "stream_us_per_op": v} for k, v in ops.items()}
    for v in res.values():
        v["iso"] = _summ(v["iso_us"])
    ov = {"gemm_alone_us": 800.0, "ar_alone_us": 754.0, "gemm_concurrent_us": 900.0,
          "ar_concurrent_us": 800.0, "wall_concurrent_us": 9000.0, "serial_sum_us": 14032.0,
          "hidden_fraction_of_ar": 0.8}
    out = {"schema": "prismaquant.nccl_sweep.child/1", "config": cfg_name, "fake": True,
           "config_env": json.loads(os.environ.get("SWEEP_CFG_ENV", "{}")), "rank": a.rank,
           "host": socket.gethostname(), "nccl_env": {}, "nccl_version": 0, "nccl_library": "",
           "init_s": 0.0, "t_start_unix": t_start, "t_end_unix": time.time(), "collectives": res,
           "overlap": {"reps": [ov], "median": ov}}
    with open(os.path.join(a.out, f"rank{a.rank}", f"{a.index:02d}-{cfg_name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[nccl-sweep] rank{a.rank} {cfg_name}: fake", flush=True)


# ---------------------------------------------------------------- box
def box(a):
    t0 = time.time()
    os.makedirs(os.path.join(a.out, f"rank{a.rank}"), exist_ok=True)
    import torch
    from datetime import timedelta
    ctl = torch.distributed.TCPStore(a.master, a.port, 2, is_master=(a.rank == 0),
                                     timeout=timedelta(seconds=a.rdzv_timeout),
                                     wait_for_workers=False)
    try:
        gpu_procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader"], capture_output=True, text=True,
                                   timeout=20).stdout.strip()
    except Exception as e:  # recorded, not fatal
        gpu_procs = f"unavailable: {e}"
    names = [c[0] for c in CONFIGS] if not a.only else a.only.split(",")
    todo = [(i, n, dict(CONFIGS)[n]) for i, n in enumerate(c[0] for c in CONFIGS) if n in names]
    log = []
    for i, name, env_add in todo:
        # Lockstep: rank 0 decides run or stop from its deadline; rank 1 follows.
        if a.rank == 0:
            left = a.deadline_s - (time.time() - t0)
            ctl.set(f"plan{i}", "run" if left > a.child_timeout + 5 else "stop")
        plan = ctl.get(f"plan{i}").decode()
        ctl.add(f"arrive{i}", 1)
        t_wait = time.time()
        while int(ctl.add(f"arrive{i}", 0)) < 2:
            if time.time() - t_wait > a.rdzv_timeout:
                raise SystemExit(f"peer did not reach configuration {i} ({name})")
            time.sleep(0.05)
        if plan == "stop":
            log.append({"index": i, "config": name, "status": "skipped_deadline"})
            continue
        env = dict(os.environ)
        env.update(env_add)
        env["SWEEP_CFG_NAME"] = name
        env["SWEEP_CFG_ENV"] = json.dumps(env_add)
        if name == "auto-c64":   # show the protocol NCCL picks per size in the serve's config
            env.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,TUNING",
                        "NCCL_DEBUG_FILE": os.path.join(a.out, f"rank{a.rank}", "nccl-debug-auto-c64.%h.%p.log")})
        cmd = [sys.executable, os.path.abspath(__file__), "child", "--rank", str(a.rank),
               "--master", a.master, "--port", str(a.port + 1 + i), "--out", a.out,
               "--index", str(i), "--rdzv-timeout", str(a.rdzv_timeout)]
        t1 = time.time()
        try:
            p = subprocess.run(cmd, env=env, timeout=a.child_timeout, capture_output=True, text=True)
            status = "ok" if p.returncode == 0 else f"rc={p.returncode}"
            tail = (p.stdout + p.stderr)[-1500:]
        except subprocess.TimeoutExpired as e:
            status = "timeout"
            dec = lambda x: x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
            tail = (dec(e.stdout) + dec(e.stderr))[-1500:]
        log.append({"index": i, "config": name, "status": status, "wall_s": time.time() - t1,
                    "output_tail": tail})
        print(f"[nccl-sweep] rank{a.rank} {i:02d} {name}: {status} in {time.time() - t1:.1f} s", flush=True)
    summary = {"schema": "prismaquant.nccl_sweep.box/1", "rank": a.rank, "host": socket.gethostname(),
               "master": a.master, "port": a.port, "gpu_compute_apps_at_start": gpu_procs,
               "t_start_unix": t0, "t_end_unix": time.time(), "deadline_s": a.deadline_s,
               "child_timeout_s": a.child_timeout, "configs": log}
    with open(os.path.join(a.out, f"rank{a.rank}", "box.json"), "w") as f:
        json.dump(summary, f, indent=1)
    # Hold the control store until the peer is done with it.
    ctl.add("done", 1)
    t_wait = time.time()
    while int(ctl.add("done", 0)) < 2 and time.time() - t_wait < 30:
        time.sleep(0.1)
    bad = [c for c in log if c["status"] not in ("ok", "skipped_deadline")]
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------- merge
def merge(a):
    rows = {}
    for r in (0, 1):
        d = os.path.join(a.out, f"rank{r}")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json") and fn != "box.json":
                j = json.load(open(os.path.join(d, fn)))
                rows.setdefault(j["config"], {})[r] = j
    boxes = {}
    for r in (0, 1):
        p = os.path.join(a.out, f"rank{r}", "box.json")
        if os.path.exists(p):
            boxes[r] = json.load(open(p))
    order = [c[0] for c in CONFIGS]
    out = {"schema": "prismaquant.nccl_sweep/1",
           "method": "pairwise minimum of rank0/rank1 barrier-aligned samples = transfer estimate; "
                     "payload GB/s = bytes / time (2-rank ring: every rank sends that many bytes)",
           "boxes": boxes, "configs": []}
    for name in sorted(rows, key=lambda n: order.index(n) if n in order else 99):
        pair = rows[name]
        ent = {"config": name, "env": (pair.get(0) or pair.get(1))["config_env"],
               "ranks_present": sorted(pair), "collectives": {}, "overlap": {}}
        if len(pair) == 2:
            for op in pair[0]["collectives"]:
                x, y = pair[0]["collectives"][op], pair[1]["collectives"][op]
                mins = [min(u, v) for u, v in zip(x["iso_us"], y["iso_us"])]
                med = statistics.median(mins)
                ent["collectives"][op] = {
                    "bytes": x["bytes"], "transfer_us": _summ(mins),
                    "payload_GBps_at_median": x["bytes"] / med / 1e3,
                    "rank_iso_med_us": [x["iso"]["med"], y["iso"]["med"]],
                    "stream_us_per_op": [x["stream_us_per_op"], y["stream_us_per_op"]]}
            for r in (0, 1):
                ent["overlap"][f"rank{r}"] = pair[r]["overlap"]["median"]
            ent["nccl_version"] = pair[0]["nccl_version"]
            ent["init_s"] = [pair[0]["init_s"], pair[1]["init_s"]]
            ent["t_unix"] = [min(p["t_start_unix"] for p in pair.values()),
                             max(p["t_end_unix"] for p in pair.values())]
        out["configs"].append(ent)
    base = next((c for c in out["configs"] if c["config"] == "auto-c64" and c["collectives"]), None)
    if base:
        b = base["collectives"]["allreduce_16.8MB"]["transfer_us"]["med"]
        for c in out["configs"]:
            if c["collectives"]:
                t = c["collectives"]["allreduce_16.8MB"]["transfer_us"]["med"]
                # 91 all-reduces of 16.8 MB per 2048-token chunk in the A8SE-SH trace
                c["ms_per_chunk_vs_auto_c64_at_91_allreduces"] = (t - b) * 91 / 1e3
    path = os.path.join(a.out, "nccl-sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'config':18s} {'AR16.8MB us':>11s} {'GB/s':>6s} {'RS us':>7s} {'AG us':>7s} "
          f"{'AR8KB us':>8s} {'AR64KB us':>9s} {'gemm+%':>7s} {'ar+%':>6s} {'hidden':>6s} {'d ms/chunk':>10s}")
    for c in out["configs"]:
        if not c["collectives"]:
            print(f"{c['config']:18s} ranks {c['ranks_present']} (incomplete)")
            continue
        k = c["collectives"]
        ov = c["overlap"]["rank0"]
        print(f"{c['config']:18s} {k['allreduce_16.8MB']['transfer_us']['med']:11.0f} "
              f"{k['allreduce_16.8MB']['payload_GBps_at_median']:6.1f} "
              f"{k['reducescatter_16.8MB_in']['transfer_us']['med']:7.0f} "
              f"{k['allgather_8.4MB_in']['transfer_us']['med']:7.0f} "
              f"{k['allreduce_8KB_decode_m1']['transfer_us']['med']:8.1f} "
              f"{k['allreduce_64KB_decode_m8']['transfer_us']['med']:9.1f} "
              f"{100 * (ov['gemm_concurrent_us'] / ov['gemm_alone_us'] - 1):7.1f} "
              f"{100 * (ov['ar_concurrent_us'] / ov['ar_alone_us'] - 1):6.1f} "
              f"{ov['hidden_fraction_of_ar']:6.2f} "
              f"{c.get('ms_per_chunk_vs_auto_c64_at_91_allreduces', 0):10.1f}")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["box", "child", "merge"])
    ap.add_argument("--rank", type=int)
    ap.add_argument("--master", default="10.100.96.2")
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated configuration names")
    ap.add_argument("--deadline-s", type=float, default=250.0)
    ap.add_argument("--child-timeout", type=float, default=45.0)
    ap.add_argument("--rdzv-timeout", type=float, default=60.0)
    a = ap.parse_args()
    {"box": box, "child": child, "merge": merge}[a.mode](a)


if __name__ == "__main__":
    main()
