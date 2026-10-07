"""Stage 1 check and timing for the register-direct routed kernel (eng-regdirect-build).

Cells: projection (gate/up 2 x 1024 x 4096, down 4096 x 1024 per expert; GLM-5.3 Flash TP2 rank),
rate profile (R768, R1024, and k-step-mixed R896 = half the slots at R4, then R3), M in --ms, balanced
routing (token t picks experts (8t + j) mod 288).  Today's fused routed kernel runs at the same cells
on its own synthetic wire (the D41 path) as the baseline.

Order, one process: (1) check: decoded weights (DUMP build) bitwise equal to the reference decode of
the same planes, for every touched expert at M=1 and a fixed subset at larger M; output inside an fp64
bound derived from fp32 accumulation and bf16 rounding; two runs bitwise equal.  Any failure stops the
job before timing (exit 3).  (2) timing: CUDA graph per cell; COLD-L2 timer (a 256 MB read before each
sample, untimed) as the decode measure (CEO rule 5), and the D41 warm timer beside it; a forward and a
reverse pass, the mean of the two medians.  (3) torch.profiler per cell.

--cpu-preflight (D38): imports, arguments, the generator and the reference decode on a small stack.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import zlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fragment_synth import GROUPS, TABLES, make_stack, reference_decode  # noqa: E402

TOP_K, EXPERTS, HIDDEN, INTER, SWIGLU_LIMIT = 8, 288, 4096, 1024, 10.0
GB10_BW = 236.4e9
SHAPES = {0: (INTER, HIDDEN), 2: (HIDDEN, INTER)}     # (rows N, input K) per expert
PROFILES = {"R1024": (4, 4, None), "R768": (3, 3, None), "R896": (4, 3, "half")}


def profiles_for(name, ks):
    ra, rb, ksa = PROFILES[name]
    return [(ra, rb, ks if ksa is None else ks // 2)] * EXPERTS


def balanced_routing(m, dev):
    t = torch.arange(m, device=dev).unsqueeze(1) * TOP_K + torch.arange(TOP_K, device=dev)
    return (t % EXPERTS).to(torch.int32)


def routing_tables(ids, bm):
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(EXPERTS, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = torch.zeros(EXPERTS + 1, dtype=torch.int32, device=ids.device)
    offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    order = torch.argsort(flat, stable=True).to(torch.int32).contiguous()
    item_off = torch.zeros(EXPERTS + 1, dtype=torch.int32, device=ids.device)
    item_off[1:] = torch.cumsum((counts + bm - 1) // bm, 0, dtype=torch.int32)
    return offsets, order, item_off


class MemGuard:
    """D30: abort this process (SIGTERM, then SIGKILL) if host MemAvailable < 2 GiB."""

    def __init__(self):
        self.min_avail = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import signal
        while True:
            time.sleep(0.25)
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    a = int(line.split()[1]) * 1024
                    self.min_avail = a if self.min_avail is None else min(self.min_avail, a)
                    if a < 2 * 2**30:
                        print("D30 GUARD: MemAvailable < 2 GiB; aborting", flush=True)
                        os.kill(os.getpid(), signal.SIGTERM)
                        time.sleep(5)
                        os.kill(os.getpid(), signal.SIGKILL)


# ------------------------------------------------------------------ arms
class RegDirect:
    def __init__(self, mode, profile, dev, sms):
        from tessera import regdirect_routed as rr
        self.rr, self.mode, self.profile, self.dev, self.sms = rr, mode, profile, dev, sms
        rows, k = SHAPES[mode]
        ks = k // (GROUPS[mode] * 32)
        st = make_stack(mode, EXPERTS, rows, ks, profiles_for(profile, ks), zlib.crc32(f"rd:{mode}:{profile}".encode()), dev)
        self.st = st
        self.stack = rr.FragmentStack(**st)
        self.bytes_per_expert = (st["wire"].numel() + st["hist"].numel()) * 4 // EXPERTS

    def make(self, m, x, a_scale, ids, rw_sorted=None, dump=None):
        rr = self.rr
        rows, k = SHAPES[self.mode]
        geom = rr.geometry(self.mode, m, TOP_K, rows, EXPERTS, self.stack.ks, self.sms)
        offsets, order, item_off = routing_tables(ids, geom.superblock)
        routes = m * TOP_K
        part, arrive = rr.scratch(geom, EXPERTS, routes, self.dev)
        out = torch.zeros(routes, rows, dtype=torch.bfloat16, device=self.dev)
        xz = rr.with_zero_row(x)
        a_row_mode = 0 if self.mode == 0 else 1
        if dump is not None:
            geom = rr.Geometry(geom.prefill, geom.superblock, geom.k_parts, geom.tiles,
                               self.sms * rr._ext().blocks_per_sm(self.mode, geom.prefill, True))

        def call():
            self.stack.launch(geom, xz, a_scale, offsets, order, rw_sorted, item_off, 0, EXPERTS, out, part, arrive,
                              top_k=TOP_K, a_row_mode=a_row_mode, mul_weight=self.mode == 2, limit=SWIGLU_LIMIT,
                              dump=dump)
        return call, out, {"prefill": geom.prefill, "superblock": geom.superblock, "k_parts": geom.k_parts,
                           "grid": geom.grid, "items": int(item_off[-1])}, (offsets, order, item_off)


class Baseline:
    """Today's fused routed kernel (the D41 path) on its own synthetic wire."""

    def __init__(self, mode, profile, dev, sms):
        sys.path.insert(0, os.path.join(HERE, "..", "t8r_speed"))
        from bench_rates import build_projection
        from tessera import routed_fused as rf
        self.rf, self.mode, self.dev, self.sms = rf, mode, dev, sms
        self.library = rf.library_for("e4m3")
        self.lib = rf._ext(self.library)
        rows, cols = SHAPES[mode]
        r_lo, n_hi = {"R1024": (4, 0), "R768": (3, 0), "R896": (3, cols // 2)}[profile]
        mma8 = rf.library_mma8(self.library)
        seed = zlib.crc32(f"paired:{mode}".encode())
        self.projs = [build_projection(rf, EXPERTS, rows, cols, r_lo, n_hi, seed + i, dev, mma8)
                      for i in range(2 if mode == 0 else 1)]
        self.slot_words = max(p["slot_words"] for p in self.projs)
        self.bytes_per_expert = sum(p["bytes_per_expert"] for p in self.projs)

    def make(self, m, x, a_scale, ids, rw_sorted=None):
        rf, p0, p1 = self.rf, self.projs[0], self.projs[-1]
        bm = rf.superblock_rows(self.library, self.mode, m)
        if not rf.has_width(self.library, self.mode, bm):
            bm = rf.BM
        offsets, order, item_off = routing_tables(ids, bm)
        w = rw_sorted if rw_sorted is not None else torch.full((m * TOP_K,), 1.0 / TOP_K, device=self.dev)
        rows = SHAPES[self.mode][0]
        out = torch.empty((m * TOP_K, rows), dtype=torch.bfloat16, device=self.dev)
        counter = torch.zeros(1, dtype=torch.int32, device=self.dev)

        def call():
            counter.zero_()
            self.lib.routed_fused_forward(
                self.mode, True, x, a_scale, p0["words"], p1["words"], p0["table"], p1["table"],
                p0["init"], p1["init"], p0["has_init"], p1["has_init"], p0["scale"], p1["scale"],
                p0["runs"], p1["runs"], p0["bdesc"], p1["bdesc"], p0["tile_words"], self.slot_words, False,
                offsets, order, w, item_off, counter, TOP_K, 0 if self.mode == 0 else 1, self.mode == 2,
                SWIGLU_LIMIT, out, self.sms, bm)
        return call, out, {"bm": bm}


# ----------------------------------------------------------------- check
def bf16_ulp(v):
    return torch.ldexp(torch.ones_like(v), torch.frexp(v.abs().clamp_min(2.0 ** -126))[1] - 8)


def output_check(arm, m, x, a_scale, tables, out, rw_sorted, experts):
    """The output against an fp64 reference of the reference decode, inside a bound derived from fp32
    accumulation (Higham gamma_K) and bf16 rounding (see the kill-test report, section 3)."""
    mode = arm.mode
    offsets, order, _ = tables
    off = offsets.cpu().tolist()
    k = SHAPES[mode][1]
    u32 = 2.0 ** -24
    gamma = k * u32 / (1 - k * u32)
    st = {"elements": 0, "exact": 0, "violations": 0, "max_ratio": 0.0}
    for e in experts:
        if off[e + 1] == off[e]:
            continue
        wq = reference_decode(arm.st, e).view(torch.float8_e4m3fn).to(torch.float64)
        pos = torch.arange(off[e], off[e + 1], device=x.device)
        flat = order.long()[pos]
        arow = flat // TOP_K if mode == 0 else pos
        xs = x[arow].to(torch.float64)
        sc = a_scale[arow, None].to(torch.float64)
        pre, acc_b = [], []
        for p in range(TABLES[mode]):
            ws = arm.st["wscale"][e, p][None].to(torch.float64)
            v = (xs @ wq[p].T) * sc * ws
            pre.append(v)
            acc_b.append(gamma * ((xs.abs() @ wq[p].abs().T) * sc * ws) + 3 * u32 * v.abs())
        if mode == 0:
            gf = pre[0].float().to(torch.bfloat16).float().clamp(max=SWIGLU_LIMIT)
            uf = pre[1].float().to(torch.bfloat16).float().clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            ref = (gf / (1.0 + torch.exp(-gf)) * uf).to(torch.bfloat16)
            dg = acc_b[0] + bf16_ulp(pre[0].abs() + acc_b[0])
            du = acc_b[1] + bf16_ulp(pre[1].abs() + acc_b[1])
            sg = (gf.double() / (1.0 + torch.exp(-gf.double()))).abs()
            got = out[pos].double()
            bound = 1.1 * uf.double().abs() * dg + (sg + 1.1 * dg) * du
        else:
            w = rw_sorted[pos, None].to(torch.float64)
            y = pre[0] * w
            ref = y.to(torch.bfloat16)
            got = out[flat].double()
            bound = (acc_b[0] + u32 * pre[0].abs()) * w
        b = ref.double()
        bound = bound + bf16_ulp(torch.maximum(got.abs(), b.abs())) + 1e-30
        ratio = (got - b).abs() / bound
        st["elements"] += got.numel()
        st["exact"] += int((got == b).sum())
        st["violations"] += int((ratio > 1).sum())
        st["max_ratio"] = max(st["max_ratio"], float(ratio.max()))
    st["exact_fraction"] = st["exact"] / max(1, st["elements"])
    st["pass"] = st["violations"] == 0
    return st


def check_cell(arm, m, x, a_scale, ids, rw_sorted, log):
    mode = arm.mode
    rows, k = SHAPES[mode]
    res = {"mode": mode, "profile": arm.profile, "M": m}
    dump = torch.zeros(EXPERTS, TABLES[mode], rows, k, dtype=torch.uint8, device=x.device)
    call_d, _, meta, tables = arm.make(m, x, a_scale, ids, rw_sorted, dump=dump)
    call_d(); torch.cuda.synchronize()
    off = tables[0].cpu().tolist()
    touched = [e for e in range(EXPERTS) if off[e + 1] > off[e]]
    subset = touched if len(touched) <= 16 else touched[::24]
    bad = []
    for e in subset:
        ref = reference_decode(arm.st, e)
        if not torch.equal(ref, dump[e]):
            bad.append({"expert": e, "mismatches": int((ref != dump[e]).sum()),
                        "first": (ref != dump[e]).nonzero()[:4].tolist()})
            break
    del dump
    res["decode_bitwise"] = {"experts_checked": len(subset), "pass": not bad, "bad": bad}
    call, out, meta, tables = arm.make(m, x, a_scale, ids, rw_sorted)
    call(); torch.cuda.synchronize()
    first = out.clone()
    call(); torch.cuda.synchronize()
    res["run_to_run_bitwise"] = bool(torch.equal(first, out))
    res["output"] = output_check(arm, m, x, a_scale, tables, first, rw_sorted, subset)
    res["meta"] = meta
    res["pass"] = res["decode_bitwise"]["pass"] and res["run_to_run_bitwise"] and res["output"]["pass"]
    log(f"check mode{mode} {arm.profile} M{m}: decode {res['decode_bitwise']['pass']} ({len(subset)} experts) "
        f"run-to-run {res['run_to_run_bitwise']} output {res['output']}")
    return res


# ----------------------------------------------------------------- timing
def graph_of(call):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def time_graph(graph, warmup, iters, flush=None):
    out = []
    for i in range(warmup + iters):
        if flush is not None:
            flush.sum()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        if i >= warmup:
            out.append(float(a.elapsed_time(b)))
    return out


def kernel_profile(call, reps=5):
    from torch.profiler import profile, ProfilerActivity
    call(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            call()
        torch.cuda.synchronize()
    per = {}
    for evt in prof.key_averages():
        us = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0.0)
        if us:
            per[evt.key[:140]] = round(us / reps, 2)
    return dict(sorted(per.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------- main
def cpu_preflight(args):
    dev = torch.device("cpu")
    rep = {"cpu_preflight": True, "torch": torch.__version__}
    for mode in (0, 2):
        rows, ks = 256, 4
        for prof in ((4, 4, ks), (3, 3, ks), (4, 3, 2)):
            st = make_stack(mode, 2, rows, ks, [prof, prof], 3, dev)
            ref = reference_decode(st, 1)
            assert ref.shape == (TABLES[mode], rows, ks * GROUPS[mode] * 32)
            rep[f"mode{mode}_{prof}"] = list(ref.shape)
    from tessera import regdirect_routed as rr
    rep["k_parts_gate_up_M1"] = rr.k_parts(8, 8, 96, 128)
    rep["k_parts_down_M1"] = rr.k_parts(8, 32, 96, 16)
    rep["k_parts_gate_up_M16"] = rr.k_parts(128, 8, 96, 128)
    rep["k_parts_down_M16"] = rr.k_parts(128, 32, 96, 16)
    rep["prefill_M16"] = rr.is_prefill(16, TOP_K, EXPERTS)
    rep["prefill_M2048"] = rr.is_prefill(2048, TOP_K, EXPERTS)
    json.dump(rep, open(os.path.join(args.out, "cpu-preflight.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("CPU preflight passed; no GPU results", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="0,2")
    ap.add_argument("--profiles", default="R1024,R768,R896")
    ap.add_argument("--ms", default="1,16,2048,4096")
    ap.add_argument("--check-ms", default="1,2048")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cpu-preflight", action="store_true")
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cpu_preflight:
        return cpu_preflight(args)
    guard = MemGuard()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    modes = [int(v) for v in args.modes.split(",")]
    profiles = args.profiles.split(",")
    ms = [int(v) for v in args.ms.split(",")]
    result = {"meta": {"device": torch.cuda.get_device_name(), "sms": sms, "torch": torch.__version__,
                       "host": os.environ.get("HOST_NAME"), "pb_action": os.environ.get("PB_ACTION_KEY"),
                       "head": os.environ.get("TESSERA_HEAD"), "start_unix": time.time(),
                       "statistic": "mean of forward and reverse pass medians; cold = L2 flushed before each sample"}}
    path = os.path.join(args.out, "ncu-run.json" if args.ncu else "stage1.json")

    def save():
        result["meta"]["min_mem_available_gib"] = guard.min_avail and round(guard.min_avail / 2**30, 2)
        json.dump(result, open(path, "w"), indent=1)

    def log(msg):
        print(msg, flush=True)

    inputs = {}
    for m in sorted(set(ms) | {int(v) for v in args.check_ms.split(",")}):
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"x:{m}".encode()))
        ids = balanced_routing(m, dev)
        routes = m * TOP_K
        inputs[m] = {
            0: ((torch.randn(m, HIDDEN, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn),
                torch.rand(m, device=dev, generator=g) * 0.1 + 0.01, ids, None),
            2: ((torch.randn(routes, INTER, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn),
                torch.rand(routes, device=dev, generator=g) * 0.1 + 0.01, ids,
                torch.rand(routes, device=dev, generator=g) * 0.2 + 0.05),
        }
    arms = {(md, pf): RegDirect(md, pf, dev, sms) for md in modes for pf in profiles}
    bases = {(md, pf): Baseline(md, pf, dev, sms) for md in modes for pf in profiles}

    if not args.skip_check:
        result["check"] = []
        for (md, pf), arm in arms.items():
            for m in [int(v) for v in args.check_ms.split(",")]:
                x, a_s, ids, rw = inputs[m][md]
                result["check"].append(check_cell(arm, m, x, a_s, ids, rw, log))
                save()
        if not all(c["pass"] for c in result["check"]):
            log("CORRECTNESS FAILED: stopping before any timing")
            result["stopped"] = "correctness"
            save()
            return 3

    cells = [(arm, md, pf, m) for md in modes for pf in profiles for m in ms for arm in ("baseline", "regdirect")]
    built = {}
    for arm, md, pf, m in cells:
        x, a_s, ids, rw = inputs[m][md]
        k = bases[(md, pf)] if arm == "baseline" else arms[(md, pf)]
        if arm == "baseline":
            call, out, meta = k.make(m, x, a_s, ids, rw)
        else:
            call, out, meta, _ = k.make(m, x, a_s, ids, rw)
        touched = int(torch.unique(ids).numel())
        built[(arm, md, pf, m)] = (call, out, dict(meta, wire_bytes=touched * k.bytes_per_expert, touched=touched))

    if args.ncu:
        cudart = torch.cuda.cudart()
        for key in cells:
            call = built[key][0]
            call(); call(); torch.cuda.synchronize()
            cudart.cudaProfilerStart(); call(); torch.cuda.synchronize(); cudart.cudaProfilerStop()
            log(f"ncu {key}")
        return 0

    graphs = {k: graph_of(v[0]) for k, v in built.items()}
    flush = torch.ones(64 * 2**20, dtype=torch.int32, device=dev)
    samples = {k: {} for k in cells}
    for pas in ("F", "R"):
        for k in (cells if pas == "F" else list(reversed(cells))):
            samples[k][pas + "cold"] = time_graph(graphs[k], args.warmup, args.iters, flush)
            samples[k][pas + "warm"] = time_graph(graphs[k], args.warmup, args.iters)
            log(f"{pas} {k}: cold {statistics.median(samples[k][pas + 'cold']) * 1e3:.1f} us, "
                f"warm {statistics.median(samples[k][pas + 'warm']) * 1e3:.1f} us")
    out_cells = {}
    for k in cells:
        arm, md, pf, m = k
        meta = built[k][2]
        cold = (statistics.median(samples[k]["Fcold"]) + statistics.median(samples[k]["Rcold"])) / 2
        warm = (statistics.median(samples[k]["Fwarm"]) + statistics.median(samples[k]["Rwarm"])) / 2
        out_cells[f"{arm}.mode{md}.{pf}.M{m}"] = {
            "cold_us": cold * 1e3, "warm_us": warm * 1e3, "cold_GBps": meta["wire_bytes"] / (cold * 1e-3) / 1e9,
            "cold_pct_236": meta["wire_bytes"] / (cold * 1e-3) / GB10_BW * 100, "samples": samples[k], "meta": meta,
            "torch_profiler": kernel_profile(built[k][0])}
    result["cells"] = out_cells
    result["meta"]["end_unix"] = time.time()
    save()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
