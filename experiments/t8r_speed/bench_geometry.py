"""The supported-rung geometry sweep for Tessera-8 (tessera#750 protocol).

Every q256 rung the sweep names runs through the fused window kernel at the
protocol's shapes and M, and each cell is timed twice in one job: once in a
forward pass over the cell list and once in a reverse pass, so a drift in
clock or temperature over the job moves both halves of a comparison equally.
A cell's time is the mean of its two passes' medians, and its spread is the
two passes' difference over that mean.

Shapes (per rank, TP2): routed gate/up (4096 to 2 x 1024, launch mode 0) and
down (1024 to 4096, mode 2); dense ``o_proj`` 4096 to 4096, ``q_b`` 1536 to
8192, and KDA ``in_proj_qkvbfg_a`` 4096 to 12448.  The dense identity needs
rows in multiples of 128, and 12448 is not one, so that shape is refused by
name and its first 12416 rows (97 blocks) run as ``kda_in_12416``.

M: 1, 64, 512 and 8192.  Routed cells use balanced routing (token t picks
experts (8t + j) mod 288) and, when ``--routing`` names the recorded prefill
routing, the median recorded step at M = 512 and 8192 (median by the number
of 64-route superblocks the step needs).

A cell is replayed from a captured CUDA graph (the counter reset and the
launch), so launch overhead does not enter small-M times.  The forward pass
also records the output's sha256, torch.profiler device time per kernel,
and, at ``--power-ms``, NVML board power over a back-to-back loop.  Every
cell carries its unix time, so the Netdata power series of the job can be
aligned to it.

Geometry columns per rung: one or two runs, the column rates, the bits a
lane's 8 rows hold, the 64-row half's bytes (16-byte copies on even rates,
an 8-byte tail on odd ones), the word slot, the launch's dynamic shared
memory, and registers and spills of the instantiation, read from the built
library with ``cuobjdump -res-usage``.

References: vLLM's FP8 MoE (W8A8, per-channel weight and per-token
activation scales) at the routed shapes, and ``torch._scaled_mm`` FP8
row-wise at the dense shapes, in both passes.

Usage: bench_geometry.py --out DIR --part routed|dense|all
       [--cases q256,...] [--ms 1,64,512,8192] [--routing DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_rates import (EXPERTS, HIDDEN, INTER, SWIGLU_LIMIT, TOP_K, Clock, build_projection,  # noqa: E402
                         emit, floor_ms, parse_case, q256_of, routing_tables, sha)
from bench_t8r import (ENVELOPE_W, PowerSampler, balanced_routing, kernel_profile,  # noqa: E402
                       recorded_routing, routing_files, routing_stats, summarize, time_events)

DEFAULT_RUNGS = sorted([256 * k for k in range(1, 9)] + [256 * k + f for k in range(1, 8) for f in (64, 128, 192)])
PROTOCOL_DENSE = {"o_proj": (4096, 4096), "q_b": (8192, 1536), "kda_in": (12448, 4096),
                  "kda_in_12416": (12416, 4096)}


def geometry(rf, r_lo, frac, mode, mma8):
    rates = [r_lo] if frac is None else [r_lo, r_lo + 1]
    pair = torch.tensor((r_lo, 0, 1, 0, rates[-1] if frac else 0, 1, 1 if frac else 0, 16 * r_lo),
                        dtype=torch.int32)
    slot = rf.slot_words_for_pair(pair)
    return {"runs": len(rates), "rates": rates,
            "lane_bits": [8 * r for r in rates],
            "lane_ends_on_word": [(8 * r) % 32 == 0 for r in rates],
            "half_bytes": [8 * r for r in rates],
            "half_copy": ["16B" if (8 * r) % 16 == 0 else "8B tail" for r in rates],
            "slot_words": slot, "smem_bytes": rf.smem_bytes(mode, slot, mma8=mma8)}


def resource_usage(lib):
    """``{demangled kernel: {REG, STACK, LOCAL, SHARED}}`` of the built library."""
    so = getattr(lib, "__file__", None)
    if not so:
        return {"error": "the extension has no __file__"}
    try:
        raw = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True, timeout=300).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:300]}
    out, name = {}, None
    for line in raw.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        if name and "REG:" in line:
            vals = dict(re.findall(r"(\w+(?:\[\d+\])?):(\d+)", line))
            out[name] = {k: int(vals[k]) for k in ("REG", "STACK", "SHARED", "LOCAL") if k in vals}
            name = None
    try:
        dem = subprocess.run(["c++filt"], input="\n".join(out), capture_output=True, text=True, timeout=60).stdout
        names = dem.splitlines()
        if len(names) == len(out):
            out = dict(zip(names, out.values()))
    except Exception:  # noqa: BLE001
        pass
    return {"so": so, "kernels": out}


def kernel_usage(usage, mode, dense, r_lo, two, bm):
    """The instantiation ``routed_fused_kernel<true, MODE, DENSE, SPLIT=false, RL, TWO, BMT>``'s row."""
    want = f"routed_fused_kernel<true, {mode}, {'true' if dense else 'false'}, false, {r_lo}, " \
           f"{'true' if two else 'false'}, {bm}>"
    for k, v in usage.get("kernels", {}).items():
        if want in k:
            return v
    return None


def time_call(call, warmup, iters, graph=True):
    """Graph replay (the decode path's launch), or CUDA events for a call that is not captured."""
    if not graph:
        return "events", time_events(call, warmup, iters)
    try:
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
        samples = time_events(graph.replay, warmup, iters)
        del graph
        return "graph", samples
    except Exception:  # noqa: BLE001
        torch.cuda.synchronize()
        return "events", time_events(call, warmup, iters)


def pick_recorded(root, ms):
    """The median recorded step per M, by 64-route superblocks."""
    files = routing_files(root, ms) if root else {}
    out = {}
    for m, paths in files.items():
        stats = []
        for p in paths:
            ids, _ = recorded_routing(p, m, "cpu")
            stats.append((routing_stats(ids)["superblocks"], p))
        stats.sort()
        sb, p = stats[len(stats) // 2]
        out[m] = {"path": p, "superblocks": sb, "n_files": len(paths),
                  "superblocks_range": [stats[0][0], stats[-1][0]]}
    return out


class Sweep:
    def __init__(self, args, rf, lib, library, mma8, dev, sms):
        self.args, self.rf, self.lib, self.library, self.mma8 = args, rf, lib, library, mma8
        self.dev, self.sms = dev, sms
        self.power, self.clock = PowerSampler(), Clock()
        self.cells = {}
        self.ms = [int(v) for v in args.ms.split(",")]
        self.power_ms = {int(v) for v in args.power_ms.split(",") if v}
        self.recorded = pick_recorded(args.routing, [512, 8192]) if args.routing else {}

    # -- groups: (key, build) where build() returns [(cell_key, meta, call, out)] lazily
    def groups(self):
        a = self.args
        parts = ("routed", "dense") if a.part == "all" else (a.part,)
        g = []
        rungs = [int(c[1:]) if c.startswith("q") else q256_of(*parse_case(c)) for c in a.cases.split(",")]
        if "routed" in parts:
            if a.refs:
                g.append(("vllm_fp8_moe", None))
            for q in rungs:
                for mode in (0, 2):
                    g.append(("routed", (q, mode)))
        if "dense" in parts:
            for shape in a.shapes.split(","):
                if a.refs:
                    g.append(("scaled_mm", shape))
                for q in rungs:
                    g.append(("dense", (q, shape)))
        return g

    def variants(self, kind):
        if kind in ("routed", "vllm_fp8_moe"):
            v = [(m, "balanced") for m in self.ms]
            v += [(m, "recorded") for m in sorted(self.recorded)]
            return v
        return [(m, None) for m in self.ms]

    def routing(self, m, how):
        if how == "recorded":
            ids, w = recorded_routing(self.recorded[m]["path"], m, self.dev)
        else:
            ids, w = balanced_routing(m, self.dev)
        return ids, w

    # -- builders
    def build_routed(self, q, mode):
        rf = self.rf
        r_lo, frac = parse_case(f"q{q}")
        rows, cols = (INTER, HIDDEN) if mode == 0 else (HIDDEN, INTER)
        n_hi = 0 if frac is None else round(cols * frac)
        seed = zlib.crc32(f"q{q}:{mode}".encode())
        projs = [build_projection(rf, EXPERTS, rows, cols, r_lo, n_hi, seed + i, self.dev, self.mma8)
                 for i in range(2 if mode == 0 else 1)]
        p0, p1 = projs[0], projs[-1]
        slot_words = max(p["slot_words"] for p in projs)
        head = {"kind": "routed", "q256": q, "mode": mode, "rows": rows, "cols": cols, "r_lo": r_lo,
                "n_hi": n_hi, "tile_words": p0["tile_words"],
                "geometry": geometry(rf, r_lo, frac, mode, self.mma8)}

        def make(m, how):
            bm = rf.superblock_rows(self.library, mode, m)
            if not rf.has_width(self.library, mode, bm):
                bm = rf.BM
            ids, w = self.routing(m, how)
            offsets, flat_sorted, rw_sorted, item_off = routing_tables(ids, w, EXPERTS, bm)
            routes = m * TOP_K
            g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(f"x:{q}:{mode}:{m}:{how}".encode()))
            xrows = m if mode == 0 else routes
            x = (torch.randn(xrows, cols, device=self.dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
            a_scale = torch.rand(xrows, device=self.dev, generator=g) * 0.1 + 0.01
            out = torch.empty((routes, rows), dtype=torch.bfloat16, device=self.dev)
            counter = torch.zeros(1, dtype=torch.int32, device=self.dev)

            def call():
                counter.zero_()
                self.lib.routed_fused_forward(
                    mode, True, x, a_scale, p0["words"], p1["words"], p0["table"], p1["table"],
                    p0["init"], p1["init"], p0["has_init"], p1["has_init"], p0["scale"], p1["scale"],
                    p0["runs"], p1["runs"], p0["bdesc"], p1["bdesc"], p0["tile_words"], slot_words,
                    offsets, flat_sorted, rw_sorted, item_off, counter, TOP_K,
                    0 if mode == 0 else 1, mode == 2, SWIGLU_LIMIT, out, self.sms, bm)
            touched = int(torch.unique(ids).numel())
            wire = touched * sum(p["bytes_per_expert"] for p in projs)
            out_cols = 2 * rows if mode == 0 else rows
            fl = floor_ms(wire + xrows * cols + routes * rows * 2, 2.0 * routes * out_cols * cols)
            st = routing_stats(ids)
            meta = {"bm": bm, "routes": routes, "touched": touched, "wire_bytes": wire, "floor": fl,
                    "superblocks": st["superblocks"] if bm == 64 else st["superblocks_128"]}
            return meta, call, out
        return head, make, projs

    def build_dense(self, q, shape):
        rf = self.rf
        rows, cols = PROTOCOL_DENSE[shape]
        r_lo, frac = parse_case(f"q{q}")
        n_hi = 0 if frac is None else round(cols * frac)
        head = {"kind": "dense", "q256": q, "shape": shape, "rows": rows, "cols": cols, "r_lo": r_lo,
                "n_hi": n_hi, "geometry": geometry(rf, r_lo, frac, 2, self.mma8)}
        if rows % rf.BN:
            head["refused"] = f"{rows} rows; the dense identity needs a multiple of {rf.BN}"
            return head, None, None
        p = build_projection(rf, 1, rows, cols, r_lo, n_hi, zlib.crc32(f"{shape}:q{q}".encode()),
                             self.dev, self.mma8)
        head["tile_words"] = p["tile_words"]

        def make(m, _how):
            g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(f"x:{shape}:{q}:{m}".encode()))
            x = (torch.randn(m, cols, device=self.dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
            a_scale = torch.rand(m, device=self.dev, generator=g) * 0.1 + 0.01
            out = torch.empty((m, rows), dtype=torch.bfloat16, device=self.dev)
            counter = torch.zeros(1, dtype=torch.int32, device=self.dev)
            s = rf.dense_k_split(m, rows, cols, self.sms, tile_words=p["tile_words"])
            bm = rf.superblock_rows(self.library, 2, m, dense=True) if s == 1 else rf.BM
            partial = (torch.empty((s, m, rows), dtype=torch.float32, device=self.dev) if s > 1
                       else torch.empty(0, dtype=torch.float32, device=self.dev))
            wscale = p["scale"].reshape(1, rows)

            def call():
                counter.zero_()
                self.lib.dense_forward(True, x, a_scale, p["words"], p["table"], p["init"], p["has_init"][:1],
                                       wscale, p["runs"], p["bdesc"], int(p["tile_words"]),
                                       int(p["slot_words"]), counter, int(s), partial, out, self.sms, int(bm))
            wire = p["bytes_per_expert"] * rows // (-(-rows // 512) * 512)
            act = m * cols + m * rows * 2 + (2 * s * m * rows * 4 if s > 1 else 0)
            meta = {"k_split": s, "bm": bm, "wire_bytes": wire,
                    "floor": floor_ms(wire + act, 2.0 * m * rows * cols)}
            return meta, call, out
        return head, make, p

    def build_vllm(self):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig, FusedMoEParallelConfig,
                                                                 RoutingMethodType)
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            convert_to_fp8_moe_kernel_format, make_fp8_moe_kernel, make_fp8_moe_quant_config,
            select_fp8_moe_backend)
        from vllm.model_executor.layers.quantization.utils.quant_utils import (kFp8DynamicTokenSym,
                                                                                kFp8StaticChannelSym)
        from vllm.v1.worker.workspace import init_workspace_manager
        dev = self.dev
        if not getattr(self, "_ws", False):
            init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
            self._ws = True
        E = EXPERTS
        parallel = FusedMoEParallelConfig(
            tp_size=1, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0,
            ep_rank=0, sp_size=1, use_ep=False, all2all_backend="allgather_reducescatter",
            enable_eplb=False)
        cfg = FusedMoEConfig(
            num_experts=E, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=INTER,
            num_local_experts=E, num_logical_experts=E, activation=MoEActivation.SILU,
            device=torch.device("cuda"), routing_method=RoutingMethodType.DeepSeekV3,
            moe_parallel_config=parallel, in_dtype=torch.bfloat16, swiglu_limit=SWIGLU_LIMIT)
        layer = torch.nn.Module()
        layer.moe_config = cfg
        layer.expert_map = None
        layer.apply_router_weight_on_input = False
        layer.activation = MoEActivation.SILU
        layer.global_num_experts = E
        layer.swiglu_limit = SWIGLU_LIMIT
        layer._expert_routing_tables = lambda: None

        def param(name, t):
            if hasattr(layer, name):
                delattr(layer, name)
            layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
        g = torch.Generator(device=dev).manual_seed(8008)
        w13 = (torch.randn(E, 2 * INTER, HIDDEN, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        w2 = (torch.randn(E, HIDDEN, INTER, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        s13 = torch.rand(E, 2 * INTER, 1, device=dev, generator=g) * 1e-2 + 1e-3
        s2 = torch.rand(E, HIDDEN, 1, device=dev, generator=g) * 1e-2 + 1e-3
        backend, experts_cls = select_fp8_moe_backend(
            config=cfg, weight_key=kFp8StaticChannelSym, activation_key=kFp8DynamicTokenSym,
            allow_vllm_cutlass=True)
        for name, t in (("w13_weight", w13), ("w2_weight", w2), ("w13_weight_scale", s13), ("w2_weight_scale", s2)):
            param(name, t)
        w13c, w2c, s13c, s2c = convert_to_fp8_moe_kernel_format(
            fp8_backend=backend, layer=layer, w13=w13, w2=w2, w13_scale=s13, w2_scale=s2,
            w13_input_scale=None, w2_input_scale=None)
        for name, t in (("w13_weight", w13c), ("w2_weight", w2c), ("w13_weight_scale", s13c),
                        ("w2_weight_scale", s2c)):
            param(name, t)
        quant = make_fp8_moe_quant_config(
            fp8_backend=backend, w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            a1_scale=None, a2_scale=None, per_act_token_quant=True, per_out_ch_quant=True,
            block_shape=None, gemm1_alpha=None, gemm1_beta=None, swiglu_limit=SWIGLU_LIMIT, layer=layer)
        kernel = make_fp8_moe_kernel(moe_quant_config=quant, moe_config=cfg, experts_cls=experts_cls,
                                     fp8_backend=backend, routing_tables=None)
        head = {"kind": "vllm_fp8_moe", "q256": 2048, "backend": str(getattr(backend, "name", backend)),
                "experts_cls": getattr(experts_cls, "__name__", str(experts_cls))}

        def make(m, how):
            ids, w = self.routing(m, how)
            x = (torch.randn(m, HIDDEN, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            holder = {}

            def call():
                holder["out"] = kernel.apply(x, layer.w13_weight, layer.w2_weight, w, ids,
                                             activation=MoEActivation.SILU, global_num_experts=E,
                                             expert_map=None, apply_router_weight_on_input=False)
            call()
            torch.cuda.synchronize()
            touched = int(torch.unique(ids).numel())
            wire = touched * 3 * INTER * HIDDEN
            routes = m * TOP_K
            meta = {"routes": routes, "touched": touched, "wire_bytes": wire,
                    "floor": floor_ms(wire + m * HIDDEN * 4, 2.0 * routes * 3 * INTER * HIDDEN)}
            return meta, call, holder
        return head, make, (layer, kernel)

    def build_scaled_mm(self, shape):
        rows, cols = PROTOCOL_DENSE[shape]
        dev = self.dev
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"ref:{shape}".encode()))
        w8 = (torch.randn(rows, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
        sw = torch.rand(1, rows, device=dev, generator=g) * 1e-2 + 1e-3
        head = {"kind": "scaled_mm", "q256": 2048, "shape": shape, "rows": rows, "cols": cols}

        def make(m, _how):
            x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
            sa = torch.rand(m, 1, device=dev, generator=g) * 0.1 + 0.01
            holder = {}

            def call():
                holder["out"] = torch._scaled_mm(x, w8.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)
            meta = {"wire_bytes": rows * cols,
                    "floor": floor_ms(rows * cols + m * cols + m * rows * 2, 2.0 * m * rows * cols)}
            return meta, call, holder
        return head, make, w8

    def build(self, kind, spec):
        if kind == "routed":
            return self.build_routed(*spec)
        if kind == "dense":
            return self.build_dense(*spec)
        if kind == "vllm_fp8_moe":
            return self.build_vllm()
        return self.build_scaled_mm(spec)

    # -- the passes
    def run(self, usage, save):
        a = self.args
        groups = self.groups()
        for pas in ("F", "R"):
            order = groups if pas == "F" else list(reversed(groups))
            for kind, spec in order:
                gkey = f"{kind}:{spec}"
                try:
                    head, make, keep = self.build(kind, spec)
                except Exception as exc:  # noqa: BLE001
                    self.cells.setdefault(gkey, {"kind": kind, "spec": str(spec), "cells": {}})["error"] = repr(exc)[:500]
                    emit({"group": gkey, "pass": pas, "error": repr(exc)[:300]})
                    continue
                rec = self.cells.setdefault(gkey, dict(head, cells={}))
                if kind in ("routed", "dense") and "refused" not in head:
                    rec.setdefault("usage", {})
                if make is None:
                    if pas == "F":
                        emit({"group": gkey, "refused": head.get("refused")})
                    continue
                vs = self.variants(kind)
                for m, how in (vs if pas == "F" else list(reversed(vs))):
                    ckey = f"{m}" + (f":{how}" if how else "")
                    cell = rec["cells"].setdefault(ckey, {})
                    try:
                        meta, call, out = make(m, how)
                        how_t, samples = time_call(call, a.warmup, a.iters, graph=kind != "vllm_fp8_moe")
                        cell[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                     "timer": how_t, "unix": time.time(), "clock": self.clock.read()}
                        if pas == "F":
                            cell.update({k: v for k, v in meta.items()})
                            if isinstance(out, torch.Tensor):
                                cell["out_sha256"] = sha(out)
                            cell["profile"] = kernel_profile(call, reps=a.prof_reps)
                            if m in self.power_ms:
                                cell["power"] = self.power.sample_during(call, a.power_s)
                            if kind in ("routed", "dense"):
                                u = kernel_usage(usage, head.get("mode", 2), kind == "dense", head["r_lo"],
                                                 head["n_hi"] > 0, meta.get("bm", 64))
                                rec["usage"][str(meta.get("bm", 64))] = u
                        else:
                            f, r = cell.get("F", {}).get("median_ms"), cell["R"]["median_ms"]
                            if f:
                                cell["ms"] = 0.5 * (f + r)
                                cell["spread"] = abs(f - r) / cell["ms"]
                                fl = cell.get("floor", {}).get("floor_ms")
                                if fl:
                                    cell["roofline_frac"] = fl / cell["ms"]
                        del call, out
                    except Exception as exc:  # noqa: BLE001
                        cell[pas] = {"error": repr(exc)[:500]}
                        emit({"group": gkey, "M": ckey, "pass": pas, "error": repr(exc)[:300]})
                        continue
                    line = {"g": gkey, "M": ckey, "pass": pas, "ms": round(cell[pas]["median_ms"], 4)}
                    if pas == "R" and "ms" in cell:
                        line.update({"mean_ms": round(cell["ms"], 4), "spread": round(cell["spread"], 4),
                                     "roof": round(cell.get("roofline_frac") or 0, 3)})
                    emit(line)
                del keep
                torch.cuda.empty_cache()
                save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("routed", "dense", "all"), required=True)
    ap.add_argument("--cases", default=",".join(f"q{q}" for q in DEFAULT_RUNGS))
    ap.add_argument("--ms", default="1,64,512,8192")
    ap.add_argument("--shapes", default="o_proj,q_b,kda_in,kda_in_12416")
    ap.add_argument("--routing", default="", help="recorded routing root (m512/, m8192/)")
    ap.add_argument("--refs", action="store_true", help="vLLM FP8 MoE and torch._scaled_mm references")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prof-reps", type=int, default=3)
    ap.add_argument("--power-ms", default="512,8192")
    ap.add_argument("--power-s", type=float, default=0.5)
    ap.add_argument("--library", default=None)
    ap.add_argument("--ncu", action="store_true", help="accepted for the wrapper; not used")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from tessera import routed_fused as rf
    library = args.library or rf.library_for("e4m3")
    mma8 = rf.library_mma8(library)
    lib = rf._ext(library)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    usage = resource_usage(lib)
    sw = Sweep(args, rf, lib, library, mma8, dev, sms)
    meta = {"device": torch.cuda.get_device_name(), "sms": sms, "experts": EXPERTS, "hidden": HIDDEN,
            "inter": INTER, "top_k": TOP_K, "part": args.part, "cases": args.cases, "ms": args.ms,
            "shapes": args.shapes, "recorded": sw.recorded, "library": library,
            "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": sw.power.source,
            "envelope_w": ENVELOPE_W, "start_unix": time.time(), "torch": torch.__version__,
            "statistic": "mean of the forward and reverse passes' medians (graph replay); spread = |F - R| / mean",
            "resource_usage": usage}
    path = os.path.join(args.out, f"bench_geometry_{args.part}.json")

    def save():
        json.dump({"meta": meta, "groups": sw.cells}, open(path, "w"), indent=1)
    sw.run(usage, save)
    meta["end_unix"] = time.time()
    save()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
