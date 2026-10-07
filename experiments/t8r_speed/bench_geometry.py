"""The supported-rung geometry sweep for Tessera-8 and Tessera-16 (tessera#750 protocol).

``--library`` picks the family: the E4M3 family's libraries (the default,
``routed_fused.library_for("e4m3")``), or ``value``, Tessera-16's folded
BF16 lane (bf16 activations, bf16 table, the bf16 ``mma.sync`` peak in the
floor; references vLLM's unquantized Triton MoE and bf16 ``F.linear`` --
the BF16 source passthrough).

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
import hashlib
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
from bench_rates import (EXPERTS, HIDDEN, INTER, MMA_BF16_TFLOPS, MMA_E4M3_TFLOPS, SWIGLU_LIMIT,  # noqa: E402
                         TOP_K, Clock, build_projection, emit, floor_ms, parse_case, q256_of,
                         routing_tables, sha)
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
            "slot_words": slot, "smem_bytes": rf.smem_bytes(mode, slot, mma8=mma8),
            "word_stages": (rf.word_stages(mode, slot, mma8=mma8) if hasattr(rf, "word_stages")
                            else getattr(rf, "WORD_STAGES", None))}


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


def kernel_usage(usage, mode, dense, r_lo, two, bm, fp8=True, *, split=False):
    """Actual original-order, unpaired instantiation, including its K-split specialization."""
    want = f"routed_fused_kernel<{'true' if fp8 else 'false'}, {mode}, {'true' if dense else 'false'}, {'true' if split else 'false'}, {r_lo}, " \
           f"{'true' if two else 'false'}, {bm}, false, false>"
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


def stage_geometry_inputs(args):
    """Copy admitted whole-file inputs through the existing pinned reader."""
    if not args.data_manifest:
        if args.routing and not args.cpu_preflight:
            raise ValueError('GPU recorded routing needs its admitted data manifest')
        return
    from pathlib import Path
    import tempfile
    from pb_staged_store import StagedInputs
    inputs=StagedInputs(args.data_manifest)
    args._staged_input_owner=tempfile.TemporaryDirectory(prefix='d41-geometry-inputs-',dir='/tmp')
    local_root=Path(args._staged_input_owner.name)
    try:
        for path,offset in inputs.entries:
            if offset!=0:raise ValueError('geometry routing inputs must be whole files')
            local=local_root/path.lstrip('/')
            local.parent.mkdir(parents=True,exist_ok=True)
            local.write_bytes(inputs.read(path))
        with open(os.path.join(args.out,'staged-reads.json'),'w') as stream:
            json.dump(inputs.reads,stream,indent=1)
    finally:
        inputs.close()
    for field in ('config','routing'):
        original=getattr(args,field)
        if original:setattr(args,field,str(local_root/original.lstrip('/')))



def compact_projection(p, rows, cols, *, grouped):
    """Use the public compact constructors, including actual wide BF16 tables.

    This is geometry below intake, not an override of compact_prep's served
    rate bounds. Its independent owner refusal is recorded with each group.
    """
    e = p["words"].shape[0]
    pair = p["runs"].reshape(e, 2, 4)
    nr = 2 if int(pair[0, 1, 2]) else 1
    runs = pair[:, :nr].contiguous()
    table = p["table"].view(torch.bfloat16)
    empty = torch.empty(0, dtype=torch.uint8, device=p["words"].device)
    if not grouped:
        from tessera.window_gemm import PreparedWindowGemm
        return PreparedWindowGemm(words=p["words"][0], table=table[0], codes=empty,
            native=empty, scale=p["scale"][0], runs=runs[0], init_perm=p["init"][0],
            perm=p["perm"][0], tile_words=p["tile_words"], total_words=p["words"].shape[1],
            rows=rows, cols=cols, window_bits=p["window_bits"], family="value", has_init=True,
            block_m=64, block_n=64, block_k=64, arithmetic="folded")
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa
    dev = p["words"].device
    width = p["words"].shape[1]
    return prepare_grouped_window_gemm_from_soa(words_all=p["words"], table_all=table,
        codes_all=empty, native_all=empty, scale_all=p["scale"], runs_all=runs,
        init_all=p["init"], has_init=p["has_init"],
        word_off=torch.arange(e, device=dev, dtype=torch.int32) * width,
        tile_words=torch.full((e,), p["tile_words"], device=dev, dtype=torch.int32),
        total_words=torch.full((e,), width, device=dev, dtype=torch.int32),
        run_off=torch.arange(e + 1, device=dev, dtype=torch.int32) * nr,
        perm_all=p["perm"], rows=rows, cols=cols, experts=e, window_bits=p["window_bits"],
        family="value", arithmetic="folded")


class CompilerSpy:
    """Retain exactly the CompiledKernel returned by a real Triton launch."""
    def __init__(self, jit):
        self.jit, self.resources = jit, {}

    def __getitem__(self, grid):
        launch = self.jit[grid]
        def run(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            if compiled is not None:
                self.resources[compiled.name] = {
                    "REG": compiled.n_regs, "spills": compiled.n_spills,
                    "SHARED": compiled.metadata.shared,
                    "num_warps": compiled.metadata.num_warps,
                    "compiler": "Triton actual CompiledKernel", "launch": kwargs}
            return compiled
        return run


def capture_compiler(call, grouped):
    from tessera import window_gemm, window_gemm_grouped
    module = window_gemm_grouped if grouped else window_gemm
    symbol = "_grouped_window_gemm_kernel" if grouped else "_window_gemm_kernel"
    spy = CompilerSpy(getattr(module, symbol))
    def wrapped():
        original = getattr(module, symbol)
        setattr(module, symbol, spy)
        try:
            return call()
        finally:
            setattr(module, symbol, original)
    wrapped.compiler_resources = spy.resources
    return wrapped



class Sweep:
    def __init__(self, args, rf, lib, library, mma8, dev, sms):
        self.args, self.rf, self.lib, self.library, self.mma8 = args, rf, lib, library, mma8
        self.fp8 = rf.LIBRARIES[library][1] == "e4m3"
        self.family = rf.LIBRARIES[library][1]
        self.tflops = MMA_E4M3_TFLOPS if self.fp8 else MMA_BF16_TFLOPS
        self.abytes = 1 if self.fp8 else 2          # activation bytes per element (e4m3 or bf16)
        self.dev, self.sms = dev, sms
        self.power, self.clock = PowerSampler(), Clock()
        self.cells = {}
        self.ms = [int(v) for v in args.ms.split(",")]
        self.power_ms = {int(v) for v in args.power_ms.split(",") if v}
        self.recorded = pick_recorded(args.routing, self.ms) if args.routing else {}

    # -- groups: (key, build) where build() returns [(cell_key, meta, call, out)] lazily
    def groups(self):
        a = self.args
        parts = ("routed", "dense") if a.part == "all" else (a.part,)
        g = []
        rungs = [int(c[1:]) if c.startswith("q") else q256_of(*parse_case(c, self.family)) for c in a.cases.split(",")]
        if "routed" in parts:
            if a.refs:
                g.append(("vllm_fp8_moe" if self.fp8 else "vllm_bf16_moe", None))
            for q in rungs:
                for mode in (0, 2):
                    g.append(("routed", (q, mode)))
        if "dense" in parts:
            for shape in a.shapes.split(","):
                if a.refs:
                    g.append(("scaled_mm" if self.fp8 else "bf16_linear", shape))
                for q in rungs:
                    g.append(("dense", (q, shape)))
        return g

    def variants(self, kind):
        if kind in ("routed", "vllm_fp8_moe", "vllm_bf16_moe"):
            v = [(m, "balanced") for m in self.ms] if self.args.routing_kind != "recorded" else []
            if self.args.routing_kind != "balanced":
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
        if not self.fp8 and q > 256 * rf.RATE_MAX:
            return self.build_compact(q, mode=mode)
        r_lo, frac = parse_case(f"q{q}", self.family)
        rows, cols = (INTER, HIDDEN) if mode == 0 else (HIDDEN, INTER)
        n_hi = 0 if frac is None else round(cols * frac)
        seed = zlib.crc32(f"paired:{mode}".encode())
        projs = [build_projection(rf, EXPERTS, rows, cols, r_lo, n_hi, seed + i, self.dev, self.mma8,
                                  bf16_table=not self.fp8)
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
            g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(f"x:{mode}:{m}:{how}".encode()))
            xrows = m if mode == 0 else routes
            x, a_scale = self.activation(xrows, cols, g)
            out = torch.empty((routes, rows), dtype=torch.bfloat16, device=self.dev)
            counter = torch.zeros(1, dtype=torch.int32, device=self.dev)

            def call():
                counter.zero_()
                self.lib.routed_fused_forward(
                    mode, self.fp8, x, a_scale, p0["words"], p1["words"], p0["table"], p1["table"],
                    p0["init"], p1["init"], p0["has_init"], p1["has_init"], p0["scale"], p1["scale"],
                    p0["runs"], p1["runs"], p0["bdesc"], p1["bdesc"], p0["tile_words"], slot_words, False,  # these benchmark bodies use legacy order
                    offsets, flat_sorted, rw_sorted, item_off, counter, TOP_K,
                    0 if mode == 0 else 1, mode == 2, SWIGLU_LIMIT, out, self.sms, bm)
            touched = int(torch.unique(ids).numel())
            wire = touched * sum(p["bytes_per_expert"] for p in projs)
            out_cols = 2 * rows if mode == 0 else rows
            fl = floor_ms(wire + xrows * cols * self.abytes + routes * rows * 2, 2.0 * routes * out_cols * cols,
                          self.tflops)
            st = routing_stats(ids)
            meta = {"bm": bm, "routes": routes, "touched": touched, "wire_bytes": wire, "floor": fl,
                    "superblocks": st["superblocks"] if bm == 64 else st["superblocks_128"]}
            return meta, call, out
        return head, make, projs

    def build_dense(self, q, shape):
        rf = self.rf
        if not self.fp8 and q > 256 * rf.DENSE_RATE_MAX["value"]:
            return self.build_compact(q, shape=shape)
        rows, cols = PROTOCOL_DENSE[shape]
        r_lo, frac = parse_case(f"q{q}", self.family)
        n_hi = 0 if frac is None else round(cols * frac)
        head = {"kind": "dense", "q256": q, "shape": shape, "rows": rows, "cols": cols, "r_lo": r_lo,
                "n_hi": n_hi, "geometry": geometry(rf, r_lo, frac, 2, self.mma8)}
        if rows % rf.BN:
            head["refused"] = f"{rows} rows; the dense identity needs a multiple of {rf.BN}"
            return head, None, None
        p = build_projection(rf, 1, rows, cols, r_lo, n_hi, zlib.crc32(f"paired:{shape}".encode()),
                             self.dev, self.mma8, bf16_table=not self.fp8)
        head["tile_words"] = p["tile_words"]

        def make(m, _how):
            g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(f"x:{shape}:{m}".encode()))
            x, a_scale = self.activation(m, cols, g)
            out = torch.empty((m, rows), dtype=torch.bfloat16, device=self.dev)
            counter = torch.zeros(1, dtype=torch.int32, device=self.dev)
            s = rf.dense_k_split(m, rows, cols, self.sms, tile_words=p["tile_words"])
            bm = rf.superblock_rows(self.library, 2, m, dense=True) if s == 1 else rf.BM
            partial = (torch.empty((s, m, rows), dtype=torch.float32, device=self.dev) if s > 1
                       else torch.empty(0, dtype=torch.float32, device=self.dev))
            wscale = p["scale"].reshape(1, rows)

            def call():
                counter.zero_()
                self.lib.dense_forward(self.fp8, x, a_scale, p["words"], p["table"], p["init"], p["has_init"][:1],
                                       wscale, p["runs"], p["bdesc"], int(p["tile_words"]),
                                       int(p["slot_words"]), counter, int(s), partial, out, self.sms, int(bm))
            wire = p["bytes_per_expert"] * rows // (-(-rows // 512) * 512)
            act = m * cols * self.abytes + m * rows * 2 + (2 * s * m * rows * 4 if s > 1 else 0)
            meta = {"k_split": s, "bm": bm, "wire_bytes": wire,
                    "floor": floor_ms(wire + act, 2.0 * m * rows * cols, self.tflops)}
            return meta, call, out
        return head, make, p

    def build_compact(self, q, mode=None, shape=None):
        from tessera.native_window_moe import _silu_and_mul
        routed = mode is not None
        rows, cols = ((INTER, HIDDEN) if mode == 0 else (HIDDEN, INTER)) if routed else PROTOCOL_DENSE[shape]
        r_lo, frac = parse_case(f"q{q}", "value")
        n_hi = round(cols * (frac or 0))
        seed = zlib.crc32(f"paired:{mode if routed else shape}".encode())
        projs = [build_projection(self.rf, EXPERTS if routed else 1, rows, cols,
                    r_lo, n_hi, seed + i, self.dev, False, bf16_table=True)
                 for i in range(2 if mode == 0 else 1)]
        prepared = [compact_projection(p, rows, cols, grouped=routed) for p in projs]
        width = projs[0]["window_bits"]
        head = {"kind": "routed" if routed else "dense", "q256": q,
                "rows": rows, "cols": cols, "r_lo": r_lo, "n_hi": n_hi,
                "arity": 1, "body_kind": "WINDOW", "window_bits": width,
                "tile_words": projs[0]["tile_words"],
                "geometry": geometry(self.rf, r_lo, frac, mode if routed else 2, False),
                "path": "compact_grouped_folded" if routed else "compact_dense_folded",
                "owner_refusal": f"compact_prep intake caps {'routed at 8' if routed else 'dense at 14'} bits per column; direct public compact constructor geometry, not serving intake admission"}
        head["mode" if routed else "shape"] = mode if routed else shape
        def make(m, how):
            seed_key=f"x:{mode}:{m}:{how}" if routed else f"x:{shape}:{m}"
            g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(seed_key.encode()))
            xrows = m * TOP_K if mode == 2 else m
            x, _scale = self.activation(xrows, cols, g)
            ids, weights = self.routing(m, how) if routed else (None, None)
            holder = {}
            def invoke():
                if mode == 0:
                    gate = prepared[0](x, ids, weights, preserve=True)
                    up = prepared[1](x, ids, weights, preserve=True)
                    holder["out"] = _silu_and_mul(gate, up, clamp_limit=SWIGLU_LIMIT).reshape(m * TOP_K, rows)
                elif routed:
                    holder["out"] = prepared[0](x, ids, weights, route_input=True, round_routes=True)
                else:
                    holder["out"] = prepared[0](x)
                return holder["out"]
            call = capture_compiler(invoke, routed)
            meta = {"bm": 64, "k_split": 1, "wire_bytes": sum(p["bytes_per_expert"] for p in projs),
                    "path_scope": "actual compact public projection; down includes route reduction; gate/up includes clipped SwiGLU" if routed else "actual public PreparedWindowGemm folded BF16",
                    "window_bits": width}
            return meta, call, holder
        return head, make, (projs, prepared)


    def activation(self, rows, cols, g):
        """The family's A operand: e4m3 with a per-row scale, or bf16 and no scale."""
        if self.fp8:
            x = (torch.randn(rows, cols, device=self.dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
            return x, torch.rand(rows, device=self.dev, generator=g) * 0.1 + 0.01
        x = (torch.randn(rows, cols, device=self.dev, generator=g) * 0.5).to(torch.bfloat16)
        return x, torch.empty(0, dtype=torch.float32, device=self.dev)

    def build_vllm_bf16(self):
        """vLLM's unquantized Triton MoE on bf16 experts of the same shapes: the
        BF16 source passthrough of a routed stack (16 bits per weight)."""
        from vllm.model_executor.layers.fused_moe import fused_experts
        dev, E = self.dev, EXPERTS
        g = torch.Generator(device=dev).manual_seed(8016)
        w13 = (torch.randn(E, 2 * INTER, HIDDEN, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(E, HIDDEN, INTER, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        head = {"kind": "vllm_bf16_moe", "q256": 4096, "backend": "fused_moe.fused_experts (Triton, unquantized)"}

        def make(m, how):
            ids, w = self.routing(m, how)
            x = (torch.randn(m, HIDDEN, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            holder = {}

            def call():
                holder["out"] = fused_experts(x, w13, w2, w, ids)
            call()
            torch.cuda.synchronize()
            touched = int(torch.unique(ids).numel())
            wire = touched * 3 * INTER * HIDDEN * 2
            routes = m * TOP_K
            meta = {"routes": routes, "touched": touched, "wire_bytes": wire,
                    "floor": floor_ms(wire + m * HIDDEN * 4, 2.0 * routes * 3 * INTER * HIDDEN, MMA_BF16_TFLOPS)}
            return meta, call, holder
        return head, make, (w13, w2)

    def build_bf16_linear(self, shape):
        """bf16 ``F.linear`` (cuBLAS) at the dense shape: the BF16 source passthrough."""
        rows, cols = PROTOCOL_DENSE[shape]
        dev = self.dev
        g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"bf16:{shape}".encode()))
        w = (torch.randn(rows, cols, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        head = {"kind": "bf16_linear", "q256": 4096, "shape": shape, "rows": rows, "cols": cols}

        def make(m, _how):
            x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            holder = {}

            def call():
                holder["out"] = torch.nn.functional.linear(x, w)
            meta = {"wire_bytes": rows * cols * 2,
                    "floor": floor_ms(rows * cols * 2 + m * cols * 2 + m * rows * 2, 2.0 * m * rows * cols,
                                      MMA_BF16_TFLOPS)}
            return meta, call, holder
        return head, make, w

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
        if kind == "vllm_bf16_moe":
            return self.build_vllm_bf16()
        if kind == "bf16_linear":
            return self.build_bf16_linear(spec)
        return self.build_scaled_mm(spec)

    # -- Nsight Compute: one profiled call per (group, M), no timing
    def run_ncu(self):
        """Under ``bench_t8r.sh``'s ``BENCH_NCU=1`` (``ncu --profile-from-start
        off``): each cell warms up, then ONE call runs inside a
        ``cudaProfilerStart``/``Stop`` range, so NCU profiles exactly one
        launch per (group, M) and nothing else."""
        a = self.args
        for kind, spec in self.groups():
            gkey = f"{kind}:{spec}"
            head, make, keep = self.build(kind, spec)
            if make is None:
                emit({"group": gkey, "refused": head.get("refused")})
                continue
            for m, how in self.variants(kind):
                ckey = f"{m}" + (f":{how}" if how else "")
                _meta, call, _out = make(m, how)
                for _ in range(max(1, a.warmup)):
                    call()
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                call()
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                emit({"group": gkey, "M": ckey, "ncu": "profiled"})
                del call, _out
            del keep
            torch.cuda.empty_cache()

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
                        how_t, samples = time_call(call, a.warmup, a.iters,
                                                   graph=kind not in ("vllm_fp8_moe", "vllm_bf16_moe"))
                        cell[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                     "samples_ms": samples,
                                     "timer": how_t, "unix": time.time(), "clock": self.clock.read()}
                        if pas == "F":
                            cell.update({k: v for k, v in meta.items()})
                            if isinstance(out, torch.Tensor):
                                cell["out_sha256"] = sha(out)
                            elif isinstance(out, dict) and isinstance(out.get("out"), torch.Tensor):
                                cell["out_sha256"] = sha(out["out"])
                            cell["profile"] = kernel_profile(call, reps=a.prof_reps)
                            if m in self.power_ms:
                                cell["power"] = self.power.sample_during(call, a.power_s)
                            if hasattr(call, "compiler_resources"):
                                cell["compiler_resources"] = call.compiler_resources
                            if kind in ("routed", "dense"):
                                u = kernel_usage(usage, head.get("mode", 2), kind == "dense", head["r_lo"],
                                                 head["n_hi"] > 0, meta.get("bm", 64), fp8=self.fp8, split=meta.get("k_split", 1) > 1)
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
    ap.add_argument("--routing", default="", help="recorded routing root with m<M>/ for the requested M")
    ap.add_argument("--routing-kind", choices=("balanced", "recorded", "both"), default="both")
    ap.add_argument("--refs", action="store_true", help="vLLM FP8 MoE and torch._scaled_mm references")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prof-reps", type=int, default=3)
    ap.add_argument("--power-ms", default="512,8192")
    ap.add_argument("--power-s", type=float, default=0.5)
    ap.add_argument("--library", default=None)
    ap.add_argument("--cpu-preflight", action="store_true", help="CPU imports, parsing, shapes, tiny packed-wire read; no CUDA timing")
    ap.add_argument("--config", default="", help="actual GLM config for shape provenance")
    ap.add_argument("--ncu", action="store_true",
                    help="profile one call per (group, M) under bench_t8r.sh's BENCH_NCU=1; no timing")
    ap.add_argument('--data-manifest',default='',help='admitted PB whole-file routing readset')
    args = ap.parse_args()
    if ":" in args.shapes:
        shapes = {}
        for item in args.shapes.split(","):
            name, rows, columns = item.split(":")
            rows, columns = int(rows), int(columns)
            if not name or name in shapes or rows <= 0 or columns <= 0:
                raise ValueError("Invalid or duplicate explicit dense shape")
            shapes[name] = (rows, columns)
        PROTOCOL_DENSE.update(shapes)
        args.shapes = ",".join(shapes)
    os.makedirs(args.out, exist_ok=True)
    from tessera import routed_fused as rf
    library = args.library or rf.library_for("e4m3")
    stage_geometry_inputs(args)
    mma8 = rf.library_mma8(library)
    if args.cpu_preflight:
        cases = [parse_case(c, rf.LIBRARIES[library][1]) for c in args.cases.split(",")]
        ms = [int(m) for m in args.ms.split(",")]
        assert ms and all(m > 0 for m in ms)
        shapes = {s: PROTOCOL_DENSE[s] for s in args.shapes.split(",")}
        config = json.load(open(args.config)) if args.config else None
        if config:
            text = config.get("text_config", config)
            assert (text["hidden_size"], text["moe_intermediate_size"] // 2, text["n_routed_experts"], text["num_experts_per_tok"]) == (HIDDEN, INTER, EXPERTS, TOP_K)
        tiny = []
        for r, frac in cases:
            p = build_projection(rf, 1, 512, 256, r, round(256 * (frac or 0)), 41, torch.device("cpu"), mma8, bf16_table=library == "value")
            assert p["words"].shape == (1, p["tile_words"])
            if library == "value":
                assert bool(torch.isfinite(p["table"].view(torch.bfloat16)).all())
                compact_projection(p, 512, 256, grouped=False)
                compact_projection(p, 512, 256, grouped=True)
            tiny.append({"q256": q256_of(r, frac), "tile_words": p["tile_words"], "word": int(p["words"][0, 0]), "window_bits": p["window_bits"], "table_dtype": str(p["table"].dtype)})
        recorded = pick_recorded(args.routing, ms) if args.routing else {}
        for m, entry in recorded.items():
            ids, _weights = recorded_routing(entry["path"], m, "cpu")
            assert tuple(ids.shape) == (m, TOP_K) and int(ids.min()) >= 0 and int(ids.max()) < EXPERTS
        if args.routing_kind == "recorded":
            assert recorded, "No recorded routing at the requested M"
        json.dump({"cpu_preflight": "passed", "library": library, "shapes": shapes, "M": ms, "tiny_wire_reads": tiny, "config": args.config, "recorded": recorded, "routing_kind": args.routing_kind}, open(os.path.join(args.out, "cpu-preflight.json"), "w"), indent=2)
        print("CPU preflight passed; no GPU results", flush=True)
        return 0
    lib = rf._ext(library)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    usage = resource_usage(lib)
    sw = Sweep(args, rf, lib, library, mma8, dev, sms)
    if args.ncu:
        sw.run_ncu()
        print("done", flush=True)
        return 0
    props = torch.cuda.get_device_properties(dev)
    meta = {"device": torch.cuda.get_device_name(), "sms": sms, "experts": EXPERTS, "hidden": HIDDEN,
            "architecture": f"sm_{props.major}{props.minor}", "shared_memory_available": props.shared_memory_per_block_optin,
            "library_sha256": hashlib.sha256(open(lib.__file__, "rb").read()).hexdigest(),
            "activation_contract": "float8_e4m3fn per-row float32 scale; BF16 output; seeded normal std=0.5" if sw.fp8 else "BF16 activations without activation scale; finite BF16 table; folded row scale; seeded normal std=0.5",
            "paired_seed_contract": "fixed shape/mode/M/routing seeds independent of rung; synthetic packed wires",
            "inter": INTER, "top_k": TOP_K, "part": args.part, "cases": args.cases, "ms": args.ms,
            "shapes": args.shapes, "recorded": sw.recorded, "library": library,
            "family": rf.LIBRARIES[library][1], "mma_tflops": sw.tflops,
            "dense_seed_without_routing_suffix": True,
            "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": sw.power.source,
            "envelope_w": ENVELOPE_W, "start_unix": time.time(), "torch": torch.__version__,
            "statistic": "mean of the forward and reverse passes' medians (graph replay); spread = |F - R| / mean",
            "resource_usage": usage}
    if sw.family == "value":
        meta.update(format="TESSERA_BF16_K1", rung_min=256, rung_max=4096, grid_step_q256=1,
                    grid_owner="prismaquant.tessera_formats.family_q256_bounds/realisable_rungs(default BF16 WINDOW recipe, step_q256=1)")
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
