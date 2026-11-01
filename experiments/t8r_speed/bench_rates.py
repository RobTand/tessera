"""Every Tessera-8 column rate through the fused window kernel, at GLM-5.3-Flash TP2 shapes.

The allocator may pick any q256 rung from 256 to 2048 for an E4M3 unit.  A rung
is one column rate (q256 = 256 * r, one run) or two adjacent rates (a fraction
of the columns at r + 1, two runs).  This bench times the fused kernel's raw
launches at each such pair, on random wires, against the pair's floor, so the
rates furthest from roofline are named by measurement rather than guessed.

Routed (``--part routed``): the TP2 rank's expert stack (288 experts, hidden
4096, 1024 intermediate columns per rank), the gate/up launch (mode 0, two
projections, SwiGLU epilogue) and the down launch (mode 2), balanced routing
(token t picks experts (8t + j) mod 288).  The superblock width is the one the
serve picks per launch (``routed_fused.superblock_rows``) unless ``--bm`` fixes
it.  ``--vllm-fp8`` adds vLLM's own FP8 MoE on random E4M3 weights of the same
shapes (W8A8, per-channel weight and per-token activation scales): the
zero-kernel alternative at q256 2048, whose wire carries FP8's bytes.

Dense (``--part dense``): one role of a dense Linear per shape (rows x cols per
rank), through ``dense_forward`` with the split-K choice the serve makes
(``routed_fused.dense_k_split``).  ``--dense-ref`` adds ``torch._scaled_mm``
(E4M3 x E4M3 -> bf16, row-wise scales) and bf16 ``F.linear`` at the same shape.

Per cell it records: the output's sha256 (so two kernel arms compare bitwise);
CUDA-event wall time (median, IQR); for M <= 16 the same call replayed from a
captured CUDA graph; torch.profiler self device time per kernel; NVML board
power and SM clock over a back-to-back loop (mean against the 140 W envelope,
calls per joule); and the floor.  The floor is the larger of the bytes the
launch must move (wire of the touched experts, plus the activation read and
the output write) at the measured read bandwidth, and its FLOPs at the
measured E4M3 ``mma.sync`` peak (``docs/measurements/2026-09-30-fp8-prefill-
roofline.md``: 232.2 GB/s, 246.5 TFLOPS on GB10).  ``roofline_frac`` is
floor / measured kernel time.

Words, tables, start states and scales are random: the decode's arithmetic and
memory traffic do not depend on the values, except the table gathers' bank
conflicts, which random 14-bit windows make uniform.

Usage: bench_rates.py --out DIR --part routed|dense [--cases r4,r8,r4+q,q944]
       [--ms 1,8,512,2048] [--modes 0,2] [--shapes name:rows:cols,...]
       [--vllm-fp8] [--dense-ref] [--bm auto|64|128]
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
from bench_t8r import (ENVELOPE_W, EXPERTS, HIDDEN, TOP_K, PowerSampler, balanced_routing,  # noqa: E402
                       kernel_profile, summarize, time_events)

INTER = 1024            # the TP2 rank's routed intermediate columns
BK = 32
READ_GBPS = 232.2       # measured device read bandwidth (fp8-prefill-roofline, sparky)
MMA_E4M3_TFLOPS = 246.5  # measured mma.sync m16n8k32 e4m3 peak (same doc)
MMA_BF16_TFLOPS = 123.1  # measured mma.sync m16n8k16 bf16 peak (same doc): the value family's instruction
SWIGLU_LIMIT = 10.0

# case -> (r_lo, fraction of columns at r_lo + 1, or None for one run)
CASES = {f"r{r}": (r, None) for r in range(1, 9)}
CASES.update({f"r{r}+q": (r, 0.25) for r in range(1, 8)})


def parse_case(case, family="e4m3"):
    """``rN`` / ``rN+q`` (the named cases), or ``qK``: the rung K = q256 itself --
    one run at K / 256 when 256 divides K, else the adjacent pair (K // 256,
    K // 256 + 1) with the fraction (K % 256) / 256 of the columns at the high rate."""
    if case in CASES and family == "e4m3":
        return CASES[case]
    if case.startswith("q") and case[1:].isdigit():
        k = int(case[1:])
        from tessera.control import grid_for_name
        grid = grid_for_name({"e4m3": "E4M3", "value": "BF16"}.get(family, family))
        lower, upper = 256 // grid.arity, 256 * grid.payload_bits // grid.arity
        if not lower <= k <= upper:
            raise ValueError(f"rung {k} is outside the {grid.name} producer range {lower}..{upper}")
        k *= grid.arity
        return (k // 256, None) if k % 256 == 0 else (k // 256, (k % 256) / 256)
    raise ValueError(f"unknown case {case!r}")

# Per-rank shapes at TP2, rows x cols (out_features x in_features), from the
# GLM-5.3-Flash config (hidden 4096; dense MLP 12288; shared expert 2048; KDA
# 64 heads x 128; MLA 64 heads, q_lora 1536, kv_lora 512, qk 256, v 256;
# indexer 32 heads x 128; vocab 154880; vision hidden 1024, MLP 4096).
DENSE_SHAPES = {
    "dense_gate_up": (12288, 4096), "dense_down": (4096, 6144),
    "shared_gate_up": (2048, 4096), "shared_down": (4096, 1024),
    "kda_qkv": (12288, 4096), "kda_q": (4096, 4096), "kda_o": (4096, 4096),
    "kda_fa_ga": (256, 4096), "kda_fb": (4096, 128), "kda_b": (32, 4096),
    "mla_qa_kva": (2048, 4096), "mla_qa": (1536, 4096), "mla_kva": (512, 4096),
    "mla_qb": (8192, 1536), "mla_o": (4096, 8192),
    "idx_wqb": (4096, 1536), "idx_wk": (128, 4096), "idx_wproj": (32, 4096),
    "lm_head": (77440, 4096),
    "vis_qkv": (1536, 1024), "vis_proj": (1024, 512), "vis_gate_up": (4096, 1024), "vis_down": (1024, 2048),
}


def q256_of(r_lo, frac):
    return 256 * r_lo + (0 if frac is None else round(256 * frac))


def build_projection(rf, e, rows, cols, r_lo, n_hi, seed, dev, mma8, bf16_table=False, window_bits=None):
    """Random words/table/init/scales for ``e`` experts of one projection, and its run tables,
    generated on the device from ``seed`` (deterministic, so two arms see the same bytes)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    if window_bits is None:
        from tessera.control import grid_for_name
        from tessera.export import wire_recipe
        window_bits = wire_recipe(grid_for_name("BF16" if bf16_table else "E4M3"),
                                  256 * r_lo + round(256 * n_hi / cols)).window_bits
    n_lo = cols - n_hi
    two = n_hi > 0
    pair = torch.tensor((r_lo, 0, n_lo, 0, r_lo + 1 if two else 0, n_lo, n_hi, 16 * n_lo * r_lo),
                        dtype=torch.int32)
    tile_words = rf.pair_tile_words(pair)
    # the wire pads rows to whole 512-row tiles (``compact_prep``: rows_p // TILE_ROWS)
    words_stride = -(-rows // 512) * tile_words
    words = torch.randint(-2**31, 2**31 - 1, (e, words_stride), generator=g, device=dev, dtype=torch.int32)
    if mma8:
        table = torch.randint(0, 256, (e, 1 << window_bits), generator=g, device=dev, dtype=torch.int32)
        table = torch.where((table & 0x7F) == 0x7F, table - 1, table).to(torch.uint8)
    elif bf16_table:
        # the value family's table holds bf16 weights: finite values, as a wire's are
        table = (torch.randn(e, 1 << window_bits, generator=g, device=dev) * 0.02).to(torch.bfloat16).view(torch.int16)
    else:
        table = torch.randint(-2**15, 2**15 - 1, (e, 1 << window_bits), generator=g, device=dev,
                              dtype=torch.int32).to(torch.int16)
    init = torch.randint(0, 1 << window_bits, (e, cols), generator=g, device=dev, dtype=torch.int32)
    has_init = torch.ones(e, dtype=torch.int32, device=dev)
    scale = torch.rand(e, rows, generator=g, device=dev) * 1e-2 + 1e-3
    # per expert: n_hi random columns at the high rate, in the packer's stable
    # (rate, column) order
    cidx = torch.arange(cols, device=dev).expand(e, cols)
    if n_hi:
        key = torch.rand(e, cols, generator=g, device=dev)
        hi = torch.zeros(e, cols, dtype=torch.bool, device=dev)
        hi.scatter_(1, torch.argsort(key, dim=1)[:, :n_hi], True)
        perm = torch.argsort(hi.to(torch.int64) * cols + cidx, dim=1)
    else:
        perm = cidx.clone()
    bdesc = rf.block_desc(perm, n_lo, cols)
    runs = pair.reshape(1, 8).expand(e, 8).contiguous().to(dev)
    return {"words": words, "table": table, "init": init, "has_init": has_init, "scale": scale,
            "runs": runs, "bdesc": bdesc.contiguous(), "tile_words": tile_words,
            "slot_words": rf.slot_words_for_pair(pair), "bytes_per_expert": 4 * words_stride,
            "perm": perm.to(torch.int32).contiguous(), "window_bits": window_bits}


def routing_tables(ids, w, e, bm):
    flat = ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(e, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = torch.zeros(e + 1, dtype=torch.int32, device=ids.device)
    offsets[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    order = torch.argsort(flat, stable=True)
    item_off = torch.zeros(e + 1, dtype=torch.int32, device=ids.device)
    item_off[1:] = torch.cumsum((counts + bm - 1) // bm, 0, dtype=torch.int32)
    return offsets, order.to(torch.int32).contiguous(), w.reshape(-1)[order].contiguous(), item_off


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def graph_time(call, iters):
    """The call replayed from a captured CUDA graph (the decode path's proxy)."""
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
    return summarize(time_events(graph.replay, 5, iters))


class Clock:
    """SM clock (MHz) and temperature, where NVML is importable."""

    def __init__(self):
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            pass

    def read(self):
        if self._h is None:
            return None
        try:
            return {"sm_mhz": self._nv.nvmlDeviceGetClockInfo(self._h, self._nv.NVML_CLOCK_SM),
                    "temp_c": self._nv.nvmlDeviceGetTemperature(self._h, self._nv.NVML_TEMPERATURE_GPU)}
        except Exception:  # noqa: BLE001
            return None


def measure(call, out, args, power, clock, *, graph_ok):
    call()
    torch.cuda.synchronize()
    cell = {"out_sha256": sha(out), "clock_start": clock.read()}
    cell["wall"] = summarize(time_events(call, args.warmup, args.iters))
    if graph_ok:
        try:
            cell["graph"] = graph_time(call, args.iters)
        except Exception as exc:  # noqa: BLE001
            cell["graph"] = {"error": repr(exc)[:300]}
    cell["profile"] = kernel_profile(call, reps=args.prof_reps)
    cell["power"] = power.sample_during(call, args.power_s)
    cell["clock_end"] = clock.read()
    return cell


def floor_ms(move_bytes, flops, tflops=MMA_E4M3_TFLOPS):
    t_mem = move_bytes / (READ_GBPS * 1e9) * 1e3
    t_mma = flops / (tflops * 1e12) * 1e3
    return {"move_bytes": move_bytes, "flops": flops, "mem_ms": t_mem, "mma_ms": t_mma,
            "floor_ms": max(t_mem, t_mma), "bound": "memory" if t_mem >= t_mma else "mma"}


def finish_cell(cell, fl):
    k_ms = cell["profile"]["kernel_us_per_call"] / 1e3
    cell["floor"] = fl
    cell["kernel_ms"] = k_ms
    cell["roofline_frac"] = fl["floor_ms"] / k_ms if k_ms else None
    p = cell["power"]
    if p.get("mean_w"):
        cell["j_per_call"] = p["mean_w"] * k_ms / 1e3
    return cell


def emit(rec):
    print(json.dumps(rec), flush=True)


# ------------------------------------------------------------------ routed
def run_routed(args, rf, lib, library, mma8, dev, sms, power, clock, results):
    ms = [int(v) for v in args.ms.split(",")]
    modes = [int(v) for v in args.modes.split(",")]
    for case in args.cases.split(","):
        r_lo, frac = parse_case(case)
        for mode in modes:
            rows, cols = (INTER, HIDDEN) if mode == 0 else (HIDDEN, INTER)
            n_hi = 0 if frac is None else round(cols * frac)
            seed = zlib.crc32(f"{case}:{mode}".encode())
            projs = [build_projection(rf, EXPERTS, rows, cols, r_lo, n_hi, seed + i, dev, mma8)
                     for i in range(2 if mode == 0 else 1)]
            p0, p1 = projs[0], projs[-1]
            rec = {"part": "routed", "case": case, "q256": q256_of(r_lo, frac), "mode": mode, "r_lo": r_lo,
                   "n_hi": n_hi, "rows": rows, "cols": cols, "tile_words": p0["tile_words"],
                   "slot_words": p0["slot_words"], "library": library, "cells": {}}
            for m in ms:
                bm = rf.superblock_rows(library, mode, m) if args.bm == "auto" else int(args.bm)
                if not rf.has_width(library, mode, bm):
                    bm = rf.BM
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
                        p0["runs"], p1["runs"], p0["bdesc"], p1["bdesc"], p0["tile_words"], slot_words, False,  # these benchmark bodies use legacy order
                        offsets, flat_sorted, rw_sorted, item_off, counter, TOP_K,
                        0 if mode == 0 else 1, mode == 2, SWIGLU_LIMIT, out, sms, bm)
                try:
                    cell = measure(call, out, args, power, clock, graph_ok=m <= 16)
                except Exception as exc:  # noqa: BLE001
                    cell = {"error": repr(exc)[:500]}
                    rec["cells"][str(m)] = cell
                    emit({"part": "routed", "case": case, "mode": mode, "M": m, "error": cell["error"]})
                    continue
                touched = min(EXPERTS, routes)
                wire = touched * sum(p["bytes_per_expert"] for p in projs)
                out_cols = 2 * rows if mode == 0 else rows
                act = xrows * cols + (routes * rows * 2)            # e4m3 A read + bf16 out write
                fl = floor_ms(wire + act, 2.0 * routes * out_cols * cols)
                cell.update({"bm": bm, "routes": routes, "wire_bytes": wire})
                finish_cell(cell, fl)
                rec["cells"][str(m)] = cell
                emit({"part": "routed", "case": case, "q256": rec["q256"], "mode": mode, "M": m, "bm": bm,
                      "kernel_ms": round(cell["kernel_ms"], 4), "wall_ms": round(cell["wall"]["median_ms"], 4),
                      "graph_ms": round(cell.get("graph", {}).get("median_ms", 0) or 0, 4),
                      "floor_ms": round(fl["floor_ms"], 4), "roof": round(cell["roofline_frac"] or 0, 3),
                      "W": round(cell["power"].get("mean_w") or 0, 1)})
            results.append(rec)
            del projs, p0, p1
            torch.cuda.empty_cache()
            save(args, results)


def run_vllm_fp8(args, dev, power, clock, results):
    """vLLM's FP8 MoE (W8A8: per-channel weight, per-token activation scales) on
    random E4M3 weights of the TP2 rank's shapes, end to end (quantise, sort,
    both GEMMs, SwiGLU, reduce)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig, FusedMoEParallelConfig,
                                                             RoutingMethodType)
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format, make_fp8_moe_kernel, make_fp8_moe_quant_config,
        select_fp8_moe_backend)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (kFp8DynamicTokenSym,
                                                                            kFp8StaticChannelSym)
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
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
    for name, t in (("w13_weight", w13c), ("w2_weight", w2c), ("w13_weight_scale", s13c), ("w2_weight_scale", s2c)):
        param(name, t)
    quant = make_fp8_moe_quant_config(
        fp8_backend=backend, w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
        a1_scale=None, a2_scale=None, per_act_token_quant=True, per_out_ch_quant=True,
        block_shape=None, gemm1_alpha=None, gemm1_beta=None, swiglu_limit=SWIGLU_LIMIT, layer=layer)
    kernel = make_fp8_moe_kernel(moe_quant_config=quant, moe_config=cfg, experts_cls=experts_cls,
                                 fp8_backend=backend, routing_tables=None)
    rec = {"part": "vllm_fp8_moe", "backend": str(getattr(backend, "name", backend)),
           "experts_cls": getattr(experts_cls, "__name__", str(experts_cls)), "cells": {}}
    for m in [int(v) for v in args.ms.split(",")]:
        ids, w = balanced_routing(m, dev)
        x = (torch.randn(m, HIDDEN, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        holder = {}

        def call():
            holder["out"] = kernel.apply(x, layer.w13_weight, layer.w2_weight, w, ids,
                                         activation=MoEActivation.SILU, global_num_experts=E,
                                         expert_map=None, apply_router_weight_on_input=False)
        try:
            call()
            torch.cuda.synchronize()
            cell = measure(call, holder["out"], args, power, clock, graph_ok=False)
        except Exception as exc:  # noqa: BLE001
            rec["cells"][str(m)] = {"error": repr(exc)[:500]}
            emit({"part": "vllm_fp8_moe", "M": m, "error": repr(exc)[:300]})
            continue
        routes = m * TOP_K
        touched = min(E, routes)
        wire = touched * 3 * INTER * HIDDEN
        fl = floor_ms(wire + m * HIDDEN * 2 * 2, 2.0 * routes * 3 * INTER * HIDDEN)
        finish_cell(cell, fl)
        rec["cells"][str(m)] = cell
        emit({"part": "vllm_fp8_moe", "M": m, "kernel_ms": round(cell["kernel_ms"], 4),
              "wall_ms": round(cell["wall"]["median_ms"], 4), "floor_ms": round(fl["floor_ms"], 4),
              "roof": round(cell["roofline_frac"] or 0, 3), "W": round(cell["power"].get("mean_w") or 0, 1)})
    results.append(rec)
    save(args, results)


# ------------------------------------------------------------------ dense
def run_dense(args, rf, lib, library, mma8, dev, sms, power, clock, results):
    ms = [int(v) for v in args.ms.split(",")]
    shapes = []
    for spec in args.shapes.split(","):
        if ":" in spec:
            name, r, c = spec.split(":")
            shapes.append((name, int(r), int(c)))
        else:
            shapes.append((spec, *DENSE_SHAPES[spec]))
    for name, rows, cols in shapes:
        refs = {}
        for case in args.cases.split(","):
            r_lo, frac = parse_case(case)
            n_hi = 0 if frac is None else round(cols * frac)
            rec = {"part": "dense", "shape": name, "rows": rows, "cols": cols, "case": case,
                   "q256": q256_of(r_lo, frac), "library": library, "cells": {}}
            # the dense identity's own shape predicate (rows % 128, cols % 32 and >= 128)
            why = None
            if rows % rf.BN:
                why = f"{rows} rows; the dense identity needs a multiple of {rf.BN}"
            elif cols % rf.BK or cols < rf.MIN_COLS:
                why = f"{cols} columns; the kernel needs a multiple of {rf.BK} and at least {rf.MIN_COLS}"
            if why:
                rec["refused"] = why
                results.append(rec)
                emit({"part": "dense", "shape": name, "case": case, "refused": why})
                continue
            p = build_projection(rf, 1, rows, cols, r_lo, n_hi, zlib.crc32(f"{name}:{case}".encode()), dev, mma8)
            rec.update({"tile_words": p["tile_words"], "slot_words": p["slot_words"]})
            for m in ms:
                g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"x:{name}:{case}:{m}".encode()))
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
                a_scale = torch.rand(m, device=dev, generator=g) * 0.1 + 0.01
                out = torch.empty((m, rows), dtype=torch.bfloat16, device=dev)
                counter = torch.zeros(1, dtype=torch.int32, device=dev)
                s = rf.dense_k_split(m, rows, cols, sms, tile_words=p["tile_words"])
                bm = rf.superblock_rows(library, 2, m, dense=True) if s == 1 else rf.BM
                if args.bm != "auto" and s == 1:
                    bm = int(args.bm) if rf.has_width(library, 2, int(args.bm)) else rf.BM
                partial = (torch.empty((s, m, rows), dtype=torch.float32, device=dev) if s > 1
                           else torch.empty(0, dtype=torch.float32, device=dev))
                wscale = p["scale"].reshape(1, rows)

                def call():
                    counter.zero_()
                    lib.dense_forward(True, x, a_scale, p["words"], p["table"], p["init"], p["has_init"][:1],
                                      wscale, p["runs"], p["bdesc"], int(p["tile_words"]), int(p["slot_words"]),
                                      counter, int(s), partial, out, sms, int(bm))
                try:
                    cell = measure(call, out, args, power, clock, graph_ok=m <= 16)
                except Exception as exc:  # noqa: BLE001
                    rec["cells"][str(m)] = {"error": repr(exc)[:500]}
                    emit({"part": "dense", "shape": name, "case": case, "M": m, "error": repr(exc)[:300]})
                    continue
                wire = p["bytes_per_expert"]
                act = m * cols + m * rows * 2 + (2 * s * m * rows * 4 if s > 1 else 0)
                fl = floor_ms(wire + act, 2.0 * m * rows * cols)
                cell.update({"k_split": s, "bm": bm, "wire_bytes": wire})
                finish_cell(cell, fl)
                rec["cells"][str(m)] = cell
                emit({"part": "dense", "shape": name, "q256": rec["q256"], "M": m, "S": s, "bm": bm,
                      "kernel_ms": round(cell["kernel_ms"], 4), "graph_ms": round(cell.get("graph", {}).get("median_ms", 0) or 0, 4),
                      "floor_ms": round(fl["floor_ms"], 4), "roof": round(cell["roofline_frac"] or 0, 3),
                      "W": round(cell["power"].get("mean_w") or 0, 1)})
            results.append(rec)
            del p
            torch.cuda.empty_cache()
            save(args, results)
        if args.dense_ref:
            run_dense_refs(args, name, rows, cols, ms, dev, power, clock, results)


def run_dense_refs(args, name, rows, cols, ms, dev, power, clock, results):
    """``torch._scaled_mm`` FP8 (E4M3 x E4M3 -> bf16, row-wise scales: the W8A8
    arithmetic at 8 bits per weight) and bf16 ``F.linear`` at the same shape."""
    g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"ref:{name}".encode()))
    w8 = (torch.randn(rows, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    sw = torch.rand(1, rows, device=dev, generator=g) * 1e-2 + 1e-3
    wb = (torch.randn(rows, cols, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    for kind in ("scaled_mm_fp8", "bf16_linear"):
        rec = {"part": "dense_ref", "kind": kind, "shape": name, "rows": rows, "cols": cols, "cells": {}}
        for m in ms:
            if kind == "scaled_mm_fp8":
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
                sa = torch.rand(m, 1, device=dev, generator=g) * 0.1 + 0.01
                holder = {}

                def call():
                    holder["out"] = torch._scaled_mm(x, w8.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)
                wbytes = rows * cols
                xbytes = m * cols
            else:
                x = (torch.randn(m, cols, device=dev, generator=g) * 0.5).to(torch.bfloat16)
                holder = {}

                def call():
                    holder["out"] = torch.nn.functional.linear(x, wb)
                wbytes = rows * cols * 2
                xbytes = m * cols * 2
            try:
                call()
                torch.cuda.synchronize()
                cell = measure(call, holder["out"], args, power, clock, graph_ok=m <= 16)
            except Exception as exc:  # noqa: BLE001
                rec["cells"][str(m)] = {"error": repr(exc)[:500]}
                emit({"part": "dense_ref", "kind": kind, "shape": name, "M": m, "error": repr(exc)[:300]})
                continue
            fl = floor_ms(wbytes + xbytes + m * rows * 2, 2.0 * m * rows * cols)
            finish_cell(cell, fl)
            rec["cells"][str(m)] = cell
            emit({"part": "dense_ref", "kind": kind, "shape": name, "M": m, "kernel_ms": round(cell["kernel_ms"], 4),
                  "floor_ms": round(fl["floor_ms"], 4), "roof": round(cell["roofline_frac"] or 0, 3),
                  "W": round(cell["power"].get("mean_w") or 0, 1)})
        results.append(rec)
        save(args, results)


META = {}


def save(args, results):
    json.dump({"meta": META, "results": results}, open(os.path.join(args.out, f"bench_rates_{args.part}.json"), "w"),
              indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("routed", "dense"), required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--modes", default="0,2")
    ap.add_argument("--ms", default="1,2,4,8,16,64,512,2048,8192")
    ap.add_argument("--shapes", default="dense_gate_up,dense_down,shared_gate_up,shared_down")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--prof-reps", type=int, default=5)
    ap.add_argument("--power-s", type=float, default=1.5)
    ap.add_argument("--library", default=None, help="a routed_fused.LIBRARIES key of the E4M3 family")
    ap.add_argument("--bm", default="auto", help="auto (the serve's choice), 64 or 128")
    ap.add_argument("--vllm-fp8", action="store_true")
    ap.add_argument("--dense-ref", action="store_true")
    ap.add_argument("--ncu", action="store_true", help="accepted for the wrapper; not used")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from tessera import routed_fused as rf
    library = args.library or rf.library_for("e4m3")
    mma8 = rf.library_mma8(library)
    lib = rf._ext(library)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    power, clock = PowerSampler(), Clock()
    META.update({"device": torch.cuda.get_device_name(), "sms": sms, "experts": EXPERTS, "hidden": HIDDEN,
                 "inter": INTER, "top_k": TOP_K, "part": args.part, "cases": args.cases, "ms": args.ms,
                 "kernel_sha": os.environ.get("KERNEL_SHA"), "tessera_head": os.environ.get("TESSERA_HEAD"),
                 "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
                 "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": power.source,
                 "library": library, "bm": args.bm, "read_gbps": READ_GBPS, "mma_tflops": MMA_E4M3_TFLOPS,
                 "envelope_w": ENVELOPE_W, "start_unix": time.time(), "torch": torch.__version__})
    results = []
    if args.part == "routed":
        if args.vllm_fp8:
            try:
                run_vllm_fp8(args, dev, power, clock, results)
            except Exception as exc:  # noqa: BLE001
                results.append({"part": "vllm_fp8_moe", "error": repr(exc)[:800]})
                emit({"part": "vllm_fp8_moe", "error": repr(exc)[:300]})
        run_routed(args, rf, lib, library, mma8, dev, sms, power, clock, results)
    else:
        run_dense(args, rf, lib, library, mma8, dev, sms, power, clock, results)
    META["end_unix"] = time.time()
    save(args, results)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
