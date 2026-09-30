"""Speed of the fused window kernel's E2M1 family against the NVFP4 routes (tessera#750, T-4).

    python3 experiments/t4_code/bench_e2m1.py --out DIR --part routed|dense|all
        [--ms 1,64,512,2048,8192] [--rungs 448,704,960] [--legs e2m1,a4,vllm]

Shapes are GLM-5.3-Flash per rank at TP2.  Routed: 288 experts, top 8,
hidden 4096, intermediate 1024 (gate/up rows and down columns are the rank's
half of 2048).  Dense: the per-rank shapes of the #750 dense list.  The
default leaves out the whole TP2 ``lm_head`` (77,440 rows): its window encode
does not fit a 40 GB budget, so it is timed at an eighth and a quarter of its
rows (``lm_head_r8``, ``lm_head_r4``).  Every encode's seconds and CUDA
allocator peak are recorded (``encodes``).

Legs, each over the same activations and routing:

* ``e2m1`` (after): ``routed_fused_e2m1.FusedRoutedE2M1MoE.__call__`` -- the
  routing tables, ``scaled_fp4_quant`` at the static global, the gate/up
  launch with the SwiGLU epilogue, the second quantisation, the down launch
  and the fixed-order token sum -- on window-body wires over the LUT16 plane
  (L = 14) at each ``--rungs`` q256.  Dense: ``dense_forward`` at every K
  split from 1 to the cap (``dense_split_max``); the cell keeps every split's
  time and names the fastest.
* ``a4`` (before, the served T-4 route): the native TCQ span-2 pipeline of
  ``nvfp4_moe_route``'s ``apply`` -- ``a4_grouped_apply`` for gate, up and
  down, vLLM's ``apply_moe_activation`` with the clamp, the ``index_add_``
  combine -- at TCQ q896, the rung the contract's routed cells name.  Dense:
  ``a4_dense_apply``.
* ``vllm`` (before, stock): the modelopt NVFP4 MoE kernel vLLM's own oracle
  selects on this box for ``(kNvfp4Static, kNvfp4Dynamic)`` with the clamp,
  built through ``convert_to_nvfp4_moe_kernel_format`` /
  ``make_nvfp4_moe_quant_config`` / ``make_nvfp4_moe_kernel`` as
  ``ModelOptNvFp4FusedMoE`` builds it (4.5 bits per weight).  Dense: vLLM's
  ``cutlass_scaled_fp4_mm`` after ``scaled_fp4_quant``.

Expert weights: one wire per projection is encoded from Gaussian weights and
placed in every expert's slot, so each expert has its own copy of the bytes
(the wire traffic is the real one) without 864 encodes.  Balanced routing:
token t picks experts (8t + j) mod 288.

Timing: CUDA graph replay where the call captures, CUDA events otherwise
(the cell says which); each cell is timed in a forward pass and a reverse
pass over the cell list, and its time is the mean of the two medians.  The
forward pass also records torch.profiler device time per kernel and, at
``--power-ms``, NVML board power over a back-to-back loop; every cell has its
unix time so the Netdata series can be aligned to it.

Writes ``DIR/bench_e2m1_<part>.json``.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "t8r_speed"))

from bench_t8r import (ENVELOPE_W, EXPERTS, HIDDEN, SWIGLU_LIMIT, TOP_K, PowerSampler,  # noqa: E402
                       balanced_routing, kernel_profile, time_events)

INTER = 1024                    # the TP2 rank's routed intermediate columns
L = 14
A4_Q256 = 896
#: rows x cols per rank at TP2 (GLM-5.3-Flash; the #750 dense list).  The
#: indexer's ``wk`` (128 rows) and ``weights_proj`` (32) and ``lm_head``
#: (77440 = 302.5 blocks of 256) end inside a 256-row block.
DENSE = {
    "mla_o": (4096, 8192), "mla_qb": (8192, 1536), "mla_qa_kva": (2048, 4096),
    "mla_qa": (1536, 4096), "mla_kva": (512, 4096), "kda_o": (4096, 4096),
    "idx_wqb": (4096, 1536), "idx_wk": (128, 4096), "idx_wproj": (32, 4096),
    "lm_head": (77440, 4096),
    "vis_qkv": (1536, 1024), "vis_proj": (1024, 512), "vis_gate_up": (4096, 1024), "vis_down": (1024, 2048),
    # lm_head's rows cut to a multiple of 32: an eighth (9696) and a quarter
    # (19360).  The whole unit's window encode does not fit a 40 GB budget
    # (PB 613dcd70, memory_budget_exceeded), so it is not a default shape.
    "lm_head_r8": (9696, 4096), "lm_head_r4": (19360, 4096),
}
DEFAULT_SHAPES = [s for s in DENSE if s != "lm_head"]
#: Per encode: seconds and the CUDA allocator's peak, keyed "body rows x cols q".
ENCODES = {}


def emit(rec):
    print(json.dumps(rec), flush=True)


def gaussian(rows, cols, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(rows, cols, generator=g, device=dev) * 0.02


def time_call(call, warmup, iters, graph=True):
    """Graph replay, or CUDA events when the call does not capture."""
    if graph:
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    call()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            samples = time_events(g.replay, warmup, iters)
            del g
            return "graph", samples, None
        except Exception as exc:  # noqa: BLE001
            torch.cuda.synchronize()
            why = repr(exc)[:300]
            return "events", time_events(call, warmup, iters), why
    return "events", time_events(call, warmup, iters), None


# --------------------------------------------------------------------- wires

def encode_key(label, rows, cols, q256):
    return f"{label} {rows}x{cols} q{q256}"


def _encode(label, rows, cols, q256, seed, dev, **kw):
    """One E2M1x2 encode, its seconds and allocator peak recorded in ENCODES."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import encode_linear
    from tessera.manifest import ScalePlaneKind

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    blob = encode_linear(gaussian(rows, cols, seed, dev), grid=tuple_grid(E2M1_GRID, 2), q256=q256,
                         scale_plane=ScalePlaneKind.LUT, **kw).blob
    torch.cuda.synchronize()
    ENCODES[encode_key(label, rows, cols, q256)] = {
        "secs": round(time.time() - t0, 3), "peak_bytes_above_base": torch.cuda.max_memory_allocated() - base,
        "weights": rows * cols}
    return blob


@functools.lru_cache(maxsize=None)   # bytes on the host: both passes see one encode
def window_wire(rows, cols, q256, seed, dev):
    from tessera.manifest import BodyKind

    return _encode("window", rows, cols, q256, seed, dev, body=BodyKind.WINDOW, window_bits=L)


@functools.lru_cache(maxsize=None)
def tcq_wire(rows, cols, q256, seed, dev):
    from tessera.manifest import BodyKind

    return _encode("tcq", rows, cols, q256, seed, dev, body=BodyKind.TCQ, span=2)


def window_unit(blob, dev):
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact

    return prepare_window_lut_compact(parse_compact_wire(blob, device=str(dev), name="w"), device=str(dev))


def a4_unit(blob, dev):
    from tessera.compact_prep import parse_compact_wire
    from tessera.serving.native_a4 import prepare_a4_unit

    return prepare_a4_unit(parse_compact_wire(blob, device=str(dev), name="w"))


def e2m1_bundle(unit, part, experts):
    from tessera.native_window_moe import WindowUnitAxis
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    axis = WindowUnitAxis(experts, [part], family="e2m1")
    for e in range(experts):
        axis.put(part, e, unit)
    s = axis.finish()[part]
    return prepare_grouped_window_gemm_from_soa(
        words_all=s["words"], table_all=s["table"], codes_all=s["codes"], native_all=s["native"],
        scale_all=s["scale"], runs_all=s["runs"], init_all=s["init"], has_init=s["has_init"],
        word_off=s["word_off"], tile_words=s["tile_words"], total_words=s["total_words"],
        run_off=s["run_off"], perm_all=s["perm"], rows=s["rows"], cols=s["cols"],
        experts=experts, window_bits=s["window_bits"], family="e2m1",
        scale_plane_all=s["scale_plane"], scale_lut_all=s["scale_lut"], global_all=s["global_scale"])


# --------------------------------------------------------------------- routed legs

class Routed:
    """Builders for the routed legs; each returns (head, make(m) -> (meta, call, keep))."""

    def __init__(self, dev):
        self.dev = dev
        self.gs13 = torch.tensor(448.0 * 6 / 3.0, device=dev)    # amax 3 over the bf16 test activations
        self.gs2 = torch.tensor(448.0 * 6 / 1.0, device=dev)

    def inputs(self, m):
        g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(f"x:{m}".encode()))
        x = (torch.randn(m, HIDDEN, generator=g, device=self.dev) * 0.5).to(torch.bfloat16)
        ids, w = balanced_routing(m, self.dev)
        return x, ids, w

    def e2m1(self, q256):
        from tessera.routed_fused_e2m1 import FusedRoutedE2M1MoE

        dev = self.dev
        t0 = time.time()
        gate = e2m1_bundle(window_unit(window_wire(INTER, HIDDEN, q256, 11 + q256, dev), dev), "gate_proj", EXPERTS)
        up = e2m1_bundle(window_unit(window_wire(INTER, HIDDEN, q256, 12 + q256, dev), dev), "up_proj", EXPERTS)
        down = e2m1_bundle(window_unit(window_wire(HIDDEN, INTER, q256, 13 + q256, dev), dev), "down_proj", EXPERTS)
        moe = FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=self.gs13, gs2=self.gs2)
        head = {"leg": "e2m1", "q256": q256, "window_bits": L, "build_s": time.time() - t0,
                "wire": "window body over LUT16 (E2M1x2)", "bits_per_weight": (q256 + 64) / 256}

        def make(m):
            x, ids, w = self.inputs(m)
            holder = {}

            def call():
                holder["out"] = moe(x, ids, w, swiglu_limit=SWIGLU_LIMIT)
            return {"routes": m * TOP_K}, call, (x, ids, w, holder)
        return head, make, (moe, gate, up, down)

    def a4(self):
        from vllm.model_executor.layers.fused_moe.activation import (ApplyMoEActivationConfig, MoEActivation,
                                                                     apply_moe_activation)
        from tessera.kernel_a4 import A4UnitStack
        from tessera.serving.native_a4 import a4_grouped_apply, stack_epilogues

        dev = self.dev
        t0 = time.time()
        stacks = {}
        for role, (rows, cols, seed) in {"gate": (INTER, HIDDEN, 21), "up": (INTER, HIDDEN, 22),
                                         "down": (HIDDEN, INTER, 23)}.items():
            unit = a4_unit(tcq_wire(rows, cols, A4_Q256, seed, dev), dev)
            stacks[role] = A4UnitStack.stack([unit] * EXPERTS)
        ep = {"gate": stack_epilogues(stacks["gate"], self.gs13), "up": stack_epilogues(stacks["up"], self.gs13),
              "down": stack_epilogues(stacks["down"], self.gs2)}
        config = ApplyMoEActivationConfig(clamp_limit=SWIGLU_LIMIT, alpha=1.0, beta=0.0,
                                          activation_situ_beta=None, activation_situ_linear_beta=None)
        head = {"leg": "a4", "q256": A4_Q256, "build_s": time.time() - t0,
                "wire": "span-2 TCQ over LUT16 (E2M1x2)", "bits_per_weight": (A4_Q256 + 128) / 256}
        gs13, gs2 = self.gs13, self.gs2

        def make(m):
            x, ids, w = self.inputs(m)
            holder = {}

            def call():
                flat_ids = ids.to(torch.int64).reshape(-1)
                flat_tokens = torch.arange(m, device=dev, dtype=torch.int64).repeat_interleave(TOP_K)
                flat_w = w.reshape(-1).to(torch.float32)
                order = torch.argsort(flat_ids, stable=True)
                counts = torch.zeros(EXPERTS, dtype=torch.int32, device=dev)
                counts.scatter_add_(0, flat_ids[order], torch.ones_like(flat_ids, dtype=torch.int32))
                offsets = torch.zeros(EXPERTS + 1, dtype=torch.int32, device=dev)
                offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
                route_tokens = flat_tokens[order].to(torch.int32)
                route_w = flat_w[order]
                routes = int(flat_ids.numel())
                g = a4_grouped_apply(x, stacks["gate"], gs13, expert_offsets=offsets, route_ids=route_tokens,
                                     num_routes=routes, epilogues=ep["gate"])
                u = a4_grouped_apply(x, stacks["up"], gs13, expert_offsets=offsets, route_ids=route_tokens,
                                     num_routes=routes, epilogues=ep["up"])
                gu = torch.cat([g, u], dim=-1)
                act = torch.empty((gu.shape[0], gu.shape[1] // 2), dtype=gu.dtype, device=dev)
                apply_moe_activation(MoEActivation.SILU, act, gu, activation_config=config)
                ident = torch.arange(routes, dtype=torch.int32, device=dev)
                d = a4_grouped_apply(act, stacks["down"], gs2, expert_offsets=offsets, route_ids=ident,
                                     num_routes=routes, epilogues=ep["down"])
                out = torch.zeros((m, HIDDEN), dtype=torch.float32, device=dev)
                out.index_add_(0, route_tokens.to(torch.int64), d * route_w[:, None])
                holder["out"] = out.to(x.dtype)
            return {"routes": m * TOP_K}, call, (x, ids, w, holder)
        return head, make, stacks

    def vllm(self):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig, FusedMoEParallelConfig,
                                                                 RoutingMethodType)
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            convert_to_nvfp4_moe_kernel_format, is_global_sf_supported_for_nvfp4_backend,
            make_nvfp4_moe_kernel, make_nvfp4_moe_quant_config, select_nvfp4_moe_backend)
        from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic, kNvfp4Static
        from vllm.v1.worker.workspace import init_workspace_manager
        from vllm import _custom_ops as ops

        dev, E = self.dev, EXPERTS
        t0 = time.time()
        try:
            init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
        except Exception:  # noqa: BLE001 -- already initialised
            pass
        parallel = FusedMoEParallelConfig(
            tp_size=1, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0,
            ep_rank=0, sp_size=1, use_ep=False, all2all_backend="allgather_reducescatter",
            enable_eplb=False)
        cfg = FusedMoEConfig(
            num_experts=E, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=INTER,
            num_local_experts=E, num_logical_experts=E, activation=MoEActivation.SILU,
            device=torch.device("cuda"), routing_method=RoutingMethodType.DeepSeekV3,
            moe_parallel_config=parallel, in_dtype=torch.bfloat16, swiglu_limit=SWIGLU_LIMIT)
        backend, experts_cls = select_nvfp4_moe_backend(config=cfg, weight_key=kNvfp4Static,
                                                        activation_key=kNvfp4Dynamic)
        layer = torch.nn.Module()
        layer.moe_config = cfg
        layer.expert_map = None
        layer.apply_router_weight_on_input = False
        layer.activation = MoEActivation.SILU
        layer.global_num_experts = E
        layer.num_experts = E
        layer.swiglu_limit = SWIGLU_LIMIT
        layer._expert_routing_tables = lambda: None

        def param(name, t):
            if hasattr(layer, name):
                delattr(layer, name)
            layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))

        # Weights: one real NVFP4 quantisation per projection (vLLM's own
        # quantiser, so the block scales are the kernel's layout), copied
        # into every expert.
        def quant(rows, cols, seed):
            w = gaussian(rows, cols, seed, dev).to(torch.bfloat16)
            gsw = (448.0 * 6 / w.abs().max().float()).reshape(1)
            q, sf = ops.scaled_fp4_quant(w, gsw, is_sf_swizzled_layout=False)
            return q, sf.view(torch.float8_e4m3fn).reshape(rows, cols // 16), (1.0 / gsw).reshape(())

        qg, sg, g2g = quant(INTER, HIDDEN, 31)
        qu, su, _ = quant(INTER, HIDDEN, 32)
        qd, sd, g2d = quant(HIDDEN, INTER, 33)
        w13 = torch.cat([qg, qu], 0).unsqueeze(0).expand(E, -1, -1).contiguous()
        s13 = torch.cat([sg, su], 0).unsqueeze(0).expand(E, -1, -1).contiguous()
        w2 = qd.unsqueeze(0).expand(E, -1, -1).contiguous()
        s2 = sd.unsqueeze(0).expand(E, -1, -1).contiguous()
        w13_s2 = torch.full((E,), float(g2g), device=dev)
        w2_s2 = torch.full((E,), float(g2d), device=dev)
        # modelopt's input_scale is amax / (448 * 6)
        a13 = torch.full((E, 2), 1.0 / float(self.gs13), device=dev)
        a2 = torch.full((E,), 1.0 / float(self.gs2), device=dev)
        for name, t in (("w13_weight", w13), ("w13_weight_scale", s13), ("w13_weight_scale_2", w13_s2),
                        ("w13_input_scale", a13), ("w2_weight", w2), ("w2_weight_scale", s2),
                        ("w2_weight_scale_2", w2_s2), ("w2_input_scale", a2)):
            param(name, t)
        (w13c, s13c, w13_s2c, a13c, w2c, s2c, w2_s2c, a2c) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=backend, layer=layer, w13=layer.w13_weight, w13_scale=layer.w13_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2, a13_scale=layer.w13_input_scale, w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale, w2_scale_2=layer.w2_weight_scale_2, a2_scale=layer.w2_input_scale,
            is_act_and_mul=True)
        for name, t in (("w13_weight", w13c), ("w13_weight_scale", s13c), ("w13_weight_scale_2", w13_s2c),
                        ("w13_input_scale", a13c), ("w2_weight", w2c), ("w2_weight_scale", s2c),
                        ("w2_weight_scale_2", w2_s2c), ("w2_input_scale", a2c)):
            param(name, t)
        quant_cfg = make_nvfp4_moe_quant_config(
            backend=backend, w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2, w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale, a2_scale=layer.w2_input_scale, swiglu_limit=SWIGLU_LIMIT,
            layer=layer)
        kernel = make_nvfp4_moe_kernel(moe_quant_config=quant_cfg, moe_config=cfg, experts_cls=experts_cls,
                                       backend=backend, routing_tables=None)
        pwal = getattr(getattr(kernel, "fused_experts", None), "process_weights_after_loading", None)
        if pwal is not None:
            pwal(layer)
        head = {"leg": "vllm", "q256": 1152, "bits_per_weight": 4.5, "build_s": time.time() - t0,
                "backend": str(getattr(backend, "name", backend)),
                "experts_cls": getattr(experts_cls, "__name__", str(experts_cls)),
                "global_sf": bool(is_global_sf_supported_for_nvfp4_backend(backend)),
                "wire": "modelopt NVFP4 (e2m1 + ue4m3 group 16 + fp32 global)"}

        def make(m):
            x, ids, w = self.inputs(m)
            holder = {}

            def call():
                holder["out"] = kernel.apply(x, layer.w13_weight, layer.w2_weight, w, ids,
                                             activation=MoEActivation.SILU, global_num_experts=E,
                                             expert_map=None, apply_router_weight_on_input=False)
            return {"routes": m * TOP_K}, call, (x, ids, w, holder)
        return head, make, (layer, kernel)


# --------------------------------------------------------------------- dense legs

class Dense:
    def __init__(self, dev):
        self.dev = dev
        self.gs = torch.tensor(448.0 * 6 / 3.0, device=dev)

    def x(self, m, cols):
        g = torch.Generator(device=self.dev).manual_seed(zlib.crc32(f"xd:{m}:{cols}".encode()))
        return (torch.randn(m, cols, generator=g, device=self.dev) * 0.5).to(torch.bfloat16)

    def e2m1(self, shape, q256):
        from tessera import routed_fused_e2m1 as re2

        rows, cols = DENSE[shape]
        head = {"leg": "e2m1", "shape": shape, "rows": rows, "cols": cols, "q256": q256}
        if rows % re2.DENSE_ROWS:   # before the encode
            head["refused"] = f"{rows} rows; the dense launch needs a multiple of {re2.DENSE_ROWS}"
            return head, None, None
        unit = window_unit(window_wire(rows, cols, q256, zlib.crc32(f"{shape}:{q256}".encode()), self.dev),
                           self.dev)
        why = re2.dense_role_reason(unit)
        if why is not None:
            head["refused"] = why
            return head, None, None
        role = re2.prepare_dense_role(unit, self.gs)
        head["split_cap"] = re2.dense_split_max(cols)
        head["encode"] = ENCODES.get(encode_key("window", rows, cols, q256))

        def make(m, split):
            x = self.x(m, cols)
            out = torch.empty((m, rows), dtype=torch.bfloat16, device=self.dev)

            def call():
                re2.dense_forward(role, x, k_split=split, out=out)
            return {"k_split": split}, call, (x, out)
        return head, make, role

    def a4(self, shape):
        from tessera.serving.native_a4 import a4_dense_apply

        rows, cols = DENSE[shape]
        head = {"leg": "a4", "shape": shape, "rows": rows, "cols": cols, "q256": A4_Q256}
        try:
            unit = a4_unit(tcq_wire(rows, cols, A4_Q256, zlib.crc32(f"a4:{shape}".encode()), self.dev), self.dev)
        except Exception as exc:  # noqa: BLE001
            head["refused"] = repr(exc)[:300]
            return head, None, None
        epi = unit.epilogue_for(self.gs)
        head["encode"] = ENCODES.get(encode_key("tcq", rows, cols, A4_Q256))

        def make(m, _split):
            x = self.x(m, cols)
            holder = {}

            def call():
                holder["out"] = a4_dense_apply(x, unit, self.gs, epilogue=epi)
            return {}, call, (x, holder)
        return head, make, unit

    def vllm(self, shape):
        from vllm import _custom_ops as ops

        rows, cols = DENSE[shape]
        head = {"leg": "vllm", "shape": shape, "rows": rows, "cols": cols, "q256": 1152,
                "kernel": "vllm._custom_ops.cutlass_scaled_fp4_mm"}
        w = gaussian(rows, cols, zlib.crc32(f"v:{shape}".encode()), self.dev).to(torch.bfloat16)
        gsw = (448.0 * 6 / w.abs().max().float()).reshape(1)
        wq, wsf = ops.scaled_fp4_quant(w, gsw)
        alpha = (1.0 / (gsw * self.gs)).reshape(1).float()
        del w

        def make(m, _split):
            x = self.x(m, cols)
            holder = {}

            def call():
                xq, xsf = ops.scaled_fp4_quant(x, self.gs.reshape(1))
                holder["out"] = ops.cutlass_scaled_fp4_mm(xq, wq, xsf, wsf, alpha, torch.bfloat16)
            return {}, call, (x, holder)
        return head, make, (wq, wsf)


# --------------------------------------------------------------------- driver

class Bench:
    def __init__(self, args, dev):
        self.args, self.dev = args, dev
        self.ms = [int(v) for v in args.ms.split(",")]
        self.power_ms = {int(v) for v in args.power_ms.split(",") if v}
        self.power = PowerSampler()
        self.groups = {}

    def cell_list(self):
        a = self.args
        legs = a.legs.split(",")
        rungs = [int(v) for v in a.rungs.split(",")]
        out = []
        if a.part in ("routed", "all"):
            for leg in legs:
                for q in (rungs if leg == "e2m1" else [None]):
                    out.append(("routed", leg, q, None))
        if a.part in ("dense", "all"):
            for shape in a.shapes.split(","):
                for leg in legs:
                    for q in (rungs if leg == "e2m1" else [None]):
                        out.append(("dense", leg, q, shape))
        return out

    def build(self, kind, leg, q, shape):
        if kind == "routed":
            r = Routed(self.dev)
            return r.e2m1(q) if leg == "e2m1" else (r.a4() if leg == "a4" else r.vllm())
        d = Dense(self.dev)
        return d.e2m1(shape, q) if leg == "e2m1" else (d.a4(shape) if leg == "a4" else d.vllm(shape))

    def variants(self, kind, head):
        if kind == "dense" and head.get("leg") == "e2m1":
            cap = head["split_cap"]
            splits = sorted({1, 2, 4, 8, cap} & set(range(1, cap + 1)))
            return [(m, s) for m in self.ms for s in splits]
        return [(m, None) for m in self.ms]

    def run(self, save):
        a = self.args
        cells = self.cell_list()
        for pas in ("F", "R"):
            for kind, leg, q, shape in (cells if pas == "F" else list(reversed(cells))):
                gkey = f"{kind}:{leg}:{q}:{shape}"
                try:
                    head, make, keep = self.build(kind, leg, q, shape)
                except Exception as exc:  # noqa: BLE001
                    self.groups.setdefault(gkey, {"kind": kind, "leg": leg, "cells": {}})["error"] = repr(exc)[:600]
                    emit({"group": gkey, "pass": pas, "error": repr(exc)[:300]})
                    torch.cuda.empty_cache()
                    continue
                rec = self.groups.setdefault(gkey, dict(head, kind=kind, cells={}))
                if make is None:
                    if pas == "F":
                        emit({"group": gkey, "refused": head.get("refused")})
                    continue
                vs = self.variants(kind, head)
                for m, split in (vs if pas == "F" else list(reversed(vs))):
                    ckey = f"{m}" + (f":s{split}" if split is not None else "")
                    cell = rec["cells"].setdefault(ckey, {"M": m})
                    try:
                        meta, call, hold = make(m) if kind == "routed" else make(m, split)
                        how, samples, why = time_call(call, a.warmup, a.iters)
                        cell[pas] = {"median_ms": statistics.median(samples), "min_ms": min(samples),
                                     "timer": how, "unix": time.time()}
                        if why:
                            cell[pas]["graph_refused"] = why
                        if pas == "F":
                            cell.update(meta)
                            cell["profile"] = kernel_profile(call, reps=a.prof_reps)
                            if m in self.power_ms:
                                cell["power"] = self.power.sample_during(call, a.power_s)
                        else:
                            f = cell.get("F", {}).get("median_ms")
                            if f:
                                r = cell["R"]["median_ms"]
                                cell["ms"] = 0.5 * (f + r)
                                cell["spread"] = abs(f - r) / cell["ms"]
                        del call, hold
                    except Exception as exc:  # noqa: BLE001
                        cell[pas] = {"error": repr(exc)[:600]}
                        emit({"group": gkey, "M": ckey, "pass": pas, "error": repr(exc)[:300]})
                        torch.cuda.synchronize()
                        continue
                    line = {"g": gkey, "M": ckey, "pass": pas, "ms": round(cell[pas]["median_ms"], 4),
                            "timer": cell[pas]["timer"]}
                    if "ms" in cell and pas == "R":
                        line.update({"mean_ms": round(cell["ms"], 4), "spread": round(cell["spread"], 4)})
                    emit(line)
                del keep
                torch.cuda.empty_cache()
                save()


def summary(groups):
    """``{kind: {M: {label: ms}}}``: dense e2m1 at its fastest split."""
    out = {}
    for g in groups.values():
        kind = g.get("kind")
        label = g.get("leg", "?") + (f" q{g.get('q256')}" if g.get("leg") == "e2m1" else "")
        if kind == "dense":
            label = f"{g.get('shape')} {label}"
        for c in g.get("cells", {}).values():
            if "ms" not in c:
                continue
            row = out.setdefault(kind, {}).setdefault(str(c["M"]), {})
            if label not in row or c["ms"] < row[label]["ms"]:
                row[label] = {"ms": round(c["ms"], 4), **({"k_split": c["k_split"]} if "k_split" in c else {})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("routed", "dense", "all"), required=True)
    ap.add_argument("--ms", default="1,64,512,2048,8192")
    ap.add_argument("--rungs", default="448,704,960")
    ap.add_argument("--legs", default="e2m1,a4,vllm")
    ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prof-reps", type=int, default=3)
    ap.add_argument("--power-ms", default="1,512,8192")
    ap.add_argument("--power-s", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    b = Bench(args, dev)
    meta = {"device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(dev).multi_processor_count,
            "experts": EXPERTS, "hidden": HIDDEN, "inter": INTER, "top_k": TOP_K, "swiglu_limit": SWIGLU_LIMIT,
            "window_bits": L, "a4_q256": A4_Q256, "args": vars(args), "tessera_head": os.environ.get("TESSERA_HEAD"),
            "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "power_source": b.power.source, "envelope_w": ENVELOPE_W,
            "start_unix": time.time(), "torch": torch.__version__,
            "statistic": "mean of the forward and reverse passes' medians; spread = |F - R| / mean"}
    path = os.path.join(args.out, f"bench_e2m1_{args.part}.json")

    def save():
        json.dump({"meta": meta, "summary": summary(b.groups), "groups": b.groups, "encodes": ENCODES},
                  open(path, "w"), indent=1)
    b.run(save)
    meta["end_unix"] = time.time()
    save()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
