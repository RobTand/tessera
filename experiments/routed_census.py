"""Census for the fused routed kernel (tessera#640): wire facts and ceilings.

Runs inside the pinned serving image on a GB10.  Three parts, one JSON:

* ``wires``: the parameters the new kernel must serve, read off the real
  GLM wires -- window bits, per-run rates, column permutation, tile words,
  start-state history per TP rank -- for the oracle's TP1 cache and for the
  TP2 rank-0/rank-1 cuts of the exported arms (the PACT shape table).
* ``micro``: the instruction-level ceilings the design is priced against --
  ``mma.sync`` rate for the exact instruction each family will issue, random
  shared-memory 16-bit lookup throughput, and an exactness check of the
  ``e4m3 -> f16`` conversion the E4M3 lane will use for its A operand.
* ``ceilings``: DRAM bandwidth and the dense MMA rate torch reaches on this
  image (bf16 ``mm``, fp8 ``_scaled_mm``, nvfp4 cutlass), with the SM clock
  and power sampled during each loop.

Nothing here is a claim about the kernel; it is the denominator the report
divides by, measured on the box that will run the kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPORTS = {
    "t16": "/mnt/shared/tessera-runs/moe/glm53-body-mtp-bf16-r1024-20260926/exported",
    "t8": "/mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927/a8/body-mtp-v39/exported-r2",
    "t4": "/mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927/a4/body-mtp-v39/exported-r2",
}
LAYER_PREFIX = "model.language_model.layers."
TP2_UNITS = [("t16", LAYER_PREFIX + "10.mlp.experts"),
             ("t8", LAYER_PREFIX + "10.mlp.experts"),
             ("t4", LAYER_PREFIX + "10.mlp.experts"),
             ("t16", LAYER_PREFIX + "43.mlp.experts")]   # E4M3 R896: mixed-rate runs
EXPERTS = 288


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tensor_facts(t, values=8):
    if not isinstance(t, torch.Tensor):
        return repr(t)
    out = {"shape": list(t.shape), "dtype": str(t.dtype), "numel": int(t.numel())}
    if t.numel() and t.numel() <= 64:
        out["values"] = t.detach().cpu().reshape(-1).tolist()
    elif t.numel():
        flat = t.detach().cpu().reshape(-1)
        out["head"] = flat[:values].tolist()
        if not t.is_floating_point():
            out["nonzero"] = int((flat != 0).sum())
            out["min"] = int(flat.min())
            out["max"] = int(flat.max())
    return out


# ---------------------------------------------------------------------------
# wires
# ---------------------------------------------------------------------------

def window_bundle_facts(bundle, *, fused_owner=None, projection=None):
    """Read the compact planes or the fused owner that replaced those planes."""
    if fused_owner is not None:
        runs = getattr(fused_owner, "runs_" + projection).cpu()
        groups = {}
        for expert, row in enumerate(runs.tolist()):
            groups.setdefault(json.dumps(row), []).append(expert)
        down = projection == "down"
        return {
            "representation": "fused_native",
            "family": bundle.family, "quantizer": bundle.quantizer,
            "rows": int(bundle.rows), "cols": int(bundle.cols),
            "experts": int(bundle.experts), "window_bits": int(bundle.window_bits),
            "library": fused_owner.library,
            "words": tensor_facts(getattr(fused_owner, "words_" + projection)),
            "table": tensor_facts(getattr(fused_owner, "table_" + projection)),
            "scale": tensor_facts(bundle.scale_all),
            "initial_states": tensor_facts(bundle.init_all),
            "has_init": tensor_facts(bundle.has_init),
            "run_pairs": tensor_facts(runs),
            "block_descriptors": tensor_facts(getattr(fused_owner, "bdesc_" + projection)),
            "tile_words": int(fused_owner.tile_words_down if down else fused_owner.tile_words_gate_up),
            "slot_words": int(fused_owner.slot_words_down if down else fused_owner.slot_words_gate_up),
            "distinct_run_pairs": {key: {"experts": value} for key, value in groups.items()},
            "n_distinct_run_pairs": len(groups),
        }
    E = int(bundle.experts)
    cols = int(bundle.cols)
    run_off = bundle.run_off.cpu()
    runs = bundle.runs_all.cpu().reshape(-1, 4)
    per_expert_runs = {}
    for e in range(E):
        r = runs[int(run_off[e]):int(run_off[e + 1])].tolist()
        key = json.dumps(r)
        per_expert_runs.setdefault(key, []).append(e)
    perm = bundle.perm_all.cpu()
    ident = torch.arange(cols, dtype=perm.dtype)
    perm_identity = [bool((perm[e] == ident).all()) for e in range(E)]
    init = bundle.init_all.cpu()
    return {
        "family": bundle.family, "quantizer": bundle.quantizer,
        "rows": int(bundle.rows), "cols": cols, "experts": E,
        "window_bits": int(bundle.window_bits),
        "block": [int(bundle.block_m), int(bundle.block_n), int(bundle.block_k)],
        "words_all": tensor_facts(bundle.words_all),
        "table_all": tensor_facts(bundle.table_all), "codes_all": tensor_facts(bundle.codes_all),
        "native_all": tensor_facts(bundle.native_all), "scale_all": tensor_facts(bundle.scale_all),
        "tile_words": tensor_facts(bundle.tile_words), "total_words": tensor_facts(bundle.total_words),
        "word_off": tensor_facts(bundle.word_off),
        "has_init": tensor_facts(bundle.has_init),
        "init_nonzero_per_expert": [int((init[e] != 0).sum()) for e in range(E)],
        "perm_identity_per_expert": perm_identity,
        "distinct_run_tables": {k: {"experts": v if len(v) <= 8 else f"{len(v)} experts"}
                                for k, v in per_expert_runs.items()},
        "n_distinct_run_tables": len(per_expert_runs),
    }


def a4_stack_facts(stack):
    f = {name: tensor_facts(getattr(stack, name)) for name in
         ("select", "label", "point", "nibbles", "lut_bytes", "label_lut", "code_nibbles", "globals")}
    f.update({"rows": stack.rows, "cols": stack.cols, "rate": stack.rate, "arity": stack.arity,
              "memory": stack.memory, "half": stack.half, "experts": stack.experts})
    rows = stack.rows
    steps, pairs = rows // 2, rows // 4
    field = stack.rate - 1
    f["derived"] = {"steps": steps, "pairs": pairs, "field_bits": field,
                    "select_bits_per_column": pairs + 8,
                    "label_bits_per_column": pairs * 2,
                    "point_bits_per_column": steps * field,
                    "label_lut_entries": 1 << (stack.memory + 1),
                    "code_table_entries": 4 * (1 << field)}
    return f


def tp1_wire_facts(args):
    import routed_pair_oracle as rpo

    out = {}
    experts = [0, 72, 144, 216]
    ns = argparse.Namespace(wire_root=rpo.WIRE_ROOT, scales=rpo.SCALES, layer=3, clamp=10.0)
    try:
        from vllm.v1.worker.workspace import init_workspace_manager
        init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    except Exception as exc:  # noqa: BLE001
        out["workspace_init"] = repr(exc)
    prefix = f"model.language_model.layers.{ns.layer}.mlp.experts"
    for fkey, fam in rpo.FAMILIES.items():
        entry = {}
        out[fkey] = entry
        try:
            blobs, files = rpo.load_wires(ns, fam, experts)
            entry["wire_bytes"] = [f["bytes"] for f in files][:3]
            scales = None
            if fam["family"] == "TESSERA_NVFP4":
                scales, _ = rpo.load_input_scales(ns, experts)
            scheme, facts = rpo.scheme_for(fam, blobs, len(experts))
            entry["scheme"] = scheme
            cfg = rpo.moe_config(len(experts), ns.clamp)
            layer, method, info = rpo.build_after(fam, scheme, blobs, scales, len(experts), cfg,
                                                  ns.clamp, prefix)
            entry["build"] = info
            if fam["family"] == "TESSERA_NVFP4":
                for name in ("gate", "up", "down"):
                    entry[name] = a4_stack_facts(getattr(layer, f"tessera_a4_{name}_stack"))
                entry["gs13"] = float(layer.tessera_a4_gs13)
                entry["gs2"] = float(layer.tessera_a4_gs2)
            else:
                from tessera.routed_fused import FusedRoutedWindowMoE

                nat = method._native
                owner = nat if isinstance(nat, FusedRoutedWindowMoE) else None
                for name in ("gate", "up", "down"):
                    entry[name] = window_bundle_facts(
                        getattr(nat, name), fused_owner=owner, projection=name)
            del layer, method
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()[-3000:]
    return out


class Store:
    def __init__(self, root):
        from safetensors import safe_open

        self.root = root
        self._safe_open = safe_open
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.config = json.load(open(os.path.join(root, "config.json")))
        self._open = {}
        self.schemes = {}
        for g in self.config["quantization_config"]["config_groups"].values():
            for t in g["targets"]:
                self.schemes[t] = g["scheme"]

    def get(self, name):
        shard = self.index[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(os.path.join(self.root, shard), "pt", device="cpu")
        return self._open[shard].get_tensor(name)


def repacked_facts(unit):
    """A compact ``WindowGemvUnit`` for one rank: rep, history, tables."""
    rep = unit.rep
    facts = {"unit_fields": sorted(vars(unit).keys()), "rep_type": type(rep).__name__,
             "rep_fields": sorted(vars(rep).keys())}
    for name, value in vars(rep).items():
        if isinstance(value, torch.Tensor):
            facts[f"rep.{name}"] = tensor_facts(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            facts[f"rep.{name}"] = value
    runs = getattr(rep, "runs", None)
    if isinstance(runs, torch.Tensor):
        facts["runs"] = runs.cpu().reshape(-1, 4).tolist()
    perm = getattr(rep, "perm", None)
    if isinstance(perm, torch.Tensor):
        facts["perm_identity"] = bool((perm.cpu() == torch.arange(perm.numel(), dtype=perm.dtype)).all())
    facts["window_bits"] = int(unit.window_bits)
    facts["family"] = unit.family
    facts["row_offset"] = int(unit.row_offset)
    facts["table"] = tensor_facts(unit.table)
    facts["scale"] = tensor_facts(unit.scale)
    if unit.initial_state is not None:
        facts["initial_state"] = tensor_facts(unit.initial_state)
    else:
        facts["initial_state"] = None
    if unit.codes_of_state is not None:
        facts["codes_of_state"] = tensor_facts(unit.codes_of_state)
        facts["native"] = tensor_facts(unit.native)
    return facts


def a4_unit_facts(unit):
    f = {name: tensor_facts(getattr(unit, name)) for name in
         ("select", "label", "point", "nibbles", "lut_bytes", "label_lut", "subset_nibbles",
          "code_nibbles")}
    f.update({"rows": unit.rows, "cols": unit.cols, "rate": unit.rate, "arity": unit.arity,
              "memory": unit.memory, "half": unit.half, "global_scale": unit.global_scale})
    return f


def tp2_wire_facts(args):
    from tessera.serving.scheme import validate_tessera_moe_scheme

    out = {}
    dev = torch.device("cuda")
    stores = {}
    for arm, module in TP2_UNITS:
        key = f"{arm}:{module}"
        entry = {}
        out[key] = entry
        try:
            if arm not in stores:
                stores[arm] = Store(EXPORTS[arm])
            store = stores[arm]
            scheme = store.schemes[module]
            entry["scheme"] = {k: scheme[k] for k in scheme if k != "groups"}
            entry["groups"] = scheme["groups"]
            declared = validate_tessera_moe_scheme(scheme, module)
            wires = {("w13", 0): store.get(f"{module}.0.gate_proj.wire"),
                     ("w13", 1): store.get(f"{module}.0.up_proj.wire"),
                     ("w2", 0): store.get(f"{module}.0.down_proj.wire")}
            entry["wire_numel"] = {f"{g}[{i}]": int(w.numel()) for (g, i), w in wires.items()}
            for tp_rank in (0, 1):
                rank = {}
                entry[f"rank{tp_rank}"] = rank
                if scheme["family"] == "TESSERA_NVFP4":
                    from tessera.serving.nvfp4_moe_route import _ExpertIntake
                    from tessera.serving.native_a4 import A4ExpertAxis
                    from tessera.serving.scheme import MOE_GROUPS

                    intake = _ExpertIntake(declared, module, tp_rank, 2)
                    axes = {(g, role["roles"][0][0]): A4ExpertAxis(EXPERTS)
                            for g in MOE_GROUPS for role in intake.roles[g]}
                    for (g, i), w in wires.items():
                        ready = intake.take(g, i, 0, w.contiguous().numpy().tobytes(), dev, axes=axes)
                        if ready is None:
                            continue
                        if ready[0] == "direct":
                            rank[f"{g}.direct"] = repr(ready[1:])[:400]
                        else:
                            units, sh = ready
                            rank[f"{g}.shared_global"] = float(sh)
                            for name, unit in units:
                                rank[f"{g}.{name}"] = a4_unit_facts(unit)
                    s = store.get(f"{module}.0.gate_proj.input_global_scale")
                    rank["input_global_scale.gate"] = float(s.reshape(-1)[0])
                else:
                    from tessera.serving.moe_route import (_RankLocalPackedIntake,
                                                           _compact_expert_units)

                    intake = _RankLocalPackedIntake(declared, module, dev, tp_rank, 2)
                    rank["compact"] = bool(intake.compact)
                    fam = "value" if scheme["family"] == "TESSERA_BF16" else "e4m3"
                    scratch = {}
                    for (g, i), w in wires.items():
                        name, unit = _compact_expert_units(
                            w.contiguous().numpy().tobytes(), intake.roles[g][i], intake.plans[g],
                            f"{module} {g} expert 0", device=dev, family=fam, scratch=scratch)
                        rank[f"{g}.{name}"] = repacked_facts(unit)
                        rank[f"{g}.plan"] = repr(intake.plans[g])[:600]
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()[-3000:]
    return out


# ---------------------------------------------------------------------------
# micro: the instruction-level ceilings
# ---------------------------------------------------------------------------

MICRO_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <functional>
#include <cstdint>

// FAM: 0 bf16 k16, 1 e4m3 k32 (kind::f8f6f4), 2 e2m1 k64 block-scaled, 3 f16 k16
template<int FAM>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                   const uint32_t (&b)[2], uint32_t sa, uint32_t sb) {
    if constexpr (FAM == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (FAM == 1) {
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if constexpr (FAM == 2) {
        const uint16_t selector = 0;
        asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                     "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3},"
                     "{%10}, {%12,%12}, {%11}, {%12,%12};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                       "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "h"(selector));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

template<int FAM>
__global__ void mma_rate_kernel(int iters, uint32_t av, uint32_t bv, float* out) {
    uint32_t a[4] = {av, av, av, av};
    uint32_t b[2] = {bv, bv};
    float d[8][4];
    #pragma unroll
    for (int c = 0; c < 8; ++c) { d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0.f; }
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) mma<FAM>(d[c], a, b, 0x38383838u, 0x38383838u);
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    if (s == -1.0f) out[0] = s;   // keeps the chains live; never true for these inputs
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = s;
}

__global__ void lds_rate_kernel(int iters, uint32_t* out) {
    __shared__ uint16_t table[16384];
    for (int i = threadIdx.x; i < 16384; i += blockDim.x) table[i] = (uint16_t)((i * 2654435761u) >> 16);
    __syncthreads();
    uint32_t s = threadIdx.x * 2654435761u + blockIdx.x * 40503u + 1u, acc = 0;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            acc += table[s & 16383];
        }
    }
    if (acc == 0xFFFFFFFFu) out[0] = acc;
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = acc;
}

// The same lookups, 8 consecutive per lane from one starting index: the
// access pattern of one lane walking one column's decoded states is random
// across lanes but each lane's own stream is arbitrary, so this is the same
// test with the xorshift replaced by a cheap increment (ALU-light bound).
__global__ void lds_rate_light_kernel(int iters, uint32_t* out) {
    __shared__ uint16_t table[16384];
    for (int i = threadIdx.x; i < 16384; i += blockDim.x) table[i] = (uint16_t)((i * 2654435761u) >> 16);
    __syncthreads();
    uint32_t s = threadIdx.x * 2654435761u + blockIdx.x * 40503u + 1u, acc = 0;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s = s * 1664525u + 1013904223u;
            acc += table[(s >> 14) & 16383];
        }
    }
    if (acc == 0xFFFFFFFFu) out[0] = acc;
    if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = acc;
}

__global__ void cvt_e4m3_f16_kernel(uint16_t* out) {
    int i = threadIdx.x;   // 0..255: byte value in the LOW half, zero in the high half
    if (i < 256) {
        uint16_t in = (uint16_t)i;   // two e4m3 bytes: the value in the low byte
        uint32_t r;
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(in));
        out[i] = (uint16_t)(r & 0xFFFFu);
        out[256 + i] = (uint16_t)(r >> 16);
    }
}

static double time_launch(std::function<void()> fn, int reps) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    fn();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) fn();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)ms / reps;
}

torch::Tensor mma_rate(int fam, int blocks, int threads, int iters, int reps) {
    auto out = torch::zeros({2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    float* p = out.data_ptr<float>();
    uint32_t av = 0x3F803F80u, bv = 0x3F803F80u;           // bf16 1.0 pairs
    if (fam == 1) { av = 0x38383838u; bv = 0x38383838u; }  // e4m3 1.0
    if (fam == 2) { av = 0x22222222u; bv = 0x22222222u; }  // e2m1 1.0 nibbles
    if (fam == 3) { av = 0x3C003C00u; bv = 0x3C003C00u; }  // f16 1.0 pairs
    auto launch = [&]() {
        switch (fam) {
            case 0: mma_rate_kernel<0><<<blocks, threads>>>(iters, av, bv, p); break;
            case 1: mma_rate_kernel<1><<<blocks, threads>>>(iters, av, bv, p); break;
            case 2: mma_rate_kernel<2><<<blocks, threads>>>(iters, av, bv, p); break;
            default: mma_rate_kernel<3><<<blocks, threads>>>(iters, av, bv, p); break;
        }
    };
    double ms = time_launch(launch, reps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto res = torch::zeros({2}, torch::dtype(torch::kFloat64));
    res[0] = ms;
    res[1] = out[1].item<float>();
    return res;
}

torch::Tensor lds_rate(int variant, int blocks, int threads, int iters, int reps) {
    auto out = torch::zeros({2}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    uint32_t* p = reinterpret_cast<uint32_t*>(out.data_ptr<int32_t>());
    auto launch = [&]() {
        if (variant == 0) lds_rate_kernel<<<blocks, threads>>>(iters, p);
        else lds_rate_light_kernel<<<blocks, threads>>>(iters, p);
    };
    double ms = time_launch(launch, reps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto res = torch::zeros({1}, torch::dtype(torch::kFloat64));
    res[0] = ms;
    return res;
}

torch::Tensor cvt_e4m3_f16() {
    auto out = torch::zeros({512}, torch::dtype(torch::kInt16).device(torch::kCUDA));
    cvt_e4m3_f16_kernel<<<1, 256>>>(reinterpret_cast<uint16_t*>(out.data_ptr<int16_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_rate", &mma_rate);
    m.def("lds_rate", &lds_rate);
    m.def("cvt_e4m3_f16", &cvt_e4m3_f16);
}
"""


def micro(args, sampler):
    from torch.utils.cpp_extension import load_inline

    out = {}
    build = Path(args.out) / "micro-build"
    build.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ext = load_inline(
        name="tessera_routed_census_micro", cpp_sources="", cuda_sources=MICRO_SRC,
        functions=None, extra_cuda_cflags=["-O3", "-std=c++17",
                                           "-gencode=arch=compute_121a,code=sm_121a"],
        build_directory=str(build), verbose=False, is_python_module=True)
    out["build_seconds"] = round(time.time() - t0, 1)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    out["device"] = {"name": props.name, "sms": sms, "capability": [props.major, props.minor],
                     "smem_per_block_optin": getattr(props, "shared_memory_per_block_optin", None),
                     "smem_per_sm": getattr(props, "shared_memory_per_multiprocessor", None),
                     "regs_per_sm": getattr(props, "regs_per_multiprocessor", None),
                     "l2_bytes": getattr(props, "L2_cache_size", None),
                     "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", None)}

    # cvt exactness: every e4m3 byte through cvt.rn.f16x2.e4m3x2 equals torch's cast
    raw = ext.cvt_e4m3_f16().cpu()
    lo = raw[:256].view(torch.float16)
    hi = raw[256:].view(torch.float16)
    ref = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float16)
    lo_eq = (lo.view(torch.int16) == ref.view(torch.int16))
    nan_mask = torch.isnan(ref)
    out["cvt_e4m3_to_f16"] = {
        "bitwise_equal_non_nan": bool(lo_eq[~nan_mask].all()),
        "mismatching_bytes": [int(i) for i in torch.nonzero(~lo_eq & ~nan_mask).reshape(-1)][:16],
        "nan_bytes": [int(i) for i in torch.nonzero(nan_mask).reshape(-1)],
        "nan_bytes_convert_to_nan": bool(torch.isnan(lo[nan_mask]).all()) if nan_mask.any() else None,
        "high_half_zero_is_zero": bool((hi.view(torch.int16) == 0).all()),
    }

    flop_per_mma = {0: 16 * 8 * 16 * 2, 1: 16 * 8 * 32 * 2, 2: 16 * 8 * 64 * 2, 3: 16 * 8 * 16 * 2}
    names = {0: "bf16.m16n8k16", 1: "e4m3.m16n8k32.f8f6f4", 2: "e2m1.m16n8k64.mxf4nvf4", 3: "f16.m16n8k16"}
    out["mma_rate"] = {}
    for fam in (0, 3, 1, 2):
        for threads in (256, 512):
            blocks = sms * 4
            iters = 2048
            rec = {"blocks": blocks, "threads": threads, "iters": iters, "chains_per_warp": 8}
            try:
                sampler.mark()
                res = ext.mma_rate(fam, blocks, threads, iters, 5)
                ms = float(res[0])
                warps = blocks * threads // 32
                flops = warps * iters * 8 * flop_per_mma[fam]
                rec.update({"ms": ms, "tflops": flops / ms / 1e9,
                            "mma_per_sm_per_us": warps * iters * 8 / sms / (ms * 1e3),
                            "sampled": sampler.since_mark()})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)
            out["mma_rate"][f"{names[fam]}@{threads}"] = rec
    out["lds_rate"] = {}
    for variant, vname in ((0, "xorshift_random"), (1, "lcg_random")):
        for threads in (256, 512, 1024):
            blocks = sms * 8
            iters = 512
            rec = {"blocks": blocks, "threads": threads, "iters": iters}
            try:
                sampler.mark()
                ms = float(ext.lds_rate(variant, blocks, threads, iters, 5)[0])
                lookups = blocks * threads * iters * 8
                rec.update({"ms": ms, "lookups_per_s": lookups / ms * 1e3,
                            "lookups_per_sm_per_ns": lookups / sms / (ms * 1e6),
                            "sampled": sampler.since_mark()})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)
            out["lds_rate"][f"{vname}@{threads}"] = rec
    return out


# ---------------------------------------------------------------------------
# ceilings: torch-level DRAM and dense MMA
# ---------------------------------------------------------------------------

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = False
        self.source = None
        self._mark = time.time()
        try:
            import pynvml

            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nv = pynvml
            self.nv.nvmlDeviceGetPowerUsage(self.h)
            self.source = "pynvml"
        except Exception as exc:  # noqa: BLE001
            self.source = f"unavailable: {exc!r}"
            self.nv = None

    def run(self):
        while not self.stop_flag and self.nv is not None:
            t = time.time()
            try:
                p = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                c = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                self.samples.append((t, p, c))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)

    def mark(self):
        torch.cuda.synchronize()
        self._mark = time.time()

    def since_mark(self):
        torch.cuda.synchronize()
        t1 = time.time()
        rows = [s for s in self.samples if self._mark <= s[0] <= t1]
        if not rows:
            return {"samples": 0, "seconds": t1 - self._mark}
        return {"samples": len(rows), "seconds": t1 - self._mark,
                "power_w_mean": sum(r[1] for r in rows) / len(rows),
                "power_w_max": max(r[1] for r in rows),
                "sm_clock_mhz_mean": sum(r[2] for r in rows) / len(rows),
                "sm_clock_mhz_min": min(r[2] for r in rows),
                "sm_clock_mhz_max": max(r[2] for r in rows)}


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def ceilings(args, sampler):
    out = {}
    dev = torch.device("cuda")
    # DRAM: a 1 GiB device copy (read + write) and a read-only reduction
    try:
        a = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
        a.fill_(1)
        b = torch.empty_like(a)
        sampler.mark()
        ms = timed(lambda: b.copy_(a), 10)
        out["dram_copy"] = {"ms": ms, "gb_per_s": 2 * (1 << 30) / ms / 1e6, "sampled": sampler.since_mark()}
        sampler.mark()
        ms = timed(lambda: a.view(torch.int64).sum(), 10)
        out["dram_read_sum"] = {"ms": ms, "gb_per_s": (1 << 30) / ms / 1e6, "sampled": sampler.since_mark()}
        del a, b
    except Exception as exc:  # noqa: BLE001
        out["dram_error"] = repr(exc)
    n = 8192
    try:
        x = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        y = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        sampler.mark()
        ms = timed(lambda: x @ y, 10)
        out["mm_bf16_8192"] = {"ms": ms, "tflops": 2 * n ** 3 / ms / 1e9, "sampled": sampler.since_mark()}
        xh, yh = x.to(torch.float16), y.to(torch.float16)
        sampler.mark()
        ms = timed(lambda: xh @ yh, 10)
        out["mm_f16_8192"] = {"ms": ms, "tflops": 2 * n ** 3 / ms / 1e9, "sampled": sampler.since_mark()}
        x8 = x.to(torch.float8_e4m3fn)
        y8 = y.t().contiguous().t().to(torch.float8_e4m3fn)   # column-major B, the _scaled_mm contract
        s = torch.ones((), device=dev)
        sampler.mark()
        ms = timed(lambda: torch._scaled_mm(x8, y8, scale_a=s, scale_b=s, out_dtype=torch.bfloat16), 10)
        out["scaled_mm_fp8_8192"] = {"ms": ms, "tflops": 2 * n ** 3 / ms / 1e9, "sampled": sampler.since_mark()}
        del x8, y8
    except Exception as exc:  # noqa: BLE001
        out["mm_error"] = repr(exc)
        out["mm_traceback"] = traceback.format_exc()[-2000:]
    try:
        import vllm._custom_ops as ops  # noqa: F401

        x = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        y = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        gx = (448.0 * 6.0 / x.abs().max()).float()
        gy = (448.0 * 6.0 / y.abs().max()).float()
        # The Python wrapper allocates the e4m3 swizzled scale tensors the stable-ABI
        # GEMM type-checks (the raw op's scale dtype failed that check, census-1).
        xq, xs = ops.scaled_fp4_quant(x, gx, True)
        yq, ys = ops.scaled_fp4_quant(y, gy, True)
        xs = xs.view(torch.float8_e4m3fn) if xs.dtype == torch.uint8 else xs
        ys = ys.view(torch.float8_e4m3fn) if ys.dtype == torch.uint8 else ys
        alpha = (1.0 / (gx * gy)).float().reshape(1)
        fn = None
        if hasattr(ops, "cutlass_scaled_fp4_mm"):
            fn = lambda: ops.cutlass_scaled_fp4_mm(xq, yq, xs, ys, alpha, torch.bfloat16)  # noqa: E731
        elif callable(getattr(torch.ops._C, "cutlass_scaled_fp4_mm", None)):
            fn = lambda: torch.ops._C.cutlass_scaled_fp4_mm(xq, yq, xs, ys, alpha, torch.bfloat16)  # noqa: E731
        if fn is None:
            out["fp4_mm_8192"] = {"error": "no cutlass_scaled_fp4_mm op on this image"}
        else:
            sampler.mark()
            ms = timed(fn, 10)
            out["fp4_mm_8192"] = {"ms": ms, "tflops": 2 * n ** 3 / ms / 1e9, "sampled": sampler.since_mark()}
    except Exception as exc:  # noqa: BLE001
        out["fp4_mm_error"] = repr(exc)
        out["fp4_mm_traceback"] = traceback.format_exc()[-2000:]
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="wires,micro,ceilings")
    ap.add_argument("--image", default=os.environ.get("ORACLE_IMAGE"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = args.parts.split(",")
    sampler = Sampler()
    sampler.start()
    report = {"schema": "tessera.routed_census.v1", "issue": "RobTand/tessera#640",
              "host": os.environ.get("HOST_NAME") or os.uname().nodename,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "image": args.image, "pb_action": os.environ.get("PB_ACTION_KEY"),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "power_sampler": sampler.source, "argv": sys.argv}
    try:
        import triton
        report["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    for part in parts:
        log("census part", part)
        try:
            if part == "wires":
                report["wires"] = {"tp1_oracle_cache": tp1_wire_facts(args),
                                   "tp2_exports": tp2_wire_facts(args)}
            elif part == "micro":
                report["micro"] = micro(args, sampler)
            elif part == "ceilings":
                report["ceilings"] = ceilings(args, sampler)
        except Exception as exc:  # noqa: BLE001
            report[part] = {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]}
        (out_dir / "census.json").write_text(json.dumps(report, indent=1, default=str))
    sampler.stop_flag = True
    (out_dir / "census.json").write_text(json.dumps(report, indent=1, default=str))
    log("census written", out_dir / "census.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
