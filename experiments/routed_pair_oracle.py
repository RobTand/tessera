#!/usr/bin/env python3
"""Routed-pair correctness oracle and historical stock-control profiles.
The three routed families can execute on sm_121. Correctness checks do not
qualify a serving cell, a price, or a performance claim.

WHAT IT CHECKS.  For each routed launch pair, real production expert wires at
the served GLM-5.3-Flash expert shapes (hidden 4096, moe_intermediate 2048,
top-8) are loaded through the SAME builder and loader callbacks a serve uses
(``moe_route.build_tessera_moe_method`` /
``nvfp4_moe_route.build_tessera_nvfp4_moe_method`` -> ``create_weights`` ->
every ``weight_loader`` -> ``process_weights_after_loading`` -> ``apply``), and
the routed output is held to an independent reference:

* ``TESSERA_E4M3_K1`` -> ``(NativeWindowMoE.__call__, native_window_moe_compact)``
* ``TESSERA_BF16_K1`` -> ``(NativeWindowMoE.__call__, native_window_moe_compact_bf16)``
* ``TESSERA_E2M1_K2`` -> ``(kernel_a4.a4_span2_grouped_gemm, native_span2_grouped)``

The launch pair is not asserted from the family: it is read off the route's
own ``emit_route`` telemetry during ``apply`` and compared with the pair above.

THE REFERENCE. The materializing reader parses each projection wire.
BF16 uses ``tessera.decode.materialize_bf16`` for raw BF16 values and
separate FP32 row scales. The FP64 dot applies those scales after its sum.
No BF16 stock checkpoint enters the numerical oracle. FP8 keeps its raw
bytes and row scales; NVFP4 uses ``stock_dequant``. Activations use the
runtime quantizers: per-token E4M3, group-16 NVFP4, or unchanged BF16.
Every product is summed in fp64 on exactly representable operands, so the
reference is exact to ~1e-15.  Placement mirrors vLLM's modular path the way
``tests/test_native_window_moe.py`` (``_per_expert``/``_routes``/``_down``) and
``tests/test_kernel_a4.py`` (``check_two_stage_moe``) do: router weights on
gemm2 only; window families round gemm1 to bf16, apply
``silu(min(g, L)) * clamp(u, -L, L)`` in fp32 and round once to bf16, quantise
per route, round each weighted route to bf16 and sum; the A4 family keeps gemm1
in fp32, rounds the activation to bf16 at the quantiser, weights the fp32 routes
and sums.

THE BOUND (dtype-derived, per output element; nothing here is fitted).
  u16 = 2^-8 (bf16 unit roundoff, RNE), u32 = 2^-24 (fp32, RNE),
  u_acc = 2^-23 (one fp32 ulp per accumulation step: covers a truncating /
  round-toward-zero MMA adder, the documented behaviour of NVIDIA tensor-core
  accumulation, not only RNE), gamma(n, u) = n u / (1 - n u),
  eps_f = 2^-15 (fp32 silu/sigmoid evaluated with a fast-math expf, <= ~104 ulp
  for |g| <= 88, plus the divide and multiply), L_silu = 1.0998 (max |silu'|).
* Stage GEMM of K terms whose products are exact in fp32 (fp8 x fp8, bf16 x bf16,
  e2m1 x e2m1 x e4m3 x e4m3 all are), with up to three fp32 epilogue multiplies:
  |n - r| <= gamma(K, u_acc) S + gamma(3, u32)(|r| + gamma(K, u_acc) S),
  S = sum_k |a_k w_k| |scales| (fp64); then the output format's rounding:
  + u16 (|r| + that) for a bf16 output, nothing more for fp32.
* Teacher-forced stages (each native stage fed the NATIVE upstream output;
  both A-side quantisers run outside the Triton kernels, so with identical
  input the quantised operands are bit-identical and no rounding-boundary flip
  can occur): the bound above, per stage; the reduced down output adds each
  route's bf16 rounding, the 8-term fp32 sum gamma(8, u32) and the final bf16.
* End to end (``apply`` vs the independent pipeline, which re-quantises its OWN
  intermediates): the stage-1 bound plus the reference's own rounding gives
  |dg|, |du|; the activation is Lipschitz,
  |df| <= L_silu |dg| (min(|u|, L) + |du|) + |silu(min(g, L))| |du|, plus eps_f and
  u16 on both sides; the quantiser is discontinuous, so its perturbation is
  bounded element by element from the formats' grids: an element whose
  scaled value lies farther than the perturbation from every rounding midpoint
  of the E4M3 (or E2M1) grid keeps its code and moves only by the scale
  change; one that could cross a midpoint moves by at most the perturbation
  plus one grid gap at that magnitude; an NVFP4 block whose E4M3 block scale
  could itself round differently is bounded by |da| + |q_r - a_r| + SF_hi/gs
  (E2M1's largest half gap is 1), and a subnormal-scale block crudely by
  |a| + 6 SF_hi / gs.  The perturbation then propagates through the down GEMM
  as sum_k |w_k| b_k plus the accumulation term, then the route roundings,
  the 8-term sum and the final bf16 of both sides.
The existing suite SCREENS are reported next to the bound and labelled as
chosen screens, never as the pass criterion: ``_tol`` of
``tests/test_window_gemm_grouped.py`` (5e-3 + 1e-2 max|ref|) and the A4
two-stage ``rel < 2e-2`` of ``tests/test_kernel_a4.py``.

PROFILES (``--mode profile``).  One routed layer per family at every expert
(288 by default), M selected by --m (default 1,64,512): "after" is the native ``apply``; "before" is the
materialised arithmetic the withdrawn cells named -- the stock tiles from
``materialize_stock`` handed to vLLM's own modular fused-MoE kernel for the
family (``make_fp8_moe_kernel``, ``make_unquantized_moe_kernel``,
``make_nvfp4_moe_kernel`` over the backend the runtime's own selector picks).
Each leg records wall time per forward (CUDA events, after warmup), a
``torch.profiler`` kernel table sorted by self device time, and a steady
unprofiled replay window whose UTC bounds are written out so the Netdata GPU
power series can be read for exactly that window; an in-process NVML power
sampler records mean and peak W beside it.

Run inside the serving image through ``experiments/routed_pair_oracle.sh``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import torch

from tessera.serving.telemetry import (DECODER_NATIVE_ROUTED_FUSED_WINDOW_BF16,
                                       DECODER_NATIVE_WINDOW_MOE_COMPACT_BF16)

WIRE_ROOT = ("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
             "activation-runtime-allocation-20260911/union-a4a8a16-01/cache/wire")
SCALES = ("/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
          "activation-runtime-allocation-20260911/union-a4a8a16-01/cache/"
          "input_scales.safetensors")
HIDDEN, INTER, TOP_K = 4096, 2048, 8
PROJ = ("gate_proj", "up_proj", "down_proj")
SHARDS = (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))

FAMILIES = {
    "e4m3": {"payload": "TESSERA_E4M3_K1", "rung": "R1024", "family": "TESSERA_FP8",
             "pair": ("tessera.native_window_moe.NativeWindowMoE.__call__",
                      "native_window_moe_compact"),
             "fused_pair": ("tessera.routed_fused.FusedRoutedWindowMoE.__call__",
                            "native_routed_fused_window")},
    "bf16": {"payload": "TESSERA_BF16_K1", "rung": "R1024", "family": "TESSERA_BF16",
             "pair": ("tessera.native_window_moe.NativeWindowMoE.__call__",
                      DECODER_NATIVE_WINDOW_MOE_COMPACT_BF16),
             "fused_pair": ("tessera.routed_fused.FusedRoutedWindowMoE.__call__",
                            DECODER_NATIVE_ROUTED_FUSED_WINDOW_BF16)},
    "e2m1": {"payload": "TESSERA_E2M1_K2", "rung": "R896", "family": "TESSERA_NVFP4",
             "pair": ("tessera.kernel_a4.a4_span2_grouped_gemm", "native_span2_grouped")},
}

FUSED_ADAPTER = "FusedRoutedWindowMoE"


def expected_pair(fam, method):
    """The launch pair the built method must emit.

    ``PackedWindowMoeBundles.adapter`` (tessera#640) takes the fused
    warp-specialised lane for every stack it admits and the compact Triton
    adapter otherwise; the expectation is keyed on WHICH adapter class was
    built, never read off the adapter's own ``launch_pair`` (that would make
    the telemetry check circular).  The A4 family's pair is fixed.
    """
    nat = getattr(method, "_native", None)
    if type(nat).__name__ == FUSED_ADAPTER:
        # The E4M3 family's library is a construction fact of the adapter
        # (``routed_fused.library_for`` at ``from_bundles``), not its telemetry.
        if getattr(nat, "library", None) == "e4m3mma":
            return (fam["fused_pair"][0], "native_routed_fused_window_e4m3mma")
        return tuple(fam["fused_pair"])
    return tuple(fam["pair"])


def f16_twin(method):
    """The fused lane on the f16 instruction (``tessera_routed_fused_e4m3``)
    over the SAME prepared bundles as an E4M3-instruction ``method._native``,
    or None for any other adapter: the two differ in the tensor-core
    instruction only, so their outputs differ by fp32 summation order."""
    nat = getattr(method, "_native", None)
    if type(nat).__name__ != FUSED_ADAPTER or getattr(nat, "library", None) != "e4m3mma":
        return None
    from tessera import routed_fused as rf

    old = os.environ.get(rf.ENV_E4M3_MMA)
    os.environ[rf.ENV_E4M3_MMA] = "f16"
    try:
        twin = rf.FusedRoutedWindowMoE.from_bundles(nat.gate, nat.up, nat.down,
                                                    activation=nat.activation)
    finally:
        if old is None:
            os.environ.pop(rf.ENV_E4M3_MMA, None)
        else:
            os.environ[rf.ENV_E4M3_MMA] = old
    assert twin.library == "e4m3", twin.library
    return twin


def compact_twin(method, activation="silu"):
    """The compact Triton adapter over the SAME prepared bundles as a fused
    ``method._native`` (no second copy of the weights), or None when the
    method's adapter is not the fused lane."""
    nat = getattr(method, "_native", None)
    if type(nat).__name__ != FUSED_ADAPTER:
        return None
    from tessera.native_window_moe import native_window_moe_from_bundles

    return native_window_moe_from_bundles(nat.down, gate=nat.gate, up=nat.up,
                                          activation=getattr(nat, "activation", activation))


# ---- the dtype constants the bound is built from ---------------------------
U16 = 2.0 ** -8
U32 = 2.0 ** -24
U_ACC = 2.0 ** -23
EPS_F = 2.0 ** -15
SILU_LIP = 1.0998
F64_SLACK = 1e-12          # fp64 summation of exact products, relative to S
QUANT_MARGIN = 2.0 ** -20  # the runtime quantisers' own fp32 scale arithmetic
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def gamma(n: int, u: float) -> float:
    return n * u / (1.0 - n * u)


BOUND_CONSTANTS = {
    "u16": U16, "u32": U32, "u_acc": U_ACC, "eps_f": EPS_F, "silu_lipschitz": SILU_LIP,
    "fp64_slack_relative_to_S": F64_SLACK, "quantiser_margin_relative": QUANT_MARGIN,
    "why": {
        "u16": "bf16 has an 8-bit significand; RNE unit roundoff 2^-8",
        "u32": "fp32 RNE unit roundoff for the epilogue multiplies",
        "u_acc": "one fp32 ulp per accumulation step, covering a round-toward-zero "
                 "tensor-core adder rather than assuming RNE",
        "eps_f": "fp32 silu with a fast-math expf: <= 2 + 1.16|x| ulp for |x| <= 88 "
                 "(~104 ulp) plus the divide and the product, rounded up to 256 ulp",
        "silu_lipschitz": "max over x of |d/dx x*sigmoid(x)| = 1.0998",
        "quantiser_margin_relative": "the runtime quantisers compute their scaled "
                                     "value with fp32 reciprocals; midpoint "
                                     "distances are widened by 2^-20 relative",
    },
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def src_tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((root / "src" / "tessera").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".json"):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def wire_path(root, layer, expert, proj, fam):
    return Path(root) / (f"model__language_model__layers__{layer}__mlp__experts__{expert}__"
                         f"{proj}__{fam['payload']}_{fam['rung']}.tessera")


def load_wires(args, fam, experts):
    """Read every (expert, projection) wire by its constructed path (no listing),
    with a bounded read-ahead pool; digest each."""
    jobs = [(e, p, wire_path(args.wire_root, args.layer, e, p, fam)) for e in experts for p in PROJ]
    blobs = {p: [b""] * len(experts) for p in PROJ}
    files = []
    index = {e: i for i, e in enumerate(experts)}
    from tessera import fused, unit_artifact

    def framed(data, proj):
        # The cache holds bare unit artifacts (magic "\x89TESSERA"); a served
        # checkpoint stores ONE single-member ``tessera.fused`` container per
        # expert projection (scheme.expert_role_declarations), named by the
        # projection with the unit's own row count.  Frame it exactly so.
        if data[:8] == fused.FUSED_MAGIC:
            return data
        rows = int(unit_artifact.parse_unit_metadata(data).rows)
        return fused.pack_fused([(proj, rows, data)])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for (e, p, path), data in zip(jobs, pool.map(lambda j: j[2].read_bytes(), jobs)):
            blob = framed(data, p)
            blobs[p][index[e]] = blob
            files.append({"expert": e, "projection": p, "path": str(path),
                          "bytes": len(data), "sha256": sha256_bytes(data),
                          "framed_bytes": len(blob)})
    return blobs, files


def load_input_scales(args, experts):
    from safetensors import safe_open

    out = {p: [] for p in PROJ}
    with safe_open(args.scales, framework="pt") as f:
        meta = f.metadata()
        for e in experts:
            for p in PROJ:
                key = f"model.language_model.layers.{args.layer}.mlp.experts.{e}.{p}.input_global_scale"
                out[p].append(float(f.get_tensor(key).reshape(-1)[0]))
    return out, meta


def scheme_for(fam, blobs, n_experts):
    """The sidecar scheme, built from the wires' own verified metadata."""
    from tessera import fused, unit_artifact

    def facts(blob):
        members = fused.parse_fused(bytes(blob))
        if len(members) != 1:
            raise SystemExit(f"a projection container frames {len(members)} members")
        meta = unit_artifact.parse_unit_metadata(members[0].blob)
        return {"name": members[0].name, "grid": meta.grid.name, "body": meta.body.name,
                "plane": meta.manifest.scale_plane.kind.name,
                "q256": int(meta.manifest.branch.root_q256) // int(meta.grid.arity),
                "rows": int(meta.rows), "columns": int(meta.columns)}

    g, u, d = (facts(blobs[p][0]) for p in PROJ)
    for f, p in zip((g, u, d), PROJ):
        if f["name"] != p:
            raise SystemExit(f"container member {f['name']!r} is not {p!r}")
    for key in ("grid", "body", "plane"):
        if len({g[key], u[key], d[key]}) != 1:
            raise SystemExit(f"{key} differs across projections: {g[key]}, {u[key]}, {d[key]}")
    scheme = {
        "family": fam["family"], "structure": "routed_moe", "grid": g["grid"],
        "body": g["body"], "plane": g["plane"], "experts": int(n_experts),
        "groups": {
            "w13": {"rows": g["rows"] + u["rows"], "columns": g["columns"],
                    "q256": [g["q256"], u["q256"]],
                    "wire_stride": max(len(b) for p in ("gate_proj", "up_proj") for b in blobs[p]),
                    "roles": [["gate_proj", g["rows"]], ["up_proj", u["rows"]]]},
            "w2": {"rows": d["rows"], "columns": d["columns"], "q256": d["q256"],
                   "wire_stride": max(len(b) for b in blobs["down_proj"]),
                   "roles": [["down_proj", d["rows"]]]},
        },
    }
    return scheme, {"gate": g, "up": u, "down": d}


def moe_config(n_experts, clamp):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                             FusedMoEParallelConfig,
                                                             RoutingMethodType)

    parallel = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0,
        ep_rank=0, sp_size=1, use_ep=False, all2all_backend="allgather_reducescatter",
        enable_eplb=False)
    return FusedMoEConfig(
        num_experts=n_experts, experts_per_token=TOP_K, hidden_dim=HIDDEN,
        intermediate_size=INTER, num_local_experts=n_experts, num_logical_experts=n_experts,
        activation=MoEActivation.SILU, device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepSeekV3, moe_parallel_config=parallel,
        in_dtype=torch.bfloat16, swiglu_limit=clamp)


def make_layer(cfg, n_experts, clamp) -> Any:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    layer: Any = torch.nn.Module()  # vLLM dynamically attaches loader parameters.
    layer.moe_config = cfg
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer.activation = MoEActivation.SILU
    layer.global_num_experts = n_experts
    layer.swiglu_limit = clamp
    layer._expert_routing_tables = lambda: None
    return layer


class RouteRecorder:
    """Wrap ``emit_route`` in the two route modules: the launch pair is read off
    what the route itself reports during ``apply``, never inferred."""

    def __init__(self):
        self.calls = []
        self._saved = []

    def install(self):
        from tessera.serving import moe_route, nvfp4_moe_route

        for module in (moe_route, nvfp4_moe_route):
            real = module.emit_route
            self._saved.append((module, real))

            def wrapped(layer, *a, _real=real, **k):
                self.calls.append({"symbol": k.get("symbol"), "decoder": k.get("decoder"),
                                   "contract": k.get("contract"), "policy": k.get("policy"),
                                   "state": k.get("state"), "kind": k.get("kind")})
                return _real(layer, *a, **k)
            setattr(module, "emit_route", wrapped)

    def take(self):
        out, self.calls = self.calls, []
        return out


def build_after(fam, scheme, blobs, scales, n_experts, cfg, clamp, prefix):
    """The route's own method, through every loader callback a serve makes."""
    layer = make_layer(cfg, n_experts, clamp)
    t0 = time.time()
    if fam["family"] == "TESSERA_NVFP4":
        from tessera.serving.nvfp4_moe_route import build_tessera_nvfp4_moe_method

        method = build_tessera_nvfp4_moe_method(scheme, prefix, "resident", layer)
        method.create_weights(layer, n_experts, HIDDEN, INTER, torch.bfloat16,
                              global_num_experts=n_experts)
    else:
        from tessera.serving import moe_route

        method = moe_route.build_tessera_moe_method(scheme, prefix, "resident", layer)
        method.create_weights(layer, n_experts, HIDDEN, INTER, torch.bfloat16)
    layer.quant_method = method
    for e in range(n_experts):
        for shard, proj in SHARDS:
            param = layer.w2_wire if shard == "w2" else layer.w13_wire
            ok = param.weight_loader(param, torch.frombuffer(bytearray(blobs[proj][e]),
                                                             dtype=torch.uint8),
                                     f"{prefix}.{e}.{proj}.wire", shard, e,
                                     return_success=True)
            if ok is False:
                raise SystemExit(f"loader refused expert {e} {proj}")
            if fam["family"] == "TESSERA_NVFP4":
                sparam = (layer.w2_input_global_scale if shard == "w2"
                          else layer.w13_input_global_scale)
                sparam.weight_loader(sparam, torch.tensor([scales[proj][e]], dtype=torch.float32),
                                     f"{prefix}.{e}.{proj}.input_global_scale", shard, e,
                                     return_success=True)
    method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    info = {"tessera_decoder": getattr(layer, "tessera_decoder", None),
            "tessera_backend": getattr(layer, "tessera_backend", None),
            "load_seconds": round(time.time() - t0, 3)}
    if fam["family"] == "TESSERA_NVFP4":
        info["gs13"] = float(layer.tessera_a4_gs13)
        info["gs2"] = float(layer.tessera_a4_gs2)
    else:
        assert method._native is not None
        info["native_adapter"] = type(method._native).__name__
        info["launch_pair"] = list(method._native.launch_pair)
        info["family_of_bundles"] = getattr(method._native.down, "family", None)
        info["fused_gate_up"] = method._native.gate_up is not None
    return layer, method, info


# ---------------------------------------------------------------------------
# reference weights
# ---------------------------------------------------------------------------

def parse_roles(scheme, prefix):
    from tessera.serving.scheme import expert_role_declarations, validate_tessera_moe_scheme

    declared = validate_tessera_moe_scheme(scheme, prefix)
    roles = {}
    for group in ("w13", "w2"):
        for decl in expert_role_declarations(declared["groups"][group]):
            roles[decl["roles"][0][0]] = decl
    return declared, roles


def parsed_unit(blob, role, target, device):
    from tessera.serving.scheme import parse_tessera_expert_blob

    parsed = parse_tessera_expert_blob(blob, role, target, device=device)
    if len(parsed) != 1:
        raise SystemExit(f"{target}: {len(parsed)} roles")
    return parsed[0][1]


def _reference_factors(parsed, family, device):
    """Read canonical factors without a per-weight BF16 conversion."""
    if family == "TESSERA_BF16":
        from tessera.decode import materialize_bf16

        values, scale = materialize_bf16(parsed.unit, parsed.forests, parsed.code)
        return values.to(device), scale.to(device)

    from tessera.stock import materialize_stock, stock_dequant

    tiles = materialize_stock(parsed.unit, parsed.forests, parsed.code)
    if family == "TESSERA_FP8":
        return (tiles["weight"].to(device).float().contiguous(),
                tiles["weight_scale"].to(device).reshape(-1).float())
    tiles = {name: value.to(device) for name, value in tiles.items()}
    return stock_dequant(tiles).float().contiguous(), None


def reference_weights(fam, scheme, blobs, n_experts, prefix, device="cuda"):
    """Read each projection into values and optional row multipliers."""
    _declared, roles = parse_roles(scheme, prefix)
    ref = {p: {"W": [], "m": []} for p in PROJ}
    t0 = time.time()
    for e in range(n_experts):
        for p in PROJ:
            parsed = parsed_unit(blobs[p][e], roles[p], f"{prefix} {p} expert {e}", device)
            values, scale = _reference_factors(parsed, fam["family"], device)
            ref[p]["W"].append(values)
            ref[p]["m"].append(scale)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return ref, round(time.time() - t0, 3)


# ---------------------------------------------------------------------------
# the contract's activation quantisers (the runtime's own ops)
# ---------------------------------------------------------------------------

def fp8_quant_rows(a):
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    a = a.contiguous()
    out = torch.empty(a.shape, dtype=torch.float8_e4m3fn, device=a.device)
    scale = torch.empty((a.shape[0], 1), dtype=torch.float32, device=a.device)
    if a.shape[0]:
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, a, scale, None)
    return out, scale


_E2M1_TABLE = {}


def e2m1_table(device):
    if device not in _E2M1_TABLE:
        vals = [(-1.0 if i >> 3 else 1.0) * E2M1_VALUES[i & 7] for i in range(16)]
        _E2M1_TABLE[device] = torch.tensor(vals, dtype=torch.float64, device=device)
    return _E2M1_TABLE[device]


def nvfp4_quant_rows(a_bf16, gs):
    """vLLM's ``scaled_fp4_quant`` in the linear scale layout; returns the codes'
    values (fp64), the block scales (fp64) and the dequantised activation."""
    from vllm import _custom_ops as ops

    rows, cols = a_bf16.shape
    packed, sf = ops.scaled_fp4_quant(a_bf16.contiguous(), gs.reshape(1).float(),
                                      is_sf_swizzled_layout=False)
    packed = packed.view(torch.uint8).reshape(rows, -1)[:, : cols // 2]
    sf = sf.view(torch.uint8).reshape(rows, -1)[:, : cols // 16].contiguous()
    table = e2m1_table(a_bf16.device)
    codes = torch.empty((rows, cols), dtype=torch.long, device=a_bf16.device)
    codes[:, 0::2] = (packed & 0xF).long()
    codes[:, 1::2] = (packed >> 4).long()
    values = table[codes]
    sf_val = sf.view(torch.float8_e4m3fn).double()
    deq = values * torch.repeat_interleave(sf_val, 16, dim=1) / float(gs)
    return values, sf_val, deq


# ---------------------------------------------------------------------------
# grids and the quantiser perturbation bounds
# ---------------------------------------------------------------------------

_GRIDS = {}


def grid(kind, device):
    key = (kind, str(device))
    if key not in _GRIDS:
        if kind == "e4m3":
            v = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
            v = v.double()
            v = torch.unique(v[torch.isfinite(v) & (v >= 0)])
        else:
            v = torch.tensor(E2M1_VALUES, dtype=torch.float64)
        v = v.to(device)
        mids = (v[1:] + v[:-1]) / 2
        gaps = v[1:] - v[:-1]
        _GRIDS[key] = (v, mids, gaps)
    return _GRIDS[key]


def dist_to_midpoint(y_abs, kind):
    """Distance from |y| to the nearest rounding boundary of the grid (inf above
    the saturation value, where every value maps to the maximum)."""
    _v, mids, _g = grid(kind, y_abs.device)
    idx = torch.searchsorted(mids, y_abs.contiguous())
    lo = mids[(idx - 1).clamp(min=0)]
    hi = mids[idx.clamp(max=mids.numel() - 1)]
    d_lo = torch.where(idx > 0, (y_abs - lo).abs(), torch.full_like(y_abs, math.inf))
    d_hi = torch.where(idx < mids.numel(), (hi - y_abs).abs(), torch.full_like(y_abs, math.inf))
    return torch.minimum(d_lo, d_hi)


def cell_gap(v_abs, kind):
    """The spacing of the grid cell containing |v| (the last gap beyond max)."""
    vals, _m, gaps = grid(kind, v_abs.device)
    idx = (torch.searchsorted(vals, v_abs.contiguous(), right=True) - 1).clamp(0, gaps.numel() - 1)
    return gaps[idx]


def quant_bound_fp8_rows(a_r, q_r, s_r, delta):
    """|Q(a_n) - Q(a_r)| elementwise for per-row dynamic E4M3, given
    |a_n - a_r| <= delta.  a_r, q_r, delta: [R, K] fp64; s_r: [R] fp64."""
    s_r = s_r.reshape(-1, 1)
    d_amax = delta.max(dim=1, keepdim=True).values
    amax = a_r.abs().max(dim=1, keepdim=True).values
    ds = d_amax / 448.0 + 4 * U32 * s_r
    s_lo = torch.clamp(torch.minimum((amax - d_amax) / 448.0, s_r - ds), min=1e-30)
    y_r = a_r / s_r
    c_r = q_r / s_r
    dy = (delta + a_r.abs() * ds / s_lo) / s_lo + QUANT_MARGIN * (y_r.abs() + 1e-30)
    cand = dist_to_midpoint(y_r.abs(), "e4m3") <= dy
    s_hi = s_r + ds
    flip = s_hi * (dy + cell_gap(y_r.abs() + dy, "e4m3")) + c_r.abs() * ds
    keep = c_r.abs() * ds
    return torch.where(cand, flip, keep), cand


def quant_bound_nvfp4(a_r, q_r, sf_r, gs, delta):
    """|Q(a_n) - Q(a_r)| elementwise for NVFP4 group-16 under a static global gs.
    a_r, q_r, delta: [R, K] fp64; sf_r: [R, K/16] fp64 (the reference's block
    scales).  Returns (bound, element_candidates, block_candidates, subnormal_blocks)."""
    rows, cols = a_r.shape
    blocks = cols // 16
    a_b = a_r.reshape(rows, blocks, 16)
    d_b = delta.reshape(rows, blocks, 16)
    q_b = q_r.reshape(rows, blocks, 16)
    vmax = a_b.abs().max(-1).values
    dmax = d_b.max(-1).values
    t_r = gs * vmax / 6.0
    dt = gs * dmax / 6.0 + QUANT_MARGIN * t_r
    sf_cand = dist_to_midpoint(t_r, "e4m3") <= dt
    min_normal = 2.0 ** -6
    subnormal = (t_r - dt) < min_normal * (1 + 2.0 ** -4)
    sf_hi = torch.clamp((t_r + dt) * (1 + 2.0 ** -4) + 2.0 ** -10, max=448.0)
    # element level, same block scale on both sides
    sf = sf_r.reshape(rows, blocks, 1)
    safe = torch.where(sf > 0, sf, torch.ones_like(sf))
    y_r = torch.where(sf > 0, a_b * gs / safe, torch.zeros_like(a_b))
    dy = d_b * gs / safe + QUANT_MARGIN * y_r.abs()
    el_cand = dist_to_midpoint(y_r.abs().reshape(-1), "e2m1").reshape(y_r.shape) <= dy
    el_flip = (safe / gs) * (dy + cell_gap((y_r.abs() + dy).reshape(-1), "e2m1").reshape(y_r.shape))
    same_sf = torch.where(el_cand, el_flip, torch.zeros_like(el_flip))
    same_sf = torch.where(sf > 0, same_sf, torch.zeros_like(same_sf))  # SF 0 both sides: q = 0
    # the block scale itself may round differently
    err_r = (q_b - a_b).abs()
    normal_blk = d_b + err_r + (sf_hi / gs).unsqueeze(-1)
    crude_blk = d_b + err_r + (vmax + dmax).unsqueeze(-1) + (6.0 * sf_hi / gs).unsqueeze(-1)
    blk = torch.where(subnormal.unsqueeze(-1), crude_blk, normal_blk)
    bound = torch.where(sf_cand.unsqueeze(-1), blk, same_sf)
    return (bound.reshape(rows, cols), el_cand.reshape(rows, cols), sf_cand, subnormal)


# ---------------------------------------------------------------------------
# routing and per-expert exact GEMMs
# ---------------------------------------------------------------------------

def make_inputs(m, n_experts, seed, sigma):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, HIDDEN, generator=g) * sigma).bfloat16().cuda()
    ids = torch.rand(m, n_experts, generator=g).argsort(dim=1)[:, :TOP_K].to(torch.int32).cuda()
    w = torch.rand(m, TOP_K, generator=g) + 0.1
    w = (w / w.sum(dim=1, keepdim=True)).float().cuda()
    return x, ids, w


def expert_gemm(a_rows, route_expert, ref, proj, extra_abs=None, extra=None):
    """Per expert (host loop over E only): r = A W^T m, S = |A| |W|^T |m| in fp64;
    optionally Sx = extra_abs |W|^T |m| and Px = extra |W|^T |m|."""
    n_rows = ref[proj]["W"][0].shape[0]
    R = a_rows.shape[0]
    r = torch.zeros(R, n_rows, dtype=torch.float64, device=a_rows.device)
    S = torch.zeros_like(r)
    Sx = torch.zeros_like(r) if extra_abs is not None else None
    Px = torch.zeros_like(r) if extra is not None else None
    for e in range(len(ref[proj]["W"])):
        idx = (route_expert == e).nonzero().reshape(-1)
        if idx.numel() == 0:
            continue
        W = ref[proj]["W"][e].double()
        m = ref[proj]["m"][e]
        mm = m.double().reshape(1, -1) if m is not None else None
        A = a_rows.index_select(0, idx)
        Wa = W.abs()
        re = A @ W.t()
        se = A.abs() @ Wa.t()
        if mm is not None:
            re = re * mm
            se = se * mm.abs()
        r.index_copy_(0, idx, re)
        S.index_copy_(0, idx, se)
        if extra_abs is not None:
            assert Sx is not None
            sx = extra_abs.index_select(0, idx) @ Wa.t()
            Sx.index_copy_(0, idx, sx * mm.abs() if mm is not None else sx)
        if extra is not None:
            assert Px is not None
            px = extra.index_select(0, idx) @ Wa.t()
            Px.index_copy_(0, idx, px * mm.abs() if mm is not None else px)
        del W, Wa, A, re, se
    return r, S, Sx, Px


def gemm_bound(S, r, K, epilogue_mults=3):
    acc = gamma(K, U_ACC) * S + F64_SLACK * S
    return acc + gamma(epilogue_mults, U32) * (r.abs() + acc)


def summarize(diff, bound, ref):
    ratio = diff / torch.clamp(bound, min=1e-300)
    worst = int(torch.argmax(ratio.reshape(-1)))
    return {
        "max_abs_diff": float(diff.max()),
        "max_ref_abs": float(ref.abs().max()),
        "max_bound": float(bound.max()),
        "max_diff_over_bound": float(ratio.max()),
        "at_worst": {"diff": float(diff.reshape(-1)[worst]), "bound": float(bound.reshape(-1)[worst]),
                     "ref": float(ref.reshape(-1)[worst])},
        "violations": int((diff > bound).sum()),
        "elements": int(diff.numel()),
        "pass": bool((diff <= bound).all()),
    }


def bf16_ulp_stats(native, ref):
    """Descriptive, not a bound.  An output row is a sum with cancellation, so an
    element near zero carries the absolute error of the row's magnitudes; the
    unit is therefore the bf16 ulp at the ROW's max |ref| (2^(e-7) for
    |ref|_max in [2^e, 2^(e+1)))."""
    r = ref.double()
    d = (native.double() - r).abs()
    rowmax = r.abs().amax(dim=-1, keepdim=True).clamp(min=2.0 ** -126)
    ulp = torch.exp2(torch.floor(torch.log2(rowmax)) - 7)
    q = d / ulp
    return {"max_diff_in_bf16_ulps_of_row_max": float(q.max()),
            "elements_over_1_row_ulp": int((q > 1).sum()),
            "elements_differing": int((d > 0).sum()), "elements": int(d.numel()),
            "max_abs_diff_over_max_abs_ref": float(d.max() / r.abs().max().clamp(min=1e-300))}


def silu_f(g, u, lim):
    gc = torch.clamp(g, max=lim) if lim is not None else g
    uc = torch.clamp(u, min=-lim, max=lim) if lim is not None else u
    return torch.nn.functional.silu(gc) * uc, torch.nn.functional.silu(gc).abs(), uc.abs()


# ---------------------------------------------------------------------------
# the oracle, one family at one M
# ---------------------------------------------------------------------------

def a4_dispatch(ids, weights, n_experts):
    """The route's own dispatch (nvfp4_moe_route apply), rebuilt for the staged
    teacher-forced calls; E2E uses ``apply`` itself."""
    device = ids.device
    T, K = ids.shape
    flat_ids = ids.to(torch.int64).reshape(-1)
    flat_tokens = torch.arange(T, device=device, dtype=torch.int64).repeat_interleave(K)
    flat_weights = weights.reshape(-1).to(torch.float32)
    order = torch.argsort(flat_ids, stable=True)
    counts = torch.zeros(n_experts, dtype=torch.int32, device=device)
    counts.scatter_add_(0, flat_ids[order], torch.ones_like(flat_ids, dtype=torch.int32))
    offsets = torch.zeros(n_experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return order, offsets, flat_tokens[order].to(torch.int32), flat_weights[order]


def instruction_pair_case(nat, twin, x, ids, w, clamp, keep):
    """The E4M3-instruction lane against its f16 twin on the same inputs:
    both stages teacher-forced on ONE intermediate (the twin's), K = hidden
    for gate/up and K = intermediate for down, then the forward.  Each side
    is separately held to the exact reference by ``oracle_case`` (the native)
    and here (the twin's forward against the same end-to-end bound); the
    difference between them is reported in bf16 ulps and as a fraction of the
    end-to-end bound -- the size of the accumulation-order difference on
    real wires."""
    out = {}
    gu_m = nat.gate_up(x, ids, w, preserve=True)
    gu_f = twin.gate_up(x, ids, w, preserve=True)
    out["gate_up_K_hidden"] = {"ulps": bf16_ulp_stats(gu_m, gu_f.double()),
                               "bitwise_equal_fraction": float((gu_m == gu_f).double().mean())}
    P = ids.numel()
    from tessera import native_window_moe as nwm
    act = nwm._silu_and_mul(gu_f[..., :INTER].reshape(P, INTER), gu_f[..., INTER:].reshape(P, INTER),
                            clamp_limit=clamp)
    dn_m = nat.down_routes(act, ids, w, route_input=True, round_routes=True)
    dn_f = twin.down_routes(act, ids, w, route_input=True, round_routes=True)
    out["down_K_intermediate"] = {"ulps": bf16_ulp_stats(dn_m, dn_f.double()),
                                  "bitwise_equal_fraction": float((dn_m == dn_f).double().mean())}
    o_m = nat(x, ids, w, swiglu_limit=clamp, apply_router_weight_on_input=False)
    o_f = twin(x, ids, w, swiglu_limit=clamp, apply_router_weight_on_input=False)
    torch.cuda.synchronize()
    diff = (o_m.double() - o_f.double()).abs()
    twin_err = (o_f.double() - keep["out_r"].double()).abs()
    out["forward"] = {"max_abs_diff": float(diff.max()),
                      "max_diff_over_e2e_bound": float((diff / keep["Eout"]).max()),
                      "within_twice_e2e_bound": bool((diff <= 2 * keep["Eout"]).all()),
                      "bitwise_equal_fraction": float((o_m == o_f).double().mean()),
                      "ulps": bf16_ulp_stats(o_m, o_f.double()),
                      "twin_within_e2e_bound": bool((twin_err <= keep["Eout"]).all()),
                      "twin_max_diff_over_e2e_bound": float((twin_err / keep["Eout"]).max())}
    out["pass"] = out["forward"]["twin_within_e2e_bound"] and out["forward"]["within_twice_e2e_bound"]
    return out


def oracle_case(fkey, fam, layer, method, ref, x, ids, w, clamp, n_experts, recorder):
    from tessera import native_window_moe as nwm

    T = x.shape[0]
    P = T * TOP_K
    route_expert = ids.reshape(-1).long()
    route_token = torch.arange(T, device=x.device).repeat_interleave(TOP_K)
    rw = w.reshape(-1).double()
    out = {"M": T, "routes": P, "experts_hit": int(torch.unique(route_expert).numel())}

    # ---------------- stage 1 exact reference ---------------------------
    if fam["family"] == "TESSERA_FP8":
        xq, a1 = fp8_quant_rows(x)
        A = xq.double() * a1.double()
    elif fam["family"] == "TESSERA_BF16":
        A = x.double()
    else:
        gs13 = layer.tessera_a4_gs13
        _v, _s, A = nvfp4_quant_rows(x, gs13)
    A_routes = A.index_select(0, route_token)
    g_ex, S_g, _, _ = expert_gemm(A_routes, route_expert, ref, "gate_proj")
    u_ex, S_u, _, _ = expert_gemm(A_routes, route_expert, ref, "up_proj")
    epi1 = 3 if fam["family"] != "TESSERA_NVFP4" else 2
    E1g = gemm_bound(S_g, g_ex, HIDDEN, epi1)
    E1u = gemm_bound(S_u, u_ex, HIDDEN, epi1)
    bf16_s1 = fam["family"] != "TESSERA_NVFP4"
    B1g = E1g + U16 * (g_ex.abs() + E1g) if bf16_s1 else E1g
    B1u = E1u + U16 * (u_ex.abs() + E1u) if bf16_s1 else E1u

    # ---------------- native stages (teacher-forced) --------------------
    stages = {}
    dnat_route = None
    if fam["family"] != "TESSERA_NVFP4":
        nat = method._native
        if nat.gate_up is not None:
            gu = nat.gate_up(x, ids, w, preserve=True, apply_router_weight_on_input=False)
            g_n, u_n = gu[..., :INTER], gu[..., INTER:]
        else:
            g_n = nat.gate(x, ids, w, preserve=True, apply_router_weight_on_input=False)
            u_n = nat.up(x, ids, w, preserve=True, apply_router_weight_on_input=False)
        g_n = g_n.reshape(P, INTER)
        u_n = u_n.reshape(P, INTER)
        act_n = nwm._silu_and_mul(g_n, u_n, clamp_limit=clamp)          # bf16 [P, I]
        # the fused lane spells the staged down projection ``down_routes``
        # (its ``down`` is the prepared bundle); the compact adapter's bundle
        # is itself the callable.
        down_fn = getattr(nat, "down_routes", None) or nat.down
        down_n = down_fn(act_n, ids, w, route_input=True,
                         apply_router_weight_on_input=False, round_routes=True)   # [T, H]
    else:
        order, offsets, route_tokens_s, route_weights_s = a4_dispatch(ids, w, n_experts)
        from tessera.serving.native_a4 import a4_grouped_apply
        from vllm.model_executor.layers.fused_moe.activation import (
            ApplyMoEActivationConfig, apply_moe_activation)

        gs = a4_grouped_apply(x, layer.tessera_a4_gate_stack, layer.tessera_a4_gs13,
                              expert_offsets=offsets, route_ids=route_tokens_s, num_routes=P,
                              epilogues=layer.tessera_a4_gate_epilogues)
        us = a4_grouped_apply(x, layer.tessera_a4_up_stack, layer.tessera_a4_gs13,
                              expert_offsets=offsets, route_ids=route_tokens_s, num_routes=P,
                              epilogues=layer.tessera_a4_up_epilogues)
        cfg = ApplyMoEActivationConfig(
            clamp_limit=clamp, alpha=1.0, beta=0.0,
            activation_situ_beta=getattr(method.moe, "activation_situ_beta", None),
            activation_situ_linear_beta=getattr(method.moe, "activation_situ_linear_beta", None))
        gu_s = torch.cat([gs, us], dim=-1)
        act_s = torch.empty((P, INTER), dtype=gu_s.dtype, device=gu_s.device)
        apply_moe_activation(layer.activation, act_s, gu_s, activation_config=cfg)
        identity = torch.arange(P, dtype=torch.int32, device=x.device)
        dn_s = a4_grouped_apply(act_s, layer.tessera_a4_down_stack, layer.tessera_a4_gs2,
                                expert_offsets=offsets, route_ids=identity, num_routes=P,
                                epilogues=layer.tessera_a4_down_epilogues)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(P, device=order.device)
        g_n = gs.index_select(0, inv)
        u_n = us.index_select(0, inv)
        act_n = act_s.index_select(0, inv)
        dnat_route = dn_s.index_select(0, inv)                      # fp32 [P, H], unweighted
        comb = torch.zeros((T, HIDDEN), dtype=torch.float32, device=x.device)
        comb.index_add_(0, route_tokens_s.to(torch.int64), dn_s * route_weights_s[:, None])
        down_n = comb.to(torch.bfloat16)
    torch.cuda.synchronize()
    stages["gate"] = summarize((g_n.double() - g_ex).abs(), B1g, g_ex)
    stages["up"] = summarize((u_n.double() - u_ex).abs(), B1u, u_ex)
    f_tf, _sg, _uc = silu_f(g_n.double(), u_n.double(), clamp)
    if bf16_s1:
        B_act = (EPS_F + U16 * (1 + EPS_F)) * f_tf.abs() + 1e-300
    else:
        B_act = EPS_F * f_tf.abs() + 1e-300
    stages["activation"] = summarize((act_n.double() - f_tf).abs(), B_act, f_tf)

    # stage 3, teacher-forced: the native activation through the contract quantiser
    if fam["family"] == "TESSERA_FP8":
        aq, a2 = fp8_quant_rows(act_n)
        Q_tf = aq.double() * a2.double()
    elif fam["family"] == "TESSERA_BF16":
        Q_tf = act_n.double()
    else:
        _v, _s, Q_tf = nvfp4_quant_rows(act_n.to(torch.bfloat16), layer.tessera_a4_gs2)
    d_tf, S_d, _, _ = expert_gemm(Q_tf, route_expert, ref, "down_proj")
    if fam["family"] != "TESSERA_NVFP4":
        route_ex = d_tf * rw[:, None]
        pre = rw[:, None] * gamma(INTER, U_ACC) * S_d + F64_SLACK * rw[:, None] * S_d
        pre = pre + gamma(3, U32) * (route_ex.abs() + pre)
        Er = pre + U16 * (route_ex.abs() + pre)
        sum_ex = route_ex.reshape(T, TOP_K, HIDDEN).sum(1)
        Er3 = Er.reshape(T, TOP_K, HIDDEN)
        Esum = Er3.sum(1) + gamma(TOP_K, U32) * (route_ex.abs().reshape(T, TOP_K, HIDDEN).sum(1)
                                                 + Er3.sum(1))
        Eout = Esum + U16 * (sum_ex.abs() + Esum)
        stages["down_reduced"] = summarize((down_n.double() - sum_ex).abs(), Eout, sum_ex)
    else:
        assert dnat_route is not None
        Ed = gemm_bound(S_d, d_tf, INTER, 2)
        stages["down_route"] = summarize((dnat_route.double() - d_tf).abs(), Ed, d_tf)

    # ---------------- end to end: apply vs the independent pipeline -------
    recorder.take()
    out_n = method.apply(layer, x, w, ids, None, None)
    out_n2 = method.apply(layer, x, w, ids, None, None)
    torch.cuda.synchronize()
    routes_seen = recorder.take()
    pairs = sorted({(c["symbol"], c["decoder"]) for c in routes_seen})
    expected = expected_pair(fam, method)
    out["route_telemetry"] = {"pairs": [list(p) for p in pairs], "calls": len(routes_seen),
                              "contract": sorted({str(c["contract"]) for c in routes_seen}),
                              "expected_pair": list(expected),
                              "adapter": type(getattr(method, "_native", None)).__name__,
                              "pair_matches": pairs == [expected]}
    out["repeat_apply_max_abs_diff"] = float((out_n.float() - out_n2.float()).abs().max())
    out["apply_vs_staged_composition_max_abs_diff"] = float(
        (out_n.float() - down_n.float()).abs().max())

    if fam["family"] != "TESSERA_NVFP4":
        g_r = g_ex.to(torch.bfloat16)
        u_r = u_ex.to(torch.bfloat16)
        gf, uf = g_r.float(), u_r.float()
        gc = torch.clamp(gf, max=clamp)
        uc = torch.clamp(uf, min=-clamp, max=clamp)
        act_r = (torch.nn.functional.silu(gc) * uc).to(torch.bfloat16)
        dg = B1g + U16 * g_ex.abs()
        du = B1u + U16 * u_ex.abs()
    else:
        g_r = g_ex.float()
        u_r = u_ex.float()
        gc = torch.clamp(g_r, max=clamp)
        uc = torch.clamp(u_r, min=-clamp, max=clamp)
        act_r = (torch.nn.functional.silu(gc) * uc).to(torch.bfloat16)
        dg = B1g + U32 * g_ex.abs()
        du = B1u + U32 * u_ex.abs()
    f_r, silu_abs, u_abs = silu_f(g_r.double(), u_r.double(), clamp)
    Df = SILU_LIP * dg * (u_abs + du) + silu_abs * du
    D_act = Df + (EPS_F + U16 + EPS_F * U16) * (2 * f_r.abs() + Df)
    quant_info = {}
    if fam["family"] == "TESSERA_FP8":
        aq_r, a2_r = fp8_quant_rows(act_r)
        Q_r = aq_r.double() * a2_r.double()
        bq, cand = quant_bound_fp8_rows(act_r.double(), Q_r, a2_r.double().reshape(-1), D_act)
        quant_info = {"flip_candidate_elements": int(cand.sum()), "elements": int(cand.numel())}
    elif fam["family"] == "TESSERA_BF16":
        Q_r = act_r.double()
        bq = D_act
    else:
        gs2 = layer.tessera_a4_gs2
        _v, sf_r, Q_r = nvfp4_quant_rows(act_r, gs2)
        bq, el_cand, sf_cand, subn = quant_bound_nvfp4(act_r.double(), Q_r, sf_r, float(gs2), D_act)
        quant_info = {"flip_candidate_elements": int(el_cand.sum()), "elements": int(el_cand.numel()),
                      "scale_flip_candidate_blocks": int(sf_cand.sum()),
                      "subnormal_scale_blocks": int(subn.sum()), "blocks": int(sf_cand.numel()),
                      "reference_block_scales_subnormal": int((sf_r < 2.0 ** -6).sum()),
                      "reference_block_scales_zero": int((sf_r == 0).sum())}
    d_r, _S, S_n, Pq = expert_gemm(Q_r, route_expert, ref, "down_proj",
                                   extra_abs=Q_r.abs() + bq, extra=bq)
    assert Pq is not None and S_n is not None
    Dd = Pq + gamma(INTER, U_ACC) * S_n + F64_SLACK * S_n
    wcol = rw[:, None]
    if fam["family"] != "TESSERA_NVFP4":
        route_r = (d_r * wcol).to(torch.bfloat16)
        pre = wcol * Dd + gamma(3, U32) * wcol * (d_r.abs() + Dd)
        Er = pre + U16 * (2 * wcol * d_r.abs() + pre)
    else:
        route_r = d_r * wcol
        Er = wcol * Dd + gamma(3, U32) * wcol * (d_r.abs() + Dd)
    sum_r = route_r.double().reshape(T, TOP_K, HIDDEN).sum(1)
    out_r = sum_r.to(torch.bfloat16)
    Er3 = Er.reshape(T, TOP_K, HIDDEN)
    absr = (wcol * d_r.abs()).reshape(T, TOP_K, HIDDEN)
    Esum = Er3.sum(1) + gamma(TOP_K, U32) * (2 * absr.sum(1) + Er3.sum(1))
    Eout = Esum + U16 * (2 * sum_r.abs() + Esum)
    e2e = summarize((out_n.double() - out_r.double()).abs(), Eout, out_r.double())
    e2e["quantiser_perturbation"] = quant_info
    e2e["ulps"] = bf16_ulp_stats(out_n, out_r)
    e2e["bound_over_ref_max"] = float(Eout.max()) / max(float(out_r.double().abs().max()), 1e-300)
    e2e["discriminating"] = bool(e2e["bound_over_ref_max"] < 1.0)
    e2e["role"] = ("observation: worst-case propagation through the discontinuous "
                   "activation quantiser after a 4096-term accumulation is admitted by the "
                   "dtype constants but not tight; the pass criterion is the teacher-forced "
                   "stages plus the telemetry pair, and this block reports the end-to-end "
                   "difference in bf16 ulps")
    e2e["activation_perturbation_max"] = float(D_act.max())
    diff_max = float((out_n.float() - out_r.float()).abs().max())
    ref_max = float(out_r.float().abs().max())
    if fam["family"] != "TESSERA_NVFP4":
        tol = 5e-3 + 1e-2 * ref_max
        e2e["existing_screen"] = {
            "name": "tests/test_window_gemm_grouped.py::_tol (5e-3 + 1e-2*max|ref|), "
                    "a chosen screen, not a derived bound",
            "value": tol, "pass": diff_max < tol}
    else:
        rel = diff_max / max(ref_max, 1e-12)
        e2e["existing_screen"] = {
            "name": "tests/test_kernel_a4.py::check_two_stage_moe rel < 2e-2, "
                    "a chosen screen, not a derived bound",
            "rel": rel, "value": 2e-2, "pass": rel < 2e-2}
    out["stages_teacher_forced"] = stages
    out["end_to_end"] = e2e
    out["pass"] = bool(all(s["pass"] for s in stages.values())
                       and out["route_telemetry"]["pair_matches"])
    out["pass_criterion"] = "all teacher-forced stages within their dtype-derived bound AND the emitted launch pair equals the expected pair"
    return out, {"x": x, "ids": ids, "w": w, "out_r": out_r, "Eout": Eout}


# ---------------------------------------------------------------------------
# "before": the materialised arithmetic through vLLM's own fused-MoE kernels
# ---------------------------------------------------------------------------

def _param(layer, name, tensor):
    if hasattr(layer, name):
        delattr(layer, name)
    layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


def build_before(fkey, fam, scheme, blobs, scales, n_experts, cfg, clamp, prefix):
    """Stock tiles from ``materialize_stock`` handed to the runtime's own
    modular fused-MoE kernel for the family.  Returns (callable, info)."""
    from tessera.stock import materialize_stock
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    _declared, roles = parse_roles(scheme, prefix)
    layer = make_layer(cfg, n_experts, clamp)
    info = {}
    t0 = time.time()
    dev = "cuda"
    E = n_experts
    if fam["family"] == "TESSERA_FP8":
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            convert_to_fp8_moe_kernel_format, make_fp8_moe_kernel, make_fp8_moe_quant_config,
            select_fp8_moe_backend)
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTokenSym, kFp8StaticChannelSym)

        w13 = torch.empty((E, 2 * INTER, HIDDEN), dtype=torch.float8_e4m3fn, device=dev)
        s13 = torch.empty((E, 2 * INTER, 1), dtype=torch.float32, device=dev)
        w2 = torch.empty((E, HIDDEN, INTER), dtype=torch.float8_e4m3fn, device=dev)
        s2 = torch.empty((E, HIDDEN, 1), dtype=torch.float32, device=dev)
        for e in range(E):
            for p, (wt, st, lo, hi) in (("gate_proj", (w13, s13, 0, INTER)),
                                        ("up_proj", (w13, s13, INTER, 2 * INTER)),
                                        ("down_proj", (w2, s2, 0, HIDDEN))):
                pu = parsed_unit(blobs[p][e], roles[p], f"{prefix} {p} expert {e}", dev)
                t = materialize_stock(pu.unit, pu.forests, pu.code)
                wt[e, lo:hi] = t["weight"].to(dev)
                st[e, lo:hi] = t["weight_scale"].to(dev).reshape(-1, 1)
                del pu, t
        backend, experts_cls = select_fp8_moe_backend(
            config=cfg, weight_key=kFp8StaticChannelSym, activation_key=kFp8DynamicTokenSym,
            allow_vllm_cutlass=True)
        info["backend"] = str(getattr(backend, "name", backend))
        info["experts_cls"] = getattr(experts_cls, "__name__", str(experts_cls))
        for name, t in (("w13_weight", w13), ("w2_weight", w2),
                        ("w13_weight_scale", s13), ("w2_weight_scale", s2)):
            _param(layer, name, t)
        w13c, w2c, s13c, s2c = convert_to_fp8_moe_kernel_format(
            fp8_backend=backend, layer=layer, w13=w13, w2=w2, w13_scale=s13, w2_scale=s2,
            w13_input_scale=None, w2_input_scale=None)
        for name, t in (("w13_weight", w13c), ("w2_weight", w2c),
                        ("w13_weight_scale", s13c), ("w2_weight_scale", s2c)):
            _param(layer, name, t)
        quant = make_fp8_moe_quant_config(
            fp8_backend=backend, w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            a1_scale=None, a2_scale=None, per_act_token_quant=True, per_out_ch_quant=True,
            block_shape=None, gemm1_alpha=None, gemm1_beta=None, swiglu_limit=clamp, layer=layer)
        kernel = make_fp8_moe_kernel(moe_quant_config=quant, moe_config=cfg,
                                     experts_cls=experts_cls, fp8_backend=backend,
                                     routing_tables=None)
    elif fam["family"] == "TESSERA_BF16":
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend, convert_to_unquantized_kernel_format,
            make_unquantized_moe_kernel, select_unquantized_moe_backend)

        w13 = torch.empty((E, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device=dev)
        w2 = torch.empty((E, HIDDEN, INTER), dtype=torch.bfloat16, device=dev)
        for e in range(E):
            for p, (wt, lo, hi) in (("gate_proj", (w13, 0, INTER)),
                                    ("up_proj", (w13, INTER, 2 * INTER)),
                                    ("down_proj", (w2, 0, HIDDEN))):
                pu = parsed_unit(blobs[p][e], roles[p], f"{prefix} {p} expert {e}", dev)
                t = materialize_stock(pu.unit, pu.forests, pu.code)
                wt[e, lo:hi] = t["weight"].to(dev)
                del pu, t
        quant = FusedMoEQuantConfig.make(gemm1_clamp_limit=clamp)
        try:
            backend, experts_cls = select_unquantized_moe_backend(moe_config=cfg)
            info["selected_backend"] = str(getattr(backend, "name", backend))
            info["selected_experts_cls"] = getattr(experts_cls, "__name__", str(experts_cls))
            w13c, w2c = convert_to_unquantized_kernel_format(backend, cfg, w13, w2)
            kernel = make_unquantized_moe_kernel(quant_config=quant, moe_config=cfg,
                                                 backend=backend, experts_cls=experts_cls,
                                                 routing_tables=None)
            info["backend"] = info["selected_backend"]
            info["experts_cls"] = info["selected_experts_cls"]
        except Exception as exc:  # noqa: BLE001 -- recorded, then the named fallback
            info["auto_selection_failed"] = f"{type(exc).__name__}: {exc}"[:600]
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

            backend, experts_cls = UnquantizedMoeBackend.TRITON, TritonExperts
            w13c, w2c = w13, w2
            kernel = make_unquantized_moe_kernel(quant_config=quant, moe_config=cfg,
                                                 backend=backend, experts_cls=experts_cls,
                                                 routing_tables=None)
            info["backend"] = "TRITON (explicit fallback)"
            info["experts_cls"] = "TritonExperts"
        _param(layer, "w13_weight", w13c)
        _param(layer, "w2_weight", w2c)
    else:
        import dataclasses as _dc

        from tessera.fused import shared_lut_global
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            convert_to_nvfp4_moe_kernel_format, make_nvfp4_moe_kernel,
            make_nvfp4_moe_quant_config, select_nvfp4_moe_backend)
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kNvfp4Dynamic, kNvfp4Static)

        w13 = torch.empty((E, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=dev)
        s13 = torch.empty((E, 2 * INTER, HIDDEN // 16), dtype=torch.uint8, device=dev)
        w2 = torch.empty((E, HIDDEN, INTER // 2), dtype=torch.uint8, device=dev)
        s2 = torch.empty((E, HIDDEN, INTER // 16), dtype=torch.uint8, device=dev)
        g13 = torch.empty((E,), dtype=torch.float32)
        g2 = torch.empty((E,), dtype=torch.float32)
        for e in range(E):
            gpu = parsed_unit(blobs["gate_proj"][e], roles["gate_proj"], f"{prefix} gate {e}", dev)
            upu = parsed_unit(blobs["up_proj"][e], roles["up_proj"], f"{prefix} up {e}", dev)
            shared, moved = shared_lut_global(
                [gpu.unit.scale_lut, upu.unit.scale_lut],
                [float(gpu.unit.scale_global), float(upu.unit.scale_global)],
                ["gate_proj", "up_proj"])
            for pu, table, lo, hi in ((gpu, moved[0], 0, INTER), (upu, moved[1], INTER, 2 * INTER)):
                unit = _dc.replace(pu.unit, scale_lut=table.cpu() if pu.unit.scale_lut.device.type == "cpu"
                                   else table, scale_global=float(shared))
                t = materialize_stock(unit, pu.forests, pu.code)
                w13[e, lo:hi] = t["weight_packed"].to(dev)
                s13[e, lo:hi] = t["weight_scale"].view(torch.uint8).to(dev)
            g13[e] = float(shared)
            dpu = parsed_unit(blobs["down_proj"][e], roles["down_proj"], f"{prefix} down {e}", dev)
            t = materialize_stock(dpu.unit, dpu.forests, dpu.code)
            w2[e] = t["weight_packed"].to(dev)
            s2[e] = t["weight_scale"].view(torch.uint8).to(dev)
            g2[e] = 1.0 / float(t["weight_global_scale"].reshape(-1)[0])
            del gpu, upu, dpu, t
        a13 = torch.tensor([[1.0 / scales["gate_proj"][e], 1.0 / scales["up_proj"][e]]
                            for e in range(E)], dtype=torch.float32, device=dev)
        a2 = torch.tensor([1.0 / scales["down_proj"][e] for e in range(E)],
                          dtype=torch.float32, device=dev)
        backend, experts_cls = select_nvfp4_moe_backend(
            config=cfg, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic)
        info["backend"] = str(getattr(backend, "name", backend))
        info["experts_cls"] = getattr(experts_cls, "__name__", str(experts_cls))
        tiles = {"w13_weight": w13, "w13_weight_scale": s13.view(torch.float8_e4m3fn),
                 "w13_weight_scale_2": g13.to(dev).reshape(E, 1).expand(E, 2).contiguous(),
                 "w13_input_scale": a13, "w2_weight": w2,
                 "w2_weight_scale": s2.view(torch.float8_e4m3fn),
                 "w2_weight_scale_2": g2.to(dev), "w2_input_scale": a2}
        for name, t in tiles.items():
            _param(layer, name, t)
        (w13c, s13c, g13c, a13c, w2c, s2c, g2c, a2c) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=backend, layer=layer, w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2[:, 0].contiguous(),
            a13_scale=layer.w13_input_scale, w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale, w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale, is_act_and_mul=True, use_a16=False)
        for name, t in (("w13_weight", w13c), ("w13_weight_scale", s13c),
                        ("w13_weight_scale_2", g13c), ("w13_input_scale", a13c),
                        ("w2_weight", w2c), ("w2_weight_scale", s2c),
                        ("w2_weight_scale_2", g2c), ("w2_input_scale", a2c)):
            if t is not None:
                _param(layer, name, t)
        quant = make_nvfp4_moe_quant_config(
            backend=backend, w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2, w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale, a2_scale=layer.w2_input_scale,
            swiglu_limit=clamp, swiglu_alpha=None, swiglu_beta=None, layer=layer, use_a16=False)
        kernel = make_nvfp4_moe_kernel(moe_quant_config=quant, moe_config=cfg,
                                       experts_cls=experts_cls, backend=backend,
                                       routing_tables=None)
        experts_obj = getattr(kernel, "fused_experts", None) or getattr(kernel, "experts", None)
        if experts_obj is not None and hasattr(experts_obj, "process_weights_after_loading"):
            experts_obj.process_weights_after_loading(layer)
            info["experts_process_weights_after_loading"] = True
    torch.cuda.synchronize()
    info["build_seconds"] = round(time.time() - t0, 3)

    def run(x, w, ids):
        return kernel.apply(x, layer.w13_weight, layer.w2_weight, w, ids,
                            activation=MoEActivation.SILU, global_num_experts=E,
                            expert_map=None, apply_router_weight_on_input=False)
    return run, info, layer


# ---------------------------------------------------------------------------
# power sampling and profiling
# ---------------------------------------------------------------------------

class PowerSampler(threading.Thread):
    def __init__(self, hz=10.0):
        super().__init__(daemon=True)
        self.period = 1.0 / hz
        self.samples = []
        self.stop_flag = False
        self.source = None
        self._read = None
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(h)
            self._read = lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            self.source = "pynvml.nvmlDeviceGetPowerUsage"
        except Exception:  # noqa: BLE001
            def _smi():
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                return float(out.strip().splitlines()[0])
            try:
                _smi()
                self._read = _smi
                self.source = "nvidia-smi --query-gpu=power.draw"
            except Exception as exc:  # noqa: BLE001
                self.source = f"unavailable: {exc}"

    def run(self):
        while not self.stop_flag and self._read is not None:
            t = time.time()
            try:
                self.samples.append((t, float(self._read())))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(max(0.0, self.period - (time.time() - t)))

    def window(self, t0, t1):
        vals = [w for t, w in self.samples if t0 <= t <= t1]
        if not vals:
            return {"samples": 0}
        return {"samples": len(vals), "mean_w": sum(vals) / len(vals), "peak_w": max(vals),
                "min_w": min(vals), "mean_fraction_of_140w_envelope": sum(vals) / len(vals) / 140.0}


def kernel_table(prof, limit=25):
    rows = []
    for evt in prof.key_averages():
        dev = getattr(evt, "self_device_time_total", None)
        if dev is None:
            dev = getattr(evt, "self_cuda_time_total", 0.0)
        if not dev:
            continue
        rows.append({"name": evt.key, "self_device_us_total": float(dev), "count": int(evt.count),
                     "self_device_us_per_call": float(dev) / max(int(evt.count), 1)})
    rows.sort(key=lambda r: -r["self_device_us_total"])
    return rows[:limit]


def netdata_window(t0, t1):
    """Retain exact-window samples, not the dashboard's current reading."""
    base = os.environ.get("NETDATA_URL", "http://127.0.0.1:19999")

    def get(path, query=None):
        url = base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)

    charts = get("/api/v1/charts")["charts"]
    wanted = {key: value for key, value in charts.items()
              if value.get("context") == "nvidia_smi.gpu_power_draw"
              or key == "system.cpu"
              or key.startswith(("system.cpu_", "system.memory_", "system.io_"))
              and "pressure" in key}
    result = {"source": base, "utc_start": t0, "utc_end": t1, "charts": {}}
    for key, chart in wanted.items():
        data = get("/api/v1/data", {"chart": key, "after": math.floor(t0),
                                   "before": math.ceil(t1), "points": 0,
                                   "format": "json"})
        samples = [row for row in data["data"] if t0 <= row[0] <= t1]
        result["charts"][key] = {"context": chart.get("context"),
                                   "units": chart.get("units"),
                                   "labels": data["labels"], "data": samples}
    return result


def profile_leg(name, fn, args, sampler, iters_wall, iters_prof, out_dir):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters_wall):
        fn()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters_wall
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters_prof):
            fn()
        torch.cuda.synchronize()
    try:
        table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
    except Exception:  # noqa: BLE001
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    (out_dir / f"{name}.profiler_table.txt").write_text(table)
    kernels = kernel_table(prof)
    prof.export_chrome_trace(str(out_dir / f"{name}.trace.json"))
    del prof
    time.sleep(args.idle_s)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.power_s:
        fn()
        n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    t1 = time.time()
    return {"leg": name, "ms_per_forward_cuda_events": ms, "iters_wall": iters_wall,
            "iters_profiled": iters_prof, "top_kernels_by_self_device_time": kernels,
            "profiler_table": f"{name}.profiler_table.txt",
            "profiler_trace": f"{name}.trace.json",
            "netdata_window": netdata_window(t0, t1),
            "power_window": {"utc_start": t0, "utc_end": t1,
                             "utc_start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                             "utc_end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)),
                             "iters": n, "forwards_per_s": n / (t1 - t0),
                             "in_process_power": sampler.window(t0 + 1.0, t1)}}


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def provenance(args):
    here = Path(__file__).resolve().parents[1]
    return {"host": os.environ.get("HOST_NAME") or socket.gethostname(),
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "vllm": __import__("vllm").__version__,
            "image": args.image, "image_tag": "a5424378-mtpmap1",
            "image_named_in_issue": "a5424378 (tag of the same serving image lineage)",
            "tessera_head": args.tessera_head,
            "tessera_worktree_state": args.tessera_state,
            "src_tree_sha256": src_tree_digest(here),
            "pb_action": os.environ.get("PRISMABUILD_ACTION_KEY") or os.environ.get("PB_ACTION_KEY"),
            "argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_oracle(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    experts = [int(e) for e in args.oracle_experts.split(",")] if "," in args.oracle_experts \
        else [i * (288 // int(args.oracle_experts)) for i in range(int(args.oracle_experts))]
    E = len(experts)
    report = {"schema": "tessera.routed_pair_oracle.v1", "issue": "RobTand/tessera#604",
              "provenance": provenance(args), "layer": args.layer, "experts": experts,
              "shapes": {"hidden": HIDDEN, "moe_intermediate": INTER, "top_k": TOP_K,
                         "experts_loaded": E, "served_experts": 288},
              "swiglu_limit": args.clamp,
              "swiglu_limit_source": "docs/tessera-serving-and-moe-contract.md 9.1: "
                                     "GLM-5.3-Flash config.json sets swiglu_limit 10.0",
              "input": {"x": f"N(0, {args.sigma}^2) bf16", "routing": "top-8 distinct of the "
                        "loaded experts per token, weights U(0.1,1.1) renormalised to 1",
                        "apply_router_weight_on_input": False},
              "bound_constants": BOUND_CONSTANTS, "families": {}}
    recorder = RouteRecorder()
    recorder.install()
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts"
    for fkey in args.families.split(","):
        fam = FAMILIES[fkey]
        entry = {"payload_family": fam["payload"], "rung": fam["rung"],
                 "scheme_family": fam["family"], "expected_launch_pair": list(fam["pair"])}
        report["families"][fkey] = entry
        try:
            log(fkey, "loading wires")
            blobs, files = load_wires(args, fam, experts)
            entry["wires"] = files
            scales = None
            if fam["family"] == "TESSERA_NVFP4":
                scales, meta = load_input_scales(args, experts)
                entry["input_scales"] = {"file": args.scales, "metadata": meta,
                                         "per_expert": {p: scales[p] for p in PROJ}}
            scheme, facts = scheme_for(fam, blobs, E)
            entry["scheme"] = scheme
            entry["wire_facts"] = facts
            cfg = moe_config(E, args.clamp)
            log(fkey, "building native method")
            layer, method, info = build_after(fam, scheme, blobs, scales, E, cfg, args.clamp, prefix)
            entry["native"] = info
            entry["expected_launch_pair"] = list(expected_pair(fam, method))
            entry["compact_launch_pair"] = list(fam["pair"])
            if fam["family"] == "TESSERA_NVFP4":
                entry["native"]["note"] = (
                    "gs13/gs2 are the route's own reduction over the LOADED experts "
                    f"({E} of 288), so the scalar differs from a 288-expert serve; the "
                    "oracle tests arithmetic identity under the contract's scalar")
            log(fkey, "materialising reference", info)
            ref, secs = reference_weights(fam, scheme, blobs, E, prefix)
            entry["reference_materialise_seconds"] = secs
            before = None
            if not args.skip_before:
                try:
                    log(fkey, "building before leg")
                    before, binfo, _blayer = build_before(fkey, fam, scheme, blobs, scales, E, cfg,
                                                          args.clamp, prefix)
                    entry["before_leg"] = binfo
                except Exception as exc:  # noqa: BLE001
                    entry["before_leg"] = {"error": f"{type(exc).__name__}: {exc}",
                                           "traceback": traceback.format_exc()[-4000:]}
                    log(fkey, "before leg failed", exc)
            entry["cases"] = []
            twin = f16_twin(method)
            if twin is not None:
                entry["f16_twin"] = {"library": twin.library, "launch_pair": list(twin.launch_pair),
                                     "shares_bundles_with_native": True}
            for m in [int(v) for v in args.m.split(",")]:
                x, ids, w = make_inputs(m, E, args.seed + m, args.sigma)
                log(fkey, "oracle M", m)
                case, keep = oracle_case(fkey, fam, layer, method, ref, x, ids, w, args.clamp,
                                         E, recorder)
                if twin is not None:
                    case["e4m3_instruction_vs_f16_twin"] = instruction_pair_case(
                        method._native, twin, x, ids, w, args.clamp, keep)
                if before is not None:
                    try:
                        ob = before(x, w, ids)
                        torch.cuda.synchronize()
                        diff = (ob.double() - keep["out_r"].double()).abs()
                        case["before_leg_vs_reference"] = {
                            "max_abs_diff": float(diff.max()),
                            "max_diff_over_e2e_bound": float((diff / keep["Eout"]).max()),
                            "within_e2e_bound": bool((diff <= keep["Eout"]).all()),
                            "ulps": bf16_ulp_stats(ob, keep["out_r"]),
                            "note": "sanity only: the stock kernel's own activation-scale "
                                    "handling and placement may differ from the native "
                                    "contract; not a pass criterion"}
                    except Exception as exc:  # noqa: BLE001
                        case["before_leg_vs_reference"] = {"error": f"{type(exc).__name__}: {exc}"}
                entry["cases"].append(case)
                log(fkey, "M", m, "pass", case["pass"],
                    {k: v["max_diff_over_bound"] for k, v in case["stages_teacher_forced"].items()},
                    "e2e", case["end_to_end"]["max_diff_over_bound"])
            entry["pass"] = all(c["pass"] for c in entry["cases"])
            del layer, method, ref, before
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-6000:]
            entry["pass"] = False
            log(fkey, "FAILED", exc)
            traceback.print_exc()
        (out_dir / "oracle.json").write_text(json.dumps(report, indent=1, default=str))
    report["pass"] = all(f.get("pass") for f in report["families"].values())
    sub = []
    for fk, f in report["families"].items():
        for c in f.get("cases", []):
            q = c.get("end_to_end", {}).get("quantiser_perturbation", {})
            if "reference_block_scales_subnormal" in q:
                sub.append({"family": fk, "M": c["M"], "down_input_blocks": q["blocks"],
                            "subnormal_e4m3_block_scales": q["reference_block_scales_subnormal"],
                            "zero_block_scales": q["reference_block_scales_zero"],
                            "gs2": f.get("native", {}).get("gs2")})
    if sub:
        report["finding_down_input_subnormal_block_scales"] = {
            "cases": sub,
            "what": "NVFP4 block scale = gs2*amax_block/6 in E4M3; below 2^-6 it is subnormal "
                    "(fewer significant bits). Native and reference agree bit-for-bit, so this "
                    "is a precision property of the static-scale contract, not an oracle "
                    "disagreement.",
            "caveats": "synthetic N(0, sigma^2) hidden states; gs2 reduced over the loaded "
                       "experts only (the route's own min-over-experts reduction)"}
    (out_dir / "oracle.json").write_text(json.dumps(report, indent=1, default=str))
    log("oracle pass", report["pass"])
    return 0 if report["pass"] else 3


def run_profile(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(args.profile_experts)
    experts = list(range(n)) if n == 288 else [i * (288 // n) for i in range(n)]
    E = len(experts)
    sampler = PowerSampler(hz=10.0)
    sampler.start()
    report = {"schema": "tessera.routed_pair_profiles.v1", "issue": "RobTand/tessera#604",
              "provenance": provenance(args), "layer": args.layer, "experts_loaded": E,
              "experts": "all 288" if E == 288 else experts,
              "regime": "eager (no CUDA graph), resident, TP1, one routed layer",
              "power_sampler": sampler.source, "envelope_w": 140.0,
              "idle_windows": [], "families": {}}
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts"
    time.sleep(args.idle_s)
    t0 = time.time()
    time.sleep(max(args.idle_s, 5.0))
    report["idle_windows"].append({"utc_start": t0, "utc_end": time.time(),
                                   "in_process_power": sampler.window(t0, time.time())})
    for fkey in args.families.split(","):
        fam = FAMILIES[fkey]
        entry = {"payload_family": fam["payload"], "rung": fam["rung"], "legs": {}}
        report["families"][fkey] = entry
        try:
            log(fkey, "loading", E, "experts")
            blobs, files = load_wires(args, fam, experts)
            entry["wire_bytes_total"] = sum(f["bytes"] for f in files)
            entry["wires_sha256_of_list"] = sha256_bytes(json.dumps(
                [(f["path"], f["sha256"]) for f in files]).encode())
            scales = None
            if fam["family"] == "TESSERA_NVFP4":
                scales, _meta = load_input_scales(args, experts)
            scheme, _facts = scheme_for(fam, blobs, E)
            cfg = moe_config(E, args.clamp)
            legs = {}
            layer, method, info = build_after(fam, scheme, blobs, scales, E, cfg, args.clamp, prefix)
            entry["after_build"] = info
            legs["after"] = lambda x, w, ids, _m=method, _l=layer: _m.apply(_l, x, w, ids, None, None)
            compact = compact_twin(method)
            if compact is not None:
                # tessera#640: the lane this build replaces, over the same
                # resident bundles, so after/legacy differ in the kernel only.
                entry["legacy_build"] = {"adapter": type(compact).__name__,
                                         "launch_pair": list(compact.launch_pair),
                                         "shares_bundles_with_after": True}
                legs["legacy"] = lambda x, w, ids, _c=compact, _lim=args.clamp: _c(
                    x, ids, w, swiglu_limit=_lim, apply_router_weight_on_input=False)
            try:
                before, binfo, _bl = build_before(fkey, fam, scheme, blobs, scales, E, cfg,
                                                  args.clamp, prefix)
                entry["before_build"] = binfo
                legs["before"] = before
            except Exception as exc:  # noqa: BLE001
                entry["before_build"] = {"error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()[-4000:]}
                log(fkey, "before build failed", exc)
            del blobs
            for m in [int(v) for v in args.m.split(",")]:
                x, ids, w = make_inputs(m, E, args.seed + m, args.sigma)
                for other in ("before", "legacy"):
                    if other not in legs:
                        continue
                    key = f"after_vs_{other}"
                    try:
                        a = legs["after"](x, w, ids)
                        b = legs[other](x, w, ids)
                        torch.cuda.synchronize()
                        entry.setdefault(key, {})[str(m)] = {
                            "max_abs_diff": float((a.float() - b.float()).abs().max()),
                            "max_abs_after": float(a.float().abs().max())}
                    except Exception as exc:  # noqa: BLE001
                        entry.setdefault(key, {})[str(m)] = {"error": str(exc)}
                for leg, fn in legs.items():
                    name = f"{fkey}_{leg}_M{m}"
                    log("profiling", name)
                    try:
                        iters_wall = 200 if m == 1 else 30
                        iters_prof = 20 if m == 1 else 5
                        rec = profile_leg(name, lambda _f=fn, _x=x, _w=w, _ids=ids: _f(_x, _w, _ids), args, sampler,
                                          iters_wall, iters_prof, out_dir)
                        entry["legs"][name] = rec
                        log(name, round(rec["ms_per_forward_cuda_events"], 4), "ms",
                            rec["power_window"]["in_process_power"])
                    except Exception as exc:  # noqa: BLE001
                        entry["legs"][name] = {"error": f"{type(exc).__name__}: {exc}",
                                               "traceback": traceback.format_exc()[-4000:]}
                        log(name, "FAILED", exc)
                    (out_dir / "profiles.json").write_text(json.dumps(report, indent=1, default=str))
            entry["legs_built"] = sorted(legs)
            del legs, layer, method, compact
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-6000:]
            traceback.print_exc()
        (out_dir / "profiles.json").write_text(json.dumps(report, indent=1, default=str))
    t0 = time.time()
    time.sleep(max(args.idle_s, 5.0))
    report["idle_windows"].append({"utc_start": t0, "utc_end": time.time(),
                                   "in_process_power": sampler.window(t0, time.time())})
    sampler.stop_flag = True
    requested_m = [int(value) for value in args.m.split(",")]
    report["pass"] = bool(report["families"]) and bool(requested_m)
    for family, entry in report["families"].items():
        built = entry.get("legs_built") or ["after", "before"]
        if "before" not in built:
            built = [*built, "before"]      # a before leg that failed to build is a failure
        expected = {f"{family}_{leg}_M{m}" for leg in built for m in requested_m}
        complete = (not entry.get("error")
                    and set(entry["legs"]) == expected
                    and all(not leg.get("error") for leg in entry["legs"].values()))
        entry["pass"] = complete
        report["pass"] = report["pass"] and complete
    (out_dir / "profiles.json").write_text(json.dumps(report, indent=1, default=str))
    return 0 if report["pass"] else 3


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--mode", choices=("oracle", "profile"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wire-root", default=WIRE_ROOT)
    ap.add_argument("--scales", default=SCALES)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--families", default="e4m3,bf16,e2m1")
    ap.add_argument("--oracle-experts", default="16",
                    help="a count (evenly spread over 288) or a comma list of expert ids")
    ap.add_argument("--profile-experts", default="288")
    ap.add_argument("--m", default="1,64,512")
    ap.add_argument("--clamp", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=604)
    ap.add_argument("--skip-before", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--power-s", type=float, default=20.0)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--image", default=os.environ.get("ORACLE_IMAGE", ""))
    ap.add_argument("--tessera-head", default=os.environ.get("TESSERA_HEAD", ""))
    ap.add_argument("--tessera-state", default=os.environ.get("TESSERA_STATE", ""))
    ap.add_argument("--rung", default=None,
                    help="wire rung to read for every selected family (e.g. R832) in place of the "
                         "family's pinned one; tessera#694 reads the mixed-rate E4M3 rungs "
                         "R832/R928/R960/R1088 of the same cache")
    args = ap.parse_args()
    if args.rung:
        for key in args.families.split(","):
            FAMILIES[key] = dict(FAMILIES[key], rung=str(args.rung))
    torch.manual_seed(args.seed)
    log("host", os.environ.get("HOST_NAME"), "device", torch.cuda.get_device_name(0))
    if args.mode == "oracle":
        return run_oracle(args)
    return run_profile(args)


if __name__ == "__main__":
    sys.exit(main())
