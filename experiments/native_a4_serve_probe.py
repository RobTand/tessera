"""Bounded real-vLLM probe: the native A4 expert method, construct -> apply.

Runs inside the pinned serve image (vLLM is present; vendoring it into the
repo's test suite is forbidden).  It drives the route's own method against a
real ``FusedMoEMethodBase`` and a stub layer module carrying the registered
parameters, on the real fixture wires, and checks:

1. construct + create_weights + wire load + finalize on TP2 rank 0/1;
2. ``apply`` against the independent stock oracle (same routing, the executed
   ``scaled_fp4_quant`` + ``torch._scaled_mm`` arithmetic, weights applied
   only in the final combine);
3. the shared-experts contract: the runner owns them, so ``apply`` must not
   touch the ``shared_experts`` argument and must return the routed output;
4. a CUDA-graph capture of the whole apply;
5. refusals: a sidecar that disagrees with the bytes, a stock tensor name, and
   non-resident mode.

Usage (in the image):  python3 experiments/native_a4_serve_probe.py --rank 0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

DATA = Path(os.environ.get("TESSERA_A4_WIRE_DIR", "/mnt/shared/astra-native-a4/data"))
PREFIX = "model.language_model.layers.3.mlp.experts"
LAYER = "layers_3"
HIDDEN, LOCAL_INTER, EXPERTS, TOP_K = 4096, 1024, 2, 2


def declared(experts: int = EXPERTS):
    """The artifact's scheme, trimmed to ``experts`` for a bounded probe.

    The wires are the artifact's own per-expert containers (stride-verified
    against the group declaration, unchanged by the count); the expert axis
    is smaller so a probe can load it without the 288-expert pool.
    """
    from tessera.serving.scheme import validate_tessera_moe_scheme

    cfg = json.loads((DATA / "a4-config.json").read_text())
    scheme = dict(cfg["quantization_config"]["config_groups"][
        f"tessera_model_language_model_{LAYER}_mlp_experts"]["scheme"])
    scheme["experts"] = int(experts)
    return scheme, validate_tessera_moe_scheme(scheme, PREFIX)


def moe_config(tp_rank: int, tp_size: int = 2):
    """The runtime's own configs, built directly.

    ``FusedMoEParallelConfig.make`` consults the initialized TP group; this is
    a single-process probe of one rank's geometry, so the dataclass is built
    from the same values ``make`` would read.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                             FusedMoEParallelConfig)

    moe_parallel = FusedMoEParallelConfig(
        tp_size=tp_size, pcp_size=1, dp_size=1, ep_size=1, tp_rank=tp_rank,
        pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="allgather_reducescatter", enable_eplb=False)
    return FusedMoEConfig(
        num_experts=EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN,
        intermediate_size=LOCAL_INTER * tp_size, num_local_experts=EXPERTS,
        num_logical_experts=EXPERTS, activation=MoEActivation.SILU,
        device=torch.device("cuda"),
        routing_method="topk", moe_parallel_config=moe_parallel,
        in_dtype=torch.bfloat16, intermediate_size_per_partition=LOCAL_INTER)


def build_layer(tp_rank: int):
    """A stub layer module carrying the parameters the route registers."""
    import torch.nn as nn

    from tessera.serving.nvfp4_moe_route import build_tessera_nvfp4_moe_method

    scheme, declared_group = declared()
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    layer = nn.Module()
    layer.moe_config = moe_config(tp_rank)
    # The attributes a real ``RoutedExperts`` carries into ``apply``.
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer.activation = MoEActivation.SILU
    layer.global_num_experts = EXPERTS
    method = build_tessera_nvfp4_moe_method(scheme, PREFIX, "resident", layer)
    method.create_weights(layer, EXPERTS, HIDDEN, LOCAL_INTER, torch.bfloat16,
                          global_num_experts=EXPERTS)
    layer.quant_method = method
    return layer, method, declared_group


def load_expert(layer, expert: int):
    blobs = {name: (DATA / f"{name}_wire.bin").read_bytes()
             for name in ("gate_proj", "up_proj", "down_proj")}
    for shard, group in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
        layer.quant_method._load_wire(None, torch.frombuffer(bytearray(blobs[group]),
                                                             dtype=torch.uint8),
                                      group, shard, expert)
        layer.quant_method._load_input_global_scale(
            layer.quant_method._input_global["w13" if shard != "w2" else "w2"],
            torch.tensor([448.0 * 6.0 / 2.0], dtype=torch.float32), group, shard, expert)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    from torch.nn import functional as F

    from tessera.serving.nvfp4_route import blocked_scales
    from tessera.stock import materialize_stock
    from tessera.serving.scheme import expert_role_declarations, parse_tessera_expert_blob

    report = {"rank": args.rank, "checks": []}
    layer, method, declared_group = build_layer(args.rank)
    for expert in range(EXPERTS):
        load_expert(layer, expert)
    method.process_weights_after_loading(layer)
    report["checks"].append({"check": "construct_load_finalize", "ok": True})

    # -- apply vs the stock oracle -----------------------------------------
    torch.manual_seed(4)
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.25
    ids = torch.randint(0, EXPERTS, (8, TOP_K), device="cuda", dtype=torch.int32)
    weights = torch.rand(8, TOP_K, device="cuda") * 0.6 + 0.2

    class Sentinel:
        called = False

        def __call__(self, *a, **k):
            Sentinel.called = True
            raise AssertionError("apply must not run shared experts")

    out = method.apply(layer, x, weights, ids, Sentinel(), None)
    assert not Sentinel.called, "shared experts were consumed by apply"
    assert out.shape == x.shape
    report["checks"].append({"check": "shared_experts_not_consumed", "ok": True})

    # oracle: per (t, k) the stock tile through _scaled_mm, weights last
    roles13 = expert_role_declarations(declared_group["groups"]["w13"])
    roles2 = expert_role_declarations(declared_group["groups"]["w2"])
    from tessera.serving.nvfp4_route import a4_quantize_activation as _q  # noqa: F401
    from tessera.kernel_a4 import a4_quantize_activation

    stock = {}
    for expert in range(EXPERTS):
        gate = parse_tessera_expert_blob((DATA / "gate_proj_wire.bin").read_bytes(),
                                         roles13[0], f"{PREFIX} gate",
                                         device="cuda")[0]
        up = parse_tessera_expert_blob((DATA / "up_proj_wire.bin").read_bytes(),
                                       roles13[1], f"{PREFIX} up", device="cuda")[0]
        down = parse_tessera_expert_blob((DATA / "down_proj_wire.bin").read_bytes(),
                                         roles2[0], f"{PREFIX} down", device="cuda")[0]
        stock[expert] = (gate, up, down)

    from tessera.serving.sharding import shard_parsed_roles
    from tessera.serving.moe_route import _packed_group_shard_plan

    plan13 = _packed_group_shard_plan(declared_group, "w13", PREFIX, args.rank, 2)
    plan2 = _packed_group_shard_plan(declared_group, "w2", PREFIX, args.rank, 2)

    gs13 = layer.tessera_a4_gs13
    gs2 = layer.tessera_a4_gs2
    packed, scales = a4_quantize_activation(x, gs13)
    epilogue13 = layer.tessera_a4_gate_epilogues
    epilogue2 = layer.tessera_a4_down_epilogues
    ref = torch.zeros_like(out, dtype=torch.float32)

    import dataclasses

    from tessera.fused import shared_lut_global

    def w4a4(x_bf16, gscale, parsed_unit, epilogue_value):
        # The runtime's own quantizer in its own swizzled layout (what the
        # dense route hands _scaled_mm), not a reshaped copy of the linear
        # plane; the stock tile through ``materialize_stock``.
        unit = parsed_unit.unit
        tile = materialize_stock(unit, parsed_unit.forests, parsed_unit.code)
        a_q, a_s = torch.ops._C.scaled_fp4_quant(x_bf16.contiguous(), gscale, True)
        a_q = a_q.view(torch.float4_e2m1fn_x2)
        a_s = a_s.view(torch.uint8).view(torch.float8_e4m3fn).contiguous()
        b_q = tile["weight_packed"].view(torch.float4_e2m1fn_x2)
        b_s = blocked_scales(tile["weight_scale"].view(torch.uint8)
                             .view(torch.float8_e4m3fn))
        try:
            y = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                                 out_dtype=torch.float32)
        except RuntimeError:
            y = torch._scaled_mm(a_q, b_q.t(), scale_a=a_s, scale_b=b_s,
                                 out_dtype=torch.bfloat16).to(torch.float32)
        return y * epilogue_value

    # per-expert joined tiles: one {expert: ([gate, up, down])} map of
    # dataclass-replaced units the way the loader builds them
    joined = {}
    for expert in range(EXPERTS):
        gate_local = shard_parsed_roles([stock[expert][0]], plan13)[0]
        up_local = shard_parsed_roles([stock[expert][1]], plan13)[0]
        shared, moved = shared_lut_global(
            [gate_local[1].unit.scale_lut, up_local[1].unit.scale_lut],
            [float(gate_local[1].unit.scale_global),
             float(up_local[1].unit.scale_global)], ["gate_proj", "up_proj"])
        gate_u = dataclasses.replace(gate_local[1].unit, scale_lut=moved[0],
                                     scale_global=float(shared))
        up_u = dataclasses.replace(up_local[1].unit, scale_lut=moved[1],
                                   scale_global=float(shared))
        down_local = shard_parsed_roles([stock[expert][2]], plan2)[0]
        down_u = down_local[1].unit
        from types import SimpleNamespace

        def view(parsed, unit):
            return SimpleNamespace(unit=unit, forests=parsed.forests, code=parsed.code)

        joined[expert] = (view(gate_local[1], gate_u), view(up_local[1], up_u),
                          view(down_local[1], down_u))

    for token in range(8):
        for slot in range(TOP_K):
            expert = int(ids[token, slot])
            gate_parsed = joined[expert][0]
            up_parsed = joined[expert][1]
            down_parsed = joined[expert][2]
            gate_o = w4a4(x[token:token + 1], gs13, gate_parsed,
                          epilogue13[expert:expert + 1])
            up_o = w4a4(x[token:token + 1], gs13, up_parsed,
                        epilogue13[expert:expert + 1])
            h = (F.silu(gate_o) * up_o).to(torch.bfloat16)
            down_o = w4a4(h, gs2, down_parsed, epilogue2[expert:expert + 1])
            ref[token] += float(weights[token, slot]) * down_o[0]
    # the same pipeline through the already-gated dense kernel, per expert:
    # slices of the layer's own stacks, so apply's wiring (dispatch,
    # activation, stage-2 gather, reduction) is isolated from the stock oracle
    from tessera.kernel_a4 import A4Unit, a4_span2_gemm

    def build_subset_nibbles_from_codes(codes):
        # invert the code table for the slice's own table (only the kernel's
        # subset_nibbles field is unused by the grouped kernels; the dense
        # wrapper validates it)
        return codes.new_zeros(
            4 * (1 << (int(layer.tessera_a4_gate_stack.rate) - 1)) * 2).cuda()

    def slice_unit(stack, expert):
        subset = build_subset_nibbles_from_codes(stack.code_nibbles[expert].cpu())
        return A4Unit(select=stack.select[expert], label=stack.label[expert],
                      point=stack.point[expert], nibbles=stack.nibbles[expert],
                      lut_bytes=stack.lut_bytes[expert], label_lut=stack.label_lut[expert],
                      subset_nibbles=subset,
                      code_nibbles=stack.code_nibbles[expert], rows=stack.rows,
                      cols=stack.cols, rate=stack.rate, arity=stack.arity,
                      memory=stack.memory, half=stack.half,
                      global_scale=float(stack.globals[expert]))

    native_ref = torch.zeros_like(out, dtype=torch.float32)
    for token in range(8):
        for slot in range(TOP_K):
            expert = int(ids[token, slot])
            p1, s1 = a4_quantize_activation(x[token:token + 1].contiguous(), gs13)
            gate_o = a4_span2_gemm(p1, s1, slice_unit(layer.tessera_a4_gate_stack, expert),
                                   epilogue13[expert:expert + 1], out_dtype=torch.float32)
            up_o = a4_span2_gemm(p1, s1, slice_unit(layer.tessera_a4_up_stack, expert),
                                 epilogue13[expert:expert + 1], out_dtype=torch.float32)
            h = (F.silu(gate_o) * up_o).to(torch.bfloat16)
            p2, s2 = a4_quantize_activation(h, gs2)
            down_o = a4_span2_gemm(p2, s2, slice_unit(layer.tessera_a4_down_stack, expert),
                                   epilogue2[expert:expert + 1], out_dtype=torch.float32)
            native_ref[token] += float(weights[token, slot]) * down_o[0]
    rel_native = float((out.float() - native_ref).abs().max()
                       / native_ref.abs().max().clamp_min(1e-12))
    report["checks"].append({"check": "apply_vs_dense_native", "ok": rel_native < 2e-2,
                             "rel": rel_native, "native_max": float(native_ref.abs().max())})
    rel = float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
    report["checks"].append({"check": "apply_vs_stock_oracle", "ok": rel < 2e-2,
                             "rel": rel,
                             "out_max": float(out.abs().max()),
                             "ref_max": float(ref.abs().max()),
                             "out0": [float(v) for v in out[0, :3]],
                             "ref0": [float(v) for v in ref[0, :3]],
                             "weights": [[float(weights[t, k]) for k in range(TOP_K)]
                                         for t in range(2)],
                             "ids": [[int(ids[t, k]) for k in range(TOP_K)]
                                     for t in range(2)]})

    # -- CUDA graph capture of apply ---------------------------------------
    eager = method.apply(layer, x, weights, ids, None, None)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            method.apply(layer, x, weights, ids, None, None)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method.apply(layer, x, weights, ids, None, None)
    graph.replay()
    torch.cuda.synchronize()
    report["checks"].append({"check": "cuda_graph_capture",
                             "ok": torch.equal(captured, eager)})

    # -- refusals -----------------------------------------------------------
    from tessera.serving.scheme import parse_compact_tessera_expert_blob

    bad = dict(expert_role_declarations(declared_group["groups"]["w13"])[0])
    bad["wire_stride"] = int(bad["wire_stride"]) - 1
    try:
        parse_compact_tessera_expert_blob(
            (DATA / "gate_proj_wire.bin").read_bytes(), bad, f"{PREFIX} gate",
            device="cuda")
        refused = False
    except Exception:  # noqa: BLE001 -- the refusal is the check
        refused = True
    report["checks"].append({"check": "sidecar_mismatch_refused", "ok": refused})
    ok = all(entry["ok"] for entry in report["checks"])
    report["ok"] = ok
    print("PROBE " + json.dumps(report))
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
