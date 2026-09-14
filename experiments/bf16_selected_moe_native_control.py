#!/usr/bin/env python3
"""Small native stock-vLLM control for folded BF16 selected routed experts.

This is a numerical kernel/load test with locally generated wires. It does not
qualify full GLM, TP2, memory fit, latency or a production runtime cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import torch

E, H, N, TOPK, Q256 = 4, 256, 128, 2, 1792
PREFIX = "model.layers.0.mlp.experts"


def make_wires():
    from tessera.alphabet import BF16_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.unit_artifact import read_unit_artifact

    generator = torch.Generator(device="cpu").manual_seed(53)
    wires, folded, digests = {}, {}, {}
    strides = {"w13": 0, "w2": 0}
    for expert in range(E):
        for projection, rows, cols, group in (
                ("gate_proj", N, H, "w13"),
                ("up_proj", N, H, "w13"),
                ("down_proj", H, N, "w2")):
            source = (torch.randn(rows, cols, generator=generator) * 0.08).to(
                device="cuda", dtype=torch.float32)
            encoded, _unit, _forest = encode_linear_planes(
                source, grid=BF16_GRID, q256=Q256, name=projection, verify=False)
            blob = pack_fused([(projection, rows, encoded.blob)])
            key = f"{expert}.{projection}.wire"
            wires[key] = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            folded[(expert, projection)] = read_unit_artifact(
                encoded.blob, device="cuda").to(torch.bfloat16)
            digests[key] = hashlib.sha256(blob).hexdigest()
            strides[group] = max(strides[group], len(blob))
    scheme = {
        "family": "TESSERA_BF16", "structure": "routed_moe", "grid": "BF16",
        "body": "WINDOW", "plane": "CHANNEL", "experts": E,
        "groups": {
            "w13": {"rows": 2 * N, "columns": H, "q256": Q256,
                    "wire_stride": strides["w13"],
                    "roles": [["gate_proj", N], ["up_proj", N]]},
            "w2": {"rows": H, "columns": N, "q256": Q256,
                   "wire_stride": strides["w2"],
                   "roles": [["down_proj", H]]}}}
    return wires, folded, digests, scheme


def make_layer(scheme, selected):
    from vllm.model_executor.layers.fused_moe import RoutedExperts, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, MoEActivation)
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from tessera.serving.config import TesseraConfig

    parallel = FusedMoEParallelConfig(
        tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0, dp_size=1, dp_rank=0,
        ep_size=1, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="naive", enable_eplb=False)
    moe = FusedMoEConfig(
        num_experts=E, experts_per_token=TOPK, hidden_dim=H,
        intermediate_size=N, num_local_experts=E, num_logical_experts=E,
        activation=MoEActivation.SILU, device=torch.device("cuda"),
        routing_method=RoutingMethodType.TopK, moe_parallel_config=parallel,
        in_dtype=torch.bfloat16, intermediate_size_per_partition=N,
        moe_backend="triton", swiglu_limit=10.0)
    manager = ExpertMapManager(
        max_num_batched_tokens=64, top_k=TOPK, global_num_experts=E,
        num_redundant_experts=0, num_expert_group=None, moe_parallel_config=parallel,
        placement_strategy="linear", enable_eplb=False)
    qconfig = None
    if selected:
        qconfig = TesseraConfig.from_config({
            "quant_method": "tessera", "format": "tessera", "ignore": [],
            "config_groups": {"selected_bf16": {"format": "TESSERA",
                                                 "targets": [PREFIX], "scheme": scheme}},
            "research_selected_moe": {
                "schema": "tessera.research_selected_moe.v1",
                "max_experts_per_chunk": 2, "decode_backend": "triton",
                "expected_tensor_parallel_size": 1}})
    return RoutedExperts(
        layer_name=PREFIX, params_dtype=torch.bfloat16, moe_config=moe,
        quant_config=qconfig, expert_map_manager=manager,
        swiglu_limit=10.0).to("cuda")


def reference(x, ids, weights, first, second, clamp):
    output = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            gate, up = (first[expert].float() @ x[token].float()).chunk(2)
            if clamp is not None:
                gate = gate.clamp(max=clamp)
                up = up.clamp(-clamp, clamp)
            value = torch.nn.functional.silu(gate) * up
            output[token] += weights[token, choice].float() * (second[expert].float() @ value)
    return output


def compare(actual, expected):
    delta = actual.float() - expected.float()
    return {"max_abs": float(delta.abs().max()),
            "rel_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
            "exact": bool(torch.equal(actual, expected))}


def run():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (ensure_model_parallel_initialized,
                                  init_distributed_environment)
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.cuda.set_device(0)
    config = VllmConfig()
    config.model_config = SimpleNamespace(enforce_eager=True, is_moe=True)
    with set_current_vllm_config(config, check_compile=False):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        init_distributed_environment(
            world_size=1, rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=0, backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        init_workspace_manager(torch.device("cuda"))
        wires, folded, digests, scheme = make_wires()
        selected_layer = make_layer(scheme, True)
        method = selected_layer.quant_method
        anchors = {name: int(value.numel()) for name, value in selected_layer.named_parameters()}
        loaded = sorted(selected_layer.load_weights(sorted(wires.items())))
        method.process_weights_after_loading(selected_layer)
        assert not dict(selected_layer.named_parameters())
        assert method.research_resident_bytes() > 0

        first = torch.stack([torch.cat([folded[(e, "gate_proj")],
                                        folded[(e, "up_proj")]]) for e in range(E)])
        second = torch.stack([folded[(e, "down_proj")] for e in range(E)])
        stock_layer = make_layer(scheme, False)
        stock_layer.w13_weight.data.copy_(first)
        stock_layer.w2_weight.data.copy_(second)
        stock_layer.quant_method.process_weights_after_loading(stock_layer)

        ids = torch.tensor([[3, 1], [1, 3], [0, 1], [0, 3], [3, 0]],
                           device="cuda", dtype=torch.int32)
        weights = torch.tensor([[.7, .3], [.4, .6], [.65, .35], [.2, .8], [.55, .45]],
                               device="cuda", dtype=torch.float32)
        x = (torch.randn(5, H, generator=torch.Generator(device="cuda").manual_seed(17),
                         device="cuda", dtype=torch.bfloat16) * 32).contiguous()
        expected = stock_layer.quant_method.apply(stock_layer, x, weights, ids, None, None)
        got = method.apply(selected_layer, x, weights, ids, None, None)
        torch.cuda.synchronize()
        parity = compare(got, expected)
        assert parity["exact"], parity

        selected_ids = torch.unique(ids)
        chosen = method._packed.decode_folded(selected_ids,
                                              max_experts_per_chunk=2, backend="triton")
        assert torch.equal(chosen.w13_weight, first.index_select(0, selected_ids.long()))
        assert torch.equal(chosen.w2_weight, second.index_select(0, selected_ids.long()))

        # Exercise the stock kernel on a deliberately wrong compact mapping.
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import make_unquantized_moe_kernel
        mapping = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        mapping[selected_ids.long()] = torch.arange(selected_ids.numel(),
                                                    dtype=torch.int32, device="cuda")
        wrong_map = mapping.clone()
        wrong_map[selected_ids[:2].long()] = mapping[selected_ids[:2].flip(0).long()]
        kernel = make_unquantized_moe_kernel(
            quant_config=FusedMoEQuantConfig.make(gemm1_clamp_limit=10.0),
            moe_config=method.moe, backend=method.bf16_backend,
            experts_cls=method.experts_cls,
            routing_tables=selected_layer._expert_routing_tables())
        wrong = kernel.apply(x, chosen.w13_weight, chosen.w2_weight, weights, ids,
                             activation=selected_layer.activation, global_num_experts=E,
                             expert_map=wrong_map, apply_router_weight_on_input=False)
        wrong_delta = compare(wrong, expected)
        assert wrong_delta["max_abs"] > 0, wrong_delta

        clamped = reference(x, ids, weights, first, second, 10.0)
        unclamped = reference(x, ids, weights, first, second, None)
        clamp_reference = compare(got, clamped)
        clamp_effect = compare(unclamped, clamped)
        assert clamp_effect["rel_l2"] > .05, clamp_effect
        assert clamp_reference["rel_l2"] < .15, clamp_reference

        # The stock MoERunner owns shared-expert addition, outside the selected
        # kernel. Exercise its real forward rather than assuming method.apply
        # itself returns the shared result.
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        class FixedRouter(torch.nn.Module):
            def select_experts(self, **kwargs):
                assert kwargs["hidden_states"].shape == x.shape
                return weights, ids

        shared = torch.nn.Linear(H, H, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            shared.weight.fill_(0.001)
        runner = MoERunner(
            layer_name=PREFIX + ".runner", moe_config=method.moe,
            router=FixedRouter(), routed_experts=selected_layer,
            shared_experts=shared)
        with set_forward_context(None, config, num_tokens=x.shape[0]):
            combined = runner(x, torch.zeros(x.shape[0], E, device="cuda"))
        torch.cuda.synchronize()
        shared_expected = got + shared(x)
        shared_parity = compare(combined, shared_expected)
        assert shared_parity["exact"], shared_parity

        return {"schema": "tessera.bf16_selected_native_control.v1",
                "stock_image_scope": "external pinned Docker image; recorded by caller",
                "source_scope": "locally generated 4-expert BF16 wires",
                "q256": Q256, "tokens": int(x.shape[0]), "topk": TOPK,
                "selected_global_experts": selected_ids.tolist(),
                "unselected_experts": sorted(set(range(E)) - set(selected_ids.tolist())),
                "wire_sha256": digests, "wire_anchor_bytes": anchors,
                "loaded_parameter_names": loaded,
                "backend": str(getattr(method.bf16_backend, "value", method.bf16_backend)),
                "experts_cls": method.experts_cls.__name__,
                "resident_packed_bytes": method.research_resident_bytes(),
                "selected_vs_full_stock_unquantized": parity,
                "deliberately_wrong_map": wrong_delta,
                "native_vs_fp32_clamped": clamp_reference,
                "clamp_effect_fp32": clamp_effect,
                "shared_expert_runner_checked": True,
                "shared_expert_runner_vs_stock_sum": shared_parity,
                "performance_claim": False,
                "runtime_cell_promoted": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(args.out), "parity": result["selected_vs_full_stock_unquantized"]}))


if __name__ == "__main__":
    main()
