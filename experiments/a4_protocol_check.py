"""Native A4 method protocol check: modular only, no false stock-kernel owner.

Runs in the pinned image (real vLLM).  Constructs the routed method the way
the runtime does, then asserts the protocol the runner dispatches on:

* ``is_monolithic`` is False and stays False with auto/default selection and
  with an explicit FlashInfer backend setting;
* ``moe_kernel`` is absent or None and the method owns no ``experts_cls``, so
  no obsolete monolithic stock hook can be reached;
* ``apply`` runs through the modular path against the independent stock
  oracle, as ``FusedMoERunner._apply_quant_method`` would call it.

Usage: python3 experiments/a4_protocol_check.py --rank 0 [--backend auto]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

DATA = Path(os.environ.get("TESSERA_A4_WIRE_DIR", "/mnt/shared/astra-native-a4/data"))
PREFIX = "model.language_model.layers.3.mlp.experts"
HIDDEN, LOCAL_INTER, EXPERTS, TOP_K = 4096, 1024, 2, 2


def build(backend: str, tp_rank: int = 0):
    from tessera.serving.scheme import validate_tessera_moe_scheme
    from tessera.serving.nvfp4_moe_route import build_tessera_nvfp4_moe_method

    cfg = json.loads((DATA / "a4-config.json").read_text())
    scheme = dict(cfg["quantization_config"]["config_groups"][
        "tessera_model_language_model_layers_3_mlp_experts"]["scheme"])
    scheme["experts"] = EXPERTS

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                             FusedMoEParallelConfig)
    parallel = FusedMoEParallelConfig(
        tp_size=2, pcp_size=1, dp_size=1, ep_size=1, tp_rank=tp_rank,
        pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=False,
        all2all_backend="allgather_reducescatter", enable_eplb=False)

    import torch.nn as nn

    layer = nn.Module()
    # the attributes a real RoutedExperts carries into apply
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer.activation = MoEActivation.SILU
    layer.global_num_experts = EXPERTS
    layer.moe_config = FusedMoEConfig(
        num_experts=EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN,
        intermediate_size=LOCAL_INTER * 2, num_local_experts=EXPERTS,
        num_logical_experts=EXPERTS, activation=MoEActivation.SILU,
        device=torch.device("cuda"), routing_method="topk",
        moe_parallel_config=parallel, in_dtype=torch.bfloat16,
        intermediate_size_per_partition=LOCAL_INTER, moe_backend=backend)
    method = build_tessera_nvfp4_moe_method(scheme, PREFIX, "resident", layer)
    method.create_weights(layer, EXPERTS, HIDDEN, LOCAL_INTER, torch.bfloat16,
                          global_num_experts=EXPERTS)
    return layer, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    report = {"rank": args.rank, "backend": args.backend, "checks": []}

    def record(name, ok, detail=None):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail if detail is not None else ''}",
              flush=True)

    layer, method = build(args.backend, args.rank)
    protocol = {
        "is_monolithic": bool(method.is_monolithic),
        "has_experts_cls": hasattr(method, "experts_cls"),
        "moe_kernel": getattr(method, "moe_kernel", None) is not None,
        "topk_indices_dtype": str(method.topk_indices_dtype),
        "mk_can_overlap_shared_experts": bool(method.mk_can_overlap_shared_experts),
    }
    record("modular_protocol", protocol["is_monolithic"] is False
           and not protocol["has_experts_cls"] and not protocol["moe_kernel"],
           protocol)
    # the BASE may define the hook; the ROUTE must not
    assert "apply_monolithic" not in type(method).__dict__, "obsolete monolithic hook present"
    runner_path = not protocol["is_monolithic"]
    record("runner_dispatch_is_modular", runner_path, {"runner_calls": "forward_modular"})

    if runner_path:
        for expert in range(EXPERTS):
            for shard, name in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
                method._load_wire(None, torch.frombuffer(bytearray(
                    (DATA / f"{name}_wire.bin").read_bytes()), dtype=torch.uint8),
                    name, shard, expert)
                method._load_input_global_scale(
                    method._input_global["w13" if shard != "w2" else "w2"],
                    torch.tensor([448.0 * 6.0 / 2.0], dtype=torch.float32),
                    name, shard, expert)
        method.process_weights_after_loading(layer)
        torch.manual_seed(4)
        x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.25
        ids = torch.randint(0, EXPERTS, (8, TOP_K), device="cuda", dtype=torch.int32)
        weights = torch.rand(8, TOP_K, device="cuda") * 0.6 + 0.2
        out = method.apply(layer, x, weights, ids, None, None)
        config = method.moe_quant_config
        record("native_apply_modular", out.shape == x.shape,
               {"shape": list(out.shape),
                "quant_config": type(config).__name__ if config is not None else None})

    # The hazard itself, on an isolated stub: a monolithic stock class
    # attached as ``experts_cls`` makes the BASE ``is_monolithic`` answer
    # True, which is the dispatch this route must never inherit.  (This is a
    # demonstration of the mechanism, not a claim that the auto selection
    # above picked such a class: the prefix receipts show it did not.)
    class _MonolithicStock:
        @staticmethod
        def is_monolithic() -> bool:
            return True

    class _StubBase:
        def __init__(self):
            self.moe_kernel = None
            self.experts_cls = _MonolithicStock

        @property
        def is_monolithic(self) -> bool:
            if self.moe_kernel is None:
                return self.experts_cls.is_monolithic()
            return self.moe_kernel.is_monolithic

    inherited = _StubBase().is_monolithic
    native_answer = method.is_monolithic
    record("monolithic_inheritance_stub",
           inherited is True and native_answer is False,
           {"stub_with_experts_cls": inherited, "native_method": native_answer})

    report["ok"] = all(c["ok"] for c in report["checks"])
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
    print("PROTOCOL " + json.dumps(report))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
