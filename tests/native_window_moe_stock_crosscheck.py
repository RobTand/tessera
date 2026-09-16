"""Stock-oracle crosscheck for the REAL window MoE method.

Drives `build_tessera_moe_method` through create_weights -> every loader
callback -> process_weights_after_loading -> apply and compares the ROUTED
output against **actual stock vLLM `fused_experts`** on **independently
decoded reference weights** (the materialising reader / read_unit_artifact,
never the native path's own bundles):

* FP8 ordinary route, TP1 and both TP2 geometry cuts (rank-local weights and
  the rank-local stock result; no all-reduce in this component check);
* folded BF16 through the research route (TP2 rank 0).

Stock is the oracle for the boundary in question: dynamic per-token FP8 at
both stages, the per-route bf16 down output and `moe_sum` rounding.  The
numbers are reported, not buried: each arm prints max abs, max relative and
the deviation in bf16 ulps of the reference maximum.

Run directly (actual vLLM work is exempt from PrismaBuild), in the pinned
image:

  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp \
    -e PYTHONPATH=/work/src:/work/tests \
    -v <worktree>:/work:ro --entrypoint bash <image> -lc \
    "python3 -m pip install -q --user pytest; python3 /work/tests/native_window_moe_stock_crosscheck.py"
"""
import json
import sys

import torch

sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work/tests")

from tessera.serving import moe_route                       # noqa: E402
from tessera.serving.scheme import validate_tessera_moe_scheme  # noqa: E402
from test_serving_moe_route import _stack, HIDDEN, INTER, EXPERTS   # noqa: E402
from test_native_window_moe_method import (                 # noqa: E402
    _load_all, _native_layer, bf16_wires_native_data)


def _bf16_ulp(value: float) -> float:
    import struct

    if value == 0.0:
        return 1.0
    bits = struct.unpack("<I", struct.pack("<f", abs(value)))[0]
    upper = struct.unpack("<f", struct.pack("<I", bits + 0x8000))[0]
    return float(upper - abs(value)) if upper != abs(value) else float(abs(value) * 2 ** -8)


def _arm(name, native, stock, report):
    diff = (native.float() - stock.float()).abs()
    mag = float(stock.float().abs().max())
    ulp = _bf16_ulp(mag)
    entry = {
        "arm": name,
        "max_abs": float(diff.max()),
        "max_over_mag": float(diff.max() / max(mag, 1e-6)),
        "ulps_of_max": float(diff.max() / ulp),
        "shapes": [tuple(native.shape), tuple(stock.shape)],
    }
    report["arms"].append(entry)
    print("ARM", json.dumps(entry), flush=True)
    return float(diff.max() / ulp) <= 4.0


def _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    w1 = torch.stack([
        torch.cat([r["gate"]["weight"][lo:hi], r["up"]["weight"][lo:hi]]) for r in reference
    ]).cuda().contiguous()
    s1 = torch.stack([
        torch.cat([r["gate"]["weight_scale"][lo:hi], r["up"]["weight_scale"][lo:hi]]) for r in reference
    ]).cuda().contiguous()
    w2 = torch.stack([r["down"]["weight"][:, lo:hi] for r in reference]).cuda().contiguous()
    s2 = torch.stack([r["down"]["weight_scale"] for r in reference]).cuda().contiguous()
    return fused_experts(
        x, w1, w2, weights, ids, activation=MoEActivation.SILU,
        quant_config=quant(w1_scale=s1, w2_scale=s2))


def _fp8_quant_config():
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend, make_fp8_moe_quant_config)

    def build(w1_scale, w2_scale):
        return make_fp8_moe_quant_config(
            fp8_backend=Fp8MoeBackend.TRITON, w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=None, a2_scale=None, per_act_token_quant=True,
            per_out_ch_quant=True, block_shape=None,
            gemm1_alpha=None, gemm1_beta=None, swiglu_limit=None, layer=None)
    return build


def main():
    torch.manual_seed(11)
    report = {"device": torch.cuda.get_device_name(), "arms": []}
    ok = True

    # --- FP8 ordinary route: TP1 and both TP2 cuts -------------------------
    w13_blobs, w2_blobs, scheme, reference = _stack()
    declared = validate_tessera_moe_scheme(scheme, "m")
    blobs = {"w13": w13_blobs, "w2": [[b] for b in w2_blobs]}
    reference_cpu = moe_route.prepare_tessera_moe_experts(blobs, declared, "m", device="cpu")
    reference = [{"gate": {"weight": reference_cpu.w13_weight[e][:INTER].cpu(),
                           "weight_scale": reference_cpu.w13_weight_scale[e][:INTER].cpu()},
                  "up": {"weight": reference_cpu.w13_weight[e][INTER:].cpu(),
                         "weight_scale": reference_cpu.w13_weight_scale[e][INTER:].cpu()},
                  "down": {"weight": reference_cpu.w2_weight[e].cpu(),
                           "weight_scale": reference_cpu.w2_weight_scale[e].cpu()}}
                 for e in range(EXPERTS)]
    x = (torch.randn(8, HIDDEN) * 0.5).bfloat16().cuda()
    ids = torch.randint(0, EXPERTS, (8, 2), dtype=torch.int32, device="cuda")
    weights = torch.rand(8, 2, device="cuda")
    quant = _fp8_quant_config()
    for tp_rank, tp_size in ((0, 1), (0, 2), (1, 2)):
        layer = _native_layer(tp_rank=tp_rank, tp_size=tp_size)
        method = moe_route.build_tessera_moe_method(scheme, "m", "resident", layer)
        method.create_weights(layer, EXPERTS, HIDDEN, INTER // tp_size, torch.bfloat16)
        _load_all(method, layer, w13_blobs, w2_blobs)
        method.process_weights_after_loading(layer)
        native = method.apply(layer, x, weights, ids, None, None)
        stock = _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size)
        ok &= _arm(f"fp8_tp{tp_size}_rank{tp_rank}", native, stock, report)

    # --- folded BF16 research route, TP2 rank 0 ----------------------------
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    b13, b2, bscheme, expected = bf16_wires_native_data()
    layer = _native_layer(tp_rank=0, tp_size=2)
    import types
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(
            types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True))):
        method = moe_route.build_tessera_moe_method(
            bscheme, "m", "resident", layer,
            research_selected=moe_route.ResearchSelectedMoeConfig(
                max_experts_per_chunk=2, expected_tensor_parallel_size=2))
    method.create_weights(layer, 2, HIDDEN, INTER // 2, torch.bfloat16)
    _load_all(method, layer, b13, [pair[0] for pair in b2])
    method.process_weights_after_loading(layer)
    lo, hi = 0, INTER // 2
    w1 = torch.stack([
        torch.cat([expected[e][0][lo:hi], expected[e][0][INTER + lo:INTER + hi]])
        for e in range(2)]).cuda().contiguous()
    w2 = torch.stack([expected[e][1][:, lo:hi] for e in range(2)]).cuda().contiguous()
    # routing for 2 experts on the BF16 stacks
    ids2 = torch.randint(0, 2, (8, 2), dtype=torch.int32, device="cuda")
    native = method.apply(layer, x, weights, ids2, None, None)
    stock = fused_experts(x, w1.cuda(), w2.cuda(), weights, ids2,
                          activation=MoEActivation.SILU, global_num_experts=2)
    ok &= _arm("bf16_folded_tp2_rank0", native, stock, report)

    report["passed"] = bool(ok)
    print(json.dumps(report))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
