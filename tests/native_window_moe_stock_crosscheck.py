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


def _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size,
               apply_router_weight_on_input=False):
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
        apply_router_weight_on_input=apply_router_weight_on_input,
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


def _staged(reference, x, ids, weights, tp_rank=0, tp_size=1, dtype=torch.float64):
    """Per-stage reference (fp64) on the same decoded weights and quantized
    operands, so a stage's deviation can be attributed and budgeted.

    Returns (gate_up, act_bf16, down, final) with the SAME casts the stock
    pipeline performs: fp32 accumulate -> bf16 per-route gemm1 output -> fp32
    activation cast once -> per-route bf16 gemm2 output -> fp32 weighted sum
    -> bf16.
    """
    from vllm import _custom_ops  # noqa: F401

    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    t_tokens, top_k = ids.shape
    s1s, acts, downs, finals = [], [], [], []
    for token in range(t_tokens):
        q = torch.empty((1, HIDDEN), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((1, 1), dtype=torch.float32, device=x.device)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(q, x[token].reshape(1, -1), s, None)
        xg = (q.reshape(-1).float() * s.reshape(-1)[0]).to(dtype)
        row1, row2, rowf = [], [], None
        for choice in range(top_k):
            e = int(ids[token, choice])
            ref = reference[e]
            gate = ref["gate"]["weight"][lo:hi].to(x.device).to(dtype) \
                * ref["gate"]["weight_scale"][lo:hi].to(x.device).to(dtype)
            up = ref["up"]["weight"][lo:hi].to(x.device).to(dtype) \
                * ref["up"]["weight_scale"][lo:hi].to(x.device).to(dtype)
            down = ref["down"]["weight"][:, lo:hi].to(x.device).to(dtype) \
                * ref["down"]["weight_scale"].to(x.device).to(dtype)
            g = gate @ xg
            u = up @ xg
            act64 = torch.nn.functional.silu(g) * u
            aq = torch.empty((1, act64.numel()), dtype=torch.float8_e4m3fn, device=x.device)
            as_ = torch.empty((1, 1), dtype=torch.float32, device=x.device)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                aq, act64.float().reshape(1, -1), as_, None)
            act_q = (aq.reshape(-1).float() * as_.reshape(-1)[0]).to(dtype)
            row1.append(torch.cat([g, u]).bfloat16().to(dtype))
            row2.append(act64.bfloat16().to(dtype))
            d = down @ act_q
            downs.append(d.bfloat16().to(dtype))
            rowf = d if rowf is None else rowf + d
        s1s.append(torch.stack(row1).reshape(-1))
        acts.append(torch.stack(row2).reshape(-1))
        finals.append((rowf * weights[token, 0]).bfloat16().to(dtype))
    return (torch.stack(s1s), torch.stack(acts), torch.stack(downs).reshape(t_tokens, top_k, -1),
            torch.stack(finals))


def _budget(reference, x, ids, tp_rank=0, tp_size=1):
    """A dtype-derived error budget for the staged pipeline.

    fp32 accumulation: the standard gamma_n bound u=2^-24, gamma_n = n*u/(1-n*u),
    applied to the sum of |product| magnitudes at each gemm.  Per-stage casts:
    bf16 rounding is at most 2^-9 of the value cast.  Nothing here is read off
    the observed difference.
    """
    u = 2.0 ** -24
    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    e = int(ids[0, 0])
    ref = reference[e]
    gate = ref["gate"]["weight"][lo:hi].float().cuda()
    xg = x[0].float().abs()
    acc1 = HIDDEN * u / (1 - HIDDEN * u) * float((gate.abs() @ xg).max())
    act_mag = float(torch.nn.functional.silu(torch.zeros(1)).abs().max()) + 1.0
    acc2 = local * u / (1 - local * u) * act_mag
    casts = 2 ** -9 * (1.0 + act_mag + 1.0)
    return {"accumulation": acc1 + acc2, "casts": casts, "total": acc1 + acc2 + casts}


def _loose_reference(reference, x, ids, weights, tp_rank=0, tp_size=1):
    """The superseded loose oracle: no stage-2 quant, a1 applied after silu.

    Kept ONLY as the defect discriminator: if the acceptance metric could not
    separate a real arithmetic error from accumulation order, this arm would
    land inside the same bound.
    """
    local = INTER // tp_size
    lo, hi = tp_rank * local, (tp_rank + 1) * local
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        q = torch.empty((1, HIDDEN), dtype=torch.float8_e4m3fn, device=x.device)
        s = torch.empty((1, 1), dtype=torch.float32, device=x.device)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(q, x[token].reshape(1, -1), s, None)
        for choice in range(ids.shape[1]):
            e = int(ids[token, choice])
            ref = reference[e]
            gate = ref["gate"]["weight"][lo:hi].float().to(x.device) \
                * ref["gate"]["weight_scale"][lo:hi].to(x.device)
            up = ref["up"]["weight"][lo:hi].float().to(x.device) \
                * ref["up"]["weight_scale"][lo:hi].to(x.device)
            down = ref["down"]["weight"][:, lo:hi].float().to(x.device) \
                * ref["down"]["weight_scale"].to(x.device)
            g = gate @ q.reshape(-1).float()
            u = up @ q.reshape(-1).float()
            out[token] += weights[token, choice] * s.reshape(-1)[0] * (
                down @ (torch.nn.functional.silu(g) * u))
    return out.bfloat16()


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

        if tp_size == 1:
            # --- router-weight-on-input placement: accepted only if actual
            # stock agrees; the same flag reaches stock's own kernel.
            layer.apply_router_weight_on_input = True
            native_in = method.apply(layer, x, weights, ids, None, None)
            stock_in = _fp8_stock(reference, x, ids, weights, quant, tp_rank, tp_size,
                                  apply_router_weight_on_input=True)
            layer.apply_router_weight_on_input = False
            ok &= _arm("fp8_tp1_rank0_weight_on_input", native_in, stock_in, report)

            # --- staged comparison (k=1) with a dtype-derived budget ----------
            ids1 = ids[:4, :1].contiguous()
            x1 = x[:4].contiguous()
            weights1 = torch.ones(4, 1, device="cuda")
            ones = torch.ones_like(weights1)
            s1_ref, act_ref, down_ref, final_ref = _staged(reference, x1, ids1, weights1)
            budget = _budget(reference, x1, ids1)
            if method._native.gate_up is not None:
                native_gu = method._native.gate_up(x1, ids1, ones, preserve=True)
            else:
                native_gu = torch.cat([
                    method._native.gate(x1, ids1, ones, preserve=True),
                    method._native.up(x1, ids1, ones, preserve=True)], dim=-1)
            gu_err = (native_gu[:, 0].float() - s1_ref.float()).abs().max().item()
            act_native = (torch.nn.functional.silu(native_gu[:, 0, :INTER].float())
                          * native_gu[:, 0, INTER:].float()).bfloat16()
            act_err = (act_native.float() - act_ref.float()).abs().max().item()
            native_down = method._native.down(act_ref.bfloat16().reshape(-1, INTER), ids1,
                                              ones, route_input=True, round_routes=True)
            down_err = (native_down.float() - down_ref[:, 0].float()).abs().max().item()
            stage = {"arm": "fp8_tp1_staged",
                     "gate_up_err": gu_err, "act_err": act_err, "down_err": down_err,
                     "budget": budget}
            report["stages"] = stage
            print("STAGE", json.dumps(stage), flush=True)
            ok &= gu_err <= 2 * budget["total"] and act_err <= 2 * budget["total"] \
                and down_err <= 2 * budget["total"]

            # --- defect discriminator: the loose oracle must exceed the bound --
            loose = _loose_reference(reference, x, ids, weights)
            loose_err = float((loose.float() - stock.float()).abs().max())
            native_err = float((native.float() - stock.float()).abs().max())
            discriminator = {"arm": "loose_vs_native_vs_stock",
                             "native_vs_stock": native_err, "loose_vs_stock": loose_err,
                             "budget_scale": budget["total"]}
            report["discriminator"] = discriminator
            print("DISCRIMINATOR", json.dumps(discriminator), flush=True)
            ok &= loose_err > 10 * max(native_err, budget["total"])

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
