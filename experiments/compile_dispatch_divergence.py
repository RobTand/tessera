#!/usr/bin/env python3
"""Why an eager arm and a compiled arm of one checkpoint are two different programs.

WHAT ISSUE #16 LEFT OPEN.  ``docs/measurements/serving-compile-divergence-2026-09-02.md``
established that the eager-vs-compiled KL is *deterministic* per (weights, build)
-- 0.0269 on the FP8 route, 0.2445 on the NVFP4 route, reproduced to six decimals
by independent serves in separate containers -- and attributed its size to
"compiling an NVFP4 forward on this model".  It did not say what the compiler
changed.  This harness answers that, and the answer is not a compiler artefact:
under a compiled forward vLLM 0.28 **runs different implementations of the same
math**, chosen by two config defaults that both key off "is inductor going to
run":

* ``vllm/config/vllm.py:1392-1399`` appends the ``custom_ops`` base mode:
  ``"none"`` when ``backend == "inductor" and mode != NONE``, else ``"all"``.
  With ``none`` every ``CustomOp`` (``SiluAndMul``, ``RotaryEmbedding``, ...)
  runs ``forward_native`` -- the torch decomposition -- instead of its CUDA
  kernel.
* ``vllm/platforms/cuda.py:690-700`` sets the IR-op priority: ``["native"]``
  when compiling, ``["vllm_c", "native"]`` otherwise, with the comment "Native
  used by default when compiling, use vllm_c kernels where available when no
  codegen".  ``RMSNorm.forward_native``/``forward_cuda`` both call
  ``ir.ops.rms_norm``; the priority list is what picks the kernel
  (``vllm/ir/op.py:327`` ``dispatch``).

Both arms of the divergence recorded that switch in their own startup log, which
is the attestation this file rests on rather than any reading of ours::

  eager    (--enforce-eager):  'custom_ops': ['all']
                               ir_op_priority=IrOpPriorityConfig(
                                   rms_norm=['vllm_c', 'native'],
                                   fused_add_rms_norm=['vllm_c', 'native'])
  compiled (default)        :  'custom_ops': ['none']
                               ir_op_priority=IrOpPriorityConfig(
                                   rms_norm=['native'], fused_add_rms_norm=['native'])

  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log:12
  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log:12

Both arms of that pair also ran with **every fusion pass off**
(``'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False,
'fuse_attn_quant': False, ...}`` in the compiled arm's own config line), so the
issue's "fusion changing accumulation order" hypothesis is not what happened
there.

WHAT THIS MEASURES.  On the real activations of the KL corpus's first chunk,
through the real Qwen3-0.6B weights: run each producer op both ways, then push
both results through the *same* activation quantizer the W4A4 and W8A8 routes
use, and count what changes.  The chain is the served one: a bf16-ulp difference
in the norm/activation output is re-drawn by an FP4 quantizer whose codes are
~40% apart, so a difference far below bf16 resolution becomes a difference in
the code that reaches the tensor cores.

SCOPE.  This is an op-level attribution, not the end-to-end proof.  It says the
two dispatch choices produce different quantized activations and by how much; it
does not by itself say that accounts for all of the 0.2445.  The end-to-end arm
that does is a serve with the dispatch pinned (see the receipt).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _fp4_lut(device: torch.device) -> torch.Tensor:
    """Signed E2M1 code -> value, indexed by the 4-bit code."""
    vals = FP4_VALUES + [-v for v in FP4_VALUES]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """``[rows, cols/2]`` uint8 of nibble-packed E2M1 -> ``[rows, cols]`` codes."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return out.to(torch.int64)


def dequant_fp4(packed: torch.Tensor, scales: torch.Tensor,
                global_scale: torch.Tensor) -> torch.Tensor:
    """NVFP4 tile -> float32, the ``(code * block_scale) / global_scale`` convention."""
    codes = unpack_fp4(packed)
    lut = _fp4_lut(packed.device)
    vals = lut[codes]
    blocks = scales.to(torch.float32)
    rows, groups = blocks.shape
    vals = vals.view(rows, groups, -1) * blocks.unsqueeze(-1)
    return (vals.view(rows, -1) / global_scale.to(torch.float32)).contiguous()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    denom = a32.norm().item()
    return float((a32 - b32).norm().item() / denom) if denom else float("nan")


def capture_activations(model_dir: str, corpus: str, layer: int, device: str):
    """Real residual stream, attention output and gate/up activations at one layer.

    The tokens are the KL corpus's first chunk -- the same ids the served KL is
    scored on -- so the distributions here are the distributions the divergence
    was measured under, not a random draw.
    """
    from transformers import AutoModelForCausalLM

    with open(corpus) as fh:
        contract = json.load(fh)
    ids = torch.tensor([contract["chunks"][0]], dtype=torch.long, device=device)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16).to(device).eval()
    block = model.model.layers[layer]
    grab: dict[str, torch.Tensor] = {}

    handles = [
        block.register_forward_pre_hook(
            lambda _m, args, _g=grab: _g.__setitem__("resid", args[0].detach().clone())),
        block.self_attn.register_forward_hook(
            lambda _m, _a, out, _g=grab: _g.__setitem__(
                "attn_out", (out[0] if isinstance(out, tuple) else out).detach().clone())),
        block.mlp.gate_proj.register_forward_hook(
            lambda _m, _a, out, _g=grab: _g.__setitem__("gate", out.detach().clone())),
        block.mlp.up_proj.register_forward_hook(
            lambda _m, _a, out, _g=grab: _g.__setitem__("up", out.detach().clone())),
    ]
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    hidden = model.config.hidden_size
    out = {
        "resid": grab["resid"].reshape(-1, hidden).contiguous(),
        "attn_out": grab["attn_out"].reshape(-1, hidden).contiguous(),
        "gate_up": torch.cat([grab["gate"], grab["up"]], dim=-1)
                        .reshape(-1, 2 * model.config.intermediate_size).contiguous(),
        "input_layernorm": block.input_layernorm.weight.detach().clone(),
        "post_attention_layernorm": block.post_attention_layernorm.weight.detach().clone(),
    }
    del model
    torch.cuda.empty_cache()
    return out


def producer_arms(acts: dict, eps: float) -> dict:
    """Run each producer op the eager way and the compiled way.

    ``vllm_c``/``native`` are the two providers the two arms' ``ir_op_priority``
    select; ``forward_cuda``/``forward_native`` are the two the ``custom_ops``
    base mode selects.  Nothing here chooses an implementation on its own -- it
    calls both of the ones vLLM would.
    """
    import torch.nn.functional as F
    from vllm import ir
    import vllm.kernels.vllm_c  # noqa: F401  (registers the vllm_c IR providers)

    rms = ir.ops.rms_norm
    fused = ir.ops.fused_add_rms_norm
    w_in = acts["input_layernorm"]
    w_post = acts["post_attention_layernorm"]

    arms: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}

    x = acts["resid"]
    arms["rms_norm"] = (
        rms.impls["vllm_c"].func_impl_fn(x, w_in, eps),
        rms.impls["native"].func_impl_fn(x, w_in, eps),
        "input_layernorm output -- the A side of q/k/v",
    )

    xa, ra = acts["attn_out"].clone(), acts["resid"].clone()
    xb, rb = acts["attn_out"].clone(), acts["resid"].clone()
    out_c, _ = fused.impls["vllm_c"].func_impl_fn(xa, ra, w_post, eps)
    out_n, _ = fused.impls["native"].func_impl_fn(xb, rb, w_post, eps)
    arms["fused_add_rms_norm"] = (
        out_c, out_n, "post_attention_layernorm output -- the A side of gate/up")

    gu = acts["gate_up"]
    d = gu.shape[-1] // 2
    cuda_out = torch.empty(gu.shape[:-1] + (d,), dtype=gu.dtype, device=gu.device)
    torch.ops._C.silu_and_mul(cuda_out, gu)
    arms["silu_and_mul"] = (
        cuda_out, F.silu(gu[..., :d]) * gu[..., d:],
        "SiluAndMul output -- the A side of down_proj",
    )
    return arms


def quant_legs(a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor) -> dict:
    """Push both arms through the two activation quantizers the routes execute."""
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    leg: dict = {}

    pa, sa = torch.ops._C.scaled_fp4_quant(a, global_scale, False)
    pb, sb = torch.ops._C.scaled_fp4_quant(b, global_scale, False)
    ca, cb = unpack_fp4(pa), unpack_fp4(pb)
    sa8 = sa.view(torch.float8_e4m3fn)[: a.shape[0], : a.shape[1] // 16]
    sb8 = sb.view(torch.float8_e4m3fn)[: a.shape[0], : a.shape[1] // 16]
    da = dequant_fp4(pa, sa8, global_scale)
    db = dequant_fp4(pb, sb8, global_scale)
    leg["nvfp4"] = {
        "codes": int(ca.numel()),
        "codes_differing": int((ca != cb).sum().item()),
        "code_flip_rate": float((ca != cb).float().mean().item()),
        "block_scales": int(sa8.numel()),
        "block_scales_differing": int((sa8.float() != sb8.float()).sum().item()),
        "block_scale_flip_rate": float((sa8.float() != sb8.float()).float().mean().item()),
        # How the arms' disagreement compares to the quantizer's own error: the
        # denominator is what the route pays for being 4-bit at all.
        "rel_l2_between_arms": rel_l2(da, db),
        "rel_l2_quant_error_eager": rel_l2(a, da),
        "rel_l2_quant_error_compiled": rel_l2(b, db),
        "dequant_sanity_max_rel": float(
            ((da - a.to(torch.float32)).abs().max() / a.abs().max()).item()),
    }

    qa = torch.empty(a.shape, dtype=torch.float8_e4m3fn, device=a.device)
    qb = torch.empty(b.shape, dtype=torch.float8_e4m3fn, device=b.device)
    ta = torch.empty((a.shape[0], 1), dtype=torch.float32, device=a.device)
    tb = torch.empty((b.shape[0], 1), dtype=torch.float32, device=b.device)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(qa, a, ta, None)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(qb, b, tb, None)
    fa = qa.to(torch.float32) * ta
    fb = qb.to(torch.float32) * tb
    ba = qa.view(torch.uint8)
    bb = qb.view(torch.uint8)
    leg["fp8"] = {
        "bytes": int(ba.numel()),
        "bytes_differing": int((ba != bb).sum().item()),
        "byte_flip_rate": float((ba != bb).float().mean().item()),
        "per_token_scales_differing": int((ta != tb).sum().item()),
        "rel_l2_between_arms": rel_l2(fa, fb),
        "rel_l2_quant_error_eager": rel_l2(a, fa),
        "rel_l2_quant_error_compiled": rel_l2(b, fb),
    }
    return leg


def gemm_leg(a: torch.Tensor, b: torch.Tensor, ckpt: str, weight_prefix: str,
             global_scale: torch.Tensor, bf16_dir: str, bf16_key: str) -> dict:
    """What the two arms' A sides do to the product, against the BF16 product.

    The weight is the served checkpoint's own NVFP4 tile and is identical in both
    arms; only the A side differs.  The product is taken in float32 on the
    dequantized tile rather than through ``_scaled_mm`` -- this leg is about the
    values the codes stand for, and a cuBLAS mainloop would add its own
    accumulation order to a comparison that is not about accumulation order.
    """
    from safetensors import safe_open

    with safe_open(ckpt, "pt") as fh:
        packed = fh.get_tensor(f"{weight_prefix}.weight_packed").cuda()
        wscale = fh.get_tensor(f"{weight_prefix}.weight_scale").cuda()
        wglobal = fh.get_tensor(f"{weight_prefix}.weight_global_scale").cuda()
    with safe_open(bf16_dir, "pt") as fh:
        w_bf16 = fh.get_tensor(bf16_key).cuda().to(torch.float32)

    w = dequant_fp4(packed, wscale, wglobal)

    def quantized_product(x: torch.Tensor) -> torch.Tensor:
        p, s = torch.ops._C.scaled_fp4_quant(x, global_scale, False)
        s8 = s.view(torch.float8_e4m3fn)[: x.shape[0], : x.shape[1] // 16]
        return dequant_fp4(p, s8, global_scale) @ w.t()

    ya, yb = quantized_product(a), quantized_product(b)
    ref = a.to(torch.float32) @ w_bf16.t()
    return {
        "weight": weight_prefix,
        "rel_l2_arms_vs_each_other": rel_l2(ya, yb),
        "rel_l2_eager_vs_bf16": rel_l2(ref, ya),
        "rel_l2_compiled_vs_bf16": rel_l2(ref, yb),
        "arm_gap_over_quant_error": rel_l2(ya, yb) / rel_l2(ref, ya),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16-model", default="/home/rob/models/Qwen3-0.6B")
    ap.add_argument("--nvfp4-checkpoint",
                    default="/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-k2-q896-nvfp4")
    ap.add_argument("--corpus", default="/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this harness measures CUDA kernels; no CUDA device is visible")

    acts = capture_activations(args.bf16_model, args.corpus, args.layer, "cuda")
    arms = producer_arms(acts, args.eps)

    ckpt = str(Path(args.nvfp4_checkpoint) / "model.safetensors")
    bf16 = str(Path(args.bf16_model) / "model.safetensors")
    from safetensors import safe_open
    with safe_open(ckpt, "pt") as fh:
        gs = {name: fh.get_tensor(
            f"model.layers.{args.layer}.{name}.input_global_scale").cuda()
            for name in ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")}

    consumer = {
        "rms_norm": "self_attn.q_proj",
        "fused_add_rms_norm": "mlp.gate_proj",
        "silu_and_mul": "mlp.down_proj",
    }
    report: dict = {
        "schema": "tessera.compile_dispatch_divergence/1",
        "params": {
            "bf16_model": args.bf16_model,
            "nvfp4_checkpoint": args.nvfp4_checkpoint,
            "corpus": args.corpus,
            "layer": args.layer,
            "eps": args.eps,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "ops": {},
    }
    try:
        import vllm
        report["params"]["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        report["params"]["vllm"] = None

    for op, (a, b, what) in arms.items():
        entry = {
            "what": what,
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "eager_impl": "vllm_c / forward_cuda",
            "compiled_impl": "native / forward_native",
            "elements": int(a.numel()),
            "elements_differing": int((a != b).sum().item()),
            "elementwise_disagreement_rate": float((a != b).float().mean().item()),
            "max_abs_diff": float((a.float() - b.float()).abs().max().item()),
            "rel_l2": rel_l2(a, b),
            "consumer": consumer[op],
            "quant": quant_legs(a, b, gs[consumer[op]]),
        }
        report["ops"][op] = entry

    a, b, _ = arms["rms_norm"]
    report["gemm"] = gemm_leg(
        a, b, ckpt, f"model.layers.{args.layer}.self_attn.q_proj",
        gs["self_attn.q_proj"], bf16,
        f"model.layers.{args.layer}.self_attn.q_proj.weight")

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
