#!/usr/bin/env python3
"""Native correctness check for the research GLM53 CUSTOM vLLM backend.

Run in the pinned vLLM image on SM121. --stock is the unmodified regression
arm and must fail at the zero-RoPE cache write. No model weights are required.
The oracle decodes the same packed cache and the native kernel variant's FP8
query representation, then computes FP32 softmax attention. It also reports
deviation from unquantized BF16 queries separately.
"""
import argparse
import json
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
from tessera.serving.glm53_nope import TesseraGLM53NoPEImpl, require_stock_runtime


def check(stock=False, heads=64, repeat=1, padded=False):
    require_stock_runtime()
    torch.manual_seed(20260913)
    device = "cuda"
    width, block_size = 512, 256
    tokens = 2304
    latent = torch.randn(tokens, width, dtype=torch.bfloat16, device=device)
    rope = torch.empty(tokens, 1, 0, dtype=torch.bfloat16, device=device)
    slots = torch.arange(tokens, device=device)
    storage = torch.zeros(tokens // block_size, block_size * (2 if padded else 1), 656, dtype=torch.uint8, device=device)
    cache = storage[:, :block_size]
    impl = (FlashInferMLASparseSM120Impl if stock else TesseraGLM53NoPEImpl).__new__(
        FlashInferMLASparseSM120Impl if stock else TesseraGLM53NoPEImpl)
    impl.kv_cache_dtype = "fp8_ds_mla"
    impl.kv_lora_rank, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 512, 256, 0
    impl.num_heads, impl.scale, impl.kv_scale_format = heads, 256 ** -0.5, "arbitrary_fp32"
    impl._workspace_buffer = None
    impl.do_kv_cache_update(latent, rope, cache, slots, "fp8_ds_mla", torch.ones(1, device=device))
    torch.cuda.synchronize()
    # Four FP32 scales per 512-value latent followed by 64 BF16 zero RoPE.
    rows = cache.reshape(tokens, 656)
    scales = rows[:, 512:528].contiguous().view(torch.float32)
    decoded = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(tokens, 4, 128)
    decoded = (decoded * scales.unsqueeze(-1)).reshape(tokens, 512)
    assert torch.count_nonzero(rows[:, 528:]).item() == 0
    assert torch.isfinite(decoded).all()
    # Empty, one-element, partial with holes, complete2048, expanded kpool2049.
    counts = (0, 1, 73, 2048, 2049) * repeat
    topk = torch.full((len(counts), 2176), -1, dtype=torch.int32, device=device)
    indices = []
    for row, count in enumerate(counts):
        selected = torch.randperm(tokens, device=device)[:count]
        positions = torch.randperm(2176, device=device)[:count]
        topk[row, positions] = selected.to(torch.int32)
        indices.append(selected)
    impl.topk_indices_buffer = topk
    metadata = SimpleNamespace(req_id_per_token=torch.zeros(len(counts), dtype=torch.int32, device=device),
        block_table=torch.arange(tokens // block_size, dtype=torch.int32, device=device).unsqueeze(0),
        block_size=block_size, topk_tokens=2048)
    q = torch.randn(len(counts), heads, width, dtype=torch.bfloat16, device=device)
    out, lse = impl.forward_mqa(q, cache, metadata, None)
    assert lse is None and out.shape == q.shape
    # Native decode quantizes Q per128 with power-of-two-ceiled E4M3 scales
    # (FlashInfer common/fp8_quant.cuh:212), independently of packed KV scales.
    # Compare its represented operands, and separately retain BF16-Q deviation.
    qgroups = q.float().reshape(len(counts), heads, 4, 128)
    from flashinfer.mla._sparse_mla_sm120_plan import plan, _MODEL_TYPE_GLM53_NOPE
    selected_plan = plan(len(counts), heads, 2176, _MODEL_TYPE_GLM53_NOPE, 64, False, 0, q.device)
    qscale = qgroups.abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448
    # swapAB folds arbitrary FP32 Q scales; split-K/MG use UE8M0 pow2.
    if selected_plan.variant.name != "PREFILL_SWAPAB":
        qscale = torch.exp2(torch.ceil(torch.log2(qscale)))
    print(f"native_plan={selected_plan}", flush=True)
    qdecoded = ((qgroups / qscale).to(torch.float8_e4m3fn).float() * qscale).reshape_as(q)
    expected = torch.zeros_like(q, dtype=torch.float32)
    bf16_q_expected = torch.zeros_like(expected)
    for row, selected in enumerate(indices):
        if selected.numel():
            kv = decoded[selected]
            scores = qdecoded[row] @ kv.T * impl.scale
            expected[row] = scores.softmax(-1) @ kv
            bf16_q_expected[row] = (q[row].float() @ kv.T * impl.scale).softmax(-1) @ kv
    delta = (out.float() - expected).abs()
    # Native output is BF16, permitting accumulation and final rounding error.
    tolerance = 2 * torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(out.float(), expected, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(out[0]).item() == 0
    repeated, _ = impl.forward_mqa(q, cache, metadata, None)
    torch.testing.assert_close(out, repeated, atol=0, rtol=0)
    # Capture the same cache update + attention path, then replay and compare.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        impl.do_kv_cache_update(latent, rope, cache, slots, "fp8_ds_mla", torch.ones(1, device=device))
        impl.forward_mqa(q, cache, metadata, None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    scale = torch.ones(1, device=device)
    with torch.cuda.graph(graph):
        impl.do_kv_cache_update(latent, rope, cache, slots, "fp8_ds_mla", scale)
        graph_out, _ = impl.forward_mqa(q, cache, metadata, None)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, graph_out, atol=0, rtol=0)
    print(json.dumps(dict(status="passed", counts=counts, heads=heads, padded=padded, native_plan=str(selected_plan), capacity=2176, max_abs=float(delta.max()),
        rms=float(delta.square().mean().sqrt()), bf16_query_max_abs=float((out.float()-bf16_q_expected).abs().max()), graph_exact=True, repeat_exact=True,
        dtype=str(out.dtype), device=torch.cuda.get_device_name()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--padded", action="store_true")
    args = parser.parse_args()
    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        check(args.stock, args.heads, args.repeat, args.padded)
