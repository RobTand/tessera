"""Explicit research-only GLM5-next NoPE MLA on the pinned stock SM121 runtime.

This uses vLLM's public CUSTOM backend registration. It does not modify stock
classes or attest a serving cell. FlashInfer owns the native attention kernel;
vLLM owns packed cache writes, metadata, sparse index conversion, and workspace.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)

from .backend import probed_platform_token

# Compatibility guards, not device qualification. These are the unmodified
# sources from eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c.
_STOCK_SHA256 = {
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py": "a0023f72125cb0d5599b5bf940c86be1f0c9985bd62b0919f243a8fda76f4449",
    "v1/attention/backends/mla/flashinfer_mla_sparse.py": "093181e4e0198b34075a3713fa62267c3f1fe74b1c96bb08298d553727d44478",
    "v1/attention/backends/mla/sparse_utils.py": "20372237899fb0a0c9152e12eab40396119b76c1ac5f1c9bdfdbed53f8146559",
    "v1/attention/backend.py": "8ddf8dd73c2b953a79f99e8de74db616b135590ac1f74e21e20b9ddc366b9994",
}


def require_stock_runtime() -> None:
    import vllm
    import flashinfer

    if vllm.__version__ != "0.28.1rc1.dev397+gfd4a15126.d20260904":
        raise RuntimeError(f"Tessera GLM53 NoPE requires pinned vLLM; got {vllm.__version__}")
    if flashinfer.__version__ != "0.6.18":
        raise RuntimeError(f"Tessera GLM53 NoPE requires FlashInfer 0.6.18; got {flashinfer.__version__}")
    root = Path(vllm.__file__).parent
    for relative, expected in _STOCK_SHA256.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Tessera GLM53 NoPE requires unchanged pinned stock source: {relative}")


def _config_reason(config) -> str | None:
    # Isolated attention graph equality does not qualify the hybrid model's
    # whole-engine graph path: the matched stub differs by 0.67253 logprob nats.
    if (not config.model_config.enforce_eager
            or config.compilation_config.mode != CompilationMode.NONE
            or config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE):
        return "Tessera GLM53 NoPE is eager-only; require --enforce-eager with compilation and CUDA graphs disabled"
    hf = config.model_config.hf_text_config
    expected = dict(model_type="glm5_next_text", kv_lora_rank=512,
                    qk_nope_head_dim=256, qk_rope_head_dim=0,
                    index_topk=2048, index_kpool=4)
    for name, value in expected.items():
        if getattr(hf, name, None) != value:
            return f"Tessera GLM53 NoPE requires {name}={value!r}"
    parallel = config.parallel_config
    for name in ("decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            return f"Tessera GLM53 NoPE requires {name}=1"
    if config.kernel_config.enable_flashinfer_autotune is not False:
        return "Tessera GLM53 NoPE requires kernel_config.enable_flashinfer_autotune=false"
    return None


class TesseraGLM53NoPEBackend(FlashInferMLASparseSM120Backend):
    """Opt-in CUSTOM backend; stock enums retain their own implementations."""

    supported_kv_cache_dtypes = ["fp8_ds_mla"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TesseraGLM53NoPEImpl

    @classmethod
    def get_supported_head_sizes(cls):
        return [512]

    @classmethod
    def supports_compute_capability(cls, capability):
        return (capability.major, capability.minor) == (12, 1)

    @classmethod
    def supports_combination(cls, *args, **kwargs):
        reason = super().supports_combination(*args, **kwargs)
        return reason or _config_reason(get_current_vllm_config())


class TesseraGLM53NoPEImpl(FlashInferMLASparseSM120Impl):
    def __init__(self, *args, **kwargs):
        require_stock_runtime()
        if probed_platform_token(device=torch.cuda.current_device(), torch=torch) != "sm_121":
            raise RuntimeError("Tessera GLM53 NoPE requires SM121")
        reason = _config_reason(get_current_vllm_config())
        if reason:
            raise RuntimeError(reason)
        super().__init__(*args, **kwargs)
        if (self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim) != (512, 256, 0):
            raise ValueError("Tessera GLM53 NoPE received incompatible MLA dimensions")

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                           kv_cache_dtype, k_scale):
        if k_pe.shape[-1] != 0 or kv_c_normed.shape[-1] != 512 or kv_cache_dtype != "fp8_ds_mla":
            raise ValueError("Tessera GLM53 NoPE requires 512 latent / 0 RoPE / fp8_ds_mla")
        # Stock 656-byte records retain 64 BF16 RoPE values. FlashInfer's
        # GLM53_NOPE reader ignores those values, but the writer requires them.
        padding = kv_c_normed.new_zeros((kv_c_normed.shape[0], 1, 64))
        super().do_kv_cache_update(kv_c_normed, padding, kv_cache, slot_mapping,
                                   kv_cache_dtype, k_scale)

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if q.dtype != torch.bfloat16 or q.shape[-1] != 512:
            raise ValueError("Tessera GLM53 NoPE requires BF16 queries with latent width 512")
        num_tokens = q.shape[0]
        topk = self.topk_indices_buffer[:num_tokens]
        # Hybrid MLA/KDA cache pages may have a stride beyond their logical
        # block size. Reuse stock row-stride semantics from the SM100 sibling.
        cache_rows, block_stride = flat_kv_row_view(kv_c_and_k_pe_cache, attn_metadata.block_size)
        if cache_rows.shape[0] % 64:
            raise ValueError("Tessera GLM53 NoPE requires cache storage divisible into 64-token pages")
        # Native NoPE decode is instantiated for 64-token physical pages.
        # A view preserves globally indexed rows even for larger hybrid pages.
        native_cache = cache_rows.view(-1, 64, 656)
        # Stock non-power-of-two compaction races column tiles via atomic_add.
        # Pad with invalid entries to its supported single-program row path,
        # then retain the original 2176-wide kernel capacity. This keeps stable
        # candidate order across repeats and CUDA graph replay.
        padded_width = 1 << (topk.shape[1] - 1).bit_length()
        padded_topk = torch.nn.functional.pad(topk, (0, padded_width - topk.shape[1]), value=-1)
        physical, counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens], attn_metadata.block_table,
            padded_topk, BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride, NUM_TOPK_TOKENS=padded_width,
            return_valid_counts=True,
        )
        physical = physical[:, :topk.shape[1]].contiguous()
        empty = counts == 0
        physical[:, 0] = physical[:, 0].masked_fill(empty, 0)
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla

        output = q.new_empty((num_tokens, self.num_heads, self.kv_lora_rank))
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1), kv_cache=native_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer, qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank, qk_rope_head_dim=0,
            block_tables=physical.unsqueeze(1), seq_lens=counts,
            max_seq_len=physical.shape[1], out=output.unsqueeze(1),
            bmm1_scale=self.scale, bmm2_scale=1.0,
            sparse_mla_top_k=physical.shape[1], kv_scale_format=self.kv_scale_format,
            sparse_mla_top_k_lens=counts.clamp(min=1),
        )
        out.masked_fill_(empty.view(-1, 1, 1, 1), 0.0)
        return out.squeeze(1), None
