import inspect

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl as impl,
)

assert impl.supports_dense_mha_prefill is False
assert "do_kv_cache_update" in impl.__dict__
init_src = inspect.getsource(impl.__init__)
assert "self.rope_pad = 64" in init_src
fwd_src = inspect.getsource(impl.forward_mqa)
assert "torch.nn.functional.pad(q, (0, self.rope_pad))" in fwd_src
assert "qk_rope_head_dim=self.kernel_qk_rope_head_dim" in fwd_src
assert "return_valid_counts=True" in fwd_src
assert "sparse_mla_top_k=sparse_topk_capacity" in fwd_src
assert "seq_lens=topk_lengths" in fwd_src
assert "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)" in fwd_src
assert "attn_metadata.topk_tokens" not in fwd_src

from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    src = (site / rel).read_text()
    compile(src, rel, "exec")
    assert "buffer_width = topk_tokens\n" in src
    assert "kpool - 1 if kpool > 1" not in src
kpool_src = (site / "model_executor/layers/sparse_attn_indexer_kpool.py").read_text()
compile(kpool_src, "sparse_attn_indexer_kpool.py", "exec")
assert kpool_src.count("pool_ids[:, : select_k - 1]") == 2

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend as sm120_backend,
)

assert sm120_backend.get_supported_kernel_block_sizes() == [64]
align_src = (site / "platforms/cuda.py").read_text()
compile(align_src, "cuda.py", "exec")
assert "if capability is not None and capability.major == 12:" in align_src
assert "return index_kpool * min(page_sizes)" in align_src
assert "return major in (9, 10)" in align_src
warmup_src = (site / "model_executor/warmup/kernel_warmup.py").read_text()
compile(warmup_src, "kernel_warmup.py", "exec")
assert "flashinfer_sparse_mla_decode_autotune_warmup(worker)" not in warmup_src
assert "Skipping FlashInfer autotune on SM121" in warmup_src
print("glm53 NoPE sparse-MLA overlay verify OK")
