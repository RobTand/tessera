# Source: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
# Commit: a246263a9891407e0d640db42d111545005dd83f
# Extracted Python patch block. The patch operations are unchanged.
from pathlib import Path

target = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "flashinfer_mla_sparse_sm120.py"
)
source = target.read_text()


def replace_once(old: str, new: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch target: {old!r}")
    source = source.replace(old, new)


replace_once(
    '    """SM120 FlashInfer sparse-MLA implementation."""\n\n    is_sparse = True\n',
    '    """SM120 FlashInfer sparse-MLA implementation."""\n\n'
    "    is_sparse = True\n"
    "    supports_dense_mha_prefill = False\n",
)

replace_once(
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        from vllm.config import get_current_vllm_config\n",
    '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
    "        self.rope_pad = 0\n"
    "        if self.qk_rope_head_dim == 0:\n"
    "            if self.kv_lora_rank != 512:\n"
    "                raise NotImplementedError(\n"
    '                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "\n'
    '                    "576-wide GLM_NSA geometry, which requires "\n'
    '                    f"kv_lora_rank=512; got {self.kv_lora_rank}."\n'
    "                )\n"
    "            self.rope_pad = 64\n"
    "        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad\n"
    "        from vllm.config import get_current_vllm_config\n",
)

replace_once(
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "\n"
    "        num_actual_toks = q.shape[0]\n",
    "        if isinstance(q, tuple):\n"
    "            q = torch.cat(q, dim=-1)\n"
    "        if self.rope_pad:\n"
    "            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n"
    "\n"
    "        num_actual_toks = q.shape[0]\n",
)

replace_once(
    "            qk_rope_head_dim=self.qk_rope_head_dim,\n",
    "            qk_rope_head_dim=self.kernel_qk_rope_head_dim,\n",
)

replace_once(
    "        topk_indices_physical = cast(\n"
    "            torch.Tensor,\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "            ),\n"
    "        )\n",
    "        topk_indices_physical, topk_lengths = cast(\n"
    "            tuple[torch.Tensor, torch.Tensor],\n"
    "            triton_convert_req_index_to_global_index(\n"
    "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
    "                attn_metadata.block_table,\n"
    "                topk_indices,\n"
    "                BLOCK_SIZE=attn_metadata.block_size,\n"
    "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
    "                return_valid_counts=True,\n"
    "            ),\n"
    "        )\n"
    "        sparse_topk_capacity = topk_indices_physical.shape[1]\n"
    "        empty_rows = topk_lengths == 0\n"
    "        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
    "            empty_rows, 0\n"
    "        )\n"
    "        topk_lengths = topk_lengths.clamp(min=1)\n",
)

replace_once(
    "            seq_lens=None,\n            max_seq_len=attn_metadata.topk_tokens,\n",
    "            seq_lens=topk_lengths,\n            max_seq_len=sparse_topk_capacity,\n",
)

replace_once(
    "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
    "            sparse_mla_top_k=sparse_topk_capacity,\n",
)

replace_once(
    "        return out.squeeze(1), None\n",
    "        out = out.squeeze(1)\n"
    "        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
    "        return out, None\n"
    "\n"
    "    def do_kv_cache_update(\n"
    "        self,\n"
    "        kv_c_normed: torch.Tensor,\n"
    "        k_pe: torch.Tensor,\n"
    "        kv_cache: torch.Tensor,\n"
    "        slot_mapping: torch.Tensor,\n"
    "        kv_cache_dtype: str,\n"
    "        k_scale: torch.Tensor,\n"
    "    ) -> None:\n"
    "        if self.rope_pad:\n"
    "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
    "        super().do_kv_cache_update(\n"
    "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
    "        )\n",
)

target.write_text(source)

site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
old_width = "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)"
for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
    path = site / rel
    text = path.read_text()
    if text.count(old_width) != 1:
        raise RuntimeError(f"expected one buffer_width target in {rel}")
    path.write_text(text.replace(old_width, "buffer_width = topk_tokens"))

sparse_backend = site / "v1/attention/backends/mla/flashinfer_mla_sparse.py"
text = sparse_backend.read_text()
old_block_sizes = (
    "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
    "        return [64, 256]\n"
)
if text.count(old_block_sizes) != 1:
    raise RuntimeError("expected one SM120 kernel-block-size target")
sparse_backend.write_text(
    text.replace(
        old_block_sizes,
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [64]\n",
    )
)

platform = site / "platforms/cuda.py"
text = platform.read_text()
old_align = "        return index_kpool * min(PAGED_MQA_PAGE_SIZES)\n"
if text.count(old_align) != 1:
    raise RuntimeError("expected one indexer block alignment target")
platform.write_text(
    text.replace(
        old_align,
        "        page_sizes = PAGED_MQA_PAGE_SIZES\n"
        "        capability = cls.get_device_capability()\n"
        "        if capability is not None and capability.major == 12:\n"
        "            page_sizes = tuple(p for p in page_sizes if p == 64)\n"
        "        return index_kpool * min(page_sizes)\n",
    )
)

indexer = site / "model_executor/layers/sparse_attn_indexer_kpool.py"
text = indexer.read_text()
for old, new in (
    (
        "                    expanded = expand_pools_and_append_tail(\n"
        "                        pool_ids, q_seq, index_kpool\n"
        "                    )\n",
        "                    expanded = expand_pools_and_append_tail(\n"
        "                        pool_ids[:, : select_k - 1], q_seq, index_kpool\n"
        "                    )\n",
    ),
    (
        "            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
        "            out = expand_pools_and_append_tail(\n"
        "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
        "            )\n",
    ),
):
    if text.count(old) != 1:
        raise RuntimeError(f"expected one expand-pools target: {old!r}")
    text = text.replace(old, new)
indexer.write_text(text)

warmup = site / "model_executor/warmup/kernel_warmup.py"
text = warmup.read_text()
old_sparse_warmup = (
    "    flashinfer_sparse_mla_decode_autotune_warmup(worker)\n"
    "    deepseek_v4_sparse_mla_attention_warmup(worker)\n"
)
if text.count(old_sparse_warmup) != 1:
    raise RuntimeError("expected one FlashInfer sparse-MLA warmup target")
text = text.replace(
    old_sparse_warmup,
    "    # GLM53_SKIP_FI_SPARSE_WARMUP: SM120 autotune wedges rank 0 on GB10.\n"
    "    deepseek_v4_sparse_mla_attention_warmup(worker)\n",
)
old_autotune = "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n"
if text.count(old_autotune) != 1:
    raise RuntimeError("expected one FlashInfer autotuner import")
text = text.replace(
    old_autotune,
    "    logger.info_once(\"Skipping FlashInfer autotune on SM121\")\n"
    "    return\n"
    "    from flashinfer.autotuner import AutoTuner, set_autotune_process_group\n",
)
warmup.write_text(text)

platform = site / "platforms/cuda.py"
text = platform.read_text()
old_pdl = (
    "            return False\n"
    "        return major >= 9\n"
)
if text.count(old_pdl) != 1:
    raise RuntimeError("expected one PDL capability gate")
platform.write_text(
    text.replace(
        old_pdl,
        "            return False\n"
        "        # PDL lowering races KDA state kernels on SM12x (GB10).\n"
        "        return major in (9, 10)\n",
    )
)
