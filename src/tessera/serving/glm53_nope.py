"""Explicit research-only GLM5-next NoPE MLA on the pinned stock SM121 runtime.

This uses vLLM's public CUSTOM backend registration. It does not modify stock
classes or attest a serving cell. FlashInfer owns the native attention kernel;
vLLM owns packed cache writes, metadata, sparse index conversion, and workspace.
"""
from __future__ import annotations

import functools
import hashlib
import sys
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


#: The V2 model-runner source the decode-graph path was measured on
#: (tessera#508): the pinned image's runner with vLLM #57317 (upstream
#: 70df48dc3d01) backported, so the kpool-tail KV group takes no generic slot
#: mapping. Measured in image
#: localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b,
#: whose vLLM differs from the stock-runner image's in this one file.
_GRAPH_RUNNER = "v1/worker/gpu/model_runner.py"
_GRAPH_RUNNER_SHA256 = "1c30b8c0d3ffc96172cba57965cb7c3648156ce693ea6b3232edd2f36a9d1781"
#: The same file as the pinned image ships it, before the backport.
_STOCK_RUNNER_SHA256 = "39a5edd7b1e76b13c039be4b22a72dc512a17c3b78b9fa9aa34158ce9a7d78c3"


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


@functools.cache
def _runner_sha256() -> str:
    import vllm

    return hashlib.sha256((Path(vllm.__file__).parent / _GRAPH_RUNNER).read_bytes()).hexdigest()


def _graph_mode(compilation) -> CUDAGraphMode:
    """The CUDA-graph mode vLLM will run for this backend.

    ``resolve_cudagraph_mode_and_sizes`` settles the mode after the backend is
    chosen, from the least capable metadata builder. This backend's builder
    (``FlashInferMLASparseMetadataBuilder``) supports ``UNIFORM_BATCH``, never
    ``ALWAYS``, so a ``FULL`` request loses its mixed-batch half there: to
    ``FULL_AND_PIECEWISE`` when attention is a splitting op, and to
    ``FULL_DECODE_ONLY`` otherwise. The gate judges the mode that will run.
    """
    mode = compilation.cudagraph_mode
    if mode is None:
        return CUDAGraphMode.NONE
    if mode == CUDAGraphMode.FULL:
        return (CUDAGraphMode.FULL_AND_PIECEWISE
                if compilation.splitting_ops_contain_attention()
                else CUDAGraphMode.FULL_DECODE_ONLY)
    return mode


def _padded_token_counts(config, graph: CUDAGraphMode) -> list[int]:
    """Token counts that replay a larger captured graph than their own.

    vLLM's V2 runner (``CudaGraphManager._init_candidates``) runs a batch of n
    tokens in the smallest captured graph of at least n tokens, so every n
    below the largest capture size that is not itself a capture size runs
    padded. The decode half (``FULL_DECODE_ONLY``, and ``FULL_AND_PIECEWISE``'s
    FULL graphs) captures the sizes up to ``max_num_seqs``, one token per
    request (a drafter is refused, so there is no wider decode query); the
    mixed half (``PIECEWISE``, and ``FULL_AND_PIECEWISE``'s piecewise graphs)
    captures every size.
    """
    sizes = sorted(set(config.compilation_config.cudagraph_capture_sizes or ()))
    padded: set[int] = set()
    if graph in (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE):
        decode = [n for n in sizes if n <= config.scheduler_config.max_num_seqs]
        if decode:
            padded.update(n for n in range(1, decode[-1] + 1) if n not in decode)
    if graph in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE) and sizes:
        padded.update(n for n in range(1, sizes[-1] + 1) if n not in sizes)
    return sorted(padded)


#: The op implementations compilation mode NONE resolves, which the eager
#: reference runs: vLLM appends custom op ``"all"`` unless inductor compiles
#: (``config/vllm.py``), and CUDA orders both IR ops ``vllm_c`` before
#: ``native`` without codegen (``platforms/cuda.py``,
#: ``get_default_ir_op_priority``). Every other compilation mode defaults to
#: ``"none"`` and ``native``.
_EAGER_IR_OPS = ("rms_norm", "fused_add_rms_norm")


def _op_implementation_gap(config) -> str | None:
    """How this configuration's op implementations differ from eager's, or None."""
    custom = list(config.compilation_config.custom_ops or ())
    gaps = []
    if "all" not in custom or any(str(op).startswith("-") for op in custom):
        gaps.append(f"custom_ops resolves to {custom}, not ['all']")
    priority = config.kernel_config.ir_op_priority
    for op in _EAGER_IR_OPS:
        order = list(getattr(priority, op, None) or ())
        if order[:1] != ["vllm_c"]:
            gaps.append(f"IR op {op} resolves to {order}, not ['vllm_c', 'native']")
    return "; ".join(gaps) or None


def eager_equivalence_gap(config) -> str | None:
    """Why this admitted configuration's outputs are not claimed equal to eager's.

    Admission (``_config_reason``) and this claim are separate verdicts. A
    configuration admitted here runs correctly; ``None`` further claims it
    runs eager's arithmetic, so a quality measured on an eager serve holds for
    it. A string names what differs, with its measurement: that serve's
    outputs are valid but are a different computation, and its quality must
    be measured on it rather than inherited from eager. On the tessera#508
    stub, eager itself is not repeat-exact (the fused-MoE finalize reduces
    with atomics), so "equal" means every greedy choice and top-20 logprob
    list of the equality suite is one an eager serve of the same image also
    produced.
    """
    compilation = config.compilation_config
    gaps = []
    if compilation.mode != CompilationMode.NONE:
        ops = _op_implementation_gap(config)
        if ops:
            gaps.append(
                f"compilation mode {compilation.mode.name} compiles nothing for this model "
                f"(vLLM: it does not support torch.compile) but selects other op "
                f"implementations than eager: {ops}. Each switch alone moved all 48 "
                "completions of the equality suite outside eager's outcomes (custom_ops "
                "'none': top-20 logprobs by up to 1.0608 nats on an identical prefix, 25 "
                "completions changed a generated token; IR ops native: up to 0.0644, 9 "
                "changed a token); with custom_ops ['all'] and both IR ops ['vllm_c', "
                "'native'] all 48 were eager outcomes (tessera#508)")
    graph = _graph_mode(compilation)
    padded = _padded_token_counts(config, graph)
    if padded:
        gaps.append(
            f"CUDA-graph mode {graph.name} replays token counts {padded} in larger "
            "captured graphs, and vLLM picks some kernels by token count (the mHC "
            "TileLang op splits its reduction 8 ways below 8 tokens and 4 ways from 8 "
            "to 16, tilelang.py mhc_fused_post_pre_tilelang): a batch of 5 replayed "
            "at 8 moved all 5 completions outside eager's outcomes, top-20 logprobs by "
            "up to 0.98081 nats on an identical prefix, and 2 of them changed a "
            "generated token (tessera#508). Capture every size from 1 to the largest "
            "to run eager's arithmetic")
    return "; ".join(gaps) or None


def _execution_reason(config) -> str | None:
    """Admit the execution modes a receipt shows run correctly; refuse the rest by name.

    Whether an admitted mode also runs eager's arithmetic is the separate
    claim ``eager_equivalence_gap`` makes.
    """
    compilation = config.compilation_config
    mode = compilation.mode
    graph = _graph_mode(compilation)
    if mode == CompilationMode.STOCK_TORCH_COMPILE:
        return ("Tessera GLM53 NoPE refuses compilation mode STOCK_TORCH_COMPILE: the engine "
                "fails to start (Dynamo raises while tracing the model, tessera#508); "
                "serve with compilation mode NONE")
    if mode not in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE,
                    CompilationMode.DYNAMO_TRACE_ONCE):
        return (f"Tessera GLM53 NoPE refuses compilation mode {mode.name}: no receipt "
                "measures it (tessera#508); serve with compilation mode NONE")
    if graph == CUDAGraphMode.NONE:
        return None
    if mode == CompilationMode.DYNAMO_TRACE_ONCE:
        return ("Tessera GLM53 NoPE refuses CUDA graphs under compilation mode "
                "DYNAMO_TRACE_ONCE: it is measured without graphs only (tessera#508); "
                "serve graphs with compilation mode NONE")
    if mode == CompilationMode.VLLM_COMPILE and graph != CUDAGraphMode.FULL_DECODE_ONLY:
        return (f"Tessera GLM53 NoPE refuses CUDA-graph mode {graph.name} under compilation "
                "mode VLLM_COMPILE: this model is not torch-compiled, so piecewise graphs "
                "need vLLM's breakable CUDA graph, which forces compilation mode NONE (with "
                "it off, vLLM refuses to start, tessera#508); serve with compilation mode "
                "NONE, or FULL_DECODE_ONLY")
    if not config.use_v2_model_runner:
        return (f"Tessera GLM53 NoPE admits CUDA-graph mode {graph.name} on vLLM's V2 model "
                "runner only; the V1 runner's graph path is not measured (tessera#508)")
    if config.speculative_config is not None:
        return (f"Tessera GLM53 NoPE refuses CUDA-graph mode {graph.name} with speculative "
                "decoding: no drafter graph path is measured (the qualifying stub carries "
                "no MTP layer, tessera#508); serve the drafter eager")
    runner = _runner_sha256()
    if runner == _GRAPH_RUNNER_SHA256:
        return None
    if runner == _STOCK_RUNNER_SHA256:
        return (f"Tessera GLM53 NoPE refuses CUDA-graph mode {graph.name} on the stock V2 "
                "runner: it maps the kpool-tail KV group through the generic slot-mapping "
                "kernel, which reads that group's 32-entry block-table row by absolute "
                "position, so a prompt past 128 tokens reads other rows and, further on, "
                "past the table (compute-sanitizer: invalid global reads at "
                "block_table.py:344, tessera#508; a two-chunk 3649-token prefill raised an "
                "illegal memory access in 6 of 10 decode-graph serves, tessera#581). Serve "
                f"the vLLM #57317 backport ({_GRAPH_RUNNER} sha256 {_GRAPH_RUNNER_SHA256})")
    return (f"Tessera GLM53 NoPE admits CUDA-graph mode {graph.name} on one measured runner "
            f"({_GRAPH_RUNNER} sha256 {_GRAPH_RUNNER_SHA256}); this runtime's is {runner}")


#: Equivalence verdicts already reported by this process (one line per distinct verdict).
_REPORTED: set = set()


def _report_equivalence(config) -> None:
    """Say once per process whether this serve runs eager's arithmetic, and if not, why."""
    gap = eager_equivalence_gap(config)
    if gap in _REPORTED:
        return
    _REPORTED.add(gap)
    graph = _graph_mode(config.compilation_config).name
    if gap is None:
        print(f"Tessera GLM53 NoPE: compilation mode {config.compilation_config.mode.name}, "
              f"CUDA-graph mode {graph}: runs eager's arithmetic (tessera#508)",
              file=sys.stderr, flush=True)
    else:
        print(f"Tessera GLM53 NoPE: WARNING: this serve's outputs are not claimed equal to "
              f"eager's; measure its quality on it: {gap}", file=sys.stderr, flush=True)


def _config_reason(config) -> str | None:
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
    return _execution_reason(config)


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
        config = get_current_vllm_config()
        reason = _config_reason(config)
        if reason:
            raise RuntimeError(reason)
        _report_equivalence(config)
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
