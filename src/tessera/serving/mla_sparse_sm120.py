"""Default-off eager pure-prefill override of the stock SM120 sparse-MLA backend.

The stock implementation still owns conversion, residency, mixed batches and
all decode/capture paths. Only its resolved FP8 MG prefill call is replaced.
"""
from __future__ import annotations
import functools
import hashlib
import importlib
from pathlib import Path
import sys

import torch
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

from . import contract, flags, telemetry
from .mla_prefill import MlaPrefillLibrary

FLAG = "TESSERA_RESEARCH_MLA_MASK_SKIP"
BACKEND_PATH = "tessera.serving.mla_sparse_sm120.TesseraMLASparseSM120Backend"
LIBRARY_PREFIX = "tessera_mla_prefill_"
STOCK_OBJECT = {"backend": "FLASHINFER_MLA_SPARSE_SM120", "kernel": "sparse_mla_prefill_mg_kernel"}
# Read from immutable image5be13705, the same source as the numerical gates.
STOCK_SOURCES = {
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120": "102ca08793d567f95598eefeb34c3f6ec50b3b9d704f162d1402b97b12b5777b",
    "flashinfer.mla._sparse_mla_sm120._prepared": "bda9a9d24387bb1ed410b89695deb0bb526267d4b5871647488fb645c67672a3",
    "flashinfer.mla._sparse_mla_sm120._policy": "cb1d76de87342ad939b149367b4c200d59dead049fb1330be28ff379ebae7bd2",
}
STOCK_HEADER_TREE_SHA256 = "b553db1f7f4518dd81d04ad6d28d75eefbb0c157c6c81dbc5b630595fd4be142"

#: The dispatch words this override emits through ``telemetry.emit_route``.
#: One home for the emit side: ``_emit`` and its callers name these, and
#: the trace audit in ``tools/tessera_attest.py`` pins each value to this
#: source, so a renamed word breaks the audit instead of slipping past it.
MASK_SKIP_POLICY = "mla_mask_skip_eager_dispatch"
MASK_SKIP_KIND = "attention_backend"
MASK_SKIP_CONTRACT = "stock_kernel_overrides"
NATIVE_DECODER = "native_mg_mask_skip"
STOCK_DECODER = "stock"
NATIVE_SYMBOL = "tessera_mla_prefill_mg_l0"
STOCK_SYMBOL = "stock"
NATIVE_SCHEDULE = "mg_mask_skip_pass_buffers"
STOCK_SCHEDULE = "stock_mg"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.cache
def qualified_source_refusal():
    for name, expected in STOCK_SOURCES.items():
        module = importlib.import_module(name)
        if _digest(module.__file__) != expected:
            return f"unqualified stock source {name}"
    from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
    root = Path(FLASHINFER_INCLUDE_DIR)/"flashinfer/attention/sparse_mla_sm120"
    rows = [str(path.relative_to(root))+":"+_digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()]
    if hashlib.sha256("\n".join(rows).encode()).hexdigest() != STOCK_HEADER_TREE_SHA256:
        return "unqualified stock sparse-MLA header tree"
    return None


def install():
    if not flags.latched_bool(FLAG, meaning="eager pure-prefill sparse MLA masked-tile override"):
        return False
    refusal = contract.stock_kernel_override_refusal(
        FLAG, kind="attention_backend", overrides=STOCK_OBJECT,
        loaded_by=__name__, library_prefix=LIBRARY_PREFIX)
    if refusal:
        raise RuntimeError(refusal)
    refusal = qualified_source_refusal()
    if refusal:
        print(f"Tessera MLA mask skip stays stock: {refusal}", file=sys.stderr, flush=True)
        return False
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    if backend.is_overridden() and backend.get_path() != BACKEND_PATH:
        raise RuntimeError("Tessera MLA mask skip refuses another stock-backend override")
    register_backend(backend, BACKEND_PATH)
    return True


@functools.cache
def library_for_device(device):
    # One compiled owner per process/device, shared by attention instances.
    with torch.cuda.device(device):
        return MlaPrefillLibrary(None, p0_buffers=True)


class TesseraMLASparseSM120Backend(FlashInferMLASparseSM120Backend):
    # Inherited get_name retains stock identity for KV layout and warmup.
    @staticmethod
    def get_impl_cls():
        return TesseraMLASparseSM120Impl


class TesseraMLASparseSM120Impl(FlashInferMLASparseSM120Impl):
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if torch.compiler.is_compiling():
            return super().forward_mqa(q,kv_c_and_k_pe_cache,attn_metadata,layer)
        previous = getattr(self, "_tessera_mla_context", None)
        self._tessera_mla_context = (
            attn_metadata.num_decode_tokens == 0
            and attn_metadata.num_decodes == 0
            and attn_metadata.num_prefills > 0
            and attn_metadata.prefill is not None
            and not isinstance(self.index_group, HiSparseMLAIndexGroup), layer)
        try:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        finally:
            self._tessera_mla_context = previous

    def _refusal(self, q, kv, indices):
        context = getattr(self, "_tessera_mla_context", None)
        if not context or not context[0]:
            return "decode, mixed or staged prefill uses stock"
        if torch.compiler.is_compiling():
            return "Torch compilation uses stock; eager prefill only"
        if torch.cuda.is_current_stream_capturing():
            return "CUDA graph capture uses stock; eager prefill only"
        if not flags.latched_bool(FLAG, meaning="eager pure-prefill sparse MLA masked-tile override"):
            return "override is disabled"
        if qualified_source_refusal():
            return qualified_source_refusal()
        if (self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim,
                self.qk_rope_head_dim, self.kv_scale_format, self.scale) != (32,512,256,0,"arbitrary_fp32",1/16):
            return "unqualified model geometry or scale"
        if (q.ndim != 3 or q.shape[0] <= 64 or q.shape[1:] != (32,512)
                or q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous()
                or kv.ndim != 3 or kv.shape[1:] != (64,656) or kv.dtype != torch.uint8
                or not kv.is_contiguous() or kv.device != q.device
                or indices.shape != (q.shape[0],2176) or indices.dtype != torch.int32
                or indices.device != q.device or not indices.is_contiguous()
                or indices.data_ptr() % 16):
            return "unqualified tensor geometry, dtype, device or alignment"
        if torch.cuda.get_device_capability(q.device) != (12,1):
            return "unqualified GPU capability"
        return None

    def _emit(self, q, symbol, reason=None):
        # These are eager HOST dispatch counts. Graph replays are never claimed.
        context = getattr(self, "_tessera_mla_context", None)
        if not context or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return
        telemetry.emit_route(context[1], kind=MASK_SKIP_KIND,
            policy=MASK_SKIP_POLICY, symbol=symbol,
            shape=f"T{q.shape[0]}:H{self.num_heads}:D{self.kv_lora_rank}",
            contract=MASK_SKIP_CONTRACT, state="served", reason=reason,
            decoder=STOCK_DECODER if reason else NATIVE_DECODER,
            kernel_schedule=STOCK_SCHEDULE if reason else NATIVE_SCHEDULE)

    def _run_mqa_kernel(self, q, kv_cache, topk_indices_physical):
        refusal = self._refusal(q, kv_cache, topk_indices_physical)
        if refusal:
            self._emit(q, STOCK_SYMBOL, refusal)
            return super()._run_mqa_kernel(q, kv_cache, topk_indices_physical)
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[0],32,dtype=torch.float32,device=q.device)
        # Ask FlashInfer's OWN cached/calibrated resolver; do not reconstruct its
        # policy. Calibration can change MG to another numerical route.
        from flashinfer.mla._sparse_mla_sm120 import _prepared
        tensors=(q,kv_cache,topk_indices_physical,output,None,None,None,None,None,lse,None,None)
        resolved=_prepared._functional_plan(tensors,3,False,False).plan.inspect()
        if (str(resolved["numeric_route"]),str(resolved["implementation"]),
                str(resolved["merge"]),int(resolved["variant"])) != ("fp8","mg","direct",2):
            self._emit(q,STOCK_SYMBOL,"stock resolver selected another numerical route")
            return super()._run_mqa_kernel(q,kv_cache,topk_indices_physical)
        library=getattr(self,"_tessera_mla_library",None)
        if library is None:
            library=self._tessera_mla_library=library_for_device(q.device)
        library.call(1,q,kv_cache,topk_indices_physical,output,lse)
        self._emit(q,NATIVE_SYMBOL)
        return output
