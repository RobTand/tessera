"""Runtime-scoped causal plugin registration test; pinned stock vLLM required."""
import pytest
pytest.importorskip('vllm')
from tessera.serving import flags, register
from vllm.v1.attention.backends.registry import AttentionBackendEnum

def test_explicit_mask_skip_flag_registers_stock_enum(monkeypatch):
    backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    previous=backend.get_path() if backend.is_overridden() else None
    try:
        backend.clear_override();flags.reset_for_tests('TESSERA_RESEARCH_MLA_MASK_SKIP')
        monkeypatch.delenv('TESSERA_RESEARCH_GLM53_NOPE',raising=False)
        monkeypatch.setenv('TESSERA_RESEARCH_MLA_MASK_SKIP','1')
        register()
        assert backend.is_overridden(), 'explicit mask-skip flag did not register stock enum'
        assert backend.get_path()=='tessera.serving.mla_sparse_sm120.TesseraMLASparseSM120Backend'
    finally:
        backend.clear_override();flags.reset_for_tests('TESSERA_RESEARCH_MLA_MASK_SKIP')
        if previous:
            from vllm.v1.attention.backends.registry import register_backend
            register_backend(backend,previous)

@pytest.fixture
def impl(monkeypatch):
    from tessera.serving import mla_sparse_sm120 as module
    module.flags.reset_for_tests(module.FLAG)
    monkeypatch.setenv(module.FLAG,'1')
    monkeypatch.setattr(module,'qualified_source_refusal',lambda:None)
    monkeypatch.setattr(module.telemetry,'emit_route',lambda *a,**k:None)
    instance=object.__new__(module.TesseraMLASparseSM120Impl)
    instance._tessera_mla_context=(True,object())
    instance.num_heads=32;instance.kv_lora_rank=512;instance.qk_nope_head_dim=256
    instance.qk_rope_head_dim=0;instance.kv_scale_format='arbitrary_fp32';instance.scale=1/16
    yield module,instance
    module.flags.reset_for_tests(module.FLAG)

@pytest.mark.parametrize('scope,capturing',[(False,False),(True,True)])
def test_decode_mixed_or_capture_never_enters_native(impl,monkeypatch,scope,capturing):
    import torch
    module,instance=impl;instance._tessera_mla_context=(scope,object())
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:capturing)
    sentinel=object();calls=[]
    monkeypatch.setattr(module.FlashInferMLASparseSM120Impl,'_run_mqa_kernel',lambda self,*args:calls.append(args) or sentinel)
    monkeypatch.setattr(module,'library_for_device',lambda *args:pytest.fail('unsupported route entered native'))
    q=torch.empty(2048,32,512,dtype=torch.bfloat16)
    assert instance._run_mqa_kernel(q,object(),object()) is sentinel
    assert len(calls)==1 and calls[0][0] is q


def test_unknown_stock_source_stays_stock(impl,monkeypatch):
    import torch
    module,instance=impl
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(module,'qualified_source_refusal',lambda:'unknown source')
    sentinel=object()
    monkeypatch.setattr(module.FlashInferMLASparseSM120Impl,'_run_mqa_kernel',lambda self,*args:sentinel)
    assert instance._run_mqa_kernel(torch.empty(512,32,512),object(),object()) is sentinel


def test_stock_calibrated_non_mg_plan_never_enters_native(impl,monkeypatch):
    import torch
    from types import SimpleNamespace
    from flashinfer.mla._sparse_mla_sm120 import _prepared
    module,instance=impl
    monkeypatch.setattr(instance,'_refusal',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(_prepared,'_functional_plan',lambda *args:SimpleNamespace(plan=SimpleNamespace(inspect=lambda:{'numeric_route':'fp8','implementation':'swapab','merge':'direct','variant':4})))
    monkeypatch.setattr(module,'library_for_device',lambda *args:pytest.fail('non-MG plan entered native'))
    sentinel=object();monkeypatch.setattr(module.FlashInferMLASparseSM120Impl,'_run_mqa_kernel',lambda self,*args:sentinel)
    assert instance._run_mqa_kernel(torch.empty(512,32,512,dtype=torch.bfloat16),object(),object()) is sentinel


@pytest.mark.parametrize('decodes,decode_tokens,prefills,has_prefill,expected',[
    (0,0,1,True,True),(1,1,1,True,False),(1,2048,0,False,False),(0,0,0,False,False)])
def test_prefill_scope_comes_from_metadata_and_is_restored(impl,monkeypatch,decodes,decode_tokens,prefills,has_prefill,expected):
    from types import SimpleNamespace
    module,instance=impl;instance.index_group=object();previous=instance._tessera_mla_context
    seen=[]
    def forward(self,*args):
        seen.append(self._tessera_mla_context[0]);raise RuntimeError('stock forward failed')
    monkeypatch.setattr(module.FlashInferMLASparseSM120Impl,'forward_mqa',forward)
    metadata=SimpleNamespace(num_decodes=decodes,num_decode_tokens=decode_tokens,num_prefills=prefills,prefill=object() if has_prefill else None)
    with pytest.raises(RuntimeError,match='stock forward failed'):instance.forward_mqa(object(),object(),metadata,object())
    assert seen==[expected];assert instance._tessera_mla_context is previous


def test_registration_default_off_does_not_replace_stock(monkeypatch):
    from tessera.serving import mla_sparse_sm120 as module
    backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    previous=backend.get_path() if backend.is_overridden() else None
    try:
        backend.clear_override();module.flags.reset_for_tests(module.FLAG);monkeypatch.delenv(module.FLAG,raising=False)
        assert module.install() is False;assert not backend.is_overridden()
    finally:
        backend.clear_override();module.flags.reset_for_tests(module.FLAG)
        if previous:
            from vllm.v1.attention.backends.registry import register_backend
            register_backend(backend,previous)


def test_registration_refuses_competing_override(monkeypatch):
    from tessera.serving import mla_sparse_sm120 as module
    from vllm.v1.attention.backends.registry import register_backend
    backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    previous=backend.get_path() if backend.is_overridden() else None
    try:
        backend.clear_override();module.flags.reset_for_tests(module.FLAG);monkeypatch.setenv(module.FLAG,'1')
        monkeypatch.setattr(module,'qualified_source_refusal',lambda:None)
        register_backend(backend,'other.backend.Class')
        with pytest.raises(RuntimeError,match='another stock-backend override'):module.install()
    finally:
        backend.clear_override();module.flags.reset_for_tests(module.FLAG)
        if previous:register_backend(backend,previous)


def test_compiled_forward_keeps_stock_scope_without_gpu_queries(impl,monkeypatch):
    import torch
    module,instance=impl;previous=instance._tessera_mla_context
    monkeypatch.setattr(torch.compiler,'is_compiling',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:pytest.fail('compiled path queried GPU capture'))
    sentinel=object();monkeypatch.setattr(module.FlashInferMLASparseSM120Impl,'forward_mqa',lambda self,*args:sentinel)
    assert instance.forward_mqa(object(),object(),object(),object()) is sentinel
    assert instance._tessera_mla_context is previous
