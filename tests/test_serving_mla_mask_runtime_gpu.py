"""Small actual vLLM SM120 implementation gate; no model/serve window."""
from types import SimpleNamespace
from pathlib import Path
import os
import pytest
import torch
pytest.importorskip('vllm')
from tessera.serving import mla_sparse_sm120 as module
from tessera.serving.mla_prefill import MlaPrefillLibrary

@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual SM121 CUDA runtime')
@pytest.mark.parametrize('T,E',[(512,512),(2048,8192)])
def test_actual_eager_override_matches_stock_and_graph_capture_stays_stock(T,E,monkeypatch):
    from d1_bench import make_cache,make_indices,HEADS,D_LATENT
    module.flags.reset_for_tests(module.FLAG);monkeypatch.setenv(module.FLAG,'1')
    assert module.qualified_source_refusal() is None
    library=MlaPrefillLibrary(Path(os.environ['MLA_RUNTIME_TEST_OUT'])/'build')
    monkeypatch.setattr(module,'library_for_device',lambda device:library)
    impl=object.__new__(module.TesseraMLASparseSM120Impl)
    impl.num_heads=32;impl.kv_lora_rank=512;impl.qk_nope_head_dim=256;impl.qk_rope_head_dim=0
    impl.kv_scale_format='arbitrary_fp32';impl.scale=1/16;impl._workspace_buffer=None
    impl._tessera_mla_context=(True,SimpleNamespace(prefix='runtime_gpu.mla'))
    gen=torch.Generator(device='cuda').manual_seed(1000+E+100000*T)
    kv=make_cache(E,gen,'cuda');indices=make_indices(E,T,gen,'cuda','pools')
    q=(torch.randn(T,HEADS,D_LATENT,generator=gen,device='cuda')*2).to(torch.bfloat16)
    assert impl._refusal(q,kv,indices) is None
    stock=module.FlashInferMLASparseSM120Impl._run_mqa_kernel(impl,q,kv,indices)
    calls=[];original=library.call
    monkeypatch.setattr(library,'call',lambda *args:calls.append(args[0]) or original(*args))
    candidate=impl._run_mqa_kernel(q,kv,indices);torch.cuda.synchronize()
    assert calls==[1]
    assert torch.equal(candidate.view(torch.uint8),stock.view(torch.uint8))
    calls.clear()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=impl._run_mqa_kernel(q,kv,indices)
    graph.replay();torch.cuda.synchronize()
    assert not calls, 'capture entered L0 instead of unchanged stock'
    assert torch.equal(captured.view(torch.uint8),stock.view(torch.uint8))
    module.flags.reset_for_tests(module.FLAG)
