"""Small actual vLLM SM120 implementation gate; no model/serve window."""
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import pytest
import torch
pytest.importorskip('vllm')
from tessera.serving import mla_sparse_sm120 as module
from tessera.serving import mla_prefill

@pytest.fixture(scope="module")
def workload():
    # Match the repository's file-based experiment loading convention. Do not
    # add the experiment directory to sys.path or expose its generic library.
    path = Path(__file__).resolve().parents[1] / "experiments/mla_prefill/d1_bench.py"
    spec = importlib.util.spec_from_file_location("_mla_prefill_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native_library():
    # Bind the existing Torch cache to the retained candidate before this gate.
    # Exercise the production factory and forbid any replacement native build.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mla_prefill, 'load',
                      lambda **kw: pytest.fail('runtime gate requires the retained MLA DSO'))
        module.library_for_device.cache_clear()
        try:
            library=module.library_for_device(torch.device('cuda',torch.cuda.current_device()))
            assert library.build.manifest['p0_buffers'] is True
            assert library.build.manifest['p0_wrong_pass'] is False
            yield library
        finally:
            module.library_for_device.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual SM121 CUDA runtime')
@pytest.mark.parametrize('T,E',[(512,512),(2048,8192)])
def test_actual_eager_override_matches_stock_and_graph_capture_stays_stock(T,E,monkeypatch,workload,native_library):
    make_cache,make_indices=workload.make_cache,workload.make_indices
    HEADS,D_LATENT=workload.HEADS,workload.D_LATENT
    module.flags.reset_for_tests(module.FLAG);monkeypatch.setenv(module.FLAG,'1')
    assert module.qualified_source_refusal() is None
    library=native_library
    assert module.library_for_device(torch.device('cuda',torch.cuda.current_device())) is library
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
