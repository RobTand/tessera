"""Research packed ownership and loader lifecycle; stock execution is a native control.

The CPU stub supplies only the vLLM constructor/kernel seam. Real Tessera wires,
packing, decoding, lifecycle and compact expert mapping remain the implementation.
The separate GLM container control validates the actual stock factory and kernel.
"""
from __future__ import annotations

import enum
import sys
import types
import weakref

import pytest

torch = pytest.importorskip('torch')

from tessera.serving import moe_route
from tessera.errors import GrammarError
from tessera.serving.scheme import validate_tessera_moe_scheme
from tests.test_serving_moe_route import _stack, EXPERTS, HIDDEN, INTER


@pytest.fixture(scope='module')
def original_wires():
    return _stack()


def _blobs(original_wires):
    first, second, scheme, reference = original_wires
    return {'w13': first, 'w2': [[x] for x in second]}, scheme, reference


def test_packed_preparation_never_materializes_the_full_fp8_stack(original_wires, monkeypatch):
    import tessera.decode

    original = tessera.decode.materialize_fp8
    references = []
    def per_role(*args, **kwargs):
        weight, scale = original(*args, **kwargs)
        assert weight.numel() == HIDDEN * INTER
        references.append(weakref.ref(weight))
        return weight, scale
    monkeypatch.setattr(tessera.decode, 'materialize_fp8', per_role)
    blobs, scheme, reference = _blobs(original_wires)
    owner = moe_route.prepare_tessera_packed_moe_experts(
        blobs, validate_tessera_moe_scheme(scheme, 'm'), 'm', device='cpu')
    assert len(references) == 3 * EXPERTS
    assert all(ref() is None for ref in references), 'per-role reference tiles must not stay resident'
    ids = torch.tensor([2, 0, 2, 1], dtype=torch.int32)
    first = owner.decode(ids, max_experts_per_chunk=2)
    for slot, expert in enumerate(ids.tolist()):
        ref = reference[expert]
        assert torch.equal(first.w13_weight[slot].view(torch.uint8),
                           torch.cat([ref['gate']['weight'], ref['up']['weight']]).view(torch.uint8))
        assert torch.equal(first.w13_weight_scale[slot].flatten(),
                           torch.cat([ref['gate']['weight_scale'].flatten(), ref['up']['weight_scale'].flatten()]))
        assert torch.equal(first.w2_weight[slot].view(torch.uint8), ref['down']['weight'].view(torch.uint8))
        assert torch.equal(first.w2_weight_scale[slot].flatten(), ref['down']['weight_scale'].flatten())
    empty = owner.decode(torch.empty(0, dtype=torch.int32), max_experts_per_chunk=2)
    assert empty.w13_weight.shape == (0, 2 * INTER, HIDDEN)
    assert empty.w2_weight.shape == (0, HIDDEN, INTER)
    assert owner.resident_bytes() > 0


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5, '8'])
def test_research_construction_needs_an_explicit_positive_chunk_bound(chunk):
    with pytest.raises(ValueError, match='positive integer'):
        moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=chunk)


@pytest.fixture
def stub_runtime(monkeypatch):
    names = ('vllm', 'vllm.config', 'vllm.model_executor', 'vllm.model_executor.layers',
             'vllm.model_executor.layers.fused_moe',
             'vllm.model_executor.layers.fused_moe.fused_moe_method_base',
             'vllm.model_executor.layers.fused_moe.oracle',
             'vllm.model_executor.layers.fused_moe.oracle.fp8',
             'vllm.model_executor.layers.quantization',
             'vllm.model_executor.layers.quantization.utils',
             'vllm.model_executor.layers.quantization.utils.quant_utils',
             'vllm.model_executor.utils')
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    modules['vllm.config'].get_current_vllm_config = lambda: types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=True))
    base = modules['vllm.model_executor.layers.fused_moe.fused_moe_method_base']

    class Base:
        def __init__(self, moe):
            self.moe, self.moe_kernel, self.moe_quant_config = moe, None, None
        @property
        def is_monolithic(self):
            return False
    base.FusedMoEMethodBase = Base
    fp8 = modules['vllm.model_executor.layers.fused_moe.oracle.fp8']
    fp8.Fp8MoeBackend = enum.Enum('Fp8MoeBackend', ['TRITON', 'MARLIN'])
    experts_cls = types.SimpleNamespace(is_monolithic=lambda: False)
    fp8.select_fp8_moe_backend = lambda **kwargs: (fp8.Fp8MoeBackend.TRITON, experts_cls)
    fp8.make_fp8_moe_quant_config = lambda **kwargs: kwargs
    fp8.convert_to_fp8_moe_kernel_format = lambda **kw: tuple(kw[k] for k in ('w13','w2','w13_scale','w2_scale'))
    calls = []

    class Kernel:
        is_monolithic = False
        def __init__(self, config):
            self.config = config
        def apply(self, x, w13, w2, weights, ids, **kwargs):
            # Index mapping and kwargs reach a consuming kernel, not just a spy.
            mapping = kwargs['expert_map']
            local = mapping[ids.long()]
            assert bool((local >= 0).all())
            out = torch.zeros_like(x)
            for token in range(x.shape[0]):
                for choice in range(ids.shape[1]):
                    e = int(local[token, choice])
                    first = w13[e].float() * self.config['w1_scale'][e]
                    second = w2[e].float() * self.config['w2_scale'][e]
                    gate, up = (first @ x[token].float()).chunk(2)
                    out[token] += weights[token, choice] * (second @ (torch.nn.functional.silu(gate) * up))
            calls.append({'selected':w13.shape[0], 'map':mapping.clone(), 'kwargs':kwargs})
            return out
    fp8.make_fp8_moe_kernel = lambda **kw: Kernel(kw['moe_quant_config'])
    quant = modules['vllm.model_executor.layers.quantization.utils.quant_utils']
    quant.kFp8DynamicTokenSym, quant.kFp8StaticChannelSym = object(), object()
    utils = modules['vllm.model_executor.utils']
    utils.set_weight_attrs = lambda param, attrs: [setattr(param,k,v) for k,v in attrs.items()]
    utils.replace_parameter = lambda layer,name,value: setattr(layer,name,torch.nn.Parameter(value,requires_grad=False))
    return fp8, calls


def _layer():
    layer = torch.nn.Module()
    layer.moe_config = types.SimpleNamespace(is_act_and_mul=True,
        moe_parallel_config=types.SimpleNamespace(tp_size=1, tp_rank=0, ep_size=1, dp_size=1,
            pcp_size=1, sp_size=1, use_ep=False, enable_eplb=False),
        experts_per_token=2)
    layer.activation = 'silu'
    layer.global_num_experts = EXPERTS
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    layer._expert_routing_tables = lambda: None
    return layer


def _build(scheme, layer):
    config = moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2)
    return moe_route.build_tessera_moe_method(scheme,'m','resident',layer,research_selected=config)


def _load(method, layer, original_wires):
    first, second, _, _ = original_wires
    for expert in range(EXPERTS):
        for shard, blob in [('w1',first[expert][0]),('w3',first[expert][1]),('w2',second[expert])]:
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            assert param.weight_loader(param,torch.frombuffer(bytearray(blob),dtype=torch.uint8),
                'wire',shard,expert,return_success=True)


def test_builder_retains_only_packed_owners_and_maps_each_invocation(original_wires, stub_runtime, monkeypatch):
    _, _, scheme, _ = original_wires
    layer = _layer()
    method = _build(scheme,layer)
    method.create_weights(layer,EXPERTS,HIDDEN,INTER,torch.bfloat16)
    assert set(dict(layer.named_parameters())) == {'w13_wire','w2_wire'}
    assert layer.tessera_mode == 'research_selected'
    _load(method,layer,original_wires)
    method.process_weights_after_loading(layer)
    assert not dict(layer.named_parameters())
    assert method.moe_kernel is None and method.moe_quant_config is None
    before = method.research_resident_bytes()
    assert before > 0
    recorded = []
    monkeypatch.setattr(moe_route,'emit_route',lambda *a,**k: recorded.append(k))
    x = torch.randn(2,HIDDEN)
    ids = torch.tensor([[2,0],[0,2]],dtype=torch.int32)
    weights = torch.full((2,2),.5)
    out = method.apply(layer,x,weights,ids,None,None)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert stub_runtime[1][-1]['selected'] == 2
    mapping = stub_runtime[1][-1]['map']
    assert mapping[1] == -1 and set(mapping[ids].flatten().tolist()) == {0,1}
    assert method.research_resident_bytes() == before
    assert method.moe_kernel is None and method.moe_quant_config is None
    assert recorded == [], 'research invocation must not emit a production served record'
    empty = method.apply(layer,x[:0],weights[:0],ids[:0],None,None)
    assert empty.shape == (0,HIDDEN) and len(stub_runtime[1]) == 1


def test_research_loader_rejects_incomplete_duplicate_and_late_loading(original_wires, stub_runtime):
    _, _, scheme, _ = original_wires
    layer = _layer()
    method = _build(scheme,layer)
    method.create_weights(layer,EXPERTS,HIDDEN,INTER,torch.bfloat16)
    param = layer.w13_wire
    blob = torch.frombuffer(bytearray(original_wires[0][0][0]),dtype=torch.uint8)
    param.weight_loader(param,blob,'wire','w1',0)
    with pytest.raises(ValueError,match='already loaded'):
        param.weight_loader(param,blob,'wire','w1',0)
    with pytest.raises(GrammarError, match='wire length 0'):
        method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError,match='loading'):
        param.weight_loader(param,blob,'wire','w1',1)
    with pytest.raises(RuntimeError,match='ready'):
        method.apply(layer,torch.empty(0,HIDDEN),torch.empty(0,2),torch.empty(0,2,dtype=torch.int32),None,None)


@pytest.mark.parametrize('field', ['tp_size','ep_size','dp_size'])
def test_research_builder_refuses_multirank_without_borrowing_dense_support(original_wires,stub_runtime,field):
    layer = _layer()
    setattr(layer.moe_config.moe_parallel_config,field,2)
    with pytest.raises(ValueError,match='TP1/EP1/DP1'):
        _build(original_wires[2],layer)


def test_production_streamed_gate_is_not_an_alias_for_research(original_wires):
    with pytest.raises(ValueError,match="expert route serves 'resident' only"):
        moe_route.build_tessera_moe_method(original_wires[2],'m','streamed',_layer())


@pytest.mark.parametrize('bad_id', [-1, EXPERTS])
def test_invalid_global_routing_is_refused_before_any_decode(original_wires,stub_runtime,monkeypatch,bad_id):
    layer = _layer()
    method = _build(original_wires[2],layer)
    method.create_weights(layer,EXPERTS,HIDDEN,INTER,torch.bfloat16)
    _load(method,layer,original_wires)
    method.process_weights_after_loading(layer)
    def forbidden(*args,**kwargs):
        raise AssertionError('invalid routing reached the decoder')
    monkeypatch.setattr(method._packed,'decode',forbidden)
    with pytest.raises(ValueError,match='invalid global expert ID'):
        method.apply(layer,torch.randn(1,HIDDEN),torch.ones(1,2),
                     torch.tensor([[0,bad_id]],dtype=torch.int32),None,None)


def test_research_backend_is_explicit_and_reaches_selected_owner(original_wires, stub_runtime, monkeypatch):
    with pytest.raises(ValueError, match='backend'):
        moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2, decode_backend='auto')
    assert moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2).decode_backend == 'torch'
    _, _, scheme, _ = original_wires
    layer = _layer()
    method = moe_route.build_tessera_moe_method(scheme, 'm', 'resident', layer,
        research_selected=moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2, decode_backend='triton'))
    method.create_weights(layer, EXPERTS, HIDDEN, INTER, torch.bfloat16)
    _load(method, layer, original_wires)
    method.process_weights_after_loading(layer)
    assert layer.tessera_decoder == 'research_selected_triton_window'
    calls = []
    decode = method._packed.decode
    def observed(ids, *, max_experts_per_chunk, backend):
        calls.append(backend)
        return decode(ids, max_experts_per_chunk=max_experts_per_chunk, backend='torch')
    monkeypatch.setattr(method._packed, 'decode', observed)
    method.apply(layer, torch.randn(1, HIDDEN), torch.ones(1, 2),
                 torch.tensor([[0, 2]], dtype=torch.int32), None, None)
    assert calls == ['triton']
