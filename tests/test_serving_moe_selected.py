"""CPU reference decoder controls and routed-constructor protocol boundaries.

The vLLM constructor stub never loads or executes native weights. Actual class
loading, mapped routing and captured execution are checked in the serving image.
"""
from __future__ import annotations

import enum
import sys
import types
import weakref

import pytest

torch = pytest.importorskip('torch')

from tessera.serving import moe_route
from tessera.serving.scheme import validate_tessera_moe_scheme
from test_serving_moe_route import _stack, EXPERTS, HIDDEN, INTER  # the tests dir is on sys.path (conftest)


@pytest.fixture(scope='module')
def bf16_wires():
    from tessera.alphabet import BF16_GRID
    from tessera.export import encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.unit_artifact import read_unit_artifact

    w13_blobs, w2_blobs, expected = [], [], []
    for expert in range(2):
        parts = {}
        for projection, rows, cols in (('gate_proj', INTER, HIDDEN),
                                       ('up_proj', INTER, HIDDEN),
                                       ('down_proj', HIDDEN, INTER)):
            weight = torch.randn(rows, cols,
                                 generator=torch.Generator().manual_seed(600 + expert * 3 + len(parts)))
            written, _unit, _forests = encode_linear_planes(
                weight, grid=BF16_GRID, q256=512, name=projection,
                window_bits=8, verify=False)
            parts[projection] = (pack_fused([(projection, rows, written.blob)]),
                                 read_unit_artifact(written.blob).to(torch.bfloat16))
        w13_blobs.append([parts['gate_proj'][0], parts['up_proj'][0]])
        w2_blobs.append([parts['down_proj'][0]])
        expected.append((torch.cat([parts['gate_proj'][1], parts['up_proj'][1]]),
                         parts['down_proj'][1]))
    scheme = {
        'family': 'TESSERA_BF16', 'structure': 'routed_moe', 'grid': 'BF16',
        'body': 'WINDOW', 'plane': 'CHANNEL', 'experts': 2,
        "expert_ids": [0, 1],
        "expert_classes": [{"start": 0, "end": 2,
                            "q256": {"w13": [512, 512], "w2": [512]}}],
        'groups': {
            'w13': {'rows': 2 * INTER, 'columns': HIDDEN, 'q256': 512,
                    'wire_stride': max(len(blob) for pair in w13_blobs for blob in pair),
                    'roles': [['gate_proj', INTER], ['up_proj', INTER]]},
            'w2': {'rows': HIDDEN, 'columns': INTER, 'q256': 512,
                   'wire_stride': max(len(pair[0]) for pair in w2_blobs),
                   'roles': [['down_proj', HIDDEN]]}}}
    return w13_blobs, w2_blobs, scheme, expected


def test_bf16_selected_owner_matches_actual_folded_wire_weights(bf16_wires):
    w13_blobs, w2_blobs, scheme, expected = bf16_wires
    owner = moe_route.prepare_tessera_packed_bf16_moe_experts(
        {'w13': w13_blobs, 'w2': w2_blobs},
        validate_tessera_moe_scheme(scheme, 'm'), 'm', device='cpu')
    ids = torch.tensor([1, 0, 1], dtype=torch.int32)
    selected = owner.decode_folded(ids, max_experts_per_chunk=2)
    for slot, expert in enumerate(ids.tolist()):
        assert torch.equal(selected.w13_weight[slot], expected[expert][0])
        assert torch.equal(selected.w2_weight[slot], expected[expert][1])
    assert owner.resident_bytes() > 0


def test_bf16_selected_owner_tp2_cuts_original_wires_into_exact_rank_tiles(bf16_wires):
    w13_blobs, w2_blobs, scheme, expected = bf16_wires
    ids = torch.tensor([1, 0], dtype=torch.int32)
    for rank in (0, 1):
        owner = moe_route.prepare_tessera_packed_bf16_moe_experts(
            {'w13': w13_blobs, 'w2': w2_blobs},
            validate_tessera_moe_scheme(scheme, 'm'), 'm', device='cpu',
            tp_rank=rank, tp_size=2)
        selected = owner.decode_folded(ids, max_experts_per_chunk=2)
        lo, hi = rank * (INTER // 2), (rank + 1) * (INTER // 2)
        for slot, expert in enumerate(ids.tolist()):
            full13, full2 = expected[expert]
            local13 = torch.cat([full13[lo:hi], full13[INTER + lo:INTER + hi]])
            assert torch.equal(selected.w13_weight[slot], local13)
            assert torch.equal(selected.w2_weight[slot], full2[:, lo:hi])


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
    # Only the framework constructor is supplied; no Tessera reader or
    # execution path is replaced. Device execution belongs to the image tests.
    names = ('vllm', 'vllm.config', 'vllm.model_executor', 'vllm.model_executor.layers',
             'vllm.model_executor.layers.fused_moe',
             'vllm.model_executor.layers.fused_moe.fused_moe_method_base',
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
    utils = modules['vllm.model_executor.utils']
    utils.set_weight_attrs = lambda param, attrs: [setattr(param,k,v) for k,v in attrs.items()]


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






_MISSING_PARALLEL_FIELD = object()
# Discover dimensions and flags from the existing constructor seam, rather
# than maintain a second topology roster. The tests pin exact integers equal
# to one and literal False, independently of the guard's implementation.
_PARALLEL_DEFAULTS = vars(_layer().moe_config.moe_parallel_config)
_SIDE_DEGREES = tuple(field for field in _PARALLEL_DEFAULTS
                      if field.endswith('_size') and field != 'tp_size')
_PARALLEL_FLAGS = tuple(field for field, value in _PARALLEL_DEFAULTS.items()
                       if type(value) is bool)


def _parallel_method(scheme, layer, expected_tp):
    config = moe_route.ResearchSelectedMoeConfig(
        max_experts_per_chunk=2, expected_tensor_parallel_size=expected_tp)
    return moe_route.build_tessera_moe_method(
        scheme, 'm', 'resident', layer, research_selected=config)


def _parallel_field(layer, field, value):
    parallel = layer.moe_config.moe_parallel_config
    if value is _MISSING_PARALLEL_FIELD:
        delattr(parallel, field)
    else:
        setattr(parallel, field, value)





@pytest.mark.parametrize('expected_tp,live_tp', [(1, 2), (2, 1), (2, 3)])
def test_research_builder_refuses_degree_mismatch(
        original_wires, stub_runtime, expected_tp, live_tp):
    layer = _layer()
    layer.moe_config.moe_parallel_config.tp_size = live_tp
    with pytest.raises(ValueError, match=f'TP{expected_tp}/EP1/DP1/PCP1/SP1'):
        _parallel_method(original_wires[2], layer, expected_tp)


@pytest.mark.parametrize('expected_tp', [1, 2])
@pytest.mark.parametrize('alias', ['bool', 'float', 'string', 'none'])
def test_research_builder_refuses_degree_type_aliases(
        original_wires, stub_runtime, expected_tp, alias):
    layer = _layer()
    value = {'bool': bool(expected_tp), 'float': float(expected_tp),
             'string': str(expected_tp), 'none': None}[alias]
    layer.moe_config.moe_parallel_config.tp_size = value
    with pytest.raises(ValueError, match=f'TP{expected_tp}/EP1/DP1/PCP1/SP1'):
        _parallel_method(original_wires[2], layer, expected_tp)


@pytest.mark.parametrize('field', _SIDE_DEGREES)
@pytest.mark.parametrize('value', [2, True, 1.0, None,
                         pytest.param(_MISSING_PARALLEL_FIELD, id='missing')])
def test_research_builder_requires_explicit_single_rank_side_dimensions(
        original_wires, stub_runtime, field, value):
    layer = _layer()
    _parallel_field(layer, field, value)
    with pytest.raises(ValueError, match='TP1/EP1/DP1/PCP1/SP1'):
        _parallel_method(original_wires[2], layer, 1)


@pytest.mark.parametrize('field', _PARALLEL_FLAGS)
@pytest.mark.parametrize('value', [True, 0, None,
                         pytest.param(_MISSING_PARALLEL_FIELD, id='missing')])
def test_research_builder_requires_literal_false_parallel_flags(
        original_wires, stub_runtime, field, value):
    layer = _layer()
    _parallel_field(layer, field, value)
    with pytest.raises(ValueError, match='no EP/EPLB'):
        _parallel_method(original_wires[2], layer, 1)


@pytest.mark.parametrize('expected_tp', [1, 2])
@pytest.mark.parametrize('boundary', ['below', 'past'])
def test_research_builder_refuses_rank_outside_declared_degree(
        original_wires, stub_runtime, expected_tp, boundary):
    layer = _layer()
    parallel = layer.moe_config.moe_parallel_config
    parallel.tp_size = expected_tp
    parallel.tp_rank = -1 if boundary == 'below' else expected_tp
    with pytest.raises(ValueError):
        _parallel_method(original_wires[2], layer, expected_tp)


@pytest.mark.parametrize('value', [False, 0.0, '0', None,
                         pytest.param(_MISSING_PARALLEL_FIELD, id='missing')])
def test_research_builder_requires_explicit_integer_rank(
        original_wires, stub_runtime, value):
    layer = _layer()
    _parallel_field(layer, 'tp_rank', value)
    with pytest.raises(ValueError):
        _parallel_method(original_wires[2], layer, 1)


@pytest.mark.parametrize('expected_tp', [1, 2])
def test_research_builder_refuses_deferred_finalize(
        original_wires, stub_runtime, expected_tp):
    layer = _layer()
    layer.moe_config.moe_parallel_config.tp_size = expected_tp
    layer.moe_config.defer_moe_finalize = True
    with pytest.raises(ValueError, match='refuse deferred finalize'):
        _parallel_method(original_wires[2], layer, expected_tp)


def test_research_builder_refuses_skipping_final_reduction_at_two_ranks(
        original_wires, stub_runtime):
    layer = _layer()
    layer.moe_config.moe_parallel_config.tp_size = 2
    layer.moe_config.skip_final_all_reduce = True
    with pytest.raises(ValueError):
        _parallel_method(original_wires[2], layer, 2)




def test_production_streamed_gate_is_not_an_alias_for_research(original_wires):
    with pytest.raises(ValueError):
        moe_route.build_tessera_moe_method(original_wires[2],'m','streamed',_layer())




@pytest.mark.parametrize('mode,graph', [('COMPILE','NONE'),('NONE','FULL')])
def test_standalone_selected_context_refuses_compilation_or_graphs(stub_runtime,original_wires,monkeypatch,mode,graph):
    compilation=types.ModuleType('vllm.config.compilation')
    compilation.CompilationMode=enum.Enum('CompilationMode',['NONE','COMPILE'])
    compilation.CUDAGraphMode=enum.Enum('CUDAGraphMode',['NONE','FULL'])
    monkeypatch.setitem(sys.modules,'vllm.config.compilation',compilation)
    config=types.SimpleNamespace(model_config=None,compilation_config=types.SimpleNamespace(
        mode=compilation.CompilationMode[mode],cudagraph_mode=compilation.CUDAGraphMode[graph]))
    monkeypatch.setattr(sys.modules['vllm.config'],'get_current_vllm_config',lambda:config)
    with pytest.raises(ValueError,match='explicit eager'):_build(original_wires[2],_layer())


def test_model_selected_context_still_requires_explicit_enforce_eager(stub_runtime,original_wires,monkeypatch):
    config=types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=False))
    monkeypatch.setattr(sys.modules['vllm.config'],'get_current_vllm_config',lambda:config)
    with pytest.raises(ValueError,match='require enforce_eager'):_build(original_wires[2],_layer())
