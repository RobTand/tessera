"""Research TP2 loader ownership; native stock TP2 arithmetic is a separate gate."""
from __future__ import annotations

import pytest
import torch

from tessera.serving import moe_route
from tessera.serving.scheme import validate_tessera_moe_scheme
from tests.test_serving_moe_route import _stack, _encode
from tests.test_serving_moe_selected import stub_runtime, _layer

H, N, E = 64, 32, 3


@pytest.fixture(scope='module')
def wires():
    # Uniform cap-rate columns can split freely; the source-bound GLM design
    # oracle separately covers q512 full-superblock cuts and nonzero states.
    return _stack(experts=E, hidden=H, inter=N)


def layer_for(rank):
    layer = _layer()
    parallel = layer.moe_config.moe_parallel_config
    for field,value in dict(tp_size=2,tp_rank=rank,ep_size=1,dp_size=1,
                            pcp_size=1,sp_size=1,use_ep=False,enable_eplb=False).items():
        setattr(parallel,field,value)
    layer.moe_config.defer_moe_finalize = False
    return layer


def method_for(wires, layer):
    config = moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2,
                                                expected_tensor_parallel_size=2)
    return moe_route.build_tessera_moe_method(wires[2], 'm', 'resident', layer,
                                             research_selected=config)


def load(method,layer,wires):
    method.create_weights(layer,E,H,N//2,torch.bfloat16)
    for expert in range(E):
        for shard,blob in [('w1',wires[0][expert][0]),('w3',wires[0][expert][1]),('w2',wires[1][expert])]:
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            param.weight_loader(param,torch.frombuffer(bytearray(blob),dtype=torch.uint8),'wire',shard,expert)
    method.process_weights_after_loading(layer)


@pytest.mark.parametrize('rank',[0,1])
def test_native_loader_shape_produces_rank_local_packed_tiles(wires,stub_runtime,rank):
    layer = layer_for(rank)
    method = method_for(wires,layer)
    load(method,layer,wires)
    assert not dict(layer.named_parameters())
    assert layer.tessera_rows == N and layer.tessera_columns == H
    assert method.moe_kernel is None and method.moe_quant_config is None
    ids = torch.tensor([2,0,2,1],dtype=torch.int32)
    tiles = method._packed.decode(ids,max_experts_per_chunk=2)
    lo,hi = rank*N//2,(rank+1)*N//2
    for slot,expert in enumerate(ids.tolist()):
        ref = wires[3][expert]
        expected13 = torch.cat([ref[r]['weight'][lo:hi] for r in ('gate','up')])
        assert torch.equal(tiles.w13_weight[slot].view(torch.uint8),expected13.view(torch.uint8))
        assert torch.equal(tiles.w2_weight[slot].view(torch.uint8),ref['down']['weight'][:,lo:hi].contiguous().view(torch.uint8))
        expected_scale = torch.cat([ref[r]['weight_scale'].flatten()[lo:hi] for r in ('gate','up')])
        assert torch.equal(tiles.w13_weight_scale[slot].flatten(),expected_scale)
        assert torch.equal(tiles.w2_weight_scale[slot].flatten(),ref['down']['weight_scale'].flatten())
    empty = method._packed.decode(ids[:0],max_experts_per_chunk=2)
    assert empty.w13_weight.shape == (0,N,H) and empty.w2_weight.shape == (0,H,N//2)
    owner_bytes = method.research_resident_bytes()
    x = torch.randn(2,H,generator=torch.Generator().manual_seed(7))
    routing = torch.tensor([[2,0],[1,2]],dtype=torch.int32)
    weights = torch.tensor([[.2,.8],[.7,.3]])
    got = method.apply(layer,x,weights,routing,None,None)
    expected = torch.zeros_like(x)
    for token in range(2):
        for choice in range(2):
            ref = wires[3][int(routing[token,choice])]
            gate,up = [ref[r]['weight'][lo:hi].float()*ref[r]['weight_scale'].reshape(-1,1)[lo:hi] for r in ('gate','up')]
            down = ref['down']['weight'][:,lo:hi].float()*ref['down']['weight_scale'].reshape(-1,1)
            expected[token] += weights[token,choice]*(down@(torch.nn.functional.silu(gate@x[token])* (up@x[token])))
    torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-5)
    assert stub_runtime[1][-1]['kwargs']['global_num_experts'] == E
    assert method.research_resident_bytes() == owner_bytes
    assert method.moe_kernel is None and method.moe_quant_config is None


@pytest.mark.parametrize('value',[0,3,4,True,2.0,'2'])
def test_only_explicit_tp1_or_tp2_config_is_accepted(value):
    with pytest.raises(ValueError,match='tensor_parallel_size'):
        moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=1,expected_tensor_parallel_size=value)


@pytest.mark.parametrize('field,value', [('tp_size',1),('tp_rank',2),('ep_size',2),('dp_size',2),
    ('pcp_size',2),('sp_size',2),('use_ep',True),('enable_eplb',True)])
def test_unmeasured_parallel_contracts_refuse_before_allocation(wires,stub_runtime,field,value):
    layer = layer_for(0)
    setattr(layer.moe_config.moe_parallel_config,field,value)
    with pytest.raises(ValueError,match='research selected'):
        method_for(wires,layer)
    assert not dict(layer.named_parameters())


def test_deferred_finalize_refuses_at_construction_and_before_decode(wires,stub_runtime,monkeypatch):
    layer = layer_for(0)
    layer.moe_config.defer_moe_finalize = True
    with pytest.raises(ValueError,match='deferred'):
        method_for(wires,layer)
    layer.moe_config.defer_moe_finalize = False
    method = method_for(wires,layer)
    load(method,layer,wires)
    layer.moe_config.defer_moe_finalize = True
    monkeypatch.setattr(method._packed,'decode',lambda *a,**k:pytest.fail('deferred finalize reached decoder'))
    with pytest.raises(ValueError,match='deferred'):
        method.apply(layer,torch.zeros(1,H),torch.ones(1,2),torch.zeros(1,2,dtype=torch.int32),None,None)


def test_tp2_refuses_runtime_padding_and_original_wire_shape_disagreement(wires,stub_runtime):
    layer = layer_for(0)
    method = method_for(wires,layer)
    with pytest.raises(ValueError,match='intermediate size'):
        method.create_weights(layer,E,H,N//2+256,torch.bfloat16)
    assert not dict(layer.named_parameters())
    declared = validate_tessera_moe_scheme(wires[2],'m')
    smaller,_ = _encode(N//2,H,'gate_proj',42)
    first = [list(pair) for pair in wires[0]]
    first[0][0] = smaller
    with pytest.raises(ValueError):
        moe_route.prepare_tessera_packed_moe_experts(
            {'w13':first,'w2':[[b] for b in wires[1]]},declared,'m',device='cpu',tp_rank=0,tp_size=2)


def test_tp2_requires_stock_final_reduction(wires,stub_runtime):
    layer = layer_for(0)
    layer.moe_config.skip_final_all_reduce = True
    with pytest.raises(ValueError,match='final all-reduce'):
        method_for(wires,layer)


@pytest.mark.parametrize('rank', [0, 1])
def test_tp2_construction_does_not_allocate_checkpoint_sized_wire_staging(wires, stub_runtime, rank):
    # Stock constructs ALL owners before any load callback. A second rank
    # cannot halve memory if both first allocate the full checkpoint wire bank.
    layers = [layer_for(rank) for _ in range(4)]
    methods = [method_for(wires, layer) for layer in layers]
    for method, layer in zip(methods, layers):
        method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    assert sum(p.numel() for layer in layers for p in layer.parameters()) == 0
    assert all(set(dict(layer.named_parameters())) == {'w13_wire', 'w2_wire'} for layer in layers)


@pytest.mark.parametrize('rank', [0, 1])
def test_tp2_each_load_keeps_only_rank_local_packed_projection(wires, stub_runtime, monkeypatch, rank):
    import weakref
    import tessera.serving.fp8_route as fp8

    layer = layer_for(rank)
    method = method_for(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    prepared, source_refs, parsed_refs = [], [], []
    real_prepare = fp8.prepare_tessera_fp8_module
    def observed(roles, device=None):
        assert len(roles) == 1
        name, parsed = roles[0]
        rows, cols = parsed.unit.body_bits.shape
        assert (rows, cols) == ((H, N // 2) if name == 'down_proj' else (N // 2, H))
        result = real_prepare(roles, device=device)
        prepared.append(result.wire_bytes_resident())
        parsed_refs.append(weakref.ref(parsed.unit.body_bits))
        return result
    monkeypatch.setattr(fp8, 'prepare_tessera_fp8_module', observed)
    # Load in role-major reverse-expert order: never wait for an adjacent
    # gate/up pair while retaining unsliced source projections.
    for shard, blobs in [('w3', [x[1] for x in wires[0]]),
                         ('w2', wires[1]), ('w1', [x[0] for x in wires[0]])]:
        for expert in reversed(range(E)):
            loaded = torch.frombuffer(bytearray(blobs[expert]), dtype=torch.uint8)
            source_refs.append(weakref.ref(loaded))
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            param.weight_loader(param, loaded, 'wire', shard, expert)
            del loaded
            assert all(ref() is None for ref in source_refs)
            assert len(prepared) == len(source_refs), 'each original must be sliced during its load callback'
            assert all(ref() is None for ref in parsed_refs), 'unpacked per-role temporaries must be released'
    method.process_weights_after_loading(layer)
    assert all(ref() is None for ref in parsed_refs)
    assert method.research_resident_bytes() > 0
    ids = torch.tensor([2, 0, 1], dtype=torch.int32)
    actual = method._packed.decode(ids, max_experts_per_chunk=2)
    lo, hi = rank * N // 2, (rank + 1) * N // 2
    for slot, expert in enumerate(ids.tolist()):
        ref = wires[3][expert]
        assert torch.equal(actual.w13_weight[slot].view(torch.uint8),
            torch.cat([ref[role]['weight'][lo:hi] for role in ('gate', 'up')]).view(torch.uint8))
        assert torch.equal(actual.w2_weight[slot].view(torch.uint8),
            ref['down']['weight'][:, lo:hi].contiguous().view(torch.uint8))


@pytest.mark.parametrize('defect', ['missing', 'stride', 'corrupt', 'wrong_role'])
def test_tp2_incremental_intake_preserves_original_integrity_gates(wires, stub_runtime, defect):
    import copy
    from tessera.errors import GrammarError, SchemaError

    fixture = list(wires)
    fixture[2] = copy.deepcopy(wires[2])
    if defect == 'stride':
        fixture[2]['groups']['w13']['wire_stride'] += 1
    layer = layer_for(0)
    method = method_for(fixture, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    if defect in ('corrupt', 'wrong_role'):
        blob = bytearray(wires[0][0][1 if defect == 'wrong_role' else 0])
        if defect == 'corrupt':
            blob[-1] ^= 1
        with pytest.raises((ValueError, GrammarError, SchemaError)):
            layer.w13_wire.weight_loader(layer.w13_wire,
                torch.frombuffer(blob, dtype=torch.uint8), 'wire', 'w1', 0)
        assert method._rank_local_intake is None
    elif defect == 'stride':
        for expert in range(E):
            for shard, blob in [('w1', wires[0][expert][0]),
                                ('w3', wires[0][expert][1]), ('w2', wires[1][expert])]:
                param = layer.w2_wire if shard == 'w2' else layer.w13_wire
                param.weight_loader(param, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
                                    'wire', shard, expert)
        with pytest.raises(GrammarError, match='max'):
            method.process_weights_after_loading(layer)
    else:
        with pytest.raises(GrammarError, match='zero|nonpositive|length'):
            method.process_weights_after_loading(layer)
    assert method._research_phase == 'failed'
    with pytest.raises(RuntimeError, match='not loading'):
        method.process_weights_after_loading(layer)
    assert method._packed is None


def test_tp2_incremental_intake_refuses_duplicate_wrong_parameter_and_late_wires(wires, stub_runtime):
    layer = layer_for(0)
    method = method_for(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    blob = torch.frombuffer(bytearray(wires[0][0][0]), dtype=torch.uint8)
    with pytest.raises(ValueError, match='belong to group'):
        layer.w13_wire.weight_loader(layer.w2_wire, blob, 'wire', 'w1', 0)
    with pytest.raises(ValueError, match='global expert ID'):
        layer.w13_wire.weight_loader(layer.w13_wire, blob, 'wire', 'w1', E)
    layer.w13_wire.weight_loader(layer.w13_wire, blob, 'wire', 'w1', 0)
    with pytest.raises(ValueError, match='already loaded'):
        layer.w13_wire.weight_loader(layer.w13_wire, blob, 'wire', 'w1', 0)
    method._research_phase = 'ready'
    with pytest.raises(RuntimeError, match='not loading'):
        layer.w13_wire.weight_loader(layer.w13_wire, blob, 'wire', 'w1', 0)


def test_cpu_created_tp2_intake_uses_current_cuda_device_before_first_parse(wires, stub_runtime, monkeypatch):
    # Stock permits an explicit load device, and the ordinary finalizer already
    # promotes non-CUDA staging to the current CUDA device. Incremental intake
    # must resolve the same target before it owns its first packed projection.
    layer = layer_for(0)
    method = method_for(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    assert layer.w13_wire.device.type == 'cpu'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    observed = []
    class StopBeforeDeviceWork(Exception):
        pass
    def observe(blob, declared, target, device):
        observed.append(device)
        raise StopBeforeDeviceWork
    monkeypatch.setattr(moe_route, 'parse_tessera_expert_blob', observe)
    with pytest.raises(StopBeforeDeviceWork):
        layer.w13_wire.weight_loader(layer.w13_wire,
            torch.frombuffer(bytearray(wires[0][0][0]), dtype=torch.uint8), 'wire', 'w1', 0)
    assert observed == [torch.device('cuda', 1)]


def test_tp2_intake_refuses_device_change_after_first_packed_projection(wires):
    declared = validate_tessera_moe_scheme(wires[2], 'm')
    intake = moe_route._RankLocalPackedIntake(declared, 'm', torch.device('cpu'), 0, 2)
    wire = torch.frombuffer(bytearray(wires[0][0][0]), dtype=torch.uint8)
    intake.load('w13', 0, 0, wire, device=torch.device('cpu'))
    with pytest.raises(ValueError, match='cannot change device'):
        intake.load('w13', 0, 1, wire, device=torch.device('meta'))
    assert intake.device == torch.device('cpu')
    assert intake.prepared['w13'][1][0] is None
