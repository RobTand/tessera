"""Research TP2 native loader ownership: one validated original wire becomes one
rank-local packed projection, placed on its group's ``WindowUnitAxis`` inside
its own load callback.

The compact native lane is CUDA-only -- the window kernels build the packed
planes -- so this file runs in the pinned vLLM image on a device.  The
materialising fallback is not a thing a test mock may resurrect.  What is
pinned here is the loader's half of the native contract:

* construction allocates no checkpoint-sized wire bank (zero-byte loader
  anchors only) and no packed plane until its first callback;
* every original wire is parsed by the shared compact boundary once, cut to
  this rank and released inside its own callback; each part's stacked tensor
  is allocated once and only filled afterwards;
* finishing copies no packed expert: the adapter is wrapped from the SoA the
  callbacks already filled, so its byte count is the intake's;
* a CPU-created wire anchor resolves the CURRENT CUDA device before its first
  parse, and a later device change refuses;
* the packed bundles and the adapter are the rank-local slice of the
  materialising reference -- geometry, CHANNEL scale bytes, and served
  arithmetic;
* malformed bytes and wrong declarations refuse at their own seam: a corrupt
  container (checksum), a wrong role or rung, a missing/over-long group, a
  wrong parameter, a duplicate callback, or a late one.
"""
from __future__ import annotations

import types

import pytest
import torch

from tessera.serving import moe_route
from tessera.serving.scheme import validate_tessera_moe_scheme
# Sibling test modules by their own names: ``tests/conftest.py`` puts this
# directory on ``sys.path``, while ``tests.<module>`` needs the checkout root
# there too and resolves to whatever top-level ``tests`` package an
# interpreter happens to carry (the GB10 venv carries one).
from test_serving_moe_route import _stack, _encode, EXPERTS, HIDDEN, INTER
from test_native_window_moe_method import _SharedSpy, _fp8_reference, _native_layer

H, N, E = HIDDEN, INTER, EXPERTS

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="the native window MoE runs CUDA kernels")


@pytest.fixture(scope='module')
def wires():
    # Uniform cap-rate columns can split freely; the source-bound GLM design
    # oracle separately covers q512 full-superblock cuts and nonzero states.
    return _stack(experts=E, hidden=H, inter=N)


def _eager():
    from vllm.config import set_current_vllm_config

    return set_current_vllm_config(
        types.SimpleNamespace(model_config=types.SimpleNamespace(enforce_eager=True)))


def _layer_for(rank):
    """The real vLLM layer shape, at the research TP2 contract."""
    return _native_layer(tp_rank=rank, tp_size=2)


def _method(wires, layer):
    config = moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=2,
                                                expected_tensor_parallel_size=2)
    with _eager():
        return moe_route.build_tessera_moe_method(wires[2], 'm', 'resident', layer,
                                                 research_selected=config)


def _load_all(layer, w13_blobs, w2_blobs):
    for expert in range(E):
        for shard, blob in (('w1', w13_blobs[expert][0]), ('w3', w13_blobs[expert][1]),
                            ('w2', w2_blobs[expert])):
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            assert param.weight_loader(
                param, torch.frombuffer(bytearray(blob), dtype=torch.uint8),
                'wire', shard, expert, return_success=True)


def _materialised(wires, rank, ids):
    """The materialising owner's rank-local tiles: the reference the packed
    bundles must reproduce.  This is the retained (non-native) decode path --
    a second decoder with its own slicer -- not the window path read back."""
    owner = moe_route.prepare_tessera_packed_moe_experts(
        {'w13': wires[0], 'w2': [[b] for b in wires[1]]},
        validate_tessera_moe_scheme(wires[2], 'm'), 'm',
        device='cuda', tp_rank=rank, tp_size=2)
    return owner.decode(ids, max_experts_per_chunk=2)


def _assert_rank_local_scales(packed, ref, ids):
    # ``decode`` returns slots in the order of ``ids``; the bundles are in
    # global expert order.
    for slot, expert in enumerate(ids.tolist()):
        assert torch.equal(packed.gate.scale_all[expert],
                           ref.w13_weight_scale[slot][:N // 2].reshape(-1))
        assert torch.equal(packed.up.scale_all[expert],
                           ref.w13_weight_scale[slot][N // 2:].reshape(-1))
        assert torch.equal(packed.down.scale_all[expert],
                           ref.w2_weight_scale[slot].reshape(-1))


def _fp8_reference_weight_on_input(reference, x, ids, weights, tp_rank, tp_size):
    """The materialising reference with the route weight in gemm1's fp32
    accumulator -- the placement actual stock ``fused_experts`` uses when
    ``apply_router_weight_on_input=True`` (vLLM's MUL_ROUTED_WEIGHT): the
    activation and the stage-2 quant consume weighted values, and the down
    stage does not weight again."""
    from vllm import _custom_ops  # noqa: F401  (registers torch.ops._C)

    local_inter = INTER // tp_size
    lo, hi = tp_rank * local_inter, (tp_rank + 1) * local_inter

    def _quant(row):
        q = torch.empty((1, row.numel()), dtype=torch.float8_e4m3fn, device=row.device)
        s = torch.empty((1, 1), dtype=torch.float32, device=row.device)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(q, row.reshape(1, -1), s, None)
        return q.reshape(-1).float() * s.reshape(-1)[0]

    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        xg = _quant(x[token].float())
        for choice in range(ids.shape[1]):
            ref = reference[int(ids[token, choice])]
            weight = float(weights[token, choice])
            gate = ref["gate"]["weight"][lo:hi].float().to(x.device) \
                * ref["gate"]["weight_scale"][lo:hi].to(x.device)
            up = ref["up"]["weight"][lo:hi].float().to(x.device) \
                * ref["up"]["weight_scale"][lo:hi].to(x.device)
            down = ref["down"]["weight"][:, lo:hi].float().to(x.device) \
                * ref["down"]["weight_scale"].to(x.device)
            act_q = _quant(torch.nn.functional.silu(weight * (gate @ xg))
                           * (weight * (up @ xg)))
            out[token] += down @ act_q
    return out.bfloat16()


@cuda
@pytest.mark.parametrize('rank', [0, 1])
def test_native_loader_shape_produces_rank_local_packed_tiles(wires, rank):
    layer = _layer_for(rank)
    method = _method(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    _load_all(layer, wires[0], wires[1])
    method.process_weights_after_loading(layer)
    assert not dict(layer.named_parameters())
    assert layer.tessera_rows == N and layer.tessera_columns == H
    assert method.moe_kernel is None and method.moe_quant_config is None
    assert layer.tessera_decoder == 'native_window_moe_compact'
    assert method._native is not None

    from tessera.native_window_moe import PackedWindowMoeBundles

    packed = method._packed
    assert isinstance(packed, PackedWindowMoeBundles)
    assert packed.experts == E and packed.family == 'e4m3'
    assert packed.device == torch.device('cuda', torch.cuda.current_device())
    # Rank-local physical shapes: half the intermediate rows for gate/up (a
    # row cut) and half the intermediate columns for down (a column cut).  No
    # bundle holds the full [2N, K] or [K, N] tile, in any plane.
    assert (packed.gate.rows, packed.gate.cols) == (N // 2, H)
    assert (packed.up.rows, packed.up.cols) == (N // 2, H)
    assert (packed.down.rows, packed.down.cols) == (H, N // 2)
    assert packed.gate.arithmetic == 'epilogue' and packed.down.arithmetic == 'epilogue'
    assert packed.gate.words_all.shape[0] == E and packed.down.words_all.shape[0] == E
    assert packed.gate.scale_all.shape == (E, N // 2)     # rung-local rows
    assert packed.down.scale_all.shape == (E, H)          # whole rows, cut columns
    assert packed.down.perm_all.shape == (E, N // 2)

    # Bytes: the CHANNEL scale plane is the materialising reference's rank
    # slice, byte for byte, and the served arithmetic is that reference's
    # arithmetic over this rank's tiles -- a wrong packed body word moves the
    # output and fails here.
    ids = torch.tensor([2, 0, 2, 1], dtype=torch.int32, device='cuda')
    _assert_rank_local_scales(packed, _materialised(wires, rank, ids), ids)

    x = (torch.randn(2, H, generator=torch.Generator().manual_seed(7))
         * 0.5).bfloat16().cuda()
    routing = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32, device='cuda')
    weights = torch.tensor([[.2, .8], [.7, .3]], device='cuda')
    got = method.apply(layer, x, weights, routing, _SharedSpy(), None)
    expected = _fp8_reference(wires[3], x, routing, weights, tp_rank=rank, tp_size=2)
    diff = (got.float() - expected.float()).abs()
    assert float(diff.max()) < 5e-2 + 2e-2 * float(expected.float().abs().max())
    # The route weight's placement is stock's: on input it multiplies gemm1's
    # fp32 accumulator (the method-level stock oracle attests that placement);
    # at this rank's cut the product must be the same placement over the
    # rank-local tiles, with no second weighting at the down stage.
    layer.apply_router_weight_on_input = True
    got_wi = method.apply(layer, x, weights, routing, _SharedSpy(), None)
    expected_wi = _fp8_reference_weight_on_input(wires[3], x, routing, weights,
                                                 tp_rank=rank, tp_size=2)
    diff = (got_wi.float() - expected_wi.float()).abs()
    assert float(diff.max()) < 5e-3 + 1e-2 * float(expected_wi.float().abs().max())
    layer.apply_router_weight_on_input = False
    empty = method.apply(layer, x[:0], weights[:0], routing[:0], None, None)
    assert tuple(empty.shape) == (0, H)
    owner_bytes = method.research_resident_bytes()
    method.apply(layer, x, weights, routing, _SharedSpy(), None)
    assert method.research_resident_bytes() == owner_bytes
    assert method.moe_kernel is None and method.moe_quant_config is None


@pytest.mark.parametrize('value', [0, 3, 4, True, 2.0, '2'])
def test_only_explicit_tp1_or_tp2_config_is_accepted(value):
    with pytest.raises(ValueError, match='tensor_parallel_size'):
        moe_route.ResearchSelectedMoeConfig(max_experts_per_chunk=1,
                                            expected_tensor_parallel_size=value)


@cuda
@pytest.mark.parametrize('field,value', [('tp_size', 1), ('tp_rank', 2), ('ep_size', 2),
    ('dp_size', 2), ('pcp_size', 2), ('sp_size', 2), ('use_ep', True), ('enable_eplb', True)])
def test_unmeasured_parallel_contracts_refuse_before_allocation(wires, field, value):
    layer = _layer_for(0)
    setattr(layer.moe_config.moe_parallel_config, field, value)
    with pytest.raises(ValueError, match='research selected'):
        _method(wires, layer)
    assert not dict(layer.named_parameters())


@cuda
def test_deferred_finalize_refuses_at_construction_and_before_decode(wires):
    # The native owner is constructed for one finalize contract (eager,
    # resident, stock final all-reduce).  The flag is read by the same
    # construction that validates the parallel contract, so a deferred layer
    # never owns a packed plane and there is no decoder a forward could reach;
    # a flip after construction is outside the published contract.
    layer = _layer_for(0)
    layer.moe_config.defer_moe_finalize = True
    with pytest.raises(ValueError, match='deferred'):
        _method(wires, layer)
    assert not dict(layer.named_parameters())
    assert not hasattr(layer, 'w13_wire')
    assert not hasattr(layer, 'tessera_decoder')


@cuda
def test_tp2_refuses_runtime_padding_and_original_wire_shape_disagreement(wires):
    from tessera.errors import GrammarError

    layer = _layer_for(0)
    method = _method(wires, layer)
    with pytest.raises(ValueError, match='intermediate size'):
        method.create_weights(layer, E, H, N // 2 + 256, torch.bfloat16)
    assert not dict(layer.named_parameters())

    # The same method, built correctly: a container whose encoded geometry is
    # not the role the sidecar declared refuses inside its own load callback,
    # before the rank-local axis ever sees a plane.
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    smaller, _ = _encode(N // 2, H, 'gate_proj', 42)
    with pytest.raises((ValueError, GrammarError)):
        layer.w13_wire.weight_loader(
            layer.w13_wire, torch.frombuffer(bytearray(smaller), dtype=torch.uint8),
            'wire', 'w1', 0)
    assert method._rank_local_intake is None and method._packed is None
    assert method._research_phase == 'failed'
    with pytest.raises(RuntimeError, match='not loading'):
        method.process_weights_after_loading(layer)


@cuda
def test_tp2_requires_stock_final_reduction(wires):
    layer = _layer_for(0)
    layer.moe_config.skip_final_all_reduce = True
    with pytest.raises(ValueError, match='final all-reduce'):
        _method(wires, layer)


@cuda
def test_native_tp2_refuses_unsupported_activation_and_swiglu_modifiers(wires):
    # The native adapter reproduces silu exactly (the method-level stock oracle
    # covers that arithmetic). Another activation or a swiglu modifier has no
    # exact implementation here and must refuse before a kernel consumes it.
    layer = _layer_for(0)
    method = _method(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    _load_all(layer, wires[0], wires[1])
    method.process_weights_after_loading(layer)
    x = torch.randn(1, H).bfloat16().cuda()
    routing = torch.tensor([[0, 1]], dtype=torch.int32, device='cuda')
    weights = torch.ones(1, 2, device='cuda')
    layer.activation = 'gelu'
    with pytest.raises(ValueError, match='silu'):
        method.apply(layer, x, weights, routing, None, None)
    layer.activation = 'silu'
    layer.swiglu_alpha = 1.1
    with pytest.raises(ValueError, match='swiglu_alpha'):
        method.apply(layer, x, weights, routing, None, None)


@cuda
@pytest.mark.parametrize('rank', [0, 1])
def test_ordinary_fp8_native_tp2_uses_the_same_rank_local_slices(wires, rank):
    # The ordinary (non-research) FP8 route takes the same native intake at
    # TP2: the epilogue CHANNEL row scale is the materialising reference's
    # rank-local bytes and the served arithmetic is that reference's over this
    # rank's cut. The stock FP8 tile is allocated at create time exactly as the
    # per-channel method allocates it and dropped when the native bundle takes
    # over.
    layer = _layer_for(rank)
    method = moe_route.build_tessera_moe_method(wires[2], 'm', 'resident', layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    assert layer.tessera_mode == 'resident'
    _load_all(layer, wires[0], wires[1])
    method.process_weights_after_loading(layer)
    assert not dict(layer.named_parameters())
    assert method._native is not None and method.moe_kernel is None
    assert layer.tessera_decoder == 'native_window_moe_compact'
    packed = method._packed
    assert (packed.gate.rows, packed.up.rows, packed.down.cols) == (N // 2, N // 2, N // 2)
    assert packed.gate.arithmetic == 'epilogue'
    ids = torch.tensor([2, 0, 1], dtype=torch.int32, device='cuda')
    _assert_rank_local_scales(packed, _materialised(wires, rank, ids), ids)
    x = (torch.randn(2, H, generator=torch.Generator().manual_seed(9))
         * 0.5).bfloat16().cuda()
    routing = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32, device='cuda')
    weights = torch.tensor([[.4, .6], [.9, .1]], device='cuda')
    got = method.apply(layer, x, weights, routing, _SharedSpy(), None)
    expected = _fp8_reference(wires[3], x, routing, weights, tp_rank=rank, tp_size=2)
    diff = (got.float() - expected.float()).abs()
    assert float(diff.max()) < 5e-2 + 2e-2 * float(expected.float().abs().max()), \
        f"rank {rank}: max abs diff {float(diff.max())}"


@cuda
@pytest.mark.parametrize('rank', [0, 1])
def test_tp2_construction_does_not_allocate_checkpoint_sized_wire_staging(wires, rank):
    # Stock constructs ALL owners before any load callback. A second rank
    # cannot halve memory if both first allocate the full checkpoint wire bank:
    # the native intake keeps zero-byte loader anchors and allocates a packed
    # plane only when a callback fills its first expert.
    layers = [_layer_for(rank) for _ in range(4)]
    methods = [_method(wires, layer) for layer in layers]
    for method, layer in zip(methods, layers):
        method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    assert sum(p.numel() for layer in layers for p in layer.parameters()) == 0
    assert all(set(dict(layer.named_parameters())) == {'w13_wire', 'w2_wire'}
               for layer in layers)
    assert all(method._rank_local_intake is not None
               and method._rank_local_intake.resident_bytes() == 0
               and method._rank_local_intake.placed_projections() == 0
               for method in methods)


@cuda
@pytest.mark.parametrize('rank', [0, 1])
def test_tp2_each_load_keeps_only_rank_local_packed_projection(wires, monkeypatch, rank):
    import weakref

    from tessera.serving import scheme as scheme_mod

    layer = _layer_for(rank)
    method = _method(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    intake = method._rank_local_intake
    assert intake is not None and intake.compact

    parsed = []
    real = scheme_mod.parse_compact_tessera_expert_blob

    def observed(blob, declared_role, target, device='cpu'):
        roles = real(blob, declared_role, target, device=device)
        assert [name for name, _ in roles] == [declared_role['roles'][0][0]]
        parsed.append(weakref.ref(roles[0][1]))
        return roles

    monkeypatch.setattr(scheme_mod, 'parse_compact_tessera_expert_blob', observed)
    first_words_id = {}
    sources = []
    # Load in role-major reverse-expert order: never wait for an adjacent
    # gate/up pair while retaining unsliced source projections.
    for shard, blobs in [('w3', [x[1] for x in wires[0]]),
                         ('w2', wires[1]), ('w1', [x[0] for x in wires[0]])]:
        for expert in reversed(range(E)):
            loaded = torch.frombuffer(bytearray(blobs[expert]), dtype=torch.uint8)
            sources.append(weakref.ref(loaded))
            param = layer.w2_wire if shard == 'w2' else layer.w13_wire
            param.weight_loader(param, loaded, 'wire', shard, expert)
            del loaded
            assert all(ref() is None for ref in sources), 'a source outlived its callback'
            assert len(parsed) == len(sources), 'each original must be parsed in its callback'
            assert all(ref() is None for ref in parsed), 'a parsed wire outlived its callback'
            for group in ('w13', 'w2'):
                for part, slot in intake.axis[group]._slots.items():
                    first_words_id.setdefault((group, part), id(slot['words']))
    assert intake.placed_projections() == len(sources) == 3 * E
    assert set(intake.axis['w13']._slots) == {'gate_proj', 'up_proj'}
    assert set(intake.axis['w2']._slots) == {'down_proj'}
    # One allocation per packed plane: each part's stacked tensor was created
    # at that part's first callback and every later callback filled it in
    # place, so expert E-1's plane is not a fresh allocation.
    for group in ('w13', 'w2'):
        for part, slot in intake.axis[group]._slots.items():
            assert id(slot['words']) == first_words_id[(group, part)]
            assert slot['words'].shape[0] == E
    assert intake.resident_bytes() > 0

    method.process_weights_after_loading(layer)
    assert all(ref() is None for ref in parsed)
    assert method._native is not None
    packed = method._packed
    assert not hasattr(packed, 'decode') and not hasattr(packed, 'decode_folded'), \
        'the retired materialising owner returned'
    ids = torch.tensor([2, 0, 1], dtype=torch.int32, device='cuda')
    _assert_rank_local_scales(packed, _materialised(wires, rank, ids), ids)


@cuda
@pytest.mark.parametrize('defect', ['missing', 'stride', 'corrupt', 'wrong_role', 'rung'])
def test_tp2_incremental_intake_preserves_original_integrity_gates(wires, defect):
    import copy

    from tessera.errors import GrammarError, SchemaError

    fixture = list(wires)
    fixture[2] = copy.deepcopy(wires[2])
    if defect == 'stride':
        fixture[2]['groups']['w13']['wire_stride'] += 1
    if defect == 'rung':
        fixture[2]['groups']['w13']['q256'] = 256
    layer = _layer_for(0)
    method = _method(fixture, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    blob = bytearray(wires[0][0][1 if defect == 'wrong_role' else 0])
    if defect == 'corrupt':
        blob[-1] ^= 1
    if defect in ('corrupt', 'wrong_role', 'rung'):
        with pytest.raises((ValueError, GrammarError, SchemaError)):
            layer.w13_wire.weight_loader(
                layer.w13_wire, torch.frombuffer(blob, dtype=torch.uint8), 'wire', 'w1', 0)
        assert method._rank_local_intake is None
    elif defect == 'stride':
        for expert in range(E):
            for shard, wire in [('w1', wires[0][expert][0]), ('w3', wires[0][expert][1]),
                                ('w2', wires[1][expert])]:
                param = layer.w2_wire if shard == 'w2' else layer.w13_wire
                param.weight_loader(param, torch.frombuffer(bytearray(wire), dtype=torch.uint8),
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


@cuda
def test_tp2_incremental_intake_refuses_duplicate_wrong_parameter_and_late_wires(wires):
    layer = _layer_for(0)
    method = _method(wires, layer)
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


@cuda
def test_cpu_created_tp2_intake_uses_current_cuda_device_before_first_parse(wires, monkeypatch):
    # Stock permits an explicit load device, and the ordinary finalizer already
    # promotes non-CUDA staging to the current CUDA device. Incremental intake
    # must resolve the same target before it owns its first packed projection.
    from tessera.serving import scheme as scheme_mod

    layer = _layer_for(0)
    method = _method(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    assert layer.w13_wire.device.type == 'cpu'
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    observed = []

    class StopBeforeDeviceWork(Exception):
        pass

    def observe(blob, declared_role, target, device='cpu'):
        observed.append(torch.device(device))
        raise StopBeforeDeviceWork

    monkeypatch.setattr(scheme_mod, 'parse_compact_tessera_expert_blob', observe)
    with pytest.raises(StopBeforeDeviceWork):
        layer.w13_wire.weight_loader(
            layer.w13_wire, torch.frombuffer(bytearray(wires[0][0][0]), dtype=torch.uint8),
            'wire', 'w1', 0)
    assert observed == [torch.device('cuda', 1)]


@cuda
def test_tp2_intake_refuses_device_change_after_first_packed_projection(wires):
    declared = validate_tessera_moe_scheme(wires[2], 'm')
    device = torch.device('cuda', torch.cuda.current_device())
    intake = moe_route._RankLocalPackedIntake(declared, 'm', device, 0, 2)
    wire = torch.frombuffer(bytearray(wires[0][0][0]), dtype=torch.uint8)
    intake.load('w13', 0, 0, wire, device=device)
    with pytest.raises(ValueError, match='cannot change device'):
        intake.load('w13', 0, 1, wire, device=torch.device('meta'))
    assert intake.device == device
    assert intake.placed_projections() == 1
    assert intake.resident_bytes() > 0


def _fresh_storage_bytes(fn):
    """Bytes of every storage an operator creates while ``fn`` runs.

    A view shares its input's storage and is not counted, so this is what the
    call allocates, counted at the dispatcher, with no device needed."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class Fresh(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.bytes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            given = {t.untyped_storage().data_ptr() for t in tree_leaves((args, kwargs))
                     if isinstance(t, torch.Tensor)}
            for t in tree_leaves(out):
                if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in given:
                    self.bytes += t.untyped_storage().nbytes()
            return out

    with Fresh() as fresh:
        fn()
    return fresh.bytes


@cuda
@pytest.mark.parametrize('rank', [0, 1])
def test_tp2_finishing_the_load_copies_no_packed_expert(wires, rank):
    # vLLM finishes a layer only after EVERY layer has loaded. An intake that
    # kept each projection's owner until then and stacked at finish held the
    # whole model's routed experts twice (tessera#501). Each projection is
    # placed on its expert axis inside its own callback, so finishing
    # allocates less than one expert's packed bytes: the joined scale only.
    layer = _layer_for(rank)
    method = _method(wires, layer)
    method.create_weights(layer, E, H, N // 2, torch.bfloat16)
    _load_all(layer, wires[0], wires[1])
    intake = method._rank_local_intake
    assert intake is not None
    held = intake.resident_bytes()
    assert held > 0
    allocated = _fresh_storage_bytes(lambda: method.process_weights_after_loading(layer))
    resident = method.research_resident_bytes()
    assert resident >= held
    assert allocated < resident // E, (allocated, resident)
