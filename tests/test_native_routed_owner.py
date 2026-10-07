"""CPU ownership controls; these do not qualify native CUDA arithmetic."""
import dataclasses
import gc
from pathlib import Path
import sys
import weakref

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tessera import native_window_moe as nwm
from tessera import routed_fused as rf
from tessera.errors import GrammarError
from tessera.window_gemm_grouped import PreparedGroupedWindowGemm

SUPPORTED = rf.fused_routed_window_supported


def projection(rows=512, cols=128, experts=2):
    words = rows * cols // 8
    return PreparedGroupedWindowGemm(
        words_all=torch.zeros(experts, words, dtype=torch.int32),
        table_all=torch.empty(0, dtype=torch.bfloat16),
        codes_all=torch.arange(rf.TABLE_ENTRIES, dtype=torch.int32).to(torch.uint8)
            .expand(experts, -1).clone(),
        native_all=torch.arange(256, dtype=torch.int32).to(torch.uint8)
            .expand(experts, -1).clone(),
        scale_all=torch.ones(experts, rows),
        runs_all=torch.tensor([[[4, 0, cols, 0]]]*experts, dtype=torch.int32),
        init_all=torch.zeros(experts, cols, dtype=torch.int32),
        has_init=torch.zeros(experts, dtype=torch.int32),
        word_off=torch.arange(experts, dtype=torch.int32) * words,
        tile_words=torch.full((experts,), 16*4*cols, dtype=torch.int32),
        total_words=torch.full((experts,), words, dtype=torch.int32),
        run_off=torch.arange(experts+1, dtype=torch.int64),
        perm_all=torch.arange(cols, dtype=torch.int32).expand(experts, -1).clone(),
        rows=rows, cols=cols, experts=experts, window_bits=rf.WINDOW_BITS,
        family='e4m3', block_m=32, block_n=64, block_k=64)


@pytest.fixture
def native_constructor(monkeypatch):
    # Isolate ownership construction while exercising actual lookup/descriptor
    # composition and native views. CUDA/device/toolchain admission is separate.
    monkeypatch.setattr(rf, 'fused_routed_window_supported', lambda *args: None)
    monkeypatch.setattr(rf, 'library_for', lambda family: 'e4m3mma')
    monkeypatch.setattr(rf, '_ext', lambda library: object())
    monkeypatch.setattr(rf, "_make_dispatch_resources", lambda device: rf._DispatchResources(
        (object(), object()), object(), (object(), object()),
        torch.empty(0, dtype=torch.float32, device=device)))


def native_owner(packed):
    return packed.native_owner()


def test_success_retires_unheld_compact_planes_preserves_required_storage(native_constructor):
    roles = [projection() for _ in range(3)]
    refs = {f'{index}.{field}': weakref.ref(getattr(role, field))
            for index, role in enumerate(roles) for field in ('codes_all', 'native_all', 'perm_all')}
    required = [(role.words_all.data_ptr(), role.init_all.data_ptr(), role.scale_all.data_ptr())
                for role in roles]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    packed.adapter()
    owner = native_owner(packed)
    del packed, roles
    gc.collect()
    assert all(ref() is None for ref in refs.values())
    assert required == [(role.words_all.data_ptr(), role.init_all.data_ptr(), role.scale_all.data_ptr())
                        for role in (owner.gate, owner.up, owner.down)]
    assert owner.adapter() is owner.adapter()
    assert all('codes_all' not in name and 'native_all' not in name and 'perm_all' not in name
               for name, tensor in owner.named_tensors())


def test_external_original_bundles_and_views_remain_usable(native_constructor):
    roles = [projection() for _ in range(3)]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    held_codes = roles[0].codes_all[:1]
    snapshot = held_codes.clone()
    ref = weakref.ref(roles[0].codes_all)
    packed.adapter()
    owner = native_owner(packed)
    assert owner is not packed
    assert owner.gate is not roles[0]
    assert ref() is roles[0].codes_all
    assert torch.equal(held_codes, snapshot)
    assert roles[0].perm_all is not None
    assert packed.gate is roles[0]


def test_external_view_alone_keeps_storage_until_its_last_reference(native_constructor):
    roles = [projection() for _ in range(3)]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    held_codes = roles[0].codes_all[:1]
    snapshot = held_codes.clone()
    ref = weakref.ref(roles[0].codes_all)
    packed.adapter()
    owner = native_owner(packed)
    del packed, roles
    gc.collect()
    assert ref() is not None
    assert torch.equal(held_codes, snapshot)
    assert owner.gate.codes_all is None
    del held_codes
    gc.collect()
    assert ref() is None


def test_retired_projection_cannot_enter_compact_execution(native_constructor):
    packed = nwm.PackedWindowMoeBundles(*[projection() for _ in range(3)], family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    packed.adapter()
    owner = native_owner(packed)
    with pytest.raises(GrammarError, match='retired|native projection'):
        owner.gate(torch.empty(0, owner.gate.cols, dtype=torch.bfloat16),
                   torch.empty(0, 1, dtype=torch.int32), torch.empty(0, 1))


def test_retired_projection_cannot_be_recomposed(native_constructor, monkeypatch):
    packed = nwm.PackedWindowMoeBundles(*[projection() for _ in range(3)], family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    owner = native_owner(packed)
    monkeypatch.setattr(rf, 'fused_routed_window_supported', SUPPORTED)
    reason = rf.fused_routed_window_supported(owner.gate, owner.up, owner.down)
    assert 'retired' in reason
    with pytest.raises(GrammarError, match='retired'):
        rf.FusedRoutedWindowMoE.from_bundles(owner.gate, owner.up, owner.down,
                                           expert_classes=owner.expert_classes)


@pytest.mark.parametrize('failure', [RuntimeError, GrammarError])
def test_late_descriptor_failure_keeps_original_owner(native_constructor, monkeypatch, failure):
    roles = [projection() for _ in range(3)]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    original = rf.projection_tables
    calls = []

    def descriptor(bundle):
        calls.append(bundle)
        if len(calls) == 3:
            raise failure('third projection failed')
        return original(bundle)

    monkeypatch.setattr(rf, 'projection_tables', descriptor)
    with pytest.raises(failure, match="third projection failed"):
        packed.adapter()
    assert "_adapter" not in packed.__dict__
    assert all(held is role and role.codes_all is not None and role.perm_all is not None
               for held, role in zip((packed.gate, packed.up, packed.down), roles))



def storage_union(tensors):
    allocations = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        allocations[(tensor.device, storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(allocations.values())


@pytest.mark.parametrize('library', ['e4m3mma', 'e4m3', 'value'])
def test_owner_bytes_charge_selected_storage_once_and_include_counters(
        native_constructor, monkeypatch, library):
    family = 'value' if library == 'value' else 'e4m3'
    roles = [projection() for _ in range(3)]
    if family == 'value':
        roles = [dataclasses.replace(role, family='value', arithmetic='folded',
                 table_all=torch.zeros(role.experts, rf.TABLE_ENTRIES, dtype=torch.bfloat16),
                 codes_all=torch.empty(0, dtype=torch.uint8),
                 native_all=torch.empty(0, dtype=torch.uint8)) for role in roles]
    monkeypatch.setattr(rf, 'library_for', lambda family: library)
    packed = nwm.PackedWindowMoeBundles(*roles, family=family, expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    adapter = packed.adapter()
    expected_dtype = torch.uint8 if library == 'e4m3mma' else torch.int16
    assert adapter.table_gate.dtype == expected_dtype
    # The complete caller-held owner still has its compact inputs. Aliased
    # BF16 table views hold one allocation, and counters are native storage.
    expected = storage_union([t for _, t in packed.named_tensors()] + [adapter.counters])
    assert packed.resident_bytes() == expected
    owner = native_owner(packed)
    expected = storage_union([t for _, t in owner.named_tensors()] + [adapter.counters])
    assert owner.resident_bytes() == expected
    assert dict(adapter.named_tables())['routed_classes.counters'] is adapter.counters


def test_a_retained_slice_charges_the_whole_backing_allocation():
    roles = [projection() for _ in range(3)]
    gate = roles[0]
    backing = torch.empty(3, gate.words_all.shape[1], dtype=torch.int32)
    roles[0] = dataclasses.replace(gate, words_all=backing[:gate.experts])
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3', expert_classes=[{"start": 0, "end": 2, "q256": {"w13": [1024, 1024], "w2": [1024]}}])
    expected = storage_union(t for _, t in packed.named_tensors())
    assert packed.resident_bytes() == expected
