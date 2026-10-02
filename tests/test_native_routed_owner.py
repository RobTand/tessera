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


def native_owner(packed):
    # Exercise the old full-owner behavior for causal RED instead of failing
    # merely because the new ownership method does not exist yet.
    return packed.native_owner() if hasattr(packed, 'native_owner') else packed


def test_success_retires_unheld_compact_planes_preserves_required_storage(native_constructor):
    roles = [projection() for _ in range(3)]
    refs = {f'{index}.{field}': weakref.ref(getattr(role, field))
            for index, role in enumerate(roles) for field in ('codes_all', 'native_all', 'perm_all')}
    required = [(role.words_all.data_ptr(), role.init_all.data_ptr(), role.scale_all.data_ptr())
                for role in roles]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3')
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
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3')
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
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3')
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


@pytest.mark.parametrize('failure', ['unsupported', 'build'])
def test_refusal_or_build_failure_keeps_complete_compact_fallback(native_constructor, monkeypatch, failure):
    roles = [projection() for _ in range(3)]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3')
    fallback = object()
    monkeypatch.setattr(nwm, 'native_window_moe_from_bundles', lambda *args, **kwargs: fallback)
    if failure == 'unsupported':
        monkeypatch.setattr(rf, 'fused_routed_window_supported', lambda *args: 'unsupported')
    else:
        monkeypatch.setattr(rf, '_ext', lambda library: (_ for _ in ()).throw(RuntimeError('no toolchain')))
    assert packed.adapter() is fallback
    assert native_owner(packed) is packed
    assert all(packed_role is original and original.codes_all is not None and original.perm_all is not None
               for packed_role, original in zip((packed.gate, packed.up, packed.down), roles))


def test_retired_projection_cannot_enter_compact_execution(native_constructor):
    packed = nwm.PackedWindowMoeBundles(*[projection() for _ in range(3)], family='e4m3')
    packed.adapter()
    owner = native_owner(packed)
    with pytest.raises(GrammarError, match='retired|native projection'):
        owner.gate(torch.empty(0, owner.gate.cols, dtype=torch.bfloat16),
                   torch.empty(0, 1, dtype=torch.int32), torch.empty(0, 1))


def test_retired_projection_cannot_be_recomposed(native_constructor, monkeypatch):
    packed = nwm.PackedWindowMoeBundles(*[projection() for _ in range(3)], family='e4m3')
    owner = native_owner(packed)
    monkeypatch.delenv(rf.ENV_TOGGLE, raising=False)
    monkeypatch.setattr(rf, 'fused_routed_window_supported', SUPPORTED)
    reason = rf.fused_routed_window_supported(owner.gate, owner.up, owner.down)
    assert 'retired' in reason
    with pytest.raises(GrammarError, match='retired'):
        rf.FusedRoutedWindowMoE.from_bundles(owner.gate, owner.up, owner.down)


@pytest.mark.parametrize('failure', [RuntimeError, GrammarError])
def test_late_descriptor_failure_keeps_original_owner(native_constructor, monkeypatch, failure):
    roles = [projection() for _ in range(3)]
    packed = nwm.PackedWindowMoeBundles(*roles, family='e4m3')
    original = rf.projection_tables
    calls = []

    def descriptor(bundle):
        calls.append(bundle)
        if len(calls) == 3:
            raise failure('third projection failed')
        return original(bundle)

    monkeypatch.setattr(rf, 'projection_tables', descriptor)
    fallback = object()
    monkeypatch.setattr(nwm, 'native_window_moe_from_bundles', lambda *args, **kwargs: fallback)
    if failure is GrammarError:
        with pytest.raises(GrammarError, match='third projection failed'):
            packed.adapter()
        assert '_adapter' not in packed.__dict__
    else:
        assert packed.adapter() is fallback
        assert native_owner(packed) is packed
    assert '_fused_adapter' not in packed.__dict__
    assert all(held is role and role.codes_all is not None and role.perm_all is not None
               for held, role in zip((packed.gate, packed.up, packed.down), roles))


