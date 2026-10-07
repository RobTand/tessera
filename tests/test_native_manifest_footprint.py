"""Manifest bytes must equal actual frozen native bundles, not decoded tiles."""
from types import SimpleNamespace
import pytest
import torch
from tessera import kernel_window_gemv as kg
from tessera.window_gemm import prepare_window_gemm
from tessera.serving.native_window import PreparedDenseNativeModule
from tessera.serving_parts import dense_resident_bytes_resident_mode
from window_pack_reference import pack_bitstream


@pytest.mark.parametrize('family', ['TESSERA_FP8', 'TESSERA_BF16'])
@pytest.mark.parametrize('rows,rates', [(513, (1, 3, 5, 8)), (512, (4, 4, 4)), (19, (2, 7))])
def test_manifest_counts_actual_native_kernel_inputs(family, rows, rates):
    """CPU reference packing drives the production bundle freezer unchanged."""
    value_family = family == 'TESSERA_BF16'
    bits = 14
    rep = pack_bitstream(torch.zeros((rows, len(rates)), dtype=torch.int64), rates)
    unit = kg.WindowGemvUnit(
        rep=rep, table=torch.zeros(1 << bits, dtype=torch.bfloat16),
        scale=torch.ones(rows, dtype=torch.float32), window_bits=bits, plan=kg.Plan(),
        codes_of_state=None if value_family else torch.zeros(1 << bits, dtype=torch.uint8),
        native=None if value_family else torch.zeros(256, dtype=torch.uint8),
        family='value' if value_family else 'e4m3')
    frozen = prepare_window_gemm(unit, quantizer=None)
    prepared = PreparedDenseNativeModule(
        [SimpleNamespace(name='q', rows=rows, bundle=frozen)], rows=rows,
        columns=len(rates), device=torch.device('cpu'), family=unit.family)
    # Both current routes keep this independent fp32 row-scale buffer.
    route_scale = prepared.row_scale()
    actual = prepared.packed_bytes() + route_scale.numel() * route_scale.element_size()
    role = {'rows': rows, 'cols': len(rates), 'rates': rates, 'window_bits': bits, 'tile_rows': kg.TILE_ROWS}
    priced = dense_resident_bytes_resident_mode(family, rows, len(rates), native_roles=[role])
    assert priced == actual


def test_current_native_family_refuses_missing_layout():
    with pytest.raises(ValueError, match='native.*layout'):
        dense_resident_bytes_resident_mode('TESSERA_FP8', 512, 3)


@pytest.mark.parametrize('rows,rate,cols', [(32, 1, 256), (1056, 7, 256), (1024, 8, 512)])
def test_manifest_counts_native_e2m1_window_roles_and_shared_activation_scale(monkeypatch, rows, rate, cols):
    from tessera.compact_prep import WindowLutUnit
    from tessera import routed_fused_e2m1 as fp4
    from tessera.serving.residency import resident_storage_bytes

    spec = {'rows': rows, 'cols': cols, 'rates': (rate,) * cols,
            'arity': 2, 'half': 16, 'window_bits': 14, 'tile_rows': kg.TILE_ROWS}
    priced = dense_resident_bytes_resident_mode(
        'TESSERA_NVFP4', 2 * rows, cols, native_roles=[spec, spec])
    # Only compiler loading is suppressed. The production role freezer and
    # CPU reference bit packing determine the actual retained allocations.
    monkeypatch.setattr(fp4, '_ext', lambda: None)
    gs = torch.tensor(2.0, dtype=torch.float32)
    roles = []
    for global_scale in (1.0, 4.0):
        rep = pack_bitstream(torch.zeros((rows // 2, cols), dtype=torch.int64), (rate,) * cols)
        unit = WindowLutUnit(
            rep=rep, codes=torch.zeros(1 << 14, dtype=torch.uint8),
            scale_plane=torch.zeros(rows * cols // 32, dtype=torch.uint8),
            scale_lut=torch.zeros(16, dtype=torch.uint8), global_scale=global_scale,
            window_bits=14, rows=rows, cols=cols, arity=2, half=16,
            initial_state=torch.ones(cols, dtype=torch.int32), row_offset=32)
        roles.append(fp4.prepare_dense_role(unit, gs))
    actual = resident_storage_bytes(
        (f'{i}.{name}', value) for i, role in enumerate(roles)
        for name, value in role.named_tensors())
    assert priced == actual
    assert roles[0].ratio.item() == 0.5
    assert roles[1].ratio.item() == 2.0
