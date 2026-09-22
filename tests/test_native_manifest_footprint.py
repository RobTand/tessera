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


def test_manifest_prices_measured_native_a4_bundle_not_expanded_nibbles():
    # 2026-09-22 real native capture, gate_up_proj: two 3072x1024 roles.
    # Raw candidate persistent rows, independent of the pricing implementation.
    captured = [1179648, 1179648, 196608, 196608, 99336, 99336, 98304, 98304,
                2048, 1024, 1024, 512, 512, 512, 512, 256, 256, 16, 16, 4, 4, 4]
    role = {'rows': 3072, 'cols': 1024, 'rates': (7,) * 1024,
            'arity': 2, 'memory': 6, 'half': 16, 'lut_entries': 16}
    assert dense_resident_bytes_resident_mode(
        'TESSERA_NVFP4', 6144, 1024, native_roles=[role, role],
        trellis_table_bytes=4096) == sum(captured)
