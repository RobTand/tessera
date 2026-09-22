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
