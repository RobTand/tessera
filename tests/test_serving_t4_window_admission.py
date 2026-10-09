"""Real byte intake validity is separate from WINDOW serving qualification."""
from __future__ import annotations

import pytest
import torch

from tessera.compact_prep import parse_compact_wire
from tessera.fused import pack_fused
from tessera.serving import scheme
from tessera.serving.contract import load_serving_contract

from window_lut_reference import window_blob


def _dense(blob, q256, rows=32, cols=256):
    return {"family": scheme.TESSERA_NVFP4, "grid": "E2M1x2", "body": "WINDOW",
            "plane": "LUT", "q256": q256, "rows": rows, "columns": cols,
            "wire_bytes": len(blob), "roles": [["weight", rows]]}


@pytest.mark.parametrize("q256", [128 * rate for rate in range(1, 9)])
def test_actual_paired_window_bytes_validate_without_a_fabricated_receipt(q256):
    blob = pack_fused([("weight", 32, window_blob(q256, rows=32, cols=256))])
    declared = _dense(blob, q256)
    parsed = scheme.parse_compact_blob_for_scheme(blob, declared, "dense", device="cpu")
    name, wire = parsed[0]
    assert name == "weight" and wire.metadata.rows == 32
    assert wire.metadata.columns == 256 and set(wire.metadata.rates) == {q256 // 128}
    contract = load_serving_contract()
    assert not any(cell["family"] == "TESSERA_E2M1_K2"
                   for cell in contract["lane_eligibility"]["cells"])


@pytest.mark.parametrize("window_bits", [12, 16])
def test_real_research_window_bytes_are_not_served_as_lut14(window_bits):
    blob = pack_fused([("weight", 32, window_blob(896, rows=32, cols=256,
                                               window_bits=window_bits))])
    with pytest.raises(ValueError, match="window_bits"):
        scheme.parse_compact_blob_for_scheme(blob, _dense(blob, 896), "wrong-window", device="cpu")


def test_actual_tcq_container_is_not_a_serving_compatibility_path():
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import encode_linear_planes
    from tessera.manifest import BodyKind, ScalePlaneKind

    exported, _unit, _forests = encode_linear_planes(
        torch.linspace(-0.2, 0.2, 32 * 256).reshape(32, 256),
        grid=tuple_grid(E2M1_GRID, 2), q256=896,
        body=BodyKind.TCQ, span=2, scale_plane=ScalePlaneKind.LUT,
        name="weight", verify=False)
    blob = pack_fused([("weight", 32, exported.blob)])
    with pytest.raises(ValueError, match="wire is"):
        scheme.parse_compact_blob_for_scheme(blob, _dense(blob, 896), "old-tcq", device="cpu")


def test_corrupt_actual_body_is_refused_before_hardware_preparation():
    from tessera.errors import SchemaError

    blob = bytearray(window_blob(896, rows=32, cols=256))
    blob[-1] ^= 1
    with pytest.raises(SchemaError, match="payload digest"):
        parse_compact_wire(bytes(blob), device="cpu")


@pytest.mark.parametrize("rows,cols", [(31, 256), (32, 128), (32, 288)])
def test_native_unsupported_dense_geometry_is_refused_at_scheme_load(rows, cols):
    with pytest.raises(ValueError, match="native E2M1"):
        scheme.validate_tessera_scheme(_dense(b"unused", 896, rows, cols), "geometry")
