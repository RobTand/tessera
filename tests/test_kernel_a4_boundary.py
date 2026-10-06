"""Exact field/end-of-plane numerical regressions for native span-two reads."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).parents[1] / "experiments/t4_code/span2_boundary_check.py"
spec = importlib.util.spec_from_file_location("span2_boundary_check", PATH)
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


def test_exact_width_cpu_oracle_needs_no_trailing_byte():
    # Meaningful field values at the allocation's last bit/byte; a zero-width
    # POINT field is valid even when there is no POINT allocation.
    assert boundary.field_value(bytes([0xA5]), 0, 8) == 0xA5
    assert boundary.field_value(bytes([0xA5]), 1, 7) == 0x25
    assert boundary.field_value(bytes([0xA5]), 4, 4) == 5
    assert boundary.field_value(bytes([0xA5]), 6, 2) == 1
    assert boundary.field_value(b"", 0, 0) == 0
    assert boundary.field_value(bytes([0xA5, 0x5A]), 6, 6) == 0x15


@pytest.fixture(scope="module")
def native_boundary_bank(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the native bitfield-boundary numerical oracle")
    path = tmp_path_factory.mktemp("native-span2-boundary")
    boundary.prepare(path)
    return path


@pytest.mark.parametrize("rate", range(1, 8))
def test_native_boundary_states_codes_and_scales_match_stock(rate, native_boundary_bank):
    # Numerical comparisons, not bare not-throw or source-string assertions.
    result = boundary.check_wire((native_boundary_bank / f"rate{rate}.wire").read_bytes(), rate, "cuda")
    assert result["indexed_states"] == 256
    assert result["status"] == "all native states, packed codes and scale bytes matched independent oracle"
