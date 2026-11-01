"""Fixed-tolerance failures for the packed-stock gate (tessera#1182).

A fixed tolerance ignores the reduction length and the cancellation in the
measured error. The derived bound accepts a correct cancellation result that a
fixed tolerance would refuse, and refuses a wrong tiny output that a fixed
tolerance would accept. The comparison needs torch and skips without it.
"""
import pytest

torch = pytest.importorskip("torch", reason="the packed-stock comparison needs torch")

from tessera.fp4_arithmetic import (
    FP4QualificationError as Refusal,
    check_packed_stock_arithmetic,
    derive_packed_stock_bound,
)
from test_fp4_arithmetic_attestation import packed_shapes_fixture_report


def test_fixed_tolerance_would_refuse_a_correct_cancellation_result():
    report = packed_shapes_fixture_report((2, 8, 4096))
    kwargs = dict(k=4096, global_scale=1, report=report, device="CPU test fixture",
                  physical_device="fixture-device", shape=(2, 8, 4096))
    allowance, _ = derive_packed_stock_bound(64.0, **kwargs)
    fixed_tolerance = 1e-4
    measured_error = 0.01
    assert allowance["atol"] > measured_error > fixed_tolerance
    check_kwargs = {key: value for key, value in kwargs.items() if key not in ("device", "shape")}
    actual = torch.full((2, 8), measured_error, dtype=torch.float32)
    expected = torch.zeros((2, 8), dtype=torch.float32)
    receipt = check_packed_stock_arithmetic(actual, expected, 64.0, **check_kwargs)
    assert receipt["max_abs_error"] > fixed_tolerance


def test_fixed_tolerance_would_accept_a_wrong_tiny_output():
    report = packed_shapes_fixture_report((1, 4, 256))
    kwargs = dict(k=256, global_scale=1, report=report, device="CPU test fixture",
                  physical_device="fixture-device", shape=(1, 4, 256))
    allowance, _ = derive_packed_stock_bound(1e-7, **kwargs)
    assert allowance["atol"] < 1e-6 < 1e-3
    check_kwargs = {key: value for key, value in kwargs.items() if key not in ("device", "shape")}
    actual = torch.full((1, 4), 1e-6, dtype=torch.float32)
    expected = torch.zeros((1, 4), dtype=torch.float32)
    with pytest.raises(Refusal, match="exceeds"):
        check_packed_stock_arithmetic(actual, expected, 1e-7, **check_kwargs)
