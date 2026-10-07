"""CPU controls for the device gate; these do not attest a device."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import pytest

from tessera.fp4_arithmetic import (
    ATOM_ERROR, INSTRUCTION, PROPERTIES, FP4QualificationError,
    derive_attested_fp4_bound, require_probe_contract,
    require_t4_device_qualification,
)


def fixture_report():
    return {"schema": "tessera.fp4_mma_attestation.v1", "instruction": INSTRUCTION,
            "device": {"device": "CPU test fixture"}, "harness_status": "pass",
            "properties": {name: {"status": "pass", "cases": 1} for name in PROPERTIES},
            "contract": {"alignment_bits_min": 36, "atom_summands": 65,
                         "scaled_product_error": "0", "atom_error": "(65*2^-35+2^-23)*S"},
            "arithmetic_qualified": False,
            "reviews": {"kernels_parent": False, "independent": False}}


@pytest.mark.parametrize("property_name", PROPERTIES)
def test_each_failed_property_refuses_the_named_device(property_name):
    report = fixture_report()
    report["properties"][property_name]["status"] = "fail"
    with pytest.raises(FP4QualificationError, match=f"T4 refused on CPU test fixture: required probe {property_name}"):
        require_t4_device_qualification(report, device="CPU test fixture")


def test_unreviewed_native_results_do_not_qualify():
    with pytest.raises(FP4QualificationError, match="required reviews"):
        require_t4_device_qualification(fixture_report(), device="CPU test fixture")


def test_empty_population_does_not_establish_a_property():
    report = fixture_report()
    report["properties"]["product_exactness"]["cases"] = 0
    with pytest.raises(FP4QualificationError, match="product_exactness"):
        require_probe_contract(report, device="CPU test fixture")


def test_bound_uses_the_atom_recurrence_without_a_fitted_multiplier():
    report = fixture_report()
    for k in (64, 128, 4096):
        result = derive_attested_fp4_bound(Fraction(123, 8), k=k, report=report, device="CPU test fixture")
        expected = ((1 + ATOM_ERROR) ** (k // 64) - 1) * Fraction(123, 8)
        assert Fraction(result["exact_bound"]) == expected
        assert Fraction(result["absolute_bound"]) >= expected
        assert result["arithmetic_qualified"] is False
        assert result["boundary_term"] == "0"


@pytest.mark.parametrize("k", [True, 0, 63, 65, 128.0])
def test_illegal_native_reduction_lengths_refuse(k):
    with pytest.raises(ValueError, match="64-column atom"):
        derive_attested_fp4_bound(1, k=k, report=fixture_report(), device="CPU test fixture")


@pytest.mark.parametrize("magnitude", [True, -1, float("inf"), float("nan")])
def test_invalid_magnitude_refuses(magnitude):
    with pytest.raises(ValueError):
        derive_attested_fp4_bound(magnitude, k=64, report=fixture_report(), device="CPU test fixture")


def test_finite_inputs_can_still_overflow_and_refuse():
    with pytest.raises(ValueError, match="intermediate domain"):
        derive_attested_fp4_bound(2 ** 128, k=64, report=fixture_report(), device="CPU test fixture")


def test_zero_dot_without_an_output_operation_has_zero_bound():
    result = derive_attested_fp4_bound(0, k=64, report=fixture_report(), device="CPU test fixture")
    assert result["absolute_bound"] == 0


def test_only_actual_output_operations_add_boundary_terms():
    report = fixture_report()
    native = derive_attested_fp4_bound(1, k=64, report=report, device="CPU test fixture")
    scaled = derive_attested_fp4_bound(1, k=64, report=report, device="CPU test fixture", output_scale=2)
    bf16 = derive_attested_fp4_bound(1, k=64, report=report, device="CPU test fixture", output_dtype="bfloat16")
    assert Fraction(scaled["exact_bound"]) > 2 * Fraction(native["exact_bound"])
    assert Fraction(bf16["exact_bound"]) > Fraction(native["exact_bound"])
    assert native["normalization_term"].startswith("0:")


def test_independent_interpreter_and_numerical_negative_control():
    path = Path(__file__).parents[1] / "experiments/t4_code/fp4_arithmetic_attest.py"
    spec = importlib.util.spec_from_file_location("fp4_probe_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.cpu_controls()
    case = reference.base("one scaled product", "product_exactness")
    reference.term(case, 0, 3, 3, 63, 63)
    exact = reference.exact_terms(case, 0, 0)
    outputs = [reference.bits(sum(exact))] * 128
    good, _ = reference.analyze([case], outputs, {"device": "CPU test fixture"})
    assert good["properties"]["product_exactness"]["status"] == "pass"
    outputs[0] = reference.bits(sum(exact) + 1)
    bad, rows = reference.analyze([case], outputs, {"device": "CPU test fixture"})
    assert rows[0]["failures"][0]["error"] == "1"
    with pytest.raises(FP4QualificationError, match="product_exactness"):
        require_probe_contract(bad, device="CPU test fixture")
