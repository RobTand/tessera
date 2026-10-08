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
            "output_boundaries": {name: {"status": "pass", "cases": 1} for name in ("float32_multiply", "bfloat16_conversion")},
            "physical_devices": ["fixture-device"],
            "arithmetic_qualified": False,
            "reviews": {"kernels_parent": False, "independent": False}}


@pytest.mark.parametrize("property_name", PROPERTIES)
def test_each_failed_property_refuses_the_named_device(property_name):
    report = fixture_report()
    report["properties"][property_name]["status"] = "fail"
    with pytest.raises(FP4QualificationError, match=f"T4 refused on CPU test fixture: required probe {property_name}"):
        require_t4_device_qualification(report, device="CPU test fixture", physical_device="fixture-device", comparison="exact_represented_operands")


def test_unreviewed_native_results_do_not_qualify():
    with pytest.raises(FP4QualificationError, match="required reviews"):
        require_t4_device_qualification(fixture_report(), device="CPU test fixture", physical_device="fixture-device", comparison="exact_represented_operands")


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


@pytest.mark.parametrize("name,options", [("float32_multiply", {"output_scale": 2}), ("bfloat16_conversion", {"output_dtype": "bfloat16"})])
def test_failed_boundary_refuses_only_when_the_operation_is_present(name, options):
    report = fixture_report()
    report["output_boundaries"][name]["status"] = "fail"
    derive_attested_fp4_bound(1, k=64, report=report, device="CPU test fixture")
    with pytest.raises(FP4QualificationError, match=f"required output boundary {name}"):
        derive_attested_fp4_bound(1, k=64, report=report, device="CPU test fixture", **options)


def test_an_unrepresented_output_scale_refuses():
    with pytest.raises(ValueError, match="already represented float32"):
        derive_attested_fp4_bound(1, k=64, report=fixture_report(), device="CPU test fixture", output_scale=Fraction(1, 3))


def test_an_approved_contract_cannot_transfer_to_an_untested_physical_device():
    report = fixture_report()
    report["arithmetic_qualified"] = True
    report["reviews"] = {"kernels_parent": True, "independent": True}
    with pytest.raises(FP4QualificationError, match="no required native probes cover physical device untested-device"):
        require_t4_device_qualification(report, device="CPU test fixture", physical_device="untested-device", comparison="exact_represented_operands")
    assert require_t4_device_qualification(report, device="CPU test fixture", physical_device="fixture-device", comparison="exact_represented_operands") == report["contract"]


@pytest.mark.parametrize("comparison", ["complete_fused_epilogue", "tp2_arithmetic"])
def test_uncharacterized_complete_comparisons_refuse_even_after_approval(comparison):
    report = fixture_report()
    report["arithmetic_qualified"] = True
    report["reviews"] = {"kernels_parent": True, "independent": True}
    with pytest.raises(FP4QualificationError, match="needs uncharacterized fused epilogue or cross-device terms"):
        require_t4_device_qualification(report, device="CPU test fixture", physical_device="fixture-device", comparison=comparison)



def stock_fixture_report():
    report = fixture_report()
    report["stock_reference"] = {"properties": {name: {"status": "pass", "cases": 1} for name in ("activation_division", "ratio_formation", "library_fp32", "magnitude_fp64")},
        "physical_devices": ["fixture-device"], "input_global_scale": 896, "allow_tf32": False,
        "activation_division_steps": 1, "ratio_formation_steps": 1,
        "library_fp32_precision": 24, "magnitude_fp64_precision": 53,
        "reference_shapes": [[1, 64, 256], [16, 64, 256]]}
    return report


def test_complete_stock_bound_keeps_all_higher_order_terms():
    from tessera.fp4_arithmetic import EPSILON, derive_packed_stock_bound
    report = stock_fixture_report()
    result, receipt = derive_packed_stock_bound(7, k=256, global_scale=Fraction(1, 256),
        report=report, device="CPU test fixture", physical_device="fixture-device", shape=(1, 64, 256))
    u = Fraction(1, 1 << 24)
    native = (1 + ATOM_ERROR) ** 4 - 1
    gamma = 512 * EPSILON / (1 - 512 * EPSILON)
    coefficient = (((1 + u) * (1 + native) * (1 + u) - 1 + u) / (1 - u) + gamma)
    assert Fraction(receipt["exact_bound"]) == 7 * coefficient
    assert Fraction(result["atol"]) >= 7 * coefficient
    assert receipt["activation_division"] == str(u)
    assert receipt["ratio_formation"] == str(u)
    assert receipt["weight_formation_error"].startswith("0:")
    assert receipt["subnormal_allowance"].startswith("0:")
    assert receipt["arithmetic_qualified"] is False


@pytest.mark.parametrize("name", ["activation_division", "ratio_formation", "library_fp32", "magnitude_fp64"])
def test_each_missing_stock_term_refuses(name):
    from tessera.fp4_arithmetic import derive_packed_stock_bound
    report = stock_fixture_report()
    report["stock_reference"]["properties"][name]["status"] = "fail"
    with pytest.raises(FP4QualificationError, match=f"required stock-reference term {name}"):
        derive_packed_stock_bound(1, k=256, global_scale=1, report=report, device="CPU test fixture", physical_device="fixture-device")


@pytest.mark.parametrize("global_scale", [Fraction(1, 1 << 74), 1 << 117, Fraction(3, 2), 0, -1])
def test_stock_global_domain_refuses_subnormal_or_inexact_weight_cases(global_scale):
    from tessera.fp4_arithmetic import derive_packed_stock_bound
    with pytest.raises(ValueError):
        derive_packed_stock_bound(1, k=256, global_scale=global_scale, report=stock_fixture_report(), device="CPU test fixture", physical_device="fixture-device")


def test_magnitude_inflation_uses_the_actual_double_contraction():
    from tessera.fp4_arithmetic import stock_magnitude_upper
    measured = Fraction(7, 8)
    epsilon = Fraction(1, 1 << 52)
    gamma = 512 * epsilon / (1 - 512 * epsilon)
    upper = stock_magnitude_upper(float(measured), k=256, report=stock_fixture_report(), device="CPU test fixture", physical_device="fixture-device")
    assert Fraction(upper) >= measured / (1 - gamma)

