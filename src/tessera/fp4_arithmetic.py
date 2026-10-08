"""Device-specific, default-off qualification for native block-scaled FP4.

Finite probes attest a measured implementation model. They are not a universal
hardware proof. The independent reference uses exact represented operands.
"""
from __future__ import annotations

import math
from fractions import Fraction

INSTRUCTION = "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"
PROPERTIES = ("product_exactness", "scale_application_order", "accumulation_alignment_width", "rounding_mode", "subnormal_handling", "intermediate_domain")
EPSILON = Fraction(1, 1 << 23)
# The cancellation probes retain exponent gaps through 35 and discard gap 36.
# An atom has 64 products and one incoming accumulator. Each alignment loses
# less than one 36-bit quantum. Final normalization returns a 24-bit float.
ALIGNMENT_BITS = 36
ATOM_ERROR = 65 * Fraction(1, 1 << (ALIGNMENT_BITS - 1)) + EPSILON
FP32_MAX = Fraction((1 << 24) - 1) * (1 << 104)


class FP4QualificationError(ValueError):
    pass


def require_probe_contract(report, *, device):
    """Refuse a missing or failed required probe by the actual device name."""
    prefix = f"T4 refused on {device}: "
    if not isinstance(report, dict) or report.get("schema") != "tessera.fp4_mma_attestation.v1":
        raise FP4QualificationError(prefix + "native arithmetic evidence is absent")
    if report.get("device", {}).get("device") != device:
        raise FP4QualificationError(prefix + "this device has no measured arithmetic evidence")
    if report.get("instruction") != INSTRUCTION:
        raise FP4QualificationError(prefix + "the required native instruction was not exercised")
    if report.get("harness_status") != "pass":
        raise FP4QualificationError(prefix + "the probe harness did not establish its operand layout")
    results = report.get("properties", {})
    for name in PROPERTIES:
        result = results.get(name, {})
        if result.get("status") != "pass" or type(result.get("cases")) is not int or result["cases"] < 1:
            raise FP4QualificationError(prefix + f"required probe {name} failed or is absent")
    contract = report.get("contract", {})
    if contract.get("alignment_bits_min") != 36 or contract.get("atom_summands") != 65:
        raise FP4QualificationError(prefix + "the accumulation inequality is unsupported")
    if contract.get("scaled_product_error") != "0" or contract.get("atom_error") != "(65*2^-35+2^-23)*S":
        raise FP4QualificationError(prefix + "the product or atom inequality is unsupported")
    return contract


def require_t4_device_qualification(report, *, device, physical_device, comparison):
    """Qualify the explicit primitive or attested stock-reference comparison."""
    contract = require_probe_contract(report, device=device)
    if comparison == "float32_stock_reference":
        require_stock_reference_contract(report, device=device, physical_device=physical_device)
    elif comparison != "exact_represented_operands":
        raise FP4QualificationError(f"T4 refused on {device}: comparison {comparison} needs uncharacterized fused epilogue or cross-device terms")
    reviews = report.get("reviews", {})
    if report.get("arithmetic_qualified") is not True or reviews.get("kernels_parent") is not True or reviews.get("independent") is not True:
        raise FP4QualificationError(f"T4 refused on {device}: arithmetic qualification awaits the required reviews")
    if not isinstance(physical_device, str) or physical_device not in report.get("physical_devices", []):
        raise FP4QualificationError(f"T4 refused on {device}: no required native probes cover physical device {physical_device}")
    return contract


def _require_output_boundary(report, device, name):
    result = report.get("output_boundaries", {}).get(name, {})
    if result.get("status") != "pass" or type(result.get("cases")) is not int or result["cases"] < 1:
        raise FP4QualificationError(f"T4 refused on {device}: required output boundary {name} failed or is absent")


def derive_attested_fp4_bound(magnitude, *, k, report, device, output_scale=None, output_dtype="float32"):
    """Derive a diagnostic bound from the attested atom inequality.

    M is the exact sum of absolute scaled products. The accumulator starts at
    zero. Scale bytes are UE4M3 codes 0..126. K is a multiple of 64.
    The returned bound compares native output with exact represented arithmetic,
    not with an uncharacterized library GEMM or with the original unquantized
    inputs. Output scale, when present, is an already represented finite FP32
    value. No normalization division occurs in this operation.
    """
    require_probe_contract(report, device=device)
    if type(k) is not int or k < 64 or k % 64:
        raise ValueError("k must be a positive multiple of the native 64-column atom")
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float, Fraction)):
        raise ValueError("magnitude must be a finite nonnegative exact operand bound")
    if isinstance(magnitude, float) and not math.isfinite(magnitude):
        raise ValueError("magnitude must be finite")
    m = Fraction(magnitude)
    if m < 0:
        raise ValueError("magnitude must be nonnegative")
    atoms = k // 64
    factor = (1 + ATOM_ERROR) ** atoms
    if factor * m > FP32_MAX:
        raise ValueError("the native intermediate domain can overflow")
    error = (factor - 1) * m
    boundary = Fraction(0)
    if output_scale is not None:
        _require_output_boundary(report, device, "float32_multiply")
        if isinstance(output_scale, bool) or not isinstance(output_scale, (int, float, Fraction)):
            raise ValueError("output_scale must be an already represented finite float32 value")
        if isinstance(output_scale, float) and not math.isfinite(output_scale):
            raise ValueError("output_scale must be finite")
        scale = Fraction(output_scale)
        import struct
        try:
            represented = struct.unpack("<f", struct.pack("<f", float(scale)))[0]
        except OverflowError as exc:
            raise ValueError("output_scale is outside the float32 domain") from exc
        if not math.isfinite(represented) or Fraction(represented) != scale:
            raise ValueError("output_scale must be an already represented float32 value")
        # The scalar operation is FP32 multiplication with round to nearest.
        # Subnormal products need the half-minimum-subnormal absolute term.
        scaled_max = abs(scale) * (m + error)
        if scaled_max > FP32_MAX:
            raise ValueError("the output multiplication can overflow")
        boundary = scaled_max * Fraction(1, 1 << 24) + Fraction(1, 1 << 150)
        error = abs(scale) * error + boundary
        m = abs(scale) * m
    if output_dtype == "bfloat16":
        _require_output_boundary(report, device, "bfloat16_conversion")
        # This is the actual fused output boundary, not an input quantizer term.
        bf16_max = Fraction(255) * (1 << 120)
        if m + error > bf16_max:
            raise ValueError("the bfloat16 output boundary can overflow")
        term = (m + error) * Fraction(1, 256) + Fraction(1, 1 << 134)
        boundary += term
        error += term
    elif output_dtype != "float32":
        raise ValueError("output_dtype must be float32 or bfloat16")
    upper = float(error)
    if Fraction(upper) < error:
        upper = math.nextafter(upper, math.inf)
    return {"absolute_bound": upper, "exact_bound": str(error), "atoms": atoms,
            "atom_error": "(65*2^-35+2^-23)*S", "accumulation_bound": "((1+65*2^-35+2^-23)^(K/64)-1)*M",
            "boundary_term": str(boundary), "normalization_term": "0: represented scale bytes enter the native instruction directly",
            "arithmetic_qualified": False, "scope": "measured implementation contract; exact represented-operand reference"}


STOCK_PROPERTIES = ("activation_division", "ratio_formation", "library_fp32", "magnitude_fp64")


def require_stock_reference_contract(report, *, device, physical_device, shape=None):
    require_probe_contract(report, device=device)
    _require_output_boundary(report, device, "float32_multiply")
    stock = report.get("stock_reference", {})
    for name in STOCK_PROPERTIES:
        result = stock.get("properties", {}).get(name, {})
        if result.get("status") != "pass" or type(result.get("cases")) is not int or result["cases"] < 1:
            raise FP4QualificationError(f"T4 refused on {device}: required stock-reference term {name} failed or is absent")
    if physical_device not in stock.get("physical_devices", []):
        raise FP4QualificationError(f"T4 refused on {device}: the stock reference is not attested on physical device {physical_device}")
    if stock.get("input_global_scale") != 896 or stock.get("allow_tf32") is not False:
        raise FP4QualificationError(f"T4 refused on {device}: unsupported stock-reference configuration")
    if shape is not None and list(shape) not in stock.get("reference_shapes", []):
        raise FP4QualificationError(f"T4 refused on {device}: the stock reference shape {shape} is uncharacterized")
    for name in ("activation_division_steps", "ratio_formation_steps"):
        if type(stock.get(name)) is not int or stock[name] not in (1, 2):
            raise FP4QualificationError(f"T4 refused on {device}: unsupported division model {name}")
    if stock.get("library_fp32_precision") != 24 or stock.get("magnitude_fp64_precision") != 53:
        raise FP4QualificationError(f"T4 refused on {device}: unsupported library precision")
    return stock


def _gamma_exact(steps, epsilon):
    product = steps * epsilon
    if product >= 1:
        raise ValueError("the library error depth has no finite gamma denominator")
    return product / (1 - product)


def _outward_float(value):
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


def stock_magnitude_upper(measured, *, k, report, device, physical_device, shape=None):
    """Inflate the actual positive binary64 contraction, not a fitted screen."""
    require_stock_reference_contract(report, device=device, physical_device=physical_device, shape=shape)
    if type(k) is not int or k < 128 or k % 128:
        raise ValueError("the packed reader requires a positive 128-column contraction")
    if isinstance(measured, bool) or not isinstance(measured, (int, float, Fraction)):
        raise ValueError("the measured magnitude must be a finite nonnegative scalar")
    if isinstance(measured, float) and not math.isfinite(measured):
        raise ValueError("the measured magnitude must be finite")
    value = Fraction(measured)
    if value < 0:
        raise ValueError("the measured magnitude must be nonnegative")
    gamma = _gamma_exact(2 * k, Fraction(1, 1 << 52))
    if gamma >= 1:
        raise ValueError("the positive magnitude cannot establish an upper bound")
    return _outward_float(value / (1 - gamma))


def derive_packed_stock_bound(magnitude_upper, *, k, global_scale, report, device, physical_device, shape=None, input_global_scale=896):
    """Bound the original normalized FP32 stock-reference comparison.

    The four stock terms use their targeted contracts. Both sides share the
    represented activation codes and group scales. The power-of-two weight
    global makes stock weight formation exact within the stated normal domain.
    """
    stock = require_stock_reference_contract(report, device=device, physical_device=physical_device, shape=shape)
    if input_global_scale != 896:
        raise ValueError("the attested activation global is exactly 896")
    if type(k) is not int or k < 128 or k % 128:
        raise ValueError("the packed reader requires a positive 128-column contraction")
    if isinstance(global_scale, bool) or not isinstance(global_scale, (int, float, Fraction)):
        raise ValueError("the weight global must be a finite positive power of two")
    if isinstance(global_scale, float) and not math.isfinite(global_scale):
        raise ValueError("the weight global must be finite")
    g = Fraction(global_scale)
    if g <= 0 or g.numerator & (g.numerator - 1) or g.denominator & (g.denominator - 1):
        raise ValueError("the weight global must be a positive power of two")
    exponent = g.numerator.bit_length() - g.denominator.bit_length()
    if not -73 <= exponent <= 116:
        raise ValueError("the stock-reference lattice or weight formation is outside the normal finite domain")
    if isinstance(magnitude_upper, bool) or not isinstance(magnitude_upper, (int, float, Fraction)):
        raise ValueError("the operand magnitude must be a finite nonnegative upper bound")
    if isinstance(magnitude_upper, float) and not math.isfinite(magnitude_upper):
        raise ValueError("the operand magnitude must be finite")
    m = Fraction(magnitude_upper)
    if m < 0:
        raise ValueError("the operand magnitude must be nonnegative")
    u = Fraction(1, 1 << 24)
    division = (1 + u) ** stock["activation_division_steps"] - 1
    ratio = (1 + u) ** stock["ratio_formation_steps"] - 1
    native = (1 + ATOM_ERROR) ** (k // 64) - 1
    reference = _gamma_exact(2 * k, EPSILON)
    ideal_magnitude = m / (1 - division)
    native_scaled = (1 + ratio) * (1 + native) * (1 + u) - 1
    coefficient = (native_scaled + division) / (1 - division) + reference
    if (1 + reference) * m > FP32_MAX:
        raise ValueError("the stock-reference intermediate domain can overflow")
    if (1 + native) * ideal_magnitude * input_global_scale / g > FP32_MAX:
        raise ValueError("the unnormalized native intermediate domain can overflow")
    if (1 + ratio) * (1 + native) * ideal_magnitude > FP32_MAX:
        raise ValueError("the native output multiplication can overflow")
    bound = coefficient * m
    return {"atol": _outward_float(bound), "rtol": 0.0}, {
        "schema": "tessera.packed_stock_arithmetic_bound.v1", "status": "attested_contract_pending_reviews",
        "arithmetic_qualified": False, "native_arithmetic_qualified": False,
        "bound": "[((1+rho_ratio)*(1+eta)^L*(1+u32)-1+rho_div)/(1-rho_div)+gamma(2K,epsilon32)]*M_upper",
        "exact_bound": str(bound), "exact_coefficient": str(coefficient),
        "native_atoms": k // 64, "native_coefficient": str(native),
        "activation_division": str(division), "ratio_formation": str(ratio),
        "reference_gamma": str(reference), "reference_steps": 2 * k,
        "magnitude_gamma": str(_gamma_exact(2 * k, Fraction(1, 1 << 52))),
        "weight_formation_error": "0: exact normal power-of-two stock weight formation",
        "output_multiplication_roundoff": str(u), "subnormal_allowance": "0: the supported lattice keeps every nonzero intermediate normal",
        "weight_global_exponent": exponent, "physical_device": physical_device,
        "supported_comparison": "float32_stock_reference", "operand_magnitude_upper": _outward_float(m),
    }


def check_packed_stock_arithmetic(actual, expected, magnitude_upper, *, k, global_scale, report, physical_device, input_global_scale=896):
    import torch
    if actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise ValueError("the packed stock comparison requires float32 output tensors")
    if actual.shape != expected.shape or actual.dim() != 2 or actual.numel() == 0:
        raise ValueError("the packed stock comparison requires equal nonempty output shapes")
    shape = (actual.shape[0], actual.shape[1], k)
    allowance, receipt = derive_packed_stock_bound(magnitude_upper, k=k, global_scale=global_scale,
        report=report, device=report["device"]["device"], physical_device=physical_device,
        shape=shape, input_global_scale=input_global_scale)
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("the packed stock comparison requires finite outputs")
    measured_error = float((actual.double() - expected.double()).abs().max())
    error = _outward_float(Fraction(measured_error) / (1 - Fraction(1, 1 << 53)))
    if error > allowance["atol"]:
        raise FP4QualificationError(f"T4 refused on {report['device']['device']}: packed stock-reference error {error} exceeds {allowance['atol']}")
    receipt["max_abs_error"] = error
    receipt.update(allowance)
    return receipt

