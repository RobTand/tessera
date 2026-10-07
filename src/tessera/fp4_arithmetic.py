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


def require_t4_device_qualification(report, *, device):
    """Keep the device gate closed until the two required reviews pass."""
    contract = require_probe_contract(report, device=device)
    reviews = report.get("reviews", {})
    if report.get("arithmetic_qualified") is not True or reviews.get("kernels_parent") is not True or reviews.get("independent") is not True:
        raise FP4QualificationError(f"T4 refused on {device}: arithmetic qualification awaits the required reviews")
    return contract


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
