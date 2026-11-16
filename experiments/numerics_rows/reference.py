"""IEEE RN reference and exact-input checks on represented operands."""
from __future__ import annotations

from fractions import Fraction as F
import math
import random
import struct


def power(exponent):
    return F(2) ** exponent


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rn_bits(value, precision=24, emin=-126, emax=127):
    """Round a rational with integer arithmetic, including ties and subnormals."""
    value = F(value)
    sign = int(value < 0)
    value = abs(value)
    if not value:
        return sign << 31
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    exponent -= value < power(exponent)
    quantum = power(max(exponent, emin) - precision + 1)
    quotient, remainder = divmod(value / quantum, 1)
    integer = int(quotient) + (remainder > F(1, 2) or remainder == F(1, 2) and int(quotient) & 1)
    rounded = integer * quantum
    if rounded >= power(emax + 1):
        return (sign << 31) | 0x7F800000
    return f32_bits(-float(rounded) if sign else float(rounded))


def rn32(value):
    return f32_value(rn_bits(value))


def bf16_bits(value):
    return rn_bits(value, precision=8, emin=-126) >> 16


def bf16(value):
    return f32_value(bf16_bits(value) << 16)


def multiply(a, b):
    return rn32(F(a) * F(b))


def exact_input(products):
    products = [F(p) for p in products if p]
    if not products:
        return {"holds": True, "Q": "0", "T": "0", "sum": "0"}
    integers = [p.numerator for p in products]
    exponents = [(abs(n) & -abs(n)).bit_length() - 1 - (p.denominator.bit_length() - 1)
                 for n, p in zip(integers, products)]
    if any(p.denominator & (p.denominator - 1) for p in products):
        raise ValueError("operands must be binary rationals")
    q, total = power(min(exponents)), sum(map(abs, products))
    return {"holds": q >= power(-126) and total < 2**24 * q and total < power(128),
            "Q": str(q), "T": str(total), "sum": str(sum(products))}


def output(products, scales):
    facts = exact_input(products)
    if not facts["holds"]:
        raise ValueError("X is false")
    result = rn32(F(facts["sum"]))
    for scale in scales:
        result = multiply(result, scale)
    if not math.isfinite(result):
        raise ValueError("reference is not finite")
    return bf16(result)


def selftest(samples=1_000_000):
    from experiments.t4_code.fp4_arithmetic_attest import round32

    rng, mismatches, ties = random.Random(0), 0, 0
    for i in range(samples):
        exponent = rng.randrange(-149, 104)
        numerator = rng.randrange(1, 2**24)
        if i % 2 == 0:
            value = F(2 * numerator + 1) * power(exponent - 1)
            ties += 1
        else:
            value = F(rng.randrange(1, 2**40)) * power(exponent - 16)
        if i & 2:
            value = -value
        actual = rn32(value)
        mismatches += F(actual) != round32(value)
    return {"samples": samples, "ties": ties, "mismatches": mismatches}


def compare(got, ref):
    import torch

    if got.shape != ref.shape or got.dtype != ref.dtype:
        return {"equal": False, "reason": "wrong shape or dtype", "mismatches": max(got.numel(), ref.numel())}
    unequal = (got != ref) | torch.isnan(got) | torch.isnan(ref)
    signed = (got == 0) & (ref == 0) & (torch.signbit(got) != torch.signbit(ref))
    indices = unequal.nonzero()
    first = None
    if indices.numel():
        index = tuple(indices[0].tolist())
        first = {"index": list(index), "got_bits": int(got.view(torch.int16)[index]) & 65535,
                 "ref_bits": int(ref.view(torch.int16)[index]) & 65535}
    return {"equal": not bool(unequal.any()), "mismatches": int(unequal.sum()),
            "signed_zero_diffs": int(signed.sum()), "first_mismatch": first}
