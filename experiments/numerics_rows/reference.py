"""IEEE RN reference and exact-input checks on represented operands."""
from __future__ import annotations

from fractions import Fraction as F
import math
import random
import struct
from functools import lru_cache


def outward(value):
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if math.isfinite(rounded) and F(rounded) < value else rounded


@lru_cache(None)
def reduction_growth(count):
    return outward(F(2**23 + 1, 2**23) ** count)


class FiniteBound:
    """Outward binary64 bounds with the validity of every preceding stage."""
    def __init__(self, value, held=True):
        import torch
        self.value = torch.as_tensor(value, dtype=torch.float64)
        self.held = (self.value < 2.0**126) & held

    @staticmethod
    def exact_sum(values, axis):
        import numpy as np
        import torch
        array = np.moveaxis(values.detach().double().cpu().numpy(), axis, -1)
        finite = np.isfinite(array).all(-1)
        result = np.full(array.shape[:-1], math.inf)
        bits = array[finite].view(np.uint64)
        # Normal significands have 52 fraction bits and an exponent bias of 1023.
        exponent = ((bits >> 52) & 2047).astype(np.int64)
        mantissa = ((bits & (2**52 - 1)) | ((exponent != 0).astype(np.uint64) << 52)).astype(np.int64)
        mantissa = np.where(bits >> 63, -mantissa, mantissa)
        exponent = np.maximum(exponent, 1)
        base = np.min(np.where(mantissa != 0, exponent, 2047), axis=-1, keepdims=True, initial=2047)
        integers = mantissa.astype(object) << np.maximum(exponent - base, 0).astype(object)
        totals = integers.sum(-1)
        def rounded(integer, exponent):
            exponent = int(exponent) - 1075
            return outward(F(int(integer) << max(exponent, 0), 1 << max(-exponent, 0)))
        result[finite] = [rounded(n, e) for n, e in zip(totals, base[:, 0])]
        return torch.as_tensor(result, device=values.device)

    @staticmethod
    def up(value):
        import torch
        return torch.nextafter(value, torch.full_like(value, math.inf))

    @classmethod
    def dot(cls, products, k, splits, top_k=0):
        total = cls.exact_sum(products.abs(), -1)
        return cls(total).rounded(k + splits) if k + splits + top_k < 2**20 else cls(total, False)

    def rounded(self, count=1, *, bf16=False):
        if count >= 2**20:
            return FiniteBound(self.value, False)
        factor, tiny = (1 + 2**-7, 2**-133) if bf16 else (reduction_growth(count), count * 2**-149)
        return FiniteBound(self.up(self.up(self.value * factor) + tiny), self.held)

    def multiply(self, scale):
        import torch
        value = scale.value if isinstance(scale, FiniteBound) else torch.as_tensor(scale, dtype=torch.float64, device=self.value.device).abs()
        held = self.held & scale.held if isinstance(scale, FiniteBound) else self.held
        return FiniteBound(self.up(self.value * value), held).rounded()

    def activation(self, up, limit, table_held):
        clamped = FiniteBound(up.value.clamp(max=limit), up.held & table_held)
        return self.multiply(clamped).rounded(bf16=True)

    def token_sum(self, top_k):
        value = self.value.reshape(-1, top_k, self.value.shape[-1])
        held = self.held.reshape_as(value).all(1)
        return FiniteBound(self.exact_sum(value, 1), held).rounded(top_k).rounded(bf16=True)

    @classmethod
    def join(cls, values, axis=0, *, stack=False):
        import torch
        operation = torch.stack if stack else torch.cat
        return cls(operation([x.value for x in values], axis), operation([x.held for x in values], axis))

    def __getitem__(self, index):
        return FiniteBound(self.value[index], self.held[index])

    def __setitem__(self, index, value):
        self.value[index], self.held[index] = value.value, value.held

    def reshape(self, *shape):
        return FiniteBound(self.value.reshape(*shape), self.held.reshape(*shape))


def power(exponent):
    return F(2) ** exponent


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rn_bits(value, precision=24, emin=-126, emax=127):
    """Round a rational with integer arithmetic, including ties and subnormals."""
    negative_zero = isinstance(value, float) and value == 0 and math.copysign(1.0, value) < 0
    value = F(value)
    sign = int(value < 0 or negative_zero)
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
    product = F(a) * F(b)
    if not product:
        return math.copysign(0.0, math.copysign(1.0, a) * math.copysign(1.0, b))
    return rn32(product)


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


def bf16_array(values):
    import torch
    return torch.as_tensor(values, dtype=torch.float32).to(torch.bfloat16).float().numpy()



def selftest(samples=1_000_000, *, device="cpu"):
    import numpy as np
    import torch
    from experiments.t4_code.fp4_arithmetic_attest import round32

    rng, mismatches, ties, subnormal_ties, bf16_subnormal_ties = random.Random(0), 0, 0, 0, 0
    a, b = np.empty(samples, dtype=np.float32), np.empty(samples, dtype=np.float32)
    products = np.empty(samples, dtype=np.float64)
    expected_mul, expected_add = np.empty_like(a), np.empty_like(a)
    bf_inputs, expected_bf = np.empty_like(a), np.empty_like(a)
    for i in range(samples):
        if i % 4 == 0:
            # An odd subnormal times 1/2 is an FP32 midpoint, including the zero tie.
            n = 0 if i == 0 else 2**22 - 1 if i == 4 else rng.randrange(2**22)
            left, right = f32_value(2 * n + 1), 0.5
            subnormal_ties += 1
            ties += 1
        elif i % 2 == 0:
            # An odd 24-bit significand times 3/2 gives a normal midpoint.
            left = math.ldexp(2**23 + 2 * rng.randrange(2**20) + 1, rng.randrange(-120, 121) - 23)
            right = 1.5
            ties += 1
        else:
            left = rn32(math.ldexp(rng.randrange(2**23, 2**24), rng.randrange(-149, 126) - 23))
            right = math.ldexp(rng.randrange(2**23, 2**24), -23)
        if i & 2:
            left = -left
        a[i], b[i] = left, right
        product, total = F(left) * F(right), F(left) + F(right)
        products[i] = float(product)
        expected_mul[i], expected_add[i] = float(round32(product)), float(round32(total))
        mismatches += rn32(product) != expected_mul[i] or rn32(total) != expected_add[i]
        if i % 4 == 0:
            value = math.ldexp(2 * (i // 4 % 128) + 1, -134)
            bf_inputs[i] = -value if i & 4 else value
            bf16_subnormal_ties += 1
        else:
            bf_inputs[i] = expected_mul[i]
        expected_bf[i] = bf16(float(bf_inputs[i]))
    ta, tb = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
    actual = {"numpy.multiply": a * b, "numpy.add": a + b,
              "numpy.cast": products.astype(np.float32),
              "numpy.bf16": bf16_array(bf_inputs),
              "torch.multiply": (ta * tb).cpu().numpy(), "torch.add": (ta + tb).cpu().numpy(),
              "torch.cast": torch.from_numpy(products).to(device).float().cpu().numpy(),
              "torch.bf16": torch.from_numpy(bf_inputs).to(device).to(torch.bfloat16).float().cpu().numpy()}
    counts = {name: int(np.count_nonzero(values != (expected_bf if name.endswith("bf16") else
               expected_mul if name.endswith(("multiply", "cast")) else expected_add))) for name, values in actual.items()}
    return {"samples": samples, "ties": ties, "subnormal_ties": subnormal_ties,
            "bf16_subnormal_ties": bf16_subnormal_ties, "device": device, "backends": counts,
            "mismatches": mismatches + sum(counts.values())}


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
    return {"equal": not bool(unequal.any()), "elements": got.numel(), "mismatches": int(unequal.sum()),
            "signed_zero_diffs": int(signed.sum()), "first_mismatch": first}
