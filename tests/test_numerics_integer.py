"""Compare integer and batch operations with the retained Fraction oracle."""
from fractions import Fraction as F
import itertools
import json
import math
import os
from pathlib import Path
import random
import struct
import unittest

import numpy as np

from experiments.numerics_rows import reference as actual
from experiments.numerics_rows.batch_evaluator import product, round_bf16
from experiments.numerics_rows.scale_index import ModuleMemo, evaluate, joint_coverage, scale_values


def setUpModule():
    import torch
    torch.set_num_threads(1)


def oracle_bits(value, precision=24, emin=-126, emax=127):
    """Keep reference.py:96 at d9774518 as the independent oracle."""
    negative_zero = isinstance(value, float) and value == 0 and math.copysign(1.0, value) < 0
    value = F(value)
    sign = int(value < 0 or negative_zero)
    value = abs(value)
    if not value:
        return sign << 31
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    exponent -= value < F(2) ** exponent
    quantum = F(2) ** (max(exponent, emin) - precision + 1)
    quotient, remainder = divmod(value / quantum, 1)
    integer = int(quotient) + (remainder > F(1, 2) or remainder == F(1, 2) and int(quotient) & 1)
    rounded = integer * quantum
    if rounded >= F(2) ** (emax + 1):
        return (sign << 31) | 0x7F800000
    return actual.f32_bits(-float(rounded) if sign else float(rounded))


def oracle_rn32(value):
    return actual.f32_value(oracle_bits(value))


def oracle_bf16(value):
    return actual.f32_value(oracle_bits(value, 8) & 0xffff0000)


def oracle_multiply(a, b):
    value = F(a) * F(b)
    if not value:
        return math.copysign(0.0, math.copysign(1.0, a) * math.copysign(1.0, b))
    return oracle_rn32(value)


def oracle_swiglu(g, u, table):
    g = min(oracle_bf16(g), table.limit)
    u = max(min(oracle_bf16(u), table.limit), -table.limit)
    return oracle_bf16(oracle_multiply(table(g), u))


def oracle_evaluate(case, wrong, values, table):
    def side(dot, index, wscale):
        if case['scale'] in ('a_s', 'sfa'):
            return oracle_multiply(oracle_multiply(dot, values[index]), wscale)
        scaled = oracle_multiply(dot, case['a_scale'])
        if case['scale'] == 'router':
            return oracle_multiply(oracle_multiply(scaled, wscale), values[index])
        return oracle_multiply(scaled, values[index])
    g = side(case['g'], wrong[0], case['g_scale'])
    if len(wrong) == 1:
        return oracle_bf16(oracle_multiply(g, case.get('router_weight', 1.0)) if case['scale'] != 'router' else g)
    return oracle_swiglu(g, side(case['u'], wrong[1], case['u_scale']), table)


def edges():
    values = [0.0, -0.0, math.inf, -math.inf, math.nan,
              float.fromhex('0x0.0000000000001p-1022'), float.fromhex('0x1.fffffffffffffp+1023')]
    for precision in (8, 11, 24):
        emin = -14 if precision == 11 else -126
        for exponent in (emin, -1, 0, 1, 15, 127):
            for k in (0, 1, 2, 2**(precision - 1) - 2, 2**(precision - 1) - 1):
                midpoint = math.ldexp(2**precision + 2*k + 1, exponent - precision)
                values.extend((math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)))
        for k in (0, 1, 2, 2**(precision - 1) - 2, 2**(precision - 1) - 1):
            midpoint = math.ldexp(2*k + 1, emin - precision)
            values.extend((math.nextafter(midpoint, 0), midpoint, math.nextafter(midpoint, math.inf)))
    return values + [-v for v in values]


class OracleChecks(unittest.TestCase):
    def equal_operation(self, new, old, *args):
        try:
            expected = old(*args)
        except (ValueError, OverflowError, TypeError) as exc:
            with self.assertRaises(type(exc)) as got:
                new(*args)
            self.assertEqual(str(got.exception), str(exc))
        else:
            self.assertEqual(struct.pack('<d', float(new(*args))), struct.pack('<d', float(expected)), args)


class IntegerReferenceTest(OracleChecks):
    def test_rounding_random_binary64_and_all_edge_classes(self):
        rng = random.Random(1717)
        values = edges() + [struct.unpack('<d', rng.getrandbits(64).to_bytes(8, 'little'))[0] for _ in range(4096)]
        values += [F(1, 3), F(2)**4096, -F(2)**-4096]
        for value in values:
            for new, old in ((actual.rn32, oracle_rn32), (actual.bf16, oracle_bf16)):
                self.equal_operation(new, old, value)
            self.equal_operation(lambda v: actual.rn_bits(v, 11, -14, 15),
                                 lambda v: oracle_bits(v, 11, -14, 15), value)

    def test_multiply_random_binary64_and_wide_products(self):
        rng = random.Random(1718)
        values = [0.0, -0.0, math.inf, -math.inf, math.nan, 2**-1074,
                  float.fromhex('0x1.fffffffffffffp+1023'), 1.5, -1.5, 2**-126, 2**-149]
        pairs = list(itertools.product(values, repeat=2))
        pairs += [(struct.unpack('<d', rng.getrandbits(64).to_bytes(8, 'little'))[0],
                   struct.unpack('<d', rng.getrandbits(64).to_bytes(8, 'little'))[0]) for _ in range(4096)]
        pairs += [(1 + 3*2**-24 + 2**-52, 1 - 2**-52), (F(2)**4096, F(2)**-4096)]
        for pair in pairs:
            self.equal_operation(actual.multiply, oracle_multiply, *pair)

    def test_batch_casts_subnormals_ties_and_overflow(self):
        finite32 = [float(np.float32(v)) for v in edges() if math.isfinite(v) and abs(v) <= 2**127]
        finite32 += [actual.f32_value(bits) for bits in (0x7f7f0000, 0x7f7f7fff, 0x7f7f8000,
                                                       0x7f7f8001, 0x7f7fffff, 0xff7fffff)]
        finite32 += [actual.f32_value(sign | ((high << 16) | 0x8000) + offset)
                     for sign in (0, 0x80000000) for high in (0, 1, 0x7f, 0x80, 0x3f80, 0x3f81)
                     for offset in (-1, 0, 1)]
        with np.errstate(all='ignore'):
            rounded = round_bf16(np.asarray(finite32, dtype=np.float32))
            for value, result in zip(finite32, rounded):
                self.equal_operation(lambda _: float(result), oracle_bf16, value)
            values = [0.0, -0.0, 2**-149, -2**-149, 2**-126, 0.5, 1.5, 2**127,
                      actual.f32_value(0x7f7fffff), actual.f32_value(0x3f800001)]
            for a, b in itertools.product(values, repeat=2):
                self.equal_operation(lambda x, y: float(product(x, y)), oracle_multiply, a, b)

    def test_nonfinite_refusals(self):
        for value in (math.inf, -math.inf, math.nan):
            self.equal_operation(lambda v: float(round_bf16(np.asarray([v], dtype=np.float32))[0]), oracle_bf16, value)
            for other in (0.0, 1.0):
                self.equal_operation(lambda a, b: float(product(a, b)), oracle_multiply, value, other)


class MeasuredTableTest(OracleChecks):
    @classmethod
    def setUpClass(cls):
        from experiments.numerics_rows.cases import DTable
        path = Path(os.environ.get('NUMERICS_Q1', '/inputs/q1.json'))
        record = next(t for t in json.loads(path.read_text())['tables'] if t['key'] == 'e4m3mma')
        cls.table = DTable(record)

    def test_batch_upper_up_clamp_edges(self):
        coding = [{'values': [1.0, 2.0]}]
        values, group = scale_values(coding[0]), [(0, 1), (1, 2)]
        for g in (-1.0, 1.0):
            for bits in (0x411fffff, 0x41200000, 0x41200001, 0x411f0000, 0x41210000):
                case = {'read': [0, 0], 'scale': 'row', 'g': g, 'u': actual.f32_value(bits),
                        'a_scale': 1.0, 'g_scale': 1.0, 'u_scale': 1.0}
                evaluator = ModuleMemo(self.table, vectorized=True).evaluator([case], coding, {})
                own = oracle_evaluate(case, (0, 0), values, self.table)
                expected = [[oracle_evaluate(case, (i, j), values, self.table) == own
                             for j in range(2)] for i in range(2)]
                self.assertEqual(evaluator.matches((0, 0), values, group, own).tolist(), expected)

    def test_domain_points_and_clamp_edges(self):
        from experiments.numerics_rows.cases import swiglu
        points = [actual.f32_value(bits) for bits in self.table.table]
        points += [math.nextafter(sign*10.0, direction) for sign in (-1, 1) for direction in (-math.inf, math.inf)]
        points += [actual.f32_value(bits) for bits in (0x411fffff, 0x41200000, 0x41200001,
                                                     0x411f0000, 0x41210000, 0xc1200001, 0xc11fffff)]
        for point in points:
            self.equal_operation(lambda g: swiglu(g, -1.0, self.table, 10.0),
                                 lambda g: oracle_swiglu(g, -1.0, self.table), point)
        for g, u in itertools.product((10.0, -10.0, 0.0, -0.0, *points[-7:]), repeat=2):
            self.equal_operation(lambda a, b: swiglu(a, b, self.table, 10.0),
                                 lambda a, b: oracle_swiglu(a, b, self.table), g, u)
        for value in (math.inf, -math.inf, math.nan):
            for g, u in ((value, 1.0), (1.0, value)):
                self.equal_operation(lambda a, b: swiglu(a, b, self.table, 10.0),
                                     lambda a, b: oracle_swiglu(a, b, self.table), g, u)

    def test_batch_matches_fraction_outputs_and_ordered_witnesses(self):
        for scale in ('row', 'a_s', 'sfa', 'router'):
            for g, u in ((0.0, -0.0), (71680.0, -19712.0), (71680.0, 19712.0),
                         (1.0, -1.0), (1 + 2**-40, -1.0)):
                coding = [{'values': [2**-125, 2**-6, 0.25, 0.5, 512.0]}]
                values = scale_values(coding[0])
                cases = [{'read': list(read), 'scale': scale, 'g': g, 'u': u,
                          'a_scale': 1/512, 'g_scale': 1/4096, 'u_scale': 1/2048}
                         for read in itertools.product(range(5), repeat=2)]
                memo = ModuleMemo(self.table, vectorized=True)
                evaluator = memo.evaluator(cases, coding, {})
                group = [(i, 1 << i) for i in range(5)]
                for case in cases:
                    read = tuple(case['read'])
                    own = oracle_evaluate(case, read, values, self.table)
                    self.equal_operation(lambda: evaluator(read, read, values), lambda: own)
                    expected = [[oracle_evaluate(case, (i, j), values, self.table) == own for j in range(5)] for i in range(5)]
                    self.assertEqual(evaluator.matches(read, values, group, own).tolist(), expected)
                old = joint_coverage([c['read'] for c in cases], coding,
                    lambda read, wrong, v: oracle_evaluate(cases[read[0]*5 + read[1]], wrong, v, self.table), 5)
                new = joint_coverage([c['read'] for c in cases], coding, evaluator, 5, memo=memo)
                self.assertEqual(json.dumps(new, sort_keys=True), json.dumps(old, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
