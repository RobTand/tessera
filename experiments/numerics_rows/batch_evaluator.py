"""Batch represented binary32 operands. Keep other operands on the integer path."""
import numpy as np

from experiments.numerics_rows.scale_index import CachedEvaluator


def round_bf16(values):
    """Input is binary32. Add the RNE bias to its bits, including subnormal ties."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        float(values.flat[np.flatnonzero(~np.isfinite(values))[0]]).as_integer_ratio()
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def product(a, b):
    """Both operands must be finite binary32 values represented in binary64.

    Their product needs at most 48 bits and exponents from -298 through 255.
    Binary64 multiplication is exact here. The binary32 cast performs one RNE.
    No binary64 product is used for arbitrary binary64 operands.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    for value in (a, b):
        if not np.isfinite(value).all():
            float(value.flat[np.flatnonzero(~np.isfinite(value))[0]]).as_integer_ratio()
    return (a * b).astype(np.float32)


class BatchEvaluator(CachedEvaluator):
    """Keep scalar own-output caching. Bypass the LRU for candidate matrices."""
    def __init__(self, read_cases, table=None, **kwargs):
        super().__init__(read_cases, table, **kwargs)
        self.table = table

    def matches(self, read, values, group, own):
        case = self.cases[self.keys[read]]
        codes = np.asarray([values[index] for index, _ in group], dtype=np.float64)
        operands = np.asarray([case[name] for name in ('g', 'u', 'a_scale', 'g_scale', 'u_scale')] + [self.table.limit], dtype=np.float64)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            exact32 = all(np.isfinite(x).all() and np.array_equal(x, x.astype(np.float32).astype(np.float64))
                          for x in (codes, operands))
            if not exact32:
                output = np.asarray([[self(read, (first, second), values) for second, _ in group]
                                     for first, _ in group])
                if not np.isfinite(output).all():
                    raise ValueError('mutated scale reference is not finite')
                return output == own

            def side(dot, wscale):
                if case['scale'] in ('a_s', 'sfa'):
                    return product(product(dot, codes), wscale)
                scaled = product(dot, case['a_scale'])
                return product(product(scaled, wscale), codes) if case['scale'] == 'router' else product(scaled, codes)

            g = np.minimum(round_bf16(side(case['g'], case['g_scale'])), self.table.limit)
            u = np.maximum(np.minimum(round_bf16(side(case['u'], case['u_scale'])), self.table.limit), -self.table.limit)
            d = np.asarray([self.table(float(value)) for value in g], dtype=np.float32)
            output = round_bf16(product(d[:, None], u[None, :]))
        if not np.isfinite(output).all():
            raise ValueError('mutated scale reference is not finite')
        if self.metrics is not None:
            self.metrics['batch_candidate_pairs'] = self.metrics.get('batch_candidate_pairs', 0) + output.size
            self.metrics['batch_matrices'] = self.metrics.get('batch_matrices', 0) + 1
        return output == own
