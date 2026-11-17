"""Index-coding passes and the draft-7 joint-read check."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
import math
import json
from itertools import islice
import struct
import sys

from experiments.numerics_rows.reference import bf16, multiply


def digits(n, base):
    if n < 1 or not 2 <= base <= 121:
        raise ValueError("index coding requires N >= 1 and 2 <= base <= 121")
    count, capacity = 0, 1
    while capacity < n:
        count, capacity = count + 1, capacity * base
    return [[(index // base**p) % base for index in range(n)] for p in range(count)]

class CodingValues(Sequence):
    def __init__(self, entry):
        self.entry = entry
    def __len__(self):
        return self.entry["n"]
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if not 0 <= index < len(self):
            raise IndexError(index)
        e = self.entry
        held = e["up"] if e["family"] == "gate" else e["gate"] if e["family"] == "up" else []
        if held and held[0] <= index < held[1]:
            return e["up_constant"] if e["family"] == "gate" else e["gate_constant"]
        return 2.0**((index // e["base"]**e["digit"]) % e["base"] + e["low"])


def scale_values(entry):
    return entry["values"] if "values" in entry else CodingValues(entry)

def passes(n, low, high, *, gate=(), up=(), gate_constant=1.0, up_constant=1.0, compact=False):
    base = high - low + 1
    if not 2 <= base <= 121 or n < 1:
        raise ValueError("invalid scale coding range")
    count, capacity = 0, 1
    while capacity < n:
        count, capacity = count + 1, capacity * base
    families = ("gate", "up") if gate or up else ("plain",)
    result = []
    for family in families:
        for position in range(count):
            entry = {"family": family, "digit": position, "n": n, "base": base, "low": low,
                     "gate": [gate.start, gate.stop] if gate else [], "up": [up.start, up.stop] if up else [],
                     "gate_constant": gate_constant, "up_constant": up_constant}
            if not compact:
                entry["values"] = list(CodingValues(entry))
            result.append(entry)
    return {"base": base, "families": list(families), "passes_per_family": count, "passes_total": len(result),
            "coding": result}


def coding_groups(arrays):
    groups = []
    for values in arrays:
        by_value = {}
        for index, value in enumerate(values):
            if value not in by_value:
                by_value[value] = [index, 0]
            by_value[value][1] |= 1 << index
        groups.append(list(by_value.values()))
    return groups


def matching_pairs(read, arrays, groups, same, n, batch=None):
    if len(read) == 1:
        possible = (1 << n) - 1
        for position, group in enumerate(groups):
            possible &= sum(mask for index, mask in group if same((index,), position))
        while possible:
            bit = possible & -possible
            yield (bit.bit_length() - 1,)
            possible ^= bit
    else:
        allowed = []
        for position, group in enumerate(groups):
            if batch is None:
                allowed.append({arrays[position][first]: sum(mask for second, mask in group if same((first, second), position))
                                for first, _ in group})
            else:
                equal = batch(position, group)
                allowed.append({arrays[position][first]: sum(mask for (_, mask), keep in zip(group, row) if keep)
                                for (first, _), row in zip(group, equal)})
        for first in range(n):
            possible = (1 << n) - 1
            for position, masks in enumerate(allowed):
                possible &= masks[arrays[position][first]]
                if not possible: break
            while possible:
                bit = possible & -possible
                yield first, bit.bit_length() - 1
                possible ^= bit


def joint_coverage(reads, coding, evaluate, n, *, memo=None, progress=None):
    """Check all tuples. Indices affect evaluation only through scale values."""
    arrays = [scale_values(entry) for entry in coding]
    groups = memo.groups(coding, arrays) if memo else coding_groups(arrays)
    covered, uncovered = 0, []
    for read in reads:
        read = tuple(read)
        if len(read) not in (1, 2) or any(not 0 <= i < n for i in read):
            raise ValueError("scale reads must contain one or two array indices")
        own = [evaluate(read, read, values) for values in arrays]
        if not all(math.isfinite(v) for v in own):
            raise ValueError("reference is not finite")
        def same(wrong, position):
            value = evaluate(read, wrong, arrays[position])
            if not math.isfinite(value):
                raise ValueError("mutated scale reference is not finite")
            return value == own[position]
        batch = (lambda position, group: evaluate.matches(read, arrays[position], group, own[position])) if hasattr(evaluate, 'matches') else None
        search = lambda: matching_pairs(read, arrays, groups, same, n, batch=batch)
        candidates = memo.matches(evaluate.keys[read], own, search) if memo else search()
        collision = next((pair for pair in candidates if pair != read), None)
        if collision is None:
            covered += 1
        else:
            uncovered.append({"read": list(read), "aliases": list(collision)})
        if progress is not None:
            progress({"reads": covered + len(uncovered), "reads_covered": covered, "uncovered_reads": uncovered})
    return {"reads": len(reads), "reads_covered": covered, "uncovered_reads": uncovered}


def evaluate(case, wrong, values, table=None):
    """Repeat the target array's position in the epilogue, then round to BF16."""
    def side(dot, index, wscale):
        if case["scale"] == "a_s":
            return multiply(multiply(dot, values[index]), wscale)
        if case["scale"] == "sfa":
            return multiply(multiply(dot, values[index]), wscale)
        scaled = multiply(dot, case["a_scale"])
        if case["scale"] == "router":
            return multiply(multiply(scaled, wscale), values[index])
        return multiply(scaled, values[index])
    g = side(case["g"], wrong[0], case["g_scale"])
    if len(wrong) == 1:
        return bf16(multiply(g, case.get("router_weight", 1.0)) if case["scale"] != "router" else g)
    u = side(case["u"], wrong[1], case["u_scale"])
    from experiments.numerics_rows.cases import swiglu
    return swiglu(g, u, table, table.limit)


class CachedEvaluator:
    """Cache the unchanged Fraction evaluator on its exact represented inputs.

    The instance owns one immutable measured D table, including its clamp limit.
    IEEE binary64 bytes distinguish all represented inputs, including signed zero.
    Index, column and expert labels affect evaluation only through the scale values
    read at those indices. Omit only operands the selected epilogue does not read.
    The LRU bounds memory; optional tuple counts use one bit per value-code pair
    per operand signature, so evictions do not inflate the distinct-input count.
    """
    def __init__(self, read_cases, table=None, *, coding=(), metrics=None, maxsize=2**18, cached=None):
        self.keys, self.cases, self.seen = {}, {}, {}
        self.metrics, self.previous = metrics, None
        bits = lambda value: struct.pack("<d", value)
        self.bits = bits
        for case in read_cases:
            read, scale = tuple(case["read"]), case["scale"]
            names = ["g"] + (["u"] if len(read) == 2 else [])
            if scale not in ("a_s", "sfa"):
                names.append("a_scale")
            if scale in ("a_s", "sfa", "router"):
                names += ["g_scale"] + (["u_scale"] if len(read) == 2 else [])
            operands = tuple((name, bits(case[name])) for name in names)
            if len(read) == 1 and scale != "router":
                operands += (("router_weight", bits(case.get("router_weight", 1.0))),)
            key = scale, len(read), operands
            self.keys[read], self.cases[key] = key, dict(case)
        if metrics is not None:
            alphabet = set()
            for entry in coding:
                values = (entry["values"] if "values" in entry else
                          [2.0**(entry["low"] + i) for i in range(entry["base"])] +
                          [entry["gate_constant"], entry["up_constant"]])
                alphabet.update(map(bits, values))
            self.codes = {value: i for i, value in enumerate(sorted(alphabet))}
            self.stride = len(self.codes)
            metrics.update(operand_signatures=len(self.cases), scale_value_codes=self.stride, completed_reads=0)
        shared = cached
        @lru_cache(maxsize=maxsize)
        def cached(key, values):
            decoded = [struct.unpack("<d", value)[0] for value in values]
            scale, arity, operands = key
            case = {"read": list(range(arity)), "scale": scale, "g_scale": 1.0, "u_scale": 1.0,
                    **{name: struct.unpack("<d", value)[0] for name, value in operands}}
            return evaluate(case, range(arity), decoded, table)
        self.cached = shared or cached

    def __call__(self, read, wrong, values):
        key = self.keys[tuple(read)]
        codes = tuple(self.bits(values[i]) for i in wrong)
        if self.metrics is not None:
            read = tuple(read)
            if self.previous is not None and self.previous != read:
                self.metrics["completed_reads"] += 1
            self.previous = read
            position = self.codes[codes[0]]
            if len(codes) == 2:
                position = position * self.stride + self.codes[codes[1]]
            self.seen[key] = self.seen.get(key, 0) | 1 << position
        return self.cached(key, codes)

    def report(self):
        info = self.cached.cache_info()._asdict()
        return {**info, "requests": info["hits"] + info["misses"],
                "distinct_evaluate_tuples": sum(mask.bit_count() for mask in self.seen.values()) if self.metrics is not None else None}


class ModuleMemo:
    """Keep one table and bounded coding, evaluator and alias caches."""
    def __init__(self, table, *, vectorized=True):
        self.table, self.cached, self.descriptor = table, None, None
        self.vectorized = vectorized
        self.group_cache, self.group_bytes = OrderedDict(), 0
        self.group_hits = self.group_misses = 0
        self._matches = lru_cache(maxsize=2**16)(self._remember)

    def groups(self, coding, arrays):
        descriptor = json.dumps(coding, sort_keys=True, separators=(",", ":"))
        self.descriptor = descriptor
        if descriptor in self.group_cache:
            self.group_hits += 1
            self.group_cache.move_to_end(descriptor)
            return self.group_cache[descriptor][0]
        self.group_misses += 1
        groups = coding_groups(arrays)
        size = sum(sys.getsizeof(group) + sum(sys.getsizeof(pair) + sum(map(sys.getsizeof, pair))
                                            for pair in group) for group in groups)
        while self.group_cache and self.group_bytes + size > 128 * 2**20:
            _, (_, removed) = self.group_cache.popitem(last=False)
            self.group_bytes -= removed
        if size <= 128 * 2**20:
            self.group_cache[descriptor] = groups, size
            self.group_bytes += size
        return groups

    def _remember(self, descriptor, signature, own):
        return tuple(islice(self.search(), 2))

    def matches(self, signature, own, search):
        # Two least pairs suffice: excluding one correct pair leaves the least alias.
        self.search = search
        try:
            return self._matches(self.descriptor, signature, tuple(struct.pack("<d", v) for v in own))
        finally:
            self.search = None

    def evaluator(self, read_cases, coding, metrics):
        evaluator_type = CachedEvaluator
        if self.vectorized:
            from experiments.numerics_rows.batch_evaluator import BatchEvaluator
            evaluator_type = BatchEvaluator
        evaluator = evaluator_type(read_cases, self.table, coding=coding, metrics=metrics, cached=self.cached)
        self.cached = evaluator.cached
        return evaluator

    def report(self):
        return {"evaluate": self.cached.cache_info()._asdict() if self.cached is not None else {},
                "matches": self._matches.cache_info()._asdict(),
                "group_hits": self.group_hits, "group_misses": self.group_misses,
                "group_entries": len(self.group_cache), "group_bytes": self.group_bytes}


def make_plan(scale, n, read_cases, *, table=None, gate=(), up=(), ue4m3=False, metadata=None, memoize=False, metrics=None, memo=None):
    """Choose a normal power range. Check each output before claiming coverage."""
    def operand(c, side):
        value = c[side]
        if scale in ("a_s", "sfa"):
            value = multiply(value, c[side + "_scale"])
        else:
            value = multiply(value, c["a_scale"])
            if scale == "router":
                value = multiply(value, c[side + "_scale"])
        return value
    maximum = max((max(abs(operand(c, "g")), abs(operand(c, "u"))) for c in read_cases), default=0)
    low, high = (-6, 8) if ue4m3 else (-60, 60)
    mode0 = any(len(c["read"]) == 2 for c in read_cases)
    if maximum:
        ceiling = table.limit if mode0 else 2.0**127
        while high >= low and (not math.isfinite(maximum * 2.0**high) or abs(bf16(maximum * 2.0**high)) >= ceiling):
            high -= 1
        if mode0 and not ue4m3:
            low = max(-126, high - 120)
    if high - low < 1:
        return {"scale": scale, "base": 0, "families": [], "passes_per_family": 0, "passes_total": 0,
                "reads": len(read_cases), "reads_covered": 0, "uncovered_reads": [{"read": c["read"], "reason": "no normal coding range"} for c in read_cases],
                "mismatches": 0, "coding": [], "read_cases": read_cases, "array": metadata or {}}
    gmin = min((operand(c, "g") for c in read_cases if operand(c, "g") > 0), default=0)
    umin = min((abs(operand(c, "u")) for c in read_cases if operand(c, "u")), default=0)
    def held(minimum):
        if not minimum:
            return 1.0
        exponent = math.ceil(math.log2(table.limit / minimum))
        if exponent > 127:
            raise ValueError("a normal constant scale cannot hold this side at L")
        return 2.0**max(-126, exponent)
    plan = passes(n, low, high, gate=gate, up=up,
                  gate_constant=held(gmin) if mode0 else 1.0, up_constant=held(umin) if mode0 else 1.0, compact=True)
    cases = {tuple(c["read"]): c for c in read_cases}
    if memo is not None and memo.table is not table:
        raise ValueError("module memo belongs to another measured D table")
    evaluator = memo.evaluator(cases.values(), plan["coding"], metrics) if memo is not None else (
        CachedEvaluator(cases.values(), table, coding=plan["coding"], metrics=metrics) if memoize else None)
    if metrics is not None:
        metrics["coding"] = plan["coding"]
    try:
        result = joint_coverage(list(cases), plan["coding"], evaluator or (lambda read, wrong, values: evaluate(cases[read], wrong, values, table)), n, memo=memo,
                                progress=(lambda result: metrics.update(completed_result=result)) if metrics is not None else None)
    finally:
        if evaluator is not None and metrics is not None:
            metrics.update(evaluator.report())
            metrics["completed_reads"] = metrics.get("completed_result", {}).get("reads", 0)
    if evaluator is not None and metrics is not None:
        metrics["completed_reads"] = result["reads"]
    return {"scale": scale, **plan, **result, "mismatches": 0, "read_cases": read_cases, "array": metadata or {}}
