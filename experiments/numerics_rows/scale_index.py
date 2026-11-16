"""Index-coding passes and the draft-7 joint-read check."""
from __future__ import annotations

from collections.abc import Sequence
import math

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


def joint_coverage(reads, coding, evaluate, n):
    """Check all tuples. The evaluator depends on indices only through their scale values."""
    groups = []
    arrays = [scale_values(entry) for entry in coding]
    for values in arrays:
        by_value = {}
        for index, value in enumerate(values):
            if value not in by_value:
                by_value[value] = [index, 0]
            by_value[value][1] |= 1 << index
        groups.append(list(by_value.values()))
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
        collision = None
        if len(read) == 1:
            possible = (1 << n) - 1
            for position, group in enumerate(groups):
                possible &= sum(mask for index, mask in group if same((index,), position))
            possible &= ~(1 << read[0])
            if possible:
                collision = ((possible & -possible).bit_length() - 1,)
        else:
            # Each pass has <= b+1 scale values. Resolve their Cartesian product once.
            # Bit sets retain all second indices, including pairs in the wrong family.
            allowed = []
            for position, group in enumerate(groups):
                masks = {}
                for first, _ in group:
                    masks[arrays[position][first]] = sum(mask for second, mask in group if same((first, second), position))
                allowed.append(masks)
            for first in range(n):
                possible = (1 << n) - 1
                for position, masks in enumerate(allowed):
                    possible &= masks[arrays[position][first]]
                    if not possible:
                        break
                if first == read[0]:
                    possible &= ~(1 << read[1])
                if possible:
                    collision = (first, (possible & -possible).bit_length() - 1)
                    break
        if collision is None:
            covered += 1
        else:
            uncovered.append({"read": list(read), "aliases": list(collision)})
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


def make_plan(scale, n, read_cases, *, table=None, gate=(), up=(), ue4m3=False, metadata=None):
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
    result = joint_coverage(list(cases), plan["coding"], lambda read, wrong, values: evaluate(cases[read], wrong, values, table), n)
    return {"scale": scale, **plan, **result, "mismatches": 0, "read_cases": read_cases, "array": metadata or {}}
