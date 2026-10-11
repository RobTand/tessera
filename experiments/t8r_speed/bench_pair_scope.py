"""Requested cells for the synthetic pair benchmark, without CUDA imports."""
from itertools import product

CASES = {
    "r4": (4, None), "r4+1b": (4, "1b"), "r4+q": (4, 0.25),
    "r5-1b": (4, "-1b"), "r5": (5, None),
    "r3": (3, None), "r3+q": (3, 0.25), "r3+1b": (3, "1b"),
}


def requested_cells(cases, modes, ms):
    cases = cases.split(",")
    modes = [int(v) for v in modes.split(",")]
    ms = [int(v) for v in ms.split(",")]
    if (not cases or any(v not in CASES for v in cases)
            or not modes or any(v not in (0, 1, 2) for v in modes)
            or not ms or any(v <= 0 for v in ms)
            or any(len(set(values)) != len(values) for values in (cases, modes, ms))):
        raise ValueError("invalid or duplicate requested cases/modes/M")
    return set(product(cases, modes, ms))
