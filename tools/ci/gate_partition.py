#!/usr/bin/env python3
"""Deterministic duration-balanced shards shared by gate and nightly CI."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def _cost(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('durations must be finite nonnegative numbers, not booleans')
    return float(value)


def default_unknown_seconds(durations: Mapping[str, float]) -> float:
    """Nearest-rank p90; empty or all-zero inventories use one second."""
    costs = sorted(_cost(value) for value in durations.values())
    return (costs[math.ceil(len(costs) * .9) - 1] or 1.0) if costs else 1.0


def shards(files: Sequence[str], durations: Mapping[str, float], count: int,
           *, unknown_seconds: float) -> list[dict]:
    """Assign longest files first, breaking cost ties by path and shard index."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer')
    if len(files) != len(set(files)):
        raise ValueError('duplicate files')
    costs = {path: _cost(value) for path, value in durations.items()}
    unknown = _cost(unknown_seconds)
    result = [{'index': index, 'files': [], 'estimated_seconds': 0.0, 'unknown_files': []}
              for index in range(count)]
    for path in sorted(files, key=lambda path: (-costs.get(path, unknown), path)):
        row = min(result, key=lambda row: (row['estimated_seconds'], row['index']))
        row['files'].append(path)
        row['estimated_seconds'] += costs.get(path, unknown)
        if path not in costs:
            row['unknown_files'].append(path)
    for row in result:
        row['files'].sort()
        row['unknown_files'].sort()
    return result
