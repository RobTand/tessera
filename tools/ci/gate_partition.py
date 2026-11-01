#!/usr/bin/env python3
"""Deterministic duration-balanced shards shared by gate and nightly CI."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path, PurePosixPath


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




def selection_shard_count(files: Sequence[str], durations: Mapping[str, float], maximum: int,
                          *, workers: int, single_shard_seconds: float) -> int:
    """Keep small PRs on one slot; larger selections share the full gate matrix."""
    unknown = default_unknown_seconds(durations)
    # Reuse partition validation so an unknown file or invalid cost cannot be free.
    packed = shards(files, durations, workers, unknown_seconds=unknown)
    wall = max(row['estimated_seconds'] for row in packed)
    useful = sum(durations.get(path, unknown) > 0 for path in files)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('maximum must be a positive integer')
    threshold = _cost(single_shard_seconds)
    return 1 if wall <= threshold else min(maximum, max(1, useful))


def lane_matrix(files, durations, count, runners):
    """Each lane covers every candidate file; missing dependencies are reported."""
    if not runners:
        raise ValueError('runners cannot be empty')
    result = []
    for lane in ('bytes', 'torch'):
        costs = durations.get(lane, {})
        for row in shards(files, costs, count, unknown_seconds=default_unknown_seconds(costs)):
            row.update(index=len(result), lane=lane, runner=runners[row['index'] % len(runners)])
            result.append(row)
    return {'shards': count, 'include': result}
