"""Admitted CPU preflight and GPU checks for the T16 scale cutover."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pytest
import torch


class Collection:
    def __init__(self):
        self.nodeids = []

    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    targets = args.pytest_args
    if targets[:1] == ['--']:
        targets = targets[1:]
    if not targets:
        raise ValueError('The check needs explicit pytest targets.')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / 'src'), str(root / 'tests')]
    if args.cpu_preflight:
        # This exact arithmetic witness also checks the input constants.
        value = torch.tensor(1.0, dtype=torch.bfloat16).float()
        scale = torch.tensor(1.0 + 2.0 ** -8, dtype=torch.float32)
        route_weight = torch.tensor(0.75, dtype=torch.float32)
        old = ((value * scale).bfloat16().float() * route_weight).bfloat16()
        new = ((value * scale) * route_weight).bfloat16()
        assert float(old) == 0.75 and float(new) == 0.75390625
        collector = Collection()
        result = pytest.main(['--collect-only', '-q', *targets], plugins=[collector])
        record = {'phase': 'CPU preflight', 'action': os.environ.get('PRISMABUILD_ACTION_KEY'),
                  'returncode': int(result), 'targets': targets, 'nodeids': collector.nodeids,
                  'old_witness': float(old), 'new_witness': float(new),
                  'scope': 'CPU imports, parser, test collection and scale witness. No CUDA arithmetic ran.'}
        (args.out / 'cpu-preflight.json').write_text(json.dumps(record, indent=2))
        return int(result)
    return int(pytest.main(['-n', '2', '--dist', 'worksteal', '--durations=20',
                           '--strict-cuda', '--surface-json', str(args.out / 'surface.json'),
                           '-q', *targets]))


if __name__ == '__main__':
    raise SystemExit(main())
