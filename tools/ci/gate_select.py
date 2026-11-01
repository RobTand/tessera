#!/usr/bin/env python3
"""Adapt Tessera's reverse-import selector to the two hosted CPU lanes."""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition import lane_matrix, selection_shard_count


def full_gate(pr, repository):
    same = pr['head'].get('repo', {}).get('full_name') == repository
    batch = (pr['head']['ref'].startswith('mergify/merge-queue/') and
             pr.get('user', {}).get('login') == 'mergify[bot]' and
             pr.get('user', {}).get('id') == 37929162)
    canary = any(label['name'] == 'gate-canary' for label in pr.get('labels', []))
    return same and (batch or canary)


def population(root):
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
                             '--', 'tests/test_*.py'], cwd=root, capture_output=True, check=True)
    return sorted(set(result.stdout.decode().rstrip('\0').split('\0')) - {''})


def select_files(root, base, head, execute=subprocess.run):
    all_files = population(root)
    try:
        result = execute([sys.executable, str(root / 'tools/impacted_tests.py'),
                          '--ref', f'{base}...{head}', '--json'], cwd=root,
                         capture_output=True, text=True, check=False, timeout=120)
        if result.returncode:
            raise ValueError(result.stderr or f'selector exit {result.returncode}')
        if not result.stdout.strip():
            return {'verdict': 'none', 'tests': [], 'reason': 'no changed paths'}
        selection = json.loads(result.stdout)
        if selection['verdict'] == 'full':
            selection['tests'] = all_files
        elif (selection['verdict'] not in ('narrowed', 'none') or
              not set(selection['tests']) <= set(all_files)):
            raise ValueError('selector returned an invalid verdict or unknown files')
        return selection
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        return {'verdict': 'full', 'tests': all_files, 'reason': str(error)}


def issue_state(number):
    request = urllib.request.Request(f'https://api.github.com/repos/RobTand/tessera/issues/{number}',
                                     headers={'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)['state']


def quarantine(config, *, issue_state=issue_state):
    lanes = config['lanes']
    owners = config['issues']
    for lane in ('bytes', 'torch'):
        if not isinstance(lanes[lane], list) or len(lanes[lane]) != len(set(lanes[lane])):
            raise ValueError('quarantine must list unique test nodes')
        for nodeid in lanes[lane]:
            if '::' not in nodeid or not nodeid.startswith('tests/'):
                raise ValueError('quarantine requires an exact test node')
            number = owners.get(nodeid)
            if type(number) is not int or number < 1:
                raise ValueError('quarantine needs an owner issue for ' + nodeid)
    for number in set(owners[nodeid] for nodes in lanes.values() for nodeid in nodes):
        if issue_state(number) != 'open':
            raise ValueError(f'quarantine owner #{number} must remain open')
    return lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--mode', choices=['full', 'selection'], required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--runners', default='ubuntu-24.04,ubuntu-24.04-arm')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    selection = ({'verdict': 'full', 'tests': population(root), 'reason': 'full CPU gate'}
                 if args.mode == 'full' else select_files(root, args.base, args.head))
    files = population(root) if args.mode == 'full' else selection['tests']
    if args.mode == 'full' and not files:
        raise ValueError('empty CPU population refused in full gate')
    # Gate tooling changes require complete coverage, independently of graph edges.
    diff = subprocess.run(['git', 'diff', '--name-only', args.base, args.head],
                          text=True, capture_output=True, check=True).stdout.splitlines()
    if any(path.startswith(('tools/ci/', '.github/')) for path in diff):
        files = population(root)
    durations = json.loads((root / 'tools/ci/gate_durations.json').read_text())['lanes']
    count = args.shards
    if args.mode == 'selection':
        count = max(selection_shard_count(files, durations.get(lane, {}), args.shards,
                        workers=4, single_shard_seconds=120) for lane in ('bytes', 'torch'))
    runners = args.runners.split(',')
    if int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')) % 2 == 0:
        runners.reverse()
    matrix = lane_matrix(files, durations, count, runners) if files else {'shards': 0, 'include': []}
    # A no-tests PR has a complete empty plan, and launches no pytest shard.
    plan = {'schema': 'tessera.gate_plan.v1', 'base_sha': args.base, 'head_sha': args.head,
            'mode': args.mode, 'selection': selection,
            'population': {'version': hashlib.sha256('\n'.join(population(root)).encode()).hexdigest(),
                           'cuda': 'not covered: CPU wheels, CUDA_VISIBLE_DEVICES empty'},
            'selected': [{'lane': lane, 'path': path, 'estimated_seconds': durations.get(lane, {}).get(path)}
                         for lane in ('bytes', 'torch') for path in files],
            'deselect': quarantine(json.loads((root / 'tools/ci/gate_quarantine.json').read_text()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + '\n')
    args.matrix.write_text(json.dumps(matrix, indent=2) + '\n')
    print('has_tests=' + str(bool(files)).lower())
    print('matrix=' + json.dumps({'include': matrix['include'] or [{'index': 0}]}, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
