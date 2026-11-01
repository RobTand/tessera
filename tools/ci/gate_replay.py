#!/usr/bin/env python3
"""Replay merged changes against full JUnit outcomes; execute only in admitted jobs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcomes(junit, surface):
    result = defaultdict(dict)
    for case in ET.parse(junit).iter('testcase'):
        path = case.get('file')
        if not path:
            path = case.get('classname', '').split('::')[0].replace('.', '/') + '.py'
        if not path.startswith('tests/'):
            path = 'tests/' + path
        state = next((state for state in ('failure', 'error', 'skipped') if case.find(state) is not None), 'passed')
        result[path][case.get('classname', '') + '::' + case.get('name', '')] = state
    for path in json.loads(Path(surface).read_text())['not_collected']:
        result[path if path.startswith('tests/') else 'tests/' + path]['<module>'] = 'not_collected'
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before', default='75c2aa03c8b3352c313c14fc4cd1a7c1c8c76c56')
    parser.add_argument('--run-full', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = Path.cwd()
    args.output.mkdir(parents=True, exist_ok=True)
    selector = load(root / 'tools/impacted_tests.py', 'new_selector')
    gate = load(root / 'tools/ci/gate_select.py', 'gate_selector')
    old = type(sys)('old_selector')
    old.__file__ = str(root / 'tools/impacted_tests.py')
    exec(compile(subprocess.check_output(['git', 'show', args.before + ':tools/impacted_tests.py'], text=True),
                 old.__file__, 'exec'), old.__dict__)
    rows, full = [], {}
    with tempfile.TemporaryDirectory(prefix='tessera-replay-') as temporary:
        temporary = Path(temporary)
        for spec in json.loads(args.spec.read_text()):
            head = spec['merge']
            base = subprocess.check_output(['git', 'rev-parse', head + '^1'], text=True).strip()
            checkout = temporary / head
            subprocess.run(['git', 'worktree', 'add', '--detach', str(checkout), head], check=True, capture_output=True)
            changed = subprocess.check_output(['git', 'diff', '--name-only', '--no-renames', base, head], text=True).splitlines()
            before = old.select(checkout, changed)
            after = selector.select(checkout, changed)
            population = gate.population(checkout)
            if gate.documentation_only(changed):
                after = {'verdict': 'narrowed', 'tests': sorted(set(population) & gate.FAST_BOUNDARY)}
            if any(path.startswith(('tools/ci/', '.github/')) or path == 'tools/impacted_tests.py' for path in changed):
                after = {'verdict': 'full', 'tests': population}
            row = {**spec, 'base': base, 'changed': changed, 'population': len(population),
                   'before': len(population) if before['verdict'] == 'full' else len(before['tests']),
                   'after': len(population) if after['verdict'] == 'full' else len(after['tests']),
                   'verdict': after['verdict'], 'selected': after['tests']}
            if args.run_full:
                for commit in (base, head):
                    if commit in full:
                        continue
                    tree = checkout if commit == head else temporary / commit
                    if tree != checkout:
                        subprocess.run(['git', 'worktree', 'add', '--detach', str(tree), commit], check=True, capture_output=True)
                    destination = args.output / commit
                    destination.mkdir(exist_ok=True)
                    command = [sys.executable, '-m', 'pytest', '-n', str(args.workers), '--dist=loadfile',
                               '-q', '-ra', '--timeout=60', '--durations=20', '-o', 'junit_family=xunit1',
                               '--junitxml=' + str(destination / 'junit.xml'),
                               '--surface-json=' + str(destination / 'surface.json'), 'tests']
                    env = dict(os.environ, PYTHONPATH=str(tree / 'src'), CUDA_VISIBLE_DEVICES='',
                               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                    with (destination / 'pytest.log').open('w') as log:
                        run = subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if run.returncode not in (0, 1):
                        raise RuntimeError(f'full replay {commit} has no verdict: {run.returncode}')
                    full[commit] = outcomes(destination / 'junit.xml', destination / 'surface.json')
                    (destination / 'outcomes.json').write_text(json.dumps(full[commit], indent=2))
                    print('FULL', commit, run.returncode, flush=True)
                changed_outcomes = sorted(path for path in full[base].keys() | full[head].keys()
                                          if full[base].get(path) != full[head].get(path))
                missing = [] if after['verdict'] == 'full' else sorted(set(changed_outcomes) - set(after['tests']))
                row.update(changed_outcome_files=changed_outcomes, missing_outcome_files=missing)
            rows.append(row)
            (args.output / 'report.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        summary = {arm: {'p50': statistics.median(row[arm] for row in rows),
                         'p90': sorted(row[arm] for row in rows)[__import__('math').ceil(.9 * len(rows)) - 1]}
                   for arm in ('before', 'after')}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    return int(any(row.get('missing_outcome_files') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
