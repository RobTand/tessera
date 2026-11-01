#!/usr/bin/env python3
"""Run any CI-assigned gate files. Exit 0 pass, 1 test failure, 75 no verdict."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

NO_VERDICT = 75
BUDGET_ENFORCEMENT_TOLERANCE = 1.5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_junit import test_name


class BoundaryFailure(ValueError):
    """A boundary proof failed with a final verdict."""


class BudgetFailure(ValueError):
    """A definite policy failure, never an infrastructure retry."""


def _budget_limits(config):
    if not isinstance(config, dict) or not isinstance(config.get('slow'), dict):
        raise BudgetFailure('Test budget requires seconds and a slow map')
    limits = [config.get('seconds'), *config['slow'].values()]
    if (any(isinstance(limit, bool) or not isinstance(limit, (int, float)) or
            not math.isfinite(limit) or limit <= 0 for limit in limits) or
            any(not isinstance(name, str) or not name for name in config['slow'])):
        raise BudgetFailure('Test budget limits must be finite positive seconds')
    return config


def _budget_policy(root, base_sha):
    path = 'tools/ci/gate_budget.json'
    current = _budget_limits(json.loads((root / path).read_text()))
    baseline = current
    if base_sha:
        # Check the commit separately: an unavailable target is not a bootstrap.
        target = subprocess.run(['git', 'cat-file', '-e', f'{base_sha}^{{commit}}'],
                                cwd=root, capture_output=True, check=False)
        if target.returncode:
            raise RuntimeError('budget target commit is unavailable: ' + base_sha)
        tree = subprocess.run(['git', 'ls-tree', base_sha, '--', path],
                              cwd=root, text=True, capture_output=True, check=True)
        if tree.stdout.strip():
            blob = subprocess.run(['git', 'show', f'{base_sha}:{path}'], cwd=root,
                                  text=True, capture_output=True, check=True)
            baseline = _budget_limits(json.loads(blob.stdout))
    if baseline.get('enforce', False) and not current.get('enforce', False):
        raise BudgetFailure('Enforced test budgets may not be disabled')
    if (current['seconds'] > baseline['seconds'] or
            not current['slow'].keys() <= baseline['slow'].keys() or
            any(limit > baseline['slow'][name] for name, limit in current['slow'].items())):
        raise BudgetFailure('Test budgets and the explicit slow list may not grow versus the target')
    return current


def _budget_timings(junit, files, config, require_all=True):
    observed, seconds = set(), {}
    for case in ET.parse(junit).iter('testcase'):
        filename, name = case.get('file', '').removeprefix('./'), case.get('name', '')
        name = test_name(name)
        classname = case.get('classname', '')
        if not classname and case.find('skipped') is not None and (name in files or filename in files):
            continue  # A module-level importorskip is not a test timing.
        if filename not in files or not name:
            raise ValueError('JUnit timing has missing or unexpected test identity')
        module = filename.removesuffix('.py').replace('/', '.')
        if classname != module and not classname.startswith(module + '.'):
            raise ValueError('JUnit timing cannot resolve test class: ' + classname)
        classes = classname[len(module):].lstrip('.').replace('.', '::')
        nodeid = filename + '::' + (classes + '::' if classes else '') + name
        try:
            elapsed = float(case.attrib['time'])
        except (KeyError, ValueError) as error:
            raise ValueError('JUnit timing missing or invalid for ' + nodeid) from error
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('JUnit timing must be finite nonnegative seconds: ' + nodeid)
        observed.add(filename)
        seconds[nodeid] = seconds.get(nodeid, 0) + elapsed
    if require_all and observed != set(files):
        raise ValueError('JUnit timings do not cover every assigned file')
    return [{'nodeid': nodeid, 'seconds': elapsed,
             'budget_seconds': config['slow'].get(nodeid, config['seconds'])}
            for nodeid, elapsed in sorted(seconds.items())
            if elapsed > config['slow'].get(nodeid, config['seconds'])]


def _write_budget_warnings(junit, warnings):
    messages = [
        f"BUDGET_WARNING: {row['nodeid']}: {row['seconds']:.3f}s > "
        f"{row['budget_seconds']:.3f}s policy; within "
        f"{BUDGET_ENFORCEMENT_TOLERANCE:g}x tolerance "
        f"({row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE:.3f}s)"
        for row in warnings]
    text = '\n'.join(messages)
    print(text)
    tree = ET.parse(junit)
    suite = next(tree.iter('testsuite'), tree.getroot())
    output = suite.find('system-out')
    if output is None:
        output = ET.SubElement(suite, 'system-out')
    output.text = (output.text + '\n' if output.text else '') + text
    tree.write(junit, encoding='utf-8', xml_declaration=True)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a') as handle:
            handle.write('\n## Per-test budget warnings\n\n' +
                         f'{len(messages)} warnings; complete list in junit.xml and result.json.\n\n' +
                         '\n'.join('- ' + message[:1000] for message in messages[:25]) + '\n' +
                         (f'{len(messages) - 25} more warnings in junit.xml.\n' if len(messages) > 25 else ''))


def cpu_population(path):
    surface = json.loads(path.read_text())
    if surface.get('role') != 'population':
        raise ValueError('CPU population is missing its controller report')
    if surface.get('cuda') is not False or surface.get('cuda_surface', {}).get('executed') != 0:
        raise ValueError('CPU lane unexpectedly exercised CUDA')
    if not all(key in surface for key in ('counts', 'not_collected', 'skip_reasons', 'device')):
        raise ValueError('CPU population is incomplete')
    return surface


def collection_coverage(directory, files, surface, workers=1):
    uncollected = {path if path.startswith('tests/') else 'tests/' + path
                   for path in surface['not_collected']}
    observed = set()
    for index in range(workers):
        path = directory / f'collection.gw{index}.json'
        receipt = json.loads(path.read_text())
        collected = set(receipt['collected'])
        missing = set(files) - collected - uncollected
        if missing:
            raise ValueError('unaccounted assigned test files: ' + ', '.join(sorted(missing)))
        observed.update(collected)
    return {'status': 'pass', 'worker_receipts': workers, 'collected': sorted(observed),
            'uncollected': sorted(set(files) & uncollected)}


def run_plan(plan: dict, *, root: Path, output: Path, junit: Path,
             files: list[str] | None = None, workers: int | None = None, lane: str = 'torch',
             execute=subprocess.run, affinity=None) -> int:
    print('GATE_PREFLIGHT_START', flush=True)
    record = {'schema': 'prismabuild.gate_result.v1',
              'population_version': plan.get('population', {}).get('version'), 'lane': lane,
              'mode': plan.get('mode'), 'base_sha': plan.get('base_sha'), 'head_sha': plan.get('head_sha'), 'files': [], 'verdict': None, 'pytest_exit_code': None}
    try:
        eligible = {row['path'] for row in plan['selected'] if row['lane'] == lane}
        files = sorted(eligible) if files is None else files
        if not files:
            raise ValueError('empty gate assignment cannot run pytest')
        if len(files) != len(set(files)) or set(files) - eligible:
            raise ValueError('assignment contains duplicate files or files outside the plan')
        costs = {row['path']: row.get('estimated_seconds') for row in plan['selected'] if row['lane'] == lane}
        known = sorted(value for value in costs.values() if value is not None)
        unknown = (known[math.ceil(len(known) * .9) - 1] or 1.0) if known else 1.0
        files = sorted(files, key=lambda path: (-(costs[path] if costs[path] is not None else unknown), path))
        record['files'] = files
        config = _budget_policy(root, plan.get('base_sha'))
        record['budget'] = {'status': 'pending', 'baseline_sha': plan.get('base_sha')}
        cpus = os.sched_getaffinity(0) if affinity is None else affinity
        workers = len(cpus) if workers is None else workers
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or not cpus:
            raise ValueError('job needs a positive worker count and assigned CPUs')
        workers = min(workers, len(cpus))
        record['workers'] = workers
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        # setup-python exports loader overrides; proof capture must see the actual
        # installed interpreter rather than an ambient dynamic-loader override.
        environment = {name: value for name, value in environment.items() if not name.startswith('LD_')}
        environment['GATE_DESELECT'] = json.dumps([target for target in plan.get('deselect', {}).get(lane, [])
                                                  if target.split('::', 1)[0] in files])
        environment['GATE_ASSIGNED_FILES'] = json.dumps(files)
        environment['GATE_COLLECTION_DIR'] = str(output.parent.resolve())
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + str(root / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
        junit.parent.mkdir(parents=True, exist_ok=True)
        profile = ['-X', 'importtime'] if os.environ.get('GATE_PROFILE_STARTUP') == '1' else []
        command = [sys.executable, *profile, '-m', 'pytest', '-n', str(workers), '--dist=loadgroup', '--no-loadscope-reorder',
                   '-q', '-ra', '--durations=30', '--timeout=300', '-p', 'gate_collect', '--surface-json=' + str(output.with_name('surface.json')), '--rootdir=' + str(root), '--junitxml=' + str(junit), '-o', 'junit_family=xunit1',
                   str(root)]
        if lane == 'bytes':
            record['boundary'] = {'status': 'pending'}
            proof = execute([sys.executable, 'tools/ci/gate_boundary.py', '--base', plan['base_sha']],
                            cwd=root, env=environment, check=False)
            if proof.returncode == 1:
                record['boundary']['status'] = 'fail'
                raise BoundaryFailure('no-torch boundary proof failed')
            if proof.returncode != 0:
                raise RuntimeError('boundary proof ended without a verdict')
            record['boundary']['status'] = 'pass'
        print('GATE_PYTEST_START', flush=True)
        result = execute(command, cwd=root, env=environment, check=False)
        print('GATE_PYTEST_END', flush=True)
        record['pytest_exit_code'] = result.returncode
        record['verdict'] = 'fail' if result.returncode == 1 else None
        try:
            # Preserve a recorded failure even if pytest subsequently interrupts
            # or leaves a truncated report during an internal error.
            for _, element in ET.iterparse(junit):
                if element.tag in {'failure', 'error'}:
                    record['verdict'] = 'fail'
        except (OSError, ET.ParseError) as error:
            if record['verdict'] != 'fail':
                raise RuntimeError('JUnit verdict unavailable: ' + str(error)) from error
        if record['verdict'] != 'fail':
            if result.returncode not in (0, 5):
                raise RuntimeError(f'pytest exited {result.returncode}, without a complete test verdict')
            record['verdict'] = 'pass'
        try:
            record['surface'] = cpu_population(output.with_name('surface.json'))
            if lane == 'torch' and (record['surface']['not_collected'] or
                                    not record['surface']['device'].startswith('torch ')):
                raise ValueError('torch lane requires torch and zero uncollected modules')
            record['collection'] = collection_coverage(output.parent, files, record['surface'], workers)
            costs = _budget_timings(junit, files, config, require_all=False)
            overruns, warnings = [], []
            for row in costs:
                target = (overruns if row['seconds'] >
                          row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE else warnings)
                target.append(row)
            enforcing = config.get('enforce', False)
            record['budget'].update(status='fail' if overruns and enforcing else 'pass', enforce=enforcing,
                                    overruns=overruns, warnings=warnings)
            if overruns:
                if enforcing:
                    record['verdict'] = 'fail'
                print('Per-test time budget exceeded:')
                for row in overruns:
                    limit = row['budget_seconds'] * BUDGET_ENFORCEMENT_TOLERANCE
                    print(f"{row['nodeid']}: {row['seconds']:.3f}s > {limit:.3f}s "
                          f"(policy {row['budget_seconds']:.3f}s x {BUDGET_ENFORCEMENT_TOLERANCE:g})")
            if warnings:
                _write_budget_warnings(junit, warnings)
        except (OSError, ValueError, ET.ParseError) as error:
            record['budget'].update(status='unavailable', reason=str(error))
            # A definite test failure remains final even if timing evidence is lost.
            if record['verdict'] != 'fail':
                record['verdict'] = None
                raise RuntimeError(str(error)) from error
        code = 0 if record['verdict'] == 'pass' else 1
    except BoundaryFailure as error:
        record['verdict'] = 'fail'
        record['boundary'] = {'status': 'fail', 'reason': str(error)}
        code = 1
    except BudgetFailure as error:
        record['verdict'] = 'fail'
        record['budget'] = {'status': 'fail', 'reason': str(error), 'baseline_sha': plan.get('base_sha')}
        print('GATE_BUDGET_FAILURE: ' + str(error))
        code = 1
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        record['infrastructure_reason'] = str(error)
        code = NO_VERDICT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print({0: 'GATE_PASS', 1: 'GATE_TEST_FAILURE', 75: 'NO_VERDICT'}[code], record.get('infrastructure_reason', ''), flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--junitxml', type=Path, required=True)
    parser.add_argument('--lane', choices=['bytes', 'torch'], required=True)
    parser.add_argument('--workers', type=int, help='resource-class CPU allocation; clamped to inherited affinity')
    assigned = parser.add_mutually_exclusive_group()
    assigned.add_argument('--files-from', type=Path, help='one assigned repository-relative file per line')
    assigned.add_argument('--files', nargs='+', help='explicit assigned test files')
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        files = ([line.strip() for line in args.files_from.read_text().splitlines() if line.strip()]
                 if args.files_from else args.files)
        return run_plan(plan, root=args.root, output=args.result, junit=args.junitxml,
                        files=files, workers=args.workers, lane=args.lane)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'gate: no verdict: {error}', file=sys.stderr)
        return NO_VERDICT


if __name__ == '__main__':
    raise SystemExit(main())
