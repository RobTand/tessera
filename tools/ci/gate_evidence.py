#!/usr/bin/env python3
"""Summarize retained JUnit repeats without treating skips as passes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def cases(path):
    for case in ET.parse(path).iter('testcase'):
        filename = case.get('file', '').removeprefix('./')
        classname, name = case.get('classname', ''), case.get('name', '')
        if not filename:
            module = classname.split('.Test', 1)[0]
            filename = module.replace('.', '/') + '.py' if module else name
        if not filename.startswith('tests/') or not filename.endswith('.py'):
            continue
        module = filename.removesuffix('.py').replace('/', '.')
        classes = classname.removeprefix(module).strip('.').replace('.', '::')
        nodeid = filename + ('::' + classes if classes else '') + ('::' + name if classname else '')
        status = ('failed' if case.find('failure') is not None or case.find('error') is not None
                  else 'skipped' if case.find('skipped') is not None else 'passed')
        yield {'nodeid': nodeid, 'path': filename, 'status': status,
               'seconds': float(case.get('time', 0))}


def analyze_runs(runs):
    observed = defaultdict(lambda: defaultdict(list))
    file_times = defaultdict(lambda: defaultdict(list))
    largest = defaultdict(dict)
    results = []
    for run in runs:
        path = Path(run['junit'])
        totals = defaultdict(float)
        counts = defaultdict(int)
        for row in cases(path):
            observed[run['lane']][row['nodeid']].append(row['status'])
            totals[row['path']] += row['seconds']
            counts[row['status']] += 1
            largest[run['lane']][row['nodeid']] = max(largest[run['lane']].get(row['nodeid'], 0), row['seconds'])
        for name, value in totals.items():
            file_times[run['lane']][name].append(value)
        result = {**run, 'counts': dict(counts), 'junit_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        surface = run.get('surface')
        if surface:
            result['population'] = json.loads(Path(surface).read_text())
        results.append(result)
    lanes = {}
    for lane, nodes in sorted(observed.items()):
        lanes[lane] = {
            'files': {path: statistics.median(times) for path, times in sorted(file_times[lane].items())},
            'always_failing': sorted(node for node, statuses in nodes.items() if all(s == 'failed' for s in statuses)),
            'flaky': sorted(node for node, statuses in nodes.items() if 'failed' in statuses and 'passed' in statuses),
            'largest_test_seconds': dict(sorted(largest[lane].items())),
        }
    return {'schema': 'tessera.gate_evidence.v1', 'runs': results, 'lanes': lanes}



def _elapsed(start, end):
    if not start or not end:
        return None
    return (datetime.fromisoformat(end.replace('Z', '+00:00')) -
            datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()


def analyze_canaries(runs):
    """Measure through the final gate; failed runs never qualify as passes."""
    measured, passing = [], []
    for entry in runs:
        run, jobs = entry['run'], entry['jobs']
        gate = next((job for job in jobs if job['name'] == 'gate'), {})
        wall = _elapsed(run['created_at'], gate.get('completed_at'))
        passed = (run['status'] == 'completed' and run.get('conclusion') == 'success'
                  and gate.get('conclusion') == 'success' and wall is not None)
        if passed:
            passing.append(wall)
        shards = []
        for job in jobs:
            if not job['name'].startswith('shard-'):
                continue
            steps = {step['name']: _elapsed(step.get('started_at'), step.get('completed_at'))
                     for step in job.get('steps', [])}
            total = _elapsed(job.get('started_at'), job.get('completed_at'))
            dependency = steps.get('Install pinned lane dependencies')
            tests = steps.get('Run assigned population')
            other = (total - dependency - tests
                     if all(value is not None for value in (total, dependency, tests)) else None)
            shards.append({'name': job['name'], 'runners': job.get('labels', []),
                           'conclusion': job.get('conclusion'), 'job_seconds': total,
                           'dependency_setup_seconds': dependency,
                           'test_step_seconds': tests, 'other_seconds': other})
        measured.append({'run_id': run['id'], 'attempt': run.get('run_attempt'),
                         'head_sha': run.get('head_sha'), 'status': run['status'],
                         'conclusion': run.get('conclusion'), 'wall_seconds': wall,
                         'passed': passed, 'shards': shards})
    passing.sort()
    return {'runs': measured, 'passing_runs': len(passing),
            'p50_seconds': statistics.median(passing) if passing else None,
            'p90_seconds': passing[math.ceil(.9 * len(passing)) - 1] if passing else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--canary-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--durations-output', type=Path)
    parser.add_argument('--budget-output', type=Path)
    args = parser.parse_args()
    if not args.manifest and not args.canary_manifest:
        parser.error('provide --manifest or --canary-manifest')
    runs = json.loads(args.manifest.read_text()) if args.manifest else []
    report = analyze_runs(runs)
    if args.canary_manifest:
        report['canaries'] = analyze_canaries(json.loads(args.canary_manifest.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if args.durations_output:
        data = {'lanes': {lane: value['files'] for lane, value in report['lanes'].items()},
                'source': [{'lane': row['lane'], 'repeat': row['repeat'], 'action_key': row.get('action_key'),
                            'junit_sha256': row['junit_sha256']} for row in report['runs']]}
        args.durations_output.write_text(json.dumps(data, indent=2) + '\n')
    if args.budget_output:
        maximum = {}
        for lane in report['lanes'].values():
            for name, seconds in lane['largest_test_seconds'].items():
                maximum[name] = max(maximum.get(name, 0), seconds)
        args.budget_output.write_text(json.dumps({'seconds': 20, 'enforce': True,
            'slow': {name: math.ceil(2 * seconds) for name, seconds in sorted(maximum.items()) if seconds > 20},
            'source': [{'run': row.get('run_id'), 'artifact': row.get('artifact'),
                        'junit_sha256': row['junit_sha256']} for row in report['runs']]}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
