#!/usr/bin/env python3
"""Evaluate hosted gate attempt records without retrying test failures."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys


def _verdict(plan, files, record, lane):
    if not isinstance(record, dict) or record.get('lane') != lane:
        return None
    if any(record.get(field) != plan.get(field) for field in ('base_sha', 'head_sha', 'mode')):
        return None
    if record.get('population_version') != plan['population']['version']:
        return None
    if (not isinstance(record.get('files'), list) or
            any(not isinstance(path, str) for path in record['files']) or
            sorted(record['files']) != sorted(files)):
        return None
    if record.get('verdict') == 'fail' and record.get('boundary', {}).get('status') == 'fail':
        return 'fail'
    code = record.get('pytest_exit_code')
    budget = record.get('budget')
    if (record.get('verdict') == 'fail' and isinstance(budget, dict) and
            budget.get('status') == 'fail' and (code is None or type(code) is int and code in {0, 1})):
        return 'fail'
    if type(code) is not int:
        return None
    verdict = record.get('verdict')
    if not isinstance(verdict, str):
        return None
    if verdict == 'fail':
        return 'fail'
    if verdict == 'pass' and code in (0, 5) and isinstance(budget, dict) and budget.get('status') == 'pass':
        surface = record.get('surface', {})
        boundary_ok = lane != 'bytes' or record.get('boundary', {}).get('status') == 'pass'
        if (surface.get('role') == 'population' and surface.get('cuda') is False and
                surface.get('cuda_surface', {}).get('executed') == 0 and boundary_ok and
                record.get('collection', {}).get('status') == 'pass'):
            return 'pass'
    return None


def assess_attempts(plan, assignments, attempts):
    """Return retry indices and the final verdict, preserving any test failure."""
    indices = [row['index'] for row in assignments]
    assigned = Counter((row['lane'], path) for row in assignments for path in row['files'])
    selected = [(row['lane'], row['path']) for row in plan['selected']]
    if (len(indices) != len(set(indices)) or len(selected) != len(set(selected)) or
            set(assigned) != set(selected) or any(count != 1 for count in assigned.values())):
        raise ValueError('assignments must cover every selected file exactly once, with unique indices')
    retry, failed, missing, effective = [], [], [], {}
    for row in assignments:
        index, files = row['index'], row['files']
        first = _verdict(plan, files, attempts.get(1, {}).get(index), row['lane'])
        second = _verdict(plan, files, attempts.get(2, {}).get(index), row['lane'])
        if first is None:
            retry.append(index)
        if 'fail' in (first, second):
            verdict, attempt = 'fail', 1 if first == 'fail' else 2
        elif first is None:
            verdict, attempt = second, 2
        else:
            verdict, attempt = first, 1
        if verdict is None:
            missing.append(index)
        else:
            effective[index] = attempt
            if verdict == 'fail':
                failed.append(index)
    return {'schema': 'prismabuild.gate_aggregate.v1', 'retry_indices': sorted(retry),
            'failed_indices': sorted(failed), 'missing_indices': sorted(missing),
            'effective_attempts': effective, 'exit_code': 1 if failed else 75 if missing else 0}


def _read_result(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['retries', 'finish'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--prefix', default='gate-result')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--effective-output', type=Path)
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        matrix = json.loads(args.matrix.read_text())
        assignments = matrix['include']
        attempts = {attempt: {row['index']: _read_result(args.artifacts /
                    f"{args.prefix}-a{attempt}-{row['index']}" / 'result.json') for row in assignments}
                    for attempt in (1, 2)}
        result = assess_attempts(plan, assignments, attempts)
        if args.command == 'retries':
            include = [row for row in matrix['include'] if row['index'] in result['retry_indices']]
            # Keep a valid nonempty expression even when the job's `if` skips the matrix.
            output = {'include': include or matrix['include'][:1]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output) + '\n')
            print('retry=' + str(bool(include)).lower())
            print('matrix=' + json.dumps(output, separators=(',', ':')))
            return 0
        if args.effective_output:
            for index, attempt in result['effective_attempts'].items():
                source = args.artifacts / f'{args.prefix}-a{attempt}-{index}'
                target = args.effective_output / str(index)
                target.mkdir(parents=True, exist_ok=True)
                for name in ('result.json', 'junit.xml', 'surface.json'):
                    if (source / name).is_file():
                        shutil.copyfile(source / name, target / name)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        marker = {0: 'GATE_PASS', 1: 'GATE_TEST_FAILURE', 75: 'NO_VERDICT'}[result['exit_code']]
        message = f"{marker}: failed shards {result['failed_indices']}; missing verdicts {result['missing_indices']}"
        print(message)
        if args.summary:
            with args.summary.open('a') as handle:
                handle.write('## Gate verdict\n\n' + message + '\n')
        return result['exit_code']
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('NO_VERDICT: ' + str(error), file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
