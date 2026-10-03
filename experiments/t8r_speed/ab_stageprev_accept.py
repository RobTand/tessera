"""#793 numeric/timing acceptance over an explicit frozen case/rate/source manifest.

AB_EXPECTED_CASES names the manifest. An observed union is not coverage.
Historical measurement banks are never rewritten by this current owner.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def case_key(row):
    family, group, raw_m = row['family'], row['group'], row['M']
    if family not in ('routed', 'dense') or not isinstance(group, str) or not group:
        raise ValueError('invalid case family/group')
    if isinstance(raw_m, bool) or not isinstance(raw_m, (str, int)):
        raise ValueError('invalid M key')
    m = str(raw_m)
    if not re.fullmatch(r'[1-9][0-9]*(?:@[A-Za-z0-9_.-]+)?', m):
        raise ValueError(f'invalid M key: {m}')
    return family, group, m


def expected_population(expected):
    if expected.get('schema') != 'tessera.stageprev.expected.v1':
        raise ValueError('expected manifest schema differs')
    if expected.get('phase') not in ('numeric', 'timing'):
        raise ValueError('expected phase must be numeric or timing')
    arms = expected['arms']
    if (not isinstance(arms, list) or len(arms) != 2 or len(set(arms)) != 2
            or any(not isinstance(a, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', a) for a in arms)):
        raise ValueError('expected manifest requires two unique named arms')
    hashes = expected['kernel_sha']
    if not hashes or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                         for v in hashes.values()):
        raise ValueError('expected manifest lacks exact source hashes')
    cases = {}
    for row in expected['cases']:
        key = case_key(row)
        if key in cases:
            raise ValueError(f'duplicate expected case: {key}')
        rates = row['rates']
        if (not isinstance(rates, list) or len(rates) not in (1, 2)
                or any(type(r) is not int or not 1 <= r <= 8 for r in rates)
                or len(set(rates)) != len(rates)):
            raise ValueError(f'invalid declared rates: {key}')
        cases[key] = rates
    if not cases:
        raise ValueError('expected population is empty')
    required_hashes = {f'{arm}-{family}{suffix}' for family, _, _ in cases
                       for arm in arms for suffix in ('', 'b')}
    if set(hashes) != required_hashes:
        raise ValueError('expected source population differs from case families/arms')
    return cases


def accept(summary, expected, *, old=None, tol_same=0.02, slack=0.01):
    cases = expected_population(expected)
    if summary['arms'] != expected['arms'] or summary['ref'] != expected['arms'][0]:
        raise ValueError('observed arm population differs')
    if summary['kernel_sha'] != expected['kernel_sha']:
        raise ValueError('observed source hashes differ')
    observed = {}
    for row in summary['rows']:
        key = case_key(row)
        if key in observed:
            raise ValueError(f'duplicate observed case: {key}')
        observed[key] = row
    if set(observed) != set(cases):
        raise ValueError(f'case population differs: missing={sorted(set(cases)-set(observed))}, '
                         f'extra={sorted(set(observed)-set(cases))}')
    arm = expected['arms'][1]
    restoration = {}
    if old is not None:
        if len(old['arms']) != 2 or old['ref'] != old['arms'][0]:
            raise ValueError('old comparison has ambiguous arms')
        old_arm = old['arms'][1]
        for row in old['rows']:
            key = case_key(row)
            if key in restoration:
                raise ValueError(f'duplicate old case: {key}')
            values = [row.get(f'{old_arm}{suffix}_ratio') for suffix in ('', 'b')]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f'old comparison lacks finite ratios: {key}')
            restoration[key] = sum(values) / 2
        if not set(cases) <= set(restoration):
            raise ValueError('old comparison is missing declared cases')
    lines, failures = [], []
    for key, rates in cases.items():
        row = observed[key]
        why = []
        if row.get('bitwise') is not True or row.get('missing') != []:
            why.append('not bitwise or missing arm output')
        if expected.get('require_intermediates') and row.get('role_words_equal') is not True:
            why.append('intermediate role words not qualified')
        values = [row.get(f'{arm}{suffix}_ratio') for suffix in ('', 'b')]
        if expected['phase'] == 'timing':
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                why.append('finite positive timing ratios missing')
            elif len(rates) == 1:
                if any(abs(v - 1) > tol_same for v in values):
                    why.append('single-rate changed beyond declared tolerance')
            elif old is not None:
                if abs(sum(values) / 2 * restoration[key] - 1) > tol_same:
                    why.append('two-run does not restore pre-763 time')
            elif key[0] == 'routed' and int(key[2].split('@')[0]) == 2048:
                if any(v > 0.94 + slack for v in values):
                    why.append('two-run routed M2048 above original acceptance bar')
            elif key[0] == 'dense' or int(key[2].split('@')[0]) == 1:
                if any(v > 0.98 + slack for v in values):
                    why.append('two-run above original acceptance bar')
        tag = f'{key[0]} {key[1]} M={key[2]} rates={rates}'
        lines.append(('FAIL ' if why else 'ok ') + tag + (' <- ' + '; '.join(why) if why else ''))
        if why:
            failures.append(tag)
    return {'phase': expected['phase'], 'cases': len(cases), 'failures': failures, 'lines': lines}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ab_dir')
    parser.add_argument('--old')
    parser.add_argument('--tol-same', type=float, default=0.02)
    parser.add_argument('--slack', type=float, default=0.01)
    args = parser.parse_args()
    try:
        expected_path = os.environ.get('AB_EXPECTED_CASES')
        if not expected_path:
            raise ValueError('AB_EXPECTED_CASES is required')
        if (not math.isfinite(args.tol_same) or args.tol_same < 0
                or not math.isfinite(args.slack) or args.slack < 0):
            raise ValueError('timing tolerances must be finite and nonnegative')
        result = accept(load(Path(args.ab_dir) / 'ab_summary.json'), load(expected_path),
                        old=load(Path(args.old) / 'ab_summary.json') if args.old else None,
                        tol_same=args.tol_same, slack=args.slack)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2
    print('\n'.join(result['lines']))
    print(f"{result['phase']}: {result['cases']} declared cases, {len(result['failures'])} failing")
    return 1 if result['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
