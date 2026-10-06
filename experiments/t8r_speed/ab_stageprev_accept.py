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
    if expected["phase"] == "numeric" and expected.get("require_intermediates") is not True:
        raise ValueError("numeric phase requires actual intermediate words/native evidence")
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
    if expected.get("require_intermediates"):
        natives = expected.get("native_files", {})
        if set(natives) != set(arms):
            raise ValueError("declared native arm population differs")
        for arm, native in natives.items():
            if (not isinstance(native.get("path"), str) or not native["path"].startswith("/mnt/shared/")
                    or not native["path"].endswith(".so")):
                raise ValueError("declared native path differs")
            for name in ("sha256", "source_sha256", "build_action_key", "build_receipt_sha256"):
                if not isinstance(native.get(name), str) or not re.fullmatch("[0-9a-f]{64}", native[name]):
                    raise ValueError("declared native/source/build identity is incomplete")
            if any(v != native["source_sha256"] for k, v in hashes.items() if k.startswith(arm + "-")):
                raise ValueError("native build source differs from declared arm source")
    return cases


def validate_native_record(record, native):
    if record.get("declared_path") != native["path"] or record.get("source_sha256") != native["source_sha256"]:
        raise ValueError("observed native source/path differs from reviewed build identity")
    for name in ("expected_sha256", "before_load_sha256", "after_load_sha256", "mapped_sha256", "after_profile_sha256"):
        if record.get(name) != native["sha256"]:
            raise ValueError("observed native digest differs or is incomplete")
    if (not record.get("executable_mappings") or not record.get("pin_id") or not record.get("ref_id")
            or record.get("final_fence_complete") is not True
            or record.get("load_fd_closed_after_fence") is not True):
        raise ValueError("native mapping/final-fence/lease boundary evidence is incomplete")


def observed_rates(tables):
    roles = ("gate_proj", "up_proj", "down_proj")
    if set(tables) != set(roles):
        raise ValueError("observed projection run-table population differs")
    populations = {}
    for role in roles:
        if len(tables[role]) != 288:
            raise ValueError("observed expert run-table population differs")
        rates = set()
        for pair in tables[role]:
            if (len(pair) != 8 or any(type(v) is not int for v in pair)
                    or not 1 <= pair[0] <= 8 or pair[2] <= 0 or pair[1] != 0 or pair[3] != 0
                    or pair[5] != pair[2] or pair[6] < 0 or pair[7] != 16*pair[2]*pair[0]
                    or (pair[6] == 0 and pair[4] != 0)
                    or (pair[6] > 0 and (pair[4] != pair[0]+1 or pair[4] > 8))):
                raise ValueError("observed run pair differs from production grammar")
            rates.add(pair[0])
            if pair[6]: rates.add(pair[4])
        populations[role] = sorted(rates)
    return populations


def accept(summary, expected, *, old=None, tol_same=0.02, slack=0.01):
    cases = expected_population(expected)
    if summary.get("phase") != expected["phase"]:
        raise ValueError("observed phase differs; numeric evidence cannot promote timing")
    if summary['arms'] != expected['arms'] or summary['ref'] != expected['arms'][0]:
        raise ValueError('observed arm population differs')
    if summary['kernel_sha'] != expected['kernel_sha']:
        raise ValueError('observed source hashes differ')
    if expected.get("require_intermediates"):
        if summary.get("native_build_identity") != expected["native_files"]:
            raise ValueError("observed reviewed native build identity differs")
        if set(summary.get("native_code_artifact", {})) != set(expected["kernel_sha"]):
            raise ValueError("observed native artifact population differs")
        for key, record in summary["native_code_artifact"].items():
            arm_name = next(a for a in expected["arms"] if key.startswith(a + "-"))
            validate_native_record(record, expected["native_files"][arm_name])
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
        if expected.get("require_intermediates"):
            actual_rates = observed_rates(row["numeric_signature"]["run_tables"])
            if row.get("observed_rates") != actual_rates:
                why.append("reported projection rates differ from actual run tables")
        if expected.get("require_intermediates") and row.get("observed_rates") != {
                role: sorted(rates) for role in ("gate_proj", "up_proj", "down_proj")}:
            why.append("observed projection rates differ from declaration")
        if expected["phase"] == "numeric" and any(row.get(k) is not None for k in row if k.endswith("_ratio")):
            why.append("numeric evidence includes promotable timing ratios")
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
