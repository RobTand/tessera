"""CPU causal controls for the current #793 acceptance owner, not CUDA proof."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / 'experiments/t8r_speed/ab_stageprev_accept.py'


def population():
    cases = [
        {'family': 'routed', 'group': group, 'M': str(m), 'rates': rates}
        for group, rates in [('experts.R1024.L10', [4]),
                             ('experts.R1088.L11', [4, 5]),
                             ('experts.R832.L42', [3, 4])]
        for m in (1, 2048)
    ]
    hashes = {f'{arm}-routed{suffix}': ('a' if arm == 'master' else 'b') * 64
              for arm in ('master', 'fix') for suffix in ('', 'b')}
    expected = {'schema': 'tessera.stageprev.expected.v1', 'phase': 'timing',
                'arms': ['master', 'fix'], 'kernel_sha': hashes, 'cases': cases}
    rows = []
    for case in cases:
        ratio = 1.0 if len(case['rates']) == 1 else (0.92 if case['M'] == '2048' else 0.97)
        rows.append({**{k: case[k] for k in ('family', 'group', 'M')},
                     'bitwise': True, 'missing': [], 'fix_ratio': ratio, 'fixb_ratio': ratio})
    summary = {'ref': 'master', 'arms': ['master', 'fix'], 'kernel_sha': hashes, 'rows': rows}
    return expected, summary


def run_checker(tmp_path, expected, summary):
    (tmp_path / 'expected.json').write_text(json.dumps(expected))
    (tmp_path / 'ab_summary.json').write_text(json.dumps(summary))
    env = dict(os.environ, AB_EXPECTED_CASES=str(tmp_path / 'expected.json'))
    return subprocess.run([sys.executable, str(CHECKER), str(tmp_path)], env=env,
                          capture_output=True, text=True, timeout=10)


def test_actual_string_m_and_recorded_route_key(tmp_path):
    expected, summary = population()
    expected['cases'][0]['M'] = '1@ids-000001'
    summary['rows'][0]['M'] = '1@ids-000001'
    result = run_checker(tmp_path, expected, summary)
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize('defect', ['empty', 'missing', 'duplicate', 'extra', 'sha', 'arms'])
def test_expected_population_refuses_incomplete_or_unbound_results(tmp_path, defect):
    expected, summary = population()
    # Integers are also valid historical M keys; they must not hide the failure.
    for row in summary['rows']:
        row['M'] = int(row['M'])
    if defect == 'empty':
        summary['rows'] = []
    elif defect == 'missing':
        summary['rows'] = [r for r in summary['rows'] if r['group'] != 'experts.R832.L42']
    elif defect == 'duplicate':
        summary['rows'].append(deepcopy(summary['rows'][0]))
    elif defect == 'extra':
        row = deepcopy(summary['rows'][0]); row['M'] = 512; summary['rows'].append(row)
    elif defect == 'sha':
        summary['kernel_sha'] = dict(summary['kernel_sha'], **{'fix-routed': 'c' * 64})
    else:
        summary['arms'].append('unqualified')
    result = run_checker(tmp_path, expected, summary)
    assert result.returncode != 0, result.stdout
    assert 'REFUSED' in result.stderr, result.stderr


@pytest.mark.parametrize('defect', ['empty', 'duplicate'])
def test_expected_manifest_itself_cannot_be_vacuous(tmp_path, defect):
    expected, summary = population()
    if defect == 'empty': expected['cases'] = []
    else: expected['cases'].append(deepcopy(expected['cases'][0]))
    result = run_checker(tmp_path, expected, summary)
    assert result.returncode != 0
    assert 'REFUSED' in result.stderr


def test_uniform_rate_is_declared_not_inferred_from_r1024_label(tmp_path):
    expected, summary = population()
    expected['cases'][0]['group'] = summary['rows'][0]['group'] = 'experts.R768.L7'
    expected['cases'][0]['rates'] = [3]
    result = run_checker(tmp_path, expected, summary)
    assert result.returncode == 0, result.stdout + result.stderr


def test_numeric_phase_does_not_claim_timing(tmp_path):
    expected, summary = population(); expected['phase'] = 'numeric'
    for row in summary['rows']:
        row['fix_ratio'] = row['fixb_ratio'] = None
    result = run_checker(tmp_path, expected, summary)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'numeric' in result.stdout


def test_actual_kernel_guards_mixed_history_without_changing_layout():
    source = (ROOT / 'src/tessera/serving/csrc/routed_fused_window.cu').read_text()
    kernel = source[source.index('__global__ void __launch_bounds__(THREADS, 1) routed_fused_kernel'):]
    kernel = kernel[:kernel.index('\nnamespace fp4')]
    assert 'constexpr bool STAGE_PREV = PREV_STAGED && !TWO;' in kernel
    assert 'if constexpr (PREV_STAGED)' not in kernel
    assert 'if (PREV_STAGED ||' not in kernel
    assert 'if constexpr (!PREV_STAGED)' not in kernel
    assert 'constexpr int PREV_REGION_BYTES = PREV_STAGED ? WORD_STAGES * PREV_STAGE_INTS * 4 : 0;' in source
    assert source.count('if (gc >= 2) bar_sync(BAR_EMPTY0 + (gc & 1), THREADS);') == 2
