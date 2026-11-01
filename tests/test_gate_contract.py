"""Hosted gate coverage, identity and retry contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / 'ci' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('change', ['fork', 'login', 'id', 'branch'])
def test_batch_identity_requires_all_fields(change):
    gate = load('gate_select')
    pr = {'head': {'ref': 'mergify/merge-queue/master/pr-1', 'repo': {'full_name': 'RobTand/tessera'}},
          'user': {'login': 'mergify[bot]', 'id': 37929162}, 'labels': []}
    assert gate.full_gate(pr, 'RobTand/tessera')
    if change == 'fork':
        pr['head']['repo']['full_name'] = 'fork/tessera'
    elif change == 'branch':
        pr['head']['ref'] = 'tessera-1559-gate'
    elif change == 'login':
        pr['user']['login'] = 'someone'
    else:
        pr['user']['id'] = 1
    assert not gate.full_gate(pr, 'RobTand/tessera')


def test_canary_is_same_repository_only():
    gate = load('gate_select')
    pr = {'head': {'ref': 'tessera-1559-gate', 'repo': {'full_name': 'RobTand/tessera'}},
          'user': {}, 'labels': [{'name': 'gate-canary'}]}
    assert gate.full_gate(pr, 'RobTand/tessera')
    pr['head']['repo']['full_name'] = 'fork/tessera'
    assert not gate.full_gate(pr, 'RobTand/tessera')


def test_partition_covers_both_lanes_once_and_uses_both_architectures():
    partition = load('gate_partition')
    files = ['tests/test_a.py', 'tests/test_b.py', 'tests/test_c.py']
    matrix = partition.lane_matrix(files, {'torch': {files[0]: 100}, 'bytes': {}}, 2,
                                   ['ubuntu-24.04', 'ubuntu-24.04-arm'])
    for lane in ('bytes', 'torch'):
        rows = [r for r in matrix['include'] if r['lane'] == lane]
        assert sorted(p for row in rows for p in row['files']) == files
        assert {r['runner'] for r in rows} == {'ubuntu-24.04', 'ubuntu-24.04-arm'}
    assert len({r['index'] for r in matrix['include']}) == len(matrix['include'])


def fixture_plan():
    return {'population': {'version': 'cpu-v1'}, 'base_sha': 'base', 'head_sha': 'head',
            'mode': 'full', 'selected': [{'lane': 'bytes', 'path': 'tests/test_a.py'},
                                         {'lane': 'torch', 'path': 'tests/test_a.py'}]}


def verdict(plan, lane, value, code):
    return {**{k: plan[k] for k in ('base_sha', 'head_sha', 'mode')},
            'population_version': 'cpu-v1', 'lane': lane, 'files': ['tests/test_a.py'],
            'verdict': value, 'pytest_exit_code': code, 'budget': {'status': 'pass'},
            'boundary': {'status': 'pass'}, 'surface': {'role': 'population', 'cuda': False,
                                                     'cuda_surface': {'executed': 0}}}


def test_failures_are_final_and_only_missing_verdicts_retry():
    actions = load('gate_actions')
    plan = fixture_plan()
    assignments = [{'index': i, 'lane': lane, 'files': ['tests/test_a.py']}
                   for i, lane in enumerate(('bytes', 'torch'))]
    attempts = {1: {0: verdict(plan, 'bytes', 'fail', 1)},
                2: {0: verdict(plan, 'bytes', 'pass', 0),
                    1: verdict(plan, 'torch', 'pass', 0)}}
    result = actions.assess_attempts(plan, assignments, attempts)
    assert result['retry_indices'] == [1]
    assert result['failed_indices'] == [0]
    assert result['exit_code'] == 1


def test_foreign_lane_verdict_cannot_certify_coverage():
    actions = load('gate_actions')
    plan = fixture_plan()
    assignments = [{'index': i, 'lane': lane, 'files': ['tests/test_a.py']}
                   for i, lane in enumerate(('bytes', 'torch'))]
    result = actions.assess_attempts(plan, assignments,
        {1: {i: verdict(plan, 'torch', 'pass', 0) for i in (0, 1)}})
    assert result['exit_code'] == 75
    assert result['retry_indices'] == [0]


@pytest.mark.parametrize('config', [{'seconds': 21, 'slow': {}},
                                   {'seconds': 20, 'slow': {'new::test': 30}}])
def test_budget_cannot_grow_against_base(tmp_path, config):
    runner = load('gate_run')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    path = tmp_path / 'tools/ci/gate_budget.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seconds': 20, 'slow': {}}))
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'base'], cwd=tmp_path, check=True)
    path.write_text(json.dumps(config))
    with pytest.raises(runner.BudgetFailure, match='may not grow'):
        runner._budget_policy(tmp_path, 'HEAD')


def test_budget_tolerance_and_measurement_mode(tmp_path):
    runner = load('gate_run')
    assert runner.BUDGET_ENFORCEMENT_TOLERANCE == 1.5
    junit = tmp_path / 'junit.xml'
    junit.write_text('<testsuite><testcase file="tests/test_a.py" classname="tests.test_a" '
                     'name="test_one" time="31" /></testsuite>')
    rows = runner._budget_timings(junit, ['tests/test_a.py'], {'seconds': 20, 'slow': {}})
    assert rows == [{'nodeid': 'tests/test_a.py::test_one', 'seconds': 31, 'budget_seconds': 20}]


def test_selector_failure_runs_full_population(tmp_path):
    gate = load('gate_select')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_a.py').write_text('def test_a(): pass\n')
    def broken(*args, **kwargs):
        return SimpleNamespace(returncode=2, stdout='', stderr='graph unavailable')
    selection = gate.select_files(tmp_path, 'base', 'head', execute=broken)
    assert selection['verdict'] == 'full'
    assert selection['tests'] == ['tests/test_a.py']
    assert 'graph unavailable' in selection['reason']


def test_workflow_keeps_boundary_proofs_and_branch_check_separate():
    text = (ROOT / '.github/workflows/ci.yml').read_text()
    assert 'name: branch rule' in text
    assert '--dist worksteal' in text
    for proof in ('byte-layer modules import without torch', 'an empty denylist must refuse',
                  'the wheel and the sdist ship', 'boundary allowlists only shrink'):
        assert proof in text
    gate = (ROOT / '.github/workflows/gate.yml').read_text()
    assert 'types: [opened, synchronize, reopened]' in gate
    assert 'name: pr-gate' in gate and 'name: gate' in gate
    assert '37929162' in gate
    assert 'CUDA_VISIBLE_DEVICES' in gate
    assert 'self-hosted' not in gate
