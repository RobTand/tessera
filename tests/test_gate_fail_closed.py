"""Failure evidence cannot turn into an infrastructure retry or a false pass."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/ci' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_failure_is_final_without_pytest_exit_code():
    actions = load('gate_actions')
    plan = {'population': {'version': 'v1'}, 'mode': 'full', 'base_sha': 'a',
            'head_sha': 'b', 'selected': [{'lane': 'bytes', 'path': 'tests/test_a.py'}]}
    record = {'population_version': 'v1', 'mode': 'full', 'base_sha': 'a', 'head_sha': 'b',
              'lane': 'bytes', 'files': ['tests/test_a.py'], 'verdict': 'fail',
              'boundary': {'status': 'fail'}, 'pytest_exit_code': None}
    result = actions.assess_attempts(plan, [{'index': 0, 'lane': 'bytes',
                                            'files': ['tests/test_a.py']}], {1: {0: record}})
    assert result['retry_indices'] == []
    assert result['exit_code'] == 1


def test_success_needs_population_and_timing_evidence():
    actions = load('gate_actions')
    plan = {'population': {'version': 'v1'}, 'mode': 'full', 'base_sha': 'a', 'head_sha': 'b'}
    record = {'population_version': 'v1', 'mode': 'full', 'base_sha': 'a', 'head_sha': 'b',
              'lane': 'torch', 'files': ['tests/test_a.py'], 'verdict': 'pass',
              'pytest_exit_code': 0, 'budget': {'status': 'unavailable'}}
    assert actions._verdict(plan, ['tests/test_a.py'], record, 'torch') is None


def test_enforced_budget_cannot_be_disabled(tmp_path):
    runner = load('gate_run')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    path = tmp_path / 'tools/ci/gate_budget.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'seconds': 20, 'slow': {}, 'enforce': True}))
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'base'], cwd=tmp_path, check=True)
    path.write_text(json.dumps({'seconds': 20, 'slow': {}, 'enforce': False}))
    with pytest.raises(runner.BudgetFailure, match='disabled'):
        runner._budget_policy(tmp_path, 'HEAD')


def test_quarantine_requires_an_open_owner():
    gate = load('gate_select')
    config = {'lanes': {'bytes': [], 'torch': ['tests/test_a.py::test_a']}, 'issues': {}}
    with pytest.raises(ValueError, match='owner'):
        gate.quarantine(config, issue_state=lambda number: 'open')
    config['issues']['tests/test_a.py::test_a'] = 123
    with pytest.raises(ValueError, match='open'):
        gate.quarantine(config, issue_state=lambda number: 'closed')
    assert gate.quarantine(config, issue_state=lambda number: 'open') == config['lanes']


def test_aggregate_rejects_a_cuda_population_on_a_cpu_lane(tmp_path):
    runner = load('gate_run')
    population = {'role': 'population', 'cuda': True, 'cuda_surface': {'executed': 1},
                  'counts': {'passed': 1}, 'not_collected': [], 'skip_reasons': {}}
    path = tmp_path / 'surface.json'
    path.write_text(json.dumps(population))
    with pytest.raises(ValueError, match='CUDA'):
        runner.cpu_population(path)


@pytest.mark.parametrize('identity', ['name="tests/test_gpu.py"',
                                     'file="tests/test_gpu.py" name="tests.test_gpu"'])
def test_budget_excludes_module_level_skips_but_keeps_test_skips(tmp_path, identity):
    runner = load('gate_run')
    path = tmp_path / 'junit.xml'
    path.write_text('<testsuite><testcase classname="" ' + identity + ' time="0">'
                    '<skipped type="pytest.skip" /></testcase>'
                    '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_a" time="0">'
                    '<skipped /></testcase></testsuite>')
    assert runner._budget_timings(path, ['tests/test_a.py', 'tests/test_gpu.py'],
                                  {'seconds': 20, 'slow': {}}, require_all=False) == []


def test_collection_receipt_accounts_for_every_assigned_file(tmp_path):
    runner = load('gate_run')
    path = tmp_path / 'collection.gw0.json'
    path.write_text(json.dumps({'collected': ['tests/test_a.py'], 'items': {}}))
    surface = {'not_collected': []}
    with pytest.raises(ValueError, match='unaccounted'):
        runner.collection_coverage(tmp_path, ['tests/test_a.py', 'tests/test_b.py'], surface)
    surface['not_collected'] = ['test_b.py']
    record = runner.collection_coverage(tmp_path, ['tests/test_a.py', 'tests/test_b.py'], surface)
    assert record['status'] == 'pass'
    assert record['uncollected'] == ['tests/test_b.py']
