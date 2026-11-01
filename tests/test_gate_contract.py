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
            'boundary': {'status': 'pass'}, 'collection': {'status': 'pass'}, 'surface': {'role': 'population', 'cuda': False,
                                                     'cuda_surface': {'executed': 0}, 'not_collected': [], 'device': 'torch 2.11 reports no CUDA device'}}


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
    assert '--print-mode' in gate
    assert 'CUDA_VISIBLE_DEVICES' in gate
    assert 'self-hosted' not in gate


def test_full_gate_refuses_an_empty_cpu_population(tmp_path, monkeypatch):
    gate = load('gate_select')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'README.md').write_text('empty project\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'base'], cwd=tmp_path, check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, text=True).strip()
    config = tmp_path / 'tools/ci'
    config.mkdir(parents=True)
    (config / 'gate_durations.json').write_text('{"lanes": {"bytes": {}, "torch": {}}}')
    (config / 'gate_quarantine.json').write_text('{"lanes": {"bytes": [], "torch": []}, "issues": {}}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['gate_select', '--base', sha, '--head', sha,
        '--mode', 'full', '--shards', '2', '--output', 'plan.json', '--matrix', 'matrix.json'])
    with pytest.raises(ValueError, match='empty'):
        gate.main()


@pytest.mark.parametrize('grouped', [True, False])
def test_gate_builds_a_module_fixture_once_across_workers(tmp_path, monkeypatch, grouped):
    runner = load('gate_run')
    tests = tmp_path / 'tests'
    tests.mkdir()
    (tests / 'conftest.py').write_text(
        'def pytest_addoption(parser): parser.addoption("--surface-json")\n')
    marker = tmp_path / 'fixture-builds.txt'
    (tests / 'test_shared.py').write_text(
        'import pathlib, pytest, time\n'
        '@pytest.fixture(scope="module")\n'
        'def shared():\n'
        f'    with pathlib.Path({str(marker)!r}).open("a") as log: log.write("build\\n")\n'
        '    return 1\n'
        + ('@pytest.mark.xdist_group(name="test_shared")\n' if grouped else '')
        + '@pytest.mark.parametrize("case", range(8))\n'
        'def test_consumer(shared, case): assert shared == 1\n'
        '@pytest.mark.parametrize("case", range(8))\n'
        'def test_independent(case, worker_id):\n'
        f'    ready = pathlib.Path({str(tmp_path)!r}) / ("independent-" + worker_id + ".ready")\n'
        '    ready.touch()\n'
        '    deadline = time.monotonic() + 10\n'
        '    while len(list(ready.parent.glob("independent-*.ready"))) < 2:\n'
        '        assert time.monotonic() < deadline, "Independent cases did not reach both workers"\n'
        '        time.sleep(.01)\n'
        f'    with pathlib.Path({str(marker.with_name("independent-workers.txt"))!r}).open("a") as log: log.write(worker_id+"\\n")\n')
    monkeypatch.setattr(runner, '_budget_policy', lambda *args: {'seconds': 20, 'slow': {}, 'enforce': True})
    monkeypatch.setattr(runner, 'cpu_population', lambda *args: {
        'role': 'population', 'cuda': False, 'cuda_surface': {'executed': 0},
        'not_collected': [], 'device': 'torch 2.11 reports no CUDA device', 'counts': {'passed': 8}})
    output = tmp_path / 'output'
    output.mkdir()
    plan = {'mode': 'full', 'base_sha': 'base', 'head_sha': 'head',
            'population': {'version': 'fixture-regression'},
            'selected': [{'lane': 'torch', 'path': 'tests/test_shared.py', 'estimated_seconds': 1}]}
    code = runner.run_plan(plan, root=tmp_path, output=output / 'result.json',
                           junit=output / 'junit.xml', workers=2, affinity={0, 1})
    assert code == 0
    assert marker.read_text().splitlines() == ['build']
    assert len(set(marker.with_name('independent-workers.txt').read_text().splitlines())) == 2


def test_file_scheduler_preserves_the_duration_order(tmp_path, monkeypatch):
    runner = load('gate_run')
    monkeypatch.setattr(runner, '_budget_policy', lambda *args: {'seconds': 20, 'slow': {}})
    commands = []
    def capture(command, **kwargs):
        commands.append((command, json.loads(kwargs['env']['GATE_ASSIGNED_FILES'])))
        return SimpleNamespace(returncode=3)
    plan = {'selected': [{'lane': 'torch', 'path': 'tests/test_short.py', 'estimated_seconds': 1},
                         {'lane': 'torch', 'path': 'tests/test_long.py', 'estimated_seconds': 30}]}
    assert runner.run_plan(plan, root=tmp_path, output=tmp_path / 'result.json',
                           junit=tmp_path / 'junit.xml', execute=capture, affinity={0, 1}) == 75
    command, files = commands[0]
    assert files == ['tests/test_long.py', 'tests/test_short.py']
    assert '--no-loadscope-reorder' in command


@pytest.mark.parametrize('name, expected', [
    ('test_cpu[value]@test_matched_reach', 'test_cpu[value]'),
    ('test_cpu[email@test_matched_reach]', 'test_cpu[email@test_matched_reach]'),
])
def test_grouped_junit_keeps_exact_budget_and_evidence_identity(tmp_path, name, expected):
    filename = 'tests/test_matched_reach.py'
    nodeid = filename + '::' + expected
    junit = tmp_path / 'junit.xml'
    junit.write_text('<testsuite><testcase file="' + filename + '" '
                     'classname="tests.test_matched_reach" name="' + name + '" '
                     'time="177" /></testsuite>')
    runner = load('gate_run')
    assert runner._budget_timings(junit, [filename], {'seconds': 20, 'slow': {nodeid: 100}}) == [
        {'nodeid': nodeid, 'seconds': 177, 'budget_seconds': 100}]
    assert list(load('gate_evidence').cases(junit))[0]['nodeid'] == nodeid


def test_runner_preserves_a_measured_exception_above_the_global_timeout(tmp_path, monkeypatch):
    runner = load('gate_run')
    monkeypatch.setattr(runner, '_budget_policy', lambda *args: {
        'seconds': 80, 'slow': {'tests/test_a.py::test_slow': 357}, 'enforce': True})
    seen = []
    def capture(command, **kwargs):
        seen.append(kwargs['env'])
        assert json.loads(kwargs['env'].get('GATE_TEST_TIMEOUTS', '{}')) == {
            'tests/test_a.py::test_slow': 535.5}
        raise RuntimeError('stop after checking the actual pytest environment')
    assert runner.run_plan(fixture_plan(), root=tmp_path, output=tmp_path/'result.json',
                           junit=tmp_path/'junit.xml', workers=1, lane='torch',
                           execute=capture, affinity={0}) == 75
    assert len(seen) == 1


def test_collector_applies_the_measured_timeout_to_the_exact_node(tmp_path):
    import os
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/test_a.py').write_text(
        'def test_slow(request):\n'
        '    assert request.node.get_closest_marker("timeout").args == (535.5,)\n'
        'def test_ordinary(request):\n'
        '    assert request.node.get_closest_marker("timeout") is None\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'tools/ci'),
               GATE_COLLECTION_DIR=str(tmp_path/'collection'),
               GATE_ASSIGNED_FILES=json.dumps(['tests/test_a.py']), GATE_DESELECT='[]',
               GATE_TEST_TIMEOUTS=json.dumps({'tests/test_a.py::test_slow':535.5}))
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'gate_collect',
                             '--timeout=300', '--rootdir', str(tmp_path), str(tmp_path/'tests')],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
