"""Measurements retain repeat identity and distinguish failures from flakes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/ci' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_distinguishes_always_failures_from_flakes(tmp_path):
    evidence = load('gate_evidence')
    runs = []
    for repeat, failure in enumerate((True, False, True), 1):
        junit = tmp_path / f'run-{repeat}.xml'
        junit.write_text('<testsuite>'
            '<testcase file="tests/test_a.py" classname="tests.test_a" name="test_always" time="2"><failure /></testcase>'
            f'<testcase file="tests/test_a.py" classname="tests.test_a" name="test_flaky" time="{repeat}">'
            + ('<failure />' if failure else '') + '</testcase></testsuite>')
        runs.append({'lane': 'torch', 'repeat': repeat, 'junit': str(junit)})
    report = evidence.analyze_runs(runs)
    lane = report['lanes']['torch']
    assert lane['files'] == {'tests/test_a.py': 4}
    assert lane['always_failing'] == ['tests/test_a.py::test_always']
    assert lane['flaky'] == ['tests/test_a.py::test_flaky']
    assert len(report['runs']) == 3


def test_selector_timeout_runs_full_population(tmp_path):
    gate = load('gate_select')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_a.py').write_text('def test_a(): pass\n')
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('selector', 120)
    assert gate.select_files(tmp_path, 'base', 'head', execute=timeout)['verdict'] == 'full'


def test_canary_timing_uses_final_gate_and_preserves_setup_and_failures():
    evidence = load('gate_evidence')
    runs = []
    for number, seconds, conclusion in [(1, 120, 'success'), (2, 180, 'success'),
                                        (3, 240, 'success'), (4, 600, 'failure')]:
        minute = seconds // 60
        runs.append({'run': {'id': number, 'status': 'completed', 'conclusion': conclusion,
                             'created_at': '2026-11-01T00:00:00Z'},
                     'jobs': [{'name': 'shard-0-torch-a1', 'labels': ['ubuntu-24.04-arm'],
                               'started_at': '2026-11-01T00:00:10Z',
                               'completed_at': '2026-11-01T00:01:50Z',
                               'steps': [{'name': 'Install pinned lane dependencies',
                                          'started_at': '2026-11-01T00:00:20Z',
                                          'completed_at': '2026-11-01T00:00:50Z'},
                                         {'name': 'Run assigned population',
                                          'started_at': '2026-11-01T00:00:50Z',
                                          'completed_at': '2026-11-01T00:01:40Z'}]},
                              {'name': 'gate', 'conclusion': conclusion,
                               'completed_at': f'2026-11-01T00:{minute:02d}:00Z'}]})
    result = evidence.analyze_canaries(runs)
    assert result['passing_runs'] == 3
    assert result['p50_seconds'] == 180
    assert result['p90_seconds'] == 240
    assert result['runs'][3]['conclusion'] == 'failure'
    assert result['runs'][3]['wall_seconds'] == 600
    shard = result['runs'][0]['shards'][0]
    assert shard['job_seconds'] == 100
    assert shard['dependency_setup_seconds'] == 30
    assert shard['test_step_seconds'] == 50
    assert shard['other_seconds'] == 20
    assert shard['runners'] == ['ubuntu-24.04-arm']
    assert evidence.analyze_canaries([])['p90_seconds'] is None


def test_skipped_canary_jobs_do_not_report_negative_work():
    evidence = load('gate_evidence')
    result = evidence.analyze_canaries([{
        'run': {'id': 1, 'status': 'completed', 'conclusion': 'failure',
                'created_at': '2026-11-01T00:00:00Z'},
        'jobs': [{'name': 'shard-unused-a2', 'conclusion': 'skipped',
                  'started_at': '2026-11-01T00:00:10Z',
                  'completed_at': '2026-11-01T00:00:09Z'}]}])
    assert result['runs'][0]['shards'] == []
    assert result['runs'][0]['wall_seconds'] is None
    assert result['passing_runs'] == 0
    assert evidence._elapsed('2026-11-01T00:00:10Z', '2026-11-01T00:00:09Z') is None


def test_canary_rerun_timing_requires_its_own_request_timestamp():
    evidence = load('gate_evidence')
    entry = {'run': {'id': 1, 'run_attempt': 2, 'status': 'completed',
                     'conclusion': 'success', 'created_at': '2026-11-01T00:00:00Z'},
             'jobs': [{'name': 'gate', 'conclusion': 'success',
                       'completed_at': '2026-11-01T01:02:00Z'}]}
    result = evidence.analyze_canaries([entry])
    assert result['runs'][0]['wall_seconds'] is None
    assert result['passing_runs'] == 0
    entry['requested_at'] = '2026-11-01T01:00:00Z'
    result = evidence.analyze_canaries([entry])
    assert result['runs'][0]['wall_seconds'] == 120
    assert result['p90_seconds'] == 120


def test_budget_bootstrap_uses_measured_default_for_cold_consumers(tmp_path, monkeypatch):
    evidence = load('gate_evidence')
    junit = tmp_path / 'junit.xml'
    junit.write_text('<testsuite>'
        '<testcase file="tests/test_cpu.py" classname="tests.test_cpu" name="test_cold" time="39.884" />'
        '<testcase file="tests/test_cpu.py" classname="tests.test_cpu" name="test_slow" time="100" />'
        '</testsuite>')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'lane': 'torch', 'repeat': 1, 'junit': str(junit)}]))
    budget = tmp_path / 'budget.json'
    monkeypatch.setattr(sys, 'argv', ['gate_evidence', '--manifest', str(manifest),
        '--output', str(tmp_path / 'report.json'), '--budget-output', str(budget),
        '--budget-default-seconds', '80'])
    assert evidence.main() == 0
    policy = json.loads(budget.read_text())
    assert policy['seconds'] == 80
    assert policy['enforce'] is True
    assert policy['slow'] == {'tests/test_cpu.py::test_slow': 200}


def test_canary_timing_uses_verdict_and_completion_steps_when_job_clock_differs():
    evidence = load('gate_evidence')
    result = evidence.analyze_canaries([{
        'run': {'id': 1, 'status': 'completed', 'conclusion': 'success',
                'created_at': '2026-11-01T08:00:00Z'},
        'jobs': [
            {'name': 'gate', 'conclusion': 'success', 'completed_at': '2026-11-01T09:02:02Z',
             'steps': [{'name': 'Require complete lane coverage and final verdicts',
                        'completed_at': '2026-11-01T08:02:00Z'}]},
            {'name': 'shard-0-torch-a1', 'conclusion': 'success',
             'started_at': '2026-11-01T08:00:10Z', 'completed_at': '2026-11-01T09:01:52Z',
             'steps': [{'name': 'Install pinned lane dependencies',
                        'started_at': '2026-11-01T08:00:20Z', 'completed_at': '2026-11-01T08:00:50Z'},
                       {'name': 'Run assigned population',
                        'started_at': '2026-11-01T08:00:50Z', 'completed_at': '2026-11-01T08:01:40Z'},
                       {'name': 'Complete job', 'completed_at': '2026-11-01T08:01:50Z'}]}]}])
    run = result['runs'][0]
    assert run['wall_seconds'] == 120
    assert run['gate_job_completed_at'] == '2026-11-01T09:02:02Z'
    assert run['verdict_completed_at'] == '2026-11-01T08:02:00Z'
    assert run['shards'][0]['job_seconds'] == 100
    assert run['shards'][0]['other_seconds'] == 20


def test_canary_uses_job_clock_upper_bound_when_steps_precede_run_creation():
    evidence = load('gate_evidence')
    result = evidence.analyze_canaries([{
        'run': {'id': 1, 'status': 'completed', 'conclusion': 'success',
                'created_at': '2026-11-01T09:00:00Z'},
        'jobs': [
            {'name': 'gate', 'conclusion': 'success', 'completed_at': '2026-11-01T09:02:02Z',
             'steps': [{'name': 'Require complete lane coverage and final verdicts',
                        'completed_at': '2026-11-01T08:02:00Z'}]},
            {'name': 'shard-0-torch-a1', 'conclusion': 'success',
             'started_at': '2026-11-01T09:00:10Z', 'completed_at': '2026-11-01T09:01:52Z',
             'steps': [{'name': 'Install pinned lane dependencies',
                        'started_at': '2026-11-01T08:00:20Z', 'completed_at': '2026-11-01T08:00:50Z'},
                       {'name': 'Run assigned population',
                        'started_at': '2026-11-01T08:00:50Z', 'completed_at': '2026-11-01T08:01:40Z'},
                       {'name': 'Complete job', 'completed_at': '2026-11-01T08:01:50Z'}]}]}])
    run = result['runs'][0]
    assert run['wall_seconds'] == 122
    assert run['verdict_completed_at'] == '2026-11-01T08:02:00Z'
    assert run['timing_basis'] == 'job_completion_upper_bound'
    assert run['shards'][0]['job_seconds'] == 102
    assert run['shards'][0]['other_seconds'] == 22
