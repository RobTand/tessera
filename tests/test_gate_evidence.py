"""Measurements retain repeat identity and distinguish failures from flakes."""
from __future__ import annotations

import importlib.util
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
