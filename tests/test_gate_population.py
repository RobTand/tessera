"""A torch gate needs an actual torch population and an honest summary."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/ci' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('surface', [
    {'not_collected': ['test_gpu.py'], 'device': 'torch 2.11 reports no CUDA device'},
    {'not_collected': [], 'device': 'no torch in this interpreter (ImportError)'},
])
def test_aggregate_refuses_missing_torch_or_uncollected_modules(surface):
    actions = load('gate_actions')
    plan = {'population': {'version': 'v'}, 'mode': 'full', 'base_sha': 'a', 'head_sha': 'b'}
    record = {**plan, 'population_version': 'v', 'lane': 'torch', 'files': ['tests/test_a.py'],
              'verdict': 'pass', 'pytest_exit_code': 0, 'budget': {'status': 'pass'},
              'collection': {'status': 'pass'}, 'surface': {
                  'role': 'population', 'cuda': False, 'cuda_surface': {'executed': 0}, **surface}}
    assert actions._verdict(plan, record['files'], record, 'torch') is None


@pytest.mark.parametrize('change,expected', [('batch','full'),('fork','selection'),('login','selection'),
                                           ('id','selection'),('branch','selection'),('canary','full')])
def test_workflow_executes_the_tested_batch_decision(tmp_path, change, expected):
    import os
    pr = {'head': {'ref': 'mergify/merge-queue/master/pr-1', 'repo': {'full_name': 'RobTand/tessera'}},
          'user': {'login': 'mergify[bot]', 'id': 37929162}, 'labels': []}
    if change == 'fork': pr['head']['repo']['full_name'] = 'fork/tessera'
    if change == 'login': pr['user']['login'] = 'someone'
    if change == 'id': pr['user']['id'] = 1
    if change in ('branch', 'canary'): pr['head']['ref'] = 'tessera-1559-gate'
    if change == 'canary': pr['labels'] = [{'name': 'gate-canary'}]
    event = tmp_path / 'event.json'; event.write_text(json.dumps({'pull_request': pr}))
    output = tmp_path / 'output'
    workflow = (ROOT / '.github/workflows/gate.yml').read_text()
    command = next(line.split('run: ', 1)[1] for line in workflow.splitlines()
                   if 'run: ' in line and '--print-mode' in line)
    result = subprocess.run(['bash', '-c', command], cwd=ROOT,
                            env=dict(os.environ, GITHUB_EVENT_PATH=str(event),
                                     GITHUB_REPOSITORY='RobTand/tessera', GITHUB_OUTPUT=str(output)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == 'mode=' + expected
    assert "needs.plan.outputs.mode == 'full'" in workflow
    assert "needs.plan.outputs.mode != 'full'" in workflow
