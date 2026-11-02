"""Issue-owned retries are exact, bounded, visible, and expire with the issue."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = 'tests/test_probe.py::test_probe'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools/ci' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table(node=NODE):
    return {node: {'issue': 1529, 'reason': 'Deterministic retry probe'}}


def run_gate(tmp_path, source, marks, workers=0):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_probe.py').write_text(source)
    config = tmp_path / 'tools/ci'
    config.mkdir(parents=True)
    (config / 'gate_flaky.json').write_text(json.dumps(marks))
    summary = tmp_path / 'summary.md'
    summary.touch()
    # Nested pytest must not inherit the outer shard assignment or credentials.
    env = {key: value for key, value in os.environ.items()
           if not key.endswith(('_API_KEY', '_TOKEN')) and
           not key.startswith(('GATE_', 'PYTEST_'))}
    env.update(PYTHONPATH=str(ROOT / 'tools/ci'),
               GATE_ASSIGNED_FILES=json.dumps(['tests/test_probe.py']),
               GITHUB_STEP_SUMMARY=str(summary),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'gate_collect',
               '--rootdir', str(tmp_path), '--junitxml', str(tmp_path / 'junit.xml'),
               '-o', 'junit_family=xunit1']
    if workers:
        command += ['-n', str(workers), '--dist=loadgroup']
    result = subprocess.run(command + [str(tmp_path / 'tests')], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=45)
    return result, summary.read_text()


def probe(fail_until):
    return ('from pathlib import Path\nimport pytest\n'
            '@pytest.fixture(scope="module")\ndef shared(): return 1\n'
            'def test_probe(shared):\n'
            '    path = Path("attempts")\n'
            '    count = int(path.read_text()) + 1 if path.exists() else 1\n'
            '    path.write_text(str(count))\n'
            f'    assert count > {fail_until}\n')


@pytest.mark.parametrize('workers', [0, 2])
def test_marked_failure_then_pass_records_one_retry(tmp_path, workers):
    result, summary = run_gate(tmp_path, probe(1), table(), workers)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'attempts').read_text() == '2'
    for output in (result.stdout, summary):
        assert 'GATE_FLAKY_RETRY' in output
        assert NODE in output and '#1529' in output
    cases = list(ET.parse(tmp_path / 'junit.xml').iter('testcase'))
    assert len(cases) == 1
    assert cases[0].find('failure') is None
    assert cases[0].find('error') is None


def test_marked_two_failures_are_final(tmp_path):
    result, summary = run_gate(tmp_path, probe(2), table())
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'attempts').read_text() == '2'
    assert summary.count('GATE_FLAKY_RETRY') == 1
    assert NODE in summary and '#1529' in summary
    assert len(list(ET.parse(tmp_path / 'junit.xml').iter('failure'))) == 1


def test_unmarked_failure_has_no_retry(tmp_path):
    result, summary = run_gate(tmp_path, probe(1), {})
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'attempts').read_text() == '1'
    assert 'GATE_FLAKY_RETRY' not in result.stdout + summary


def test_marked_pass_has_no_retry(tmp_path):
    result, summary = run_gate(tmp_path, probe(0), table())
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'attempts').read_text() == '1'
    assert 'GATE_FLAKY_RETRY' not in result.stdout + summary


@pytest.mark.parametrize('unknown', [NODE + '_missing', 'tests/test_missing.py::test_probe'])
def test_table_refuses_unknown_nodes(tmp_path, unknown):
    result, _ = run_gate(tmp_path, probe(0), table(unknown))
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'flaky' in result.stdout + result.stderr
    assert unknown in result.stdout + result.stderr
    assert not (tmp_path / 'attempts').exists()


def test_parameter_mark_does_not_retry_a_prefix_neighbor(tmp_path):
    source = ('from pathlib import Path\nimport pytest\n'
              '@pytest.mark.parametrize("value", ["a", "ab"])\n'
              'def test_probe(value):\n'
              '    path = Path(value)\n'
              '    count = int(path.read_text()) + 1 if path.exists() else 1\n'
              '    path.write_text(str(count))\n'
              '    assert count > 1\n')
    result, summary = run_gate(tmp_path, source, table(NODE + '[a]'), workers=2)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'a').read_text() == '2'
    assert (tmp_path / 'ab').read_text() == '1'
    assert NODE + '[a]' in summary
    assert NODE + '[ab]' not in summary


@pytest.mark.parametrize('state', ['closed', 'missing', 'pull_request'])
def test_table_refuses_issue_without_an_open_deflake_owner(tmp_path, state):
    flaky = load('gate_flaky')
    select = load('gate_select')
    config = tmp_path / 'tools/ci'
    config.mkdir(parents=True)
    (config / 'gate_flaky.json').write_text(json.dumps(table()))
    calls = []
    def fetch(issue):
        calls.append(issue)
        if state == 'missing':
            raise urllib.error.HTTPError('https://api.github.com', 404, 'missing', {}, None)
        return {'state': 'closed'} if state == 'closed' else {'state': 'open', 'pull_request': {}}
    def check(entries):
        select.check_quarantine_owners(entries, fetch=fetch)
    with pytest.raises(ValueError, match='open issue'):
        flaky.check_policy(tmp_path, ['tests/test_probe.py'], check)
    assert calls == [1529]


def test_policy_checks_owners_even_outside_the_changed_selection(tmp_path):
    flaky = load('gate_flaky')
    config = tmp_path / 'tools/ci'
    config.mkdir(parents=True)
    (config / 'gate_flaky.json').write_text(json.dumps(table()))
    entries = []
    assert flaky.check_policy(tmp_path, ['tests/test_probe.py'], entries.extend) == ['tests/test_probe.py']
    assert [row['issue'] for row in entries] == [1529]
    with pytest.raises(ValueError, match='population'):
        flaky.check_policy(tmp_path, ['tests/test_other.py'], entries.extend)


@pytest.mark.parametrize('entry', [
    {'issue': True, 'reason': 'bad'}, {'issue': 0, 'reason': 'bad'},
    {'issue': 1529, 'reason': ''}, {'issue': 1529},
])
def test_table_requires_an_issue_and_reason(tmp_path, entry):
    flaky = load('gate_flaky')
    config = tmp_path / 'tools/ci'
    config.mkdir(parents=True)
    (config / 'gate_flaky.json').write_text(json.dumps({NODE: entry}))
    with pytest.raises(ValueError, match='flaky'):
        flaky.check_policy(tmp_path, ['tests/test_probe.py'], lambda entries: None)
