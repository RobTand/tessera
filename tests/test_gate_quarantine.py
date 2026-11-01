"""Quarantine cannot hide regressions in the same change."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate_select', ROOT / 'tools/ci/gate_select.py')
select = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select)
QUARANTINE = 'tools/ci/gate_quarantine.json'


def entry(target='tests/test_one.py::test_bad', issue=3201):
    return {'target': target, 'issue': issue, 'lane': 'torch'}


@pytest.mark.parametrize('current', [
    [entry(), entry('tests/test_other.py')],
    [entry('tests/test_one.py')],
    [entry(issue=3202)],
])
def test_quarantine_growth_cannot_accompany_code_changes(current):
    with pytest.raises(ValueError, match='quarantine-only'):
        select.quarantine_policy({'entries': current}, {'entries': [entry()]},
                                 [QUARANTINE, 'src/tessera/owner.py'])


@pytest.mark.parametrize('changed', [[QUARANTINE], [QUARANTINE, 'docs/ci_gate.md']])
def test_quarantine_only_change_can_add_entries(changed):
    select.quarantine_policy({'entries': [entry(), entry('tests/test_other.py')]},
                             {'entries': [entry()]}, changed)


def test_removal_and_file_to_exact_node_are_always_allowed():
    select.quarantine_policy({'entries': [entry()]},
        {'entries': [entry('tests/test_one.py'), entry('tests/test_other.py')]},
        [QUARANTINE, 'src/tessera/owner.py'])


def test_first_policy_bootstraps_when_base_has_no_policy():
    select.quarantine_policy({'entries': [entry()]}, None, [QUARANTINE, 'tools/ci/gate_select.py'])


@pytest.mark.parametrize('response', [{'state': 'closed'}, {'state': 'open', 'pull_request': {}}])
def test_owner_must_be_an_open_issue(response):
    with pytest.raises(ValueError, match='open issue'):
        select.check_quarantine_owners([entry()], fetch=lambda issue: response)


def test_owner_lookup_is_deduplicated():
    calls = []
    def fetch(issue):
        calls.append(issue)
        return {'state': 'open'}
    select.check_quarantine_owners([entry(), entry('tests/test_other.py')], fetch=fetch)
    assert calls == [3201]


def test_owner_api_failure_is_no_verdict():
    def unavailable(issue):
        raise TimeoutError('GitHub unavailable')
    with pytest.raises(RuntimeError, match='no verdict'):
        select.check_quarantine_owners([entry()], fetch=unavailable)


def test_owner_rate_limit_and_server_error_are_no_verdict():
    import urllib.error
    for status in (403, 429, 500, 503):
        def fetch(issue):
            raise urllib.error.HTTPError('https://api.github.com', status, 'unavailable', {}, None)
        with pytest.raises(RuntimeError, match='no verdict'):
            select.check_quarantine_owners([entry()], fetch=fetch)


def _captured_request(monkeypatch, token):
    import io
    import json
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        seen['headers'] = {key.lower(): value for key, value in request.header_items()}
        return Response(json.dumps({'state': 'open'}).encode())

    if token is None:
        monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    else:
        monkeypatch.setenv('GITHUB_TOKEN', token)
    monkeypatch.setattr(select.urllib.request, 'urlopen', urlopen)
    assert select.fetch_quarantine_owner(3201) == {'state': 'open'}
    return seen['headers']


def test_owner_lookup_authenticates_with_the_workflow_token(monkeypatch):
    # Unauthenticated REST shares a small per-IP limit across hosted runners
    # and returned HTTP 403 in run 41289555169; the job token avoids that.
    headers = _captured_request(monkeypatch, 'job-token')
    assert headers['authorization'] == 'Bearer job-token'


def test_owner_lookup_without_a_token_stays_anonymous(monkeypatch):
    assert 'authorization' not in _captured_request(monkeypatch, None)


@pytest.mark.parametrize('target', ['tests/test_one.py::', 'tests/test_one.py::test_bad[a',
                                    'tests/../test_one.py', '/tests/test_one.py'])
def test_quarantine_rejects_non_exact_syntax(target):
    with pytest.raises(ValueError, match='exact'):
        select.exclusion_entries({'entries': [entry(target)]})


def test_base_must_exist_before_bootstrap(tmp_path):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    with pytest.raises(subprocess.CalledProcessError):
        select.base_quarantine(tmp_path, 'a' * 40)


@pytest.mark.parametrize('target,code', [
    ('tests/test_one.py::test_bad[a]', 0),
    ('tests/test_one.py::test_bad[ab]', 1),
    ('tests/test_one.py::test_bad[a', 4),
    ('tests/test_one.py::', 4),
    ('tests/test_one.py::test_missing', 4),
    ('tests/test_one.py', 5),
])
def test_collection_deselects_only_exact_nodes(tmp_path, target, code):
    import json, os, subprocess, sys
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_one.py').write_text(
        'import pytest\n@pytest.mark.parametrize("value", ["a", "ab"])\n'
        'def test_bad(value): assert value != "a"\n')
    env = dict(os.environ, GATE_ASSIGNED_FILES=json.dumps(['tests/test_one.py']),
               GATE_DESELECT=json.dumps([target]), GATE_COLLECTION_DIR=str(tmp_path / 'out'),
               PYTHONPATH=str(ROOT / 'tools/ci'))
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'gate_collect',
                             '--rootdir', str(tmp_path), str(tmp_path)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == code, result.stdout + result.stderr
