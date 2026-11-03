"""Full fallback and post-merge populations cross separate bounded Run requests."""
import importlib.util
import json
from pathlib import Path
import shlex
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'tools/ci/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


XML = '''<query>
<rule name="//tests:a" class="py_test"/>
<rule name="//tests:b" class="py_test"/>
<rule name="//tests:c" class="py_test"/>
<rule name="//tests:runner" class="py_test"><list name="tags"><string value="runner"/></list></rule>
<rule name="//tests:manual" class="py_test"><list name="tags"><string value="manual"/></list></rule>
<rule name="//tests:gpu" class="py_test"><list name="tags"><string value="gpu"/></list></rule>
<rule name="//tools:binary" class="py_binary"/>
<rule name="//src:library" class="py_library"/>
</query>'''


def test_committed_files_partition_every_eligible_rule_and_test(tmp_path):
    shards = load('full_shards')
    shards.generate(tmp_path, XML)
    for arch in ['amd64', 'arm64']:
        parts = [shards.validate(tmp_path, XML, arch, index) for index in range(2)]
        rules = [set(part['rules']) for part in parts]
        tests = [set(part['tests']) for part in parts]
        expected_tests = {'//tests:a', '//tests:b', '//tests:c'}
        if arch == 'amd64':
            expected_tests.add('//tests:runner')
        assert not rules[0] & rules[1]
        assert set.union(*rules) == expected_tests | {'//tools:binary', '//src:library'}
        assert not tests[0] & tests[1]
        assert set.union(*tests) == expected_tests
        for part in parts:
            assert Path(part['path']).is_relative_to(tmp_path / 'tools/bazel/full_targets')
            assert Path(part['path']).read_text().splitlines() == sorted(part['rules'])


@pytest.mark.parametrize('mutation', ['file', 'live-query', 'manifest'])
def test_changed_lists_or_populations_fail_closed(tmp_path, mutation):
    shards = load('full_shards')
    shards.generate(tmp_path, XML)
    xml = XML
    if mutation == 'file':
        path = Path(shards.validate(tmp_path, XML, 'amd64', 0)['path'])
        path.write_text(path.read_text() + '//tests:unknown\n')
    elif mutation == 'manifest':
        path = tmp_path / 'tools/bazel/full_targets/manifest.json'
        path.write_text('{}')
    else:
        xml = XML.replace('</query>', '<rule name="//tools:new" class="py_binary"/></query>')
    with pytest.raises(ValueError):
        shards.validate(tmp_path, xml, 'amd64', 0)


@pytest.mark.parametrize('arch,index', [('cloud', 0), ('amd64', -1), ('arm64', 2)])
def test_invalid_shard_coordinates_fail_closed(tmp_path, arch, index):
    shards = load('full_shards')
    shards.generate(tmp_path, XML)
    with pytest.raises(ValueError):
        shards.validate(tmp_path, XML, arch, index)


def test_run_requests_use_short_shard_coordinates_and_stay_under_an_hour():
    remote = load('remote_gate')
    class Args:
        mode, base, head, checkout, branch = 'main', 'a' * 40, 'b' * 40, '', 'master'
        repo, cpu, memory, timeout = remote.REPOSITORY, 4, '8GB', 120
    initial = remote.run_request(Args)
    assert '--plan-full' in initial['steps'][0]['run']
    for arch in ['amd64', 'arm64']:
        for index in range(2):
            request = remote.run_request(Args, full_shard=(arch, index))
            assert f'--full-shard {arch}:{index}' in request['steps'][0]['run']
            assert int(request['timeout'].removesuffix('m')) < 60
            assert '//tests:' not in json.dumps(request)
            assert request['platform_properties'] == initial['platform_properties']


def results(mode='pr'):
    tests = ['//tests:a', '//tests:b']
    arches = ['amd64'] if mode == 'pr' else ['amd64', 'arm64']
    plan = {'head': 'b' * 40, 'mode': mode, 'selection': 'fallback', 'selected': tests,
            'scopes': [], 'all_scopes': True, 'inputs': [], 'graph_impact': None,
            'full_populations': {arch: {'tests': tests, 'rules': tests} for arch in arches}}
    parts = []
    for arch in arches:
        for index, label in enumerate(tests):
            parts.append({'head': plan['head'], 'mode': mode,
                          'full_shard': {'architecture': arch, 'index': index},
                          'selected': [label], 'full_rules': [label], 'failed': [],
                          'verdict': 'pass', 'bazel_exit': 0, 'budget_warnings': [], 'inputs': []})
    return plan, parts


@pytest.mark.parametrize('mode', ['pr', 'main'])
def test_merge_retains_scope_and_budget_warnings_without_a_rerun(mode):
    remote = load('remote_gate')
    plan, parts = results(mode)
    warning = {'label': '//tests:a', 'nodeid': 'tests/a.py::one',
               'seconds': 1000, 'budget_seconds': 10}
    parts[0]['budget_warnings'] = [warning]
    merged = remote.merge_full_shards(plan, parts)
    if mode == 'pr':
        assert merged['verdict'] == 'pass' and merged['failed'] == []
        assert merged['budget_warnings'] == [warning]
        assert merged['scopes'] == [] and merged['all_scopes']
    else:
        assert merged['architectures']['amd64']['budget_warnings'] == [warning]
        assert all(row['verdict'] == 'pass' for row in merged['architectures'].values())


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'head', 'omitted-rule'])
def test_incomplete_or_mismatched_shard_evidence_fails_closed(mutation):
    remote = load('remote_gate')
    plan, parts = results()
    if mutation == 'missing':
        parts.pop()
    elif mutation == 'duplicate':
        parts[1] = parts[0]
    elif mutation == 'head':
        parts[1]['head'] = 'c' * 40
    else:
        parts[1]['full_rules'] = []
    with pytest.raises(remote.RemoteError):
        remote.merge_full_shards(plan, parts)


@pytest.mark.parametrize('mode', ['pr', 'main'])
def test_hard_timeout_or_failed_build_fails_the_combined_verdict(mode):
    remote = load('remote_gate')
    plan, parts = results(mode)
    parts[0].update(verdict='fail', failed=['//tests:a'], bazel_exit=3)
    merged = remote.merge_full_shards(plan, parts)
    if mode == 'pr':
        assert merged['verdict'] == 'fail' and merged['failed'] == ['//tests:a']
    else:
        assert merged['architectures']['amd64']['verdict'] == 'fail'
    plan, parts = results(mode)
    parts[0]['bazel_exit'] = 1
    merged = remote.merge_full_shards(plan, parts)
    assert merged['inputs']


@pytest.mark.parametrize('failed', [False, True])
def test_postmerge_runs_each_shard_once_and_waits_before_submitting_next(tmp_path, monkeypatch, failed):
    mode = 'main'
    remote = load('remote_gate')
    plan, parts = results(mode)
    if failed:
        parts[0].update(verdict='fail', failed=['//tests:a'], bazel_exit=3)
    queue = [dict(row, schema=remote.SCHEMA, base='a' * 40, error=None)
             for row in [plan, *parts]]
    events = []

    class Args:
        base, head, checkout, branch = 'a' * 40, 'b' * 40, '', 'master'
        repo, cpu, memory, timeout = remote.REPOSITORY, 4, '8GB', 55
        output_dir = tmp_path
    Args.mode = mode

    class Client:
        def call(self, method, request, **options):
            assert method == 'Run'
            index = sum(event[0] == 'submit' for event in events)
            assert index == 0 or events[-1] == ('complete', index - 1)
            events.append(('submit', index))
            return {'invocationId': str(index)}

    def terminal(client, invocation_id, deadline):
        index = int(invocation_id)
        events.append(('complete', index))
        return {'success': not (failed and index == 1), 'index': index}

    monkeypatch.setattr(remote, 'await_invocation', terminal)
    monkeypatch.setattr(remote, 'diagnostics', lambda *args: [])
    monkeypatch.setattr(remote, 'fetch', lambda client, invocation, name:
                        json.dumps(queue[invocation['index']]).encode())
    merged = remote.execute(Client(), Args)
    assert len(merged['invocations']) == 1 + len(parts)
    assert len(events) == 2 * len(queue)
    if mode == 'pr':
        assert merged['verdict'] == ('fail' if failed else 'pass')
    else:
        assert merged['architectures']['amd64']['verdict'] == ('fail' if failed else 'pass')
    assert merged['inputs'] == []


@pytest.mark.parametrize('committed', [False, True])
def test_runner_uses_pattern_files_and_retains_bazel_exit(tmp_path, monkeypatch, committed):
    gate = load('bazel_gate')
    shards = load('full_shards')
    shards.generate(tmp_path, XML)
    part = shards.validate(tmp_path, XML, 'amd64', 0)
    calls = []
    workspace = gate.Workspace(tmp_path)

    def run(*command, **options):
        calls.append(command)
        return 1

    monkeypatch.setattr(workspace, 'run', run)
    monkeypatch.setattr(workspace, 'budget_warnings', lambda targets: [])
    workspace.test(part['tests'], 'remote', {}, 'full-amd64-0',
                   extra=('--noremote_accept_cached', '--nocache_test_results'),
                   pattern_file=Path(part['path']) if committed else None)
    assert workspace.last_test_exit == 1
    assert len(calls) == 1
    command = calls[0]
    patterns = next(argument.split('=', 1)[1] for argument in command
                    if argument.startswith('--target_pattern_file='))
    if committed:
        assert patterns == part['path']
    else:
        assert Path(patterns).read_text().splitlines() == part['tests']
    assert '--jobs=16' in command
    assert '--noremote_accept_cached' in command and '--nocache_test_results' in command
    assert not any(argument.startswith('//') for argument in command)


def test_fallback_returns_a_plan_without_running_tests_or_builds():
    gate = load('bazel_gate')

    class Workspace:
        def ensure_commit(self, sha): pass
        def checkout(self, sha): pass
        def ensure_history(self, sha): pass
        def changes(self, base, head): return {'.bazelrc'}, set()
        def used_sources(self): return set()
        def tests(self, architecture): return {'//tests:a'}
        def inputs_check(self): return []
        def graph_impact(self, base): return None
        def full_plan(self, architectures):
            return {'amd64': [{'rules': ['//tests:a', '//src:library'], 'tests': ['//tests:a']}]}

    result = gate.gate_pr(Workspace(), 'base', 'head', {})
    assert result['selection'] == 'fallback' and result['all_scopes']
    assert result['full_populations']['amd64']['rules'] == ['//src:library', '//tests:a']
    assert 'verdict' not in result


def test_postmerge_without_both_architectures_is_never_a_pass():
    remote = load('remote_gate')
    row = {'schema': remote.SCHEMA, 'mode': 'main', 'head': 'b' * 40,
           'architectures': {'amd64': {'verdict': 'pass', 'failed': []}}}
    with pytest.raises(remote.RemoteError, match='complete post-merge'):
        remote.parse_result(json.dumps(row), row['head'])


@pytest.mark.parametrize('case', ['fallback', 'selected', 'postmerge'])
def test_only_gate_fallback_overlaps_and_never_exceeds_two_runs(tmp_path, monkeypatch, case):
    remote = load('remote_gate')
    mode = 'main' if case == 'postmerge' else 'pr'
    plan, parts = results(mode)
    if case == 'selected':
        plan.pop('full_populations')
        plan.update(selection='impacted', verdict='pass', failed=[], budget_warnings=[])
    records = {'plan': plan, **{f"{row['full_shard']['architecture']}:{row['full_shard']['index']}": row
                               for row in parts}}
    records = {key: dict(value, schema=remote.SCHEMA, base='a' * 40, error=None)
               for key, value in records.items()}
    lock = threading.Lock()
    overlap = threading.Barrier(2)
    state = {'active': 0, 'peak': 0, 'submissions': []}
    bound = 2 if case == 'fallback' else 1

    class Args:
        base, head, checkout, branch = 'a' * 40, 'b' * 40, '', 'master'
        repo, cpu, memory, timeout = remote.REPOSITORY, 4, '8GB', 55
        output_dir = tmp_path
    Args.mode = mode

    class Client:
        def call(self, method, request, **options):
            assert method == 'Run'
            command = shlex.split(request['steps'][0]['run'])
            coordinate = (command[command.index('--full-shard') + 1]
                          if '--full-shard' in command else 'plan')
            assert not any(argument.startswith('//') for argument in command)
            with lock:
                if coordinate != 'plan':
                    assert state['submissions'][0] == 'plan'
                state['submissions'].append(coordinate)
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
                assert state['active'] <= bound
            return {'invocationId': coordinate}

    def terminal(client, invocation_id, deadline):
        if case == 'fallback' and invocation_id != 'plan':
            # A serial controller cannot reach this rendezvous. No timing race
            # is needed to prove that both separate Runs overlap.
            overlap.wait(timeout=5)
        with lock:
            state['active'] -= 1
        return {'success': True, 'id': invocation_id}

    monkeypatch.setattr(remote, 'await_invocation', terminal)
    monkeypatch.setattr(remote, 'diagnostics', lambda *args: [])
    monkeypatch.setattr(remote, 'fetch', lambda client, invocation, name:
                        json.dumps(records[invocation['id']]).encode())
    result = remote.execute(Client(), Args)
    expected = ['plan'] if case == 'selected' else ['plan', *list(records)[1:]]
    assert sorted(state['submissions']) == sorted(expected)
    assert state['active'] == 0 and state['peak'] == bound
    assert len(result['invocations']) == len(expected)
