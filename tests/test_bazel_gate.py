"""The Bazel merge gate fails closed, keeps its check names and reruns only budgets (#1589).

tools/ci/bazel_gate.py runs on a BuildBuddy Remote Bazel runner: it selects the
tests a change can affect with bazel-diff, runs them, and writes
gate-result.json. tools/ci/remote_gate.py starts it from GitHub and turns the
result into the required checks. Any doubt about what to test fails the gate.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bazel_gate = _load('bazel_gate', 'tools/ci/bazel_gate.py')
remote_gate = _load('remote_gate', 'tools/ci/remote_gate.py')

TESTS = {'//tests:test_wire', '//tests:test_wire.bytes', '//tests:test_export'}
USED = {'src/tessera/wire.py', 'src/tessera/export.py', 'tests/test_wire.py', 'tests/fixtures/a.json',
        'tests/BUILD.bazel', 'src/tessera/BUILD.bazel', 'docs/ARCHITECTURE.md',
        '.github/workflows/ci.yml'}


def plan(changed, impacted=(), deleted=(), used=USED, tests=TESTS):
    return bazel_gate.gate_decision(set(changed), set(deleted), set(impacted), set(used), set(tests))


def test_impacted_tests_are_selected_and_packages_become_scopes():
    decision = plan(['src/tessera/wire.py'], ['//src/tessera:wire', '//tests:test_wire', '//tests:test_wire.bytes'])
    assert decision['selected'] == ['//tests:test_wire', '//tests:test_wire.bytes']
    assert decision['scopes'] == ['//src/tessera', '//tests']
    assert decision['build'] == ['//src/tessera:wire']
    assert not decision['all_scopes']


@pytest.mark.parametrize('path', ['MODULE.bazel', 'MODULE.bazel.lock', '.bazelrc', '.bazelversion',
                                  '.bazelignore', 'BUILD.bazel', 'tools/bazel/python.bzl',
                                  'tools/bazel/requirements.txt', '.github/workflows/gate.yml',
                                  'tools/ci/bazel_gate.py', 'tools/ci/remote_gate.py'])
def test_build_lock_and_gate_workflow_changes_test_everything(path):
    decision = plan([path, 'src/tessera/wire.py'])
    assert decision['selection'] == 'fallback'
    assert decision['selected'] == sorted(TESTS)
    assert decision['build'] == ['//...'] and decision['all_scopes']


def test_another_workflow_is_an_ordinary_input():
    """ci.yml is read by the tests that check it, so only they rerun."""
    decision = plan(['.github/workflows/ci.yml'], ['//tests:test_export'])
    assert decision['selection'] == 'impacted' and decision['selected'] == ['//tests:test_export']


def test_an_empty_diff_fails_closed():
    with pytest.raises(bazel_gate.GateError, match='empty diff'):
        plan([])


def test_an_unmapped_python_file_fails_naming_it_and_the_fix():
    with pytest.raises(bazel_gate.GateError) as failed:
        plan(['src/tessera/new_module.py'], ['//tests:test_wire'])
    assert 'src/tessera/new_module.py' in str(failed.value)
    assert 'tools/bazel/generate.sh' in str(failed.value)


def test_a_file_no_target_reads_selects_nothing():
    """An unread note is not a test input; the generated-input check still
    fails a stale BUILD file in the same run."""
    decision = plan(['docs/measurements/2026-11-02-note.md'])
    assert decision['selection'] == 'irrelevant' and decision['selected'] == []


def test_a_relevant_change_with_nothing_impacted_fails_closed():
    with pytest.raises(bazel_gate.GateError, match='no impacted targets'):
        plan(['src/tessera/wire.py'], [])


def test_a_build_file_edit_that_moves_no_rule_hash_tests_everything():
    decision = plan(['tests/BUILD.bazel'], [])
    assert decision['selection'] == 'fallback' and decision['selected'] == sorted(TESTS)


def test_every_selected_test_must_pass_and_a_missing_status_fails():
    assert bazel_gate.verdict_of(['//a', '//b'], {'//a': 'PASSED', '//b': 'PASSED'}) == ('pass', [])
    assert bazel_gate.verdict_of(['//a', '//b'], {'//a': 'PASSED'}) == ('fail', ['//b'])
    assert bazel_gate.verdict_of([], {}) == ('skipped', [])



@pytest.mark.parametrize('status', ['PASSED', 'FAILED', 'TIMEOUT'])
@pytest.mark.parametrize('mode', ['pr', 'main'])
def test_budget_warnings_preserve_the_first_verdict_without_rerunning(status, mode):
    warning = {'label': '//tests:test_wire', 'nodeid': 'tests/test_wire.py::test_one',
               'seconds': 10000, 'budget_seconds': 20}

    class Workspace(bazel_gate.Workspace):
        def __init__(self):
            self.calls = []
        def ensure_history(self): pass
        def ensure_commit(self, sha): pass
        def checkout(self, sha): pass
        def changes(self, base, head): return {'src/tessera/wire.py'}, set()
        def used_sources(self): return USED
        def bazel_diff(self, base, head): return {'//tests:test_wire'}
        def tests(self, architecture): return {'//tests:test_wire'}
        def inputs_check(self): return []
        def graph_impact(self, base): return None
        def build(self, *args): return True
        def test(self, targets, config, metadata, log_name, extra=()):
            self.calls.append(config)
            return {'//tests:test_wire': status}, [warning]

    workspace = Workspace()
    result = getattr(bazel_gate, 'gate_' + mode)(workspace, 'base', 'head', {})
    records = [result] if mode == 'pr' else list(result['architectures'].values())
    assert len(workspace.calls) == len(records)
    assert len(set(workspace.calls)) == len(records)
    for record in records:
        assert record['verdict'] == ('pass' if status == 'PASSED' else 'fail')
        assert record['budget_warnings'] == [warning]
        assert 'budget_reruns' not in record


def test_budget_warnings_collect_each_shard_without_changing_status(tmp_path):
    folder = tmp_path / 'bazel-testlogs/tests/test_wire/shard_1_of_2/test.outputs'
    folder.mkdir(parents=True)
    warning = {'nodeid': 'tests/test_wire.py::test_one', 'seconds': 10000, 'budget_seconds': 20}
    (folder / 'budget_warnings.json').write_text(json.dumps({'warnings': [warning]}))
    assert bazel_gate.Workspace(tmp_path).budget_warnings(['//tests:test_wire']) == [
        {'label': '//tests:test_wire', **warning}]


def test_remote_summary_reports_budget_warnings_on_pr_and_postmerge():
    warning = {'label': '//tests:test_wire', 'nodeid': 'tests/test_wire.py::test_one',
               'seconds': 10000, 'budget_seconds': 20}
    record = {'verdict': 'pass', 'failed': [], 'missed': [], 'budget_warnings': [warning]}
    for mode in ('pr', 'main'):
        result = dict(record, mode=mode, base='a'*40, head='b'*40,
                      architectures={'amd64': record})
        summary = remote_gate.explain(result, [])
        assert 'BUDGET_WARNING: tests/test_wire.py::test_one' in summary
        assert '10000.000s > 20.000s' in summary

def test_gate_children_get_no_secret_from_the_runner():
    environment = bazel_gate.child_environment({'BUILDBUDDY_API_KEY': 'k', 'MERGIFY_TOKEN': 't',
                                                'GITHUB_TOKEN': 'g', 'X_API_KEY': 'x', 'PATH': '/bin'})
    assert sorted(environment) == ['PATH']


def test_the_arm64_population_leaves_out_the_runner_lane():
    assert bazel_gate.population_query('arm64').endswith('except attr(tags, "\\brunner\\b", //...)')
    assert 'runner' not in bazel_gate.population_query('amd64')


def test_the_remote_run_targets_tessera_on_the_pinned_image():
    class Args:
        mode, base, head, checkout, branch = 'pr', 'a' * 40, 'b' * 40, '', 'feature'
        repo, cpu, memory, timeout = remote_gate.REPOSITORY, 4, '8GB', 45
    request = remote_gate.run_request(Args)
    assert request['repo'] == 'https://github.com/RobTand/tessera'
    assert request['env']['GIT_REPO_DEFAULT_BRANCH'] == 'master'
    assert request['platform_properties']['container-image'] == remote_gate.IMAGE
    assert 'tools/ci/bazel_gate.py pr --base ' + 'a' * 40 in request['steps'][-1]['run']
    assert 'BUILDBUDDY_API_KEY' not in json.dumps(request)


def test_the_required_checks_need_scopes_bazel_and_the_boundary():
    good = {'bazel': {'result': 'success', 'outputs': {'scopes_uploaded': 'true', 'tests': 'success'}},
            'boundary': {'result': 'success'}}
    remote_gate.aggregate(good)
    for broken in ({**good, 'boundary': {'result': 'failure'}},
                   {**good, 'bazel': {'result': 'success', 'outputs': {'scopes_uploaded': 'false', 'tests': 'success'}}},
                   {**good, 'bazel': {'result': 'success', 'outputs': {'scopes_uploaded': 'true', 'tests': 'failure'}}}):
        with pytest.raises(RuntimeError):
            remote_gate.aggregate(broken)


def test_the_workflow_keeps_its_check_names_refs_and_scopes():
    text = (ROOT / '.github/workflows/gate.yml').read_text()
    assert '    name: pr-gate\n' in text and '    name: gate\n' in text
    assert 'action: scopes-git-refs' in text and 'action: scopes-upload' in text
    assert 'branches: [master]' in text
    assert 'labeled' not in text
    assert 'check_no_secret_leak.py' in text or 'check-no-secret-leak' in text


def test_impacted_output_is_read_in_either_form():
    """bazel-diff v70.0.2 prints a JSON list with distances when it reads dependency edges."""
    assert bazel_gate.impacted_labels('[{"label": "@@//tests:test_wire", "targetDistance": 1}]') == \
        {'//tests:test_wire'}
    assert bazel_gate.impacted_labels('//tests:test_wire\n@//src/tessera:wire\n') == \
        {'//tests:test_wire', '//src/tessera:wire'}
    with pytest.raises(bazel_gate.GateError):
        bazel_gate.impacted_labels('[{"name": "x"}]')


def test_the_gate_pins_the_shared_bazel_diff():
    assert bazel_gate.BAZEL_DIFF_URL.endswith('/v70.0.2/bazel-diff-rust-linux-amd64')
    assert bazel_gate.BAZEL_DIFF_SHA256 == '44226a1d1d5239abc8bc6158de1f886fcbbf58e44479ec178df8c01e7e32102e'


@pytest.fixture
def artifact_gate(monkeypatch, tmp_path):
    output = tmp_path / 'artifacts'
    output.mkdir()
    root = tmp_path / 'workspace'
    root.mkdir()
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(output))
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'fake-runner-secret-for-artifact-tests')
    monkeypatch.setenv('UNRELATED_TOKEN', 'fake-unrelated-token-never-printed')
    monkeypatch.delenv('BBRUN_FINAL_SCAN', raising=False)
    monkeypatch.setattr(bazel_gate, 'gate_pr', lambda *args: {'verdict': 'pass', 'inputs': []})
    write_result = bazel_gate.write_result
    monkeypatch.setattr(bazel_gate, 'write_result', lambda result, directory: write_result(result, directory, root))

    def run():
        return bazel_gate.main(['pr', '--base', 'a' * 40, '--head', 'b' * 40], workspace=object())

    return root, output, run


def test_the_final_scan_covers_both_diagnostic_logs(monkeypatch, artifact_gate):
    root, output, run = artifact_gate
    names = ['pr-bazel-remote-grpc.log', 'pr-execution_log.binpb.zst']
    for name in names:
        (root / name).write_text('safe diagnostic')
    scans = []
    scan = bazel_gate.check_no_secret_leak.main

    def record(argv):
        scans.append(argv)
        return scan(argv)

    monkeypatch.setattr(bazel_gate.check_no_secret_leak, 'main', record)
    assert run() == 0
    assert all((output / name).is_file() for name in names)
    assert len(scans) == 1
    assert str(output) in scans[0]
    assert ['--env', 'BUILDBUDDY_API_KEY'] == scans[0][1:3]
    assert ['--require', 'BUILDBUDDY_API_KEY'] == scans[0][-2:]


@pytest.mark.parametrize('name', ['pr-bazel-remote-grpc.log', 'pr-execution_log.binpb.zst', 'summary.json'])
def test_a_secret_artifact_is_removed_and_fails_the_gate(name, artifact_gate, capsys):
    root, output, run = artifact_gate
    directory = root if name.startswith('pr-') else output
    (directory / name).write_text('fake-runner-secret-for-artifact-tests')
    assert run() == 1
    assert list(output.iterdir()) == []
    text = capsys.readouterr().out
    assert name in text
    assert 'fake-runner-secret-for-artifact-tests' not in text
    assert 'fake-unrelated-token-never-printed' not in text


@pytest.mark.parametrize('name', ['raw-bep.json', 'nested/events.binpb', 'command.log',
                                  'nested/bazel_command_log.txt'])
def test_bep_and_command_logs_are_deleted_even_without_a_secret(name, artifact_gate):
    _, output, run = artifact_gate
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('safe bytes in a forbidden artifact')
    assert run() == 1
    assert not path.exists()


@pytest.mark.parametrize('key', [None, ''])
def test_a_blind_gate_scan_fails_without_deleting_artifacts(key, monkeypatch, artifact_gate, capsys):
    _, output, run = artifact_gate
    if key is None:
        monkeypatch.delenv('BUILDBUDDY_API_KEY')
    else:
        monkeypatch.setenv('BUILDBUDDY_API_KEY', key)
    (output / 'keep.json').write_text('{}')
    assert run() == 1
    assert (output / 'keep.json').is_file()
    assert 'secret scan blind: key unavailable' in capsys.readouterr().out


def test_an_incomplete_scan_clears_artifacts_and_fails(artifact_gate):
    _, output, run = artifact_gate
    (output / 'broken.gz').write_bytes(b'\x1f\x8btruncated gzip')
    assert run() == 1
    assert list(output.iterdir()) == []


def test_a_failed_gate_still_scans_its_artifacts(monkeypatch, artifact_gate):
    _, output, run = artifact_gate

    def fail(*args):
        raise bazel_gate.GateError('controlled gate failure')

    monkeypatch.setattr(bazel_gate, 'gate_pr', fail)
    (output / 'summary.json').write_text('fake-runner-secret-for-artifact-tests')
    assert run() == 1
    assert list(output.iterdir()) == []


@pytest.mark.parametrize('program', ['git', 'bazel'])
def test_no_gate_child_inherits_the_scanner_key(program, monkeypatch, tmp_path):
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'fake-runner-secret-for-artifact-tests')
    monkeypatch.setenv('MERGIFY_TOKEN', 'fake-mergify-secret')
    received_names = []

    def execute(command, **options):
        environment = options.get('env')
        received_names.extend(name for name in ('BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN')
                              if environment is None or environment.get(name))
        return bazel_gate.subprocess.CompletedProcess(command, 0, stdout='ok')

    monkeypatch.setattr(bazel_gate.subprocess, 'run', execute)
    assert bazel_gate.Workspace(tmp_path).run(program, 'version') == 'ok'
    assert received_names == []


def test_hosted_scans_fail_closed_when_the_key_is_unavailable():
    text = (ROOT / '.github/workflows/gate.yml').read_text()
    scans = text.split('python3 tools/ci/check_no_secret_leak.py gate-output')[0:-1]
    assert len(scans) == 2
    for preceding in scans:
        assert 'secret scan blind: key unavailable' in preceding.rsplit('run: |', 1)[-1]
    assert text.count('id: artifact_scan') == 2
    assert text.count("if: always() && steps.artifact_scan.outcome == 'success'") == 2


def test_bbrun_marker_defers_the_gate_scan(monkeypatch, artifact_gate, capsys):
    _, output, run = artifact_gate
    monkeypatch.setenv('BBRUN_FINAL_SCAN', '1')
    monkeypatch.delenv('BUILDBUDDY_API_KEY')
    assert run() == 0
    assert (output / 'gate-result.json').is_file()
    assert 'secret scan deferred to bbrun final scan' in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize('marker', ['', '0', 'true', '01'])
def test_only_the_exact_bbrun_marker_defers_the_gate_scan(marker, monkeypatch, artifact_gate, capsys):
    _, _, run = artifact_gate
    monkeypatch.setenv('BBRUN_FINAL_SCAN', marker)
    monkeypatch.delenv('BUILDBUDDY_API_KEY')
    assert run() == 1
    assert 'secret scan blind: key unavailable' in capsys.readouterr().out


def test_the_remote_gate_does_not_defer_its_final_scan():
    class Args:
        mode, base, head, checkout, branch = 'pr', 'a' * 40, 'b' * 40, '', 'feature'
        repo, cpu, memory, timeout = remote_gate.REPOSITORY, 4, '8GB', 45

    assert 'BBRUN_FINAL_SCAN' not in json.dumps(remote_gate.run_request(Args))


@pytest.mark.parametrize('architecture,pool', [('amd64', 'selfhosted-amd64'),
                                               ('arm64', 'selfhosted-arm64')])
def test_both_action_platforms_use_selfhosted_capacity_without_fallback(architecture, pool):
    import re
    build = (ROOT / 'tools/bazel/BUILD.bazel').read_text()
    blocks = re.findall(r'platform\(.*?\n\)', build, re.S)
    block = next(block for block in blocks if f'name = "linux_{architecture}"' in block)
    assert f'"Pool": "{pool}"' in block
    assert '"use-self-hosted-executors": "true"' in block
    config = 'remote' if architecture == 'amd64' else 'remote-arm64'
    bazelrc = (ROOT / '.bazelrc').read_text()
    assert f'build:{config} --remote_local_fallback=false' in bazelrc
    assert f'build:{config} --platforms=//tools/bazel:linux_{architecture}' in bazelrc


@pytest.mark.parametrize('mode', ['pr', 'main'])
def test_gate_and_postmerge_keep_the_cloud_runner(mode):
    class Args:
        base, head, checkout, branch = 'a'*40, 'b'*40, '', 'feature'
        repo, cpu, memory, timeout = remote_gate.REPOSITORY, 4, '8GB', 45
    Args.mode = mode
    properties = remote_gate.run_request(Args)['platform_properties']
    assert properties['workload-isolation-type'] == 'firecracker'
    assert 'Pool' not in properties and 'use-self-hosted-executors' not in properties


@pytest.mark.parametrize('pytest_status,failure', [(0, ''), (1, '<failure>assertion</failure>'),
                                                 (1, '<error>hard timeout</error>')])
def test_python_runner_warns_without_overriding_assertions_or_timeouts(
        tmp_path, monkeypatch, pytest_status, failure):
    import sys
    from types import SimpleNamespace
    runner = _load('python_runner', 'tools/bazel/python_runner.py')
    real_gate = _load('runner_gate_run', 'tools/ci/gate_run.py')
    path = 'tests/test_wire.py'
    warning = {'nodeid': path+'::test_one', 'seconds': 10000, 'budget_seconds': 20}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('TS_BAZEL_LANE', raising=False)
    monkeypatch.delenv('TS_TRACE', raising=False)
    monkeypatch.delenv('TEST_TOTAL_SHARDS', raising=False)
    monkeypatch.setenv('TEST_TMPDIR', str(tmp_path))
    outputs = tmp_path / 'outputs'
    monkeypatch.setenv('TEST_UNDECLARED_OUTPUTS_DIR', str(outputs))
    monkeypatch.setenv('XML_OUTPUT_FILE', str(tmp_path/'junit.xml'))
    monkeypatch.setattr(sys, 'argv', ['python_runner.py', path, 'torch', '--budget={}'])
    monkeypatch.setattr(runner, 'prepare_remote_test', lambda root: (tmp_path/'scratch', {}))
    monkeypatch.setattr(runner, 'budget', lambda *args: (
        real_gate, {'seconds': 20, 'slow': {}, 'enforce': True}))
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        (tmp_path/'surface.json').write_text(json.dumps({'not_collected': []}))
        (tmp_path/'junit.xml').write_text(
            '<testsuite><testcase file="tests/test_wire.py" classname="tests.test_wire" '
            f'name="test_one" time="10000">{failure}</testcase></testsuite>')
        return SimpleNamespace(returncode=pytest_status)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    assert runner.main() == pytest_status
    assert len(calls) == 1
    assert json.loads((outputs/'budget_warnings.json').read_text())['warnings'] == [warning]
    assert not (outputs/'budget_overrun.json').exists()
    if pytest_status == 0:
        assert json.loads((outputs/'inventory.json').read_text())['budget_warnings'] == [warning]
