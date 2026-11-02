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


def test_only_a_budget_overrun_or_a_timeout_is_rerun():
    statuses = {'//a': 'FAILED', '//b': 'FAILED', '//c': 'TIMEOUT', '//d': 'PASSED'}
    assert bazel_gate.budget_rerun_labels(statuses, overruns={'//a'}) == ['//a', '//c']


def test_a_budget_rerun_passes_once_and_is_recorded():
    statuses = {'//a': 'FAILED', '//c': 'TIMEOUT', '//d': 'PASSED'}
    merged, reruns = bazel_gate.apply_budget_reruns(statuses, ['//a', '//c'], {'//a': 'PASSED', '//c': 'FAILED'})
    assert merged == {'//a': 'PASSED', '//c': 'FAILED', '//d': 'PASSED'}
    assert reruns == [{'label': '//a', 'first': 'FAILED', 'rerun': 'PASSED'},
                      {'label': '//c', 'first': 'TIMEOUT', 'rerun': 'FAILED'}]


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
