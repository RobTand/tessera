"""The Bazel build matches the source it is generated from and the gate it replaces (#1589).

tools/bazel/build_files.py writes every BUILD.bazel from Tessera's dependency
evidence, and tools/bazel/requirements.py writes pip.parse's lock from the
hosted gate's pins. A committed file that differs from what they write now is
stale. tools/ci/bazel_parity.py holds Bazel's test population to the hosted
gate's: every gate test file has one target per lane, and a target the cloud
gate leaves out says why.
"""
import importlib.util
import json
from pathlib import Path

import pytest
from accounting_source_support import isolated_prismaquant_imports

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def parity():
    return _load('bazel_parity', 'tools/ci/bazel_parity.py')


def test_the_generated_build_files_are_current():
    build_files = _load('build_files', 'tools/bazel/build_files.py')
    generated = build_files.generate(ROOT)
    stale = [key for key, text in generated.items()
             if build_files.output_path(ROOT, key).read_text() != text]
    assert stale == [], 'run tools/bazel/generate.sh'


def test_the_lock_export_names_the_gate_pins():
    requirements = _load('requirements', 'tools/bazel/requirements.py')
    source = requirements.source_text(ROOT)
    assert requirements.problems(source, (ROOT / requirements.EXPORT).read_text()) == []


def test_every_gate_file_has_a_target_in_each_lane(parity):
    assert parity.missing_targets(ROOT) == []


def test_every_target_the_cloud_gate_leaves_out_says_why(parity):
    assert parity.unexplained_exclusions(ROOT) == []


def test_every_table_entry_names_a_target_and_a_reason(parity):
    assert parity.table_problems(ROOT) == []


def test_a_stale_lock_export_is_named(tmp_path):
    requirements = _load('requirements', 'tools/bazel/requirements.py')
    source = 'pytest==8.4.2\n'
    export = requirements.header(source) + 'pytest==8.4.1 \\\n    --hash=sha256:00\n'
    assert any('pytest' in line for line in requirements.problems(source, export))


def test_a_target_left_out_without_a_reason_is_named(parity):
    build = (
        'ts_py_test(\n    name = "test_a",\n    path = "tests/test_a.py",\n    tags = ["manual"],\n)\n'
        'ts_py_test(\n    name = "test_a.bytes",\n    path = "tests/test_a.py",\n)\n'
    )
    targets = parity.parse_targets(build)
    assert parity.exclusions_without_reason(targets, nonhermetic={}, quarantine=set()) == ['//tests:test_a']
    assert parity.missing_lanes(targets, ['tests/test_a.py', 'tests/test_b.py']) == [
        'tests/test_b.py: no torch target //tests:test_b', 'tests/test_b.py: no bytes target //tests:test_b.bytes']


SECRETS = {'BUILDBUDDY_API_KEY': 'k1', 'MERGIFY_TOKEN': 'k2', 'GITHUB_TOKEN': 'k3',
           'SOME_SERVICE_API_KEY': 'k4', 'A_DEPLOY_TOKEN': 'k5'}


def test_the_bazel_runner_gives_pytest_no_secret(tmp_path):
    """A Remote Bazel runner holds BUILDBUDDY_API_KEY in its environment; a child
    that inherits it can print it into an uploaded log."""
    runtime = _load('runtime', 'tools/bazel/runtime.py')
    environment = runtime.gate_environment(dict(SECRETS, PATH='/bin'), tmp_path, tmp_path)
    assert sorted(set(SECRETS) & set(environment)) == []
    assert environment['PATH'] == '/bin'


def test_the_hosted_gate_gives_pytest_no_secret():
    gate_run = _load('gate_run', 'tools/ci/gate_run.py')
    environment = gate_run.without_secrets(dict(SECRETS, PATH='/bin'))
    assert sorted(set(SECRETS) & set(environment)) == []
    assert environment['PATH'] == '/bin'


def test_no_target_timeout_cuts_off_its_slowest_recorded_node():
    build_files = _load('build_files', 'tools/bazel/build_files.py')
    durations, quarantine = build_files.gate_tables(ROOT)
    budget = json.loads((ROOT / build_files.BUDGET).read_text())
    classes = {'moderate': 300, 'long': 900, 'eternal': 3600}
    mismatches = []
    for relative in build_files.repository_files(ROOT):
        if not build_files.is_test(relative):
            continue
        slowest = max([budget['seconds']] + [seconds for node, seconds in budget['slow'].items()
                                            if node.split('::', 1)[0] == relative])
        for lane in build_files.LANES:
            attributes = build_files.test_attributes(relative, lane, durations, quarantine, budget)
            if classes[attributes['timeout']] < slowest * 3:
                mismatches.append(f"{relative} ({lane}): {attributes['timeout']} < {slowest}s x 3")
    assert mismatches == []


def test_no_lane_installs_the_prismabuild_sdk():
    """No test imports prismabuild: neither gate requirements source pins it and
    the generated lane packages and hash-locked export do not name it."""
    for source in ('tools/ci/gate_requirements.txt', 'tools/ci/gate_torch_requirements.txt',
                   'tools/bazel/requirements.txt', 'tools/bazel/lanes.bzl'):
        text = (ROOT / source).read_text().lower()
        assert 'prismabuild' not in text, source


def test_prismaquant_is_pinned_and_hash_locked_for_the_tests_that_read_it():
    requirements = _load('requirements', 'tools/bazel/requirements.py')
    assert 'tools/ci/gate_prismaquant_requirements.txt' in requirements.SOURCES
    export = (ROOT / requirements.EXPORT).read_text()
    entry = export[export.index('\nprismaquant @ '):].split('\n', 3)
    assert '/archive/' in entry[1] and entry[2].strip().startswith('--hash=sha256:')


def test_the_runner_points_box_artifacts_at_an_installed_prismaquant(
        monkeypatch, tmp_path, isolated_prismaquant_imports):
    runtime = _load('runtime', 'tools/bazel/runtime.py')
    package = tmp_path / 'site' / 'prismaquant'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    monkeypatch.syspath_prepend(str(tmp_path / 'site'))
    environment = runtime.gate_environment({'PATH': '/bin'}, tmp_path, tmp_path)
    assert environment['TESSERA_PRISMAQUANT_DIR'] == str(tmp_path / 'site')
    explicit = runtime.gate_environment({'TESSERA_PRISMAQUANT_DIR': '/elsewhere'}, tmp_path, tmp_path)
    assert explicit['TESSERA_PRISMAQUANT_DIR'] == '/elsewhere'


@pytest.mark.parametrize('target,required_input', [
    ('//tests:test_bazel_gate', '//tools/bazel:BUILD.bazel'),
    ('//tests:test_seeded_breaks', '//tests:BUILD.bazel'),
])
def test_build_contract_readers_declare_the_files_they_open(target, required_input):
    import ast
    tree = ast.parse((ROOT / 'tools/bazel/test_inputs.bzl').read_text())
    table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(name, ast.Name) and name.id == 'TEST_INPUTS'
                         for name in node.targets))
    entry = next((value for key, value in zip(table.keys, table.values)
                  if ast.literal_eval(key) == target), None)
    data = [] if entry is None else next(ast.literal_eval(field.value)
                                        for field in entry.keywords if field.arg == 'data')
    assert required_input in data, f'{target} must carry its BUILD file read in both runfiles lanes'
