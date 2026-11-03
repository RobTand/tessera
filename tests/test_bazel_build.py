"""The Bazel build matches the source it is generated from and the gate it replaces (#1589).

tools/bazel/build_files.py writes every BUILD.bazel from Tessera's dependency
evidence, and tools/bazel/requirements.py writes pip.parse's lock from the
hosted gate's pins. A committed file that differs from what they write now is
stale. tools/ci/bazel_parity.py holds Bazel's test population to the hosted
gate's: every gate test file has one target per lane, and a target the cloud
gate leaves out says why.
"""
import ast
from functools import lru_cache
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


def test_both_lanes_install_the_prismabuild_sdk():
    """The SDK is stdlib-only, so the no-torch lane runs its consumers too
    instead of skipping them (Rob, 2026-11-02: install the library a test needs)."""
    text = (ROOT / 'tools/ci/gate_requirements.txt').read_text()
    assert any(line.startswith('prismabuild @ ') for line in text.splitlines())


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


def _shared_inputs():
    """Follow the generated targets the shared pytest macro actually declares."""
    constants = {}
    for name in ('lanes.bzl', 'python.bzl'):
        for statement in ast.parse((ROOT / 'tools/bazel' / name).read_text()).body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    if target.id in {'_TEST_DEPS', '_TEST_DATA'}:
                        constants[target.id] = [constants[value.id] if isinstance(value, ast.Name)
                                                 else ast.literal_eval(value) for value in statement.value.elts]
                    elif target.id == 'CONFTEST':
                        constants[target.id] = ast.literal_eval(statement.value)

    @lru_cache(None)
    def rules(package):
        found = {}
        for statement in ast.parse((ROOT / package / 'BUILD.bazel').read_text()).body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            attributes = {keyword.arg: keyword.value for keyword in statement.value.keywords}
            if 'name' in attributes:
                found[ast.literal_eval(attributes['name'])] = attributes
        return found

    files, pending, visited = set(), constants['_TEST_DEPS'] + constants['_TEST_DATA'], set()
    while pending:
        label = pending.pop()
        if label in visited or label.startswith('@'):
            continue
        visited.add(label)
        package, name = label.removeprefix('//').split(':', 1)
        if (ROOT / package / name).is_file():
            files.add((Path(package) / name).as_posix())
            continue
        attributes = rules(package)[name]
        for kind in ('srcs', 'data', 'deps'):
            expression = attributes.get(kind)
            if expression is None:
                continue
            if isinstance(expression, ast.Call):
                assert isinstance(expression.func, ast.Name) and expression.func.id == 'glob'
                patterns = ast.literal_eval(expression.args[0])
                excluded = next((ast.literal_eval(keyword.value) for keyword in expression.keywords
                                 if keyword.arg == 'exclude'), [])
                matches = {path for pattern in patterns for path in (ROOT / package).glob(pattern)
                           if path.is_file()}
                matches -= {path for pattern in excluded for path in (ROOT / package).glob(pattern)}
                files.update(path.relative_to(ROOT).as_posix() for path in matches)
                continue
            for dependency in ast.literal_eval(expression):
                pending.append(dependency if dependency.startswith(('//', '@'))
                               else f'//{package}:{dependency.removeprefix(":")}')
    return files


def test_a_budget_edit_changes_only_its_test_action():
    build_files = _load('build_files', 'tools/bazel/build_files.py')
    left, right = 'tests/test_left.py', 'tests/test_right.py'
    before = {'seconds': 20, 'slow': {left + '::test_slow': 40}, 'enforce': True}
    after = dict(before, slow={left + '::test_slow': 30})
    for lane in build_files.LANES:
        assert build_files.test_attributes(left, lane, {}, {}, before) != build_files.test_attributes(
            left, lane, {}, {}, after)
        assert build_files.test_attributes(right, lane, {}, {}, before) == build_files.test_attributes(
            right, lane, {}, {}, after)
    assert 'tools/ci/gate_budget.json' not in _shared_inputs(), (
        'the complete budget table is a shared test input; it overrides per-file --budget isolation')


@pytest.mark.parametrize('exit_code', [0, 1])
def test_the_bazel_runner_retains_its_population_after_pytest(monkeypatch, tmp_path, exit_code):
    """The downloadable receipt survives both a passing and a failing child."""
    from types import SimpleNamespace

    runner = _load('bazel_python_runner', 'tools/bazel/python_runner.py')
    private, outputs, scratch = (tmp_path / name for name in ('private', 'outputs', 'scratch'))
    private.mkdir()
    scratch.mkdir()
    path = 'tests/test_fixture.py'
    population = {
        'role': 'population', 'cuda': False, 'device': 'fixture CPU',
        'cuda_surface': {'executed': 0},
        'counts': {'passed': int(exit_code == 0), 'failed': int(exit_code != 0), 'skipped': 0},
        'not_collected': [], 'skip_reasons': {},
    }
    monkeypatch.chdir(tmp_path)
    for name in ('TS_BAZEL_LANE', 'TS_TRACE', 'TEST_TOTAL_SHARDS', 'XML_OUTPUT_FILE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('TEST_TMPDIR', str(private))
    monkeypatch.setenv('TEST_UNDECLARED_OUTPUTS_DIR', str(outputs))
    monkeypatch.setattr(runner.sys, 'argv', ['runner', path, 'bytes', '--budget={}'])
    monkeypatch.setattr(runner, 'prepare_remote_test', lambda root: (scratch, {}))
    rules = SimpleNamespace(BUDGET_ENFORCEMENT_TOLERANCE=1.5, timings=lambda *a, **kw: [])
    monkeypatch.setattr(runner, 'budget', lambda *args: (rules, {'slow': {}}))

    def child(command, **kwargs):
        surface = Path(next(arg.removeprefix('--surface-json=') for arg in command
                            if arg.startswith('--surface-json=')))
        surface.write_text(json.dumps(population))
        junit = Path(next(arg.removeprefix('--junitxml=') for arg in command
                          if arg.startswith('--junitxml=')))
        failure = '<failure message="fixture failure"/>' if exit_code else ''
        junit.write_text('<testsuites><testsuite><testcase file="' + path +
                         '" classname="tests.test_fixture" name="test_case">' + failure +
                         '</testcase></testsuite></testsuites>')
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(runner.subprocess, 'run', child)
    assert runner.main() == exit_code
    assert not scratch.exists()
    assert json.loads((outputs / 'surface.json').read_text()) == population


def test_only_verified_executor_metadata_is_left_out_of_bazel_inputs(tmp_path, monkeypatch):
    import hashlib
    import shlex
    import subprocess
    import sys

    from tessera._dev.suite_source import measured_source

    build_files = _load('build_files', 'tools/bazel/build_files.py')
    root = tmp_path / 'snapshot'
    root.mkdir()
    for directory in ('experiments', 'docs', 'tests', 'tools/bazel', 'tools/ci', 'src/tessera/serving'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name, text in {
        'pyproject.toml': '', 'tests/conftest.py': '',
        'tools/bazel/test_inputs.bzl': '["//:files/docs"]',
        'tools/bazel/helper.txt': 'ordinary tool input',
        'tools/ci/gate_durations.json': '{"lanes": {}}',
        'tools/ci/gate_quarantine.json': '{"lanes": {}}',
        'tools/ci/gate_budget.json': '{"seconds": 1, "slow": {}}',
        'tools/ci/gate_flaky.json': '[]',
        'tools/ci/gate_requirements.txt': '', 'tools/ci/gate_torch_requirements.txt': '',
        'src/tessera/serving/runtime_contract.json': '{}',
        'docs/report.txt': 'ordinary source', 'docs/other.txt': 'another source',
    }.items():
        (root / name).write_text(text)
    reader = 'experiments/reader.py'
    (root / reader).write_text(
        'from pathlib import Path\n'
        'root = Path(__file__).resolve().parents[1]\n'
        'records = list(root.glob("*.json"))\n'
        'first_record = records[0].read_text()\n')
    generated = 'executor.metadata.json'
    nested = 'docs/executor.metadata.json'
    tool_metadata = 'tools/bazel/executor.metadata.json'
    ordinary = '.pbrun-closure.ffeeddccbbaa9988.json'
    for name in (generated, nested, tool_metadata, ordinary):
        (root / name).write_text('{}')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'snapshot'], cwd=root, check=True)
    verifier = tmp_path / 'verifier.py'
    entries = [{'path': name, 'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
               for name in (generated, nested, tool_metadata)]
    verifier.write_text('import json\nprint(json.dumps(' + repr({'generated': entries}) + '))\n')
    monkeypatch.setenv('TESSERA_SOURCE_VERIFIER', shlex.join([sys.executable, str(verifier)]))
    assert measured_source(root)['verification'] == 'verified'
    listed = build_files.repository_files(root)
    assert generated not in listed, 'verified executor metadata was treated as repository source'
    assert ordinary in listed, 'a closure-shaped basename is not proof of executor ownership'
    assert nested not in listed and tool_metadata not in listed
    raw_roster = [*listed, generated, nested, tool_metadata]
    for files in (None, raw_roster):
        evidence = build_files.Evidence(root, files=files)
        assert generated not in evidence.files
        assert generated not in evidence.facts[reader].paths
        assert generated not in evidence.data(reader), 'a resolved glob reintroduced executor metadata'
        assert ordinary in evidence.data(reader)
    rendered = build_files.generate(root)
    groups = [node for node in ast.parse(rendered['']).body
              if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == 'filegroup']
    docs_group = next(node.value for node in groups
                      if ast.literal_eval(next(arg.value for arg in node.value.keywords
                                               if arg.arg == 'name')) == 'files/docs')
    srcs = next(arg.value for arg in docs_group.keywords if arg.arg == 'srcs')
    if isinstance(srcs, ast.Call):
        patterns = ast.literal_eval(srcs.args[0])
        inputs = {str(path.relative_to(root)) for pattern in patterns
                  for path in root.glob(pattern + '/*' if pattern.endswith('/**') else pattern)
                  if path.is_file()}
    else:
        inputs = set(ast.literal_eval(srcs))
    assert nested not in inputs, 'generated filegroup reintroduced verified executor metadata'
    assert {'docs/report.txt', 'docs/other.txt'} <= inputs
    tool_rosters = [node.value for node in ast.parse(rendered['tools/bazel/lanes.bzl']).body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'BAZEL_TOOL_FILES'
                            for target in node.targets)]
    assert tool_rosters, 'handwritten tools/bazel filegroup lacks a verified source roster'
    tool_inputs = ast.literal_eval(tool_rosters[0])
    assert 'executor.metadata.json' not in tool_inputs
    assert 'helper.txt' in tool_inputs
    handwritten = ast.parse((ROOT / 'tools/bazel/BUILD.bazel').read_text())
    tool_group = next(node.value for node in handwritten.body
                      if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Name) and node.value.func.id == 'filegroup')
    tool_srcs = next(arg.value for arg in tool_group.keywords if arg.arg == 'srcs')
    assert isinstance(tool_srcs, ast.Name) and tool_srcs.id == 'BAZEL_TOOL_FILES'
    for name in (generated, nested, tool_metadata):
        (root / name).unlink()
    assert build_files.generate(root) == rendered, 'executor metadata changed generated BUILD text'
    for name in (generated, nested, tool_metadata):
        (root / name).write_text('{}')
    entries[0]['sha256'] = '0' * 64
    verifier.write_text('import json\nprint(json.dumps(' + repr({'generated': entries}) + '))\n')
    assert measured_source(root)['verification'] == 'unknown'
    assert generated in build_files.repository_files(root), 'unverified metadata is an ordinary input'
    assert generated in build_files.Evidence(root).data(reader)
    monkeypatch.delenv('TESSERA_SOURCE_VERIFIER')
    assert generated in build_files.repository_files(root), 'no verifier proves no metadata exclusion'
