"""The Bazel build matches the source it is generated from and the gate it replaces (#1589).

tools/bazel/build_files.py writes every BUILD.bazel from Tessera's dependency
evidence, and tools/bazel/requirements.py writes pip.parse's lock from the
hosted gate's pins. A committed file that differs from what they write now is
stale. tools/ci/bazel_parity.py holds Bazel's test population to the hosted
gate's: every gate test file has one target per lane, and a target the cloud
gate leaves out says why.
"""
import importlib.util
from pathlib import Path

import pytest

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
