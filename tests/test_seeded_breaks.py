"""A seeded break counts as caught only when an expected target fails inside the selection (#1589)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/ci'))
SPEC = importlib.util.spec_from_file_location('seeded_breaks', ROOT / 'tools/ci/seeded_breaks.py')
seeded_breaks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeded_breaks)


def test_an_expected_failure_inside_the_selection_is_caught():
    row = seeded_breaks.seed_verdict(['//tests:test_wire', '//tests:test_wire.bytes'],
                                     {'//tests:test_wire': 'FAILED', '//tests:test_wire.bytes': 'PASSED'},
                                     ['//tests:test_wire'])
    assert row == {'impacted_tests': 2, 'failing_selected': ['//tests:test_wire'], 'caught': True,
                   'example': '//tests:test_wire'}


def test_an_unexpected_failure_or_an_unselected_test_is_not():
    assert not seeded_breaks.seed_verdict(['//tests:test_pool'], {'//tests:test_pool': 'FAILED'},
                                          ['//tests:test_wire'])['caught']
    assert not seeded_breaks.seed_verdict([], {}, ['//tests:test_wire'])['caught']


def test_seeds_cover_at_least_eight_places_and_name_existing_files():
    assert len({place for place, *_ in seeded_breaks.SEEDS}) >= 8
    for _, path, _, expected in seeded_breaks.SEEDS:
        assert (ROOT / path).is_file(), path
        assert expected and all(label.startswith('//tests:test_') for label in expected)


@pytest.fixture
def seed_main(monkeypatch, tmp_path):
    class Workspace:
        root = tmp_path

        def checkout(self, base):
            assert base == 'a' * 40

    output = tmp_path / 'artifacts'
    output.mkdir()
    monkeypatch.setenv('BUILDBUDDY_ARTIFACTS_DIRECTORY', str(output))
    monkeypatch.setenv('BUILDBUDDY_API_KEY', 'fake-seed-runner-secret')
    monkeypatch.setenv('UNRELATED_TOKEN', 'fake-unrelated-seed-token')
    monkeypatch.delenv('BBRUN_FINAL_SCAN', raising=False)
    monkeypatch.setattr(seeded_breaks.bazel_gate, 'Workspace', Workspace)
    monkeypatch.setattr(seeded_breaks.subprocess, 'check_output', lambda *args, **kwargs: 'a' * 40 + '\n')
    places = []

    def run_seed(workspace, base, place, path, mutation, expected):
        places.append(place)
        return {'place': place, 'path': path, 'impacted_tests': 1, 'failing_selected': [expected[0]],
                'caught': True, 'example': expected[0]}

    monkeypatch.setattr(seeded_breaks, 'run_seed', run_seed)
    monkeypatch.setattr(sys, 'argv', ['seeded_breaks.py', '--base', 'a' * 40])
    return output, places


def test_the_seed_step_scans_its_actual_artifact_directory(monkeypatch, seed_main):
    output, _ = seed_main
    scans = []
    scan = seeded_breaks.bazel_gate.check_no_secret_leak.main

    def record(argv):
        scans.append(argv)
        return scan(argv)

    monkeypatch.setattr(seeded_breaks.bazel_gate.check_no_secret_leak, 'main', record)
    assert seeded_breaks.main() == 0
    assert len(scans) == 1
    assert str(output) in scans[0]
    assert ['--env', 'BUILDBUDDY_API_KEY'] == scans[0][1:3]
    assert ['--require', 'BUILDBUDDY_API_KEY'] == scans[0][-2:]


def test_a_seed_step_secret_log_is_deleted_and_fails(seed_main, capsys):
    output, _ = seed_main
    (output / 'seed-bazel-remote-grpc.log').write_text('fake-seed-runner-secret')
    assert seeded_breaks.main() == 1
    assert list(output.iterdir()) == []
    text = capsys.readouterr().out
    assert 'seed-bazel-remote-grpc.log' in text
    assert 'fake-seed-runner-secret' not in text
    assert 'fake-unrelated-seed-token' not in text


@pytest.mark.parametrize('key', [None, ''])
def test_a_blind_seed_scan_fails_without_deleting_artifacts(key, monkeypatch, seed_main, capsys):
    output, _ = seed_main
    if key is None:
        monkeypatch.delenv('BUILDBUDDY_API_KEY')
    else:
        monkeypatch.setenv('BUILDBUDDY_API_KEY', key)
    assert seeded_breaks.main() == 1
    assert (output / 'seeded-breaks.json').is_file()
    assert 'secret scan blind: key unavailable' in capsys.readouterr().out


def test_the_seed_step_rejects_bep_artifacts(seed_main):
    output, _ = seed_main
    path = output / 'seed-bep.json'
    path.write_text('{}')
    assert seeded_breaks.main() == 1
    assert not path.exists()


def test_bbrun_marker_defers_the_seed_scan(monkeypatch, seed_main, capsys):
    output, _ = seed_main
    monkeypatch.setenv('BBRUN_FINAL_SCAN', '1')
    monkeypatch.delenv('BUILDBUDDY_API_KEY')
    assert seeded_breaks.main() == 0
    assert (output / 'seeded-breaks.json').is_file()
    assert 'secret scan deferred to bbrun final scan' in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize('marker', ['', '0', 'true', '01'])
def test_only_the_exact_bbrun_marker_defers_the_seed_scan(marker, monkeypatch, seed_main, capsys):
    monkeypatch.setenv('BBRUN_FINAL_SCAN', marker)
    monkeypatch.delenv('BUILDBUDDY_API_KEY')
    assert seeded_breaks.main() == 1
    assert 'secret scan blind: key unavailable' in capsys.readouterr().out


@pytest.mark.parametrize('selector', ['0,7,10', 'library module,JSON fixture,gate tooling',
                                     '0,JSON fixture,10'])
def test_the_cli_runs_only_requested_seeds(selector, monkeypatch, seed_main):
    output, places = seed_main
    monkeypatch.setattr(sys, 'argv', ['seeded_breaks.py', '--seeds', selector])
    assert seeded_breaks.main() == 0
    assert places == ['library module', 'JSON fixture', 'gate tooling']
    import json
    rows = json.loads((output / 'seeded-breaks.json').read_text())['rows']
    assert [row['place'] for row in rows] == places


def test_omitting_the_selector_runs_all_seeds(seed_main):
    _, places = seed_main
    assert seeded_breaks.main() == 0
    assert places == [place for place, *_ in seeded_breaks.SEEDS]


def test_seed_selection_deduplicates_and_preserves_declared_order(monkeypatch, seed_main):
    _, places = seed_main
    monkeypatch.setattr(sys, 'argv', ['seeded_breaks.py', '--seeds', '10,0,library module,7'])
    assert seeded_breaks.main() == 0
    assert places == ['library module', 'JSON fixture', 'gate tooling']


@pytest.mark.parametrize('selector', ['', '11', '-1', 'unknown place', '0,,1'])
def test_an_invalid_selector_is_refused_before_running_any_seed(selector, monkeypatch, seed_main):
    _, places = seed_main
    monkeypatch.setattr(sys, 'argv', ['seeded_breaks.py', '--seeds', selector])
    with pytest.raises(SystemExit) as error:
        seeded_breaks.main()
    assert error.value.code == 2
    assert places == []
