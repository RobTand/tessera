"""The fan-out ratchet: a change to one source module selects only the tests that use it.

tools/ci/import_fanout.py measures, from the dependency evidence Bazel's
BUILD generator reads (tools/bazel/evidence.py), how many gate tests depend on
each source module. tests/fixtures/import_fanout_baseline.json holds the best
figures reached so far (#1588). A regression past a figure's tolerance fails,
and so does a baseline that no longer records a gain.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location('import_fanout', ROOT / 'tools/ci/import_fanout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def fanout():
    return _load()


@pytest.fixture(scope='module')
def live(fanout):
    return fanout.measure(ROOT)


def test_no_figure_regresses_past_the_baseline(fanout, live):
    baseline = json.loads(fanout.BASELINE.read_text())
    assert fanout.regressions(live, baseline) == []


def test_the_baseline_records_every_gain(fanout, live):
    baseline = json.loads(fanout.BASELINE.read_text())
    assert fanout.stale_entries(live, baseline) == []


def test_a_worse_figure_is_a_regression(fanout):
    baseline = {'median_selected_pct': 2.0, 'p90_selected_pct': 50.0, 'modules_over_half': 10,
                'conftest_source_modules': ['src/tessera/a.py'], 'test_to_test_imports': []}
    worse = dict(baseline, p90_selected_pct=60.0, conftest_source_modules=['src/tessera/a.py', 'src/tessera/b.py'],
                 test_to_test_imports=['tests/test_x.py -> tests/test_y.py'])
    found = fanout.regressions(worse, baseline)
    assert any('p90_selected_pct' in line for line in found)
    assert any('src/tessera/b.py' in line for line in found)
    assert any('tests/test_x.py -> tests/test_y.py' in line for line in found)


def test_a_gain_past_tolerance_makes_the_baseline_stale(fanout):
    baseline = {'median_selected_pct': 2.0, 'p90_selected_pct': 50.0, 'modules_over_half': 10,
                'conftest_source_modules': ['src/tessera/a.py', 'src/tessera/b.py'], 'test_to_test_imports': []}
    better = dict(baseline, p90_selected_pct=40.0, conftest_source_modules=['src/tessera/a.py'])
    found = fanout.stale_entries(better, baseline)
    assert any('p90_selected_pct' in line for line in found)
    assert any('src/tessera/b.py' in line for line in found)


def test_writing_the_baseline_keeps_the_best_figure(fanout):
    baseline = {'median_selected_pct': 2.0, 'p90_selected_pct': 50.0, 'modules_over_half': 10,
                'conftest_source_modules': [], 'test_to_test_imports': []}
    mixed = dict(baseline, median_selected_pct=1.5, p90_selected_pct=50.4)
    record = fanout.baseline_record(mixed, baseline)
    assert record['median_selected_pct'] == 1.5
    assert record['p90_selected_pct'] == 50.0
