"""The graph-impact check: does this change deepen the test graph? (#1588)

tools/ci/graph_impact.py compares the static dependency graph at the merge
base with the graph at the head, from the evidence Bazel's BUILD generator
reads (tools/bazel/evidence.py). It needs no run history. It fails when a new
or changed test reaches more source files than the budget, when a test-to-test
or conftest source import is added, when the uniform-change selection U
worsens past tolerance or the largest import cycle grows, and when the
baseline (tests/fixtures/graph_impact_baseline.json) no longer records a gain.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location('graph_impact', ROOT / 'tools/ci/graph_impact.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def impact():
    return _load()


@pytest.fixture(scope='module')
def head(impact):
    return impact.metrics(impact.snapshot(ROOT))


def test_no_figure_regresses_past_the_baseline(impact, head):
    assert impact.regressions(head, impact.load_baseline()) == []


def test_the_baseline_records_every_gain(impact, head):
    assert impact.stale_entries(head, impact.load_baseline()) == []


def test_the_change_does_not_deepen_the_graph(impact, head):
    """base -> head, with the merge base GRAPH_IMPACT_BASE or origin/master names."""
    base = impact.merge_base(ROOT, os.environ.get('GRAPH_IMPACT_BASE'))
    if base is None:
        pytest.skip('no merge base in this checkout; the baseline figures above still hold')
    result = impact.check(ROOT, base, head)
    print(result['summary'])
    assert result['failures'] == []


def _graph(impact, edges, tests, conftest=()):
    """A synthetic metrics record: modules a..z under src, tests under tests/."""
    return impact.metrics_from_edges(edges, tests, conftest=list(conftest))


def test_a_new_test_over_the_reach_budget_prints_its_blame_path(impact):
    edges = {'tests/test_new.py': {'src/a.py'}, 'src/a.py': {'src/hub.py'}, 'src/hub.py': {'src/b.py'},
             'tests/test_old.py': {'src/b.py'}}
    head = _graph(impact, edges, ['tests/test_new.py', 'tests/test_old.py'])
    base = _graph(impact, {'tests/test_old.py': {'src/b.py'}}, ['tests/test_old.py'])
    failures = impact.compare(base, head, changed=['tests/test_new.py'], budget=1, allowlist={})
    message = '\n'.join(failures)
    assert 'tests/test_new.py' in message
    assert 'tests/test_new.py -> src/a.py -> src/hub.py' in message


def test_an_allowlisted_test_may_reach_past_the_budget(impact):
    edges = {'tests/test_new.py': {'src/a.py', 'src/b.py'}}
    head = _graph(impact, edges, ['tests/test_new.py'])
    base = _graph(impact, {}, [])
    assert impact.compare(base, head, changed=['tests/test_new.py'], budget=1,
                          allowlist={'tests/test_new.py': 'integration-wide by design'}) == []


def test_an_added_test_to_test_import_is_named(impact):
    head = _graph(impact, {'tests/test_x.py': {'tests/test_y.py'}}, ['tests/test_x.py', 'tests/test_y.py'])
    base = _graph(impact, {}, ['tests/test_x.py', 'tests/test_y.py'])
    failures = impact.compare(base, head, changed=['tests/test_x.py'], budget=10, allowlist={})
    assert any('tests/test_x.py -> tests/test_y.py' in line for line in failures)


def test_a_grown_cycle_names_the_edge_that_closed_it(impact):
    base_edges = {'src/a.py': {'src/b.py'}, 'src/b.py': {'src/c.py'}, 'tests/test_t.py': {'src/c.py'}}
    head_edges = dict(base_edges, **{'src/c.py': {'src/a.py'}})
    base = _graph(impact, base_edges, ['tests/test_t.py'])
    head = _graph(impact, head_edges, ['tests/test_t.py'])
    failures = impact.compare(base, head, changed=['src/c.py'], budget=10, allowlist={})
    assert any('src/c.py -> src/a.py' in line and 'cycle' in line for line in failures)


def test_a_worse_uniform_selection_names_the_new_edge(impact):
    tests = [f'tests/test_{n}.py' for n in range(10)]
    base_edges = {test: {'src/leaf.py'} for test in tests}
    base_edges.update({'src/m1.py': set(), 'src/m2.py': set(), 'src/m3.py': set()})
    head_edges = dict(base_edges, **{'src/leaf.py': {'src/m1.py', 'src/m2.py', 'src/m3.py'}})
    base = _graph(impact, base_edges, tests)
    head = _graph(impact, head_edges, tests)
    failures = impact.compare(base, head, changed=['src/leaf.py'], budget=100, allowlist={})
    assert any('src/leaf.py -> src/m1.py' in line for line in failures)


def test_the_summary_line_reads_like_the_spec(impact):
    head = _graph(impact, {'tests/test_x.py': {'src/a.py'}}, ['tests/test_x.py'])
    base = _graph(impact, {'tests/test_x.py': {'src/a.py'}}, ['tests/test_x.py'])
    line = impact.summary(base, head, changed=[], budget=96)
    assert line.startswith('U 100.0% -> 100.0%, SCC max 1 -> 1, new tests 0')


def test_writing_the_baseline_keeps_the_best_figure(impact):
    baseline = {'uniform_selection_pct': 20.0, 'uniform_selection_loc_pct': 20.0, 'largest_cycle': 50,
                'reach_budget': 100, 'conftest_source_modules': [], 'test_to_test_imports': []}
    live = dict(baseline, uniform_selection_pct=19.0, largest_cycle=52, reach_budget=90)
    record = impact.baseline_record(live, baseline)
    assert record['uniform_selection_pct'] == 19.0
    assert record['largest_cycle'] == 50
    assert record['reach_budget'] == 90
