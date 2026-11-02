"""The graph-impact check: does this change deepen the test graph? (#1588)

tools/ci/graph_impact.py is the core, fed `tests: {test: reach}`,
`edges: {(from, to)}` and `loc`. Until Tessera's BUILD files are on master,
tools/ci/graph_impact_tessera.py bridges to it from the evidence Bazel's BUILD
generator reads, and adds Python's own rules; afterwards the source is
bazel-diff's deps.json (#1589). The check compares the merge base's graph with
the head's and needs no run history; tests/fixtures/graph_impact_baseline.json
keeps the best figures.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def core():
    return _load('graph_impact', 'tools/ci/graph_impact.py')


@pytest.fixture(scope='module')
def tessera():
    return _load('graph_impact_tessera', 'tools/ci/graph_impact_tessera.py')


@pytest.fixture(scope='module')
def head(tessera):
    return tessera.measure(ROOT)


def test_no_figure_regresses_past_the_baseline(tessera, head):
    assert tessera.regressions(head, tessera.load_baseline()) == []


def test_the_baseline_records_every_gain(tessera, head):
    assert tessera.stale_entries(head, tessera.load_baseline()) == []


def test_the_change_does_not_deepen_the_graph(tessera, head):
    """base -> head, with the merge base GRAPH_IMPACT_BASE or origin/master names."""
    base = tessera.merge_base(ROOT, os.environ.get('GRAPH_IMPACT_BASE'))
    if base is None:
        pytest.skip('no merge base in this checkout; the baseline figures above still hold')
    result = tessera.check(ROOT, base, head)
    print('graph-impact: ' + result['summary'])
    assert result['failures'] == []


def test_a_new_test_over_the_reach_budget_prints_its_blame_path(core):
    edges = {('tests/test_new.py', 'src/a.py'), ('src/a.py', 'src/hub.py'), ('src/hub.py', 'src/b.py'),
             ('tests/test_old.py', 'src/hub.py')}
    head = core.metrics({'tests/test_new.py': {'src/a.py', 'src/hub.py', 'src/b.py'},
                         'tests/test_old.py': {'src/hub.py', 'src/b.py'}}, edges)
    base = core.metrics({'tests/test_old.py': {'src/hub.py', 'src/b.py'}},
                        {('tests/test_old.py', 'src/hub.py'), ('src/hub.py', 'src/b.py')})
    failures = core.compare(base, head, changed=['tests/test_new.py'], budget=1, allowlist={})
    assert any('tests/test_new.py reaches 3' in line and 'tests/test_new.py -> src/a.py -> src/hub.py' in line
               for line in failures)


def test_an_allowlisted_test_may_reach_past_the_budget(core):
    old = {'tests/test_old.py': {'src/a.py', 'src/b.py'}}
    old_edges = {('tests/test_old.py', 'src/a.py'), ('tests/test_old.py', 'src/b.py')}
    head = core.metrics(dict(old, **{'tests/test_new.py': {'src/a.py', 'src/b.py'}}),
                        old_edges | {('tests/test_new.py', 'src/a.py'), ('tests/test_new.py', 'src/b.py')})
    base = core.metrics(old, old_edges)
    assert core.compare(base, head, changed=['tests/test_new.py'], budget=1,
                        allowlist={'tests/test_new.py': 'integration-wide by design'}) == []


def test_the_allowlist_grows_only_in_a_change_of_its_own(core):
    grown = {'tests/test_new.py': 'integration-wide by design'}
    path = 'tests/fixtures/graph_impact_allowlist.json'
    assert core.allowlist_problems({}, grown, [path], path) == []
    assert core.allowlist_problems({}, grown, [path, 'src/a.py'], path)
    assert core.allowlist_problems({}, {'tests/test_new.py': ''}, [path], path)


def test_a_grown_cycle_names_the_edge_that_closed_it(core):
    base_edges = {('src/a.py', 'src/b.py'), ('src/b.py', 'src/c.py'), ('tests/test_t.py', 'src/a.py')}
    reach = {'tests/test_t.py': {'src/a.py', 'src/b.py', 'src/c.py'}}
    base = core.metrics(reach, base_edges)
    head = core.metrics(reach, base_edges | {('src/c.py', 'src/a.py')})
    failures = core.compare(base, head, changed=['src/c.py'], budget=10, allowlist={})
    assert any('cycle grew to 3' in line and 'src/c.py -> src/a.py' in line for line in failures)


def test_a_worse_uniform_selection_names_the_new_edge(core):
    tests = [f'tests/test_{n}.py' for n in range(10)]
    loc = {'src/leaf.py': 10, 'src/m1.py': 10, 'src/m2.py': 10, 'src/m3.py': 10}
    base = core.metrics({test: {'src/leaf.py'} for test in tests}, {(test, 'src/leaf.py') for test in tests}, loc)
    head_edges = {(test, 'src/leaf.py') for test in tests} | {('src/leaf.py', f'src/m{n}.py') for n in (1, 2, 3)}
    head = core.metrics({test: set(loc) for test in tests}, head_edges, loc)
    failures = core.compare(base, head, changed=['src/leaf.py'], budget=100, allowlist={})
    assert any('U rose from 25.0% to 100.0%' in line and 'src/leaf.py -> src/m1.py' in line for line in failures)


def test_an_added_test_to_test_import_is_named(tessera):
    base = {'test_to_test_imports': [], 'conftest_source_modules': []}
    head = {'test_to_test_imports': ['tests/test_x.py -> tests/test_y.py'], 'conftest_source_modules': []}
    assert tessera.python_rules(base, head) == ['new test-to-test import: tests/test_x.py -> tests/test_y.py']


def test_the_summary_line_reads_like_the_spec(core):
    reach = {'tests/test_x.py': {'src/a.py'}}
    figures = core.metrics(reach, {('tests/test_x.py', 'src/a.py')})
    assert core.summary(figures, figures, changed=[], budget=96) == \
        'U 100.0% -> 100.0%, SCC max 1 -> 1, new tests 0, new edges 0, none to a hub'


def test_writing_the_baseline_keeps_the_best_figure(core):
    baseline = {'uniform_selection_pct': 20.0, 'uniform_selection_loc_pct': 20.0, 'largest_cycle': 50,
                'reach_budget': 100}
    live = dict(baseline, uniform_selection_pct=19.0, largest_cycle=52, reach_p90=90)
    record = core.baseline_record(live, baseline)
    assert record['uniform_selection_pct'] == 19.0
    assert record['largest_cycle'] == 50
    assert record['reach_budget'] == 90
