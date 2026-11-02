"""The graph-impact check: does this change deepen the test graph? (#1588)

tools/ci/graph_impact.py is the shared core, vendored byte-identical from
PrismaQuant; tools/ci/graph_impact.sha256 records the copy. Until Tessera's
BUILD files are on master, tools/ci/graph_impact_tessera.py feeds it from the
evidence Bazel's BUILD generator reads; afterwards from bazel-diff's deps.json
(#1589). The check compares the merge base's graph with the head's and needs
no run history; tests/fixtures/graph_impact_baseline.json keeps the figures
it ratchets.
"""
import hashlib
import importlib.util
import json
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
def tessera():
    return _load('graph_impact_tessera', 'tools/ci/graph_impact_tessera.py')


@pytest.fixture(scope='module')
def head(tessera):
    return tessera.measure_tree(ROOT)


def test_the_vendored_core_is_the_recorded_copy():
    recorded = (ROOT / 'tools/ci/graph_impact.sha256').read_text().split()[0]
    assert hashlib.sha256((ROOT / 'tools/ci/graph_impact.py').read_bytes()).hexdigest() == recorded


def test_no_figure_regresses_past_the_baseline(tessera, head):
    baseline = tessera.comparison_baseline(json.loads((ROOT / tessera.BASELINE).read_text()), head, head)
    assert tessera.core.regressions(head, baseline) == []


def test_the_baseline_records_every_gain(tessera, head):
    baseline = tessera.comparison_baseline(json.loads((ROOT / tessera.BASELINE).read_text()), head, head)
    assert tessera.core.stale_entries(head, baseline) == []


def test_every_test_is_in_its_bazel_package(head):
    assert set(head['test_packages'].values()) == {'tests'}


def test_conftest_reach_is_a_ratcheted_edge_group(head):
    """A new source module in conftest's reach puts it under every test."""
    group = head['edge_groups']['conftest_source']
    assert group and all(edge.startswith('tests/conftest.py -> ') for edge in group)


def test_the_change_does_not_deepen_the_graph(tessera, head, capsys):
    """base -> head, with the merge base GRAPH_IMPACT_BASE or origin/master names."""
    base = tessera.merge_base(ROOT, os.environ.get('GRAPH_IMPACT_BASE'))
    if base is None:
        pytest.skip('no merge base in this checkout; the baseline figures above still hold')
    code = tessera.check(ROOT, base, head)
    summary = capsys.readouterr().out
    print(summary)
    assert summary.startswith('U ')
    assert code == 0


#: Coordinator, 2026-11-02: Tessera's test->test edges are judged in the core's
#: transitional `reach` mode until the helper cleanup (#1588) lands, and switch
#: to `strict` by this date at the latest. Date only; no count trigger.
STRICT_BY = '2026-11-16'


def test_test_to_test_edges_are_in_reach_mode_until_the_switch_date(tessera):
    import datetime
    baseline = json.loads((ROOT / tessera.BASELINE).read_text())
    if baseline.get('test_to_test', 'strict') == 'strict':
        return
    assert baseline['test_to_test'] == 'reach'
    assert baseline['test_to_test_strict_by'] == STRICT_BY
    assert datetime.date.today() < datetime.date.fromisoformat(STRICT_BY), (
        f'test_to_test is still reach after {STRICT_BY}: set it to strict in {tessera.BASELINE}')


def test_the_baseline_records_reach_mode_with_its_switch_date(tessera):
    baseline = json.loads((ROOT / tessera.BASELINE).read_text())
    assert baseline.get('test_to_test') == 'reach' and baseline.get('test_to_test_strict_by') == STRICT_BY


def test_the_baseline_stores_no_edge_list(tessera):
    """New edges are judged by the base revision's graph against the head's. An
    edge list stored in the baseline goes stale when master gains edges through
    other merges (TL3 #464), and then edges already on master read as new."""
    baseline = json.loads((ROOT / tessera.BASELINE).read_text())
    assert 'edge_groups' not in baseline


def test_new_edges_are_judged_against_the_base_revision(tessera):
    core = tessera.core
    base = core.measure({'tests/test_a.py': {'src/x.py'}}, {('tests/test_a.py', 'src/x.py')})
    head = core.measure({'tests/test_a.py': {'src/x.py'}, 'tests/test_b.py': {'src/x.py'}},
                        {('tests/test_a.py', 'src/x.py'), ('tests/test_b.py', 'tests/test_a.py')})
    base['edge_groups']['conftest_source'] = ['tests/conftest.py -> src/x.py']
    head['edge_groups']['conftest_source'] = ['tests/conftest.py -> src/x.py', 'tests/conftest.py -> src/y.py']
    stored = {'schema': core.SCHEMA, 'figures': {}, 'tolerances': {}, 'reach_budget': 10}
    compared = tessera.comparison_baseline(stored, base, head)
    assert sorted(core.regressions(head, compared)) == ['new conftest_source edge: tests/conftest.py -> src/y.py',
                                                        'new test_to_test edge: tests/test_b.py -> tests/test_a.py']
    assert core.stale_entries(head, compared) == []


_LOADS_A_TOOL = '''import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a():
    assert _load('helper').X
'''


def test_a_relative_root_reads_the_same_graph(tessera, tmp_path, monkeypatch):
    """A path the code resolves (Path(__file__).resolve()) is absolute; the
    evidence must place it under a relative root as under an absolute one."""
    repo = tmp_path / 'repo'
    (repo / 'tests').mkdir(parents=True)
    (repo / 'tools').mkdir()
    (repo / 'pyproject.toml').write_text('[project]\nname = "x"\n')
    (repo / 'tools/helper.py').write_text('X = 1\n')
    (repo / 'tests/test_a.py').write_text(_LOADS_A_TOOL)
    files = ['pyproject.toml', 'tools/helper.py', 'tests/test_a.py']
    monkeypatch.chdir(tmp_path)
    edges = [tessera.graph.Graph(tessera.evidence.Evidence(root, files)).edges.get('tests/test_a.py')
             for root in (Path('repo'), repo)]
    assert edges == [{'tools/helper.py'}, {'tools/helper.py'}]
