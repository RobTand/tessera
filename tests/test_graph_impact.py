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
    baseline = json.loads((ROOT / tessera.BASELINE).read_text())
    assert tessera.core.regressions(head, baseline) == []


def test_the_baseline_records_every_gain(tessera, head):
    baseline = json.loads((ROOT / tessera.BASELINE).read_text())
    assert tessera.core.stale_entries(head, baseline) == []


def test_every_test_is_in_its_bazel_package(head):
    assert set(head['test_packages'].values()) == {'tests'}


def test_conftest_reach_is_a_ratcheted_edge_group(head):
    """A new source module in conftest's reach puts it under every test."""
    group = head['edge_groups']['conftest_source']
    assert group and all(edge.startswith('tests/conftest.py -> ') for edge in group)


def test_the_change_does_not_deepen_the_graph(tessera, capsys):
    """base -> head, with the merge base GRAPH_IMPACT_BASE or origin/master names."""
    base = tessera.merge_base(ROOT, os.environ.get('GRAPH_IMPACT_BASE'))
    if base is None:
        pytest.skip('no merge base in this checkout; the baseline figures above still hold')
    code = tessera.check(ROOT, base)
    summary = capsys.readouterr().out
    print(summary)
    assert summary.startswith('U ')
    assert code == 0
