#!/usr/bin/env python3
"""Tessera's graph-impact check: the adapter that feeds the shared core (#1588).

tools/ci/graph_impact.py is the shared core, vendored byte-identical from
PrismaQuant (tools/ci/graph_impact.sha256 records the copy). This adapter
owns what the core leaves to each repository: discovering the graph, git, and
Tessera's own rules.

- **The graph,** until Tessera's BUILD files are on master: the evidence
  Bazel's BUILD generator reads (tools/bazel/evidence.py, tools/bazel/
  graph.py). A test's reach is the source files in its Bazel closure: its
  imports, transitively, tests/conftest.py's, the programs it launches and
  the files it reads. Afterwards, bazel-diff's deps.json replaces it (#1589).
- **Packages:** every test file is in the `tests` Bazel package.
- **conftest_source,** an edge group the core ratchets: a new source module
  in conftest's reach puts it under every test.

    tools/ci/graph_impact_tessera.py                    figures for the working tree
    tools/ci/graph_impact_tessera.py check [--base SHA] base -> head; exit 1 on a failure
    tools/ci/graph_impact_tessera.py --write-baseline   record gains (refuses losses)
    tools/ci/graph_impact_tessera.py --replay N         what the last N merges select

An exception goes in tests/fixtures/graph_impact_allowlist.json under the
core's exact diagnostic key, with its reason.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'tests/fixtures/graph_impact_baseline.json'
ALLOWLIST = 'tests/fixtures/graph_impact_allowlist.json'
TEST_PACKAGE = 'tests'
sys.path.insert(0, str(ROOT / 'tools/bazel'))
import evidence  # noqa: E402
import graph  # noqa: E402


def _core():
    # Vendored byte-identical across repositories; loaded by path.
    spec = importlib.util.spec_from_file_location('graph_impact', Path(__file__).with_name('graph_impact.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _core()


def inputs(built: graph.Graph):
    """(tests, edges, loc, closures) in the core's input contract."""
    sources = {module for module in built.evidence.modules if not evidence.is_test_side(module)}
    closures = {test: built.closure(test) for test in built.tests}
    tests = {test: closures[test] & sources for test in built.tests}
    edges = {(source, target) for source, targets in built.edges.items() for target in targets}
    # Every test loads tests/conftest.py, so a blame path may run through it.
    edges |= {(test, graph.CONFTEST) for test in built.tests}
    loc = {module: (built.evidence.root / module).read_bytes().count(b'\n') for module in sources}
    return tests, edges, loc, closures


def measure_tree(root: Path, files=None) -> dict:
    """The core's measurement of a tree, with Tessera's packages and edge groups."""
    built = graph.Graph(evidence.Evidence(root, files))
    tests, edges, loc, closures = inputs(built)
    figures = core.measure(tests, edges, loc, test_packages=dict.fromkeys(tests, TEST_PACKAGE))
    figures['edge_groups']['conftest_source'] = sorted(
        f'{graph.CONFTEST} -> {module}' for module in built.reach(graph.CONFTEST)
        if not evidence.is_test_side(module))
    figures['selectors'] = {file: sorted(owners) for file, owners in graph.selectors(closures).items()}
    return figures


def _git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True).stdout


def merge_base(root, explicit=None):
    """The commit to compare with: `explicit`, or the merge base with origin/master or master."""
    try:
        if explicit:
            return _git(root, 'rev-parse', '--verify', explicit + '^{commit}').strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    for branch in ('origin/master', 'master'):
        try:
            return _git(root, 'merge-base', 'HEAD', branch).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def measure_commit(root, commit):
    """The measurement of a commit's tree, read from git without a checkout."""
    files = [path for path in _git(root, 'ls-tree', '-r', '--name-only', commit).splitlines()
             if path and not path.startswith(evidence.SKIPPED) and Path(path).name != 'BUILD.bazel']
    archive = subprocess.run(['git', 'archive', '--format=tar', commit], cwd=root, capture_output=True,
                             check=True).stdout
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(temporary, filter='data')
        tree = Path(temporary)
        return measure_tree(tree, [path for path in files if (tree / path).is_file()
                                   and not (tree / path).is_symlink()])


def comparison_baseline(stored, base, head):
    """The stored baseline, with each edge group taken from the base revision's
    graph (as far as the head still has it). New edges are then exactly the
    head's minus the base's, and gone edges are no failure. The baseline file
    holds no edge list: one recorded at some commit goes stale as soon as master
    gains edges through other merges (TL3 #464)."""
    groups = {group: sorted(set(base.get('edge_groups', {}).get(group, ())) & set(rows))
              for group, rows in head.get('edge_groups', {}).items()}
    return dict(stored, edge_groups=groups)


def stored_record(record):
    """What the baseline file keeps: figures, budget, tolerances, the
    test->test mode, count and switch date. Never an edge list."""
    return {key: value for key, value in record.items() if key != 'edge_groups'}


def _at(root, commit, path):
    try:
        return json.loads(_git(root, 'show', f'{commit}:{path}'))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def check(root, base_commit, head=None, *, write_baseline=False):
    """The core's check of the change from `base_commit` to the working tree; its exit code."""
    head = head or measure_tree(root)
    base = measure_commit(root, base_commit)
    changed = set(_git(root, 'diff', '--name-only', '--no-renames', base_commit).split())
    allowlist = root / ALLOWLIST
    stored = json.loads((root / BASELINE).read_text())
    compared = comparison_baseline(stored, base, head)
    previous = _at(root, base_commit, BASELINE)
    previous = (dict(previous, edge_groups=compared['edge_groups'])
                if previous and previous.get('schema') == core.SCHEMA else None)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'graph_impact_baseline.json'
        path.write_text(json.dumps(compared))
        code = core.check(
            base, head, path, allowlist if allowlist.exists() else None,
            changed_tests=sorted(changed & head['reach'].keys()),
            base_allowlist=_at(root, base_commit, ALLOWLIST) or {},
            previous_baseline=previous, write_baseline=write_baseline,
            test_packages=head['test_packages'],
            base_package_max_reach=core.package_max_reach(base),
            edge_diff=core.diff_edges(base, head))
        if write_baseline and code == 0:
            (root / BASELINE).write_text(json.dumps(stored_record(json.loads(path.read_text())), indent=2) + '\n')
    return code


def replay(root, live, count):
    """The fraction of the gate each of the last `count` merges selects, on today's graph."""
    merges = _git(root, 'log', '--first-parent', '--merges', '--format=%H', '-n', str(count), 'HEAD').split()
    population = live['test_count']
    rows = []
    for sha in merges:
        changed = _git(root, 'diff', '--name-only', '--no-renames', f'{sha}^1', sha).split()
        if any(graph.is_global(path) for path in changed):
            selected, why = population, 'global'
        else:
            selected = len(set().union(set(), *(set(live['selectors'].get(path, ())) for path in changed)))
            why = 'graph'
        rows.append({'merge': sha, 'selected': selected, 'fraction': selected / population, 'why': why})
    fractions = sorted(row['fraction'] for row in rows)
    rank = lambda q: fractions[max(0, -(-len(fractions) * q // 1) - 1)] if fractions else 0.0  # noqa: E731
    return {'merges': len(rows), 'mean_pct': round(100 * statistics.mean(fractions), 1) if rows else 0.0,
            'p50_pct': round(100 * rank(0.5), 1), 'p90_pct': round(100 * rank(0.9), 1), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=['report', 'check'], default='report')
    parser.add_argument('--base', help='the commit to compare with; default: the merge base with master')
    parser.add_argument('--write-baseline', action='store_true')
    parser.add_argument('--replay', type=int, metavar='N')
    args = parser.parse_args(argv)
    live = measure_tree(ROOT)
    if args.command == 'check' or args.write_baseline and (ROOT / BASELINE).exists():
        base = merge_base(ROOT, args.base)
        if base is None:
            sys.exit('graph-impact: no merge base; pass --base')
        return check(ROOT, base, live, write_baseline=args.write_baseline)
    print(f"U {live['u_pct']:.1f}% (LOC-weighted {live['loc_u_pct']:.1f}%), largest cycle {live['scc_max']}, "
          f"test reach median {live['reach_median']} p90 {live['reach_p90']} max {live['reach_max']}; "
          f"conftest reaches {len(live['edge_groups']['conftest_source'])} source modules; "
          f"{len(live['edge_groups']['test_to_test'])} test-to-test imports")
    if args.replay:
        result = replay(ROOT, live, args.replay)
        print(f"replay of {result['merges']} merges: mean {result['mean_pct']}%, p50 {result['p50_pct']}%, "
              f"p90 {result['p90_pct']}%")
    if args.write_baseline:
        (ROOT / BASELINE).write_text(json.dumps(stored_record(core.baseline_record(live)), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
