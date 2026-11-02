#!/usr/bin/env python3
"""Tessera's graph-impact check: its Python-import source, rules and baseline (#1588).

tools/ci/graph_impact.py is the shared core: it reads only
`tests: {test: source files}`, `edges: {(from, to)}` and `loc`. This adapter
feeds it from the evidence Bazel's BUILD generator reads (tools/bazel/
evidence.py, tools/bazel/graph.py). A test's reach is the source files in its
Bazel closure: its imports, transitively, tests/conftest.py's, the programs it
launches and the files it reads. Two rules are Python's own and live here: no
new test-to-test import, and no new source module in conftest's reach.

    tools/ci/graph_impact_tessera.py                 figures for the working tree
    tools/ci/graph_impact_tessera.py check [--base]  base -> head; exit 1 on a failure
    tools/ci/graph_impact_tessera.py --replay N      what the last N merges select
    tools/ci/graph_impact_tessera.py --write-baseline

tests/test_graph_impact.py holds tests/fixtures/graph_impact_baseline.json
and runs the check against the merge base (GRAPH_IMPACT_BASE, or the merge
base with origin/master). An integration-wide test goes in
tests/fixtures/graph_impact_allowlist.json with its reason.
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
FIX = 'run tools/ci/graph_impact_tessera.py --write-baseline'
LISTS = ('conftest_source_modules', 'test_to_test_imports')
sys.path.insert(0, str(ROOT / 'tools/bazel'))
import evidence  # noqa: E402
import graph  # noqa: E402


def _core():
    # The core is vendored byte-identical across repositories; load it by path.
    spec = importlib.util.spec_from_file_location('graph_impact', Path(__file__).with_name('graph_impact.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = _core()


def inputs(built: graph.Graph):
    """(tests, edges, loc) in the core's contract, from Tessera's dependency graph."""
    sources = {module for module in built.evidence.modules if not evidence.is_test_side(module)}
    closures = {test: built.closure(test) for test in built.tests}
    tests = {test: closures[test] & sources for test in built.tests}
    edges = {(source, target) for source, targets in built.edges.items() for target in targets}
    # Every test loads tests/conftest.py, so a blame path may run through it.
    edges |= {(test, graph.CONFTEST) for test in built.tests}
    loc = {module: (built.evidence.root / module).read_bytes().count(b'\n') for module in sources}
    return tests, edges, loc, closures


def measure(root: Path, files=None) -> dict:
    """The core's figures plus Tessera's Python-only lists."""
    built = graph.Graph(evidence.Evidence(root, files))
    tests, edges, loc, closures = inputs(built)
    figures = core.metrics(tests, edges, loc)
    figures['conftest_source_modules'] = sorted(module for module in built.reach(graph.CONFTEST)
                                                if not evidence.is_test_side(module))
    figures['test_to_test_imports'] = sorted(f'{importer} -> {imported}' for importer in built.evidence.modules
                                             if evidence.is_test_side(importer)
                                             for imported in built.edges[importer] if evidence.is_test(imported))
    figures['graph'] = built
    figures['selectors'] = graph.selectors(closures)
    return figures


def python_rules(base, head):
    found = []
    for name, label in (('test_to_test_imports', 'test-to-test import'),
                        ('conftest_source_modules', 'source module conftest reaches')):
        added = sorted(set(head[name]) - set(base[name]))
        if len(added) == 1:
            found.append(f'new {label}: {added[0]}')
        elif added:
            found.append(f'{len(added)} new {label}s: ' + ', '.join(added[:12]) + (', ...' if len(added) > 12 else ''))
    return found


def load_baseline(root=ROOT):
    return json.loads((root / BASELINE).read_text())


def regressions(live, baseline):
    return core.regressions(live, baseline) + [f'new in {name}: {entry}' for name in LISTS
                                               for entry in sorted(set(live[name]) - set(baseline[name]))]


def stale_entries(live, baseline):
    return core.stale_entries(live, baseline, FIX) + [f'gone from {name}: {entry}; {FIX}' for name in LISTS
                                                     for entry in sorted(set(baseline[name]) - set(live[name]))]


def baseline_record(live, baseline=None):
    record = core.baseline_record(live, baseline)
    record.update({name: live[name] for name in LISTS})
    record['evidence'] = 'tools/bazel/evidence.py'
    return {key: record[key] for key in sorted(record)}


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
    """The figures of a commit's tree, read from git without a checkout."""
    files = [path for path in _git(root, 'ls-tree', '-r', '--name-only', commit).splitlines()
             if path and not path.startswith(evidence.SKIPPED) and Path(path).name != 'BUILD.bazel']
    archive = subprocess.run(['git', 'archive', '--format=tar', commit], cwd=root, capture_output=True,
                             check=True).stdout
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(temporary, filter='data')
        tree = Path(temporary)
        return measure(tree, [path for path in files if (tree / path).is_file() and not (tree / path).is_symlink()])


def check(root, base, head=None):
    """{'failures': [...], 'summary': '...'} for the change from `base` to the working tree."""
    head = head or measure(root)
    changed = _git(root, 'diff', '--name-only', '--no-renames', base).split()
    allowlist = json.loads((root / ALLOWLIST).read_text()) if (root / ALLOWLIST).exists() else {}
    try:
        previous = json.loads(_git(root, 'show', f'{base}:{ALLOWLIST}'))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        previous = {}
    budget = load_baseline(root)['reach_budget']
    base_figures = measure_commit(root, base)
    failures = core.compare(base_figures, head, changed, budget, allowlist)
    failures += python_rules(base_figures, head)
    failures += core.allowlist_problems(previous, allowlist, changed, ALLOWLIST)
    return {'failures': failures, 'summary': core.summary(base_figures, head, changed, budget)}


def replay(root, live, count):
    """The fraction of the gate each of the last `count` merges selects, on today's graph."""
    merges = _git(root, 'log', '--first-parent', '--merges', '--format=%H', '-n', str(count), 'HEAD').split()
    rows = []
    for sha in merges:
        changed = _git(root, 'diff', '--name-only', '--no-renames', f'{sha}^1', sha).split()
        if any(graph.is_global(path) for path in changed):
            selected, why = live['population'], 'global'
        else:
            selected = len(set().union(set(), *(live['selectors'].get(path, set()) for path in changed)))
            why = 'graph'
        rows.append({'merge': sha, 'selected': selected, 'fraction': selected / live['population'], 'why': why,
                     'drivers': sorted(((len(live['selectors'].get(path, ())), path) for path in changed),
                                       reverse=True)[:3]})
    fractions = [row['fraction'] for row in rows]
    return {'merges': len(rows), 'mean_pct': round(100 * statistics.mean(fractions), 1) if rows else 0.0,
            'p50_pct': round(100 * core.nearest_rank(fractions, 0.5), 1),
            'p90_pct': round(100 * core.nearest_rank(fractions, 0.9), 1), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=['report', 'check'], default='report')
    parser.add_argument('--base', help='the commit to compare with; default: the merge base with master')
    parser.add_argument('--write-baseline', action='store_true')
    parser.add_argument('--replay', type=int, metavar='N')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args(argv)
    live = measure(ROOT)
    if args.command == 'check':
        base = merge_base(ROOT, args.base)
        if base is None:
            sys.exit('graph-impact: no merge base; pass --base')
        result = check(ROOT, base, live)
        print('graph-impact: ' + result['summary'])
        for failure in result['failures']:
            print('graph-impact: FAIL ' + failure)
        return 1 if result['failures'] else 0
    print(f"U {live['uniform_selection_pct']}% (LOC-weighted {live['uniform_selection_loc_pct']}%), "
          f"largest cycle {live['largest_cycle']}, test reach median {live['reach_median']} "
          f"p90 {live['reach_p90']} max {live['reach_max']}; conftest reaches "
          f"{len(live['conftest_source_modules'])} source modules; "
          f"{len(live['test_to_test_imports'])} test-to-test imports")
    for module, count in sorted(live['fanout'].items(), key=lambda item: -item[1])[:args.top]:
        print(f"  {100 * count / live['population']:5.1f}%  {module}")
    if args.replay:
        result = replay(ROOT, live, args.replay)
        print(f"replay of {result['merges']} merges: mean {result['mean_pct']}%, p50 {result['p50_pct']}%, "
              f"p90 {result['p90_pct']}%")
    if args.write_baseline:
        path = ROOT / BASELINE
        previous = load_baseline() if path.exists() else None
        if previous and 'reach_budget' not in previous:
            previous = None  # Figures from before the graph-impact check: start the record.
        path.write_text(json.dumps(baseline_record(live, previous), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
