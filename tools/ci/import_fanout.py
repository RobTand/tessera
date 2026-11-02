#!/usr/bin/env python3
"""How many gate tests a change to one source module selects (#1588).

Bazel reruns a test when any file its target holds changes: the test file,
every module it imports transitively, the modules tests/conftest.py imports,
and the data each reads. tools/bazel/evidence.py and tools/bazel/graph.py
derive those closures from the source, as Bazel's BUILD generator will. For
each source module (a non-test Python file under src/tessera, tools or
experiments), this counts the gate tests (tests/test_*.py) whose closure holds
it, as a fraction of all of them. A module half the gate depends on selects
half the gate whenever it changes.

tests/test_import_fanout.py holds the live figures against
tests/fixtures/import_fanout_baseline.json. A summary figure worse than the
baseline by more than its tolerance fails, as does a new source module in
conftest's reach or a new test-to-test import. So does a baseline that no
longer records the current state: a gain past a tolerance, or an entry that has
gone. --write-baseline records the gains and keeps the best of each figure.

--replay N reports the fraction of the gate that each of the last N merged
pull requests on master's first-parent history would select.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'tests/fixtures/import_fanout_baseline.json'
sys.path.insert(0, str(ROOT / 'tools/bazel'))
import evidence  # noqa: E402
import graph  # noqa: E402

#: How far a summary figure may move without a baseline change. Adding an
#: ordinary test shifts the fractions by about 0.2 percentage points.
PERCENT_TOLERANCE = 1.0
MODULE_TOLERANCE = 2
RATCHETED = {'median_selected_pct': PERCENT_TOLERANCE, 'p90_selected_pct': PERCENT_TOLERANCE,
             'modules_over_half': MODULE_TOLERANCE}
LISTS = ('conftest_source_modules', 'test_to_test_imports')


def nearest_rank(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else 0.0


def measure(root: Path = ROOT, built: graph.Graph | None = None) -> dict:
    """The fan-out of every source module over the gate population, and its summary."""
    built = built or graph.Graph(evidence.Evidence(root))
    selectors = graph.selectors(built.closures())
    population = len(built.tests)
    sources = [module for module in built.evidence.modules if not evidence.is_test_side(module)]
    fractions = {module: len(selectors.get(module, ())) / population for module in sources}
    return {
        'median_selected_pct': round(100 * nearest_rank(fractions.values(), 0.5), 1),
        'p90_selected_pct': round(100 * nearest_rank(fractions.values(), 0.9), 1),
        'modules_over_half': sum(fraction > 0.5 for fraction in fractions.values()),
        'conftest_source_modules': sorted(module for module in built.reach(graph.CONFTEST)
                                          if not evidence.is_test_side(module)),
        'test_to_test_imports': sorted(f'{importer} -> {imported}' for importer in built.evidence.modules
                                       if evidence.is_test_side(importer)
                                       for imported in built.edges[importer] if evidence.is_test(imported)),
        'fractions': fractions,
        'selectors': selectors,
        'population': population,
    }


def regressions(live: dict, baseline: dict) -> list[str]:
    found = [f'{name} is {live[name]}, worse than the baseline {baseline[name]} by more than {tolerance}'
             for name, tolerance in RATCHETED.items() if live[name] > baseline[name] + tolerance]
    for name in LISTS:
        found += [f'new in {name}: {entry}' for entry in sorted(set(live[name]) - set(baseline[name]))]
    return found


def stale_entries(live: dict, baseline: dict) -> list[str]:
    found = [f'{name} improved to {live[name]} from {baseline[name]}: run tools/ci/import_fanout.py '
             '--write-baseline' for name, tolerance in RATCHETED.items() if live[name] < baseline[name] - tolerance]
    for name in LISTS:
        found += [f'gone from {name}: {entry}; run tools/ci/import_fanout.py --write-baseline'
                  for entry in sorted(set(baseline[name]) - set(live[name]))]
    return found


def baseline_record(live: dict, baseline: dict | None = None) -> dict:
    """The live figures, keeping the baseline's where it is better: a figure
    inside its tolerance may wobble, and the record keeps the best reached."""
    record = {name: live[name] for name in (*RATCHETED, *LISTS)}
    if baseline:
        for name in RATCHETED:
            record[name] = min(record[name], baseline[name])
    record['evidence'] = 'tools/bazel/evidence.py'
    return {key: record[key] for key in sorted(record)}


def replay(root: Path, live: dict, count: int) -> dict:
    """The fraction of the gate each of the last `count` merges selects, on today's graph."""
    merges = subprocess.run(['git', 'log', '--first-parent', '--merges', '--format=%H', '-n', str(count), 'HEAD'],
                            cwd=root, capture_output=True, text=True, check=True).stdout.split()
    rows = []
    for sha in merges:
        changed = subprocess.run(['git', 'diff', '--name-only', '--no-renames', f'{sha}^1', sha], cwd=root,
                                 capture_output=True, text=True, check=True).stdout.split()
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
            'p50_pct': round(100 * nearest_rank(fractions, 0.5), 1),
            'p90_pct': round(100 * nearest_rank(fractions, 0.9), 1), 'rows': rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-baseline', action='store_true')
    parser.add_argument('--replay', type=int, metavar='N')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args(argv)
    live = measure(ROOT)
    print(f"single-module change selects: median {live['median_selected_pct']}%, p90 {live['p90_selected_pct']}%; "
          f"{live['modules_over_half']} modules select over half; conftest reaches "
          f"{len(live['conftest_source_modules'])} source modules; {len(live['test_to_test_imports'])} "
          'test-to-test imports')
    for module, fraction in sorted(live['fractions'].items(), key=lambda item: -item[1])[:args.top]:
        print(f'  {100 * fraction:5.1f}%  {module}')
    if args.replay:
        result = replay(ROOT, live, args.replay)
        print(f"replay of {result['merges']} merges: mean {result['mean_pct']}%, p50 {result['p50_pct']}%, "
              f"p90 {result['p90_pct']}%")
    if args.write_baseline:
        previous = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
        BASELINE.write_text(json.dumps(baseline_record(live, previous), indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
