#!/usr/bin/env python3
"""Does a change deepen the test graph? The core of the graph-impact check.

Every graph source feeds this the same input:

- tests: {test_id: set(source_file)}, each test's static reach;
- edges: set((from_file, to_file)), for blame paths and cycles; a test's
  edges start at its test_id;
- loc: {file: lines}, optional, for the line-weighted figure. Its keys are
  source files too: a file no test reaches counts in U with fanout 0.

From it alone, with no run history:

- fanout(m): the number of tests whose reach holds source file m;
- U, the uniform-change selection: the mean over source files of
  fanout(m) / T, the expected fraction of tests one random file edit reruns;
  U_loc weights each file by its lines;
- the largest strongly connected component (an import cycle) and its members;
- the reach distribution: median, p90, max.

compare() is the base -> head check: a new or changed test whose reach is over
the budget (with the import path to the hub it reaches through), U worse past
tolerance, or a cycle that grew or merged (with the edges the change added).
summary() is the one line every run prints. The baseline helpers keep the
best figures reached; the allowlist names tests that are integration-wide by
design and only shrinks.
"""
from __future__ import annotations

from collections import deque
import math
import statistics

#: How far U may move without failing.
PERCENT_TOLERANCE = 1.0
#: How far the largest cycle may move against the baseline. Against the merge
#: base, any growth fails.
CYCLE_TOLERANCE = 2
RATCHETED = {'uniform_selection_pct': PERCENT_TOLERANCE, 'uniform_selection_loc_pct': PERCENT_TOLERANCE,
             'largest_cycle': CYCLE_TOLERANCE}
#: A source file more than this fraction of the tests reach is a hub.
HUB_FRACTION = 0.5


def nearest_rank(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else 0


def strongly_connected(nodes, successors):
    """Tarjan's algorithm, iterative; components in a stable order."""
    index, low, on_stack, stack, components, counter = {}, {}, set(), [], [], 0
    for start in sorted(nodes):
        if start in index:
            continue
        work = [(start, iter(sorted(successors.get(start, ()))))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(successors.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return sorted(components)


def metrics(tests, edges, loc=None):
    """Every figure the check reads, from the input contract."""
    loc = loc or {}
    population = len(tests)
    sources = set().union(set(loc), *tests.values())
    fanout = dict.fromkeys(sources, 0)
    for reach in tests.values():
        for file in reach:
            fanout[file] += 1
    successors = {}
    for source, target in edges:
        successors.setdefault(source, set()).add(target)
    nodes = {node for edge in edges for node in edge}
    cycles = [set(component) for component in strongly_connected(nodes, successors) if len(component) > 1]
    largest = max(cycles, key=len, default=set())
    sizes = [len(reach) for reach in tests.values()]
    total_loc = sum(loc.get(file, 1) for file in sources) or 1
    uniform = statistics.mean(count / population for count in fanout.values()) if sources else 0.0
    weighted = sum(loc.get(file, 1) * count / population for file, count in fanout.items()) / total_loc \
        if population else 0.0
    return {
        'tests': tests, 'edges': set(edges), 'successors': successors, 'fanout': fanout, 'cycles': cycles,
        'population': population,
        'uniform_selection_pct': round(100 * uniform, 1),
        'uniform_selection_loc_pct': round(100 * weighted, 1),
        'largest_cycle': max(len(largest), 1), 'largest_cycle_members': sorted(largest),
        'reach_median': nearest_rank(sizes, 0.5), 'reach_p90': nearest_rank(sizes, 0.9),
        'reach_max': max(sizes, default=0),
    }


def blame_path(figures, test):
    """The shortest import path from a test to the hub its reach is heaviest on."""
    reach = figures['tests'][test]
    if not reach:
        return [test]
    hub = max(reach, key=lambda file: (figures['fanout'][file], file))
    previous, queue = {test: None}, deque([test])
    while queue:
        node = queue.popleft()
        if node == hub:
            break
        for target in sorted(figures['successors'].get(node, ())):
            if target not in previous:
                previous[target] = node
                queue.append(target)
    if hub not in previous:
        return [test, '(not by import)', hub]
    path, node = [], hub
    while node is not None:
        path.append(node)
        node = previous[node]
    return path[::-1]


def changed_tests(base, head, changed):
    changed = set(changed)
    return sorted(test for test in head['tests'] if test not in base['tests'] or test in changed)


def compare(base, head, changed, budget, allowlist):
    """Why the head's graph is worse than the base's; empty when it isn't."""
    failures = []
    for test in changed_tests(base, head, changed):
        size = len(head['tests'][test])
        if test not in allowlist and size > budget:
            failures.append(f'{test} reaches {size} source files, over the budget of {budget}: '
                            + ' -> '.join(blame_path(head, test)))
    added = sorted(head['edges'] - base['edges'])
    if head['uniform_selection_pct'] > base['uniform_selection_pct'] + PERCENT_TOLERANCE:
        grown = sorted(((count - base['fanout'].get(file, 0), file) for file, count in head['fanout'].items()),
                       reverse=True)[:5]
        failures.append(
            f"uniform-change selection U rose from {base['uniform_selection_pct']}% to "
            f"{head['uniform_selection_pct']}%; edges this change added: "
            + ', '.join(f'{source} -> {target}' for source, target in added[:20])
            + '; fanout grew most for ' + ', '.join(f'{file} (+{growth})' for growth, file in grown if growth > 0))
    for cycle in head['cycles']:
        merged = [old for old in base['cycles'] if old & cycle]
        largest = max((len(old) for old in merged), default=1)
        if len(cycle) > largest or len(merged) > 1:
            inside = [f'{source} -> {target}' for source, target in added if source in cycle and target in cycle]
            failures.append(f'an import cycle grew to {len(cycle)} modules (largest at the base: {largest}'
                            + (f', {len(merged)} cycles merged' if len(merged) > 1 else '')
                            + '); edges this change added inside it: ' + (', '.join(inside) or 'none'))
    return failures


def summary(base, head, changed, budget):
    new = changed_tests(base, head, changed)
    added = head['edges'] - base['edges']
    hubs = {file for file, count in head['fanout'].items()
            if head['population'] and count / head['population'] > HUB_FRACTION}
    to_hubs = sum(1 for _, target in added if target in hubs)
    sizes = [len(head['tests'][test]) for test in new]
    reaches = ', '.join(str(size) for size in sizes)
    within = '<=' if all(size <= budget for size in sizes) else 'vs'
    return (f"U {base['uniform_selection_pct']}% -> {head['uniform_selection_pct']}%, "
            f"SCC max {base['largest_cycle']} -> {head['largest_cycle']}, new tests {len(new)}"
            + (f' (reach {reaches} {within} budget {budget})' if new else '')
            + f", new edges {len(added)}, " + (f'{to_hubs} to a hub' if to_hubs else 'none to a hub'))


def regressions(live, baseline):
    return [f'{name} is {live[name]}, worse than the baseline {baseline[name]} by more than {tolerance}'
            for name, tolerance in RATCHETED.items() if live[name] > baseline[name] + tolerance]


def stale_entries(live, baseline, fix):
    return [f'{name} improved to {live[name]} from {baseline[name]}: {fix}'
            for name, tolerance in RATCHETED.items() if live[name] < baseline[name] - tolerance]


def baseline_record(live, baseline=None):
    """The live figures, keeping the baseline's where it is better. The reach
    budget is the p90 of test reach when first recorded, and only falls."""
    record = {name: live[name] for name in RATCHETED}
    record['reach_budget'] = live['reach_p90']
    if baseline:
        for name in (*RATCHETED, 'reach_budget'):
            if name in baseline:
                record[name] = min(record[name], baseline[name])
    return record


def allowlist_problems(previous, allowlist, changed, allowlist_path, isolated_prefixes=('docs/',)):
    """Every exception has a reason; the list grows only in a change of its own."""
    grown = sorted(set(allowlist) - set(previous))
    isolated = all(path == allowlist_path or path.startswith(isolated_prefixes) for path in changed)
    problems = [f'{allowlist_path} gives no reason for {test}' for test, reason in allowlist.items()
                if not isinstance(reason, str) or not reason.strip()]
    if grown and not isolated:
        problems.append(f'{allowlist_path} grew ({", ".join(grown)}); add an exception in a change of its own')
    return problems
