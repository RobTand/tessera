"""Read issue-owned, exact-node single-retry marks without pytest in the planner."""
from __future__ import annotations

import json
from pathlib import PurePosixPath

FLAKY_PATH = 'tools/ci/gate_flaky.json'


def read_table(root):
    path = root / FLAKY_PATH
    if not path.exists():
        return {}  # Older gate fixtures and the initial policy have no marks.
    table = json.loads(path.read_text())
    if not isinstance(table, dict):
        raise ValueError('flaky table must map exact nodes to an issue and reason')
    for node, entry in table.items():
        filename, *parts = node.split('::')
        parsed = PurePosixPath(filename)
        if (not filename.startswith('tests/') or not filename.endswith('.py') or
                str(parsed) != filename or '..' in parsed.parts or not parts or
                any(not part for part in parts) or node.count('[') != node.count(']')):
            raise ValueError('flaky table requires an exact test node: ' + node)
        if (not isinstance(entry, dict) or set(entry) != {'issue', 'reason'} or
                type(entry['issue']) is not int or entry['issue'] < 1 or
                not isinstance(entry['reason'], str) or not entry['reason'].strip()):
            raise ValueError('flaky table needs a positive issue and nonempty reason: ' + node)
    return table


def check_policy(root, population, check_owners):
    table = read_table(root)
    files = {node.split('::', 1)[0] for node in table}
    unknown = files - set(population)
    if unknown:
        raise ValueError('flaky files outside the test population: ' + ', '.join(sorted(unknown)))
    check_owners(list(table.values()))
    # Always collect marked files, including on docs-only/no-tests changes, so
    # collection checks exact node IDs even outside the changed selection.
    return sorted(files)
