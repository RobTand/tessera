#!/usr/bin/env python3
"""The hosted gate's test population and quarantine policy (tools/ci/gate_policy.py)."""
from __future__ import annotations

import os
import urllib.request
import urllib.error
import json
from pathlib import Path, PurePosixPath
import subprocess


def population(root):
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
                             '--', 'tests/test_*.py'], cwd=root, capture_output=True, check=True)
    return sorted(set(result.stdout.decode().rstrip('\0').split('\0')) - {''})


def exclusion_entries(config):
    if not isinstance(config, dict) or set(config) != {'lanes', 'issues'}:
        raise ValueError('quarantine schema requires only lanes and issues')
    lanes, issues = config['lanes'], config['issues']
    if (not isinstance(lanes, dict) or set(lanes) != {'bytes', 'torch'} or
            not all(isinstance(nodes, list) and all(isinstance(node, str) for node in nodes)
                    for nodes in lanes.values()) or not isinstance(issues, dict)):
        raise ValueError('quarantine schema requires bytes/torch lists and an issue map')
    targets = {node for nodes in lanes.values() for node in nodes}
    if set(issues) != targets:
        raise ValueError('quarantine schema requires exactly one owner per target')
    entries = [{'lane': lane, 'target': node, 'issue': issues[node]}
               for lane, nodes in lanes.items() for node in nodes]
    seen = set()
    for row in entries:
        target = row.get('target', '')
        path, *parts = target.split('::')
        parsed = PurePosixPath(path)
        if (not path.startswith('tests/') or not path.endswith('.py') or
                parsed.is_absolute() or '..' in parsed.parts or str(parsed) != path or
                any(not part for part in parts) or
                target.count('[') != target.count(']') or
                row.get('lane') not in ('bytes', 'torch')):
            raise ValueError('quarantine requires a whole file or exact test node: ' + target)
        if type(row.get('issue')) is not int or row['issue'] < 1:
            raise ValueError('quarantine needs an owner issue for ' + target)
        key = row['lane'], target
        if key in seen:
            raise ValueError('quarantine must list unique test nodes')
        seen.add(key)
    return entries


QUARANTINE_PATH = 'tools/ci/gate_quarantine.json'


def quarantine_policy(entries: list[dict], baseline: list[dict] | None, changed: list[str]) -> None:
    if baseline is None:
        return  # Initial policy, like the budget bootstrap, is reviewed with the gate.
    previous = {(row['lane'], row['target']): row for row in baseline}
    def covered(row):
        target = row['target']
        old = previous.get((row['lane'], target), previous.get((row['lane'], target.split('::', 1)[0])))
        return old is not None and row['issue'] == old['issue']
    growth = [row['target'] for row in entries if not covered(row)]
    isolated = QUARANTINE_PATH in changed and all(
        path == QUARANTINE_PATH or path.startswith('docs/') and path.endswith(('.md', '.rst')) or
        path in {'README.md', 'CHANGELOG.md'} for path in changed)
    if growth and not isolated:
        raise ValueError('Quarantine growth requires a quarantine-only PR with optional docs: ' + ', '.join(growth))


def fetch_quarantine_owner(issue: int) -> dict:
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'tessera-gate'}
    # Anonymous REST shares a small per-IP limit across hosted runners; the
    # workflow's read-only job token has its own allowance.
    if token := os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f'Bearer {token}'
    request = urllib.request.Request(
        f'https://api.github.com/repos/RobTand/tessera/issues/{issue}', headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            raise ValueError(f'Quarantine owner #{issue} must be an open issue') from error
        raise


def check_quarantine_owners(entries: list[dict], *, fetch=fetch_quarantine_owner) -> None:
    for issue in sorted({row['issue'] for row in entries}):
        try:
            owner = fetch(issue)
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                raise ValueError(f'Quarantine owner #{issue} must be an open issue') from error
            raise RuntimeError(f'Quarantine owner #{issue}: no verdict, GitHub HTTP {error.code}') from error
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f'Quarantine owner #{issue}: no verdict, GitHub unavailable: {error}') from error
        if not isinstance(owner, dict) or owner.get('state') not in {'open', 'closed'}:
            raise RuntimeError(f'Quarantine owner #{issue}: no verdict, invalid GitHub response')
        if owner['state'] != 'open' or 'pull_request' in owner:
            raise ValueError(f'Quarantine owner #{issue} must be an open issue')


def base_quarantine(root: Path, base: str) -> dict | None:
    subprocess.run(['git', 'cat-file', '-e', f'{base}^{{commit}}'], cwd=root,
                   capture_output=True, check=True)
    tree = subprocess.run(['git', 'ls-tree', base, '--', QUARANTINE_PATH], cwd=root,
                          capture_output=True, text=True, check=True)
    if not tree.stdout.strip():
        return None
    return json.loads(subprocess.check_output(['git', 'show', f'{base}:{QUARANTINE_PATH}'],
                                             cwd=root, text=True))
