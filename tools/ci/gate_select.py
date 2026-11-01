#!/usr/bin/env python3
"""Adapt Tessera's reverse-import selector to the two hosted CPU lanes."""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
import urllib.error
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition import lane_matrix, selection_shard_count


def full_gate(pr, repository):
    same = pr['head'].get('repo', {}).get('full_name') == repository
    batch = (pr['head']['ref'].startswith('mergify/merge-queue/') and
             pr.get('user', {}).get('login') == 'mergify[bot]' and
             pr.get('user', {}).get('id') == 37929162)
    canary = any(label['name'] == 'gate-canary' for label in pr.get('labels', []))
    return same and (batch or canary)


def population(root):
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
                             '--', 'tests/test_*.py'], cwd=root, capture_output=True, check=True)
    return sorted(set(result.stdout.decode().rstrip('\0').split('\0')) - {''})


FAST_BOUNDARY = {'tests/test_standalone_boundary.py', 'tests/test_packaging.py', 'tests/test_provenance.py'}


def documentation_only(paths):
    return bool(paths) and all(
        not path.startswith('docs/schema/') and
        (path in {'README.md', 'CHANGELOG.md'} or
         path.startswith('docs/') and Path(path).suffix in {'.md', '.rst'})
        for path in paths)


def select_files(root, base, head, execute=subprocess.run):
    all_files = population(root)
    try:
        diff = execute(['git', 'diff', '--name-only', '--no-renames', base, head, '--'],
                       cwd=root, capture_output=True, text=True, check=False, timeout=30)
        if diff.returncode:
            raise ValueError(diff.stderr or 'diff unavailable')
        if documentation_only(diff.stdout.splitlines()):
            return {'verdict': 'narrowed', 'tests': sorted(set(all_files) & FAST_BOUNDARY),
                    'reason': 'documentation-only change: fast boundary tier'}
        result = execute([sys.executable, str(root / 'tools/impacted_tests.py'),
                          '--ref', f'{base}...{head}', '--json'], cwd=root,
                         capture_output=True, text=True, check=False, timeout=120)
        if result.returncode:
            raise ValueError(result.stderr or f'selector exit {result.returncode}')
        if not result.stdout.strip():
            return {'verdict': 'none', 'tests': [], 'reason': 'no changed paths'}
        selection = json.loads(result.stdout)
        if selection['verdict'] == 'full':
            selection['tests'] = all_files
        elif (selection['verdict'] not in ('narrowed', 'none') or
              not set(selection['tests']) <= set(all_files)):
            raise ValueError('selector returned an invalid verdict or unknown files')
        return selection
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        return {'verdict': 'full', 'tests': all_files, 'reason': str(error)}


def exclusion_entries(config):
    if 'entries' in config:
        entries = config['entries']
    else:
        entries = [{'lane': lane, 'target': node, 'issue': config['issues'].get(node)}
                   for lane, nodes in config['lanes'].items() for node in nodes]
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


def quarantine_policy(current: dict, baseline: dict | None, changed: list[str]) -> None:
    entries = exclusion_entries(current)
    if baseline is None:
        return  # Initial policy, like the budget bootstrap, is reviewed with the gate.
    previous = {(row['lane'], row['target']): row for row in exclusion_entries(baseline)}
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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base')
    parser.add_argument('--head')
    parser.add_argument('--mode', choices=['full', 'selection'])
    parser.add_argument('--shards', type=int)
    parser.add_argument('--runners', default='ubuntu-24.04,ubuntu-24.04-arm')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--event', type=Path)
    parser.add_argument('--repository')
    parser.add_argument('--print-mode', action='store_true')
    args = parser.parse_args()
    if args.print_mode:
        event = json.loads(args.event.read_text())
        print('mode=' + ('full' if full_gate(event['pull_request'], args.repository) else 'selection'))
        return 0
    if not all((args.base, args.head, args.mode, args.shards, args.output, args.matrix)):
        parser.error('planning requires base, head, mode, shards, output and matrix')
    root = Path.cwd()
    selection = ({'verdict': 'full', 'tests': population(root), 'reason': 'full CPU gate'}
                 if args.mode == 'full' else select_files(root, args.base, args.head))
    files = population(root) if args.mode == 'full' else selection['tests']
    if args.mode == 'full' and not files:
        raise ValueError('empty CPU population refused in full gate')
    # Gate tooling changes require complete coverage, independently of graph edges.
    diff = subprocess.run(['git', 'diff', '--name-only', args.base, args.head],
                          text=True, capture_output=True, check=True).stdout.splitlines()
    if any(path.startswith(('tools/ci/', '.github/')) or path == 'tools/impacted_tests.py' for path in diff):
        files = population(root)
    config = json.loads((root / QUARANTINE_PATH).read_text())
    entries = exclusion_entries(config)
    unknown = {row['target'].split('::', 1)[0] for row in entries} - set(population(root))
    if unknown:
        raise ValueError('quarantine files outside the test population: ' + ', '.join(sorted(unknown)))
    quarantine_policy(config, base_quarantine(root, args.base), diff)
    check_quarantine_owners(entries)
    durations = json.loads((root / 'tools/ci/gate_durations.json').read_text())['lanes']
    count = args.shards
    if args.mode == 'selection':
        count = max(selection_shard_count(files, durations.get(lane, {}), args.shards,
                        workers=4, single_shard_seconds=120) for lane in ('bytes', 'torch'))
    runners = args.runners.split(',')
    if int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')) % 2 == 0:
        runners.reverse()
    matrix = lane_matrix(files, durations, count, runners) if files else {'shards': 0, 'include': []}
    # A no-tests PR has a complete empty plan, and launches no pytest shard.
    plan = {'schema': 'tessera.gate_plan.v1', 'base_sha': args.base, 'head_sha': args.head,
            'mode': args.mode, 'selection': selection,
            'population': {'version': hashlib.sha256('\n'.join(population(root)).encode()).hexdigest(),
                           'cuda': 'not covered: CPU wheels, CUDA_VISIBLE_DEVICES empty'},
            'selected': [{'lane': lane, 'path': path, 'estimated_seconds': durations.get(lane, {}).get(path)}
                         for lane in ('bytes', 'torch') for path in files],
            'deselect': config['lanes']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + '\n')
    args.matrix.write_text(json.dumps(matrix, indent=2) + '\n')
    print('has_tests=' + str(bool(files)).lower())
    print('matrix=' + json.dumps({'include': matrix['include'] or [{'index': 0}]}, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except ValueError as error:
        print('GATE_POLICY_FAILURE: ' + str(error), file=sys.stderr)
        code = 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('NO_VERDICT: ' + str(error), file=sys.stderr)
        code = 75
    raise SystemExit(code)
