#!/usr/bin/env python3
"""Start the Bazel gate on a BuildBuddy Remote Bazel runner and report its result.

Runs in the GitHub job. The runner executes tools/ci/bazel_gate.py and leaves
gate-result.json as a BuildBuddy artifact; this script fetches it. A failed,
timed-out or disconnected run, or a missing or malformed result, fails here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import sys
import time
import urllib.error
import urllib.request

API = 'https://app.buildbuddy.io/api/v1/'
IMAGE = ('docker://gcr.io/flame-public/rbe-ubuntu24-04@sha256:'
         '5f0c01e5f6d79e58d7a996330de33aa38ce599bf8431eaf27521e7602daadf50')
REPOSITORY = 'https://github.com/RobTand/tessera'
RESULT = 'gate-result.json'
SCHEMA = 'tessera.bazel-gate.v1'
# Diagnostic logs, by suffix: each test or build run names its own.
DIAGNOSTICS = ('bazel-remote-grpc.log', 'execution_log.binpb.zst')
DONE = 'COMPLETE_INVOCATION_STATUS'
LOST = 'DISCONNECTED_INVOCATION_STATUS'
GATE_FALLBACK_CONCURRENCY = 2


class RemoteError(RuntimeError):
    pass


class BuildBuddy:
    def __init__(self, key, api=API):
        if not key:
            raise RemoteError('BUILDBUDDY_API_KEY is empty')
        self.key, self.api = key, api

    def call(self, method, body, timeout=120, raw=False):
        """JSON in; JSON out, or the raw body for GetFile."""
        request = urllib.request.Request(
            self.api + method, data=json.dumps(body).encode(),
            headers={'x-buildbuddy-api-key': self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read() if raw else json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise RemoteError(f'BuildBuddy {method} failed: {error}') from error


def history_environment(environ):
    return {
        name: value for name, value in environ.items()
        if (name in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'VIRTUAL_ENV',
                     'BUILDBUDDY_ARTIFACTS_DIRECTORY', 'BBRUN_FINAL_SCAN')
            or name.startswith(('LC_', 'UV_', 'PYTHON')))
        and not name.endswith(('_API_KEY', '_TOKEN'))
    }


def prepare_full_history(expected_commit, root=None):
    """Runner-only: hydrate origin without exposing Git diagnostics or credentials."""
    import json
    from pathlib import Path
    import re
    import subprocess

    root = Path.cwd() if root is None else Path(root)
    environment = history_environment(os.environ)
    environment.pop('BBRUN_FINAL_SCAN', None)
    facts = {'fetched': False, 'objects': 0, 'HEAD': None}
    stage = 'validate-commit'

    def git(arguments, *, accepted=(0,), failure='Git command failed'):
        result = subprocess.run(['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
                                 *arguments], cwd=root, env=environment,
                                capture_output=True, text=True, timeout=300)
        if result.returncode not in accepted:
            raise RuntimeError(failure)
        return result

    try:
        if not re.fullmatch(r'[0-9a-f]{40}', expected_commit):
            raise RuntimeError('a full lowercase commit SHA is required')
        stage = 'git-version'
        version = git(['--version']).stdout
        match = re.match(r'git version (\d+)\.(\d+)', version)
        if not match or tuple(map(int, match.groups())) < (2, 36):
            raise RuntimeError('Git 2.36 or newer is required for fetch --refetch --no-filter')
        stage = 'shallow-check'
        shallow = git(['rev-parse', '--is-shallow-repository']).stdout.strip()
        if shallow not in ('true', 'false'):
            raise RuntimeError('cannot determine whether checkout is shallow')
        fetch = ['fetch', '--refetch', '--no-filter', '--tags']
        if shallow == 'true':
            fetch.append('--unshallow')
        fetch.extend(['origin', '+refs/heads/*:refs/remotes/origin/*'])
        stage = 'fetch'
        git(fetch, failure='full fetch with --refetch --no-filter failed; refusing Bazel')
        facts['fetched'] = True
        stage = 'head-check'
        head = git(['rev-parse', '--verify', 'HEAD']).stdout.strip()
        if re.fullmatch(r'[0-9a-f]{40}', head):
            facts['HEAD'] = head
        if head != expected_commit:
            raise RuntimeError('HEAD differs from requested commit after full fetch')
        stage = 'object-check'
        objects = git(['rev-list', '--objects', '--missing=print', '--all', 'HEAD'],
                      failure='cannot verify complete object graph').stdout.splitlines()
        if not objects or any(line.startswith('?') for line in objects):
            raise RuntimeError('missing objects remain after full fetch')
        if any(not re.fullmatch(r'[0-9a-f]{40}', line.split(' ', 1)[0]) for line in objects):
            raise RuntimeError('cannot verify object identifiers')
        facts['objects'] = len(objects)
        # A successful unfiltered refetch can leave old promisor pack markers.
        # Only discard their metadata after proving the reachable graph complete.
        stage = 'promisor-cleanup'
        pack = root / Path(git(['rev-parse', '--git-path', 'objects/pack']).stdout.strip())
        for marker in pack.glob('*.promisor'):
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
        stage = 'partial-clone-cleanup'
        for name in ('remote.origin.promisor', 'remote.origin.partialclonefilter',
                     'extensions.partialclone'):
            git(['config', '--local', '--unset-all', name], accepted=(0, 5),
                failure='cannot remove partial-clone metadata')
        stage = 'metadata-check'
        metadata = git(['config', '--get-regexp',
                        r'^(remote\..*\.(promisor|partialclonefilter)|extensions\.partialclone)$'],
                       accepted=(0, 1), failure='cannot verify partial-clone metadata')
        if metadata.returncode != 1 or any(pack.glob('*.promisor')):
            raise RuntimeError('promisor metadata remains after full fetch')
        stage = 'shallow-check'
        if git(['rev-parse', '--is-shallow-repository']).stdout.strip() != 'false':
            raise RuntimeError('shallow history remains after full fetch')
        return True
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f'full history preparation failed: {stage} ({type(error).__name__})',
              file=sys.stderr, flush=True)
        return False
    finally:
        print('full history: ' + json.dumps(facts, sort_keys=True), flush=True)

def run_request(args, full_shard=None):
    if not args.base or not args.head:
        raise RemoteError('empty base or head: the queue-aware refs were unavailable')
    command = ['tools/ci/bazel_gate.py', args.mode, '--base', args.base,
               '--head', args.head, '--branch', args.branch]
    if full_shard is None:
        command.append('--plan-full')
    else:
        arch, index = full_shard
        if arch not in ('amd64', 'arm64') or type(index) is not int or index not in range(2):
            raise RemoteError('invalid full-shard coordinates')
        command.extend(['--full-shard', f'{arch}:{index}'])
    return {
        'repo': args.repo,
        'branch': args.branch,
        'commit_sha': args.checkout or args.head,
        # The runner image's Python 3.12 is the version Tessera's gate runs.
        'steps': [{'run': 'python3 ' + shlex.join(command)}],
        'env': {'GIT_REPO_DEFAULT_BRANCH': 'master'},
        'platform_properties': {
            # The runner hosts only the Bazel client; tests go to remote execution.
            'EstimatedCPU': str(args.cpu),
            'EstimatedMemory': args.memory,
            # 1,030 test actions' input trees; see .bazelrc's --nobuild_runfile_links.
            'EstimatedFreeDiskBytes': '50GB',
            'container-image': IMAGE,
            'workload-isolation-type': 'firecracker',
            'recycle-runner': 'true',
        },
        # BuildBuddy's free tier disconnects every Remote Bazel run at one hour.
        'timeout': f'{min(args.timeout, 55)}m',
        'wait_until': 'STARTED',
    }


def await_invocation(client, invocation_id, deadline, interval=15, clock=time.monotonic, sleep=time.sleep,
         retries=3):
    """Wait for the run's own terminal status. Returns the invocation record."""
    errors = 0
    while True:
        try:
            found = client.call('GetInvocation', {'selector': {'invocation_id': invocation_id},
                                                  'include_artifacts': True}).get('invocation', [])
            errors = 0
        except RemoteError:
            # One API blip must not fail a whole merge-queue batch.
            errors += 1
            if errors > retries or clock() > deadline:
                raise
            sleep(interval)
            continue
        invocation = found[0] if found else {}
        status = invocation.get('invocationStatus')
        if status == DONE:
            return invocation
        if status == LOST:
            raise RemoteError(f'remote run {invocation_id} disconnected')
        if clock() > deadline:
            raise RemoteError(f'remote run {invocation_id} did not finish in time')
        sleep(interval)


def diagnostics(client, invocation):
    for artifact in invocation.get('artifacts', []):
        name = artifact.get('name', '')
        if name.endswith(DIAGNOSTICS):
            yield Path(name).name, client.call('GetFile', {'uri': artifact['uri']}, timeout=300, raw=True)


def fetch(client, invocation, name):
    for artifact in invocation.get('artifacts', []):
        if artifact.get('name') == name:
            return client.call('GetFile', {'uri': artifact['uri']}, timeout=300, raw=True)
    return None


def parse_result(raw, head):
    if raw is None:
        raise RemoteError(f'the remote run left no {RESULT}')
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RemoteError(f'{RESULT} is not JSON: {error}') from error
    if not isinstance(result, dict) or result.get('schema') != SCHEMA or result.get('head') != head:
        raise RemoteError(f'{RESULT} has the wrong schema or head')
    if result.get('mode') not in ('pr', 'main'):
        raise RemoteError(f'{RESULT} has the wrong mode')
    if (result.get('error') is None and result.get('mode') == 'pr'
            and 'full_populations' not in result and result.get('verdict') not in (
            'pass', 'fail', 'skipped')):
        raise RemoteError(f'{RESULT} has no verdict')
    if (result.get('error') is None and result['mode'] == 'main'
            and 'full_populations' not in result and 'full_shard' not in result
            and set(result.get('architectures', {})) != {'amd64', 'arm64'}):
        raise RemoteError(f'{RESULT} has no complete post-merge verdict')
    return result


def merge_full_shards(plan, parts):
    """Accept exactly two complete, disjoint result records per requested arch."""
    arches = ['amd64'] if plan['mode'] == 'pr' else ['amd64', 'arm64']
    if set(plan.get('full_populations', {})) != set(arches):
        raise RemoteError('full plan has the wrong architecture populations')
    if len(parts) != 2 * len(arches):
        raise RemoteError('full run is missing shard records')
    merged = dict(plan, inputs=list(plan.get('inputs', [])))
    records = {}
    for arch in arches:
        expected = plan['full_populations'][arch]
        rows = [part for part in parts if part.get('full_shard', {}).get('architecture') == arch]
        if (len(rows) != 2 or {part['full_shard'].get('index') for part in rows} != {0, 1}
                or any(type(part['full_shard'].get('index')) is not int for part in rows)):
            raise RemoteError('full run has missing or duplicate shard coordinates')
        rows.sort(key=lambda row: row['full_shard']['index'])
        for kind, field in (('tests', 'selected'), ('rules', 'full_rules')):
            labels = [label for row in rows for label in row.get(field, [])]
            if (not expected.get(kind) or len(expected[kind]) != len(set(expected[kind]))
                    or len(labels) != len(set(labels)) or set(labels) != set(expected[kind])):
                raise RemoteError('full shard union differs from the planned ' + kind)
        failed, warnings, bad = set(), [], False
        for row in rows:
            if row.get('head') != plan['head'] or row.get('mode') != plan['mode'] or row.get('error'):
                raise RemoteError('full shard has the wrong head, mode or an error')
            if (row.get('verdict') not in ('pass', 'fail', 'skipped')
                    or type(row.get('bazel_exit')) is not int
                    or not isinstance(row.get('failed'), list)
                    or (row.get('verdict') == 'skipped' and row['selected'])
                    or not set(row['failed']) <= set(row['selected'])):
                raise RemoteError('full shard has no valid test/build verdict')
            if row['bazel_exit'] and not (row['bazel_exit'] == 3 and row['failed']):
                merged['inputs'].append(f"bazel full shard {arch}:{row['full_shard']['index']} exited {row['bazel_exit']}")
            merged['inputs'].extend(row.get('inputs', []))
            failed.update(row['failed'])
            warnings.extend(row.get('budget_warnings', []))
            bad |= bool(row['bazel_exit'] or row['failed'] or row['verdict'] == 'fail')
        records[arch] = {'verdict': 'fail' if bad else 'pass', 'failed': sorted(failed),
                         'tests': len(expected['tests']), 'budget_warnings': warnings,
                         'missed': [] if plan.get('selection_error') else sorted(failed - set(plan['selected']))}
    merged.pop('full_populations')
    if plan['mode'] == 'pr':
        merged.update({key: records['amd64'][key]
                       for key in ('verdict', 'failed', 'budget_warnings')})
    else:
        merged['architectures'] = records
    return merged


def outputs(result):
    """GitHub step outputs. Scopes are ready only when detection succeeded."""
    ready = result.get('error') is None and result.get('mode') == 'pr'
    values = {
        'scopes_ready': str(ready).lower(),
        'scopes': ','.join(result.get('scopes', [])) if ready else '',
        'all_scopes': str(bool(result.get('all_scopes'))).lower(),
    }
    if result.get('mode') == 'pr':
        ok = not result.get('error') and not result.get('inputs')
        impact = result.get('graph_impact') or {}
        values['tests'] = ('success' if ok and result.get('verdict') in ('pass', 'skipped') and not impact.get('exit')
                           else 'failure')
    return values


def post_merge_record(result, prs, seconds):
    """One machine-readable line per post-merge run."""
    failures = {arch: info['failed'] for arch, info in result.get('architectures', {}).items()}
    missed = {arch: info['missed'] for arch, info in result.get('architectures', {}).items()}
    if result.get('error') or result.get('inputs') or result.get('selection_error'):
        verdict = 'gate-error'
    elif missed.get('amd64'):
        verdict = 'detection-miss'
    elif failures.get('amd64') or result.get('architectures', {}).get('amd64', {}).get('verdict') == 'fail':
        verdict = 'selected-but-failed'
    elif failures.get('arm64') or result.get('architectures', {}).get('arm64', {}).get('verdict') == 'fail':
        verdict = 'arm64-only'
    else:
        verdict = 'pass'
    return {'merge_sha': result['head'], 'base_sha': result['base'], 'prs': prs,
            'verdict': verdict, 'selection': result.get('selection'),
            'selected': len(result.get('selected', [])),
            'tests': {arch: row['tests'] for arch, row in result.get('architectures', {}).items()},
            'failures': failures, 'missed': missed, 'seconds': round(seconds, 1),
            'error': result.get('error'), 'selection_error': result.get('selection_error'),
            'inputs': result.get('inputs'),
            'invocations': result.get('invocations', []),
            'budget_warnings': {arch: row.get('budget_warnings', [])
                                for arch, row in result.get('architectures', {}).items()}}


def explain(result, prs):
    lines = [f"Bazel gate ({result['mode']}) {result['base'][:12]}..{result['head'][:12]}"]
    if result.get('error'):
        lines.append(f"Detection failed: {result['error']}")
    if result.get('selection_error'):
        lines.append('Post-merge selection failed: ' + result['selection_error'])
    if result.get('inputs'):
        lines.append('Stale generated inputs, parity drift or growing budgets; fix with '
                     'tools/bazel/generate.sh: ' + '; '.join(result['inputs']))
    if result['mode'] == 'pr':
        lines.append(f"Selection: {result.get('selection')}, {len(result.get('selected', []))} tests")
        impact = result.get('graph_impact')
        if impact:
            lines.append('graph-impact: ' + impact['summary'])
            lines.extend('graph-impact: FAIL ' + line for line in impact['failures'])
        lines.extend(f"BUDGET_WARNING: {row['nodeid']}: {row['seconds']:.3f}s > "
                     f"{row['budget_seconds']:.3f}s policy ({row['label']})"
                     for row in result.get('budget_warnings', []))
        lines.extend(f'FAILED: {label}' for label in result.get('failed', []))
    else:
        lines.append(f"Merged PRs: {', '.join(f'#{pr}' for pr in prs) or 'unknown'}")
        for arch, info in result.get('architectures', {}).items():
            lines.extend(f"BUDGET_WARNING: {row['nodeid']}: {row['seconds']:.3f}s > "
                         f"{row['budget_seconds']:.3f}s policy ({arch}, {row['label']})"
                         for row in info.get('budget_warnings', []))
            for label in info['failed']:
                caught = 'not selected: detection miss' if label in info['missed'] else 'was selected'
                if result.get('selection_error'):
                    caught = 'selection unavailable: detection failed'
                elif arch == 'arm64':
                    caught = 'arm64 runs only after merge'
                lines.append(f'FAILED on {arch}: {label} ({caught})')
    return '\n'.join(lines)


def aggregate(needs):
    """A required check: the Bazel job finished detection, uploaded scopes and
    passed, and the hosted boundary job (no-torch proofs, policies) passed."""
    gate = needs.get('bazel', {})
    values = gate.get('outputs', {})
    if gate.get('result') != 'success' or values.get('scopes_uploaded') != 'true':
        raise RuntimeError(f'The Bazel gate did not finish detection and scope upload: {gate}')
    if values.get('tests') != 'success':
        raise RuntimeError('The Bazel gate failed: see the Bazel job')
    if needs.get('boundary', {}).get('result') != 'success':
        raise RuntimeError('The boundary job failed: see it')


def write_github(path_variable, text):
    path = os.environ.get(path_variable)
    if path:
        with open(path, 'a') as handle:
            handle.write(text)


def execute(client, args):
    """Only gate fallback may overlap two Runs; selected and main stay serial."""

    def run(coordinates=None):
        request = run_request(args, full_shard=coordinates)
        invocation_id = client.call('Run', request, timeout=600).get('invocationId')
        if not invocation_id:
            raise RemoteError('BuildBuddy Run returned no invocation id')
        print(f'Remote Bazel run: https://app.buildbuddy.io/invocation/{invocation_id}', flush=True)
        invocation = await_invocation(
            client, invocation_id, time.monotonic() + (int(request['timeout'][:-1]) + 10) * 60)
        for name, data in diagnostics(client, invocation):
            (args.output_dir / f'{invocation_id}-{name}').write_bytes(data)
        result = parse_result(fetch(client, invocation, RESULT), args.head)
        if result.get('mode') != args.mode or result.get('base') != args.base:
            raise RemoteError('remote result has the wrong mode or base')
        # Retain each result for diagnostics; the workflow scans everything before upload.
        (args.output_dir / f'{invocation_id}-{RESULT}').write_text(json.dumps(result, indent=2))
        record = {'id': invocation_id, 'full_shard': coordinates,
                  'success': invocation.get('success') is True}
        if (invocation.get('success') is not True and result.get('verdict') != 'fail'
                and not result.get('error') and not result.get('inputs')):
            result.setdefault('inputs', []).append('remote runner did not succeed: ' + invocation_id)
        return result, record

    plan, initial = run()
    runs = [initial]
    if 'full_populations' in plan and not plan.get('error') and not plan.get('inputs'):
        arches = ['amd64'] if args.mode == 'pr' else ['amd64', 'arm64']
        coordinates = [(arch, index) for arch in arches for index in range(2)]
        if args.mode == 'pr':
            if plan.get('selection') != 'fallback':
                raise RemoteError('only gate fallback may submit parallel full shards')
            with ThreadPoolExecutor(max_workers=GATE_FALLBACK_CONCURRENCY) as workers:
                completed = list(workers.map(run, coordinates))
        else:
            completed = [run(coordinate) for coordinate in coordinates]
        runs.extend(record for _, record in completed)
        result = merge_full_shards(plan, [part for part, _ in completed])
    else:
        result = plan
    result['invocations'] = runs
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['aggregate']:
        try:
            aggregate(json.loads(os.environ['NEEDS_JSON']))
        except (RuntimeError, KeyError, ValueError) as error:
            sys.exit(str(error))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pr', 'main'])
    parser.add_argument('--base', default='')
    parser.add_argument('--head', required=True)
    parser.add_argument('--checkout', default='', help='commit the runner clones before the gate fetches HEAD')
    parser.add_argument('--branch', required=True)
    parser.add_argument('--repo', default=REPOSITORY)
    parser.add_argument('--prs', default='', help='comma-separated merged PR numbers')
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--memory', default='8GB')
    parser.add_argument('--timeout', type=int, default=45, help='minutes')
    parser.add_argument('--output-dir', type=Path, default=Path('gate-output'))
    args = parser.parse_args(argv)
    prs = [int(pr) for pr in args.prs.split(',') if pr]
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = BuildBuddy(os.environ.get('BUILDBUDDY_API_KEY', ''))
        result = execute(client, args)
    except RemoteError as error:
        print(f'::error::{error}', flush=True)
        write_github('GITHUB_OUTPUT', 'scopes_ready=false\ntests=failure\n')
        return 1
    (args.output_dir / RESULT).write_text(json.dumps(result, indent=2))
    summary = explain(result, prs)
    print(summary, flush=True)
    write_github('GITHUB_STEP_SUMMARY', '```\n' + summary + '\n```\n')
    write_github('GITHUB_OUTPUT', ''.join(f'{k}={v}\n' for k, v in outputs(result).items()))
    ok = not result.get('error') and not result.get('inputs')
    if args.mode == 'main':
        record = post_merge_record(result, prs, time.monotonic() - started)
        line = json.dumps(record)
        (args.output_dir / 'post-merge-record.jsonl').write_text(line + '\n')
        print('POST_MERGE_RECORD ' + line, flush=True)
        write_github('GITHUB_STEP_SUMMARY', f'\n`POST_MERGE_RECORD {line}`\n')
        ok = ok and record['verdict'] == 'pass'
    else:
        ok = ok and outputs(result)['tests'] == 'success'
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
