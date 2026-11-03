#!/usr/bin/env python3
"""The Bazel merge gate. Runs on a BuildBuddy Remote Bazel runner (#1589).

pr mode selects the tests a change can affect with bazel-diff, runs them on
remote execution (and the runner lane on the runner itself), and reports
Mergify scopes. main mode runs every test on amd64 and arm64 after a merge and
records whether the merged change's own selection would have caught each
failure. Both write gate-result.json to the BuildBuddy artifacts directory,
and tools/ci/remote_gate.py reads it back. Any detection failure is a gate
failure, never an empty pass.

Per-test budget overruns are warnings and are recorded without changing a
verdict. Assertion failures and hard timeouts fail on their first attempt.
There are no test reruns.
"""
import argparse
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_no_secret_leak  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'tessera.bazel-gate.v1'
RESULT = 'gate-result.json'

# The one bazel-diff pin TL3, PrismaBuild, PrismaQuant and Tessera share.
BAZEL_DIFF_URL = 'https://github.com/Tinder/bazel-diff/releases/download/v70.0.2/bazel-diff-rust-linux-amd64'
BAZEL_DIFF_SHA256 = '44226a1d1d5239abc8bc6158de1f886fcbbf58e44479ec178df8c01e7e32102e'

# A change to any of these can move every target: test everything. Patterns
# without '/' name root files. Other workflows are ordinary inputs.
FALLBACK = [
    'MODULE.bazel', 'MODULE.bazel.lock', '.bazelrc', '.bazelversion', '.bazelignore', 'BUILD.bazel',
    'tools/bazel/*', '.github/workflows/gate.yml', 'tools/ci/bazel_gate.py', 'tools/ci/remote_gate.py',
]
CONFIGS = {'amd64': 'remote', 'arm64': 'remote-arm64'}
DIAGNOSTICS = ['bazel-remote-grpc.log', 'execution_log.binpb.zst']
TESTS = 'tests(//...)'
# A Remote Bazel runner holds BuildBuddy's API key. Only the in-process
# artifact scanner reads it; children use the runner's configured Bazel auth.
SECRET_NAMES = {'BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN', 'GITHUB_TOKEN'}


class GateError(RuntimeError):
    """Detection could not establish what to test. The gate fails."""


def child_environment(environ=None):
    environ = os.environ if environ is None else environ
    return {name: value for name, value in environ.items()
            if name not in SECRET_NAMES and not name.endswith(('_API_KEY', '_TOKEN'))}


# fnmatch's * also crosses '/', so 'tools/bazel/*' covers the whole tree.
def fallback_files(changed):
    return sorted(path for path in changed if any(
        fnmatch(path, pattern) if '/' in pattern else path == pattern for pattern in FALLBACK))


def build_graph(path):
    return Path(path).name in ('BUILD.bazel', 'BUILD') or path.endswith('.bzl')


def everything(reason, tests):
    return {'selection': 'fallback', 'reason': sorted(reason), 'selected': sorted(tests),
            'build': ['//...'], 'scopes': [], 'all_scopes': True}


def gate_decision(changed, deleted, impacted, used_sources, tests):
    """Decide what to test. Pure; every input comes from git or Bazel.

    A Python file no target reads means the BUILD files are stale: fail. Any
    other file no target reads is no test's input and selects nothing; the
    generated-input check in the same run still fails a stale BUILD file.
    """
    if not changed and not deleted:
        raise GateError('no changed files between base and head; refusing to pass an empty diff')
    forced = fallback_files(changed | deleted)
    if forced:
        return everything(forced, tests)
    unmapped = sorted(path for path in changed - deleted if path not in used_sources and path.endswith('.py'))
    if unmapped:
        raise GateError('changed Python files that no Bazel target holds: ' + ', '.join(unmapped)
                        + '. Run tools/bazel/generate.sh and commit what it writes (docs/bazel.md).')
    relevant = [path for path in changed | deleted if path in used_sources or path in deleted]
    if not impacted and relevant and all(build_graph(path) for path in relevant):
        # A BUILD or .bzl edit that changes no rule hash (formatting, comments): test everything.
        return everything(relevant, tests)
    if not impacted and relevant:
        raise GateError('bazel-diff found no impacted targets for build-relevant changes')
    selected = sorted(label for label in impacted if label in tests)
    scopes = sorted({label.split(':', 1)[0] for label in impacted if label.startswith('//')})
    build = sorted(label for label in impacted if label not in tests)
    return {'selection': 'impacted' if impacted else 'irrelevant', 'reason': [], 'selected': selected,
            'build': build, 'scopes': scopes, 'all_scopes': False}


def verdict_of(selected, statuses):
    """Every selected test must report PASSED; a missing status fails."""
    failed = sorted(label for label in selected if statuses.get(label) != 'PASSED')
    return ('fail' if failed else 'pass' if selected else 'skipped'), failed


def population_query(architecture):
    """The cloud gate's tests: manual targets (gpu, quarantined files) are out,
    and on arm64 so is the runner lane, whose runner VM is amd64."""
    query = f'{TESTS} except attr(tags, "\\bmanual\\b", //...)'
    if architecture == 'arm64':
        query += ' except attr(tags, "\\brunner\\b", //...)'
    return query


def test_statuses(bep_path):
    statuses = {}
    if not bep_path.exists():
        return statuses
    for line in bep_path.read_text().splitlines():
        event = json.loads(line)
        summary = event.get('id', {}).get('testSummary')
        if summary:
            statuses[canonical_label(summary['label'])] = event.get('testSummary', {}).get('overallStatus',
                                                                                           'NO_STATUS')
    return statuses


def canonical_label(label):
    return label[1:] if label.startswith('@//') else label.removeprefix('@@')


def impacted_labels(output):
    """get-impacted-targets output: a JSON list of {label, distances} when it
    read dependency edges, else one label per line."""
    if output.lstrip().startswith('['):
        values = json.loads(output)
        if any(not isinstance(value, dict) or not isinstance(value.get('label'), str) for value in values):
            raise GateError('malformed impacted-target output')
        return {canonical_label(value['label']) for value in values}
    return {canonical_label(label) for label in output.split()}


class Workspace:
    """Git and Bazel on the runner. Every failure raises GateError."""

    def __init__(self, root=ROOT, bazel='bazel'):
        self.root, self.bazel = root, bazel

    def run(self, *command, capture=True, check=True):
        print('$ ' + shlex.join(map(str, command)), flush=True)
        environment = child_environment()
        result = subprocess.run(list(map(str, command)), cwd=self.root, text=True, env=environment,
                                stdout=subprocess.PIPE if capture else None)
        if check and result.returncode:
            raise GateError(f'{shlex.join(map(str, command))} exited {result.returncode}')
        return result.stdout if capture else result.returncode

    def ensure_history(self, head):
        """Hydrate the exact tested revision before queries or runner-lane tests."""
        from remote_gate import prepare_full_history
        if not prepare_full_history(head, root=self.root):
            raise GateError('full history preparation failed; refusing Bazel')

    def ensure_commit(self, sha):
        if not sha:
            raise GateError('empty base or head; the queue-aware refs were unavailable')
        present = subprocess.run(['git', 'cat-file', '-e', f'{sha}^{{commit}}'], cwd=self.root,
                                 stderr=subprocess.DEVNULL, env=child_environment())
        if present.returncode:
            self.run('git', 'fetch', '--quiet', '--no-tags', 'origin', sha)

    def changes(self, base, head):
        output = self.run('git', 'diff', '--name-status', '--no-renames', '-z', base, head)
        parts = output.split('\0')
        changed, deleted = set(), set()
        for status, path in zip(parts[0::2], parts[1::2]):
            (deleted if status.startswith('D') else changed).add(path)
        return changed, deleted

    def checkout(self, sha):
        self.run('git', 'checkout', '--quiet', '--force', '--end-of-options', sha, '--')

    def query(self, expression):
        return self.run(self.bazel, 'query', expression, '--output=label', '--keep_going=false').split()

    def tests(self, architecture='amd64'):
        return set(map(canonical_label, self.query(population_query(architecture))))

    def used_sources(self):
        labels = self.query('kind("source file", deps(//...))') + self.query('buildfiles(//...)')
        return {label.removeprefix('//').replace(':', '/').lstrip('/') for label in map(canonical_label, labels)
                if label.startswith('//')}

    def bazel_diff(self, base, head):
        """bazel-diff's impacted targets between two commits; it leaves each
        revision's dependency edges (deps.json) beside the hashes."""
        executable = Path(tempfile.gettempdir(), 'bazel-diff-v70.0.2-linux-amd64')
        if not executable.exists() or hashlib.sha256(executable.read_bytes()).hexdigest() != BAZEL_DIFF_SHA256:
            with urllib.request.urlopen(BAZEL_DIFF_URL, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != BAZEL_DIFF_SHA256:
                raise GateError('bazel-diff binary checksum mismatch')
            executable.write_bytes(data)
        executable.chmod(0o755)
        directory = Path(tempfile.mkdtemp(prefix='tessera-bazel-diff-'))
        hashes = {name: directory / f'{name}-hashes.json' for name in ['base', 'head']}
        edges = {name: directory / f'{name}-deps.json' for name in ['base', 'head']}
        impacted = directory / 'impacted.json'
        for name, sha in [('base', base), ('head', head)]:
            self.checkout(sha)
            self.run(executable, 'generate-hashes', '--workspacePath', self.root, '--bazelPath', self.bazel,
                     '--excludeExternalTargets', '--includeTargetType', '--depEdgesFile', edges[name], hashes[name])
        self.run(executable, 'get-impacted-targets', '--workspacePath', self.root, '--bazelPath', self.bazel,
                 '--targetType', 'Rule', '--startingHashes', hashes['base'], '--finalHashes', hashes['head'],
                 '--depEdgesFile', edges['head'], '--output', impacted)
        self.diff_files = {'hashes': hashes, 'edges': edges, 'executable': executable}
        return impacted_labels(impacted.read_text())

    def test(self, targets, config, metadata, log_name, extra=()):
        """Run targets; return (per-label status, per-test budget warnings)."""
        if not targets:
            return {}, []
        patterns = Path(tempfile.gettempdir(), f'{log_name}-targets.txt')
        patterns.write_text('\n'.join(targets) + '\n')
        bep = Path(tempfile.gettempdir(), f'{log_name}-bep.json')
        bep.unlink(missing_ok=True)
        self.run(self.bazel, 'test', f'--config={config}', '--config=ci', '--keep_going',
                 f'--target_pattern_file={patterns}', f'--build_event_json_file={bep}',
                 *diagnostic_logs(log_name), *extra,
                 *[f'--build_metadata={key}={value}' for key, value in metadata.items()],
                 capture=False, check=False)
        return test_statuses(bep), self.budget_warnings(targets)

    def budget_warnings(self, targets):
        warnings = []
        logs = self.root / 'bazel-testlogs'
        for label in targets:
            package, _, name = label.removeprefix('//').partition(':')
            for path in sorted((logs / package / name).glob('**/test.outputs/budget_warnings.json')):
                for row in json.loads(path.read_text())['warnings']:
                    warnings.append({'label': label, **row})
        return warnings

    def build(self, targets, config, metadata, log_name):
        if not targets:
            return True
        patterns = Path(tempfile.gettempdir(), f'{log_name}-build-targets.txt')
        patterns.write_text('\n'.join(targets) + '\n')
        return not self.run(self.bazel, 'build', f'--config={config}', '--config=ci', '--keep_going',
                            f'--target_pattern_file={patterns}', *diagnostic_logs(f'{log_name}-build'),
                            *[f'--build_metadata={key}={value}' for key, value in metadata.items()],
                            capture=False, check=False)

    def inputs_check(self):
        """Generated files current, lock export current, population parity."""
        problems = []
        for command in ([sys.executable, 'tools/bazel/build_files.py', '--check'],
                        [sys.executable, 'tools/bazel/requirements.py', '--check'],
                        [sys.executable, 'tools/ci/bazel_parity.py']):
            if self.run(*command, capture=False, check=False):
                problems.append(shlex.join(map(str, command)))
        return problems

    def graph_impact(self, base):
        """The graph-impact check's summary line and verdict, when the tool is present."""
        tool = self.root / 'tools/ci/graph_impact_tessera.py'
        if not tool.exists():
            return None
        result = subprocess.run([sys.executable, str(tool), 'check', '--base', base], cwd=self.root, text=True,
                                capture_output=True, env=child_environment())
        print(result.stdout + result.stderr, flush=True)
        return {'summary': result.stdout.strip().splitlines()[0] if result.stdout.strip() else '',
                'failures': result.stderr.strip().splitlines(), 'exit': result.returncode}


def diagnostic_logs(prefix):
    return [f'--remote_grpc_log={prefix}-{DIAGNOSTICS[0]}',
            f'--execution_log_compact_file={prefix}-{DIAGNOSTICS[1]}']


def write_result(result, directory, root=ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULT).write_text(json.dumps(result, indent=2) + '\n')
    for suffix in DIAGNOSTICS:
        for log in root.glob(f'*{suffix}'):
            log.replace(directory / log.name)


def scan_artifacts(directory):
    """Refuse raw event/command logs and scan the actual upload directory."""
    deferred = os.environ.get('BBRUN_FINAL_SCAN') == '1'
    names = ['BUILDBUDDY_API_KEY'] + sorted(
        name for name, value in os.environ.items() if value and name != 'BUILDBUDDY_API_KEY'
        and (name in SECRET_NAMES or name.endswith(('_API_KEY', '_TOKEN'))))
    arguments = [str(directory), *[arg for name in names for arg in ('--env', name)],
                 '--require', 'BUILDBUDDY_API_KEY']
    if not deferred and not os.environ.get('BUILDBUDDY_API_KEY'):
        print('secret scan blind: key unavailable', flush=True)
        check_no_secret_leak.main(arguments)
        return False
    forbidden = False
    for path in directory.rglob('*'):
        name = path.name.lower()
        if (path.is_file() or path.is_symlink()) and (
                'bep' in name or name.endswith('.binpb') or name == 'command.log' or 'command_log' in name):
            display = str(path.relative_to(directory))
            for variable in names:
                if value := os.environ.get(variable):
                    for form in check_no_secret_leak.secret_byte_forms(value):
                        display = display.replace(form.decode(errors='replace'), '')
            print(display, flush=True)
            path.unlink()
            forbidden = True
    if deferred:
        print('secret scan deferred to bbrun final scan', flush=True)
        return not forbidden
    status = check_no_secret_leak.main(arguments)
    if status:
        for path in directory.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    return status == 0 and not forbidden


def gate_pr(workspace, base, head, metadata):
    started = time.monotonic()
    for sha in [base, head]:
        workspace.ensure_commit(sha)
    workspace.checkout(head)
    workspace.ensure_history(head)
    changed, deleted = workspace.changes(base, head)
    used = workspace.used_sources()
    forced = fallback_files(changed | deleted)
    reads = {path for path in changed | deleted if path in used or path in deleted}
    impacted = set() if forced or not reads else workspace.bazel_diff(base, head)
    workspace.checkout(head)
    tests = workspace.tests('amd64')
    decision = gate_decision(changed, deleted, impacted, used, tests)
    selected_seconds = time.monotonic() - started
    statuses, warnings = workspace.test(decision['selected'], CONFIGS['amd64'], metadata, 'pr')
    tested_seconds = time.monotonic() - started
    verdict, failed = verdict_of(decision['selected'], statuses)
    inputs = workspace.inputs_check()
    if not workspace.build(decision['build'], CONFIGS['amd64'], metadata, 'pr'):
        inputs.append('bazel build of impacted targets failed: ' + ' '.join(decision['build']))
    return {**decision, 'verdict': verdict, 'failed': failed, 'budget_warnings': warnings, 'inputs': inputs,
            'graph_impact': workspace.graph_impact(base), 'changed': sorted(changed | deleted),
            'durations': {'selection': round(selected_seconds, 1),
                          'tests': round(tested_seconds - selected_seconds, 1),
                          'total': round(time.monotonic() - started, 1)}}


def gate_main(workspace, base, head, metadata):
    """Full run after a merge, on both architectures, plus what selection would have run."""
    started = time.monotonic()
    for sha in [base, head]:
        workspace.ensure_commit(sha)
    workspace.checkout(head)
    workspace.ensure_history(head)
    changed, deleted = workspace.changes(base, head)
    try:
        used = workspace.used_sources()
        forced = fallback_files(changed | deleted)
        impacted = set() if forced else workspace.bazel_diff(base, head)
        workspace.checkout(head)
        selection = gate_decision(changed, deleted, impacted, used, workspace.tests('amd64'))
    except GateError as error:
        workspace.checkout(head)
        selection = {'selection': 'error', 'reason': [str(error)], 'selected': sorted(workspace.tests('amd64'))}
    architectures = {}
    for architecture, config in CONFIGS.items():
        begun = time.monotonic()
        population = sorted(workspace.tests(architecture))
        statuses, warnings = workspace.test(population, config, metadata, f'main-{architecture}')
        verdict, failed = verdict_of(population, statuses)
        architectures[architecture] = {
            'verdict': verdict, 'failed': failed, 'tests': len(population), 'budget_warnings': warnings,
            'missed': sorted(label for label in failed if label not in selection['selected']),
            'seconds': round(time.monotonic() - begun, 1)}
    if any(arch['tests'] == 0 for arch in architectures.values()):
        raise GateError('an architecture executed no tests: ' + json.dumps(
            {name: arch['tests'] for name, arch in architectures.items()}))
    return {'selection': selection['selection'], 'reason': selection['reason'],
            'selected': selection['selected'], 'architectures': architectures,
            'inputs': workspace.inputs_check(), 'changed': sorted(changed | deleted),
            'durations': {'total': round(time.monotonic() - started, 1)}}


def passed(result):
    if result.get('error') or result.get('inputs'):
        return False
    if result['mode'] == 'main':
        return all(not arch['failed'] for arch in result['architectures'].values())
    impact = result.get('graph_impact')
    return result['verdict'] != 'fail' and not (impact and impact['exit'])


def main(argv=None, workspace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pr', 'main'])
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--branch', default='')
    args = parser.parse_args(argv)
    output = Path(os.environ.get('BUILDBUDDY_ARTIFACTS_DIRECTORY', ROOT / 'gate-artifacts'))
    metadata = {'COMMIT_SHA': args.head, 'BRANCH_NAME': args.branch}
    result = {'schema': SCHEMA, 'mode': args.mode, 'base': args.base, 'head': args.head}
    try:
        gate = gate_pr if args.mode == 'pr' else gate_main
        result.update(gate(workspace or Workspace(), args.base, args.head, metadata))
        result['error'] = None
    except GateError as error:
        result['error'] = str(error)
    try:
        write_result(result, output)
    finally:
        safe_artifacts = scan_artifacts(output)
    print(json.dumps({key: result.get(key) for key in ['selection', 'error', 'verdict', 'inputs', 'budget_warnings']}))
    return 0 if safe_artifacts and passed(result) else 1


if __name__ == '__main__':
    sys.exit(main())
