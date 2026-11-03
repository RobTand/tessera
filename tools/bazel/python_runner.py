"""Run one gate test file in one lane as tools/ci/gate_run.py runs a hosted shard.

pytest runs from the repository root with tests/conftest.py, pyproject.toml,
the gate's environment, its collection plugin, its node-level quarantine and
its per-test timeouts. The per-test budgets in tools/ci/gate_budget.json apply
with gate_run's tolerance. A file Bazel shards runs one share of its tests,
keeping the tests that share a module-scoped fixture together. After the run,
test.outputs/inventory.json records each node's outcome for parity, and the
file's seconds. A run that fails only on a per-test budget writes
test.outputs/budget_overrun.json, so the gate may rerun it once; an assertion
failure never does.

With TS_TRACE=1 (`--test_env=TS_TRACE=1`), every Python process the test
starts records the repository files it imports or opens, and the inventory
lists them as `executed`: the dynamic floor that static selection is measured
against (#1588).

The bytes lane has no torch. A file that needs a third-party package that lane
lacks is not collected there (tests/conftest.py decides, as on the hosted
gate) and passes with no tests; the torch lane must collect every file.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

# Import siblings by path: tools/ci has modules with common names.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import checkout_of, prepare_remote_test  # noqa: E402

PYTEST_TIMEOUT_SECONDS = 300

TRACE_SITE = '''
import atexit
import os
import sys

_ROOT = os.environ['TS_TRACE_ROOT']
_OPENED = set()


def _audit(event, args):
    if event == 'open' and args:
        try:
            path = os.fspath(args[0])
        except TypeError:
            return
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        path = os.path.abspath(path)
        if path.startswith(_ROOT):
            _OPENED.add(path)


sys.addaudithook(_audit)


@atexit.register
def _record():
    files = set(_OPENED)
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and os.path.abspath(name).startswith(_ROOT):
            files.add(os.path.abspath(name))
    with open(os.path.join(os.environ['TS_TRACE_DIR'], f'{os.getpid()}.txt'), 'a') as handle:
        handle.write(''.join(path + '\\n' for path in sorted(files)))
'''


def traced(trace, root):
    """Repository-relative files every traced process imported or opened."""
    found = set()
    for record in trace.glob('*.txt'):
        for line in record.read_text().splitlines():
            if line.startswith(str(root) + os.sep):
                found.add(line[len(str(root)) + 1:])
    return sorted(found)

SHARD_PLUGIN = '''
import os
import zlib

def _group(item):
    fixtures = item._fixtureinfo.name2fixturedefs.values()
    if any(definition.scope == 'module' for definitions in fixtures for definition in definitions):
        return item.path.name
    return item.nodeid

def pytest_collection_modifyitems(config, items):
    index, total = int(os.environ['TS_SHARD_INDEX']), int(os.environ['TS_TOTAL_SHARDS'])
    groups = sorted({_group(item) for item in items})
    mine = {group for position, group in enumerate(groups) if position % total == index}
    keep = [item for item in items if _group(item) in mine]
    config.hook.pytest_deselected(items=[item for item in items if _group(item) not in mine])
    items[:] = keep
'''


def outcomes(junit):
    """Node ID to passed, skipped or failed, from an xunit1 report."""
    found = {}
    for case in ET.parse(junit).iter('testcase'):
        filename = case.get('file', '').removeprefix('./')
        module = filename.removesuffix('.py').replace('/', '.')
        classes = case.get('classname', '')[len(module):].lstrip('.').replace('.', '::')
        node = filename + '::' + (classes + '::' if classes else '') + case.get('name', '')
        if case.find('failure') is not None or case.find('error') is not None:
            found[node] = 'failed'
        elif case.find('skipped') is not None:
            found[node] = 'skipped' if found.get(node) != 'failed' else 'failed'
        else:
            found.setdefault(node, 'passed')
    return found


def budget(root, argument):
    """The budget rules and this file's policy slice, which
    tools/bazel/build_files.py passes as --budget."""
    sys.path.insert(0, str(root / 'tools/ci'))
    import gate_budget
    return gate_budget, gate_budget.limits(json.loads(argument))


def main():
    root = Path.cwd().resolve()
    if os.environ.get('TS_BAZEL_LANE') == 'runner':
        # Hosted shards run pytest in a git checkout; so does the runner lane.
        root = checkout_of(root)
    path, lane, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    deselect = [argument.removeprefix('--deselect=') for argument in extra if argument.startswith('--deselect=')]
    budget_argument = next(argument.removeprefix('--budget=') for argument in extra if argument.startswith('--budget='))
    extra = [argument for argument in extra if not argument.startswith(('--deselect=', '--budget='))]
    scratch, environment = prepare_remote_test(root)
    private = Path(os.environ['TEST_TMPDIR'])
    outputs = Path(os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', private))
    outputs.mkdir(parents=True, exist_ok=True)
    junit = Path(os.environ.get('XML_OUTPUT_FILE') or outputs / 'junit.xml')
    surface = outputs / 'surface.json'
    gate_budget, config = budget(root, budget_argument)
    environment['GATE_ASSIGNED_FILES'] = json.dumps([path])
    environment['GATE_DESELECT'] = json.dumps(deselect)
    environment['GATE_TEST_TIMEOUTS'] = json.dumps({
        node: limit * gate_budget.BUDGET_ENFORCEMENT_TOLERANCE for node, limit in config['slow'].items()
        if node.split('::', 1)[0] == path and limit * gate_budget.BUDGET_ENFORCEMENT_TOLERANCE > PYTEST_TIMEOUT_SECONDS})
    plugins = ['-p', 'gate_collect']
    if os.environ.get('TEST_TOTAL_SHARDS'):
        Path(os.environ['TEST_SHARD_STATUS_FILE']).touch()
        (private / 'bazel_shard.py').write_text(SHARD_PLUGIN)
        environment['PYTHONPATH'] = str(private) + os.pathsep + environment['PYTHONPATH']
        environment['TS_SHARD_INDEX'] = os.environ['TEST_SHARD_INDEX']
        environment['TS_TOTAL_SHARDS'] = os.environ['TEST_TOTAL_SHARDS']
        plugins += ['-p', 'bazel_shard']
    # The repository root, as gate_run passes it: gate_collect keeps only the
    # assigned file, and tests/conftest.py's collect_ignore applies, which a
    # file named on the command line would bypass.
    command = [sys.executable, '-m', 'pytest', str(root), '-q', '-ra', '--durations=10',
               f'--timeout={PYTEST_TIMEOUT_SECONDS}', '--surface-json=' + str(surface),
               '--rootdir=' + str(root), '--junitxml=' + str(junit), '-o', 'junit_family=xunit1',
               '-o', 'cache_dir=' + str(private / 'pytest-cache'), *plugins, *extra]
    trace = None
    if os.environ.get('TS_TRACE') == '1':
        trace, site = private / 'trace', private / 'trace-site'
        trace.mkdir()
        site.mkdir()
        (site / 'sitecustomize.py').write_text(TRACE_SITE)
        environment['PYTHONPATH'] = str(site) + os.pathsep + environment['PYTHONPATH']
        environment['TS_TRACE_DIR'], environment['TS_TRACE_ROOT'] = str(trace), str(root)
    started = time.monotonic()
    try:
        status = subprocess.run(command, cwd=root, env=environment, check=False).returncode
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record = {'file': path, 'lane': lane, 'shard': os.environ.get('TEST_SHARD_INDEX'),
              'seconds': round(time.monotonic() - started, 3)}
    if trace is not None:
        record['executed'] = traced(trace, root)
    try:
        population = json.loads(surface.read_text())
        not_collected = {name if name.startswith('tests/') else 'tests/' + name
                         for name in population.get('not_collected', [])}
    except (OSError, ValueError) as error:
        print(f'no CPU population for {path}: {error}', file=sys.stderr)
        return status or 1
    if population.get('cuda_surface', {}).get('executed'):
        print(f'{path} exercised CUDA in a CPU lane', file=sys.stderr)
        return 1
    if path in not_collected:
        if lane == 'torch':
            print(f'the torch lane must collect {path}', file=sys.stderr)
            return 1
        print(f'{path} is not collected without torch, as on the hosted no-torch lane')
        (outputs / 'inventory.json').write_text(json.dumps(dict(record, collected=False, nodes={})))
        return 0 if status in (0, 5) else status
    try:
        nodes = outcomes(junit)
    except (OSError, ET.ParseError) as error:
        print(f'no JUnit verdict for {path}: {error}', file=sys.stderr)
        return status or 1
    if any(outcome == 'failed' for outcome in nodes.values()):
        status = status or 1
    if status == 5:
        # gate_run's rule: no collected test, such as a module-level
        # importorskip, or a shard that holds no group of a small file.
        status = 0
    if status == 0 and any(outcome != 'skipped' for outcome in nodes.values()):
        try:
            timings = gate_budget.timings(junit, [path], config, require_all=False)
        except ValueError as error:
            print(f'no per-test timings for {path}: {error}', file=sys.stderr)
            return 1
        overruns = [row for row in timings
                    if row['seconds'] > row['budget_seconds'] * gate_budget.BUDGET_ENFORCEMENT_TOLERANCE]
        for row in overruns:
            print(f"Per-test time budget exceeded: {row['nodeid']}: {row['seconds']:.3f}s > "
                  f"{row['budget_seconds']:.3f}s policy", file=sys.stderr)
        status = 1 if overruns and config.get('enforce', False) else 0
        if status:
            # Only a budget failed: every assertion passed. The gate reruns
            # such a test once, uncached, before its verdict stands.
            (outputs / 'budget_overrun.json').write_text(json.dumps(
                {'file': path, 'lane': lane, 'overruns': overruns}))
    if status == 0:
        (outputs / 'inventory.json').write_text(json.dumps(
            dict(record, collected=True, nodes=nodes, skip_reasons=population.get('skip_reasons', {}))))
    return status


if __name__ == '__main__':
    sys.exit(main())
