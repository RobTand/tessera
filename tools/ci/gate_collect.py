"""Collect an exact gate assignment with one traversal of each directory."""
from __future__ import annotations

from fnmatch import fnmatch
from copy import copy
import json
import os
from pathlib import Path, PurePosixPath

import pytest
from _pytest.runner import runtestprotocol

from gate_flaky import read_table
from gate_junit import test_name


class GateCollection:
    def __init__(self, root, files, config):
        self.root = root
        self.config = config
        try:
            self.flaky = read_table(root)
            for node in self.flaky:
                if not (root / node.split('::', 1)[0]).is_file():
                    raise ValueError('flaky node names a missing file: ' + node)
        except (OSError, ValueError) as error:
            raise pytest.UsageError(str(error)) from error
        self.ranks = {root / path: rank for rank, path in enumerate(files)}
        self.collected = set()
        self.directories = {parent for path in self.ranks for parent in path.parents
                            if parent == root or root in parent.parents}

    def pytest_ignore_collect(self, collection_path):
        if collection_path not in self.ranks and collection_path not in self.directories:
            return True
        return None

    def pytest_collectreport(self, report):
        path = report.nodeid.split('::', 1)[0]
        if path and report.outcome in ('passed', 'skipped'):
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            if candidate in self.ranks:
                self.collected.add(candidate.relative_to(self.root).as_posix())

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        targets = json.loads(os.environ.get('GATE_DESELECT', '[]'))
        timeouts = json.loads(os.environ.get('GATE_TEST_TIMEOUTS', '{}'))
        nodes = {item.nodeid for item in items}
        files = {item.path.relative_to(self.root).as_posix() for item in items}
        for node in self.flaky:
            filename = node.split('::', 1)[0]
            if filename in self.collected or filename in files:
                if node not in nodes:
                    raise pytest.UsageError('flaky mark is not an exact collected node: ' + node)
        removed = []
        for target in targets:
            filename = target.split('::', 1)[0]
            # The bytes lane cannot collect torch modules; the torch lane
            # independently requires complete collection and validates these.
            if filename not in self.collected and filename not in files:
                continue
            if target not in nodes and target not in self.collected and target not in files:
                raise pytest.UsageError('quarantine is not an exact collected node or file: ' + target)
        kept = []
        for item in items:
            filename = item.path.relative_to(self.root).as_posix()
            (removed if item.nodeid in targets or filename in targets else kept).append(item)
        items[:] = kept
        if removed:
            config.hook.pytest_deselected(items=removed)
        items.sort(key=lambda item: self.ranks[item.path])
        for item in items:
            if item.nodeid in timeouts:
                item.add_marker(pytest.mark.timeout(timeouts[item.nodeid]), append=False)
            fixtures = item._fixtureinfo.name2fixturedefs.values()
            if any(definition.scope == 'module'
                   for definitions in fixtures for definition in definitions):
                group = 'module_' + item.path.relative_to(self.root).as_posix().replace('/', '.')
                item.add_marker(pytest.mark.xdist_group(name=group))
        yield

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        node = test_name(item.nodeid)
        if node not in self.flaky:
            return None
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
        failed = next((report for report in reports if report.failed), None)
        if failed is not None:
            entry = self.flaky[node]
            # One nonfailure report carries the first attempt through xdist.
            # Use call phase so JUnit closes only on the final teardown, and
            # charge all of the first attempt's time to this node's budget.
            retry = copy(failed)
            retry.outcome, retry.when = 'rerun', 'call'
            retry.duration = sum(report.duration for report in reports)
            retry.gate_flaky_retry = (
                f"GATE_FLAKY_RETRY: {node} issue #{entry['issue']} "
                f"retry 1/1: {entry['reason']}")
            item.ihook.pytest_runtest_logreport(report=retry)
            item.user_properties.extend([('gate_flaky_issue', entry['issue']),
                                         ('gate_flaky_attempts', 2)])
            reports = runtestprotocol(item, nextitem=nextitem, log=False)
        for report in reports:
            item.ihook.pytest_runtest_logreport(report=report)
        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        return True

    @pytest.hookimpl(tryfirst=True)
    def pytest_report_teststatus(self, report):
        if getattr(report, 'gate_flaky_retry', None):
            return 'rerun', 'R', 'RERUN'

    def pytest_runtest_logreport(self, report):
        message = getattr(report, 'gate_flaky_retry', None)
        if not message or hasattr(self.config, 'workerinput'):
            return
        terminal = self.config.pluginmanager.get_plugin('terminalreporter')
        if terminal:
            terminal.write_line(message)
        else:
            print(message, flush=True)
        if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
            with Path(summary).open('a') as handle:
                handle.write('\n' + message + '\n')

    def pytest_collection_finish(self, session):
        destination = os.environ.get('GATE_COLLECTION_DIR')
        worker = os.environ.get('PYTEST_XDIST_WORKER')
        if destination and worker:
            counts = {path.relative_to(self.root).as_posix(): 0 for path in self.ranks}
            for item in session.items:
                counts[item.path.relative_to(self.root).as_posix()] += 1
            path = Path(destination) / f'collection.{worker}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'collected': sorted(self.collected), 'items': counts}) + '\n')


@pytest.hookimpl(specname='pytest_configure')
def pytest_gate_configure(config):
    try:
        files = json.loads(os.environ['GATE_ASSIGNED_FILES'])
        if (not isinstance(files, list) or not files or
                any(not isinstance(path, str) or not path or
                    PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                    for path in files) or len(files) != len(set(files))):
            raise ValueError('assignment must contain unique repository-relative files')
        root = Path(config.rootpath)
        if any(not (root / path).is_file() for path in files):
            raise ValueError('an assigned test file is missing')
    except (KeyError, ValueError, TypeError) as error:
        raise pytest.UsageError('invalid gate collection assignment: ' + str(error)) from error
    # Match explicit unusual filenames without assertion-rewriting every imported
    # production dependency, which would change Python's bytecode policy.
    for filename in sorted({Path(path).name for path in files}):
        if not any(fnmatch(filename, pattern) for pattern in config.getini('python_files')):
            config.addinivalue_line('python_files', filename)
    config.pluginmanager.register(GateCollection(root, files, config), 'gate-assigned-collection')
