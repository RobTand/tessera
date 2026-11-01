"""Collect an exact gate assignment with one traversal of each directory."""
from __future__ import annotations

from fnmatch import fnmatch
import json
import os
from pathlib import Path, PurePosixPath

import pytest


class GateCollection:
    def __init__(self, root, files):
        self.root = root
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
        nodes = {item.nodeid for item in items}
        files = {item.path.relative_to(self.root).as_posix() for item in items}
        removed = []
        for target in targets:
            filename = target.split('::', 1)[0]
            # The bytes lane cannot collect torch modules; the torch lane
            # independently requires complete collection and validates these.
            if filename not in self.collected and filename not in files:
                continue
            if target not in nodes and target not in files:
                raise pytest.UsageError('quarantine is not an exact collected node or file: ' + target)
        kept = []
        for item in items:
            filename = item.path.relative_to(self.root).as_posix()
            (removed if item.nodeid in targets or filename in targets else kept).append(item)
        items[:] = kept
        if removed:
            config.hook.pytest_deselected(items=removed)
        yield

        items.sort(key=lambda item: self.ranks[item.path])
        for item in items:
            fixtures = item._fixtureinfo.name2fixturedefs.values()
            if any(definition.scope == 'module'
                   for definitions in fixtures for definition in definitions):
                group = 'module_' + item.path.relative_to(self.root).as_posix().replace('/', '.')
                item.add_marker(pytest.mark.xdist_group(name=group))

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
    config.pluginmanager.register(GateCollection(root, files), 'gate-assigned-collection')
