"""Collect an exact gate assignment with one traversal of each directory."""
from __future__ import annotations

from fnmatch import fnmatch
import json
import os
from pathlib import Path, PurePosixPath

import pytest


class GateCollection:
    def __init__(self, root, files):
        self.ranks = {root / path: rank for rank, path in enumerate(files)}
        self.directories = {parent for path in self.ranks for parent in path.parents
                            if parent == root or root in parent.parents}

    def pytest_ignore_collect(self, collection_path):
        if collection_path not in self.ranks and collection_path not in self.directories:
            return True
        return None

    def pytest_collection_modifyitems(self, items):
        items.sort(key=lambda item: self.ranks[item.path])


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
