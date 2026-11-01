#!/usr/bin/env python3
"""Inventory test modules naming documentation inputs without importing them."""
from __future__ import annotations

import ast
from collections import defaultdict, deque
import json
from pathlib import Path


def documentation_readers(root: Path, files: list[str]) -> list[str]:
    readers = set()
    sources = set(files) | {
        path.relative_to(root).as_posix() for path in (root / 'tests').rglob('*.py')
    }
    aliases = defaultdict(set)
    for path in sources:
        parts = list(Path(path).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        # pytest can import a helper from its package or its test directory.
        # Keep every candidate when those spellings are ambiguous.
        for start in range(len(parts)):
            aliases['.'.join(parts[start:])].add(path)
    importers = defaultdict(set)
    for path in sorted(sources):
        tree = ast.parse((root / path).read_text(), filename=path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ''
                if node.level:
                    package = Path(path).parent.parts
                    prefix = '.'.join((*package[:len(package) - node.level + 1], prefix)).rstrip('.')
                names = [prefix, *(f'{prefix}.{alias.name}' for alias in node.names)]
            for name in names:
                parts = name.split('.')
                for end in range(1, len(parts) + 1):
                    for target in aliases.get('.'.join(parts[:end]), ()):
                        importers[target].add(path)
        # Citations in comments/docstrings are not file inputs. Keep all other
        # literals, including paths passed through helpers and receipt records;
        # requiring a local read_text call would miss those consumers.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
                node.body = node.body[1:]
        if any(isinstance(node, ast.Constant) and isinstance(node.value, str) and
               node.value.endswith(('.md', '.rst', 'issues-snapshot.json'))
               for node in ast.walk(tree)):
            readers.add(path)
    pending = deque(sorted(readers))
    while pending:
        for importer in sorted(importers[pending.popleft()] - readers):
            readers.add(importer)
            pending.append(importer)
    return sorted(readers & set(files))


if __name__ == '__main__':
    root = Path.cwd()
    print(json.dumps(documentation_readers(root, sorted(
        str(path.relative_to(root)) for path in (root / 'tests').glob('test_*.py'))), indent=2))
