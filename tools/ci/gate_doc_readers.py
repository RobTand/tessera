#!/usr/bin/env python3
"""Inventory test modules naming documentation inputs without importing them."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def documentation_readers(root: Path, files: list[str]) -> list[str]:
    readers = []
    for path in files:
        tree = ast.parse((root / path).read_text(), filename=path)
        # Citations in comments/docstrings are not file inputs. Keep all other
        # literals, including paths passed through helpers and receipt records;
        # requiring a local read_text call would miss those consumers.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
                node.body = node.body[1:]
        if any(isinstance(node, ast.Constant) and isinstance(node.value, str) and
               node.value.endswith(('.md', '.rst', 'issues-snapshot.json'))
               for node in ast.walk(tree)):
            readers.append(path)
    return readers


if __name__ == '__main__':
    root = Path.cwd()
    print(json.dumps(documentation_readers(root, sorted(
        str(path.relative_to(root)) for path in (root / 'tests').glob('test_*.py'))), indent=2))
