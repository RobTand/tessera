"""Tessera's serving half depends on no other runtime.

The plugin under ``tessera.serving`` is a rewrite-in-place of Gridbook's
Tessera lane.  The whole point of the move is that Tessera now owns the
runtime that reads its bytes: a serve installs ONE package, and a producer
reading the packaged contract pulls in no second quantization stack.  An
``import gridbook`` anywhere under ``src/tessera`` would silently reinstate
that dependency, so it is a test, not a convention.

The check is on IMPORTS, parsed with ``ast`` -- not on the substring.  The
docstrings deliberately say where this code came from (``ops``, ``telemetry``,
``flags`` and ``window`` all name Gridbook to explain a design decision), and
forbidding the word would forbid the history.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "tessera"
FORBIDDEN = "gridbook"


def _python_files(root: Path):
    return sorted(p for p in root.rglob("*.py"))


def _imported_names(tree: ast.AST):
    """Every module name any import statement in ``tree`` names."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # ``module`` is None for a bare relative import (``from . import x``).
            names.append(node.module or "")
            names.extend(f"{node.module or ''}.{alias.name}" for alias in node.names)
    return names


def test_the_tessera_package_never_imports_gridbook():
    files = _python_files(SRC)
    assert files, f"no python sources found under {SRC}"
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            root = name.split(".")[0]
            if root == FORBIDDEN:
                offenders.append(f"{path}: {name}")
    assert offenders == []


def test_no_serving_import_statement_mentions_gridbook_at_all():
    """Stricter, on the serving package: not even as a dotted component.

    A prose mention in a docstring is allowed and expected; an import that
    merely CONTAINS the name (a shim, a compat module, a vendored path) is
    not, because it would be a dependency the plugin's own docstring denies.
    """
    files = _python_files(SRC / "serving")
    assert files, "the serving package has no python sources"
    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported_names(tree):
            if FORBIDDEN in name.lower():
                offenders.append(f"{path}: {name}")
    assert offenders == []


@pytest.mark.parametrize("module", ["tessera.serving.contract", "tessera.serving.scheme"])
def test_the_producer_facing_modules_import_no_torch(module):
    """A producer reads the packaged contract on a machine with no GPU.

    ``contract`` and the ``scheme`` vocabulary it validates against are the two
    modules a PrismaQuant-side gate imports, so neither may pull torch in.
    """
    path = SRC / "serving" / f"{module.rsplit('.', 1)[1]}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = {name.split(".")[0] for name in _imported_names(tree)}
    assert "torch" not in roots and "vllm" not in roots
