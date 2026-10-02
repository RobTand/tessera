"""Bind an explicit PrismaQuant pricing root to its canonical import origin."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

from tessera.errors import TesseraError


class AccountingSourceError(TesseraError):
    """The requested accountant cannot be attributed to its source root."""


def _require(condition, reason):
    if not condition:
        raise AccountingSourceError("PrismaQuant accounting source refused: " + reason)


def _origin(module, expected, name):
    declared = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    _require(declared and origin, f"{name} has no canonical module origin")
    file = Path(declared)
    _require(file.is_absolute() and file.resolve() == file
             and Path(origin) == file and file == expected,
             f"{name} was imported from {declared}, expected {expected}; "
             "use one qualified root in this process or a fresh process")


def accountant(root):
    """Reuse a same-origin import, refuse a foreign cached one before pricing.

    No module is evicted and no pricing implementation is copied. A caller
    that needs different sources uses separate processes with explicit roots.
    """
    try:
        root = Path(root).resolve(strict=True)
    except OSError as error:
        raise AccountingSourceError(f"PrismaQuant accounting source root is unavailable: {root}") from error
    package = root / "prismaquant/__init__.py"
    expected = root / "prismaquant/tessera_formats.py"
    _require(package.is_file() and expected.is_file()
             and package.resolve() == package and expected.resolve() == expected,
             f"{root} does not contain canonical PrismaQuant accounting files")
    for name, path in (("prismaquant", package), ("prismaquant.tessera_formats", expected)):
        cached = sys.modules.get(name)
        if cached is not None:
            _origin(cached, path, name)
    cached = sys.modules.get("prismaquant.tessera_formats")
    if cached is not None:
        return cached
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        module = importlib.import_module("prismaquant.tessera_formats")
        _origin(sys.modules["prismaquant"], package, "prismaquant")
        _origin(module, expected, "prismaquant.tessera_formats")
        return module
    except AccountingSourceError:
        raise
    except Exception as error:
        raise AccountingSourceError(f"PrismaQuant accounting source {root} is not importable: {error}") from error
    finally:
        sys.path[:] = previous
