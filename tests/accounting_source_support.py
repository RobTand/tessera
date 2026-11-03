"""Shared fixtures that isolate PrismaQuant packages and pricing roots."""
import importlib
import sys

import pytest


@pytest.fixture
def isolated_prismaquant_imports(monkeypatch):
    """Keep each test's explicit pricing root separate from cached imports."""
    saved = {key: value for key, value in sys.modules.items() if key == "prismaquant" or key.startswith("prismaquant.")}
    for key in saved:
        monkeypatch.delitem(sys.modules, key)
    yield
    for key in list(sys.modules):
        if key == "prismaquant" or key.startswith("prismaquant."):
            sys.modules.pop(key)
    sys.modules.update(saved)


@pytest.fixture
def package_roots(tmp_path, monkeypatch, isolated_prismaquant_imports):
    roots = []
    for name, price in (("first", 3), ("second", 7)):
        root = tmp_path / name
        package = root / "prismaquant"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "tessera_formats.py").write_text(f"from fractions import Fraction\ndef artifact_bpp(*args, **kwargs):\n    return Fraction({price})\n")
        roots.append(root)
    monkeypatch.syspath_prepend(str(roots[0]))
    importlib.import_module("prismaquant.tessera_formats")
    yield roots
