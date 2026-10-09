"""Explicit pricing roots cannot borrow another root's cached accountant."""
import importlib
from pathlib import Path
import sys

import pytest
from tessera.errors import TesseraError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _accounting_source import accountant


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


def test_explicit_foreign_root_is_refused_before_pricing(package_roots):
    first, second = package_roots
    with pytest.raises(TesseraError, match="accounting source"):
        accountant(second)


def test_same_root_uses_the_existing_accountant(package_roots):
    first, _ = package_roots
    assert accountant(first) is sys.modules["prismaquant.tessera_formats"]


def test_a_foreign_parent_package_is_refused_even_without_a_cached_accountant(package_roots):
    _, second = package_roots
    sys.modules.pop("prismaquant.tessera_formats")
    with pytest.raises(TesseraError, match="accounting source"):
        accountant(second)


def test_an_explicit_missing_root_does_not_omit_accounting(tmp_path):
    with pytest.raises(TesseraError, match="accounting source"):
        accountant(tmp_path / "absent")


def test_a_fresh_accountant_uses_the_explicit_root_and_restores_the_import_path(package_roots):
    _, second = package_roots
    sys.modules.pop("prismaquant.tessera_formats")
    sys.modules.pop("prismaquant")
    previous = list(sys.path)
    module = accountant(second)
    assert module.__file__ == str(second / "prismaquant/tessera_formats.py")
    assert module.artifact_bpp() == 7
    assert sys.path == previous


def test_same_origin_alias_root_is_resolved_before_import(package_roots, tmp_path):
    first, _ = package_roots
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    assert accountant(alias) is sys.modules["prismaquant.tessera_formats"]
