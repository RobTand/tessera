"""Explicit pricing roots cannot borrow another root's cached accountant."""
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest
from tessera.errors import TesseraError


@pytest.fixture
def package_roots(tmp_path, monkeypatch):
    saved = {key: value for key, value in sys.modules.items() if key == "prismaquant" or key.startswith("prismaquant.")}
    for key in saved:
        monkeypatch.delitem(sys.modules, key)
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
    for key in list(sys.modules):
        if key == "prismaquant" or key.startswith("prismaquant."):
            sys.modules.pop(key)
    sys.modules.update(saved)


def planner():
    path = Path(__file__).resolve().parents[1] / "experiments/plan_from_layer_config.py"
    spec = importlib.util.spec_from_file_location("_accounting_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_foreign_root_is_refused_before_pricing(package_roots):
    first, second = package_roots
    with pytest.raises(TesseraError, match="accounting source"):
        planner().charged_bits(second, "E4M3", 1024, (8, 8))


def test_same_root_uses_the_existing_accountant(package_roots):
    first, _ = package_roots
    assert planner().charged_bits(first, "E4M3", 1024, (8, 8)) == 192


def test_a_foreign_parent_package_is_refused_even_without_a_cached_accountant(package_roots):
    _, second = package_roots
    sys.modules.pop("prismaquant.tessera_formats")
    with pytest.raises(TesseraError, match="accounting source"):
        planner().charged_bits(second, "E4M3", 1024, (8, 8))


def test_an_explicit_missing_root_does_not_omit_accounting(tmp_path):
    with pytest.raises(TesseraError, match="accounting source"):
        planner().charged_bits(tmp_path / "absent", "E4M3", 1024, (8, 8))


def test_unspecified_accounting_remains_explicitly_absent():
    assert planner().charged_bits(None, "E4M3", 1024, (8, 8)) is None


def test_same_origin_alias_root_is_resolved_before_import(package_roots, tmp_path):
    first, _ = package_roots
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    assert planner().charged_bits(alias, "E4M3", 1024, (8, 8)) == 192
