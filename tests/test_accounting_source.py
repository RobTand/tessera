"""Explicit pricing roots cannot borrow another root's cached accountant."""
import importlib
from pathlib import Path
import sys

import pytest
from tessera.errors import TesseraError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from _accounting_source import accountant

from accounting_source_support import (
    isolated_prismaquant_imports,
    package_roots,
)


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
