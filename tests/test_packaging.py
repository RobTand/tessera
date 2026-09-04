"""What a wheel ships, checked from the tree the wheel is cut from.

An editable install reads the source tree and cannot notice a data file the
package-data table forgot; the first wheel would, at a user's first call,
as a FileNotFoundError from a JIT build or a missing runtime contract.  These
tests hold the table to the tree and the tree to the table, and hold the
plugin entry point to a name that resolves.  ``tools/check_wheel.py`` is the
other half: the same properties asserted on a built wheel, in CI, before it
is published.  Torch-free by construction so the bytes-only job runs them.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

#: Every non-Python file a route or a producer opens at run time.  A new one
#: belongs in this list AND in pyproject's package-data table; the tests below
#: refuse either half on its own.
RUNTIME_DATA_SUFFIXES = (".cu", ".json")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _package_data() -> dict[str, list[str]]:
    return _pyproject()["tool"]["setuptools"]["package-data"]


def _package_dir(package: str) -> Path:
    return SRC.joinpath(*package.split("."))


def test_every_package_data_glob_matches_a_file():
    """A glob that matches nothing is a package-data line that ships nothing."""
    for package, globs in _package_data().items():
        base = _package_dir(package)
        assert base.is_dir(), f"package-data names {package!r}, which is not a package"
        for pattern in globs:
            assert list(base.glob(pattern)), (
                f"package-data {package!r}: {pattern!r} matches no file under {base}"
            )


def test_every_runtime_data_file_in_the_tree_is_declared():
    """The other direction: a data file the table does not name is not in
    the wheel, and an editable install would never say so."""
    table = _package_data()
    undeclared = []
    for path in sorted(SRC.joinpath("tessera").rglob("*")):
        if not path.is_file() or path.suffix not in RUNTIME_DATA_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        declared = any(
            path in _package_dir(package).glob(pattern)
            for package, globs in table.items()
            for pattern in globs
        )
        if not declared:
            undeclared.append(str(path.relative_to(SRC)))
    assert not undeclared, (
        "runtime data files no package-data glob covers (a wheel would omit "
        f"them): {undeclared}"
    )


def test_the_vllm_plugin_entry_point_resolves_to_a_callable():
    """vLLM discovers the plugin by this entry point and calls what it names.
    The name has to resolve without vLLM present: registration imports vLLM
    lazily, inside the call, and this test runs where there is none."""
    entry_points = _pyproject()["project"]["entry-points"]["vllm.general_plugins"]
    assert list(entry_points) == ["tessera"]
    module_name, _, attribute = entry_points["tessera"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute)), entry_points["tessera"]


def test_the_runtime_contract_is_read_from_inside_the_serving_package():
    """The contract must resolve through importlib.resources to the file the
    package-data table ships, never through repo-root arithmetic, so a wheel,
    an editable install and a checkout read the same bytes."""
    from tessera.serving import contract

    path = contract.contract_path()
    assert path.is_file()
    shipped = _package_dir("tessera.serving") / contract.CONTRACT_FILENAME
    assert Path(str(path)).resolve() == shipped.resolve()
    assert contract.load_serving_contract()["schema"] == contract.CONTRACT_SCHEMA


def test_the_kernel_sources_resolve_as_package_resources():
    """The routes JIT-build these through importlib.resources; both csrc
    directories have to be reachable that way, not only as tree paths."""
    from importlib import resources

    for package, name in (
        ("tessera", "csrc/window_gemv.cu"),
        ("tessera.serving", "csrc/window_gemv.cu"),
        ("tessera.serving", "csrc/tessera_nvfp4.cu"),
    ):
        assert resources.files(package).joinpath(name).is_file(), f"{package}: {name}"
