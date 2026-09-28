"""The Tessera code a cell's ``serving_source_sha256`` names.

The digest is only useful if it cannot be SHORT: a source file it skipped is
code that can change under a cell without moving the digest. These tests hold
it to every source file in the package, check it is a function of those files'
paths and bytes and nothing else, and check the contract's grammar for it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tessera.serving.contract import cell_runtime_code, load_serving_contract, validate_serving_contract
from tessera.serving.source_identity import (
    SOURCE_IDENTITY_ALGORITHM,
    serving_source_files,
    serving_source_sha256,
)

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "tessera"


def _copy(tmp_path, name="src"):
    root = tmp_path / name
    shutil.copytree(PACKAGE, root / "tessera", ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_the_digest_is_a_sha256_named_by_its_algorithm():
    digest = serving_source_sha256(SRC)
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert serving_source_sha256(SRC) == digest
    assert SOURCE_IDENTITY_ALGORITHM == "tessera.package_source.v1"


def test_every_python_and_native_source_is_covered_and_nothing_else():
    covered = {path.relative_to(SRC).as_posix() for path in serving_source_files(SRC)}
    python = {path.relative_to(SRC).as_posix() for path in PACKAGE.rglob("*.py")
              if "__pycache__" not in path.parts}
    assert python <= covered
    assert "tessera/serving/csrc/window_gemv.cu" in covered
    assert "tessera/serving/runtime_contract.json" not in covered
    assert all(Path(name).suffix in {".py", ".cu", ".cuh", ".cpp", ".c", ".cc", ".h", ".hpp"}
               for name in covered)


def test_the_default_root_is_the_package_this_module_came_from():
    assert serving_source_sha256() == serving_source_sha256(SRC)


def test_the_digest_depends_on_the_files_not_where_they_sit(tmp_path):
    assert serving_source_sha256(_copy(tmp_path)) == serving_source_sha256(SRC)


def test_bytecode_and_data_files_do_not_move_the_digest(tmp_path):
    root = _copy(tmp_path)
    base = serving_source_sha256(root)
    cache = root / "tessera" / "serving" / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "scheme.cpython-314.pyc").write_bytes(b"\x00bytecode")
    contract = root / "tessera" / "serving" / "runtime_contract.json"
    contract.write_text(contract.read_text() + "\n")
    fresh = tmp_path / "fresh"
    shutil.copytree(root, fresh)
    assert serving_source_sha256(fresh) == base


@pytest.mark.parametrize("relative", [
    "tessera/serving/fp8_route.py",          # a serving route
    "tessera/kernel_a4.py",                  # reached by a function-local import
    "tessera/serving/csrc/window_gemv.cu",   # a native source
])
def test_a_source_edit_moves_the_digest(tmp_path, relative):
    root = _copy(tmp_path)
    base = serving_source_sha256(root)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n// edit\n" if path.suffix == ".cu"
                     else path.read_bytes() + b"\n# edit\n")
    edited = tmp_path / "edited"
    shutil.copytree(root, edited)
    assert serving_source_sha256(edited) != base


def test_a_tree_without_the_serving_plugin_is_refused(tmp_path):
    (tmp_path / "tessera").mkdir()
    with pytest.raises(FileNotFoundError, match="serving plugin"):
        serving_source_sha256(tmp_path)


def _cell_with(**runtime):
    contract = json.loads(json.dumps(load_serving_contract()))
    contract["lane_eligibility"]["cells"][0]["runtime"].update(runtime)
    return contract


def test_the_contract_accepts_a_cell_that_names_its_code():
    contract = _cell_with(tessera_commit="a" * 40, serving_source_sha256="b" * 64)
    validate_serving_contract(contract)
    cell = contract["lane_eligibility"]["cells"][0]
    assert cell_runtime_code(cell) == ("a" * 40, "b" * 64)


def test_no_published_cell_names_its_code_yet():
    cells = load_serving_contract()["lane_eligibility"]["cells"]
    assert all(cell_runtime_code(cell) is None for cell in cells)


@pytest.mark.parametrize("runtime, match", [
    ({"tessera_commit": "a" * 40}, "both fields or neither"),
    ({"serving_source_sha256": "b" * 64}, "both fields or neither"),
    ({"tessera_commit": "A" * 40, "serving_source_sha256": "b" * 64}, "40-hex"),
    ({"tessera_commit": "a" * 40, "serving_source_sha256": "b" * 63}, "64 lowercase hex"),
    ({"tessera_commit": "a" * 40, "serving_source_sha256": None}, "64 lowercase hex"),
])
def test_the_contract_refuses_a_half_or_malformed_code_scope(runtime, match):
    with pytest.raises(ValueError, match=match):
        validate_serving_contract(_cell_with(**runtime))
