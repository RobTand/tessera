"""The host runtime seal follows installed aliases and measures actual bytes."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("suite_runtime", ROOT / "tools/_suite_runtime.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    module = load()
    stdlib, site, vendor = (tmp_path / name for name in ("stdlib", "site", "vendor"))
    for directory in (stdlib, site, vendor): directory.mkdir()
    executable = tmp_path / "python"; executable.write_bytes(b"interpreter")
    library = tmp_path / "libpython.so"; library.write_bytes(b"runtime-library")
    (stdlib / "json.py").write_bytes(b"stdlib")
    for name in ("pytest", "pytest-xdist"):
        info = site / (name + ".dist-info"); info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    (vendor / "dependency.py").write_bytes(b"before")
    (site / "vendor").symlink_to(vendor, target_is_directory=True)
    monkeypatch.setattr(module.sys, "executable", str(executable))
    monkeypatch.setattr(module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(module.sysconfig, "get_path", lambda key: str(stdlib if key == "stdlib" else site))
    monkeypatch.setattr(module.sysconfig, "get_config_var", lambda key: str(tmp_path) if key == "LIBDIR" else "libpython.so")
    return module, vendor, stdlib, executable


@pytest.mark.parametrize("part", ["dependency", "stdlib", "interpreter"])
def test_native_runtime_content_seal_follows_installed_directory_alias(runtime, part):
    module, vendor, stdlib, executable = runtime
    before = module.identity()
    assert before["schema"] == "tessera.suite_runtime.v1"
    target = {"dependency": vendor / "dependency.py", "stdlib": stdlib / "json.py", "interpreter": executable}[part]
    target.write_bytes(b"different bytes")
    assert module.identity()["sha256"] != before["sha256"]


def test_runtime_alias_cycle_terminates_and_alias_target_is_sealed(runtime, tmp_path):
    module, vendor, _, _ = runtime
    (vendor / "cycle").symlink_to(vendor, target_is_directory=True)
    before = module.identity()
    (vendor / "cycle").unlink()
    assert module.identity()["sha256"] != before["sha256"]


def test_wrapper_runtime_does_not_claim_a_pytest_population(runtime, tmp_path):
    module, vendor, stdlib, executable = runtime
    site = tmp_path / "site"
    for path in site.glob("*.dist-info/METADATA"):
        path.unlink()
    with pytest.raises(ValueError, match="pytest"):
        module.identity()
    measured = module.identity(require_suite=False)
    assert measured["require_suite"] is False
    assert measured["executable"] == str(executable)
    before = measured["sha256"]
    executable.write_bytes(b"different actual wrapper bytes")
    assert module.identity(require_suite=False)["sha256"] != before
