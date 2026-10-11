"""Benchmark imports must not change the interpreter's vLLM bindings."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("runtime_present", [False, True])
def test_benchmark_import_preserves_the_runtime_bindings(runtime_present):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    if runtime_present:
        pytest.importorskip("vllm.config")
    code = r'''
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

root = Path.cwd()
if sys.argv[1] == "True":
    import vllm.config
else:
    class NoVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "vllm" or fullname.startswith("vllm."):
                raise ModuleNotFoundError("vLLM is absent for this import probe", name=fullname)
    sys.meta_path.insert(0, NoVllm())

def bindings():
    return {name: module for name, module in sys.modules.items()
            if name == "vllm" or name.startswith("vllm.")}

before = bindings()
mode = os.environ.get("TESSERA_SERVE_MODE")
spec = importlib.util.spec_from_file_location(
    "bench_t8r", root / "experiments/t8r_speed/bench_t8r.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert bindings() == before, "benchmark import changed the vLLM bindings"
assert os.environ.get("TESSERA_SERVE_MODE") == mode
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(runtime_present)], cwd=CHECKOUT,
        env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fixture_module", ["test_serving_dispatch", "test_serving_moe_dispatch"])
@pytest.mark.parametrize("failure", ["config-import", "stub-install"])
@pytest.mark.parametrize("runtime_present", [False, True])
def test_dispatch_setup_failure_restores_runtime_bindings(fixture_module, failure, runtime_present):
    pytest.importorskip("torch")
    if runtime_present:
        pytest.importorskip("vllm.config")
    code = r'''
import builtins
import importlib
import importlib.abc
import sys

import pytest

sys.path.insert(0, "tests")
if sys.argv[3] == "True":
    import vllm.config
    import tessera.serving.config
else:
    class NoVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "vllm" or fullname.startswith("vllm."):
                raise ModuleNotFoundError("vLLM is absent for this setup probe", name=fullname)
    sys.meta_path.insert(0, NoVllm())

module = importlib.import_module(sys.argv[1])
import tessera.serving as package

def bindings():
    return {name: value for name, value in sys.modules.items() if module._is_isolated(name)}

before = bindings()
missing = object()
package_attr = getattr(package, "config", missing)
original_import = builtins.__import__
original_install = module._install_vllm_stubs

def refuse_config(name, *args, **kwargs):
    if name == "tessera.serving.config":
        raise ImportError("forced config setup failure")
    return original_import(name, *args, **kwargs)

def refuse_install():
    original_install()
    raise ImportError("forced stub setup failure")

if sys.argv[2] == "config-import":
    builtins.__import__ = refuse_config
else:
    module._install_vllm_stubs = refuse_install
try:
    with pytest.raises(ImportError, match="forced .* setup failure"):
        next(module.runtime_modules.__wrapped__())
finally:
    builtins.__import__ = original_import
    module._install_vllm_stubs = original_install

# A later consumer must still see the real config, or the original absence.
if sys.argv[3] == "True":
    from vllm.config import VllmConfig
    assert VllmConfig is before["vllm.config"].VllmConfig
else:
    with pytest.raises(pytest.skip.Exception):
        pytest.importorskip("vllm.config")
assert bindings() == before, "fixture setup changed the runtime bindings"
assert getattr(package, "config", missing) is package_attr
'''
    result = subprocess.run(
        [sys.executable, "-c", code, fixture_module, failure, str(runtime_present)],
        cwd=CHECKOUT, env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
