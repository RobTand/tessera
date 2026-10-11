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
