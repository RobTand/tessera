"""tessera#1031: the intake guard must see the real activation submodule.

A stale ``vllm.model_executor.layers`` stub makes ``import vllm`` succeed
while the fused MoE activation submodule stays unimportable. The intake
file must skip then. It must not collect tests that fail at call time.
"""
from __future__ import annotations

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")

import test_serving_dispatch as dispatch
import test_serving_moe_dispatch as moe_dispatch


def _save_vllm():
    return {name: mod for name, mod in sys.modules.items()
            if name == "vllm" or name.startswith("vllm.")}


def _drop_vllm():
    for name in [name for name in sys.modules
                 if name == "vllm" or name.startswith("vllm.")]:
        del sys.modules[name]


_INTAKE = "test_serving_moe_bf16_tp1_intake"
_MISSING = object()


def _pollute():
    saved = _save_vllm()
    consumer = sys.modules.pop(_INTAKE, _MISSING)
    _drop_vllm()
    dispatch._install_vllm_stubs()
    return saved, consumer


def _clean(saved):
    modules, consumer = saved
    _drop_vllm()
    sys.modules.update(modules)
    sys.modules.pop(_INTAKE, None)
    if consumer is not _MISSING:
        sys.modules[_INTAKE] = consumer


def test_stale_layers_stub_makes_the_intake_file_skip():
    before = _save_vllm()
    consumer = sys.modules.get(_INTAKE, _MISSING)
    saved = _pollute()
    try:
        assert "vllm.model_executor.layers" in sys.modules
        with pytest.raises(pytest.skip.Exception, match="fused_moe.activation"):
            importlib.import_module(_INTAKE)
    finally:
        _clean(saved)
    assert _save_vllm() == before
    assert sys.modules.get(_INTAKE, _MISSING) is consumer


def _finish(running):
    try:
        next(running)
    except StopIteration:
        return
    raise AssertionError("the fixture did not finish")


def _stub_leftovers():
    return sorted(name for name, mod in sys.modules.items()
                  if (name == "vllm" or name.startswith("vllm."))
                  and getattr(mod, "__file__", None) is None)


@pytest.mark.parametrize("module", [dispatch, moe_dispatch])
def test_stub_installers_leave_no_vllm_stub_behind(module):
    before = _stub_leftovers()
    running = module.runtime_modules.__wrapped__
    started = running()
    next(started)
    try:
        assert "vllm.model_executor.layers" in sys.modules
    finally:
        _finish(started)
    assert _stub_leftovers() == before
