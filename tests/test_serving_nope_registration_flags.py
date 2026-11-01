"""CPU tests of the real plugin entry point; unrelated vLLM installs are stubbed."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tessera.serving import flags, register


FLAG = "TESSERA_RESEARCH_GLM53_NOPE"
MASK_FLAG = "TESSERA_RESEARCH_MLA_MASK_SKIP"
BACKEND_PATH = "tessera.serving.glm53_nope.TesseraGLM53NoPEBackend"


@pytest.fixture
def registry(monkeypatch):
    """Keep flag parsing real without requiring vLLM, torch, or CUDA installs."""
    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__path__ = []
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    backend = SimpleNamespace(path=None)
    backend.is_overridden = lambda: backend.path is not None
    backend.get_path = lambda: backend.path
    calls = []

    def register_backend(selected, path):
        assert selected is backend
        calls.append(path)
        backend.path = path

    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends"):
        module(name)
    module("vllm.model_executor.layers.quantization",
           register_quantization_config=lambda name: lambda config: config)
    module("vllm.v1.attention.backends.registry",
           AttentionBackendEnum=SimpleNamespace(CUSTOM=backend),
           register_backend=register_backend)
    module("tessera.serving.config", TesseraConfig=object)
    module("tessera.serving.endpoint_runtime", install=lambda: None)
    module("tessera.serving.projection_routes", install=lambda: None)
    module("tessera.serving.glm53_graphs", install_operator_pin=lambda: None)

    for flag in (FLAG, MASK_FLAG):
        flags.reset_for_tests(flag)
        monkeypatch.delenv(flag, raising=False)
    try:
        yield backend, calls
    finally:
        for flag in (FLAG, MASK_FLAG):
            flags.reset_for_tests(flag)


def test_true_is_rejected_with_accepted_spellings(registry, monkeypatch):
    backend, calls = registry
    monkeypatch.setenv(FLAG, "true")
    with pytest.raises(ValueError, match=FLAG) as error:
        register()
    message = str(error.value)
    assert "'1'" in message and "'0'" in message and "unset" in message
    assert backend.path is None
    assert calls == []


@pytest.mark.parametrize("initial", [None, "0", "1", " 1 "],
                         ids=["unset", "disabled", "enabled", "whitespace"])
def test_registration_latches_the_first_normalized_value(registry, monkeypatch, initial):
    backend, calls = registry
    if initial is not None:
        monkeypatch.setenv(FLAG, initial)
    enabled = initial is not None and initial.strip() == "1"
    register()
    assert backend.path == (BACKEND_PATH if enabled else None)

    # Repeated registration of the same normalized value remains idempotent.
    if initial == " 1 ":
        monkeypatch.setenv(FLAG, "1")
    register()
    assert backend.path == (BACKEND_PATH if enabled else None)
    assert bool(calls) is enabled

    before = list(calls)
    monkeypatch.setenv(FLAG, "0" if enabled else "1")
    with pytest.raises(RuntimeError, match=FLAG) as error:
        register()
    assert "restart" in str(error.value)
    assert calls == before, "a changed flag must be refused before backend registration"
    assert backend.path == (BACKEND_PATH if enabled else None)
