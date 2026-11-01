"""CPU build-owner controls for the default-off folded BF16 arm (#874)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch.utils.cpp_extension
from tessera import routed_fused as rf
from tessera.errors import GrammarError


@pytest.mark.parametrize("distance", ["0", "4"])
@pytest.mark.parametrize("fp4_distance", ["0", "4"])
def test_value_prefetch_build_identity(monkeypatch, distance, fp4_distance):
    rf._ext.cache_clear()
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH", distance)
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_FP4_A_PREFETCH", fp4_distance)
    calls = []
    def load(**kw):
        calls.append(kw)
        return SimpleNamespace()
    monkeypatch.setattr("torch.utils.cpp_extension.load", load)
    def build(module, source_module, compile_fn):
        assert source_module == rf.MODULE_NAME_VALUE
        assert module == (rf.MODULE_NAME_VALUE if distance == "0" else "tessera_routed_fused_value_prefetch4")
        compile_fn("source.cu", "build", "sm_121", False)
        # Stop before runtime ABI checks; this test checks the actual loader arguments.
        raise RuntimeError("captured build")
    monkeypatch.setattr(rf, "build_library", build)
    with pytest.raises(RuntimeError, match="captured build"):
        rf._ext("value")
    assert calls[0]["name"] == (rf.MODULE_NAME_VALUE if distance == "0" else "tessera_routed_fused_value_prefetch4")
    assert ("-DTESSERA_ROUTED_FUSED_VALUE_A_PREFETCH=4" in calls[0]["extra_cuda_cflags"]) == (distance == "4")
    rf._ext.cache_clear()


@pytest.mark.parametrize("distance", ["1", "2", "-1", "bad"])
def test_value_prefetch_rejects_unqualified_distance(monkeypatch, distance):
    rf._ext.cache_clear()
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH", distance)
    with pytest.raises(GrammarError, match="must be 0 or 4"):
        rf._ext("value")


