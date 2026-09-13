"""Runtime-scoped tests; run inside the pinned vLLM image."""
from types import SimpleNamespace

import pytest
pytest.importorskip("vllm")

from tessera.serving.glm53_nope import _config_reason, TesseraGLM53NoPEBackend


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
            model_type="glm5_next_text", kv_lora_rank=512, qk_nope_head_dim=256,
            qk_rope_head_dim=0, index_topk=2048, index_kpool=4)),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1,
                                        prefill_context_parallel_size=1),
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=False),
    )


@pytest.mark.parametrize("field,value", [("model_type", "deepseek_v3"),
    ("qk_rope_head_dim", 64), ("kv_lora_rank", 256), ("qk_nope_head_dim", 128),
    ("index_topk", 1024), ("index_kpool", 1)])
def test_wrong_geometry_refused(field, value):
    candidate = config()
    setattr(candidate.model_config.hf_text_config, field, value)
    assert field in _config_reason(candidate)


def test_configuration_and_device_guards():
    candidate = config()
    assert _config_reason(candidate) is None
    candidate.parallel_config.decode_context_parallel_size = 2
    assert "decode_context_parallel_size" in _config_reason(candidate)
    candidate.parallel_config.decode_context_parallel_size = 1
    candidate.kernel_config.enable_flashinfer_autotune = True
    assert "enable_flashinfer_autotune" in _config_reason(candidate)
    assert TesseraGLM53NoPEBackend.supports_compute_capability(SimpleNamespace(major=12, minor=1))
    assert not TesseraGLM53NoPEBackend.supports_compute_capability(SimpleNamespace(major=12, minor=0))


def test_registration_opt_in_preserves_stock_and_refuses_custom_collision(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as Backend
    from vllm.v1.attention.backends.registry import register_backend
    from tessera.serving import register

    previous = Backend.CUSTOM.get_path() if Backend.CUSTOM.is_overridden() else None
    stock = Backend.FLASHINFER_MLA_SPARSE_SM120.get_path()
    try:
        Backend.CUSTOM.clear_override()
        monkeypatch.delenv("TESSERA_RESEARCH_GLM53_NOPE", raising=False)
        register()
        assert not Backend.CUSTOM.is_overridden()
        monkeypatch.setenv("TESSERA_RESEARCH_GLM53_NOPE", "1")
        register()
        register()
        assert Backend.CUSTOM.get_class() is TesseraGLM53NoPEBackend
        assert Backend.FLASHINFER_MLA_SPARSE_SM120.get_path() == stock
        register_backend(Backend.CUSTOM, "another.PluginBackend")
        with pytest.raises(RuntimeError, match="another CUSTOM backend"):
            register()
    finally:
        Backend.CUSTOM.clear_override()
        if previous:
            register_backend(Backend.CUSTOM, previous)
