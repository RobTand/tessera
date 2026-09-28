"""Runtime-scoped tests; run inside the pinned vLLM image."""
from types import SimpleNamespace

import pytest
pytest.importorskip("vllm")

from tessera.serving import glm53_nope
from tessera.serving.glm53_nope import _config_reason, TesseraGLM53NoPEBackend
from vllm.config.compilation import CompilationMode, CUDAGraphMode


#: vLLM's default capture list at max_num_seqs=8 without a drafter
#: (``VllmConfig._set_cudagraph_sizes``: 1, 2, 4, then multiples of 8 up to
#: twice max_num_seqs).
DEFAULT_SIZES = [1, 2, 4, 8, 16]
CONTIGUOUS = list(range(1, 9))
EAGER_IR = ["vllm_c", "native"]


def config(*, mode=CompilationMode.NONE, graph=CUDAGraphMode.NONE, attention_splits=False,
           enforce_eager=True, sizes=CONTIGUOUS, max_num_seqs=8, custom_ops=("all",),
           ir=EAGER_IR):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager, hf_text_config=SimpleNamespace(
            model_type="glm5_next_text", kv_lora_rank=512, qk_nope_head_dim=256,
            qk_rope_head_dim=0, index_topk=2048, index_kpool=4)),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1,
                                        prefill_context_parallel_size=1),
        kernel_config=SimpleNamespace(
            enable_flashinfer_autotune=False,
            ir_op_priority=SimpleNamespace(rms_norm=list(ir), fused_add_rms_norm=list(ir))),
        compilation_config=SimpleNamespace(
            mode=mode, cudagraph_mode=graph, cudagraph_capture_sizes=list(sizes),
            custom_ops=list(custom_ops),
            splitting_ops_contain_attention=lambda: attention_splits),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        use_v2_model_runner=True,
        speculative_config=None,
    )


@pytest.fixture
def runner(monkeypatch):
    """Pin the runner digest the gate reads, independent of the installed image."""
    def use(digest):
        monkeypatch.setattr(glm53_nope, "_runner_sha256", lambda: digest)
    use(glm53_nope._GRAPH_RUNNER_SHA256)
    return use


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


# Every (compilation mode, CUDA-graph mode) pair, and FULL both ways: vLLM
# resolves FULL to FULL_DECODE_ONLY for this backend unless attention is a
# splitting op, when it becomes FULL_AND_PIECEWISE (resolve_cudagraph_mode_and_sizes).
_GRAPH_CASES = [(graph, splits) for graph in CUDAGraphMode for splits in (False, True)]


def _resolved(graph, splits):
    if graph == CUDAGraphMode.FULL:
        return CUDAGraphMode.FULL_AND_PIECEWISE if splits else CUDAGraphMode.FULL_DECODE_ONLY
    return graph


def _admitted(mode, graph, splits):
    """The receipts of tessera#508: which pairs ran the equality suite correctly."""
    resolved = _resolved(graph, splits)
    if mode == CompilationMode.NONE:
        return True
    if mode == CompilationMode.VLLM_COMPILE:
        return resolved in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
    if mode == CompilationMode.DYNAMO_TRACE_ONCE:
        return resolved == CUDAGraphMode.NONE
    return False


@pytest.mark.parametrize("mode", list(CompilationMode), ids=lambda m: m.name)
@pytest.mark.parametrize("graph,splits", _GRAPH_CASES,
                         ids=[f"{g.name}-{'attn_split' if s else 'no_split'}" for g, s in _GRAPH_CASES])
def test_every_execution_mode_is_admitted_or_refused_by_name(runner, mode, graph, splits):
    reason = _config_reason(config(mode=mode, graph=graph, attention_splits=splits,
                                   enforce_eager=False))
    resolved = _resolved(graph, splits)
    if _admitted(mode, graph, splits):
        assert reason is None
    elif mode == CompilationMode.STOCK_TORCH_COMPILE:
        assert "refuses compilation mode STOCK_TORCH_COMPILE" in reason and "fails to start" in reason
    elif mode == CompilationMode.DYNAMO_TRACE_ONCE:
        assert "under compilation mode DYNAMO_TRACE_ONCE" in reason
        assert "measured without graphs only" in reason
    elif mode == CompilationMode.VLLM_COMPILE:
        assert f"refuses CUDA-graph mode {resolved.name} under compilation mode VLLM_COMPILE" in reason
        assert "breakable CUDA graph" in reason
    else:
        assert f"refuses compilation mode {mode.name}" in reason and "no receipt" in reason


def test_eager_and_cudagraph_none_are_admitted_on_any_runner(runner):
    runner("0" * 64)
    assert _config_reason(config()) is None
    assert _config_reason(config(enforce_eager=False)) is None


_GRAPHS = (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE,
           CUDAGraphMode.FULL_AND_PIECEWISE)


def test_graphs_on_the_stock_runner_are_refused_with_the_fault(runner):
    runner(glm53_nope._STOCK_RUNNER_SHA256)
    for graph in _GRAPHS:
        reason = _config_reason(config(graph=graph, enforce_eager=False))
        assert "stock V2 runner" in reason and "vLLM #57317" in reason
        assert "6 of 10" in reason and glm53_nope._GRAPH_RUNNER_SHA256 in reason


def test_graphs_on_an_unmeasured_runner_name_its_digest(runner):
    runner("f" * 64)
    for graph in _GRAPHS:
        reason = _config_reason(config(graph=graph, enforce_eager=False))
        assert "f" * 64 in reason and glm53_nope._GRAPH_RUNNER_SHA256 in reason


def test_graphs_need_the_v2_runner_and_no_drafter(runner):
    for graph in _GRAPHS:
        candidate = config(graph=graph, enforce_eager=False)
        candidate.use_v2_model_runner = False
        assert "V2 model runner" in _config_reason(candidate)
        candidate = config(graph=graph, enforce_eager=False)
        candidate.speculative_config = SimpleNamespace(method="mtp")
        assert "speculative decoding" in _config_reason(candidate)


@pytest.mark.parametrize("graph,sizes,max_num_seqs,padded", [
    # vLLM's default list pads decode batches of 3, 5, 6 and 7 requests.
    (CUDAGraphMode.FULL_DECODE_ONLY, DEFAULT_SIZES, 8, [3, 5, 6, 7]),
    (CUDAGraphMode.FULL_DECODE_ONLY, CONTIGUOUS, 8, []),
    # Decode graphs stop at max_num_seqs: a larger capture size pads nothing.
    (CUDAGraphMode.FULL_DECODE_ONLY, CONTIGUOUS + [16], 8, []),
    # Decode batches of 5 and 6 have no graph at max_num_seqs 6 and run eager.
    (CUDAGraphMode.FULL_DECODE_ONLY, [1, 2, 4, 8], 6, [3]),
    # The release shape: four requests; the default pads only a batch of 3.
    (CUDAGraphMode.FULL_DECODE_ONLY, [1, 2, 4, 8], 4, [3]),
    (CUDAGraphMode.FULL_DECODE_ONLY, [1, 2, 3, 4], 4, []),
    # Piecewise graphs take mixed batches of every size up to the largest.
    (CUDAGraphMode.PIECEWISE, DEFAULT_SIZES, 8, [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]),
    (CUDAGraphMode.PIECEWISE, CONTIGUOUS, 8, []),
    (CUDAGraphMode.FULL_AND_PIECEWISE, DEFAULT_SIZES, 8, [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]),
    (CUDAGraphMode.FULL_AND_PIECEWISE, CONTIGUOUS, 8, []),
    (CUDAGraphMode.NONE, DEFAULT_SIZES, 8, []),
])
def test_padded_token_counts_follow_the_runner_dispatch(graph, sizes, max_num_seqs, padded):
    candidate = config(graph=graph, sizes=sizes, max_num_seqs=max_num_seqs, enforce_eager=False)
    assert glm53_nope._padded_token_counts(candidate, graph) == padded


def test_eager_equivalence_is_claimed_only_without_padding_or_an_op_switch(runner):
    gap = glm53_nope.eager_equivalence_gap
    assert gap(config()) is None
    for graph in _GRAPHS:
        contiguous = config(graph=graph, enforce_eager=False)
        assert _config_reason(contiguous) is None and gap(contiguous) is None
        padded = config(graph=graph, sizes=DEFAULT_SIZES, enforce_eager=False)
        # Admitted: a padded graph runs correctly; it is a different computation.
        assert _config_reason(padded) is None
        assert "[3, 5, 6, 7" in gap(padded) and "0.98081" in gap(padded)
    # VLLM_COMPILE compiles nothing here; it only switches op implementations.
    restored = config(mode=CompilationMode.VLLM_COMPILE)
    assert gap(restored) is None
    # The verdict names each switch that differs from eager, and only those.
    custom_gap, ir_gap = "custom_ops resolves to", "IR op rms_norm resolves to"
    by_default = config(mode=CompilationMode.VLLM_COMPILE, custom_ops=("none",), ir=["native"])
    assert custom_gap in gap(by_default) and ir_gap in gap(by_default)
    assert "compiles nothing" in gap(by_default)
    ir_only = gap(config(mode=CompilationMode.VLLM_COMPILE, ir=["native"]))
    assert ir_gap in ir_only and custom_gap not in ir_only
    custom_only = gap(config(mode=CompilationMode.VLLM_COMPILE, custom_ops=("none",)))
    assert custom_gap in custom_only and ir_gap not in custom_only
    excluded = config(mode=CompilationMode.VLLM_COMPILE, custom_ops=("all", "-rms_norm"))
    assert "custom_ops resolves to ['all', '-rms_norm']" in gap(excluded)
    # Mode NONE is the reference, whatever its op settings.
    assert gap(config(custom_ops=("none",), ir=["native"])) is None


def test_the_verdict_is_reported_once_per_distinct_verdict(runner, monkeypatch, capsys):
    monkeypatch.setattr(glm53_nope, "_REPORTED", set())
    equal = config(graph=CUDAGraphMode.FULL_DECODE_ONLY, enforce_eager=False)
    padded = config(graph=CUDAGraphMode.FULL_DECODE_ONLY, sizes=DEFAULT_SIZES, enforce_eager=False)
    for candidate in (equal, equal, padded, padded):
        glm53_nope._report_equivalence(candidate)
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2
    assert "runs eager's arithmetic" in err[0]
    assert "not claimed equal to eager's" in err[1] and "0.98081" in err[1]


def test_both_call_sites_apply_the_gate(runner, monkeypatch):
    """supports_combination (backend selection) and Impl.__init__ (construction)."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend

    monkeypatch.setattr(FlashInferMLASparseSM120Backend, "supports_combination",
                        classmethod(lambda cls, *args, **kwargs: None))
    monkeypatch.setattr(glm53_nope, "require_stock_runtime", lambda: None)
    monkeypatch.setattr(glm53_nope, "probed_platform_token", lambda **kwargs: "sm_121")
    monkeypatch.setattr(glm53_nope.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(glm53_nope, "_REPORTED", set())
    args = (512, None, "fp8_ds_mla", 64, True, False, True, False, None)

    def construct():
        # The gate refuses before the stock constructor runs; an admitted
        # configuration reaches it and stops at this sentinel.
        monkeypatch.setattr(glm53_nope.FlashInferMLASparseSM120Impl, "__init__",
                            lambda self, *a, **k: (_ for _ in ()).throw(LookupError("stock init")))
        with pytest.raises((RuntimeError, LookupError)) as info:
            glm53_nope.TesseraGLM53NoPEImpl()
        return info.value

    for mode, graph, splits, admitted in (
            (CompilationMode.NONE, CUDAGraphMode.FULL, False, True),
            (CompilationMode.NONE, CUDAGraphMode.FULL, True, True),
            (CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY, False, True),
            (CompilationMode.NONE, CUDAGraphMode.PIECEWISE, False, True),
            (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, False, True),
            (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL, True, False),
            (CompilationMode.VLLM_COMPILE, CUDAGraphMode.PIECEWISE, False, False)):
        candidate = config(mode=mode, graph=graph, attention_splits=splits, enforce_eager=False)
        monkeypatch.setattr(glm53_nope, "get_current_vllm_config", lambda: candidate)
        reason = TesseraGLM53NoPEBackend.supports_combination(*args)
        error = construct()
        if admitted:
            assert reason is None
            assert isinstance(error, LookupError)
        else:
            assert "under compilation mode VLLM_COMPILE" in reason
            assert isinstance(error, RuntimeError) and str(error) == reason
    runner(glm53_nope._STOCK_RUNNER_SHA256)
    candidate = config(graph=CUDAGraphMode.FULL, enforce_eager=False)
    monkeypatch.setattr(glm53_nope, "get_current_vllm_config", lambda: candidate)
    assert "vLLM #57317" in TesseraGLM53NoPEBackend.supports_combination(*args)
    assert "vLLM #57317" in str(construct())
