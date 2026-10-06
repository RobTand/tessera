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
DEFAULT_MAX_MODEL_LEN = 4096


def config(*, mode=CompilationMode.NONE, graph=CUDAGraphMode.NONE, attention_splits=False,
           enforce_eager=True, sizes=CONTIGUOUS, max_num_seqs=8, custom_ops=("all",),
           ir=EAGER_IR, speculative=None, max_model_len=4096):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager, max_model_len=max_model_len,
                                     hf_text_config=SimpleNamespace(
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
        speculative_config=speculative,
    )


def drafter(method="mtp", k=2, share=True, **extra):
    """A speculative configuration as vLLM resolves it; the draft config carries
    ``index_share_for_mtp_iteration`` from the model's own config."""
    return SimpleNamespace(method=method, num_speculative_tokens=k, **extra,
                           draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
                               index_share_for_mtp_iteration=share)))


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


def test_eager_and_cudagraph_none_preserve_nonstock_runner_admission(runner):
    runner("0" * 64)
    assert _config_reason(config()) is None
    assert _config_reason(config(enforce_eager=False)) is None


_GRAPHS = (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE,
           CUDAGraphMode.FULL_AND_PIECEWISE)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_known_stock_v2_runner_is_refused_without_cuda_graphs(runner, enforce_eager):
    runner(glm53_nope._STOCK_RUNNER_SHA256)
    reason = _config_reason(config(graph=CUDAGraphMode.NONE, enforce_eager=enforce_eager))
    assert reason is not None, "known out-of-bounds stock V2 runner was admitted without graphs"
    assert "stock V2 runner" in reason and "vLLM #57317" in reason


def test_known_stock_v2_digest_does_not_reject_eager_v1_path(runner):
    runner(glm53_nope._STOCK_RUNNER_SHA256)
    candidate = config()
    candidate.use_v2_model_runner = False
    assert _config_reason(candidate) is None


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


def test_graphs_need_the_v2_runner(runner):
    for graph in _GRAPHS:
        for speculative in (None, drafter()):
            candidate = config(graph=graph, enforce_eager=False, speculative=speculative)
            candidate.use_v2_model_runner = False
            assert "V2 model runner" in _config_reason(candidate)


def test_dflash_is_refused_by_name_in_every_mode(runner):
    for mode in (CompilationMode.NONE, CompilationMode.VLLM_COMPILE):
        for graph in (CUDAGraphMode.NONE,) + _GRAPHS:
            reason = _config_reason(config(mode=mode, graph=graph, speculative=drafter("dflash", k=7)))
            assert "refuses speculative method 'dflash' in every mode" in reason
            assert "SupportsEagle3" in reason and "_get_kv_cache_groups_glm5_next" in reason
            assert "tessera#695" in reason


def test_eager_drafters_other_than_dflash_keep_their_admission(runner):
    """The gate claims nothing about an eager drafter but dflash's load failure."""
    runner("0" * 64)
    for method in ("mtp", "ngram", "eagle"):
        for k in (1, 2, 7):
            assert _config_reason(config(speculative=drafter(method, k=k))) is None
            assert _config_reason(config(enforce_eager=False, speculative=drafter(method, k=k))) is None


def test_speculative_graphs_are_refused_without_a_receipt(runner):
    assert glm53_nope._SPECULATIVE_GRAPH_RECEIPTS == {}
    for graph in _GRAPHS:
        resolved = _resolved(graph, False).name
        for k in (1, 2, 3):
            reason = _config_reason(config(graph=graph, enforce_eager=False, speculative=drafter(k=k)))
            assert f"refuses speculative method 'mtp' at {k} draft tokens" in reason
            assert (f"compilation mode NONE, CUDA-graph mode {resolved}, max_model_len "
                    f"{DEFAULT_MAX_MODEL_LEN}: no receipt") in reason
            assert ("sparse indices shared across draft steps" in reason) == (k > 1)
            assert "measured: none; tessera#695" in reason
        reason = _config_reason(config(graph=graph, enforce_eager=False, speculative=drafter(
            k=2, num_speculative_tokens_per_batch_size=[(1, 4, 2)])))
        assert "dynamic speculative decoding" in reason and "tessera#695" in reason
    # The runner comes first: no receipt admits a drafter on the stock runner.
    runner(glm53_nope._STOCK_RUNNER_SHA256)
    reason = _config_reason(config(graph=CUDAGraphMode.FULL_DECODE_ONLY, enforce_eager=False,
                                   speculative=drafter()))
    assert "vLLM #57317" in reason


def test_a_receipt_admits_exactly_its_drafter_graph_path(runner, monkeypatch):
    key = ("mtp", 2, True, CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY,
           DEFAULT_MAX_MODEL_LEN)
    monkeypatch.setattr(glm53_nope, "_SPECULATIVE_GRAPH_RECEIPTS", {key: None})
    fdo = dict(graph=CUDAGraphMode.FULL_DECODE_ONLY, enforce_eager=False)
    assert _config_reason(config(**fdo, speculative=drafter())) is None
    # FULL resolves to FULL_DECODE_ONLY without attention splitting: the same path.
    assert _config_reason(config(graph=CUDAGraphMode.FULL, enforce_eager=False,
                                 speculative=drafter())) is None
    measured = ("measured: speculative method 'mtp' at 2 draft tokens, sparse indices shared "
                "across draft steps, compilation mode NONE, CUDA-graph mode FULL_DECODE_ONLY, "
                f"max_model_len {DEFAULT_MAX_MODEL_LEN}")
    for other in (drafter(k=1), drafter(k=3), drafter(share=False), drafter("eagle")):
        reason = _config_reason(config(**fdo, speculative=other))
        assert "no receipt measures this drafter graph path" in reason and measured in reason
    for graph in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE):
        reason = _config_reason(config(graph=graph, enforce_eager=False, speculative=drafter()))
        assert f"CUDA-graph mode {graph.name}, max_model_len {DEFAULT_MAX_MODEL_LEN}: no receipt" in reason
    compiled = config(mode=CompilationMode.VLLM_COMPILE, **fdo, speculative=drafter())
    assert ("compilation mode VLLM_COMPILE, CUDA-graph mode FULL_DECODE_ONLY, "
            f"max_model_len {DEFAULT_MAX_MODEL_LEN}: no receipt") in _config_reason(compiled)


def test_a_receipt_does_not_speak_for_another_max_model_len(runner, monkeypatch):
    """The captured indexer branch follows max_model_len (tessera#695/#702), so the
    receipt is scoped to it: measured at 2048 it admits 2048 and refuses 4096,
    and its None verdict cannot claim a 4096 serve runs eager's arithmetic."""
    at_2048 = ("mtp", 1, None, CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY, 2048)
    monkeypatch.setattr(glm53_nope, "_SPECULATIVE_GRAPH_RECEIPTS", {at_2048: None})
    fdo = dict(graph=CUDAGraphMode.FULL_DECODE_ONLY, enforce_eager=False)
    assert _config_reason(config(**fdo, max_model_len=2048, speculative=drafter(k=1))) is None
    reason = _config_reason(config(**fdo, speculative=drafter(k=1)))
    assert ("no receipt measures this drafter graph path" in reason
            and "max_model_len 2048" in reason and "max_model_len 4096" in reason)
    gap = glm53_nope.eager_equivalence_gap(config(**fdo, speculative=drafter(k=1)))
    assert "no receipt compares this drafter graph path with eager" in gap


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


#: vLLM's default capture lists at max_num_seqs 4 with k draft tokens
#: (``_set_cudagraph_sizes``: 1, 2, 4 and multiples of 8 up to 2 * 4 * (1 + k),
#: plus (1 + k) tokens times 1, 2 and 4 requests).
SPEC_DEFAULT_SIZES = {1: [1, 2, 4, 8, 16], 2: [1, 2, 3, 4, 6, 8, 12, 16, 24],
                      3: [1, 2, 4, 8, 16, 24, 32]}
_MIXED_K2 = [5, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]


@pytest.mark.parametrize("graph,k,sizes,max_num_seqs,padded", [
    # Two draft tokens: verification of 1 to 4 requests is 3, 6, 9 and 12 tokens,
    # 9 from capture size 8 rounded up to whole requests; the drafter's later
    # steps decode 1 to 4 tokens. Nothing pads.
    (CUDAGraphMode.FULL_DECODE_ONLY, 2, SPEC_DEFAULT_SIZES[2], 4, []),
    # One draft token: verifying 3 requests (6 tokens) replays in the 8-token graph.
    (CUDAGraphMode.FULL_DECODE_ONLY, 1, SPEC_DEFAULT_SIZES[1], 4, [6]),
    # Three: 3 requests verify in the 16-token graph and draft-decode in the 4-token one.
    (CUDAGraphMode.FULL_DECODE_ONLY, 3, SPEC_DEFAULT_SIZES[3], 4, [3, 12]),
    # A verification wider than every uniform graph runs eager, unpadded.
    (CUDAGraphMode.FULL_DECODE_ONLY, 2, [1, 2, 3], 4, []),
    # One request pads nothing (the default list at max_num_seqs 1, 3 draft tokens).
    (CUDAGraphMode.FULL_DECODE_ONLY, 3, [1, 2, 4, 8], 1, []),
    # Mixed batches pad at every gap in the list, with or without a drafter.
    (CUDAGraphMode.FULL_AND_PIECEWISE, 2, SPEC_DEFAULT_SIZES[2], 4, _MIXED_K2),
    (CUDAGraphMode.PIECEWISE, 2, SPEC_DEFAULT_SIZES[2], 4, _MIXED_K2),
    (CUDAGraphMode.FULL_AND_PIECEWISE, 1, SPEC_DEFAULT_SIZES[1], 4,
     [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]),
    # Every size up to max_num_seqs * (1 + k) pads nothing in any family.
    (CUDAGraphMode.FULL_DECODE_ONLY, 2, list(range(1, 13)), 4, []),
    (CUDAGraphMode.FULL_AND_PIECEWISE, 2, list(range(1, 13)), 4, []),
    (CUDAGraphMode.PIECEWISE, 2, list(range(1, 13)), 4, []),
])
def test_padded_token_counts_with_a_drafter(graph, k, sizes, max_num_seqs, padded):
    candidate = config(graph=graph, sizes=sizes, max_num_seqs=max_num_seqs, enforce_eager=False,
                       speculative=drafter(k=k))
    assert glm53_nope._padded_token_counts(candidate, graph) == padded


def test_a_drafter_is_claimed_equal_only_to_an_eager_serve_of_the_same(runner, monkeypatch):
    gap = glm53_nope.eager_equivalence_gap
    # Eager with a drafter is that drafter's reference.
    assert gap(config(speculative=drafter())) is None
    fdo = dict(graph=CUDAGraphMode.FULL_DECODE_ONLY, enforce_eager=False, max_num_seqs=4)
    contiguous = config(**fdo, sizes=range(1, 13), speculative=drafter())
    assert "no receipt compares this drafter graph path with eager" in gap(contiguous)
    key = ("mtp", 2, True, CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY,
           DEFAULT_MAX_MODEL_LEN)
    monkeypatch.setattr(glm53_nope, "_SPECULATIVE_GRAPH_RECEIPTS", {key: None})
    assert gap(contiguous) is None
    assert gap(config(**fdo, sizes=SPEC_DEFAULT_SIZES[2], speculative=drafter())) is None
    three = ("mtp", 3, True, CompilationMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY,
             DEFAULT_MAX_MODEL_LEN)
    monkeypatch.setattr(glm53_nope, "_SPECULATIVE_GRAPH_RECEIPTS", {key: None, three: None})
    padded = gap(config(**fdo, sizes=SPEC_DEFAULT_SIZES[3], speculative=drafter(k=3)))
    assert ("[3, 12] (target verification and draft prefill: [12]; draft decode: [3]) in larger"
            in padded)
    # A receipt that found a difference with every size captured is reported as measured.
    monkeypatch.setattr(glm53_nope, "_SPECULATIVE_GRAPH_RECEIPTS", {key: "the measured difference"})
    assert gap(contiguous) == "the measured difference"


def test_the_dflash_load_blockers_hold_in_this_image():
    """The dflash refusal's two blockers, read off the pinned runtime itself."""
    import torch
    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.models.glm5next import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_glm5_next
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec, SlidingWindowSpec

    # The V2 runner's set_eagle3_aux_hidden_state_layers raises on this predicate.
    assert not supports_eagle3(Glm5NextForCausalLM)
    assert not supports_eagle3(Glm5NextForConditionalGeneration)

    class Unread:
        def __getattr__(self, name):
            raise LookupError(f"read vllm_config.{name}")

    body = {
        "model.layers.0.self_attn.kda": MambaSpec(block_size=64, shapes=((1,),),
                                                  dtypes=(torch.float32,)),
        "model.layers.1.self_attn.attn": MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=656, dtype=torch.uint8),
        "model.layers.1.self_attn.indexer": MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=132, dtype=torch.uint8, tokens_per_state=4),
    }
    sliding = {"model.layers.45.self_attn.attn": SlidingWindowSpec(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16, sliding_window=2048)}
    # A sliding-window layer ends the GLM5-next grouping before it reads the config ...
    assert _get_kv_cache_groups_glm5_next(Unread(), {**body, **sliding}) is None
    # ... where the same layers without it pass that check (and fail later, on these toy pages).
    try:
        outcome = _get_kv_cache_groups_glm5_next(Unread(), body)
    except Exception as error:
        outcome = error
    assert outcome is not None


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
