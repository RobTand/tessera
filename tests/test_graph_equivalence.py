"""The serve-time graph verdict on the vLLM nightly path (tessera#702), CPU only.

``graph_equivalence`` answers, for a running vLLM configuration, whether a
graph serve is one a published graph receipt measured equal to eager.  These
drive it with configuration doubles and a digest stub, so they need no vLLM:
the enums below stand in for ``CompilationMode`` / ``CUDAGraphMode`` (the
module reads members by name and builds a resolved mode from the input's own
enum class), and ``digests`` stands in for hashing the installed runner.
"""
from __future__ import annotations

import copy
import enum
from types import SimpleNamespace

import pytest

from tessera.serving import graph_equivalence as ge


class Mode(enum.IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class Graph(enum.IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


RUNNER = {"v1/worker/gpu/model_runner.py": "a" * 64}
RECEIPT = {
    "id": "glm5next_test", "image": "x@sha256:" + "0" * 64, "vllm": "v", "torch": "t",
    "model_type": "glm5_next_text", "runner_sha256": RUNNER,
    "serve": {"compilation_config": {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"},
              "kernel_config": {"enable_flashinfer_autotune": False},
              "env": {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
              "capture_sizes": "every_count_to_max_decode_tokens"},
    "equivalence": {"verdict": "eager_equivalent", "criterion": "tessera#508 membership",
                    "receipt": "docs/measurements/x.md", "data": "experiments/results/x.json"},
    "speculative": [{"method": "mtp", "num_speculative_tokens": 1,
                     "verdict": "eager_equivalent", "criterion": "tessera#508 membership",
                     "receipt": "docs/measurements/x.md", "data": "experiments/results/y.json"}],
}
ENV = {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}


def config(*, mode=Mode.NONE, graph=Graph.FULL_DECODE_ONLY, sizes=range(1, 9), max_num_seqs=8,
           custom_ops=("all",), model_type="glm5_next_text", speculative=None,
           autotune=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        compilation_config=SimpleNamespace(
            mode=mode, cudagraph_mode=graph, cudagraph_capture_sizes=list(sizes),
            custom_ops=list(custom_ops), splitting_ops_contain_attention=lambda: False),
        kernel_config=SimpleNamespace(
            enable_flashinfer_autotune=autotune,
            ir_op_priority=SimpleNamespace(rms_norm=["vllm_c", "native"],
                                           fused_add_rms_norm=["vllm_c", "native"])),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=speculative)


def mtp(k=1):
    return SimpleNamespace(method="mtp", num_speculative_tokens=k)


def same(paths):
    return dict(RUNNER) if dict(paths) == RUNNER else {p: None for p in paths}


def other(paths):
    return {p: "b" * 64 for p in paths}


def verdict(cfg, *, digests=same, environ=ENV, receipts=(RECEIPT,)):
    return ge.graph_verdict(cfg, list(receipts), digests=digests, environ=environ)


def test_the_full_request_resolves_as_the_sparse_mla_builder_does():
    cfg = config(graph=Graph.FULL)
    assert ge.graph_mode(cfg.compilation_config) is Graph.FULL_DECODE_ONLY
    cfg.compilation_config.splitting_ops_contain_attention = lambda: True
    assert ge.graph_mode(cfg.compilation_config) is Graph.FULL_AND_PIECEWISE
    cfg.compilation_config.cudagraph_mode = None
    assert ge.graph_mode(cfg.compilation_config) is None


def test_default_sizes_pad_decode_batches_and_contiguous_sizes_do_not():
    # vLLM's default list at max_num_seqs 8 without a drafter.
    padded = config(sizes=[1, 2, 4, 8, 16])
    assert ge.padded_token_counts(padded, Graph.FULL_DECODE_ONLY) == [3, 5, 6, 7]
    assert ge.padded_token_counts(config(), Graph.FULL_DECODE_ONLY) == []
    # One draft token: verification runs 2 tokens per request.
    assert ge.padded_token_counts(config(sizes=[1, 2, 4, 8, 16], speculative=mtp()),
                                  Graph.FULL_DECODE_ONLY) == [6, 10, 12, 14]


def test_an_eager_serve_claims_nothing_and_needs_no_receipt():
    assert verdict(config(graph=Graph.NONE)) == (None, None)
    assert verdict(config(graph=None), receipts=()) == (None, None)


def test_the_measured_graph_serve_is_claimed_equal_to_eager():
    assert verdict(config()) == ("glm5next_test", None)
    assert verdict(config(sizes=range(1, 17))) == ("glm5next_test", None)


@pytest.mark.parametrize("change, named", [
    (dict(mode=Mode.VLLM_COMPILE), "compilation_config.mode is 'VLLM_COMPILE'"),
    (dict(graph=Graph.FULL_AND_PIECEWISE), "compilation_config.cudagraph_mode"),
    (dict(sizes=[1, 2, 4, 8, 16]), "leave token counts [3, 5, 6, 7]"),
    (dict(autotune=True), "kernel_config.enable_flashinfer_autotune"),
    (dict(speculative=mtp(2), sizes=range(1, 25)), "drafter 'mtp' at 2 draft tokens"),
])
def test_any_departure_from_the_receipt_is_named(change, named):
    rid, gap = verdict(config(**change))
    assert rid == "glm5next_test"
    assert gap is not None and named in gap, gap


def test_the_breakable_graph_environment_is_part_of_the_receipt():
    rid, gap = verdict(config(), environ={})
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH is None, not '0'" in gap


def test_a_measured_drafter_needs_every_verification_count_captured():
    assert verdict(config(speculative=mtp(1), sizes=range(1, 17))) == ("glm5next_test", None)
    _, gap = verdict(config(speculative=mtp(1), sizes=range(1, 9)))
    assert "leave token counts [9, 10, 11, 12, 13, 14, 15, 16] of 1..16" in gap


@pytest.mark.parametrize("change", [dict(digests=other), dict(receipts=())])
def test_a_runtime_no_receipt_measured_is_never_claimed_equal(change):
    rid, gap = verdict(config(), **change)
    assert rid is None and "no graph receipt measures this runtime's graph path" in gap


def test_another_model_type_is_not_covered_by_the_receipt():
    rid, gap = verdict(config(model_type="qwen3"))
    assert rid is None and "'qwen3'" in gap


def test_a_receipt_names_only_what_it_measured():
    """Fields the receipt leaves out are not compared: its compilation config is
    what the consumer passes, not the whole resolved object."""
    receipt = copy.deepcopy(RECEIPT)
    receipt["serve"]["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert verdict(config(mode=Mode.VLLM_COMPILE), receipts=[receipt]) == (
        "glm5next_test", None)


def test_report_records_the_verdict_once_and_skips_eager(monkeypatch, capsys):
    seen = []
    from tessera.serving import telemetry

    monkeypatch.setattr(telemetry, "record_backend_execution_identity",
                        lambda **kw: seen.append(kw))
    monkeypatch.setattr(ge, "_REPORTED", set())
    monkeypatch.setattr(ge, "runner_digests", same)
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    ge.report_once(config(graph=Graph.NONE), [RECEIPT])
    assert seen == [] and capsys.readouterr().err == ""
    ge.report_once(config(), [RECEIPT])
    ge.report_once(config(), [RECEIPT])
    assert seen == [dict(backend="vllm", compilation_mode="NONE",
                         cuda_graph_mode="FULL_DECODE_ONLY", eager_equivalence_gap=None)] * 2
    assert capsys.readouterr().err.count("reproduces graph receipt 'glm5next_test'") == 1
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH")
    ge.report_once(config(), [RECEIPT])
    assert seen[-1]["eager_equivalence_gap"] is not None
    assert "WARNING: this serve's outputs are not claimed equal" in capsys.readouterr().err
