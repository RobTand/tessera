"""CPU checks for ``tessera.graph_receipt``: the graph-equals-eager rule and ``verify``."""
from __future__ import annotations

import copy

import pytest

from tessera import graph_receipt as gr

IMAGE = "localhost/prismaquant/spark-vllm-nccl230@sha256:" + "5be13705" + "0" * 56
RELEASE_CC = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}


def _arm(name="rGR", members=48, replays=1, class_replays=1, cc=None):
    return {
        "name": name, "compilation_config": cc or dict(RELEASE_CC), "speculative_tokens": 0,
        "max_model_len": 8448, "max_num_seqs": 8, "tensor_parallel_size": 1,
        "graph": {"managers": {"ModelCudaGraphManager": {"captured_sizes": [1, 2, 4, 8],
                                                         "replayed_sizes": {"1": replays, "2": 3,
                                                                            "4": 2, "8": 1}}},
                  "classes": {"captured": {"ModelCudaGraphManager": {"2048": 4, "8448": 4}},
                              "replays": {"ModelCudaGraphManager|2048": 40,
                                          "ModelCudaGraphManager|8448": class_replays}}},
        "passes": [{"name": name, "members": members, "choices": 48},
                   {"name": f"{name}-r2", "members": 48, "choices": 48}],
    }


def _receipt(*arms):
    return gr.finish({"schema": gr.SCHEMA, "runtime": {"image": IMAGE},
                      "model": {"config_sha256": "m" * 64}, "tessera": {"src_sha256": "s" * 64},
                      "arms": list(arms) or [_arm()]})


def _serve(**over):
    serve = {"image": IMAGE, "model_config_sha256": "m" * 64, "tessera_src_sha256": "s" * 64,
             "compilation_config": dict(RELEASE_CC), "speculative_tokens": 0,
             "max_model_len": 8448, "max_num_seqs": 8, "tensor_parallel_size": 1}
    serve.update(over)
    return serve


def test_an_equal_arm_attests_exactly_its_own_serve():
    receipt = _receipt()
    assert receipt["verdict"] == "equal"
    assert gr.verify(receipt, _serve()) is None
    # Key order is not scope: the canonical spelling compares.
    assert gr.verify(receipt, _serve(compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                                         "mode": "NONE"})) is None


@pytest.mark.parametrize("field, value", [
    ("tensor_parallel_size", 2), ("max_model_len", 4096), ("max_num_seqs", 4),
    ("speculative_tokens", 1), ("compilation_config", {"cudagraph_mode": "FULL_DECODE_ONLY"}),
    ("image", IMAGE.replace("5be13705", "e813795a")), ("model_config_sha256", "x" * 64),
])
def test_nothing_is_extrapolated_beyond_the_measured_scope(field, value):
    why = gr.verify(_receipt(), _serve(**{field: value}))
    assert why is not None and field in why


@pytest.mark.parametrize("arm, needle", [
    (_arm(members=47), "rGR"),             # one choice off eager
    (_arm(replays=0), "rGR"),              # a captured size never replayed
    (_arm(class_replays=0), "rGR"),        # the long class captured, never replayed
])
def test_a_graph_that_did_not_run_eagers_arithmetic_or_did_not_run_is_not_equal(arm, needle):
    receipt = _receipt(arm)
    assert receipt["verdict"] == "not_equal"
    assert needle in gr.verify(receipt, _serve())


def test_one_unequal_arm_makes_the_receipt_not_equal():
    receipt = _receipt(_arm(), _arm(name="rG1", members=0, cc={"cudagraph_mode": "FULL_DECODE_ONLY"}))
    assert receipt["verdict"] == "not_equal"


def test_an_edited_verdict_is_refused_by_the_rule():
    receipt = _receipt(_arm(members=0))
    forged = copy.deepcopy(receipt)
    forged["verdict"] = "equal"
    forged["arms"][0]["equal"] = True
    forged["attests"] = [gr.attestation(forged, forged["arms"][0])]
    assert "not equal" in gr.verify(forged, _serve())


def test_an_unknown_schema_is_refused():
    assert "schema" in gr.verify({"schema": "x"}, _serve())


def test_a_serve_that_does_not_name_its_scope_is_refused():
    serve = _serve()
    del serve["tensor_parallel_size"]
    assert "does not name" in gr.verify(_receipt(), serve)
