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


def _receipt(*arms, fabric="none"):
    return gr.finish({"schema": gr.SCHEMA, "runtime": {"image": IMAGE, "fabric": fabric},
                      "model": {"config_sha256": "m" * 64}, "tessera": {"src_sha256": "s" * 64},
                      "arms": list(arms) or [_arm()]})


def _serve(**over):
    serve = {"image": IMAGE, "model_config_sha256": "m" * 64, "tessera_src_sha256": "s" * 64,
             "compilation_config": dict(RELEASE_CC), "speculative_tokens": 0,
             "max_model_len": 8448, "max_num_seqs": 8, "tensor_parallel_size": 1,
             "fabric": "none"}
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


@pytest.mark.parametrize("damage", [
    lambda r: r["arms"][0].pop("graph"),
    lambda r: r["arms"][0]["passes"][0].pop("members"),
    lambda r: r.pop("runtime"),
    lambda r: r.update(arms="not a list"),
])
def test_a_malformed_receipt_is_refused_with_a_reason_not_an_exception(damage):
    receipt = _receipt()
    damage(receipt)
    why = gr.verify(receipt, _serve())
    assert isinstance(why, str) and "malformed" in why


# ------------------------------------------------------------------ v2: the fabric is scope


def _tp2_arm(**over):
    arm = _arm(**over)
    arm["tensor_parallel_size"] = 2
    return arm


def test_the_schema_is_v2_and_the_fabric_is_scope():
    assert gr.SCHEMA == "tessera.graph_equals_eager.v2"
    assert "fabric" in gr.SCOPE_FIELDS


def test_a_socket_receipt_does_not_attest_a_roce_serve():
    """dec-1005-003356-6ba2: an all-reduce over sockets and one over RoCE are two serves."""
    receipt = _receipt(_tp2_arm(), fabric="socket")
    why = gr.verify(receipt, _serve(tensor_parallel_size=2, fabric="roce"))
    assert why is not None and "fabric" in why
    assert gr.verify(receipt, _serve(tensor_parallel_size=2, fabric="socket")) is None


def test_a_roce_receipt_attests_a_roce_serve():
    receipt = _receipt(_tp2_arm(), fabric="roce")
    assert gr.verify(receipt, _serve(tensor_parallel_size=2, fabric="roce")) is None


@pytest.mark.parametrize("fabric, tp", [("none", 2), ("socket", 1), ("infiniband", 2), (None, 2)])
def test_a_fabric_that_does_not_fit_the_serve_is_refused(fabric, tp):
    """A tensor-parallel serve names socket or roce; a one-rank serve has no fabric ("none")."""
    receipt = _receipt(_tp2_arm() if tp == 2 else _arm(), fabric=fabric)
    why = gr.verify(receipt, _serve(tensor_parallel_size=tp, fabric=fabric))
    assert why is not None and "fabric" in why


def test_a_v1_receipt_stays_readable_but_verifies_no_card():
    receipt = _receipt()
    receipt["schema"] = "tessera.graph_equals_eager.v1"
    del receipt["runtime"]["fabric"]
    assert gr.finish(receipt)["verdict"] == "equal"           # still readable
    why = gr.verify(receipt, _serve())
    assert why is not None and "v1" in why and "fabric" in why
