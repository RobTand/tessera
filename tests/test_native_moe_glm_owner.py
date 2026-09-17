"""The GLM-5.3-Flash routed owner geometry in the producer.

Tessera side of PrismaQuant #658. The producer's supported owner used to be one
complete 32-expert LFM stack; GLM-5.3-Flash is 288 experts with top-8
`noaux_tc` selection, a live FP32 correction bias, `norm_topk_prob`, routed
scale 2.5, a SwiGLU clamp of 10.0 and a served TP2 cut. These tests fix the
producer's half of the shared contract, and fix that the LFM half is unchanged.

CPU only: geometry, routing and member-shape validation, which is what both
sides must agree on before any panel is worth a GPU.
"""
from __future__ import annotations

import copy

import pytest

from experiments import bench_native_moe_operator as moe


def _glm_shape(**overrides):
    shape = {"geometry_version": 1, "geometry_id": "glm53_next_routed_stack_v1",
             "source_id": "glm5_next", "n_routed_experts": 288, "top_k": 8,
             "hidden_size": 4096, "intermediate_size": 2048,
             "shared_experts": 1,
             "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
             "scoring_func": "sigmoid", "norm_topk_prob": True,
             "routed_scaling_factor": 2.5, "swiglu_limit": 10.0, "gated": True,
             "tensor_parallel": 1, "tensor_parallel_cut_axis": "intermediate"}
    shape.update(overrides)
    assert set(shape) == set(moe.GLM_SHAPE_FIELDS), sorted(set(shape) ^ set(moe.GLM_SHAPE_FIELDS))
    return shape


def _glm_members(shape):
    roster = moe._roster_shape(shape)
    n, k = roster["intermediate_size"], roster["hidden_size"]
    return [{"unit": f"model.language_model.layers.3.mlp.experts.{expert}.{role}",
             "expert": expert, "role": role, "format": moe.FORMAT,
             "shape": [k, n] if role == "w2" else [n, k]}
            for expert in range(roster["experts"]) for role in ("w1", "w3", "w2")]


def _glm_routing(**overrides):
    routing = {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
               "routed_scaling_factor": 2.5, "apply_router_weight_on_input": False,
               "expert_map": None, "input_dtype": "torch.bfloat16",
               "topk_weights_dtype": "torch.float32", "topk_ids_dtype": "torch.int32",
               "device": "cuda:0",
               "weights_contract": "post_renormalization_and_routed_scaling",
               "swiglu_limit": 10.0, "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
               "source_protocol": {"router_class": "Glm5NextTopKRouter",
                                   "router_source_sha256": "a" * 64,
                                   "scoring_func": "sigmoid", "topk_method": "noaux_tc",
                                   "normalization_epsilon": 1e-6,
                                   "correction_bias": {"content_sha256": "b" * 64,
                                                       "dtype": "torch.float32"},
                                   "expert_bias_affects": "selection_only",
                                   "norm_topk_prob": True}}
    routing.update(overrides)
    return routing


def test_the_source_facts_match_the_shared_contract():
    """The producer's constants are PQ's, so the two sides cannot drift apart."""
    assert moe.GLM_SOURCE_FACTS["n_routed_experts"] == 288
    assert moe.GLM_SOURCE_FACTS["top_k"] == 8
    assert moe.GLM_SOURCE_FACTS["swiglu_limit"] == 10.0
    assert moe.GLM_SOURCE_FACTS["routed_scaling_factor"] == 2.5
    assert moe.GLM_GEOMETRY_VERSION == 1


def test_is_glm_geometry_is_a_property_of_the_shape_not_a_flag():
    assert moe.is_glm_geometry(_glm_shape())
    assert not moe.is_glm_geometry({"experts": 32, "hidden_size": 4, "intermediate_size": 4, "top_k": 2})
    assert not moe.is_glm_geometry(None)


def test_a_complete_glm_shape_validates():
    assert moe.validate_shape(_glm_shape())["n_routed_experts"] == 288


@pytest.mark.parametrize("override,expected", [
    ({"n_routed_experts": 32}, "n_routed_experts"),
    ({"top_k": 1}, "top_k"),
    ({"swiglu_limit": None}, "swiglu_limit"),
    ({"routed_scaling_factor": 1.0}, "routed_scaling_factor"),
    ({"topk_method": "greedy"}, "topk_method"),
    ({"norm_topk_prob": False}, "norm_topk_prob"),
    ({"hidden_size": 2048}, "hidden_size"),
])
def test_a_glm_shape_that_is_not_the_captured_source_refuses(override, expected):
    with pytest.raises(ValueError) as caught:
        moe.validate_shape(_glm_shape(**override))
    assert expected in str(caught.value), str(caught.value)


def test_an_unknown_glm_geometry_version_refuses():
    with pytest.raises(ValueError) as caught:
        moe.validate_shape(_glm_shape(geometry_version=2))
    assert "geometry version" in str(caught.value), str(caught.value)


@pytest.mark.parametrize("tp,expected", [(1, 2048), (2, 1024)])
def test_the_rank_local_member_shapes_follow_the_declared_cut(tp, expected):
    shape = moe.validate_shape(_glm_shape(tensor_parallel=tp))
    members = _glm_members(shape)
    assert len(members) == 288 * 3
    assert moe.validate_member_order(members, shape) is members
    assert members[0]["shape"] == [expected, 4096]
    assert members[2]["shape"] == [4096, expected]


def test_a_cut_the_operator_does_not_implement_refuses():
    with pytest.raises(ValueError) as caught:
        moe.validate_shape(_glm_shape(tensor_parallel_cut_axis="hidden"))
    assert "cut" in str(caught.value), str(caught.value)
    with pytest.raises(ValueError):
        moe.validate_shape(_glm_shape(tensor_parallel=4))


def test_a_glm_member_roster_of_the_wrong_length_refuses():
    shape = moe.validate_shape(_glm_shape())
    with pytest.raises(ValueError) as caught:
        moe.validate_member_order(_glm_members(shape)[:-1], shape)
    assert "expert-role members" in str(caught.value), str(caught.value)


def test_the_captured_glm_routing_validates_as_glm():
    moe.validate_routing(_glm_routing(), glm=True)
    # And the same object through the LFM door is refused rather than quietly
    # read with LFM's rules.
    with pytest.raises(ValueError):
        moe.validate_routing(_glm_routing())


@pytest.mark.parametrize("override,expected", [
    ({"swiglu_limit": 5.0}, "SwiGLU clamp"),
    ({"routed_scaling_factor": 1.0}, "routed scale"),
    ({"topk_method": "greedy"}, "top-k method"),
    ({"n_group": 2}, "group selection"),
])
def test_a_coordinate_the_route_reads_is_checked(override, expected):
    with pytest.raises(ValueError) as caught:
        moe.validate_routing(_glm_routing(**override), glm=True)
    assert expected in str(caught.value), str(caught.value)


def test_a_non_renormalizing_glm_source_refuses():
    routing = _glm_routing()
    routing["source_protocol"] = {**routing["source_protocol"], "norm_topk_prob": False}
    with pytest.raises(ValueError) as caught:
        moe.validate_routing(routing, glm=True)
    assert "norm_topk_prob" in str(caught.value), str(caught.value)


def test_a_correction_bias_that_is_not_the_sources_fp32_bias_refuses():
    for bad in ({"content_sha256": "b" * 64, "dtype": "torch.bfloat16"},
                {"content_sha256": "b" * 64}, "b" * 64):
        routing = _glm_routing()
        routing["source_protocol"] = {**routing["source_protocol"], "correction_bias": bad}
        with pytest.raises(ValueError) as caught:
            moe.validate_routing(routing, glm=True)
        assert "correction bias" in str(caught.value), str(caught.value)


def test_the_lfm_owner_is_unchanged_by_the_second_geometry():
    """The original 32-expert contract, field for field, including its refusals."""
    shape = {"experts": 32, "hidden_size": 4, "intermediate_size": 4, "top_k": 2}
    assert moe.validate_shape(shape) == shape
    assert not moe.is_glm_geometry(shape)
    for patch in ({"experts": 31}, {"top_k": 33}):
        with pytest.raises(ValueError) as caught:
            moe.validate_shape({**shape, **patch})
        assert "32-expert owner" in str(caught.value), str(caught.value)
    lfm = {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
           "routed_scaling_factor": 1.0, "apply_router_weight_on_input": False,
           "expert_map": None, "input_dtype": "torch.bfloat16",
           "topk_weights_dtype": "torch.float32", "topk_ids_dtype": "torch.int32",
           "device": "cuda:0",
           "weights_contract": "post_renormalization_and_routed_scaling",
           "source_protocol": {"router_class": "test.CapturedRouter",
                               "router_source_sha256": "a" * 64, "selection_bias": None,
                               "normalization_epsilon": 1e-6,
                               "expert_bias_affects": "selection_only"}}
    moe.validate_routing(copy.deepcopy(lfm))
    scaled = copy.deepcopy(lfm)
    scaled["routed_scaling_factor"] = 2.5
    with pytest.raises(ValueError):
        moe.validate_routing(scaled)


def test_the_shared_glm_contract_matches_prismaquants_consumer():
    """The producer's half of the contract, checked against PQ's own constants.

    Two repositories validate the same panel. The values they agree on are read
    from PQ's module directly rather than restated, so this test fails if either
    side moves a coordinate -- the drift a shared contract exists to prevent.
    """
    pq = pytest.importorskip("prismaquant.native_moe_panel")
    assert pq.GEOMETRY_VERSION == moe.GLM_GEOMETRY_VERSION
    assert pq.GLM_SOURCE_GEOMETRY["n_routed_experts"] == moe.GLM_SOURCE_FACTS["n_routed_experts"]
    assert pq.GLM_SOURCE_GEOMETRY["top_k"] == moe.GLM_SOURCE_FACTS["top_k"]
    assert pq.GLM_SOURCE_GEOMETRY["swiglu_limit"] == moe.GLM_SOURCE_FACTS["swiglu_limit"]
    assert pq.GLM_SOURCE_GEOMETRY["routed_scaling_factor"] == moe.GLM_SOURCE_FACTS["routed_scaling_factor"]
    assert pq.GLM_SOURCE_GEOMETRY["topk_method"] == moe.GLM_SOURCE_FACTS["topk_method"]
    assert pq.GLM_SOURCE_GEOMETRY["norm_topk_prob"] == moe.GLM_SOURCE_FACTS["norm_topk_prob"]
    assert pq.GLM_TP_CUT_AXIS == "intermediate"
    assert set(pq.SUPPORTED_TP_SIZES) == {1, 2}
    # The field sets are the same object on both sides, not two spellings.
    assert set(pq.GLM_SHAPE_FIELDS) == set(moe.GLM_SHAPE_FIELDS)
    assert set(pq.GLM_SOURCE_GEOMETRY) | {"geometry_version", "tensor_parallel",
                                          "tensor_parallel_cut_axis"} == set(moe.GLM_SHAPE_FIELDS)
