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

# The bench modules below reach the native wire, which needs torch.  The
# bytes-only CI job installs pytest alone, so this module refuses collection
# there rather than running a test whose subject cannot be imported; the
# sibling owner-runtime and operator-receipt modules carry the same guard.
torch = pytest.importorskip("torch")

from experiments import bench_native_moe_operator as moe
from experiments.bench_native_operator import tensor_identity as dense_tensor_identity

GLM_UNIT = "model.language_model.layers.3.mlp.experts"


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


def test_a_missing_or_inconsistent_renormalization_capture_refuses():
    """The consumer's cross-check, mirrored: the producer must refuse it too.

    Root review found both sides accepted `correction_bias=None`, which reads as
    a source without a bias, and `renormalize=False` while the source protocol
    said `norm_topk_prob=True`. `noaux_tc` selects on the bias and the served
    route normalizes before applying the routed scale, so neither capture
    describes this model.
    """
    routing = _glm_routing()
    routing["source_protocol"] = {**routing["source_protocol"], "correction_bias": None}
    with pytest.raises(ValueError) as caught:
        moe.validate_routing(routing, glm=True)
    assert "correction bias" in str(caught.value), str(caught.value)

    for renormalize, norm_topk_prob in ((False, True), (True, False)):
        routing = _glm_routing(renormalize=renormalize)
        routing["source_protocol"] = {**routing["source_protocol"],
                                      "norm_topk_prob": norm_topk_prob}
        with pytest.raises(ValueError) as caught:
            moe.validate_routing(routing, glm=True)
        message = str(caught.value)
        assert "renormalize" in message or "norm_topk_prob" in message, message


@pytest.mark.parametrize("key,value", [
    ("geometry_version", True), ("geometry_version", 1.0), ("tensor_parallel", True),
    ("tensor_parallel", 1.0), ("n_routed_experts", True), ("top_k", 8.0),
])
def test_a_non_integer_geometry_coordinate_refuses_before_width_arithmetic(key, value):
    """`True` is an int and `1.0 == 1`; neither may become a slice bound."""
    with pytest.raises(ValueError) as caught:
        moe.validate_shape(_glm_shape(**{key: value}))
    assert key in str(caught.value), str(caught.value)


# --------------------------------------------------------------------------
# GEOMETRY AND CONTRACT, NOT A DEVICE RESULT.
#
# These tests establish the geometry and routing contracts, and that a GLM
# owner's shape/roster/execution record reach the producer's own validators.
# They do not run a device: the harness's own runtime path -- the owner's
# format becoming a family/grid/rung, a TP2 serving config, a real rank, and a
# panel the validator accepts at that family -- is
# `tests/test_native_moe_tp_owner_runtime.py`'s, and the CUDA layer
# construction plus the native decode are the GPU qualification step named
# there.
#
# `verify_routing_bias` reads whichever spelling the protocol block declares --
# `selection_bias` for LFM, `correction_bias` for a GLM `noaux_tc` owner -- and
# checks it against the payload this process was handed, so the cross-repository
# payload test below drives the REQUEST path rather than a hand translation.
# --------------------------------------------------------------------------

def _pq_owner_view(tp, format_name, unit=GLM_UNIT):
    """Build a GLM owner the way PQ's producer does, through PQ's own helpers."""
    pq = pytest.importorskip("prismaquant.native_moe_panel")
    validated = pq.validate_geometry({
        "geometry_version": pq.GEOMETRY_VERSION, "geometry_id": "glm53_next_routed_stack_v1",
        "source_id": "glm5_next", "n_routed_experts": 288, "top_k": 8,
        "hidden_size": 4096, "intermediate_size": 2048, "shared_experts": 1,
        "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
        "scoring_func": "sigmoid", "norm_topk_prob": True,
        "routed_scaling_factor": 2.5, "swiglu_limit": 10.0, "gated": True,
        "tensor_parallel": tp, "tensor_parallel_cut_axis": pq.GLM_TP_CUT_AXIS})
    shape = {**validated, "format": format_name}
    view = pq._shape_for_roster(shape)
    members = [{"unit": f"{unit}.{expert}.{role}", "expert": expert, "role": role,
                "format": format_name}
               for expert in range(view["experts"]) for role in ("w1", "w3", "w2")]
    return pq, shape, view, members


@pytest.mark.parametrize("format_name", ["TESSERA_E2M1_K2_R896", "TESSERA_E4M3_K1_R1024",
                                         "TESSERA_BF16_K1_R1024"])
def test_a4_a8_and_a16_each_reach_the_whole_owner_validator(format_name):
    """The contract accepts these families, and the roster carries the format.

    This checks the geometry/routing contract and the roster's format
    parameterization. It does not run a device: what the harness then RESOLVES
    from that format -- family, grid, rung, sidecar and route -- is asserted in
    `tests/test_native_moe_tp_owner_runtime.py`.
    """
    pq, shape, view, members = _pq_owner_view(1, format_name)
    assert view["format"] == format_name
    assert {m["format"] for m in members} == {format_name}

    # A real request carries the owner's FORMAT beside the geometry, because a
    # whole routed owner holds one format for all of its members; the shape the
    # consumer prepared has it, so the producer's roster check reads the same
    # rung rather than the module constant.
    producer_shape = {k: v for k, v in shape.items() if k in moe.GLM_SHAPE_FIELDS}
    producer_shape["format"] = format_name
    member_inputs = [{**m, "blob": b"", "record": {}} for m in members]
    assert len(moe.validate_member_order_public(member_inputs, producer_shape)) == 288 * 3
    # And the rung the producer reads for the wire is the OWNER's, not E4M3 R1024.
    assert moe._parse_owner_format(format_name) == ("E2M1", 896) if "E2M1" in format_name \
        else moe._parse_owner_format(format_name)[1] == 1024


@pytest.mark.parametrize("tp", [1, 2])
def test_the_declared_tensor_parallel_reaches_the_whole_receipt_execution_check(tp):
    """The execution RECORD the harness stamps is this geometry's own cut.

    `validate_execution`, the prepared operator's runtime manifest and the
    panel's execution check all read `owner_execution(shape)`, so a TP2 owner
    is stamped TP2 and can no longer be relabelled with the LFM record.
    """
    pq, shape, view, members = _pq_owner_view(tp, "TESSERA_E4M3_K1_R1024")
    execution = pq.owner_execution(shape, format_name="TESSERA_E4M3_K1_R1024")
    assert execution["tensor_parallel"] == tp
    producer_shape = {k: v for k, v in shape.items() if k in moe.GLM_SHAPE_FIELDS}
    producer_shape["format"] = "TESSERA_E4M3_K1_R1024"
    # The owner's own record is accepted...
    assert moe.validate_execution(execution, list(moe.ROLE_ORDER), shape=producer_shape) is None
    # ...and the LFM TP1 record is refused for a TP2 owner.
    if tp == 2:
        with pytest.raises(ValueError) as caught:
            moe.validate_execution(dict(moe.EXECUTION), list(moe.ROLE_ORDER), shape=producer_shape)
        assert "tensor-parallel" in str(caught.value), str(caught.value)
    # The member widths both sides price are the rank's own -- 2048//tp -- and
    # they are carried under each side's own spelling. The producer's roster
    # view puts the rank-local width in `intermediate_size` (`_roster_shape`
    # divides there); PQ's view keeps the FULL declared `intermediate_size` and
    # carries the rank-local width separately as `rank_local_intermediate`
    # (`_shape_for_roster` / `rank_local_intermediate`), which is the key its
    # member geometry reads at `_expect_member_roster`. Comparing the two
    # `intermediate_size` keys compares a rank-local width against a declared
    # source width, so the keys that mean the same thing must be the ones
    # compared.
    producer_view = moe._roster_shape(producer_shape)
    rank_local = 2048 // tp
    assert producer_view["intermediate_size"] == view["rank_local_intermediate"] == rank_local
    # ...and PQ's declared geometry is untouched by its own derived view: the
    # full source width is still declared, and stripping the view's derived
    # keys returns the declared geometry. `format` is one of those derived
    # keys, so the comparison is against the geometry without it -- the shape
    # the caller prepared carries the owner's format, which the view does not
    # declare.
    assert shape["intermediate_size"] == 2048
    assert view["intermediate_size"] == 2048
    declared = {k: v for k, v in shape.items() if k != "format"}
    assert pq.geometry_only(view) == declared
    assert pq.geometry_family(pq.geometry_only(view)) == "glm53_next_routed_stack_v1"

    # The rank-local width is not a statement about the values: it must reach
    # the actual member tensor shapes both sides expect. The producer's member
    # geometry and PQ's member geometry are read from their real helpers, and
    # PQ's roster check is driven with members carrying exactly those shapes.
    member_shapes = {role: moe._member_shape(producer_shape, role) for role in moe.ROLE_ORDER}
    member_roster = [{**m, "shape": list(member_shapes[m["role"]])} for m in members]
    assert pq._member_roster(GLM_UNIT, member_roster, shape) == member_roster
    for role in ("w1", "w3"):
        assert member_shapes[role] == [rank_local, 4096]
    assert member_shapes["w2"] == [4096, rank_local]
    # A w1 carrying a width the rank does not declare is refused -- otherwise
    # the keys above could agree while the actual member shapes did not. The
    # negative has to be built from THIS rank's own width: for TP1 the full
    # 2048 IS the rank-local width, so a full-width w1 is valid there and only
    # the TP2 case is a mismatch.
    planted = 2048 if tp == 2 else rank_local + 1
    wrong = [{**m, "shape": [planted, 4096]} if m["role"] == "w1"
             else {**m, "shape": list(member_shapes[m["role"]])} for m in members]
    with pytest.raises(ValueError):
        pq._member_roster(GLM_UNIT, wrong, shape)

    # This asserts the CONTRACT's rank-local geometry, not a device result. The
    # rank-local member rows a request carries and the module rows a container
    # frames are separated in the harness by `_declared_member_rows`, and
    # `tests/test_native_moe_tp_owner_runtime.py` covers that split and the
    # TP2 serving config that reaches it.


# --------------------------------------------------------------------------
# The cross-repository payload translation root asked for: PQ's GLM protocol
# spells the correction bias `correction_bias`; the producer's
# `verify_routing_bias:538` reads `source_protocol["selection_bias"]`. Those are
# two spellings of one captured object across the boundary, and until this test
# nothing connected them -- so a real GLM panel would have been refused (or
# worse, compared the wrong tensor) with no test saying so.
# --------------------------------------------------------------------------

def test_the_producer_reads_the_bias_under_the_consumers_glm_spelling(monkeypatch):
    """A PQ-shaped GLM protocol must reach the producer's bias check intact.

    This is a payload-translation test, not a geometry test: it builds the
    routing document in PRISMAQUANT's shape (where the field is
    `correction_bias`, carrying a content digest and dtype), converts it the way
    a real request must, and then drives the producer's own
    `verify_routing_bias` against an actual FP32 tensor. Both the translation
    and the comparison are exercised, so the missing link is visible.
    """
    import torch

    pq = pytest.importorskip("prismaquant.native_moe_panel")
    routing = _glm_routing()
    source = routing["source_protocol"]
    # PQ's shape, verbatim: correction_bias, a digest + dtype, no tensor.
    assert "correction_bias" in source and "selection_bias" not in source
    assert set(source["correction_bias"]) == {"content_sha256", "dtype"}

    # The producer's field is `selection_bias`, and it compares the FULL tensor
    # record (shape, dtype, logical_bytes, content digest) against the tensor it
    # was given, so the translation carries the record `tensor_identity`
    # produces -- not the digest-and-dtype pair PQ stores, which is a different
    # schema and is exactly what `tensor_record` refuses.
    bias = torch.zeros(288, dtype=torch.float32)
    producer_protocol = {
        "router_class": source["router_class"],
        "router_source_sha256": source["router_source_sha256"],
        "selection_bias": dense_tensor_identity(bias),
        "normalization_epsilon": source["normalization_epsilon"],
        "expert_bias_affects": source["expert_bias_affects"],
    }
    # The two schemas are NOT interchangeable, and that is the cross-repository
    # fact this test exists to record: PQ stores {content_sha256, dtype} and the
    # producer requires {shape, dtype, logical_bytes, content_sha256}. A request
    # builder must materialize the tensor record.
    assert set(producer_protocol["selection_bias"]) == {
        "shape", "dtype", "logical_bytes", "content_sha256"}
    # The translation keeps `topk_method`, which is what the producer's width
    # rule keys on; dropping it would size the bias at the LFM 32 and refuse the
    # 288-wide GLM tensor. That is a real request-path hazard, recorded below.
    producer_routing = {**routing, "source_protocol": producer_protocol}

    # The producer's own bias check accepts the translated payload.
    assert moe.verify_routing_bias(producer_routing, bias) is bias
    # The REQUEST path takes PQ's spelling directly: `verify_routing_bias`
    # reads `correction_bias` when the protocol declares it, checks the
    # declared dtype and digest against the payload this process was handed,
    # and holds it to the captured source's 288 experts. No translation step
    # is needed at the call site, so nothing can drop `topk_method` on the way.
    request_routing = _glm_routing()
    request_routing["source_protocol"]["correction_bias"] = {
        "dtype": producer_protocol["selection_bias"]["dtype"],
        "content_sha256": producer_protocol["selection_bias"]["content_sha256"]}
    assert moe.verify_routing_bias(request_routing, bias) is bias
    assert "routing_bias" in moe.request_tensor_roster(request_routing, [
        {"unit": "u.0.w1"}, {"unit": "u.0.w3"}, {"unit": "u.0.w2"}])

    # And it refuses a bias that is not the captured one, which is the point of
    # translating rather than passing the name through.
    other = torch.ones(288, dtype=torch.float32)
    with pytest.raises(ValueError) as caught:
        moe.verify_routing_bias(producer_routing, other)
    assert "captured source" in str(caught.value), str(caught.value)
    # Under PQ's own spelling the same wrong tensor is refused by the digest.
    with pytest.raises(ValueError) as caught:
        moe.verify_routing_bias(request_routing, other)
    assert "not the captured source's bytes" in str(caught.value), str(caught.value)

    # The producer's width rule follows the geometry's own expert count: 288
    # for GLM, not the LFM 32.
    with pytest.raises(ValueError):
        moe.verify_routing_bias(producer_routing, torch.zeros(32, dtype=torch.float32))
