"""CPU boundary tests for the whole-owner receipt; never native GPU evidence."""
import copy
import hashlib

import pytest

torch = pytest.importorskip("torch")
from experiments import bench_native_moe_operator as moe
from experiments import bench_native_operator as dense


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _shape():
    return {"experts": 32, "hidden_size": 256, "intermediate_size": 128, "top_k": 4}


def _members():
    return [{"unit": f"model.layers.1.feed_forward.experts.{expert}.{role}",
             "expert": expert, "role": role, "format": moe.FORMAT}
            for expert in range(32) for role in ("w1", "w3", "w2")]


def _routing():
    return {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
            "routed_scaling_factor": 1.0, "apply_router_weight_on_input": False,
            "expert_map": None, "input_dtype": "torch.bfloat16", "topk_weights_dtype": "torch.float32",
            "topk_ids_dtype": "torch.int32", "device": "cuda:0",
            "weights_contract": "post_renormalization_and_routed_scaling",
            "source_protocol": {"router_class": "test.CapturedRouter",
                "router_source_sha256": _sha("CPU source fixture"), "selection_bias": None,
                "normalization_epsilon": 1e-6, "expert_bias_affects": "selection_only"}}


def test_complete_owner_preserves_explicit_w1_w3_w2_role_order():
    members = _members()
    assert moe.validate_member_order(members, _shape()) is members
    assert [member["role"] for member in members[:3]] == ["w1", "w3", "w2"]
    # Lexical w1,w2,w3 sorting changes the actual gated operator.
    with pytest.raises(ValueError, match="role"):
        moe.validate_member_order(sorted(members, key=lambda m: (m["expert"], m["unit"])), _shape())


@pytest.mark.parametrize("mutation", [
    lambda m: m.pop(), lambda m: m.append(copy.deepcopy(m[0])),
    lambda m: m.__setitem__(3, copy.deepcopy(m[0])),
    lambda m: m[0].update(expert=True), lambda m: m[3].update(expert=0),
    lambda m: m[1].update(role="w2"), lambda m: m[1].update(unit=m[0]["unit"]),
    lambda m: m[0].update(unit=""), lambda m: m[0].update(format="TESSERA_E4M3_K1_R512"),
])
def test_missing_extra_reordered_duplicate_or_wrong_rung_member_refuses(mutation):
    members = _members()
    mutation(members)
    with pytest.raises(ValueError):
        moe.validate_member_order(members, _shape())


@pytest.mark.parametrize("patch", [{"experts": 31}, {"experts": True}, {"hidden_size": 0},
                                    {"intermediate_size": -1}, {"top_k": 33}, {"top_k": False}])
def test_unsupported_or_bool_geometry_refuses(patch):
    with pytest.raises(ValueError):
        moe.validate_shape({**_shape(), **patch})


@pytest.mark.parametrize("patch", [{"tensor_parallel": 2}, {"tensor_parallel": True},
    {"expert_parallel": 2}, {"owner_kind": "single_expert"}, {"include_router": True},
    {"shared_experts": True}, {"monolithic": True}, {"bias": 0}, {"mode": "streamed"},
    {"execution_mode": "compiled"}])
def test_unsupported_execution_cannot_relabel_a_whole_receipt(patch):
    with pytest.raises(ValueError):
        moe.validate_execution({**moe.EXECUTION, **patch}, list(moe.ROLE_ORDER))


@pytest.mark.parametrize("patch", [{"scoring_func": "softmax"}, {"activation": "gelu"},
    {"renormalize": 1}, {"routed_scaling_factor": True}, {"routed_scaling_factor": 2.0},
    {"apply_router_weight_on_input": True}, {"expert_map": [0]}, {"device": "cuda:1"},
    {"input_dtype": "torch.float32"}, {"topk_ids_dtype": "torch.float32"},
    {"topk_weights_dtype": "torch.int32"}])
def test_routing_drift_or_unsupported_execution_refuses(patch):
    with pytest.raises(ValueError):
        moe.validate_routing({**_routing(), **patch})


@pytest.mark.parametrize("weights_dtype", ["torch.bfloat16", "torch.float32"])
@pytest.mark.parametrize("ids_dtype", ["torch.int32", "torch.int64"])
def test_routing_preserves_captured_or_losslessly_transported_dtypes(weights_dtype, ids_dtype):
    routing = {**_routing(), "topk_weights_dtype": weights_dtype, "topk_ids_dtype": ids_dtype}
    assert moe.validate_routing(routing) == routing


@pytest.mark.parametrize("source,supplied,key", [
    (torch.tensor([[0.25, 0.125]], dtype=torch.bfloat16), torch.tensor([[0.25, 0.125]], dtype=torch.float32), "topk_weights"),
    (torch.tensor([[1, 3]], dtype=torch.int64), torch.tensor([[1, 3]], dtype=torch.int32), "topk_ids"),
])
def test_lossless_transport_reconstructs_the_raw_capture_hash(source, supplied, key):
    record = {"source": dense.tensor_identity(source), "supplied": dense.tensor_identity(supplied),
              "operation": "lossless_dtype_conversion"}
    moe.validate_transport(record, supplied, key=key)
    changed = supplied.clone()
    changed[0, 0] += 1
    # Even updating the supplied hash cannot hide changing the raw source.
    record["supplied"] = dense.tensor_identity(changed)
    with pytest.raises(ValueError, match="lossless"):
        moe.validate_transport(record, changed, key=key)


@pytest.mark.parametrize("source,supplied,key", [
    (torch.tensor([[0.10001]], dtype=torch.float32), torch.tensor([[0.10001]], dtype=torch.bfloat16), "topk_weights"),
    (torch.tensor([[2**40 + 1]], dtype=torch.int64), torch.tensor([[1]], dtype=torch.int32), "topk_ids"),
])
def test_lossy_rounding_and_integer_overflow_cannot_claim_lossless_transport(source, supplied, key):
    record = {"source": dense.tensor_identity(source), "supplied": dense.tensor_identity(supplied),
              "operation": "lossless_dtype_conversion"}
    with pytest.raises(ValueError, match="lossless"):
        moe.validate_transport(record, supplied, key=key)


def test_identity_transport_requires_exact_source_and_supplied_bytes():
    supplied = torch.tensor([[0, 1]], dtype=torch.int32)
    record = {"source": dense.tensor_identity(supplied), "supplied": dense.tensor_identity(supplied),
              "operation": "identity"}
    moe.validate_transport(record, supplied, key="topk_ids")
    record["source"]["content_sha256"] = _sha("different captured IDs")
    with pytest.raises(ValueError, match="identity transport"):
        moe.validate_transport(record, supplied, key="topk_ids")


def test_native_configuration_has_no_unstable_repr_fallback():
    with pytest.raises(ValueError, match="unobserved native configuration"):
        moe._plain(object())


def _record(shape, dtype="torch.bfloat16"):
    import math
    return {"shape": shape, "dtype": dtype, "logical_bytes": math.prod(shape) * moe.DTYPE_BYTES[dtype],
            "content_sha256": _sha(str(shape) + dtype)}


def _panel():
    shape = _shape()
    members = _members()
    for member in members:
        geometry = moe._member_shape(shape, member["role"])
        record = {"blob_sha256": _sha(member["unit"]), "blob_bytes": 100}
        member.update(shape=geometry, source_weight=_record(geometry), rendered_weight=_record(geometry),
            activation={"clip_enabled": False, "input_global_scale": None},
            wire={**record, "record": dict(record)})
    route = {"kind": "moe", "policy": "TESSERA_FP8:resident", "decoder": "torch_materialize_stock",
             "contract": "fp8_per_token_dynamic", "symbol": "vllm.fused_moe.modular_kernel:TEST_ONLY"}
    phases = {}
    routing = _routing()
    for phase, m in (("prefill", 8), ("decode", 1)):
        phases[phase] = {"m": m, "expected_route": route, "transport": {}}
        for key in moe.TENSOR_KEYS:
            width = shape["top_k"] if key in ("topk_ids", "topk_weights") else shape["hidden_size"]
            dtype = routing[key + "_dtype"] if key in ("input", "topk_ids", "topk_weights") else "torch.bfloat16"
            phases[phase][key] = _record([m, width], dtype)
        for key in ("topk_ids", "topk_weights"):
            phases[phase]["transport"][key] = {"source": dict(phases[phase][key]),
                "supplied": dict(phases[phase][key]), "operation": "identity"}
    workspace = {"schema": moe.WORKSPACE_SCHEMA, "owner": "vllm.WorkspaceManager", "num_ubatches": 1,
                 "num_lanes": 1, "locked": True, "slots": [{"index": 0, "allocation": None}], "resident_bytes": 0}
    panel = {"schema": moe.PANEL_SCHEMA, "unit": "model.layers.1.feed_forward.experts", "format": moe.FORMAT,
        "shape": shape, "members": members, "profile_role_order": list(moe.ROLE_ORDER), "routing": routing,
        "probe_scope": None, "execution": dict(moe.EXECUTION), "runtime": {"schema": moe.RUNTIME_SCHEMA, "execution": dict(moe.EXECUTION)},
        "numerics": {"atol": 2**-6, "rtol": 2**-6}, "phases": phases,
        "workspace": workspace, "workspace_sha256": dense.identity_sha256(workspace),
        "runtime_binding": {"member_formats": {m["unit"]: m["format"] for m in members},
            "member_shapes": {m["unit"]: m["shape"] for m in members},
            "member_operator_identity_sha256": {m["unit"]: _sha(m["unit"] + "joint") for m in members},
            "operator_route": route["symbol"]}}
    for key in ("routing_capture_sha256", "source_sha256", "calibration_sha256", "cost_sha256",
                "probe_identity_sha256", "native_tensors_sha256", "scheme_sha256", "config_sha256", "serving_config_sha256"):
        panel[key] = _sha(key)
    return panel


def test_frozen_panel_accepts_complete_members_and_unallocated_workspace_slot():
    panel = _panel()
    assert moe.validate_panel(panel) == panel


@pytest.mark.parametrize("qualification", [None, _sha("independent source execution proof")])
def test_panel_preserves_explicit_source_execution_and_qualification(qualification):
    panel = _panel()
    panel.update(source_execution={"schema": "prismaquant.joint_aura.source_execution.v1",
        "modules": {"": {"attention": "eager", "experts": "grouped_mm"},
                    "model.layers.2": {"experts": {"text": "grouped_mm"}}}},
        source_execution_qualification_sha256=qualification)
    original = dense.identity_sha256(panel)
    assert moe.validate_panel(panel) == panel
    assert dense.identity_sha256(panel) == original
    panel["source_execution"]["modules"][""]["experts"] = "eager"
    assert dense.identity_sha256(panel) != original


@pytest.mark.parametrize("defect", ["missing_descriptor", "missing_qualification", "bad_sha",
    "bad_schema", "extra_field", "no_root", "empty_selectors", "unknown_selector", "bad_selector"])
def test_panel_refuses_malformed_source_execution_join(defect):
    panel = _panel()
    panel.update(source_execution={"schema": "prismaquant.joint_aura.source_execution.v1",
        "modules": {"": {"attention": "eager", "experts": "grouped_mm"}}},
        source_execution_qualification_sha256=None)
    execution = panel["source_execution"]
    if defect == "missing_descriptor":
        panel.pop("source_execution")
    elif defect == "missing_qualification":
        panel.pop("source_execution_qualification_sha256")
    elif defect == "bad_sha":
        panel["source_execution_qualification_sha256"] = "unverified"
    elif defect == "bad_schema":
        execution["schema"] = "unknown"
    elif defect == "extra_field":
        execution["assertion"] = "measured"
    elif defect == "no_root":
        execution["modules"] = {"child": {"experts": "grouped_mm"}}
    elif defect == "empty_selectors":
        execution["modules"][""] = {}
    elif defect == "unknown_selector":
        execution["modules"][""]["other"] = "ignored"
    elif defect == "bad_selector":
        execution["modules"][""]["experts"] = ["grouped_mm"]
    with pytest.raises(ValueError):
        moe.validate_panel(panel)


@pytest.mark.parametrize("mutate", [
    lambda p: p["runtime_binding"]["member_operator_identity_sha256"].pop(next(iter(p["runtime_binding"]["member_formats"]))),
    lambda p: p["runtime_binding"].update(operator_route="vllm.fused_moe.modular_kernel:OTHER"),
    lambda p: p["members"][0]["wire"]["record"].update(blob_sha256=_sha("changed wire")),
    lambda p: p["members"][0]["activation"].update(clip_enabled=True),
    lambda p: p["members"][0].update(unit="other.owner.0.w1"),
    lambda p: p["workspace"].update(locked=False),
    lambda p: p["workspace"].update(num_lanes=True),
    lambda p: p["phases"]["decode"]["transport"]["topk_weights"]["source"].update(content_sha256=_sha("changed raw weights")),
    lambda p: p["phases"]["decode"]["transport"]["topk_ids"].update(operation="renormalize"),
    lambda p: p["phases"]["decode"]["topk_ids"].update(dtype="torch.float32"),
    lambda p: p["phases"]["decode"].update(m=True),
    lambda p: p.update(serving_config_sha256="missing"),
    lambda p: p.update(summed_expert_median=1.0),
])
def test_panel_refuses_incomplete_join_routing_drift_and_unpriced_extra_fields(mutate):
    panel = _panel()
    mutate(panel)
    with pytest.raises(ValueError):
        moe.validate_panel(panel)


def test_factory_bias_is_actual_captured_values_not_a_zero_substitute():
    routing = _routing()
    captured = torch.arange(32, dtype=torch.float32) / 32
    routing["source_protocol"]["selection_bias"] = dense.tensor_identity(captured)
    assert moe.verify_routing_bias(routing, captured) is captured
    with pytest.raises(ValueError, match="differs"):
        moe.verify_routing_bias(routing, torch.zeros_like(captured))
    with pytest.raises(ValueError, match="actual FP32"):
        moe.verify_routing_bias(routing, None)
    with pytest.raises(ValueError, match="unexpected"):
        moe.verify_routing_bias(_routing(), captured)


def _fake_whole_lifecycle(monkeypatch, *, bad_decode=False):
    """Explicit CPU substitutions test ordering, never native numerical evidence."""
    from types import SimpleNamespace
    from tessera.serving.telemetry import emit_route
    panel = _panel()
    values = {}
    for phase in moe.PHASES:
        m = panel["phases"][phase]["m"]
        x = torch.ones(m, 256, dtype=torch.bfloat16)
        ids = torch.arange(4, dtype=torch.int32).repeat(m, 1)
        weights = torch.full((m, 4), .25)
        values[phase] = {"input": x, "topk_ids": ids, "topk_weights": weights,
                         "reference_qdq": x.clone(), "reference_output": x.clone()}
        for key, value in values[phase].items():
            panel["phases"][phase][key] = dense.tensor_identity(value)
        for key in ("topk_ids", "topk_weights"):
            ident = dense.tensor_identity(values[phase][key])
            panel["phases"][phase]["transport"][key] = {"source": ident, "supplied": ident, "operation": "identity"}
    layer = torch.nn.Module()
    layer.register_buffer("weight", torch.ones(32, dtype=torch.bfloat16))
    layer.tessera_mode = "resident"
    layer.quant_method = SimpleNamespace(is_monolithic=False)
    config = {"fixture": "CPU_ONLY"}
    panel["runtime"]["arithmetic"] = dense.observe_arithmetic()
    operator = {key: copy.deepcopy(panel[key]) for key in
        ("shape", "routing", "profile_role_order", "routing_capture_sha256", "serving_config_sha256")}
    operator.update(native_tensors=dense._native_tensors(layer), scheme={"fixture": "CPU_ONLY"}, config=config,
        declared_route=panel["phases"]["prefill"]["expected_route"],
        phases={phase: {"transport": panel["phases"][phase]["transport"]} for phase in moe.PHASES},
        members=[{**{key: member[key] for key in ("unit", "expert", "role", "format", "shape", "source_weight", "rendered_weight")},
                  "wire_sha256": member["wire"]["blob_sha256"],
                  "wire_record_sha256": dense.identity_sha256(member["wire"]["record"])} for member in panel["members"]])
    for key in ("native_tensors", "scheme", "config"):
        panel[key + "_sha256"] = dense.identity_sha256(operator[key])
    prepared = {"layer": layer, "operator": operator, "runtime": copy.deepcopy(panel["runtime"]),
                "workspace": copy.deepcopy(panel["workspace"]), "workspace_pointers": (None,)}
    calls = []
    def apply(actual, tensors):
        assert actual is layer
        m = tensors["input"].shape[0]
        calls.append(("apply", m))
        emit_route(layer, **operator["declared_route"], shape=f"M{m}:N256:K256", state="served", reason=None)
        out = tensors["input"].clone()
        if bad_decode and m == 1:
            out[0, 0] += 10
        return out
    def time(apply, **kwargs):
        calls.append(("time", None))
        apply()
        return {"method": "cuda_events", "sample_unit": "single_apply", "warmup_iterations": 2, "samples_ms": [1., 2., 3.]}
    monkeypatch.setattr(moe, "_require_cuda", lambda x: None)
    monkeypatch.setattr(dense, "_require_eager_context", lambda: None)
    monkeypatch.setattr(dense, "_check_native_library_scope", lambda runtime: None)
    monkeypatch.setattr(dense, "_resident_bytes", lambda layer: 64)
    monkeypatch.setattr(moe, "_native_config", lambda layer: config)
    monkeypatch.setattr(moe, "observe_workspace", lambda: (copy.deepcopy(panel["workspace"]), (None,)))
    monkeypatch.setattr(moe, "apply_whole", apply)
    monkeypatch.setattr(dense, "represented_native_input", lambda layer, x: x.clone())
    monkeypatch.setattr(dense, "time_apply", time)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1500)
    return prepared, panel, values, calls


def _measure_whole(fixture, collector=None):
    prepared, panel, values, _ = fixture
    return moe.measure_prepared_operator(prepared, panel, values, warmup_iterations=2,
                                         iterations=3, resource_collector=collector)


def test_whole_both_phase_gates_precede_any_samples_and_global_resources_stay_unknown(monkeypatch):
    fixture = _fake_whole_lifecycle(monkeypatch)
    result = _measure_whole(fixture)
    assert result["status"] == "timing_admissible"
    assert fixture[-1][:3] == [("apply", 8), ("apply", 1), ("time", None)]
    assert result["resources"]["resident_bytes"] == 64
    assert result["resources"]["workspace_resident_bytes"] == 0
    assert "fixed_and_full_model_resources" in result["resources"]["unknown"]
    assert result["resources"]["status"] == "incomplete"


def test_whole_bad_decode_refuses_even_prefill_samples(monkeypatch):
    fixture = _fake_whole_lifecycle(monkeypatch, bad_decode=True)
    result = _measure_whole(fixture)
    assert result["status"] == "numerical_refused"
    assert not any(kind == "time" for kind, _ in fixture[-1])
    assert all(phase["measurement"] is None for phase in result["phases"].values())


@pytest.mark.parametrize("key", ["topk_ids", "topk_weights", "input"])
def test_whole_captured_routing_or_input_mutation_during_timing_refuses(monkeypatch, key):
    fixture = _fake_whole_lifecycle(monkeypatch)
    original = dense.time_apply
    def mutate(apply, **kwargs):
        result = original(apply, **kwargs)
        fixture[2]["prefill"][key][0, 0] += 1
        return result
    monkeypatch.setattr(dense, "time_apply", mutate)
    with pytest.raises(ValueError):
        _measure_whole(fixture)


def test_workspace_pointer_reallocation_refuses_even_when_portable_layout_matches(monkeypatch):
    fixture = _fake_whole_lifecycle(monkeypatch)
    monkeypatch.setattr(moe, "observe_workspace", lambda: (fixture[1]["workspace"], (999,)))
    with pytest.raises(ValueError, match="workspace"):
        _measure_whole(fixture)
    assert fixture[-1] == []


def test_whole_resource_observation_has_no_decision_timing_until_collector_stops(monkeypatch):
    from types import SimpleNamespace
    fixture = _fake_whole_lifecycle(monkeypatch)
    def observe(apply, phase, **kwargs):
        return apply(), {"fixture": "CPU_ONLY"}
    collector = SimpleNamespace(_finished=False, observe_apply=observe)
    result = _measure_whole(fixture, collector)
    assert result["status"] == "resources_observed"
    assert not any(kind == "time" for kind, _ in fixture[-1])
    result["resources"]["status"] = "complete_operator_bound"
    with pytest.raises(ValueError, match="closed"):
        moe.time_after_resource_collection(*fixture[:3], result, collector=collector, warmup_iterations=2, iterations=3)
    collector._finished = True
    moe.time_after_resource_collection(*fixture[:3], result, collector=collector, warmup_iterations=2, iterations=3)
    assert result["status"] == "timing_admissible"
    assert result["timing_scope"] == "cuda_events_after_resource_collector_stop"


def test_native_configuration_sets_have_deterministic_type_preserving_identity():
    assert moe._plain({"b", "a"}) == {"set": ["a", "b"]}
    assert moe._plain(frozenset({"a", "b"})) == {"frozenset": ["a", "b"]}
    with pytest.raises(ValueError, match="unobserved"):
        moe._plain({object()})


def _factory_configuration_fixture():
    from types import SimpleNamespace as NS
    routing, shape = _routing(), _shape()
    layer = NS(renormalize=True, scoring_func="sigmoid", routed_scaling_factor=1.0,
        apply_router_weight_on_input=False, expert_map=None, global_num_experts=32, local_num_experts=32,
        top_k=4, use_grouped_topk=True, num_expert_group=1, topk_group=1, custom_routing_function=None,
        swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None, is_fused_checkpoint_transposed=False,
        tessera_mode="resident", tessera_family="TESSERA_FP8", activation="silu", e_score_correction_bias=None,
        quant_method=NS(is_monolithic=False), moe_config=NS(num_experts=32, num_local_experts=32,
            num_logical_experts=32, experts_per_token=4, hidden_dim=256, intermediate_size=128,
            intermediate_size_per_partition=128, max_num_tokens=2048, has_bias=False, is_lora_enabled=False,
            in_dtype=torch.bfloat16, device=torch.device("cuda:0"),
            moe_parallel_config=NS(tp_size=1, ep_size=1, dp_size=1, pcp_size=1, sp_size=1)))
    return layer, shape, routing


def test_factory_observed_configuration_agrees_with_requested_scope():
    moe.verify_native_configuration(*_factory_configuration_fixture(), 2048)


@pytest.mark.parametrize("name,value", [("renormalize", False), ("top_k", True), ("use_grouped_topk", False),
    ("num_expert_group", 2), ("topk_group", 2), ("global_num_experts", 33),
    ("local_num_experts", 16), ("activation", "gelu"), ("swiglu_limit", 7.0)])
def test_factory_actual_routing_or_owner_drift_refuses(name, value):
    layer, shape, routing = _factory_configuration_fixture()
    setattr(layer, name, value)
    with pytest.raises(ValueError):
        moe.verify_native_configuration(layer, shape, routing, 2048)


@pytest.mark.parametrize("name,value", [("max_num_tokens", 512), ("has_bias", True), ("is_lora_enabled", True),
                                        ("device", "cuda:1"), ("in_dtype", torch.float16)])
def test_factory_actual_scheduler_bias_dtype_device_drift_refuses(name, value):
    layer, shape, routing = _factory_configuration_fixture()
    setattr(layer.moe_config, name, value)
    with pytest.raises(ValueError):
        moe.verify_native_configuration(layer, shape, routing, 2048)


def test_probe_subset_scope_remains_explicit_and_joins_the_panel_calibration():
    panel = _panel()
    panel["probe_scope"] = {"schema": "prismaquant.native_probe_subset.v1",
        "parent_calibration_sha256": _sha("full captured calibration"),
        "subset_calibration_sha256": panel["calibration_sha256"],
        "sample_indices": [0], "scope": "first_sequence_integration_screen"}
    assert moe.validate_panel(panel)["probe_scope"] == panel["probe_scope"]
    for patch in ({"sample_indices": [False]}, {"sample_indices": [1]},
                  {"subset_calibration_sha256": _sha("another subset")}, {"scope": "full_quality"}):
        changed = copy.deepcopy(panel)
        changed["probe_scope"].update(patch)
        with pytest.raises(ValueError, match="subset scope"):
            moe.validate_panel(changed)
