"""CPU byte intake and loader protocol for the native fused W4A4 owner.

Only vLLM's abstract Python protocol is stubbed. Serialized data, metadata
validation, TP cuts, materializing reference packing and e2m1 resident axes
are real. CPU tests do not execute or impersonate a CUDA serving forward;
the integrated serving-owner harness covers native eager and graph numerics.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from tessera import compact_prep
from tessera.compact_prep import parse_compact_wire
from tessera.errors import GrammarError
from tessera.fused import pack_fused
from tessera.native_window_moe import WindowUnitAxis
from tessera.serving import nvfp4_moe_route as route
from tessera.serving.scheme import TESSERA_NVFP4, validate_tessera_moe_scheme

from window_lut_reference import prepare_reference, window_blob

HIDDEN = INTER = 512
EXPERTS = 2
SHARDS = ("w1", "w3", "w2")
PARTS = ("gate_proj", "up_proj", "down_proj")


def _stack(q256=896):
    wires, units = {}, {}
    for expert in range(EXPERTS):
        for index, (shard, part) in enumerate(zip(SHARDS, PARTS)):
            blob = window_blob(q256, seed=expert * 7 + index, name=part,
                               global_scale=float(2 ** (expert * 3 + index)))
            wires[expert, shard] = pack_fused([(part, 512, blob)])
            units[expert, part] = prepare_reference(parse_compact_wire(blob, device="cpu"))
    declared = {"family": TESSERA_NVFP4, "structure": "routed_moe", "grid": "E2M1x2",
                "body": "WINDOW", "plane": "LUT", "experts": EXPERTS,
                "groups": {
                    "w13": {"rows": 2 * INTER, "columns": HIDDEN, "q256": q256,
                            "wire_stride": max(len(wires[e, s]) for e in range(EXPERTS)
                                               for s in SHARDS[:2]),
                            "roles": [["gate_proj", INTER], ["up_proj", INTER]]},
                    "w2": {"rows": HIDDEN, "columns": INTER, "q256": q256,
                           "wire_stride": max(len(wires[e, "w2"]) for e in range(EXPERTS)),
                           "roles": [["down_proj", HIDDEN]]}}}
    return wires, declared, units


@pytest.fixture(scope="module")
def stack():
    return _stack()


@pytest.fixture
def runtime(monkeypatch):
    """The abstract runtime interface only; never a quantized arithmetic spy."""
    names = ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe",
             "vllm.model_executor.layers.fused_moe.fused_moe_method_base",
             "vllm.model_executor.utils")
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    class Base:
        def __init__(self, moe):
            self.moe = moe

    modules[names[-2]].FusedMoEMethodBase = Base
    modules[names[-1]].set_weight_attrs = lambda parameter, attrs: [
        setattr(parameter, name, value) for name, value in attrs.items()]
    from tessera.serving import backend
    monkeypatch.setattr(backend, "require_platform_backs", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(compact_prep, "prepare_window_lut_compact", prepare_reference)


def _layer(tp_size=1, tp_rank=0):
    layer = torch.nn.Module()
    layer.moe_config = types.SimpleNamespace(
        is_act_and_mul=True, experts_per_token=2,
        moe_parallel_config=types.SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, ep_size=1, use_ep=False, enable_eplb=False))
    layer.activation = "silu"
    layer.expert_map = None
    layer.global_num_experts = EXPERTS
    layer.apply_router_weight_on_input = False
    layer.swiglu_limit = 10.0
    return layer


def _build(declared, layer, mode="resident"):
    return route.build_tessera_nvfp4_moe_method(declared, "model.mlp.experts", mode, layer)


def _load(method, layer, wires, *, omit=None):
    method.create_weights(layer, EXPERTS, HIDDEN, INTER // method._tp_size, torch.bfloat16)
    for expert in reversed(range(EXPERTS)):
        for shard in ("w3", "w2", "w1"):
            if (expert, shard) == omit:
                continue
            parameter = layer.w2_wire if shard == "w2" else layer.w13_wire
            parameter.weight_loader(parameter, torch.frombuffer(bytearray(wires[expert, shard]),
                                                                 dtype=torch.uint8),
                                    "wire", shard, expert)
    return method.intake_axes()


@pytest.mark.parametrize("q256", [128 * rate for rate in range(1, 9)] + [448])
@pytest.mark.parametrize("rank", [0, 1])
def test_real_intake_keeps_rank_local_bytes_and_each_projection_global(q256, rank, monkeypatch):
    wires, raw_scheme, _ = _stack(q256)
    declared = validate_tessera_moe_scheme(raw_scheme, "real-intake")
    intake = route._ExpertIntake(declared, "real-intake", rank, 2)
    monkeypatch.setattr(compact_prep, "prepare_window_lut_compact", prepare_reference)
    axes = {"w13": WindowUnitAxis(EXPERTS, PARTS[:2], family="e2m1"),
            "w2": WindowUnitAxis(EXPERTS, PARTS[2:], family="e2m1")}
    for expert in reversed(range(EXPERTS)):
        for shard, part in zip(SHARDS, PARTS):
            group, index = ("w2", 0) if shard == "w2" else ("w13", SHARDS.index(shard))
            intake.take(group, index, expert, wires[expert, shard], "cpu", axes=axes)
    bundles = route._window_bundles(axes, EXPERTS)
    for part, bundle in zip(PARTS, (bundles.gate, bundles.up, bundles.down)):
        assert bundle.family == "e2m1"
        for expert in range(EXPERTS):
            assert float(bundle.global_all[expert]) == float(2 ** (expert * 3 + PARTS.index(part)))
            assert bundle.scale_lut_all[expert].numel() == 16
        assert bundle.rows == (512 if part == "down_proj" else 256)
        assert bundle.cols == (256 if part == "down_proj" else 512)
        if part != "down_proj" and rank == 1:
            assert bool(bundle.has_init.all()) and bool(bundle.init_all.any())
    assert not torch.equal(bundles.gate.global_all, bundles.up.global_all)
    with pytest.raises(GrammarError, match="gs13 and gs2"):
        bundles.adapter()


def test_axes_finish_keeps_allocations_and_drops_temporary_unit_storage(stack, runtime):
    wires, scheme, units = stack
    layer = _layer()
    method = _build(scheme, layer)
    axes = _load(method, layer, wires)
    axis_words = axes["w13"]._slots["gate_proj"]["words"]
    bundles = route._window_bundles(axes, EXPERTS)
    assert bundles.gate.words_all is axis_words
    for expert in range(EXPERTS):
        original = units[expert, "gate_proj"]
        assert torch.equal(bundles.gate.words_all[expert], original.rep.words)
        assert bundles.gate.words_all.untyped_storage().data_ptr() != original.rep.words.untyped_storage().data_ptr()
        assert torch.equal(bundles.gate.scale_plane_all[expert], original.scale_plane)
    assert axes["w13"].resident_bytes() == 0


def test_static_input_global_preserves_existing_checkpoint_reduction():
    values = torch.tensor([[3.25, 9.0], [7.0, 2.75]], dtype=torch.float32)
    expected = (1.0 / (1.0 / values).max()).reshape(())
    assert torch.equal(route._static_input_global(values, "cpu"), expected)
    assert torch.equal(route._static_input_global(values[:, 0], "cpu"),
                       (1.0 / (1.0 / values[:, 0]).max()).reshape(()))


def test_native_method_remains_modular_and_shared_experts_stay_with_runner(stack, runtime):
    _wires, scheme, _ = stack
    layer = _layer()
    method = _build(scheme, layer)
    assert layer.prefix == "model.mlp.experts"
    assert not method.is_monolithic and method.moe_kernel is None
    assert method.topk_indices_dtype is None
    assert not method.mk_can_overlap_shared_experts and not method.supports_eplb
    assert not hasattr(method, "experts_cls")
    assert "apply_monolithic" not in type(method).__dict__


@pytest.mark.parametrize("kind", ["streamed", "ep", "nongated", "rank"])
def test_constructor_refuses_unsupported_runtime_protocol(stack, runtime, kind):
    _wires, scheme, _ = stack
    layer = _layer()
    mode = "resident"
    if kind == "streamed":
        mode = kind
    elif kind == "ep":
        layer.moe_config.moe_parallel_config.use_ep = True
    elif kind == "nongated":
        layer.moe_config.is_act_and_mul = False
    else:
        layer.moe_config.moe_parallel_config.tp_rank = 1
    with pytest.raises(ValueError):
        _build(scheme, layer, mode)


def test_wire_and_scale_loaders_refuse_real_bad_input_and_duplicates(stack, runtime):
    wires, scheme, _ = stack
    layer = _layer()
    method = _build(scheme, layer)
    method.create_weights(layer, EXPERTS, HIDDEN, INTER, torch.bfloat16)
    parameter = layer.w13_wire
    good = torch.frombuffer(bytearray(wires[0, "w1"]), dtype=torch.uint8)
    for wire, shard, expert, reason in ((good, "w9", 0, "shard_id"),
                                      (good, "w1", EXPERTS, "outside"),
                                      (good.float(), "w1", 0, "uint8"),
                                      (good[:0], "w1", 0, "wire_stride")):
        with pytest.raises(ValueError, match=reason):
            parameter.weight_loader(parameter, wire, "wire", shard, expert)
    parameter.weight_loader(parameter, good, "wire", "w1", 0)
    with pytest.raises(ValueError, match="already loaded"):
        parameter.weight_loader(parameter, good, "wire", "w1", 0)
    scale = layer.w13_input_global_scale
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            scale.weight_loader(scale, torch.tensor([bad]), "input_global_scale", "w1", 0)
    scale.weight_loader(scale, torch.tensor([4.0]), "input_global_scale", "w1", 0)
    with pytest.raises(ValueError, match="already loaded"):
        scale.weight_loader(scale, torch.tensor([4.0]), "input_global_scale", "w1", 0)
    with pytest.raises(ValueError, match="stock tensor"):
        layer.w13_weight.weight_loader(layer.w13_weight, torch.ones(1), "weight", "w1", 0)


@pytest.mark.parametrize("missing", ["wire", "scale"])
def test_finalization_refuses_incomplete_actual_intake(stack, runtime, missing):
    wires, scheme, _ = stack
    layer = _layer()
    method = _build(scheme, layer)
    _load(method, layer, wires, omit=(1, "w3") if missing == "wire" else None)
    with pytest.raises((ValueError, GrammarError), match="wire length|input_global_scale"):
        method.process_weights_after_loading(layer)


def test_cpu_is_not_misrepresented_as_native_serving(stack, runtime):
    wires, scheme, _ = stack
    layer = _layer()
    method = _build(scheme, layer)
    _load(method, layer, wires)
    for expert in range(EXPERTS):
        for shard in SHARDS:
            parameter = layer.w2_input_global_scale if shard == "w2" else layer.w13_input_global_scale
            parameter.weight_loader(parameter, torch.tensor([2.0]), "input_global_scale", shard, expert)
    with pytest.raises(GrammarError, match="CUDA"):
        method.process_weights_after_loading(layer)
    assert not hasattr(layer, "tessera_routed_fused")


@pytest.mark.parametrize("group", ["w13", "w2"])
def test_expert_rate_matrix_refuses_different_strides_before_intake(stack, group):
    import copy

    _wires, scheme, _units = stack
    bad = copy.deepcopy(scheme)
    bad["groups"][group]["q256"] = ([[896, 896], [1024, 1024]] if group == "w13"
                                        else [[896], [1024]])
    with pytest.raises(ValueError, match="experts disagree on their run tables"):
        validate_tessera_moe_scheme(bad, "mixed-stride")


def test_uniform_expert_rate_matrix_is_not_a_new_refusal(stack):
    import copy

    _wires, scheme, _units = stack
    uniform = copy.deepcopy(scheme)
    uniform["groups"]["w13"]["q256"] = [[896, 896], [896, 896]]
    uniform["groups"]["w2"]["q256"] = [[896], [896]]
    declared = validate_tessera_moe_scheme(uniform, "uniform-matrix")
    assert declared["groups"]["w13"]["role_q256"] == [896, 896]


def test_gate_and_up_refuse_different_tile_strides(stack):
    import copy

    _wires, scheme, _units = stack
    different = copy.deepcopy(scheme)
    different["groups"]["w13"]["q256"] = [896, 1024]
    with pytest.raises(ValueError, match="one tile stride for both"):
        validate_tessera_moe_scheme(different, "gate-up-stride")


def test_real_equal_rate_counts_keep_different_expert_permutations():
    from tessera.routed_fused import _run_stack_reason

    axes = {"w13": WindowUnitAxis(EXPERTS, PARTS[:2], family="e2m1"),
            "w2": WindowUnitAxis(EXPERTS, PARTS[2:], family="e2m1")}
    for expert in range(EXPERTS):
        for part in PARTS:
            blob = window_blob(448, reverse_rates=bool(expert), name=part)
            unit = prepare_reference(parse_compact_wire(blob, device="cpu"))
            axes["w2" if part == "down_proj" else "w13"].put(part, expert, unit)
    bundles = route._window_bundles(axes, EXPERTS)
    assert not torch.equal(bundles.gate.perm_all[0], bundles.gate.perm_all[1])
    for part in ("gate", "up", "down"):
        assert _run_stack_reason(part, getattr(bundles, part), EXPERTS) is None



def test_plan_rate_guard_keeps_down_rate_independent():
    from tessera.serving.scheme import e2m1_expert_rate_reason

    assert e2m1_expert_rate_reason([[896, 896, 512], [896, 896, 512]]) is None
    assert "one tile stride" in e2m1_expert_rate_reason([[896, 768, 512], [896, 768, 512]])
    assert "one schedule per stack" in e2m1_expert_rate_reason([[896, 896, 512], [896, 896, 896]])

