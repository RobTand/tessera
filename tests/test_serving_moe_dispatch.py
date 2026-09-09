"""The MoE gate derives from vLLM's MoE module instead of sniffing names.

``get_quant_method`` refused routed-MoE experts because returning ``None``
hands vLLM ``UnquantizedFusedMoEMethod`` -- uninitialised or BF16 expert
memory served in silence.  But the refusal keyed on a hand-maintained
frozenset (``RoutedExperts``/``FusedMoE``/``SharedFusedMoE``) beside an import
that only tried two of those names: on a vLLM build that renames or moves the
MoE class again, both halves miss and the layer falls through to ``return
None`` -- the silent fallback the branch exists to prevent.  A rename already
happened once (``RoutedExperts`` vs ``FusedMoE`` across versions).

These tests stage that second rename inside a stubbed vLLM: the stub MoE
module defines ``FutureMoE`` alongside ``RoutedExperts``, and the gate must
refuse it without anyone editing the plugin.  What the tests pin is the rule
(every layer class the MoE module defines; every name it defines; never a
silent ``None`` for a MoE-looking layer), never the roster.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import lane                                    # noqa: E402
from tessera.serving.lane import TESSERA_MODE_ENV                   # noqa: E402
from tessera.serving.scheme import TESSERA_NVFP4                    # noqa: E402


def test_shared_config_preserves_existing_import():
    from tessera.moe_execution import ResearchSelectedMoeConfig
    from tessera.serving.moe_route import ResearchSelectedMoeConfig as ExistingImport
    assert ExistingImport is ResearchSelectedMoeConfig


def _research_checkpoint(tp=1, backend="triton"):
    return {"schema": "tessera.research_selected_moe.v1",
            "max_experts_per_chunk": 3, "decode_backend": backend,
            "expected_tensor_parallel_size": tp}


def _expert_config(block):
    from tessera.serving.scheme import TESSERA_FP8
    scheme = {"family": TESSERA_FP8, "structure": "routed_moe", "grid": "E4M3",
              "body": "WINDOW", "plane": "CHANNEL", "experts": 4,
              "groups": {
                  "w13": {"rows": 128, "columns": 128, "q256": 512,
                          "wire_stride": 10000, "roles": [["gate_proj", 64], ["up_proj", 64]]},
                  "w2": {"rows": 128, "columns": 64, "q256": 512,
                         "wire_stride": 10000, "roles": [["down_proj", 128]]}}}
    return {"quant_method": "tessera", "config_groups": {
        "expert": {"targets": ["model.layers.1.mlp.experts"], "scheme": scheme}},
        "ignore": ["model.layers.2.mlp.experts"], "research_selected_moe": block}


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_checkpoint_reconstruction_selects_existing_packed_owner(monkeypatch, tp, backend):
    from tessera.serving import config as config_module, moe_route
    import vllm.model_executor.layers.fused_moe as moe

    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    block = _research_checkpoint(tp, backend)
    # The worker receives JSON, never the native control's local config subclass.
    payload = json.loads(json.dumps(_expert_config(block)))
    config = TesseraConfig.from_config(payload)
    facts, calls = [], []
    monkeypatch.setattr(config_module, "declare_compile_identity", lambda **kw: facts.append(kw))
    result = object()
    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    monkeypatch.setattr(moe_route, "build_tessera_moe_method", build)
    layer = moe.RoutedExperts()
    assert config.get_quant_method(layer, "model.layers.1.mlp.experts") is result
    selected = calls[0][1].get("research_selected")
    assert isinstance(selected, moe_route.ResearchSelectedMoeConfig), (
        "checkpoint reconstructed the ordinary FP8 materialized owner")
    assert selected.expected_tensor_parallel_size == tp
    assert selected.decode_backend == backend
    assert selected.max_experts_per_chunk == 3
    assert facts[0]["serve_mode"] == "resident"
    assert "research_selected_moe" in facts[0]
    assert config.get_quant_method(layer, "model.layers.2.mlp.experts") is None
    assert len(calls) == 1


@pytest.mark.parametrize("block", [None, [], {},
    {**_research_checkpoint(), "schema": "future"},
    {**_research_checkpoint(), "extra": 1},
    {k: v for k, v in _research_checkpoint().items() if k != "decode_backend"},
    *[{**_research_checkpoint(), "max_experts_per_chunk": value} for value in [True, 0, -1, 1.0, "3"]],
    *[{**_research_checkpoint(), "expected_tensor_parallel_size": value} for value in [True, 0, 3, 1.0, "2"]],
    *[{**_research_checkpoint(), "decode_backend": value} for value in [None, [], "auto"]],
])
def test_checkpoint_research_request_refuses_malformed_before_loading(monkeypatch, block):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="research_selected_moe"):
        TesseraConfig.from_config(_expert_config(block))


def test_checkpoint_research_request_refuses_streamed_and_no_routed_target(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "streamed")
    with pytest.raises(ValueError, match="research_selected_moe.*resident"):
        TesseraConfig.from_config(_expert_config(_research_checkpoint()))
    lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    dense = _config()
    dense["research_selected_moe"] = _research_checkpoint()
    with pytest.raises(ValueError, match="research_selected_moe.*routed"):
        TesseraConfig.from_config(dense)


def test_ordinary_checkpoint_retains_builder_and_compile_identity(monkeypatch):
    from tessera.serving import config as config_module, moe_route
    import vllm.model_executor.layers.fused_moe as moe
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    payload = _expert_config(_research_checkpoint())
    del payload["research_selected_moe"]
    config = TesseraConfig.from_config(payload)
    facts, calls = [], []
    monkeypatch.setattr(config_module, "declare_compile_identity", lambda **kw: facts.append(kw))
    monkeypatch.setattr(moe_route, "build_tessera_moe_method", lambda *a, **kw: calls.append((a, kw)))
    config.get_quant_method(moe.RoutedExperts(), "model.layers.1.mlp.experts")
    assert calls[0][1] == {}
    assert facts == [{"serve_mode": "resident"}]


def test_checkpoint_execution_identity_distinguishes_every_execution_choice(monkeypatch):
    from tessera.serving import config as config_module, moe_route
    import vllm.model_executor.layers.fused_moe as moe
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    facts = []
    monkeypatch.setattr(config_module, "declare_compile_identity", lambda **kw: facts.append(kw))
    monkeypatch.setattr(moe_route, "build_tessera_moe_method", lambda *a, **kw: None)
    base = _research_checkpoint()
    variants = [base, {**base, "expected_tensor_parallel_size": 2},
                {**base, "decode_backend": "torch"}, {**base, "max_experts_per_chunk": 5}]
    for block in [*variants, dict(reversed(list(base.items())))]:
        config = TesseraConfig.from_config(_expert_config(block))
        config.get_quant_method(moe.RoutedExperts(), "model.layers.1.mlp.experts")
    hashes = [fact["research_selected_moe"] for fact in facts]
    assert len(set(hashes[:4])) == 4
    assert hashes[0] == hashes[-1]


# Every routed stack a real LFM2.5-8B-A1B MoE checkpoint declares: layers 2..23.
_LFM_STACKS = tuple(f"model.layers.{layer}.feed_forward.experts" for layer in range(2, 24))


def _routed_scheme(family=None, **over):
    from tessera.serving.scheme import TESSERA_FP8
    if family is None or family == TESSERA_FP8:
        route = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL"}
    else:
        # A different route that is a valid Tessera scheme, so the refusal below
        # comes from require_targets and not from scheme validation.
        route = {"family": family, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT"}
    return {**route, "structure": "routed_moe", "experts": 32,
            "groups": {
                "w13": {"rows": 128, "columns": 128, "q256": 1024,
                        "wire_stride": 10000, "roles": [["gate_proj", 64], ["up_proj", 64]]},
                "w2": {"rows": 128, "columns": 64, "q256": 1024,
                       "wire_stride": 10000, "roles": [["down_proj", 128]]}}, **over}


def _multi_stack_config(block, *, off_route=None):
    """One declaration over all 22 routed stacks, optionally breaking one of them.

    The single-stack fixture above cannot separate "checks the first routed
    target" from "checks every routed target"; a real checkpoint declares 22.
    """
    from tessera.serving.scheme import TESSERA_NVFP4
    groups = {}
    for index, stack in enumerate(_LFM_STACKS):
        family = TESSERA_NVFP4 if index == off_route else None
        groups[f"experts_{index}"] = {"format": "TESSERA", "targets": [stack],
                                      "scheme": _routed_scheme(family)}
    return {"quant_method": "tessera", "config_groups": groups, "ignore": [],
            "research_selected_moe": block}


def test_every_declared_stack_selects_the_packed_owner(monkeypatch):
    from tessera.serving import config as config_module, moe_route
    import vllm.model_executor.layers.fused_moe as moe

    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    block = _research_checkpoint(tp=2, backend="triton")
    payload = json.loads(json.dumps(_multi_stack_config(block)))
    config = TesseraConfig.from_config(payload)
    facts, calls = [], []
    monkeypatch.setattr(config_module, "declare_compile_identity", lambda **kw: facts.append(kw))
    monkeypatch.setattr(moe_route, "build_tessera_moe_method",
                        lambda *a, **kw: calls.append((a, kw)) or object())
    layer = moe.RoutedExperts()
    for stack in _LFM_STACKS:
        assert config.get_quant_method(layer, stack) is not None
    assert len(calls) == len(_LFM_STACKS) == 22
    selected = {id(kwargs["research_selected"]) for _, kwargs in calls}
    assert len(selected) == 1, "each stack rebuilt its own execution object"
    only = calls[0][1]["research_selected"]
    assert only.expected_tensor_parallel_size == 2 and only.decode_backend == "triton"
    # The execution declaration is a checkpoint fact, so every stack reports one
    # identity, not one per layer.
    assert facts and len({fact["research_selected_moe"] for fact in facts}) == 1


@pytest.mark.parametrize("off_route", [0, 11, 21])
def test_one_off_route_stack_among_many_refuses_before_loading(monkeypatch, off_route):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    payload = _multi_stack_config(_research_checkpoint(), off_route=off_route)
    with pytest.raises(ValueError, match="research_selected_moe.*TESSERA_FP8/E4M3"):
        TesseraConfig.from_config(payload)


def _module(name):
    value = types.ModuleType(name)
    sys.modules[name] = value
    return value


def _moe_class(module, name, base=torch.nn.Module):
    cls = type(name, (base,), {})
    cls.__module__ = module.__name__
    return cls


def _install_vllm_stubs():
    _module("vllm")
    _module("vllm.model_executor")
    _module("vllm.model_executor.layers")
    _module("vllm.model_executor.layers.quantization")
    linear = _module("vllm.model_executor.layers.linear")
    linear.LinearBase = type("LinearBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    parameter = _module("vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kwargs):
            pass
    parameter.ModelWeightParameter = StubParameter
    parameter.BasevLLMParameter = StubParameter
    parameter.ChannelQuantScaleParameter = StubParameter
    parameter.PerTensorScaleParameter = StubParameter
    base = _module("vllm.model_executor.layers.quantization.base_config")
    base.QuantizationConfig = type("QuantizationConfig", (), {})
    base.QuantizeMethodBase = object
    embedding = _module("vllm.model_executor.layers.vocab_parallel_embedding")
    embedding.UnquantizedEmbeddingMethod = type("UEM", (), {})
    embedding.VocabParallelEmbedding = type("VPE", (), {})
    embedding.ParallelLMHead = type(
        "ParallelLMHead", (embedding.VocabParallelEmbedding,), {})
    fused = _module("vllm.model_executor.layers.fused_moe")
    fused.RoutedExperts = _moe_class(fused, "RoutedExperts")
    # The rename the hand-maintained roster never learned: a second layer
    # class the MoE module defines, which the old frozenset did not contain.
    fused.FutureMoE = _moe_class(fused, "FutureMoE")
    # Not a layer: the scan must not treat every type in the module as one.
    fused.FusedMoEConfig = _moe_class(fused, "FusedMoEConfig", base=object)
    distributed = _module("vllm.distributed")
    distributed.get_tensor_model_parallel_world_size = lambda: 1


_ISOLATED = ("tessera.serving.config",)


def _is_isolated(name: str) -> bool:
    return name == "vllm" or name.startswith("vllm.") or name in _ISOLATED


@pytest.fixture(scope="module", autouse=True)
def runtime_modules():
    import tessera.serving as package

    before = {name: mod for name, mod in sys.modules.items() if _is_isolated(name)}
    missing = object()
    package_attr = getattr(package, "config", missing)
    for name in list(sys.modules):
        if _is_isolated(name) and (name in _ISOLATED
                                   or getattr(sys.modules.get(name), "__file__", None) is None):
            sys.modules.pop(name, None)
    vars(package).pop("config", None)
    _install_vllm_stubs()
    from tessera.serving.config import TesseraConfig
    globals()["TesseraConfig"] = TesseraConfig
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _is_isolated(name) and (name in _ISOLATED
                                       or getattr(sys.modules.get(name), "__file__", None) is None):
                sys.modules.pop(name, None)
        vars(package).pop("config", None)
        sys.modules.update(before)
        if package_attr is not missing:
            setattr(package, "config", package_attr)


@pytest.fixture(autouse=True)
def _fresh_mode(monkeypatch):
    lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    lane.reset_for_tests()


TARGET = "model.layers.0.self_attn.qkv_proj"


def _scheme(**over):
    return {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT", "q256": 896,
            "rows": 2048, "columns": 1024, "wire_bytes": 1048576,
            "roles": [["q_proj", 1024], ["k_proj", 512], ["v_proj", 512]], **over}


def _config(targets=(TARGET,), ignore=()):
    return {"quant_method": "tessera", "format": "tessera",
            "config_groups": {"tessera": {"format": "TESSERA", "targets": list(targets),
                                          "scheme": _scheme()}},
            "ignore": list(ignore)}


def _resolved(monkeypatch, **kw):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    return TesseraConfig.from_config(_config(**kw))


def _moe_layer_classes():
    """The rule, from the code that owns it: every layer the MoE module defines."""
    import vllm.model_executor.layers.fused_moe as moe

    return tuple(obj for name in dir(moe)
                 if isinstance(obj := getattr(moe, name, None), type)
                 and issubclass(obj, torch.nn.Module)
                 and obj.__module__ == moe.__name__)


def test_every_layer_class_the_moe_module_defines_refuses(monkeypatch):
    """A rename the plugin never heard of is still an MoE layer.

    The stub MoE module defines ``FutureMoE`` beside ``RoutedExperts``; the
    old frozenset contained no such name, so it fell through to ``None`` --
    vLLM's silent unquantized fallback.  Derived from the module, not listed.
    """
    config = _resolved(monkeypatch)
    for cls in _moe_layer_classes():
        assert cls.__name__ not in ("RoutedExperts",) or True
        with pytest.raises(ValueError, match="routed-MoE"):
            config.get_quant_method(object.__new__(cls),
                                    "model.layers.0.mlp.experts")


def test_a_class_named_like_a_module_moe_layer_refuses(monkeypatch):
    """A subclass in another module, carrying a name the MoE module defines.

    The name set is generated from the imported classes rather than
    hand-maintained beside them, so ``FutureMoE`` refuses wherever it is
    defined -- the old set knew only the three names typed next to it.
    """
    import vllm.model_executor.layers.fused_moe as moe

    assert "FutureMoE" in {c.__name__ for c in _moe_layer_classes()}, \
        "the stub rename is gone; this test no longer stages a rename"
    assert not hasattr(moe, "NotAMoeModuleClass")
    alien = type("FutureMoE", (torch.nn.Module,), {})
    assert alien.__module__ != moe.__name__
    config = _resolved(monkeypatch)
    with pytest.raises(ValueError, match="routed-MoE"):
        config.get_quant_method(object.__new__(alien),
                                "model.layers.0.mlp.experts")


@pytest.mark.parametrize("name", ["FusedMoE2", "ShardedExpertsV2"])
def test_a_renamed_moe_layer_never_falls_through_to_none(monkeypatch, name):
    """Fail closed: a MoE-looking layer nobody imports refuses, not ``None``.

    Neither the module scan nor the generated name set can know ``name`` --
    that is exactly the rename-again future this pins.  ``None`` here would
    serve uninitialised or BF16 expert memory; the refusal names the expert
    route that is designed and not built.
    """
    import vllm.model_executor.layers.fused_moe as moe

    assert name not in {c.__name__ for c in _moe_layer_classes()}
    assert name not in {n for n in dir(moe)}
    cls = type(name, (torch.nn.Module,), {})
    config = _resolved(monkeypatch)
    with pytest.raises(ValueError, match="routed-MoE"):
        assert config.get_quant_method(object.__new__(cls),
                                       "model.layers.0.mlp.experts") is None


def test_a_benign_non_linear_layer_still_takes_vllms_own_method(monkeypatch):
    """The backstop is narrow: the LM head is no MoE layer and stays ``None``."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    config = _resolved(monkeypatch)
    assert config.get_quant_method(object.__new__(ParallelLMHead), "lm_head") is None


def test_a_non_module_value_still_takes_vllms_own_method(monkeypatch):
    """The backstop fires on layers, not on junk: a non-module is vLLM's
    default ``None``, refused nowhere by this plugin."""
    config = _resolved(monkeypatch)
    assert config.get_quant_method(object(), "some.attn.layer") is None


def test_an_ignored_moe_looking_layer_is_declared_bf16(monkeypatch):
    """An explicit ``ignore`` still wins: the checkpoint DECLARED these
    experts BF16, so vLLM's own unquantized MoE method is the answer and
    saying so is not a silent fallback -- even for a name nobody imports."""
    cls = type("FusedMoE2", (torch.nn.Module,), {})
    config = _resolved(monkeypatch, ignore=("model.layers.0.mlp.experts",))
    assert config.get_quant_method(
        object.__new__(cls), "model.layers.0.mlp.experts") is None


def _bound_checkpoint(tmp_path, quantization):
    import hashlib
    path = tmp_path / 'checkpoint-config.json'
    raw = (json.dumps({'quantization_config': quantization}, sort_keys=True) + '\n').encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize('tp', [1, 2])
def test_native_checkpoint_control_uses_registered_ordinary_config(monkeypatch, tmp_path, tp):
    from experiments.glm_packed_moe_control import checkpoint_quant_config
    import vllm.model_executor.layers.quantization as registry
    from tessera.serving import config as config_module, moe_route
    import vllm.model_executor.layers.fused_moe as moe
    monkeypatch.setenv(TESSERA_MODE_ENV, 'resident')
    registered = {}
    monkeypatch.setattr(registry, 'register_quantization_config',
        lambda name: lambda cls: registered.setdefault(name, cls), raising=False)
    monkeypatch.setattr(registry, 'get_quantization_config', lambda name: registered[name], raising=False)
    expected = _expert_config(_research_checkpoint(tp))
    binding = _bound_checkpoint(tmp_path, expected)
    quant, evidence = checkpoint_quant_config(binding, expected)
    assert type(quant) is TesseraConfig
    assert registered == {'tessera': TesseraConfig}
    assert evidence['checkpoint_config'] == binding
    assert evidence['research_selected_moe'] == expected['research_selected_moe']
    assert evidence['config_class'] == 'tessera.serving.config.TesseraConfig'
    calls = []
    monkeypatch.setattr(config_module, 'declare_compile_identity', lambda **kw: None)
    monkeypatch.setattr(moe_route, 'build_tessera_moe_method', lambda *a, **kw: calls.append(kw))
    quant.get_quant_method(moe.RoutedExperts(), 'model.layers.1.mlp.experts')
    assert calls[0]['research_selected'].as_checkpoint() == expected['research_selected_moe']


@pytest.mark.parametrize('mutation', ['hash', 'missing', 'null', 'tp', 'backend', 'chunk', 'target', 'scheme'])
def test_native_checkpoint_control_refuses_before_runtime_lookup(monkeypatch, tmp_path, mutation):
    import copy
    from experiments.glm_packed_moe_control import checkpoint_quant_config
    import vllm.model_executor.layers.quantization as registry
    expected = _expert_config(_research_checkpoint())
    altered = copy.deepcopy(expected)
    if mutation == 'missing':
        del altered['research_selected_moe']
    elif mutation == 'null':
        altered['research_selected_moe'] = None
    elif mutation in ('tp', 'backend', 'chunk'):
        field, value = {'tp': ('expected_tensor_parallel_size', 2),
            'backend': ('decode_backend', 'torch'), 'chunk': ('max_experts_per_chunk', 4)}[mutation]
        altered['research_selected_moe'][field] = value
    elif mutation == 'target':
        altered['config_groups']['expert']['targets'] = ['wrong.experts']
    elif mutation == 'scheme':
        altered['config_groups']['expert']['scheme']['groups']['w2']['wire_stride'] += 1
    binding = _bound_checkpoint(tmp_path, altered)
    if mutation == 'hash':
        binding['sha256'] = '0' * 64
    def forbidden(*a, **kw):
        pytest.fail('invalid checkpoint reached runtime registration')
    monkeypatch.setattr(registry, 'register_quantization_config', forbidden, raising=False)
    with pytest.raises(ValueError, match='checkpoint'):
        checkpoint_quant_config(binding, expected)
