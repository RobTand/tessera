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

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import lane                                    # noqa: E402
from tessera.serving.lane import TESSERA_MODE_ENV                   # noqa: E402
from tessera.serving.scheme import TESSERA_NVFP4                    # noqa: E402


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
