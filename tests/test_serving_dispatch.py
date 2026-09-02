"""Config-level dispatch: a Tessera checkpoint reaches its routes, or refuses.

``TesseraConfig`` is what vLLM builds when a checkpoint declares
``quantization_config.quant_method: "tessera"``.  THERE IS NO ENABLE FLAG any
more: Gridbook's ``GRIDBOOK_TESSERA`` selected a lane inside another runtime,
and here the bytes select the plugin.  The one operator knob is the residency,
``TESSERA_SERVE_MODE``, and an unset or misspelt one refuses by name at config
parse -- which is what the ported "unset flag refuses by name" tests became.

The other half of this file is the refusals that must NOT degrade: an
undeclared Linear, a routed-MoE experts layer (where a ``None`` would hand vLLM
``UnquantizedFusedMoEMethod`` and serve BF16 expert memory), and TP > 1.

vLLM is STUBBED here exactly as the Gridbook original stubs it; loading owes a
container run.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import lane                                    # noqa: E402
from tessera.serving.lane import TESSERA_MODE_ENV                   # noqa: E402
from tessera.serving.scheme import TESSERA_FP8, TESSERA_NVFP4       # noqa: E402


def _install_vllm_stubs():
    def module(name):
        value = types.ModuleType(name)
        sys.modules[name] = value
        return value
    module("vllm")
    module("vllm.model_executor")
    module("vllm.model_executor.layers")
    module("vllm.model_executor.layers.quantization")
    linear = module("vllm.model_executor.layers.linear")
    linear.LinearBase = type("LinearBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.register_weight_loader_v2_supported_method = lambda cls: cls
    parameter = module("vllm.model_executor.parameter")

    class StubParameter(torch.nn.Parameter):
        def __new__(cls, data, **kwargs):
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, data, **kwargs):
            pass
    parameter.ModelWeightParameter = StubParameter
    parameter.BasevLLMParameter = StubParameter
    parameter.ChannelQuantScaleParameter = StubParameter
    parameter.PerTensorScaleParameter = StubParameter
    base = module("vllm.model_executor.layers.quantization.base_config")
    base.QuantizationConfig = type("QuantizationConfig", (), {})
    base.QuantizeMethodBase = object
    embedding = module("vllm.model_executor.layers.vocab_parallel_embedding")
    embedding.UnquantizedEmbeddingMethod = type("UEM", (), {})
    embedding.VocabParallelEmbedding = type("VPE", (), {})
    embedding.ParallelLMHead = type("ParallelLMHead", (embedding.VocabParallelEmbedding,), {})
    fused = module("vllm.model_executor.layers.fused_moe")
    fused.RoutedExperts = type("RoutedExperts", (), {})
    # A world size of ONE by default: without this module ``_require_tp1``'s
    # import fails, the broad ``except`` swallows it, and the TP gate would be
    # untested rather than passing.
    distributed = module("vllm.distributed")
    distributed.get_tensor_model_parallel_world_size = lambda: 1


#: Modules whose binding this file replaces and must restore.  ``ops`` and the
#: route modules are deliberately NOT here: ``ops`` registers two
#: ``torch.library.custom_op``s at import and a second import raises.
_ISOLATED = ("tessera.serving.config",)


def _is_isolated(name: str) -> bool:
    return name == "vllm" or name.startswith("vllm.") or name in _ISOLATED


@pytest.fixture(scope="module", autouse=True)
def runtime_modules():
    """Give this file a private vLLM import graph, and restore the real one.

    Pytest imports every selected test file before running module fixtures, so
    stubs installed here would otherwise leak into later files and make results
    depend on the file order.
    """
    import tessera.serving as package

    before = {name: mod for name, mod in sys.modules.items() if _is_isolated(name)}
    missing = object()
    package_attr = getattr(package, "config", missing)
    for name in list(sys.modules):
        # Only a stub (no ``__file__``) may be REMOVED: importing real vLLM
        # registers opaque types with Torch that no sys.modules bookkeeping
        # can undo, so a second import of it dies.
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
    """A process that has never read the residency, per test.

    The mode is LATCHED to the first value the process observes, which is right
    in a serve and wrong across a session.
    """
    lane.reset_for_tests()
    monkeypatch.delenv(TESSERA_MODE_ENV, raising=False)
    yield
    lane.reset_for_tests()


TARGET = "model.layers.0.self_attn.qkv_proj"
FP8_TARGET = "model.layers.0.mlp.down_proj"
IGNORED = "lm_head"


def _scheme(**over):
    return {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT", "q256": 896,
            "rows": 2048, "columns": 1024, "wire_bytes": 1048576,
            "roles": [["q_proj", 1024], ["k_proj", 512], ["v_proj", 512]], **over}


def _fp8_scheme(**over):
    return {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
            "q256": 1024, "rows": 1024, "columns": 2048, "wire_bytes": 262144,
            "roles": [["down_proj", 1024]], **over}


def _config(scheme=None, targets=(TARGET,), ignore=(), extra_groups=None, quant_method="tessera"):
    groups = {"tessera": {"format": "TESSERA", "targets": list(targets),
                          "scheme": _scheme() if scheme is None else scheme}}
    if extra_groups:
        groups.update(extra_groups)
    return {"quant_method": quant_method, "format": "tessera",
            "config_groups": groups, "ignore": list(ignore)}


def _resolved(cfg=None):
    return TesseraConfig.from_config(cfg or _config())


def _layer():
    from vllm.model_executor.layers.linear import LinearBase
    return object.__new__(LinearBase)


def _lm_head():
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    return object.__new__(ParallelLMHead)


# --- dispatch ----------------------------------------------------------------

def test_a_declared_target_dispatches_to_its_route(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(
        extra_groups={"tessera_fp8": {"format": "TESSERA", "targets": [FP8_TARGET],
                                      "scheme": _fp8_scheme()}}))
    assert type(config.get_quant_method(_layer(), TARGET)).__name__ == "TesseraNvfp4LinearMethod"
    assert type(config.get_quant_method(_layer(), FP8_TARGET)).__name__ == "TesseraFp8LinearMethod"
    assert set(config.target_scheme) == {TARGET, FP8_TARGET}


def test_an_ignored_linear_is_served_unquantized(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(ignore=(IGNORED,)))
    method = config.get_quant_method(_layer(), IGNORED)
    assert type(method).__name__ == "UnquantizedLinearMethod"


def test_a_linear_that_is_neither_declared_nor_ignored_refuses_by_name(monkeypatch):
    """No silent BF16: one mistyped target name must be a refusal, not an
    artifact that merely looks disappointing."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    other = "model.layers.7.mlp.gate_up_proj"
    with pytest.raises(ValueError, match=other):
        _resolved().get_quant_method(_layer(), other)


def test_a_non_linear_non_moe_layer_takes_vllms_own_method(monkeypatch):
    """``ParallelLMHead`` and friends: ``None`` is right here, and only here."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    assert _resolved().get_quant_method(_lm_head(), "lm_head") is None


@pytest.mark.parametrize("build", ["real", "named"])
def test_a_routed_moe_layer_refuses_and_never_returns_none(monkeypatch, build):
    """vLLM hands an MoE layer ``UnquantizedFusedMoEMethod`` when a config
    returns ``None``, so ``None`` here would serve uninitialised or BF16 expert
    memory rather than say no.  Both the real class and a class merely NAMED
    ``RoutedExperts`` (a version rename, a subclass in another module) must
    refuse."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    if build == "real":
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        layer = object.__new__(RoutedExperts)
    else:
        layer = object.__new__(type("RoutedExperts", (), {}))
    config = _resolved()
    with pytest.raises(ValueError) as excinfo:
        config.get_quant_method(layer, "model.layers.0.mlp.experts")
    message = str(excinfo.value)
    assert "flashinfer_b12x" in message and "compressed-tensors" in message


def test_tensor_parallelism_above_one_refuses(monkeypatch):
    """A unit is one blob per module against a shared rate schedule, so a shard
    is not a byte range."""
    import vllm.distributed as distributed

    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved()
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 2)
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        config.get_quant_method(_layer(), TARGET)


def test_the_config_declares_its_resolved_mode_once(monkeypatch):
    """The residency is folded into vLLM's compile-cache key from
    ``get_quant_method`` -- after the config is current, before any hash --
    and exactly once per config."""
    from tessera.serving import config as config_module

    monkeypatch.setenv(TESSERA_MODE_ENV, "streamed")
    seen = []
    monkeypatch.setattr(config_module, "declare_compile_identity",
                        lambda **facts: seen.append(facts))
    config = _resolved()
    config.get_quant_method(_layer(), TARGET)
    assert seen == [{"serve_mode": "streamed"}]
    config.get_quant_method(_layer(), TARGET)
    assert seen == [{"serve_mode": "streamed"}]


# --- refusals at config parse ------------------------------------------------

def test_the_mode_is_declared_not_defaulted_and_there_is_no_enable_flag(monkeypatch):
    """The ported "unset flag refuses by name", rewritten for the new contract.

    Gridbook needed ``GRIDBOOK_TESSERA=1`` to reach the lane at all.  Here the
    checkpoint selects the plugin and the only refusal left is the residency,
    which the plugin will not choose for the operator because it changes the
    footprint the artifact occupies.
    """
    assert not hasattr(lane, "TESSERA_FLAG")
    with pytest.raises(ValueError, match=TESSERA_MODE_ENV):
        _resolved()
    lane.reset_for_tests()
    monkeypatch.setenv(TESSERA_MODE_ENV, "residnet")       # a misspelling, not a mode
    with pytest.raises(ValueError, match=TESSERA_MODE_ENV):
        _resolved()


def test_from_config_refuses_another_runtimes_checkpoint(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="quant_method"):
        TesseraConfig.from_config(_config(quant_method="compressed-tensors"))
    with pytest.raises(ValueError, match="config_groups"):
        TesseraConfig.from_config({"quant_method": "tessera"})


def test_a_group_that_is_not_a_tessera_scheme_is_refused(monkeypatch):
    """Not skipped: a group this plugin cannot read is a checkpoint written for
    another runtime, and skipping it would serve that module BF16."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    foreign = {"cb": {"format": "nvfp4-pack-quantized", "targets": ["model.layers.1.mlp.up_proj"],
                      "scheme": {"num_bits": 4, "type": "float", "group_size": 16}}}
    with pytest.raises(ValueError, match="not a Tessera scheme"):
        _resolved(_config(extra_groups=foreign))


def test_a_malformed_scheme_is_refused_at_config_parse(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="stack to"):
        _resolved(_config(_scheme(rows=4096)))
    with pytest.raises(ValueError, match="E2M1-based"):
        _resolved(_config(_scheme(grid="E4M3")))


def test_a_routed_moe_structure_is_refused_by_name_at_parse(monkeypatch):
    """The field exists so that adding the expert route is a value, not a
    schema change -- and until it is built, the value is refused here rather
    than mis-served through the dense method."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="routed_moe"):
        _resolved(_config(_scheme(structure="routed_moe")))


def test_a_target_may_not_be_both_declared_and_ignored(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    with pytest.raises(ValueError, match="both declared and ignored"):
        _resolved(_config(ignore=(TARGET,)))


# --- FAMILY = ROUTE is a table, not an if-chain -----------------------------
# A third family (TESSERA_BF16: the same WINDOW body and CHANNEL plane, its
# 2^L alphabet snapped to bf16, decoded to a bf16 tile for the stock GEMM) must
# cost one route module plus one ROUTES entry plus contract rows.  These tests
# hold that boundary: they register a fake family at runtime and assert the
# dispatcher and the family list pick it up with no edit anywhere else.

def test_the_family_list_is_derived_from_the_route_table():
    from tessera.serving import scheme
    assert scheme.TESSERA_FAMILIES == tuple(scheme.ROUTES), \
        "a hand-written family tuple is a second place to remember"


def test_every_route_names_the_builder_that_serves_it():
    from importlib import import_module
    from tessera.serving import scheme
    for family, route in scheme.ROUTES.items():
        module_name, builder_name = route["builder"]
        assert callable(getattr(import_module(module_name), builder_name)), family


def test_a_new_family_needs_only_a_routes_entry(monkeypatch):
    """Register a family at runtime; the dispatcher must find it."""
    import sys
    import types
    from tessera.serving import lane, scheme

    seen = {}

    def _build(scheme_dict, prefix, mode):
        seen.update(prefix=prefix, mode=mode, family=scheme_dict["family"])
        return "the-fake-method"

    fake = types.ModuleType("tessera_fake_route")
    fake.build_fake_method = _build
    monkeypatch.setitem(sys.modules, "tessera_fake_route", fake)
    monkeypatch.setitem(scheme.ROUTES, "TESSERA_FAKE", {
        "grids": ("E4M3",), "plane": "CHANNEL", "short": "FAKE",
        "tile": "a fake tile", "columns_multiple": 16,
        "activation_contract": "fake_contract",
        "builder": ("tessera_fake_route", "build_fake_method"),
    })
    got = lane.build_tessera_method({"family": "TESSERA_FAKE"}, "m", mode="resident")
    assert got == "the-fake-method"
    assert seen == {"prefix": "m", "mode": "resident", "family": "TESSERA_FAKE"}


def test_an_unknown_family_is_still_refused_by_name():
    import pytest
    from tessera.serving import lane
    with pytest.raises(ValueError, match="family must be one of"):
        lane.build_tessera_method({"family": "TESSERA_NOPE"}, "m", mode="resident")


# --------------------------------------------------------------------------
# A checkpoint's module names are not vLLM's module names.
#
# vLLM hands every quantization config the model's ``hf_to_vllm_mapper`` and
# expects the config to translate its own target lists
# (``model_loader/utils.py:277-279``; ``models/interfaces.py:1160`` for a
# ``SupportsQuant`` class).  The base method is a no-op
# (``base_config.py:229-241``), so inheriting it meant every target stayed in
# checkpoint space.  On ``Glm5NextForConditionalGeneration`` -- mapper
# ``{"model.language_model." -> "language_model.model.", "model.visual." ->
# "visual.", "lm_head." -> "language_model.lm_head."}`` -- not one target would
# have matched, and every Linear would have been refused at load.
#
# It never showed because every Tessera artifact served so far is Qwen3-0.6B,
# whose class declares no mapper at all.
# --------------------------------------------------------------------------

class _Mapper:
    """The one method of ``WeightsMapper`` this contract uses."""

    def __init__(self, prefixes, drop=()):
        self.prefixes = dict(prefixes)
        self.drop = set(drop)

    def apply_list(self, values):
        out = []
        for value in values:
            if value in self.drop:
                continue
            for old, new in self.prefixes.items():
                if value.startswith(old):
                    value = new + value[len(old):]
                    break
            out.append(value)
        return out


GLM = {"model.language_model.": "language_model.model.", "model.visual.": "visual."}


def test_targets_and_ignores_move_into_the_module_namespace(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    declared = "model.language_model.layers.0.self_attn.qkv_proj"
    passed = "model.language_model.layers.1.mlp.experts"
    config = _resolved(_config(targets=(declared,), ignore=(passed, "lm_head")))

    config.apply_vllm_mapper(_Mapper(GLM))

    assert "language_model.model.layers.0.self_attn.qkv_proj" in config.target_scheme
    assert declared not in config.target_scheme, "the checkpoint name survived the mapping"
    assert "language_model.model.layers.1.mlp.experts" in config.ignore
    assert "lm_head" in config.ignore, "a bare module name is not a path and is left alone"


def test_the_mapped_target_is_the_one_that_dispatches(monkeypatch):
    """The point of the whole exercise."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(targets=("model.language_model.layers.0.mlp.gate_up_proj",)))
    config.apply_vllm_mapper(_Mapper(GLM))

    layer = _layer()
    assert config.get_quant_method(
        layer, "language_model.model.layers.0.mlp.gate_up_proj") is not None
    with pytest.raises(ValueError, match="declares no wire"):
        config.get_quant_method(layer, "model.language_model.layers.0.mlp.gate_up_proj")


def test_a_regex_target_is_left_alone(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(targets=("re:.*gate_up_proj$",)))
    config.apply_vllm_mapper(_Mapper(GLM))
    assert "re:.*gate_up_proj$" in config.target_scheme


def test_a_dropped_target_refuses_rather_than_vanishing(monkeypatch):
    """compressed-tensors drops it silently; a dead wire is worse than a stop."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    gone = "model.language_model.layers.0.mlp.gate_up_proj"
    config = _resolved(_config(targets=(gone,)))
    with pytest.raises(ValueError, match="drops"):
        config.apply_vllm_mapper(_Mapper(GLM, drop=(gone,)))


def test_two_checkpoint_modules_cannot_share_one_vllm_module(monkeypatch):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(targets=("a.x.proj", "b.x.proj")))
    with pytest.raises(ValueError, match="both map to the module"):
        config.apply_vllm_mapper(_Mapper({"a.": "c.", "b.": "c."}))


def test_an_overlap_created_by_the_mapping_is_still_refused(monkeypatch):
    """The declared/ignored check has to run again on the mapped names."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(targets=("a.x.proj",), ignore=("b.x.proj",)))
    with pytest.raises(ValueError, match="both declared and ignored"):
        config.apply_vllm_mapper(_Mapper({"a.": "c.", "b.": "c."}))


def test_a_model_with_no_mapper_is_untouched(monkeypatch):
    """Qwen3-0.6B: vLLM never calls the hook, and nothing may change if it does."""
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    config = _resolved(_config(targets=(TARGET,), ignore=("model.embed_tokens",)))
    before = dict(config.target_scheme), tuple(config.ignore)
    config.apply_vllm_mapper(_Mapper({}))
    assert (config.target_scheme, config.ignore) == before
