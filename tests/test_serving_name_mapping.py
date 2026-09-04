"""The name-mapping hook, against the real vLLM -- not a duck type.

``tests/test_serving_dispatch.py`` covers the SEMANTICS of
``TesseraConfig.apply_vllm_mapper`` under a stubbed vLLM.  What a stub cannot
check is that the contract still exists: that vLLM still hands quant configs a
mapper, still calls it before any layer is built, and that its ``WeightsMapper``
still exposes ``apply_list``.  Those are claims about another runtime, so they
are asserted against that runtime or not at all (principle 14).

This file runs where a real vLLM is importable -- the serving image -- and
skips elsewhere.  It is the reason the hook cannot silently stop being called:
if a future build renames it, moves the call site, or drops ``apply_list``, one
of these fails instead of every target quietly staying in checkpoint space.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.layers.quantization.base_config import (  # noqa: E402
    QuantizationConfig)
from vllm.model_executor.models.utils import WeightsMapper                # noqa: E402

from tessera.serving.config import TesseraConfig                          # noqa: E402
from tessera.serving.lane import TESSERA_MODE_ENV                         # noqa: E402
from tessera.serving.scheme import TESSERA_FP8                            # noqa: E402

#: The mapper ``Glm5NextForConditionalGeneration`` declares, copied as data.
#: The test does not import the model class -- a multimodal GLM build is not a
#: precondition for this contract -- it uses the same SHAPE of mapper.
GLM_PREFIXES = {"model.language_model.": "language_model.model.",
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head."}

SCHEME = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
          "q256": 1024, "rows": 2048, "columns": 4096, "wire_bytes": 1048576,
          "window_bits": 14, "roles": [["weight", 2048]]}


def _config(monkeypatch, targets, ignore=()):
    monkeypatch.setenv(TESSERA_MODE_ENV, "resident")
    return TesseraConfig.from_config({
        "quant_method": "tessera", "format": "tessera",
        "config_groups": {"tessera": {"format": "TESSERA", "targets": list(targets),
                                      "scheme": SCHEME}},
        "ignore": list(ignore)})


def test_vllm_still_offers_the_hook_this_plugin_needs():
    """If this fails, targets are silently not being translated any more."""
    assert hasattr(QuantizationConfig, "apply_vllm_mapper"), (
        "vLLM no longer offers apply_vllm_mapper; Tessera's targets are written in the "
        "checkpoint's namespace and something else must now translate them")
    assert TesseraConfig.apply_vllm_mapper is not QuantizationConfig.apply_vllm_mapper, (
        "TesseraConfig is inheriting the base no-op again")


def test_vllm_still_calls_the_hook_before_any_layer_is_built():
    """The hook is useless if nothing invokes it.

    Both call sites are checked by source: ``configure_quant_config`` for a
    plain model class, and ``SupportsQuant`` for one that carries the mixin.
    """
    from vllm.model_executor.model_loader import utils as loader_utils
    source = inspect.getsource(loader_utils.configure_quant_config)
    assert "apply_vllm_mapper" in source, (
        "vLLM's configure_quant_config no longer applies the mapper to the quant config")

    from vllm.model_executor.models import interfaces
    assert "apply_vllm_mapper" in inspect.getsource(interfaces), (
        "the SupportsQuant path no longer applies the mapper")


def test_the_real_weights_mapper_translates_tessera_targets(monkeypatch):
    mapper = WeightsMapper(orig_to_new_prefix=GLM_PREFIXES).get_unstacked_mapper()
    declared = "model.language_model.layers.0.self_attn.qkv_proj"
    experts = "model.language_model.layers.1.mlp.experts"
    config = _config(monkeypatch, targets=(declared,), ignore=(experts, "lm_head"))

    config.apply_vllm_mapper(mapper)

    assert "language_model.model.layers.0.self_attn.qkv_proj" in config.target_scheme, (
        f"the real mapper did not translate the target: {sorted(config.target_scheme)}")
    assert "language_model.model.layers.1.mlp.experts" in config.ignore, sorted(config.ignore)
    # ``lm_head`` has no dot: a module class name, not a path, and left alone by
    # the same rule compressed-tensors uses.
    assert "lm_head" in config.ignore, sorted(config.ignore)


def test_the_stacked_names_this_exporter_writes_survive_the_unstacked_mapper(monkeypatch):
    """``get_unstacked_mapper`` drops the stacked map; our targets ARE stacked.

    The exporter declares the fused module (``qkv_proj``, ``gate_up_proj``)
    because vLLM builds one method per fused module.  vLLM passes the config the
    UNSTACKED mapper, whose purpose is to leave constituent names alone -- so
    the question is whether it leaves the stacked ones alone too.
    """
    mapper = WeightsMapper(
        orig_to_new_prefix=GLM_PREFIXES,
        orig_to_new_stacked={".q_proj.": (".qkv_proj.", "q")},
    ).get_unstacked_mapper()
    targets = ("model.language_model.layers.0.self_attn.qkv_proj",
               "model.language_model.layers.0.mlp.gate_up_proj")
    config = _config(monkeypatch, targets=targets)

    config.apply_vllm_mapper(mapper)

    assert set(config.target_scheme) == {
        "language_model.model.layers.0.self_attn.qkv_proj",
        "language_model.model.layers.0.mlp.gate_up_proj"}, sorted(config.target_scheme)


def test_a_mapper_that_changes_nothing_changes_nothing(monkeypatch):
    """Qwen3-0.6B's class declares no mapper, and every artifact served so far is one."""
    config = _config(monkeypatch, targets=("model.layers.0.self_attn.qkv_proj",),
                     ignore=("lm_head",))
    before = dict(config.target_scheme), tuple(config.ignore)
    config.apply_vllm_mapper(WeightsMapper().get_unstacked_mapper())
    assert (config.target_scheme, config.ignore) == before


# --- vllm_module_name vs the real WeightsMapper (#108) ---------------------
#
# ``contract.vllm_module_name`` is the producer's replay of
# ``WeightsMapper._map_name_with_shard``: the one place in this repo that
# computes what vLLM WOULD do rather than reading what it DID.  The algorithm
# is code, not a table, so it cannot be derived from the census receipt -- the
# receipt publishes the table and vLLM keeps the loop.  What principle 14 gets
# instead is this: the same names through both, inside the pinned image, with
# any disagreement a failure.  ``tests/test_serving_construction.py`` describes
# the semantics; this attests them.

import itertools  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from tessera.serving.contract import (  # noqa: E402
    _MAPPER_FIELDS_REPLAYED, vllm_module_name)

RECEIPTS = sorted(
    (Path(__file__).resolve().parents[1] / "docs" / "measurements" / "construction")
    .glob("*.json"))

#: Mapper tables to attest.  The committed receipts are the ones that ship;
#: the synthetic ones are the shapes the replay was WRONG on before #108 --
#: a substring twice in a name, prefix rules that chain, suffix rules that
#: chain -- which no committed receipt exercises and which therefore have to
#: be constructed to be attested at all.
SYNTHETIC_TABLES = {
    "substr_twice": {"orig_to_new_substr": {".block.": ".layer."}},
    "prefix_chain": {"orig_to_new_prefix": {"model.": "language_model.",
                                            "language_model.": "lm."}},
    "suffix_chain": {"orig_to_new_suffix": {".a_proj": ".b_proj", ".b_proj": ".c_proj"}},
    "substr_then_prefix_then_suffix": {
        "orig_to_new_substr": {"decoder.": "layers."},
        "orig_to_new_prefix": {"model.": "language_model.model."},
        "orig_to_new_suffix": {".gate_up": ".gate_up_proj"}},
    "a_dropping_prefix": {"orig_to_new_prefix": {"model.visual.": None,
                                                 "model.": "language_model.model."}},
    "a_dropping_substr": {"orig_to_new_substr": {".mtp.": None}},
    "a_dropping_suffix": {"orig_to_new_suffix": {".inv_freq": None}},
}

#: Leaves a real checkpoint names, so the probe set is not only rule fragments.
LEAVES = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
          "mlp.experts.3.down_proj", "attn.qkv", "a_proj", "gate_up")


def _probe_names(table):
    """Names that exercise every rule in ``table``, plus ordinary ones.

    Each rule key is placed at the front, at the end and TWICE in the middle,
    because the three divergences #108 found were exactly a rule firing in a
    position the replay handled differently from vLLM.
    """
    names = {f"model.layers.0.{leaf}" for leaf in LEAVES}
    names |= {f"model.visual.blocks.1.{leaf}" for leaf in LEAVES}
    keys = [k for field in _MAPPER_FIELDS_REPLAYED for k in (table.get(field) or {})]
    for key in keys:
        names |= {key,
                  f"{key}layers.0.mlp.down_proj",
                  f"model.layers.0{key}",
                  f"model.layers.0.{key}mlp.{key}down_proj",
                  f"prefix{key}middle{key}suffix"}
    for a, b in itertools.permutations(keys, 2):
        names.add(f"{a}model.layers.0{b}")
    return sorted(names)


def _tables():
    yield from SYNTHETIC_TABLES.items()
    for path in RECEIPTS:
        receipt = json.loads(path.read_text())
        table = receipt.get("hf_to_vllm_mapper_unstacked") or {}
        yield path.stem, table


def _real_mapper(table):
    """vLLM's own mapper, built from the census table.

    Only the replayed fields are constructible from JSON -- a regex key is a
    compiled pattern and a renaming is a transformers object -- which is the
    same boundary ``_require_replayable_mapper`` refuses at.  A table carrying
    one of those is skipped here and refused there; neither is silently mapped.
    """
    kwargs = {field: dict(table[field]) for field in _MAPPER_FIELDS_REPLAYED
              if table.get(field)}
    return WeightsMapper(**kwargs).get_unstacked_mapper()


@pytest.mark.parametrize("name,table", list(_tables()), ids=lambda v: v if isinstance(v, str) else "")
def test_vllm_module_name_agrees_with_the_real_weights_mapper(name, table):
    """The attestation #108 asks for: our replay against vLLM's loop, name by name."""
    unreplayable = sorted(f for f, v in table.items() if v and f not in _MAPPER_FIELDS_REPLAYED)
    if unreplayable:
        pytest.skip(f"{name} uses {unreplayable}, which the producer refuses rather than replays")
    mapper = _real_mapper(table)
    entry = {"architecture": name, "hf_to_vllm_mapper_unstacked": table}
    probes = _probe_names(table)
    assert probes, "an attestation over no names attests nothing"
    for probe in probes:
        expected = mapper._map_name(probe)
        try:
            got = vllm_module_name(entry, probe)
        except ValueError:
            got = None
        assert got == expected, (
            f"{name}: vllm_module_name({probe!r}) = {got!r}, but the real WeightsMapper "
            f"in this image says {expected!r}. The producer's replay of "
            f"_map_name_with_shard has diverged from the runtime it claims to describe.")


def test_the_producer_refuses_every_mapper_field_it_does_not_replay():
    """Pin the rule against the runtime's own dataclass, not against a roster.

    If a future vLLM grows a seventh ``orig_to_new_*`` field, this fails --
    which is the point: the producer must refuse it (already true, the refusal
    is on the complement) and someone must decide whether to replay it.
    """
    import dataclasses
    declared = {f.name for f in dataclasses.fields(WeightsMapper)}
    replayed = set(_MAPPER_FIELDS_REPLAYED)
    assert replayed <= declared, (
        f"vLLM's WeightsMapper no longer declares {sorted(replayed - declared)}; the "
        "producer is replaying a field that no longer exists")
    for field in sorted(declared - replayed):
        entry = {"architecture": "StubForCausalLM",
                 "hf_to_vllm_mapper_unstacked": {field: {"a": "b"}}}
        with pytest.raises(ValueError, match=field):
            vllm_module_name(entry, "model.layers.0.mlp.down_proj")
