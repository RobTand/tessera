"""The producer may only quantize a Linear the runtime routes to the plugin (#99).

THE DEFECT THIS PINS.  ``LinearBase.__init__`` takes ``UnquantizedLinearMethod()``
in the ``quant_config is None`` branch **without calling** ``get_quant_method``
(vLLM 0.28, ``model_executor/layers/linear.py:258``).  So a projection a model
builds with ``quant_config=None`` is invisible to every quantization plugin --
ours cannot refuse it, cannot warn, cannot even see the prefix.  On
``Glm5NextForConditionalGeneration`` that is every attention projection
(``models/glm5next/nvidia/model.py:331``), the whole KDA layer
(``kda.py:171-174``), the sparse indexer (``attention.py:263``) and the entire
vision tower (``model.py:1082``), while ``experiments/export_tessera_serving.py``
quantized all of them by default: 2-D, under ``BODY_LAYER``, planned.  The wire
went in, the ``<module>.weight`` came out, and no gate on either side said a
word.

WHY THE TABLE IS GENERATED.  Principle 14: a claim about what a serving runtime
DOES is derived from a machine-readable table the runtime publishes, never
asserted beside it.  A roster of "these modules are built unquantized" typed
into the exporter would be exactly the assertion the principle forbids -- and it
would go stale the first time a model implementation moved a ``quant_config=``.
So ``tools/tessera_construction_census.py`` OBSERVES the answer inside the
pinned image (build the model the way the loader does, with a probe
``QuantizationConfig`` that records every prefix vLLM offers it), the receipts
live under ``docs/measurements/construction/``, and this file re-derives the
contract's block from them.  A hand-edited contract row fails here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    CONSTRUCTION_CENSUS_SCHEMA, CONSTRUCTION_SCHEMA, classify_construction,
    construction_entry, construction_entry_from_receipt, load_serving_contract,
    normalise_module, validate_serving_contract, vllm_module_name)

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = sorted((ROOT / "docs" / "measurements" / "construction").glob("*.json"))


def test_receipts_exist():
    assert RECEIPTS, "the construction block is generated from receipts; none are committed"


def test_contract_block_is_derived_from_the_receipts():
    """The table is the receipts, not a transcription of them."""
    contract = load_serving_contract()
    block = contract["construction"]
    assert block["schema"] == CONSTRUCTION_SCHEMA
    derived = []
    for path in RECEIPTS:
        entry = construction_entry_from_receipt(json.loads(path.read_text()))
        entry["receipt"] = str(path.relative_to(ROOT))
        derived.append(entry)
    derived.sort(key=lambda e: e["architecture"])
    assert block["architectures"] == derived, (
        "runtime_contract.json's construction block does not match the census receipts. "
        "Regenerate it from docs/measurements/construction/ rather than editing it: a row "
        "typed by hand is the assertion principle 14 refuses.")


@pytest.mark.parametrize("path", RECEIPTS, ids=lambda p: p.stem)
def test_receipt_is_a_census_of_one_image(path):
    receipt = json.loads(path.read_text())
    assert receipt["schema"] == CONSTRUCTION_CENSUS_SCHEMA
    for key in ("image", "image_id", "vllm"):
        assert receipt["runtime"][key] not in ("", None, "unstamped"), (
            f"{path.name} does not say which runtime it observed; a construction answer is a "
            "property of the image and an unstamped one cannot be checked against a serve")
    assert receipt["linears"], "a census that found no Linear observed nothing"


def test_normalise_collapses_every_repeat_index():
    assert (normalise_module("language_model.model.layers.7.self_attn.o_proj")
            == "language_model.model.layers.*.self_attn.o_proj")
    # Not only ``layers.N``: the vision tower spells its stack ``blocks.N``, and
    # a census that normalised one spelling published 300 near-identical rows.
    assert normalise_module("visual.blocks.23.attn.qkv") == "visual.blocks.*.attn.qkv"
    assert normalise_module("model.layers.0.mlp.experts.11.down_proj") == \
        "model.layers.*.mlp.experts.*.down_proj"


def test_glm_attention_is_not_routed_and_its_mlp_is():
    """The measured answer, in the shape the exporter's gate reads it."""
    entry = construction_entry(["Glm5NextForConditionalGeneration"])
    assert entry is not None, "the GLM census receipt is not reaching the contract"
    # Its checkpoint namespace is not vLLM's: the mapper swaps the two leading
    # segments, and a producer that skipped it would join nothing.
    assert (vllm_module_name(entry, "model.language_model.layers.1.self_attn.o_proj")
            == "language_model.model.layers.1.self_attn.o_proj")

    never = ["self_attn.o_proj", "self_attn.q_b_proj", "self_attn.kv_b_proj",
             "self_attn.indexer.wq_b", "self_attn.indexer.wk_weights_proj",
             "self_attn.f_b_proj", "self_attn.g_b_proj", "self_attn.fused_qkv_a_proj",
             "self_attn.in_proj_qkvbfg_a"]
    for leaf in never:
        verdict, _ = classify_construction(
            entry, f"model.language_model.layers.1.{leaf}")
        assert verdict == "never_offered", f"{leaf} should be built with quant_config=None"

    # The names the exporter's own fused rule produces on this model, which vLLM
    # does not build at all: a KDA layer's q/k/v merge into ONE
    # ``in_proj_qkvbfg_a`` and an MLA layer's q_a/kv_a into ``fused_qkv_a_proj``.
    for leaf in ("self_attn.qkv_proj", "self_attn.q_a_proj",
                 "self_attn.kv_a_proj_with_mqa", "self_attn.indexer.wk"):
        verdict, _ = classify_construction(
            entry, f"model.language_model.layers.1.{leaf}")
        assert verdict == "absent", f"{leaf} is not a module this runtime builds"

    for leaf in ("mlp.down_proj", "mlp.gate_up_proj",
                 "mlp.shared_experts.down_proj", "mlp.shared_experts.gate_up_proj"):
        verdict, _ = classify_construction(
            entry, f"model.language_model.layers.1.{leaf}")
        assert verdict == "offered", f"{leaf} is the runtime's own quantizable route"

    # The vision tower is BUILT and never offered -- so the plugin's
    # "declared or ignored" refusal never fires for it either.
    assert classify_construction(entry, "model.visual.blocks.3.attn.qkv")[0] == "never_offered"


def test_qwen_is_the_control():
    """Every Qwen Linear IS offered, which is why this went unseen for so long."""
    entry = construction_entry(["Qwen3ForCausalLM"])
    assert entry is not None
    assert entry["never_offered"] == []
    for leaf in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
        assert classify_construction(entry, f"model.layers.0.{leaf}")[0] == "offered"


def test_an_uncensused_architecture_answers_none_rather_than_yes():
    assert construction_entry(["NoSuchModelForCausalLM"]) is None


def _contract_with(block):
    contract = json.loads(json.dumps(load_serving_contract()))
    contract["construction"] = block
    return contract


def test_validator_refuses_a_module_that_is_both_offered_and_not():
    contract = _contract_with(json.loads(json.dumps(load_serving_contract()["construction"])))
    entry = contract["construction"]["architectures"][0]
    entry["offered"] = sorted(entry["offered"] + [entry["never_offered"][0]["prefix_pattern"]])
    with pytest.raises(ValueError, match="both offered and never offered"):
        validate_serving_contract(contract)


def test_validator_refuses_two_censuses_of_one_architecture():
    contract = _contract_with(json.loads(json.dumps(load_serving_contract()["construction"])))
    entries = contract["construction"]["architectures"]
    entries.append(json.loads(json.dumps(entries[0])))
    with pytest.raises(ValueError, match="censused twice"):
        validate_serving_contract(contract)


def test_validator_refuses_an_unstamped_runtime():
    contract = _contract_with(json.loads(json.dumps(load_serving_contract()["construction"])))
    contract["construction"]["architectures"][0]["runtime"]["image_id"] = ""
    with pytest.raises(ValueError, match="image_id is empty"):
        validate_serving_contract(contract)
