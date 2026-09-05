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


def test_lfm_routes_the_routed_expert_stack():
    """The EUGR launch target offers LFM's non-Linear expert stack to the plugin."""
    entry = construction_entry(["Lfm2MoeForCausalLM"])
    assert entry is not None, "the pinned LFM census receipt is not in the contract"
    assert entry["runtime"]["image"] == (
        "eugr/spark-vllm@sha256:"
        "0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c"
    )
    verdict, pattern = classify_construction(
        entry, "model.layers.2.feed_forward.experts")
    assert verdict == "offered", (verdict, pattern)
    assert pattern == "model.layers.*.feed_forward.experts"


def test_an_uncensused_architecture_answers_none_rather_than_yes():
    assert construction_entry(["NoSuchModelForCausalLM"]) is None


# --- a pattern whose members disagreed is not a clearance (#204) -----------
#
# The census does NOT let the last member win: when two modules under one
# normalised pattern answer differently it keeps the first value and records
# the exact prefixes that differ under ``disagreements``
# (``tools/tessera_construction_census.py``'s ``census``).  Derivation used to
# read only the first member's ``offered_to_quant_config`` and drop that field
# on the floor, so a pattern the census SAW disagree became an unconditional
# ``offered`` for every one of its members -- a clearance to delete a
# ``<module>.weight`` the runtime may build with ``quant_config=None``, which is
# the exact artifact this block exists to prevent.  A normalised pattern whose
# observed members disagree predicts nothing about the members that were not
# observed (the GLM receipt is a 4-layer cut of a 92-layer model), so the only
# honest verdict is that the pattern does not answer.


def _receipt(name):
    return json.loads((ROOT / "docs" / "measurements" / "construction" / name).read_text())


def _with_disagreement(receipt, pattern, field, prefixes):
    """The census's own disagreement schema, written onto a committed receipt."""
    row = next(r for r in receipt["linears"] if r["prefix_pattern"] == pattern)
    row.setdefault("disagreements", {})[field] = list(prefixes)
    return receipt


def test_a_first_offered_pattern_with_a_disagreeing_member_clears_nobody():
    receipt = _with_disagreement(
        _receipt("qwen3-0.6b.json"), "model.layers.*.mlp.down_proj",
        "offered_to_quant_config", ["model.layers.1.mlp.down_proj"])
    entry = construction_entry_from_receipt(receipt)
    for module in ("model.layers.1.mlp.down_proj", "model.layers.0.mlp.down_proj"):
        verdict, pattern = classify_construction(entry, module)
        assert (verdict, pattern) == ("disagreement", "model.layers.*.mlp.down_proj"), module
    assert entry["disagreements"] == [
        {"prefix_pattern": "model.layers.*.mlp.down_proj",
         "fields": {"offered_to_quant_config": ["model.layers.1.mlp.down_proj"]}}], (
        "the derived entry dropped the disagreement the census recorded")
    assert "model.layers.*.mlp.down_proj" not in entry["offered"], (
        "a pattern the census saw disagree cannot sit in the clearance list")
    # Every other pattern in the same receipt is unaffected.
    assert classify_construction(entry, "model.layers.0.self_attn.o_proj")[0] == "offered"


def test_the_inverse_ordering_says_disagreement_rather_than_never_offered():
    """First member unoffered, a later one offered: still not an answer.

    This ordering fails safe for the exporter -- ``never_offered`` refuses too
    -- but it refuses for a reason the census did not observe, and a refusal
    that names the wrong fact is not something a producer can act on.
    """
    pattern = "language_model.model.layers.*.self_attn.o_proj"
    receipt = _with_disagreement(
        _receipt("glm53-flash-4layer.json"), pattern, "offered_to_quant_config",
        ["language_model.model.layers.3.self_attn.o_proj"])
    entry = construction_entry_from_receipt(receipt)
    verdict, seen = classify_construction(
        entry, "model.language_model.layers.3.self_attn.o_proj")
    assert (verdict, seen) == ("disagreement", pattern)
    assert [row["prefix_pattern"] for row in entry["disagreements"]] == [pattern]


def test_the_committed_receipts_are_the_homogeneous_control():
    """No committed census observed a disagreement, so nothing above moves them."""
    for path in RECEIPTS:
        entry = construction_entry_from_receipt(json.loads(path.read_text()))
        assert entry.get("disagreements", []) == [], path.name
    entry = construction_entry(["Qwen3ForCausalLM"])
    assert classify_construction(entry, "model.layers.0.mlp.down_proj")[0] == "offered"


def test_validator_refuses_a_disagreeing_pattern_that_is_also_cleared():
    """A hand-typed block cannot list a pattern as both disagreeing and offered."""
    contract = _contract_with(json.loads(json.dumps(load_serving_contract()["construction"])))
    entry = contract["construction"]["architectures"][0]
    entry["disagreements"] = [
        {"prefix_pattern": entry["offered"][0],
         "fields": {"offered_to_quant_config": ["a.0.b"]}}]
    with pytest.raises(ValueError, match="disagreed"):
        validate_serving_contract(contract)


def test_validator_refuses_an_empty_disagreement_row():
    contract = _contract_with(json.loads(json.dumps(load_serving_contract()["construction"])))
    entry = contract["construction"]["architectures"][0]
    entry["disagreements"] = [{"prefix_pattern": "model.layers.*.mlp.down_proj", "fields": {}}]
    with pytest.raises(ValueError, match="fields"):
        validate_serving_contract(contract)


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


# --- vllm_module_name against vLLM's own WeightsMapper (#108) --------------
#
# ``vllm_module_name`` is the one place the producer computes what vLLM WOULD
# do rather than reading what it DID, and principle 14 does not have an
# exemption for "the algorithm is code, not a table".  The algorithm genuinely
# is code -- ``WeightsMapper._map_name_with_shard`` -- so it cannot be derived
# from the census table alone.  What closes the gap is a pair of gates:
#
#   * these tests, which pin the semantics vLLM 0.28 has, in the pure suite;
#   * ``tests/test_serving_name_mapping.py::test_vllm_module_name_agrees_with_
#     the_real_weights_mapper``, which runs the SAME names through the real
#     ``WeightsMapper`` inside the serving image and refuses on any
#     disagreement.  That one is the attestation; this one is the description.
#
# and the refusal below, which keeps the description from silently covering a
# field it never implemented.


def _mapper_entry(table):
    """A construction entry carrying nothing but a mapper table."""
    return {"architecture": "StubForCausalLM", "hf_to_vllm_mapper_unstacked": table,
            "offered": [], "never_offered": []}


def test_a_substring_rule_replaces_one_occurrence_not_every_one():
    """``key.replace(substr, new_key, 1)`` -- vLLM 0.28 utils.py, ``_map_name_with_shard``."""
    entry = _mapper_entry({"orig_to_new_substr": {".block.": ".layer."}})
    assert vllm_module_name(entry, "model.block.0.block.1.proj") == \
        "model.layer.0.block.1.proj"


def test_prefix_rules_fall_through_instead_of_stopping_at_the_first_match():
    """vLLM's prefix loop has no ``break``: a later rule sees the rewritten key."""
    entry = _mapper_entry({"orig_to_new_prefix": {"model.": "language_model.",
                                                  "language_model.": "lm."}})
    assert vllm_module_name(entry, "model.layers.0.mlp.down_proj") == "lm.layers.0.mlp.down_proj"


def test_suffix_rules_fall_through_too():
    entry = _mapper_entry({"orig_to_new_suffix": {".a_proj": ".b_proj", ".b_proj": ".c_proj"}})
    assert vllm_module_name(entry, "model.layers.0.a_proj") == "model.layers.0.c_proj"


def test_a_renaming_rule_is_refused_rather_than_ignored():
    """``orig_to_new_renaming`` is a list of transformers ``WeightRenaming`` objects.

    It cannot be replayed from a JSON table, so a checkpoint whose class
    declares one gets a refusal that names the field -- not a name computed as
    though the rule were not there.
    """
    entry = _mapper_entry({"orig_to_new_renaming": [{"repr": "<WeightRenaming ...>"}]})
    with pytest.raises(ValueError, match="orig_to_new_renaming"):
        vllm_module_name(entry, "model.layers.0.mlp.down_proj")


def test_a_regex_rule_is_refused_rather_than_ignored():
    entry = _mapper_entry({"orig_to_new_regex": {r"layers\.(\d+)": r"blocks.\1"}})
    with pytest.raises(ValueError, match="orig_to_new_regex"):
        vllm_module_name(entry, "model.layers.0.mlp.down_proj")


def test_a_mapper_field_this_producer_has_never_seen_is_refused():
    """Pin the rule, not the roster: a field vLLM adds tomorrow refuses today."""
    entry = _mapper_entry({"orig_to_new_something_vllm_added": {"a": "b"}})
    with pytest.raises(ValueError, match="orig_to_new_something_vllm_added"):
        vllm_module_name(entry, "model.layers.0.mlp.down_proj")


def test_a_populated_stacked_map_is_refused_because_the_table_claims_to_be_unstacked():
    """``get_unstacked_mapper`` empties it; a non-empty one means the census fell back.

    vLLM applies ``orig_to_new_stacked`` inside ``_map_name``, so ignoring a
    populated one would be a silent divergence -- and the field is named
    ``hf_to_vllm_mapper_unstacked``, so a populated one is a contradiction in
    the receipt, not a case to implement.
    """
    entry = _mapper_entry({"orig_to_new_stacked": {".q_proj.": [".qkv_proj.", "q"]}})
    with pytest.raises(ValueError, match="orig_to_new_stacked"):
        vllm_module_name(entry, "model.layers.0.self_attn.q_proj")


def test_an_empty_stacked_map_is_not_a_refusal():
    """The census records only non-empty fields, but a receipt may spell it out."""
    entry = _mapper_entry({"orig_to_new_stacked": {},
                           "orig_to_new_prefix": {"model.": "lm."}})
    assert vllm_module_name(entry, "model.layers.0.mlp.down_proj") == "lm.layers.0.mlp.down_proj"


def test_a_dropped_name_is_still_a_refusal():
    entry = _mapper_entry({"orig_to_new_prefix": {"model.visual.": None}})
    with pytest.raises(ValueError, match="DROPS"):
        vllm_module_name(entry, "model.visual.blocks.0.attn.qkv")


# --- the census must not drop a mapper field on the way into the receipt ---


def _census_module():
    """The census tool, loaded by path: its top level is stdlib-only by design."""
    import importlib.util
    path = ROOT / "tools" / "tessera_construction_census.py"
    spec = importlib.util.spec_from_file_location("tessera_construction_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_census_records_every_mapper_field_the_runtime_declares():
    """The producer can only refuse a rule the receipt admits exists.

    ``_weights_mapper_table`` used to iterate a hardcoded four-name roster, so
    a model class declaring ``orig_to_new_regex`` or ``orig_to_new_renaming``
    produced a receipt that OMITTED the rule -- and a producer reading that
    receipt would map the name as though the rule were not there, with nothing
    on either side able to notice. Read the dataclass instead (AGENTS.md rule
    3: pin the rule, not the roster).
    """
    import dataclasses
    import re as _re

    @dataclasses.dataclass
    class FakeMapper:
        """The shape of vLLM's WeightsMapper, including the fields we refuse."""
        orig_to_new_renaming: list = dataclasses.field(default_factory=list)
        orig_to_new_regex: dict = dataclasses.field(default_factory=dict)
        orig_to_new_substr: dict = dataclasses.field(default_factory=dict)
        orig_to_new_stacked: dict = dataclasses.field(default_factory=dict)
        orig_to_new_prefix: dict = dataclasses.field(default_factory=dict)
        orig_to_new_suffix: dict = dataclasses.field(default_factory=dict)
        orig_to_new_invented_next_release: dict = dataclasses.field(default_factory=dict)

        def get_unstacked_mapper(self):
            return dataclasses.replace(self, orig_to_new_stacked={})

    class FakeClass:
        hf_to_vllm_mapper = FakeMapper(
            orig_to_new_regex={_re.compile(r"layers\.(\d+)"): r"blocks.\1"},
            orig_to_new_renaming=[object()],
            orig_to_new_prefix={"model.": "lm."},
            orig_to_new_stacked={".q_proj.": (".qkv_proj.", "q")},
            orig_to_new_invented_next_release={"x": "y"})

    table = _census_module()._weights_mapper_table(FakeClass)
    assert set(table) == {"orig_to_new_regex", "orig_to_new_renaming", "orig_to_new_prefix",
                          "orig_to_new_invented_next_release"}, (
        f"the census dropped a non-empty mapper field: {sorted(table)}")
    # ``get_unstacked_mapper`` empties the stacked map, so it is absent rather
    # than refused -- the producer only refuses a POPULATED one.
    assert "orig_to_new_stacked" not in table
    assert table["orig_to_new_regex"] == {r"layers\.(\d+)": r"blocks.\1"}
    assert table["orig_to_new_prefix"] == {"model.": "lm."}
    # And what the census recorded is exactly what the producer refuses on.
    with pytest.raises(ValueError, match="orig_to_new_regex"):
        vllm_module_name({"hf_to_vllm_mapper_unstacked": table}, "model.layers.0.mlp.down_proj")


def test_the_census_still_produces_the_tables_the_committed_receipts_carry():
    """The receipt shape did not move: the two committed censuses need no re-run."""
    import dataclasses

    @dataclasses.dataclass
    class PrefixOnly:
        orig_to_new_renaming: list = dataclasses.field(default_factory=list)
        orig_to_new_regex: dict = dataclasses.field(default_factory=dict)
        orig_to_new_substr: dict = dataclasses.field(default_factory=dict)
        orig_to_new_stacked: dict = dataclasses.field(default_factory=dict)
        orig_to_new_prefix: dict = dataclasses.field(default_factory=dict)
        orig_to_new_suffix: dict = dataclasses.field(default_factory=dict)

        def get_unstacked_mapper(self):
            return dataclasses.replace(self, orig_to_new_stacked={})

    census = _census_module()
    glm = json.loads((ROOT / "docs" / "measurements" / "construction" /
                      "glm53-flash-4layer.json").read_text())
    table = glm["hf_to_vllm_mapper_unstacked"]

    class Glm:
        hf_to_vllm_mapper = PrefixOnly(orig_to_new_prefix=dict(table["orig_to_new_prefix"]))

    assert census._weights_mapper_table(Glm) == table

    class NoMapper:
        pass

    assert census._weights_mapper_table(NoMapper) is None
