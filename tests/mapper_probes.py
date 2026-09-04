"""The probe set two gates must agree on, in one place (#108).

``tests/test_serving_name_mapping.py`` runs these names through the real
``WeightsMapper`` inside the serving image and through
``contract.vllm_module_name``, and that is the attestation.
``experiments/vllm_module_name_distance.py`` then measures how far another
revision is from the attested one over the SAME names -- an inference that is
only valid while the two are literally the same set, which is why this module
exists rather than a copy in each (AGENTS.md rule 4).

Stdlib only, and no ``importorskip``: the harness must import it on a box with
no vLLM.
"""
from __future__ import annotations

import itertools

#: The ``WeightsMapper`` fields the producer replays.  Duplicated from
#: ``contract._MAPPER_FIELDS_REPLAYED`` deliberately -- this module has to
#: import on a checkout where ``tessera`` is not on the path, and
#: ``test_the_producer_refuses_every_mapper_field_it_does_not_replay`` is what
#: keeps the two honest against vLLM's own dataclass.
REPLAYED_FIELDS = ("orig_to_new_substr", "orig_to_new_prefix", "orig_to_new_suffix")

#: Tables to attest.  The committed census receipts are the ones that ship;
#: these are the shapes the replay was WRONG on before #108 -- a substring
#: twice in a name, prefix rules that chain, suffix rules that chain -- which
#: no committed receipt exercises and which therefore have to be constructed
#: to be attested at all.
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


def probe_names(table):
    """Names that exercise every rule in ``table``, plus ordinary ones.

    Each rule key is placed at the front, at the end and TWICE in the middle,
    because the three divergences #108 found were exactly a rule firing in a
    position the replay handled differently from vLLM.
    """
    names = {f"model.layers.0.{leaf}" for leaf in LEAVES}
    names |= {f"model.visual.blocks.1.{leaf}" for leaf in LEAVES}
    keys = [key for field in REPLAYED_FIELDS for key in (table.get(field) or {})]
    for key in keys:
        names |= {key,
                  f"{key}layers.0.mlp.down_proj",
                  f"model.layers.0{key}",
                  f"model.layers.0.{key}mlp.{key}down_proj",
                  f"prefix{key}middle{key}suffix"}
    for first, second in itertools.permutations(keys, 2):
        names.add(f"{first}model.layers.0{second}")
    return sorted(names)
