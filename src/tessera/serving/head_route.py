"""The vocab-parallel LM head as a Tessera dense module (tessera#750 WP3).

vLLM builds the head as ``ParallelLMHead``, a ``VocabParallelEmbedding`` and
not a ``LinearBase``, and asks the quant config for its method like any other
layer (``vocab_parallel_embedding.py``: ``quant_config.get_quant_method(self,
prefix=prefix)``).  ``LogitsProcessor`` then projects the hidden states with
``lm_head.quant_method.apply(lm_head, hidden_states, bias=...)``, gathers the
vocab shards and slices off the padded vocabulary.  So a dense Tessera method
serves the head unchanged: ``create_weights`` is called with the layer's own
TP coordinates, ``[num_embeddings_per_partition]`` as the one output partition
and ``num_embeddings_padded`` as the output size, the wire blob carries no
``output_dim`` (so ``VocabParallelEmbedding.weight_loader`` copies it whole to
every rank, as ``LinearBase`` does), and the shard plan cuts each rank's rows.

A checkpoint that does not declare the head gets ``None`` from
``TesseraConfig.get_quant_method`` exactly as before, so vLLM serves it BF16
with ``UnquantizedEmbeddingMethod``.  This module is reached only when the
checkpoint declares one.

Refused by name, before a weight is created:

* **A family no test has served as a head.**  ``HEAD_FAMILIES`` holds the
  families whose method has been driven through a ``ParallelLMHead``-shaped
  layer (``tests/test_serving_head_route.py``); a family is added there with
  its test, not assumed.
* **A structure other than dense.**  A head is one module of one role.
* **A padded or extended vocabulary.**  The declared rows must equal the
  head's ``org_vocab_size`` and ``num_embeddings_padded``, with no added
  (LoRA) vocabulary.  vLLM pads the last rank's shard with rows the checkpoint
  does not hold; the wire would have to carry them, and nothing writes that
  yet.  GLM-5.3 (154880 rows) is unpadded at TP1 and TP2.
* **A tied head** (``tie_weights``).  A tied model's head IS its input
  embedding, which stays BF16; a declared head is its own wire, and serving
  one beside a tie would answer with bytes the model does not use.

``LogitsProcessor`` refuses a ``head_dtype`` other than the model's for any
quantized head itself, so that case needs no check here.
"""
from __future__ import annotations

from typing import Mapping

from .scheme import STRUCTURE_DENSE, TESSERA_FP8

#: The families whose dense method has been served through a head-shaped
#: layer by a test.  A family is added here WITH that test.
HEAD_FAMILIES = (TESSERA_FP8,)


def _refuse_tie(prefix: str):
    def tie_weights(layer, embed_tokens):  # noqa: ARG001 -- vLLM's signature
        raise ValueError(
            f"tessera target {prefix!r}: the model ties its LM head to the input embedding, "
            "but the checkpoint declares the head as its own Tessera wire. A tied head is the "
            "embedding, which a Tessera checkpoint passes through; declare no head for a tied "
            "model.")
    return tie_weights


def require_head_geometry(scheme: Mapping, prefix: str, layer) -> None:
    """Refuse a head this route cannot serve (see the module docstring)."""
    family = scheme.get("family")
    if family not in HEAD_FAMILIES:
        raise ValueError(
            f"tessera target {prefix!r} is the LM head ({type(layer).__name__}) at family "
            f"{family!r}; a head is served only at {list(HEAD_FAMILIES)}, the families a test "
            "has driven through a head-shaped layer.")
    structure = scheme.get("structure", STRUCTURE_DENSE)
    if structure != STRUCTURE_DENSE:
        raise ValueError(
            f"tessera target {prefix!r} is the LM head, and its scheme declares structure "
            f"{structure!r}; a head is one dense module.")
    try:
        org = int(layer.org_vocab_size)
        padded = int(layer.num_embeddings_padded)
        total = int(layer.num_embeddings)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"tessera target {prefix!r}: the LM head layer {type(layer).__name__} carries no "
            "vocabulary geometry (org_vocab_size, num_embeddings_padded, num_embeddings) "
            "before its quant method is chosen, so its padding cannot be checked.") from exc
    rows = int(scheme.get("rows", -1))
    if total != org or padded != org or rows != org:
        raise ValueError(
            f"tessera target {prefix!r}: the checkpoint declares a {rows}-row head, and vLLM "
            f"builds it with org_vocab_size={org}, num_embeddings={total} and "
            f"num_embeddings_padded={padded}. A Tessera head is served only unpadded and "
            "without added vocabulary: the padded rows would have to be on the wire, and "
            "nothing writes them.")


def build_tessera_head_method(scheme: Mapping, prefix: str, mode: str, layer):
    """The Tessera method for a declared ``ParallelLMHead``."""
    from .lane import build_tessera_method
    from .moe_route import _bind_module_prefix

    require_head_geometry(scheme, prefix, layer)
    method = build_tessera_method(scheme, prefix, mode)
    # The route trace names a module by ``layer.prefix`` and
    # ``VocabParallelEmbedding`` stores none; without it every head dispatch is
    # counted unnamed and per-module qualification fails.
    _bind_module_prefix(layer, prefix)
    method.tie_weights = _refuse_tie(prefix)
    return method
