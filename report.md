# #86 — a Linear outside the decoder body is passed through but never named

## What the pinned runtime actually does (attested, not asserted)

Read out of the pinned image `prismaquant/glm53-mia-sm121:487ecf187` (file
reads inside the container; no load, no serve — the image was never started
with a GPU):

* **The vision tower is built as `LinearBase`, and the plugin is never asked
  about it.** `Glm5NextForConditionalGeneration` constructs
  `Glm5NextVisionTransformer(..., quant_config=None, ...)`
  (`models/glm5next/nvidia/model.py:1082`, with the comment that the tower is
  BF16 in the checkpoint), and `LinearBase.__init__` takes
  `UnquantizedLinearMethod()` in its `if quant_config is None` branch
  **without calling `get_quant_method`** (`model_executor/layers/linear.py`).
  So on this build the #86 refusal does **not** fire for GLM's vision tower:
  the modules exist (`QKVParallelLinear`, `MergedColumnParallelLinear`,
  `RowParallelLinear`, `ColumnParallelLinear` — `multimodal.py:102-115`,
  `:160-174`, `:312-335`), they are `LinearBase`, and nothing offers their
  prefixes to `TesseraConfig`.
* **No other case is attested to fire either.** `/mnt/shared/models/GLM-5.3-Flash`
  and `-4layer` carry no `mtp.*` tensors at all (the full model's MTP layer is
  index 45 with `num_hidden_layers=45`, so it *matches* `BODY_LAYER` and is
  planned as body). `/mnt/shared/models/Qwen3.8-Flash-Next` is
  `Qwen4ExpForConditionalGeneration`, which this image does not carry — its
  runtime is unattested here, so its `mtp.fc_embedding` / `mtp.layers.0.*`
  Linears are *unknown*, not *refused*. For the record, the closest analogue in
  the image builds the MTP `fc` **with** the real quant config
  (`models/qwen3_next_mtp.py:75-82`), which is the shape that would reach the
  refusal.

**So the honest severity is:** the code fault is real and settled by reading
(no non-body tensor could reach `ignore`), and the consequence is *latent* on
every checkpoint this box can attest. The fix is completeness of the rule the
plugin's contract states, not a repair of a refusal seen in the wild.

A latent runtime quirk found on the way, recorded because it will bite whoever
first serves a vision tower with a quant config: `Glm5NextVisionAttention`
builds its merged projection at `f"{prefix}.qkv_proj" if quant_config else
f"{prefix}.qkv"` (`multimodal.py:167`). The checkpoint name is `attn.qkv`, so
the ignore entry derived from the bytes is right only while the tower is built
unquantized. No rule derives `qkv_proj` from `qkv`; naming both would be
guessing at a runtime fact, which principle 14 forbids.

## What changed

`experiments/export_tessera_serving.py`

* **`ignored_module(tensor_name, shape)`** — one rule, applied in the shard
  loop to *every* tensor written at source precision, body or not. A fused role
  names its fused parent (vLLM builds one method per fused module, in the
  vision tower exactly as in the body); a routed leaf and a packed stack name
  the `FusedMoE` prefix; a conv or a 1-D weight names nothing. The three
  `BODY_LAYER`-gated sources are gone as sources: the two post-loop expert
  loops were the same rule restated, and what replaces them is a check that
  every plan-time passthrough came out named — so the plan-time and write-time
  mechanisms cannot drift apart in silence.
* Prose fixed on sight: the `BODY_LAYER` comment and the module docstring both
  said the tower "stays BF16 by the same rule rather than by a second exclusion
  list", which is true of the encode and false of the naming;
  `docs/tessera-serving-and-moe-contract.md:§9.2` carried the same sentence.

## Evidence

* **Fails first.** `tests/test_export_ignore_completeness.py` was committed red
  (`0960a01`) against the unmodified exporter:

  > the exporter copies these Linears through at source precision and never
  > names them, so the plugin refuses the checkpoint at load:
  > `['model.visual.blocks.0.attn.qkv', 'model.visual.blocks.0.attn.proj',
  > 'model.visual.blocks.0.mlp.gate_up_proj', ...]`

  Four tests: the rule itself, an end-to-end export with a vision tower
  (`--layers 0`, host-safe — no encode), a CUDA arm that names the tower beside
  a real body encode, and the empty-plan refusal below.
* **On the real checkpoint.** The rule run over every tensor of
  `/mnt/shared/models/GLM-5.3-Flash-4layer` (shapes read from the safetensors
  headers; nothing written) names **101 distinct non-body modules** — 24×
  `attn.qkv`, `attn.proj`, `mlp.gate_up_proj`, `mlp.down_proj`, the four
  merger modules, `lm_head`, `model.language_model.embed_tokens` — and the only
  rank≥2 tensors it declines to name are the six `*_conv1d` and the two vision
  convs (`patch_embed.proj` rank 5, `downsample` rank 4). A 45 GB passthrough
  copy was not written: it would prove nothing the header read does not.
* **Suite.** master `82cdf51` vs this branch — see the table at the end of this
  file (both runs serialised through `gpulock.sh`).

## Off-task fixes (separate commits, per the new rule)

* `c095d4f` — **an export that planned nothing was written and reported
  success.** With every dense weight passed through, `config_groups` comes out
  empty and `TesseraConfig.from_config` refuses exactly that at load. Same
  family as #86. Now a refusal at plan time; `--layers 0` stays legal because
  that is how a passthrough copy is asked for deliberately.
* `3421d30` — **the rule now reads a packed stack that carries no `.weight`
  suffix**, a fact read off `muse/ts-5-moe-route` (their branch was read, never
  edited). Without it the merge of the two branches would silently drop those
  stacks from `ignore`.

## Filed, not fixed

* **#99** — GLM's attention is built with `quant_config=None` on both layer
  kinds (`model.py:331` for MLA, `kda.py:171-174` for KDA, plus the indexer's
  `wk_weights_proj` at `attention.py:263`), so every attention wire the
  exporter writes by default is never executed *and* the `weight` the runtime
  builds is no longer in the checkpoint. Not fixed here because any fix decides
  which Linears a GLM export may quantize — bytes and bpp move — and because
  principle 14 says that runtime fact must arrive attested, not as a roster
  inside the exporter. The issue says so explicitly.

## Coordination

This diff touches the exporter's shard-loop passthrough branch and **deletes**
the two post-loop expert loops (`~784-802` on master) — which is exactly the
region `muse/ts-5-moe-route` rewrites for #5. Their version restates the two
loops as set comprehensions off the patterns; mine removes them in favour of
the one rule, having absorbed their `.weight`-suffix finding. The merge is a
deliberate resolution, not a textual one: **keep `ignored_module`, drop both
loops.** Their `quantizable()` and refusal changes do not overlap.
