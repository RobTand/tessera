# #86 — a Linear outside the decoder body is passed through but never named

## What the pinned runtime actually does (attested, not asserted)

Two ways, both inside the pinned image `prismaquant/glm53-mia-sm121:487ecf187`:
file reads, and a **real construction** of the GLM vision tower on meta device
with a *recording* quant config (run on sparklina, CPU only, no GPU and no
serve — the tower is 99 Linears and needs no weights to answer this). The run
is durable, not just recounted here: script, `docker run` line and verbatim
output are committed at
`docs/measurements/glm53-vision-tower-prefixes-2026-09-03.md`. It is a module
construction rather than a checkpoint load, which answers the question asked
because both the prefix string and the `quant_method` are fixed in `__init__`
and no weight file can change either.

* **The vision tower is built as `LinearBase` — 99 of them — and the plugin is
  never asked about it.** Constructed with a recording config, all 99 offers
  arrive and every one is a `LinearBase`; constructed the way the model class
  actually builds it (`quant_config=None`), **zero** offers arrive and all 99
  take `UnquantizedLinearMethod`. The call site: `Glm5NextForConditionalGeneration` constructs
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

* **The names also survive the mapper.** `ignore` entries are prefix-mapped,
  not only looked up: vLLM passes `hf_to_vllm_mapper.get_unstacked_mapper()`,
  and `WeightsMapper._map_name` returns `None` only for a rule whose value is
  `None`. Neither GLM mapper has one (three prefix rules on the model class;
  `orig_to_new_stacked` only on the tower) — read in the image. And a name the
  mapper *did* drop would be a refusal, not a silent thinning:
  `TesseraConfig.apply_vllm_mapper` raises.

**So the honest severity is:** the code fault is real and settled by reading
(no non-body tensor could reach `ignore`), and the consequence is *latent* on
every checkpoint this box can attest. The fix is completeness of the rule the
plugin's contract states, not a repair of a refusal seen in the wild.

**The same construction found a hole in the first version of this fix.** The
recording run offers `visual.blocks.N.attn.qkv_proj`, not `.attn.qkv`:
`Glm5NextVisionAttention` picks the name at build time —
`f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv"`
(`multimodal.py:167`) — while the tensor on disk is `attn.qkv.weight`. So the
name derived from the bytes is correct only in the world where nobody asks, and
is the #86 refusal again in the world where somebody does. `MERGED_ALIASES`
now carries both spellings, with that construction as its attestation: an
ignore entry is only ever looked up, so a second name costs nothing and a
missing one is a load-time refusal.

## What changed

`experiments/export_tessera_serving.py`

* **`ignored_modules(tensor_name, shape)`** — one rule, applied in the shard
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

  Five tests: the rule itself, an end-to-end export with a vision tower
  (`--layers 0`, host-safe — no encode), a CUDA arm that names the tower beside
  a real body encode, both spellings of a packed stack, and the empty-plan
  refusal below.
  Re-checked against **pristine master** (`/home/rob/tmp/musefix/ts-86-base`,
  `82cdf51`) on 2026-09-03 by copying only the test file in and running it
  there: it fails on the unmodified exporter with all eight vision modules
  missing, and master's `ignore` carries the six body modules only. The same
  run also shows master writing a checkpoint with `"quantized_params": 0` and
  reporting success — the empty-plan hole fixed in `c095d4f`.
* **On the real checkpoint.** The rule run over every tensor of
  `/mnt/shared/models/GLM-5.3-Flash-4layer` (shapes read from the safetensors
  headers; nothing written) names **125 distinct non-body modules** over 350 tensors (101 before the
  merged-qkv alias commit added the second `qkv_proj` spelling) — 24×
  `attn.qkv` + `attn.qkv_proj`, `attn.proj`, `mlp.gate_up_proj`,
  `mlp.down_proj`, the merger modules, `lm_head`,
  `model.language_model.embed_tokens` — and the only
  rank≥2 tensors it declines to name are the six `*_conv1d` and the two vision
  convs (`patch_embed.proj` rank 5, `downsample` rank 4). A 45 GB passthrough
  copy was not written: it would prove nothing the header read does not.
* **Suite.** The branch suite was run once through `gpulock.sh`; any failure is
  re-run per file against pristine master (`/home/rob/tmp/musefix/ts-86-base`,
  `82cdf51`). Counts are in the summary handed back to the coordinator. A full
  master baseline was started and then **killed** on instruction — fifteen were
  running concurrently and had put sparky into swap.

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
