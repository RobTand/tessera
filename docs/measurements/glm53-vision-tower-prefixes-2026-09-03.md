# What the pinned runtime builds for GLM-5.3-Flash's vision tower

**Date** 2026-09-03 · **Issue** #86 · **Image** `prismaquant/glm53-mia-sm121:487ecf187`
· **Config source** `/mnt/shared/models/GLM-5.3-Flash-4layer` (read-only, header/config only)
· **Script** `glm53-vision-tower-prefixes-2026-09-03.probe.py` (beside this file)

## Why this exists

`experiments/export_tessera_serving.py` writes every non-body tensor at source
precision and must name each of those modules in `quantization_config.ignore`,
because the plugin refuses a Linear it can neither serve nor ignore
(`src/tessera/serving/config.py`). Naming them requires knowing **the string
vLLM dispatches on**, and CLAUDE.md principle 14 says a claim about what a
serving runtime does is attested, not asserted. So it was constructed.

## What was run

`Glm5NextVisionTransformer` is instantiated twice on `torch.device("meta")`
inside the pinned image, under `set_current_vllm_config(...)` and a 1-rank gloo
`init_distributed_environment`: once with a `Recorder(QuantizationConfig)` whose
`get_quant_method(layer, prefix)` appends `(prefix, type(layer).__name__,
isinstance(layer, LinearBase))`, and once with `quant_config=None`, which is
what `Glm5NextForConditionalGeneration` actually passes on this build.

```
docker run --rm --network host \
  -v <this dir>:/probe \
  -v /mnt/shared/models/GLM-5.3-Flash-4layer:/models/GLM-5.3-Flash-4layer:ro \
  -e TRITON_CACHE_DIR=/tmp/triton --entrypoint python3 \
  prismaquant/glm53-mia-sm121:487ecf187 /probe/glm53-vision-tower-prefixes-2026-09-03.probe.py
```

**This is a module construction, not a checkpoint load, and that is sufficient
for the question asked.** Both the prefix strings and the `quant_method` object
are fixed in `__init__` — the prefix is the literal the parent passes and the
method is chosen there once — so no weight file can change either. What a load
would add (that the tensors fit) is a different question.

## Verbatim result

- **With a quant config**: 99 `LinearBase` modules, 99 offers, `NON LINEAR OFFERS: []`
  — every offer is a `LinearBase`, and nothing else is offered.
- **With `quant_config=None`** (the live path): `OFFERS WITH quant_config=None: []`,
  `LINEARBASE COUNT (none case): 99`, `QUANT METHODS (none case):
  ['UnquantizedLinearMethod']`.

So on **this** build the #86 refusal does not fire for GLM's vision tower: those
99 Linears are built with `UnquantizedLinearMethod` in `LinearBase.__init__`
without `get_quant_method` ever being called. The completeness contract is still
the thing to satisfy — a runtime that threads its quant config into the tower
(and the code has the branch for it) refuses the whole checkpoint at load.

## The finding that changed the fix

The module tree and the dispatch string **disagree** for merged qkv:

| what | value |
|---|---|
| tensor on disk | `model.visual.blocks.0.attn.qkv.weight` |
| `named_modules()` attribute path | `blocks.0.attn.qkv` |
| prefix offered to `get_quant_method` | `visual.blocks.0.attn.qkv_proj` |

because `Glm5NextVisionAttention` builds it at
`f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv"`
(`models/glm5next/nvidia/multimodal.py:167`). Which spelling exists depends on
whether the runtime passes a quant config, which the producer cannot know, so
`ignored_modules` emits **both**; an ignore entry for a module nobody builds is
never looked up, and a missing one is a load-time refusal. This is the one
conditional prefix in the tower — `gate_up_proj`, `down_proj`, `proj`, and the
merger are unconditional (`multimodal.py:107,115,174,318,327,335`), which is why
the fused `gate_proj|up_proj -> gate_up_proj` rule needs no alias.

## Names survive the mapper

`ignore` entries are prefix-mapped as well as looked up. vLLM hands the config
`hf_to_vllm_mapper.get_unstacked_mapper()`; `WeightsMapper._map_name` returns
`None` only for a rule whose value is `None`, and neither GLM mapper has one —
`Glm5NextForConditionalGeneration` inherits three prefix rules
(`model.visual.` -> `visual.`, `model.language_model.` -> `language_model.model.`,
`lm_head.` -> `language_model.lm_head.`) and the tower's own mapper is
`orig_to_new_stacked` only. Checked by reading `model_executor/models/utils.py`
and `models/glm5next/nvidia/{model,multimodal}.py` in the image. Were a rule to
drop one, `TesseraConfig.apply_vllm_mapper` raises rather than silently thins the
list, so that case is loud.

## Scope

Attested for `Glm5NextForConditionalGeneration` on this image only. GLM
checkpoints here carry no `mtp.*` tensors. Qwen3.8-Flash-Next's runtime class is
absent from this image, so nothing is attested about it — the rule names its
non-body Linears anyway, which is the cheap direction of the error.
