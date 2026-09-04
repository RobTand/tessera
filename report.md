# tessera#99 — the exporter wrote wires the runtime never routes

Branch `muse/ts-99-glmattn`, rebased onto master `5970451`. Not pushed, not merged.

## 0. The headline

**PENDING RECEIPT — §1.2 is not yet filled in.** From source read in the pinned
image, the load is *predicted* to fail loudly, from `params_dict[name]` at
`vllm/models/glm5next/nvidia/model.py:904` hitting an unmapped `...wire_bytes`
key. That is a reading, not an attestation, and it is written here as one; the
verdict in this line is not final until §1.2 carries the log.

**What IS attested, and is the more alarming half:** the belt that should have
caught the *other* side of the defect is disabled by design. The generic "weights were not initialized from checkpoint"
check — the one that would catch the *missing* `o_proj.weight` — does **not**
fire: `default_loader.py:456-464` adds every parameter of any module whose
`quant_method` has `process_weights_after_loading` to `loaded_weights`, and
`UnquantizedLinearMethod` has it (verified in the image, §1.3). So the deleted
weight is invisible; only the extra key is seen.

**The byte cost of the fix, stated loudly: on the full GLM-5.3-Flash, 335 of 427
planned dense modules leave the plan — 6.21 G of 7.74 G dense-body params
(80.2%). Priced against a 4.0-bpp body, that is +0.2331 bpp and +8.669 GiB
(+5.83%) once routed experts are exportable, and +9.62 bpp / +240% on the dense
body alone today.** §3.

## 1. What actually happens on a load

### 1.1 The artifact

`/home/rob/tessera-runs/ts99/glm4l-8e-attn`, exported from an 8-expert cut of
`/mnt/shared/models/GLM-5.3-Flash-4layer` (identical module tree; the 45 GB
full-4layer export is available on request but says nothing extra). Three
tensors carry a wire, chosen to separate the two verdicts from the control:

| tensor | census verdict |
|---|---|
| `...layers.0.self_attn.o_proj.weight` | `never_offered` (KDA layer) |
| `...layers.1.self_attn.indexer.wq_b.weight` | `never_offered` (MLA indexer) |
| `...layers.1.mlp.shared_experts.down_proj.weight` | `offered` — the control |

The other 46 body tensors pass through. The export was taken with
`--allow-unrouted`, i.e. the new gate was deliberately overridden, because the
whole point is to observe the defect the gate now prevents.

### 1.2 The load

PLACEHOLDER_LOAD

### 1.3 The belt that does not fire

Read in `prismaquant/glm53-mia-sm121:487ecf187`
(`sha256:75ea13eda532280afb4a829ab13eb572a4be49cbb47ca0a02a484a98e476ef69`, `vllm 0.1.dev20051+g487ecf187`):

```
# vllm/model_executor/model_loader/default_loader.py
456                 has_postprocess_quant = getattr(
457                     quant_method, "process_weights_after_loading", None
458                 )
461                 if has_online_quant or has_postprocess_quant:
462                     for param_name, _ in module.named_parameters():
463                         full_name = f"{name}.{param_name}" if name else param_name
464                         loaded_weights.add(full_name)
465             weights_not_loaded = weights_to_load - loaded_weights
466             if weights_not_loaded:
467                 raise ValueError(
468                     "Following weights were not initialized from "
469                     f"checkpoint: {weights_not_loaded}"
```

and in the same image:

```
>>> from vllm.model_executor.layers.linear import UnquantizedLinearMethod
>>> bool(getattr(UnquantizedLinearMethod(), "process_weights_after_loading", None))
True
```

So for *every* Linear the runtime built unquantized — which on GLM is every
attention projection — the missing-weight check is suppressed. Only the
unexpected-key path remains, and that one lives in the model class, not in the
loader:

```
# vllm/models/glm5next/nvidia/model.py, load_weights
904                     param = params_dict[name]
```

with `continue`s above it for biases, kv-scales, PP-missing parameters and
`mlp.experts.` names — none of which match `.wire_bytes`.

## 2. What was attested, and from which image

Everything below is **observed in** `prismaquant/glm53-mia-sm121:487ecf187`
(image id `sha256:75ea13eda532280afb4a829ab13eb572a4be49cbb47ca0a02a484a98e476ef69`, `vllm 0.1.dev20051+g487ecf187`), never read
off the issue.

### 2.1 The mechanism: a construction census

`tools/tessera_construction_census.py` builds the model **the way the loader
does** — `initialize_model(vllm_config=...)` on the `meta` device, inside
`set_current_vllm_config`, with distributed and model-parallel initialised —
having installed a probe `QuantizationConfig` whose `get_quant_method` records
every prefix it is asked about. Then it walks `named_modules()` and, for every
`LinearBase`, records whether the probe was asked. Nothing reads source; nothing
keeps a roster.

It writes a receipt (`tessera.construction-census.v1`) stamping the image, the
image id, the vLLM version, the model class, `packed_modules_mapping`,
`hf_to_vllm_mapper.get_unstacked_mapper()`, `supports_quant`, and one row per
normalised Linear prefix pattern.

### 2.2 What the census found on GLM-5.3-Flash

24 Linear patterns. **4 offered, 20 never offered.**

```
OFFERED language_model.model.layers.*.mlp.down_proj                | RowParallelLinear
OFFERED language_model.model.layers.*.mlp.gate_up_proj             | MergedColumnParallelLinear
OFFERED language_model.model.layers.*.mlp.shared_experts.down_proj | RowParallelLinear
OFFERED language_model.model.layers.*.mlp.shared_experts.gate_up_proj | MergedColumnParallelLinear
never   language_model.model.layers.*.mlp.gate                     | GateLinear
never   language_model.model.layers.*.self_attn.f_b_proj           | ColumnParallelLinear
never   language_model.model.layers.*.self_attn.fused_qkv_a_proj   | DeepSeekV2FusedQkvAProjLinear
never   language_model.model.layers.*.self_attn.g_b_proj           | ColumnParallelLinear
never   language_model.model.layers.*.self_attn.in_proj_qkvbfg_a   | _Glm5NextMergedColumnParallelLinear
never   language_model.model.layers.*.self_attn.indexer.wk_weights_proj | MergedColumnParallelLinear
never   language_model.model.layers.*.self_attn.indexer.wq_b       | ReplicatedLinear
never   language_model.model.layers.*.self_attn.k_conv1d           | ColumnParallelLinear
never   language_model.model.layers.*.self_attn.kv_b_proj          | ColumnParallelLinear
never   language_model.model.layers.*.self_attn.o_proj             | RowParallelLinear
never   language_model.model.layers.*.self_attn.q_b_proj           | ColumnParallelLinear
never   language_model.model.layers.*.self_attn.q_conv1d           | ColumnParallelLinear
never   language_model.model.layers.*.self_attn.v_conv1d           | ColumnParallelLinear
never   visual.blocks.*.attn.proj | visual.blocks.*.attn.qkv | visual.blocks.*.mlp.down_proj
never   visual.blocks.*.mlp.gate_up_proj | visual.merger.{down_proj,gate_up_proj,proj}
```

Three corrections to the issue's reading of the source, all measured:

* **`indexer.wq_b` is NOT offered.** The issue says `wq_b` "keeps the config"
  beside `wk_weights_proj`. On this build it is a `ReplicatedLinear` built
  unquantized like the rest. Source-reading got this one backwards; the census
  does not.
* **The vision tower is built and never offered** (all 7 patterns). Any design
  that assumes the plugin gets a chance to refuse the tower is wrong on GLM.
  This bears on #86 — see §6.
* **`lm_head` does not appear at all**: it is a `ParallelLMHead`, not a
  `LinearBase`, so the Linear census is silent about it by construction. That is
  an honest gap, not a clearance.

Control: `Qwen3ForCausalLM` on the pinned stock image — whose `image_id` in the
receipt, `sha256:61fc8a896b0a...`, is exactly the digest #100 pinned as
`versions.attested_on.image` — **4 patterns, 4 offered, 0 never offered.** That is exactly why this never bit — every Tessera artifact
served to date is Qwen.

### 2.3 How the fact travels (principle 14)

Not as a roster in the exporter. `runtime_contract.json` gains a top-level
`construction` block (**contract v11**; #100 took v10 for the image-digest pin
while this was in flight), whose rows are **generated** from the receipts by
`contract.construction_entry_from_receipt` and **re-derived** by
`tests/test_serving_construction.py`, so the table cannot drift from the
observation. `tools/tessera_update_construction_block.py` regenerates it
(`--check` asserts no drift); it splices rather than reformats, so it does not
collide with concurrent edits to the same file.

The checkpoint→vLLM naming bridge travels with it:
`hf_to_vllm_mapper_unstacked` (the very table `configure_quant_config` hands
this plugin) and `packed_modules_mapping` are published in the row, and
`contract.vllm_module_name` applies them — including vLLM's `None` spelling for
"drop this weight", which is now the same refusal `apply_vllm_mapper` raises at
load, taken at export instead.

## 3. What the fix costs, in bytes

`experiments/export_tessera_serving.py` gains a gate placed **before the first
encode**, on the *module* names (what the runtime builds and what
`config_groups` declares), with three verdicts and a refusal that names the
prefixes:

* `never_offered` — built with `quant_config=None`.
* `absent` — the runtime builds no module of that name. This is what a fused
  role named at the wrong seam looks like: the checkpoint has
  `self_attn.{q,k,v,b,f_a,g_a}_proj` where vLLM builds one
  `in_proj_qkvbfg_a`, and `{q_a,kv_a}_proj_*` where it builds one
  `fused_qkv_a_proj`.
* `uncensused` — no census covers the architecture. An honest gap, and it
  refuses, because an unknown route is an unpriced wire.

Two escapes, both stamped into the manifest's `serving_gate`:
`--passthrough-unrouted` (drop them to source precision — the safe direction)
and `--allow-unrouted` (write it anyway — how §1's artifact was produced).

### The numbers

Computed by running the exporter's own planner (`quantizable` → fused grouping →
`unrouted_modules`) over the real checkpoints. Not an estimate.

**Full `GLM-5.3-Flash`** — 427 dense modules planned today:

| verdict | modules | params |
|---|---|---|
| `offered` | 92 | 1 535.1 M (19.8%) |
| `never_offered` | 150 | 2 596.3 M (33.5%) |
| `absent` | 185 | 3 609.2 M (46.6%) |
| **leaves the plan** | **335** | **6 205.5 M (80.2%)** |

Which patterns, exactly (checkpoint names, count of modules):

```
absent         x   1  ...layers.*.eh_proj                        (MTP; see the caveat in §6)
absent         x  34  ...layers.*.self_attn.b_proj
absent         x  34  ...layers.*.self_attn.f_a_proj
absent         x  34  ...layers.*.self_attn.g_a_proj
absent         x  34  ...layers.*.self_attn.qkv_proj             -> vLLM builds in_proj_qkvbfg_a
absent         x  12  ...layers.*.self_attn.q_a_proj             -> vLLM builds fused_qkv_a_proj
absent         x  12  ...layers.*.self_attn.kv_a_proj_with_mqa   -> same
absent         x  12  ...layers.*.self_attn.indexer.wk           -> vLLM builds indexer.wk_weights_proj
absent         x  12  ...layers.*.self_attn.indexer.weights_proj -> same
never_offered  x  46  ...layers.*.self_attn.o_proj               (every layer, both attention families)
never_offered  x  34  ...layers.*.self_attn.f_b_proj
never_offered  x  34  ...layers.*.self_attn.g_b_proj
never_offered  x  12  ...layers.*.self_attn.q_b_proj
never_offered  x  12  ...layers.*.self_attn.kv_b_proj
never_offered  x  12  ...layers.*.self_attn.indexer.wq_b
offered        x  43  ...layers.*.mlp.shared_experts.gate_up_proj
offered        x  43  ...layers.*.mlp.shared_experts.down_proj
offered        x   3  ...layers.*.mlp.gate_up_proj               (the dense MLP layers)
offered        x   3  ...layers.*.mlp.down_proj
```

Priced at 4.0 bpp for what stays and 16 bpp for what reverts to BF16:

* **dense body only** (today, routed experts still BF16): 4.0000 → **13.6202
  bpp**, 3.604 → 12.273 GiB, **+8.669 GiB, +240.50%**.
* **body + routed experts** (once #5 lands and the 311.7 G expert params carry
  4.0 bpp): 4.0000 → **4.2331 bpp**, 148.729 → 157.398 GiB, **+8.669 GiB,
  +5.83%**.

**`GLM-5.3-Flash-4layer`** — 38 modules: 8 offered, 14 `never_offered`, 16
`absent`. Dense body only 4.0000 → 12.3786 bpp (+0.732 GiB, +209.47%);
body + routed 4.0000 → 4.2796 bpp (+0.732 GiB, +6.99%).

### This is not a regression, and the framing matters

The pre-gate "4.0 bpp" label on those bytes was **fiction**: 80.2% of the dense
body was being encoded into wires the runtime does not read, deleting the
weights it does. The gate does not make an artifact bigger; it makes the
artifact's size *true*. What is genuinely lost is the option of quantizing GLM
attention at all on this runtime — and the honest reading of that is
**principle 9: the allocator wanting a rung the runtime does not back is the
runtime reporting a serving gap.** Closing it means a vLLM-side change to pass
`quant_config` into those constructors, not a producer-side change.

**One decision is yours:** whether `--passthrough-unrouted` becomes the default
so a GLM export succeeds-with-BF16-attention instead of refusing. It is a
one-line default flip. I left it refusing, because a silent 240% size change is
exactly the failure mode this issue is about.

**Existing artifacts affected: yes, one.**
`/mnt/shared/models/GLM-5.3-Flash-Tessera-E2M1K2-20260901` carries `.tessera`
wires on attention across all 46 layers — dead bytes, encode already paid. It
has no `quantization_config`, so it was never servable as a Tessera artifact
and nothing shipped from it; but the encode time and the size label are both
wrong, and it should not be used as a size reference.

## 4. Test evidence

Targeted, per the branch rule.

```
$ pytest -q -p no:randomly \
    tests/test_serving_construction.py tests/test_export_construction_gate.py \
    tests/test_serving_contract.py tests/test_serving_export_gate.py \
    tests/test_serving_attested_wire.py tests/test_serving_native_extensions.py \
    tests/test_serving_loader_gates.py tests/test_serving_dispatch.py \
    tests/test_serving_sharding.py tests/test_serving_tp_axes.py \
    tests/test_serving_moe_dispatch.py tests/test_route_census_regimes.py \
    tests/test_route_grid_and_census_guard.py tests/test_no_gridbook_import.py \
    tests/test_fused_member_rungs.py tests/test_serving_residency_modes.py
234 passed, 1 skipped in 110.35s
PYTEST_EXIT=0
```

That set is: the two files this branch adds, plus every test file that imports
what it changed (`tessera.serving.contract`). The one skip is pre-existing and
unrelated (a CUDA-gated case). No failures, so nothing needed re-running
against master.

A second, wider sweep over every file importing anything from
`tessera.serving` (7 more, mostly kernel/route tests that do not touch the
contract) found exactly one failure:

```
FAILED tests/test_runtime_image_pin.py::test_the_pin_is_the_contract_field_and_nothing_else_holds_it
AssertionError: the pin digest is copied into ['src/tessera/stock.py']; it
lives in runtime_contract.json and is read from there
```

**It fails identically on a pristine master checkout** (`1 failed, 11 passed`
both places), so master is red on it today and this branch does not touch it.
Someone put the pin digest back into `src/tessera/stock.py` after #100's test
landed. Not mine to fix — it is #100's invariant and #92's file.

That sweep: `1 failed, 126 passed, 14 warnings in 174.01s`, `PYTEST_EXIT=1`.

New tests and the pre-fix failure line each one asserts:

* `tests/test_serving_construction.py` (11) — the contract block is exactly
  re-derived from the committed receipts; receipts are stamped with an image id;
  normalisation collapses every repeat index; GLM attention classifies
  `never_offered`/`absent` and GLM MLP `offered`; Qwen is the 4-of-4 control; an
  uncensused architecture returns `None`; the validator refuses
  `offered ∩ never_offered`, a duplicate architecture row, and an unstamped
  `image_id`. Before the fix: the module, the block and the receipts did not
  exist — every one is a collection error.
* `tests/test_export_construction_gate.py` (5) — on a synthetic GLM-architecture
  checkpoint: the census answers *before* encoding; an uncensused architecture
  refuses; the export refuses naming `self_attn.o_proj`, `never_offered` and
  `quant_config=None`, and writes nothing; `--passthrough-unrouted` keeps only
  `mlp.down_proj` and preserves `self_attn.o_proj.weight`; `--allow-unrouted`
  reproduces the defect (weight gone, `self_attn.o_proj.wire_bytes` present) and
  stamps it. Before the fix, `unrouted_modules` does not exist in the exporter,
  so all five are import/collection errors on master — that is the pre-fix
  failure line, and re-running the file against master is not informative.

## 5. Off-task fixes (one line each)

* `contract.vllm_module_name` mishandled vLLM's `None` = "drop this weight"
  spelling in a `WeightsMapper` — would have raised a `TypeError` from inside a
  rename loop, naming nothing. Now the same refusal `apply_vllm_mapper` raises.
  Fixed; it is mine, one line of behaviour.
* The contract-block generator lived in a scratch file while a committed test
  said "regenerate it". Committed as `tools/tessera_update_construction_block.py`
  with `--check`. Fixed; mine.

## 6. Filed, not fixed

* **#86's premise is wrong for the GLM vision tower.** The tower is *built* and
  *never offered*, so the plugin's "declared no wire and not ignored" refusal
  cannot fire for it — the plugin is never asked. Not fixed here: `muse/ts-86-ignore`
  is live in the same exporter and this is its call to make.
* **The fused-seam names are wrong for GLM regardless of this issue.** `FUSED`
  produces `qkv_proj` on KDA layers where vLLM builds one `in_proj_qkvbfg_a`;
  the same for `fused_qkv_a_proj` and `indexer.wk_weights_proj`. Today all are
  refused as `absent`, so nothing ships wrong — but the fusion table is naming a
  seam the runtime does not have. Not fixed: #86 and #5 are both live in that
  region.
* **The census is keyed by architecture, not by (architecture, image).** The
  manifest stamps the census's image; nothing yet compares it to the image a
  serve actually runs. That is principle 14's second leg and it is unbuilt.
  Not fixed: it is a decision about where the comparison belongs (producer
  preflight vs `validate_native_export`), and it is #100-adjacent.
* **The 4-layer census is applied to 46-layer models by name normalisation.**
  MTP's `eh_proj` (`mtp.py:44` passes `quant_config` through) is a module the
  4-layer cut never builds, so it classifies `absent` and would be refused —
  fail-closed, and possibly wrong. A full-model census fixes it; the bytes are
  negligible but the §3 numbers inherit the caveat.
* **The exporter's `MOE_ROUTER` comment block is a prose attestation** that the
  census now subsumes. Left alone: #5 is live there.

## 7. Consultations

One `advisor()` call before committing to the approach and one before declaring
done. No `fable-*` consultation was needed: the hard part was measurement
plumbing, not judgment.
