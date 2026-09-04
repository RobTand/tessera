# The exporter writes a routed-MoE stack, and the plugin reads what it wrote

**Date** 2026-09-04 · **Box** sparky (GB10, sm_121) · **Issue** #5 ·
**Code** `experiments/export_tessera_serving.py`,
`experiments/moe_route_load_probe.py`, `tests/test_export_moe_write.py`

Before this, `tessera.serving.moe_route` had been loaded and executed on the
pinned runtime (`tessera-moe-route-load-2026-09-04.md`), the sidecar shape and
the parameter layout existed, and **no exporter wrote any of it**. A route that
decodes wires nobody can produce serves nothing, so this is the half that was
missing: what the exporter writes, what it refuses first, and the seam between
the two.

## 1. What is closed, and what is not

| Issue #5 item | State |
|---|---|
| 1. The plugin has no MoE route | closed before this pass (`moe_route.py`, load-and-execute receipt) |
| 2. The exporter mis-plans this model silently | closed on master (`ee54038`/`d82b29e`/`3caa49c`, then `7375914`/`2975448`) |
| 3. The fused parameter layout | closed on master (`c9561e3`, `moe_layout.py`) |
| 4. Both source layouts | **unpacked: closed here. packed 3-D: still refused** — §4 |
| 5. Contract rows + a served census and KL | **OPEN.** No serve was run; §5 says why and what it costs |

**No `routed_moe` cell was added to `runtime_contract.json`,** and none should
be until a served census and KL cover it. `structures` stays `["dense"]`.

## 2. What the exporter now writes

A `--plan-json` entry keyed `<moe>.experts` — the **stack**, because vLLM
builds one method per `RoutedExperts` module and the sidecar declares one
scheme for it — puts every expert of that stack on one rung. The exporter then
writes:

* one `tessera.fused` container per expert per **projection**, under
  `<moe>.experts.{e}.{proj}.wire`. That is the granularity the checkpoint's
  tensors, the runtime's shard ids (`w1`/`w3`/`w2`) and `moe_layout`'s cells
  already have, and the suffix is what makes
  `RoutedExperts.build_expert_params_mapping` route it to `w13_wire`/`w2_wire`;
* one `config_groups` entry declaring `structure: routed_moe`, the expert
  count, and the two groups `w13` (gate then up) and `w2` (down), each with its
  geometry, roles, rung and **`wire_stride`**.

`wire_stride` is the **maximum** over that group's blobs, derived at write time
rather than passed in: the blob length follows the data (the manifest's
exact-ratio `global_scale` rides as a varint whose width follows its value), so
there is no number a caller could supply that the bytes could not contradict —
and `moe_layout.unpack_moe_wires` refuses a stride that is not what the loaded
lengths imply, which is the same check from the loader's side.

The scheme dict is put through `scheme.validate_tessera_moe_scheme` — the exact
function `TesseraConfig` calls — before `config.json` is written. The reader is
the gate, so the writer is held to it at write time rather than at load.

## 3. What is refused before the first encode

A GLM routed stack is 864 units and ~75 minutes of GPU, so every disagreement
between the checkpoint and the route is asked at plan time. Measured on the CPU
against a miniature of the real checkpoint (`tests/test_export_moe_write.py`,
and reproduced by hand at the shapes below):

| Plan | Refusal |
|---|---|
| a routed **leaf** (`...experts.0.gate_proj.weight`) | names the stack spelling that works, instead of the old "the write half does not exist" |
| the stack at `E2M1x2` | `scheme.MOE_BUILDERS names ['TESSERA_FP8']`, with the NVFP4 clamp receipt cited |
| expert 1 absent | `missing expert(s) [1] of 4` — the parameter is `[E, ...]` and the loader writes row `expert_id`, so a gap is a row of zeros served as an expert |
| expert 2 without `up_proj` | `w13 is the gate/up PAIR, so a stack missing one half has no second half for the tile` |
| one expert at a different shape | `One stack is one tile, so one shape` |
| intermediate size 40 (not a whole tuple) | `a routed stack cannot be half passed through, because vLLM builds ONE method per stack` |
| the stack plus `--layers 1` | the bound stops before the planned stack; refused rather than silently skipped |
| the stack plus `--stock-twin` | there is no per-channel FP8 expert twin writer, so the twin would be a checkpoint with no experts in it |

The construction gate covers the stack too. `classify_construction` now reads
the census receipt's `offered_non_linear` list as well as `offered`: a
`RoutedExperts` stack is not a `LinearBase`, so the census records it there, and
a classifier reading only the first list called the one module the expert route
exists for `absent` — refusing it while the same receipt said, in the same
file, that the runtime asks this plugin about it. On the real
GLM-5.3-Flash-4layer all three stacks resolve `offered` against
`language_model.model.layers.*.mlp.experts`.

## 4. The real model plans, and what it will cost

CPU-only, reading shapes off the checkpoint index — no encode:

| Fact | Value |
|---|---|
| routed expert leaves | 2592 |
| stacks | 3 (`layers.{1,2,3}.mlp.experts`), 288 experts each |
| per stack | 864 units; `w13` 4096x4096, `w2` 4096x2048 |
| construction verdict, all three | `offered` |
| routed expert parameters | 21,743,271,936 (21.74 G) |
| dense modules planned | 38, of which **30 are unrouted** (16 `absent`, 14 `never_offered`) — the KDA/MLA/indexer attention, so a real run passes `--passthrough-unrouted` |

At the held-box E4M3 whole-expert rate of 1.611 Mparam/s
(`experiments/results/moe_encode_rate_profile_exclusive.json`) that is **~75
minutes for one stack and ~3.75 h for all three**, and 1.91x that on a shared
box. `experiments/moe_export_glm_4layer.sh` is the driver;
`experiments/moe_stack_plan.py` builds the plan from the checkpoint so a layer
index cannot be typed wrong.

**The packed 3-D source layout is still refused**, and the refusal now names
the true reason. It is **two** unattested conventions, not one:

1. **Orientation.** `[E, A, B]` is ambiguous on its face.
   `packed_expert_orientation` reads it off
   `hidden_size`/`moe_intermediate_size` where the dims decide — and on
   GLM-5.3-Flash `hidden_size == 2 * moe_intermediate_size`, so a packed
   `gate_up_proj` is square and no dim comparison orients it.
2. **The gate/up split.** A packed `gate_up_proj` holds both halves on one
   axis, and whether they are chunked or interleaved is the producing library's
   convention, which the tensor does not state.

Either guessed wrong transposes or interleaves every expert in silence. GLM's
own source is unpacked, so nothing on the target model is blocked by this;
what would settle (2) is a served A/B on a packed-source model, which needs a
packed-source model this repo can serve.

## 5. What was NOT done, and why

* **No served census and no KL.** Item 5 stays open. Both GB10 GPUs in the pool
  were held for the whole of this pass — sparky's two by other agents' jobs,
  sparklina's one by an out-of-pool encode — with ready items up to 2.9 h old,
  so the encode campaign in §4 was costed and scripted but not submitted. It is
  also more than an encode: no GLM artifact has ever been served in this repo
  (every served receipt here is Qwen3-0.6B), so the serve needs a GLM teacher
  dump on the pinned image before a student KL means anything.
* **The model-level load hop is still unmeasured.** The probe drives
  `RoutedExperts.load_weights`; in a serve
  `Glm5NextForConditionalGeneration.load_weights` runs first and decides what is
  delegated, and whether a `.wire`-suffixed expert tensor survives that hop is
  the first thing a served attempt would find out.
* **Memory at 288 experts is a design item, not arithmetic.**
  `process_weights_after_loading` runs after every weight has loaded, so each
  MoE layer holds its wires *and* its decoded tile at once — ~11 GB per GLM MoE
  layer, ~33 GB transient for the 4-layer cut, and it does not fit for full GLM.
* **The compiled forward, expert parallelism, TP inside an expert and
  `streamed`** are all still refused or untested, exactly as the load receipt
  says.

## 6. The control: nothing that was written before moves

`experiments/moe_plan_baseline.py` runs the exporter end to end on the CPU over
every expert layout a checkpoint can carry and digests what it wrote.
Master (`cc3eb27`) against this branch, same command, `--work` under
`/home/rob/tmp`:

**5 of 72 rows move.** Four are the refusal-message rows (the routed-leaf
refusal and the three packed-stack refusals). The fifth is
`export/dense/manifest`, whose `routed_moe` block gained `quantized_stacks`,
`quantized_source_tensors` and a three-valued `disposition`.

Everything that decides bytes is **unchanged**:
`export/dense/tensors` (sha256 over every written tensor's name and bytes),
`export/dense/quantization_config`, `export/dense/ignore`,
`export/dense/declared` and `export/dense/tensor_names` are all identical, as
are the six other cases' export rows. So a checkpoint written before this pass
is byte-identical under the same command line, and an unplanned stack is
untouched: source precision, named in `ignore` at the FusedMoE prefix.

*(The `dense` case is the only one of the seven that reaches an export at all;
the other six are construction-gate refusals, and their refusal text is
unchanged too.)*
