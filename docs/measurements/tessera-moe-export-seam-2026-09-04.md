# The exporter writes a routed-MoE stack, and the plugin reads what it wrote

**Date** 2026-09-04 · **Box** sparky (GB10, sm_121) · **Issue** #5 ·
**Code** `experiments/export_tessera_serving.py`,
`experiments/moe_write_readback_check.py`, `experiments/moe_route_load_probe.py`,
`tests/test_export_moe_write.py`

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
| 4. Both source layouts | **unpacked: written, read back (CPU), and loaded + executed on the GPU from the exporter's own bytes. packed 3-D: still refused** — §2, §4, §5 |
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

### It was run, and read back — **on the CPU**

`experiments/moe_write_readback_check.py`, through the pool on sparky
(`260e2bcba0d5`) and re-run directly with the exporter's own verify left on
(`/home/rob/tmp/wf5_verify_on.log`), no GPU either time
(`CUDA_VISIBLE_DEVICES=`), on a miniature of the real checkpoint: 4 experts,
`hidden_size` 128, `moe_intermediate_size` 64, `E4M3` at `q256=1024`, the
default `WINDOW`/`CHANNEL` wire.

**Which flags, and why.** The export runs `--passthrough-unrouted` because this
fixture's attention is `never_offered` by the construction census and the
exporter refuses to quantize what the runtime never builds; that flag is about
the *fixture*, not the MoE path. It does **not** pass `--no-verify`: every one
of the 18 units goes through `encode_linear_planes(verify=True)`, which decodes
each written blob and compares it to the encoder's own reconstruction, and
raises `GrammarError` on any disagreement. So the bytes were re-read twice —
once by the exporter against the encoder, once by this script against the
plugin's reader. Both runs print the numbers below identically, to the byte.
Everything below is a run's own output, not a reading of the code:

| Read back with | Result |
|---|---|
| `config.json` | one `config_groups` entry, `format: TESSERA`, `targets: [<stack>]`, and the stack **not** in `ignore` |
| `scheme.validate_tessera_moe_scheme` (what `TesseraConfig` calls) | accepted: `structure routed_moe`, 4 experts, hidden 128, intermediate 64 |
| `model.safetensors` | 12 one-dimensional `uint8` `.wire` tensors under the stack, and no `.weight` left behind |
| `scheme.parse_tessera_expert_blob` per container | all 12 parse against their declared role — grid, body, plane, span, rung, geometry |
| `moe_layout.unpack_moe_wires` | padded rows plus lengths round-trip to the written blobs **byte for byte** |
| `moe_route.prepare_tessera_moe_experts(device="cpu")` | `w13_weight [4, 128, 128] float8_e4m3fn`, `w2_weight [4, 128, 64]`, scales `[4, 128, 1]` each — the stock per-channel FP8 stack |
| the decoded tile **against the source experts** | worst relative error **0.077** over all 8 legs — a transposed, interleaved or misrouted tile sits near `sqrt(2)` ≈ 1.41 |

That last row is the one that separates *right* from *plausible*: shapes and a
clean parse cannot tell a correct tile from a transposed or interleaved one,
because both are the right size. 0.077 is the quantization error of the rung;
1.41 is what a layout mistake costs on independent weights.

**The variable-length claim is now a number.** Inside `w13`, at one shape and
one rung, the eight blobs run **21293..21297 bytes — a 4-byte spread** — and
the declared `wire_stride` is 21297, the max. That is exactly the reason the
sidecar declares a stride rather than a `wire_bytes`: there is no single length
for the group to promise. (`w2`'s four blobs happened to agree at 21427; the
stride is still the max, derived the same way.)

This covers the exporter's plumbing and the plugin's *reader* acceptance. It
does **not** cover the CUDA encoder path or the fused-MoE kernel — those are
`tests/test_export_moe_write.py`'s three `@cuda` cases and the load probe, both
of which ran on the GPU late in the pass and are in §5. A CPU run is not a GPU
receipt, and this one was not: the GPU run found a defect in a `@cuda` test
that every CPU run had skipped.

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
| one expert at a different shape | `One stack is one tile, so one shape` — reproduced end to end in `moe_write_readback_check.py`, on a whole checkpoint |
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
  were held for nearly all of this pass — sparky's three tokens by other agents'
  jobs, sparklina's by an out-of-pool encode — behind a 19-deep GPU queue whose
  head-of-line item was **3.6 h old**. Two short GPU jobs did eventually land
  (below); the encode campaign in §4, which is ~75 GPU-minutes and then a
  serve, was costed and scripted but not submitted. It is also more than an encode: no GLM
  artifact has ever been served in this repo (every served receipt here is
  Qwen3-0.6B), so the serve needs a GLM teacher dump on the pinned image before
  a student KL means anything.
* **The GPU legs ran late in the pass, and one of them was red.** Both jobs
  sat unclaimed for hours behind other agents' work and then landed on sparky
  (GB10, sm121):
  * `2be23f3a9e9d` — `experiments/moe_route_load_probe.sh` in the pinned image
    `prismaquant/glm53-mia-sm121:487ecf187`, **`rc=0` in 100.8 s**. Its
    `positive_exported` arm is the matched pair this receipt exists for: the
    bytes **the exporter wrote** load through `RoutedExperts.load_weights` (12
    calls onto `w13_wire`/`w2_wire`), materialize to `w13_weight [4, 512, 512]
    float8_e4m3fn` **byte for byte equal to `materialize_stock`**, and run
    through vLLM's own fused-MoE kernel (`TRITON`/`TritonExperts`) to a
    `[17, 512]` output at `rel_l2` **0.014298978779530125** — the same digits
    as the arm the probe encoded itself, while `wire_bytes_on_disk` differs
    (999596 vs 998984). Different producers, different bytes, same tile, same
    kernel output. The full record is
    `experiments/results/moe_route_load_probe_export.json`, and
    `docs/measurements/tessera-moe-route-load-2026-09-04.md` carries it.
  * `93f5bae3b4be` — `tests/test_export_moe_write.py` +
    `tests/test_export_moe_layouts.py` on the GPU: **1 failed, 37 passed in
    68.06 s**. The failure was mine and it was real:
    `test_a_planned_stack_is_written_as_the_plugin_reads_it` read
    `parsed.unit.geometry`, but `parse_tessera_expert_blob` returns
    `[(role, unit)]` and the geometry hangs off `unit.manifest`. Nothing had
    caught it because the case is `@cuda`-gated and every CPU run skipped it —
    the exact shape of "unmeasured, not passing" this section warned about, in
    this section's own tests. Fixed here, and re-run.
  * The CPU halves had landed earlier, both through the pool on sparky:
    `tests/test_export_moe_layouts.py` — `839b1b0a1bf4`, **22 passed, 2 skipped
    in 29.74 s** — and `tests/test_export_moe_write.py` — `cb8372740b1b`,
    **11 passed, 3 skipped in 0.78 s**. Every skip in both is a `@cuda` case.
  * A prediction this receipt made and got wrong, recorded because it was
    wrong: `test_export_moe_layouts.py::test_the_exported_ignore_names_what_vllm_builds`
    was flagged as likely to fail on a GPU box because it exports a fixture
    whose attention is `never_offered` without `--passthrough-unrouted`. It
    passed.
* **The whole suite ran only in part, on CPU.** After merging `master`, the six
  files this branch touches or that `master` touched —
  `test_serve_build_identity` (the merged-in one), `test_export_moe_write`,
  `test_export_moe_layouts`, `test_serving_moe_route`, `test_serving_moe_scheme`,
  `test_moe_wire_layout` — run **89 passed, 5 skipped in 160.7 s** on CPU
  (`CUDA_VISIBLE_DEVICES=`, all 5 skips `@cuda`). The full `tests/` suite was
  submitted to the pool's CPU lane and had not been claimed when this was
  written, so "the suite is green" is a claim about those six files, not about
  the tree.
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
is byte-identical under the same command line.

The unplanned stack is **directly measured** too, rather than inferred from
that: leg 2 of `moe_write_readback_check.py` exports the same checkpoint with
no plan entry for the stack, and the run reports the stack named in `ignore`,
no `.wire` tensor anywhere under it, every source `.weight` still present, and
`routed_moe.disposition == "passed_through_bf16"`.

*(The `dense` case is the only one of the seven that reaches an export at all;
the other six are construction-gate refusals, and their refusal text is
unchanged too.)*
