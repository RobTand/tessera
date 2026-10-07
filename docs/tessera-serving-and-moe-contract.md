# Tessera: serving contract, MoE cell, and export gate

**Status update, 2026-09-04:** the historical open census/KL item below is now
superseded for the complete LFM2.5-8B-A1B artifact. The packaged contract
(v16 when this was written; v17 since #131/#133, where each cell also names
its toolchain and its `evidence` grade) attests only E4M3/q1024 routed MoE,
resident/eager on sm_121 and the exact EUGR image, for decode and batch -- the
batch cell on a prefill top-1024 KL lower bound, the decode cell on the route
census alone. Current scope and limitations are in
[`ARCHITECTURE.md` §4.4](ARCHITECTURE.md#44d-the-expert-stack-is-a-structure-not-a-module)
and the [campaign receipt §§7–9](measurements/tessera-lfm-campaign-2026-09-04.md#7-full-model-route-census-completed-at-2105-utc).
The historical body remains intact; it is not the current release checklist.

Status: **decisions**, 2026-09-01; sections 1 and 3 **superseded 2026-09-02**; the span-2 NVFP4 decoder's extension entry and load path are **retired 2026-09-16** (A4 whole-weight expansion; see ARCHITECTURE.md §3.3)
by the Tessera serving plugin (`docs/measurements/tessera-serving-plugin-2026-09-02.md`).
The superseded text is kept: it is the decision the plugin replaced, and the
reasoning in it is why the plugin looks the way it does.

Original status line: Written before the render mechanism so the
on-disk layout is not built twice. Every measured number here is cited to the
run that produced it; nothing is asserted about a runtime we have not read.

## 0. What is already true (measured, not planned)

| Fact | Value | Source |
|---|---|---|
| arity-2 body serialises | bit-exact round-trip | `132b46c`, `tests/test_ktuple.py` |
| E2M1 k=2 R=7 body rate | **exactly 4.000000 bpp** | `terminal_rate`, both accountants agree |
| built artifact at 512x256 | 4.031250 bpp (forest planes) | `built 66048 = calc 65536 + 512` |
| encode throughput, GLM expert shape, **E2M1x2 R=7** | **23.3 Mparam/s** (0.36 s per 2048x4096) | `experiments/encode_throughput_glm_expert.py` (defaults `--arity 2 --rate 7`, `E2M1_GRID`) |
| same shape, **E4M3 / CHANNEL / window L=14** | **1.651 Mparam/s** (5.08 s per 2048x4096, box held) | `experiments/results/moe_encode_rate_profile_exclusive.json`, 2026-09-04; gate/up only, so the shape matches the row above |
| the same, **sharing the box** | 0.849 Mparam/s (9.89 s per 2048x4096) | `..._contended.json`; 1.95x at matched shape, invisible from inside the process |
| the same, **over a whole expert** (gate, up, down) | 1.611 Mparam/s (5.21 s mean) | exclusive arm, all three projections; both shapes are 8,388,608 params, so this is the rate a campaign runs at |
| GLM routed-expert encode, **at the E2M1x2 rate above** | **3.72 h one box / 1.86 h two** | same, over 311,653,564,416 params |
| the same campaign **at the E4M3 rate** | **~54 h one box / ~27 h two** | same param count, at the whole-expert 1.611 Mparam/s |
| encode power | 47 W of ~140 W envelope (E2M1x2); 64-70 W of 140 W (E4M3) | neither is envelope-bound |

> **Correction, 2026-09-04 — the grid was missing, and it is the whole number.**
> The 23.3 Mparam/s row was measured on **E2M1x2 at rate 7**, which is the
> script's default and was the default wire when this doc was written. It is
> **not** the wire an expert route would encode at today: a GLM expert resolves
> through `wire_recipe(E4M3, 1024)` to the `TESSERA_FP8` route -- E4M3 grid,
> CHANNEL plane, window body L=14 -- and that encode is ~~27.6x~~ **14.1x**
> slower per parameter on the identical shape and box.  *(Both figures in this
> paragraph were struck by the amendment below: they came from an E4M3 arm that
> shared sparky, and the held-box arm is 1.91x faster.  The struck numbers are
> left visible rather than silently swapped, because what they cost is the point
> -- but nobody should be reading a number off this paragraph.)*
>
> So the conclusion below inverts. The E4M3 campaign is ~~two and a bit days on
> one box, about one on two~~ **~54 h on one box, ~27 h on two**, not an
> afternoon, and encode cost is back on the list of things that gate a served
> expert route.
>
> **Amended the same day: the quiet-box measurement this note asked for has been
> run.** It is 5.21 s per unit over a whole expert, 1.611 Mparam/s, **~54 h** --
> so the 9.94 s figure was an upper bound by a factor of **1.91**, entirely from
> sharing sparky. Compared at the *one shape* the E2M1x2 row was measured on
> (2048x4096, gate/up), the quiet box runs 5.08 s against 9.89 s, and the gap to
> that row is **14.1x** per parameter rather than 27.6x -- and that ratio is
> held-box against held-box, because the E2M1x2 arm's own **47 W** total is
> below the 63-88 W a shared sparky reads, so it cannot have been contended
> either. The two arms are a matched pair, and the pairing is what makes them
> readable:
> `torch.profiler` counted **1,056,768** `_step` invocations in both, at 96.03%
> and 96.14% of self-CUDA. Identical work; only the wall clock moved.
> Utilisation read the same on both sides -- the box-level power series is what
> separated them, 14-15 W idle before the exclusive run against 63-88 W before
> the contended one. Both files stay committed, because the contended one is the
> evidence for the sentence that follows: **schedule this campaign on a held
> box.** Sharing it costs 1.91x, and nothing inside the process can see that
> happening.
>
> Original text, kept because it is the reasoning the correction overturns:
> *"The encode campaign is an afternoon. Encode cost is **not** what gates this
> work, which removes the main argument for building the serving backend
> first."*

## 1. Serving contract (decided; loader deferred)

> **Superseded 2026-09-02 — the loader is no longer deferred, and it is ours.**
> Tessera ships its own vLLM plugin: `tessera.serving`, an entry point in the
> `vllm.general_plugins` group registering `quant_method: "tessera"`.  A serve
> installs one package -- Tessera -- and no second quantization stack.  Both
> routes below are served by it (`TESSERA_NVFP4` W4A4 and `TESSERA_FP8` W8A8),
> and since 2026-09-02 a **third** the text below did not anticipate:
> `TESSERA_BF16`, W16A16 -- the same window body and CHANNEL plane as the FP8
> route with its table snapped to bf16, decoded to an ordinary bfloat16 tile
> for the runtime's own GEMM, with the row scale applied as an fp32 epilogue
> and **never folded into the tile**.  It exists because the E4M3 alphabet, not
> the trellis, is what floors the window body above ~6 bpp.  The BF16 family is attested at
> `q256 = 1792` on `sm_121` at contract v5 -- two dense cells, `decode` and
> `batch` -- on the four-census-plus-served-KL receipt in
> `docs/measurements/tessera-bf16-route-served-2026-09-02.md`; the rule it
> waited on still holds, that a family with no receipt publishes no
> `lane_eligibility` cell.
> The streamed residency mode is the kernel lane's shape: the body stays
> compressed on disk *and in memory*, decoded inside the forward.  The only
> operator knob is `TESSERA_SERVE_MODE=resident|streamed`.  What survives from
> the text below is its reasoning: the container is self-describing from bytes
> alone, which is exactly why the loader stayed small enough to own.


The precedent is in the pinned image already: vLLM `0.1.dev20051+g487ecf187`
(Mia's exact pin) carries in-tree `exl3.py::Exl3Config`, which reads trellis
planes out of ordinary safetensors under a `quantization_config`. Tessera's
consumer is the analogous shape, and naming it now is what lets the exporter be
written once:

- **Per-unit blob as a `uint8` tensor.** One `prismaquant.tessera.v1` container
  per quantized Linear (per expert projection), stored under the unit's weight
  name. The container is already self-describing from bytes alone
  (`read_unit_artifact` takes bytes and nothing else) — that property is the
  reason a loader stays small.
- **`quantization_config.quant_method: "tessera"`**, carrying the schema id and
  the per-unit rung, so a loader never infers a grid. The grid is bound into
  `encoder_profile_id` as of `132b46c`, so the *bytes* are unambiguous even if
  the config is wrong — the config is a convenience, not the authority.
- **Two lanes, and only one needs a kernel.** Stock: decode to NVFP4 at load,
  4.5 bpp resident, no kernel, but **no disk win over NVFP4 unless the loader
  decodes** — materialising at export time forfeits the entire point. Kernel:
  body stays compressed, disk == resident, needs a vLLM quantization method.

**The size target only exists in the kernel lane.** NVFP4's floor is 4.5 bpp;
Mia's routed block is 4.0117 bpw. An NVFP4-everywhere GLM build is *+4.5%
larger* than Mia (`mia-exl3-glm53-treatment`). So "size-matched to Mia" is not
reachable by any materialise-at-export path, and the honest statement of the
goal is: it requires shipping a serving backend. That is authorized (Rob,
2026-08-31: *"Everyone knows you need one to get extreme improvements"*) but it
is the **last** thing to build, not the first.

## 2. The MoE cell (the 89%)

Mia quantizes routed experts *only* — 311.65e9 params, 89% of its bytes. A
render path that handles dense Linears and not packed experts does not touch
the goal.

- **The unit is the per-expert 2D projection weight**, keyed by qname exactly as
  every other format already is (`render_production_weight(weight, fmt, qname=…)`
  returns a dequantized tensor of the same shape). Tessera needs no new unit
  shape: `gate_proj`/`up_proj` are `[2048, 4096]`, `down_proj` is `[4096, 2048]`.
- **Per-layer expert uniformity and union-find promotion apply unchanged.**
  They are constraints on *which format name* a serving unit may carry; a
  Tessera rung is a format name. Experts stay uniform per layer, mixing across
  layers, as the existing invariant requires.
- **Diagonals (su/sv) are per-unit**, so they live in the unit's own container.
  No cross-expert sharing — that would couple units the allocator prices
  independently.
- The blocker memory frames this as *"a `glm5_next` routed-MoE cell in
  Gridbook"*. That framing is **stale**: Gridbook is explicitly not Tessera's
  substrate. The live requirement is a Tessera render mechanism in PrismaQuant,
  which is the next commit, not a Gridbook cell and not an RC wheel.

## 3. Export gate (decided)

> **Superseded 2026-09-02 for the dense cell.**  "Tessera rungs are
> `route_status: unbacked` on every serving profile today" was true of a world
> with no Tessera runtime.  Tessera now publishes its own machine-readable
> contract (`tessera/serving/runtime_contract.json`) whose dense cells are
> `backed_with_serve_flag`, carrying `requires_plugin: "tessera"` and
> `requires_serve_flags: ["TESSERA_SERVE_MODE=resident|streamed"]` -- a gate
> input, not prose (principle 14).  A producer derives eligibility from that
> table through `importlib.resources` and refuses on mismatch.  **The routed-MoE
> cell is NOT superseded:** the contract declares `structures: ["dense"]` and
> the plugin refuses a FusedMoE layer loudly, so a Tessera MoE export still
> fails closed exactly as this section says.  The declared-non-native-target
> decision below therefore still governs any MoE artifact.


Tessera rungs are `route_status: unbacked` on every serving profile today, and
export fails closed on an unbacked route (principle 9). That is the doctrine
working, not an obstacle to route around.

**Decision: the first Tessera export declares an explicit non-native target
platform, stamped on the shipcard** — not a per-run override. Reason: a
per-run override is a one-off that says nothing durable, while a declared
target is a standing, auditable statement that this artifact is not claiming a
native route on the named hardware. The bpp and any KL number it carries travel
with that declaration, per principle 12.

## 4. Order of work

1. **Render mechanism** — `TESSERA_*` in the format registry, encode->decode
   inside `render_production_weight`. Unblocks cost, allocation and KL with no
   export and no serving.
2. **Offline KL** on the rendered weights. This is the measurement that decides
   whether a serving backend is worth building. `validate_assignments_kl` needs
   no served artifact.
3. **Small-scale first** — full chain on LFM2.5-8B-A1B (shipped, MoE, small
   enough to iterate, and exercising the served routed-MoE path through its
   unpacked per-expert `w1`/`w3`/`w2` tensors) before GLM. Packed-source layout
   coverage is independent: Qwen3.8-Flash-Next supplies the real packed
   `gate_up_proj`/`down_proj` population, and the exporter requires its
   convention explicitly. GLM-5.3 is the campaign, not the testbed.
4. **Export**, then **the serving backend**, in that order and only if (2) pays.

Aqua merges at step 1/2 (the allocate step), not before. The disk-vs-resident
ambiguity in "size matched" dissolves on the kernel lane, where the body stays
compressed and disk == resident.

## 5. Queued, not blocking

Tessera decode against the **production** `flashinfer_b12x` NVFP4 GEMM at GLM
expert shapes, batched. The existing 1.049x is against a *matched Triton*
comparator, which `tessera-project-scope` already flags as not the real
comparison, and prefill needs a GEMM where only dequant-then-GEMM exists. If
that loses badly, the lane's shippability is decided by numbers rather than
after the backend is built.

---

## 7. The exporter, and what it settled (2026-09-01)

`src/tessera/export.py` closes the first of the two blockers this document
opened with. `build_unit_artifact` already inverted exactly; what was missing
was the walk around it.

- `export_checkpoint(tensors, plan, out, grid=…)` — in-memory, for tests.
- `export_checkpoint_streaming(src, out, plan, grid=…)` — one output shard per
  input shard, so a 100B-plus checkpoint never has to fit beside its own
  encoding. Encoding runs on GPU.

`plan` maps tensor name → **per-position** rate in q256 units. A name in the
plan that is absent from the checkpoint is an error, not a no-op: a plan that
silently fails to apply is how an artifact ends up heavier than the allocation
that justified it.

**Rendering identity is asserted, not assumed.** Every unit is read back off
its own bytes and compared to the encoder's reconstruction *before* it is
written. The surrogate that priced a Linear and the bytes that ship are then
the same tensor by construction.

**The arity trap, now pinned by test.** `build_unit_artifact`'s `q256` is the
per-**code** rate; a rung name's R-number is per-**position**, and a code spans
`arity` positions. Passing the rung number straight through produces a legal
artifact whose manifest declares *half* the rate it carries. `R896` at arity 2
lands on **4.001953 bpp** on a 512×4096 unit — the 0.002 is the fixed forest
planes amortising.

### 7.1 Tessera artifacts are TP-degree-specific — EXL3's are not

> **Superseded 2026-09-02 (#7). This section's conclusion inverted.** Schema
> minor 4 added the INITIAL_STATE plane — one element per column carrying the
> trellis state a row-sliced column starts from — and `tessera.layout.slice_unit`
> cuts a whole unit on either axis at load. A Tessera artifact is now
> **TP-agnostic**: the exporter never learns the TP degree and every rank cuts
> its own shard out of the same bytes (`docs/design/tensor-parallel.md`,
> `docs/schema/prismaquant.tessera.v1.md` §shard). The paragraph below is kept
> because it states the cost correctly for the wire that existed when it was
> written, and because the mechanism it names — the trellis running down rows
> within a column — is exactly why the row axis needed a new plane and why the
> span-2 route still refuses a row cut (`sharding.ROUTE_TP_AXES`).
> **What moved later (contract v29, 2026-09-15):** two-rank serves have run, and
> `runtime_contract.json` publishes `max_world_size: 2` per family on a named
> world-size receipt, graded `route_only`
> (`docs/measurements/tessera-glm53-a4-stub-tp2-world-size-2026-09-15.md`).
>
> **Also superseded 2026-09-05 (#328): the field named below no longer exists.**
> The paragraph says the config "declares `tp_size` so a loader cannot quietly
> use one at the wrong degree". It declared the constant `1` and no loader ever
> read it. The config now declares `schema_minor` and `tp_agnostic` derived from
> it (`slicing.tp_agnostic_at_minor`), and
> `TesseraConfig._require_a_cutter` reads that back through
> `sharding.require_a_cuttable_artifact` — the load-time gate the sentence
> promised. `docs/ARCHITECTURE.md` §3.8 is its home.

Reading vLLM's `exl3.py` beside our encoder settles an open question the wrong
way for us. EXL3 shards by `Tensor.narrow` on `trellis` dim 0/1 and on
`suh`/`svh` (`shard_exl3_col` / `shard_exl3_row`), because its trellis is a
structured tile tensor `[in/16, out/16, 64]`. Any TP degree is a view.

A Tessera unit is one blob of packed bit-planes with per-column rates. No byte
range is a sub-weight, and the trellis runs **down rows within a column**
(`body_bits` is `[rows/arity, cols]`), so a row-parallel split — exactly what
column-parallel Linears like gate/up need — cuts the trellis along its own
state path. **A Tessera artifact must be re-encoded per rank.** The config
declares `tp_size` so a loader cannot quietly use one at the wrong degree.

This is a real cost EXL3 does not pay. It is not a bug to fix; it follows from
the trellis being the thing that buys the rate. It matters directly here,
because the target is a 2×DGX-Spark serve.

### 7.2 Two corrections to what this repo believed

- **`exl3.py` is not in stock vLLM.** It is absent from
  `vllm/vllm-openai@905c0293` and present only in
  `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`. Serving EXL3 means
  running Mia's image.
- **Mia's artifact self-declares it is not qualified to serve.**
  `exl3-mcg-storage-abi.json` carries `"serving_reader_qualified": false`,
  `"qualified_tp_sizes": []`, and the reason *"ExLlamaV3 v0.0.43 has no audited
  GLM-5.3 TP model load/inference receipt"*. Its scope is
  `glm53_routed_experts_only`.

### 7.3 The EXL3 comparator is still open, and should move to served KL

Five offline probes could not reproduce EXL3's weights: `rel_err` ≈ √2, norm
ratio 0.999, cosine ≈ 0, identical through `execute_exl3_linear`, an identity
probe, and EXL3's own `get_weight_tensor()`. Ruled out: wrong source (`suh`
correlates with BF16 column norms, +0.09 to +0.30, layer-varying), expert
permutation (all 288 experts checked, best |cos| 0.0021), and garbage
(reconstruction kurtosis matches the true weight's to three significant figures
and tracks it per tensor). The values are right and the **basis** is wrong —
consistent with a missing Hadamard/sign leg (`su`/`sv` are `None`, `mul1` is
False).

The conclusion is not to keep probing. The comparator that decides anything is
**served KL** (principle 3), and serving lets vLLM perform the decode, which
removes this problem entirely rather than solving it.

## 8. Pricing, the budget, and the route to a served number (2026-09-01)

### 8.1 The allocator was pricing a checkpoint the exporter never writes

`FormatSpec` models a format as an integer weight rate plus a group-scale term.
Tessera's rate is fractional and `artifact_bpp` **already counts the scale
planes**, so synthesizing a spec from it charged `ceil(bpp)` *plus a second
scale term*: `TESSERA_E2M1_K2_R896` priced at **4.25 bpp** against an artifact
whose byte-exact accountant measures **4.00**.

This was live on both paths that decide anything —
`allocator_solver.py:748` reads `effective_bits_for_shape` for the DP's
per-format bit cost, and `footprint.py` reads `memory_bytes_for_shape` for the
byte-budget gate — while `tessera_footprint.py`, which the *candidate builder*
uses, priced it correctly all along. **Two accountants, one format, different
numbers**, and the allocator was ranking Tessera against NVFP4 on a 6.25%
overcharge it invented.

`FormatSpec.exact_bits_per_param` is the fix: when set it is the *whole* rate
and **replaces** both terms rather than adding to them. Every other format
leaves it `None` and prices byte-identically to before. Three tests pin it,
including one that asserts the registry and the footprint agree byte-for-byte
across the rung range.

The lesson is the ordinary one in a new form: a format whose cost model is a
*special case* of the generic one will be silently mis-priced by the generic
one, and both numbers look plausible.

### 8.2 The size target is 158.783 GiB, and Mia leaves 15.44 GiB on the table

Measured from her artifact and the BF16 source
(`docs/measurements/glm53-body-budget-2026-09-01.md`). **Layer 45 is the MTP
layer and it is a full MoE block** — 288 quantized experts, 3.729 GiB, not the
two BF16 projections a prior note recorded. Since the target excludes MTP, that
correction moves the budget by 3.7 GiB.

Mia quantizes **only** the routed experts. Attention, dense and `lm_head` —
15.44 GiB on 2.6% of the parameters — stay at 16 bpp. Pricing those at FP8
frees **7.71 GiB**.

**Corrected 2026-09-01:** those bytes buy **no rate increase on the experts**.
An earlier line here said "+0.2171 bpp on the experts at identical total size";
there is no such rung. Body and completion sum to the cap, so a Tessera family
has one size — the serialisable set is 3.5000 and 4.0000 bpp, nothing between,
nothing above (`docs/measurements/tessera-rate-ceiling-2026-09-01.md`). And
`NVFP4` at 4.5 is **Pareto-dominated** by Tessera at 4.0 on this route: more
bytes *and* 11% more functional error, because `flashinfer_b12x` serves it
W4A4.

What the bytes buy instead is a **format change on ~2.6 of the 45 expert
layers** — FP8 on 5.71% of routed-expert parameters, which packed MoE permits
because uniformity is required within a layer and not across them. Measured,
that trade wins by **7.7×** (`docs/measurements/glm53-bit-trade-2026-09-01.md`).
The heterogeneous-allocation thesis holds on this model, and it is still
structurally unavailable to a uniform-format method; only its mechanism changed
from "raise everyone's rate" to "promote the layers that need it".

### 8.3 A served number without a serving backend

The kernel lane is not optional and not merely preferable: GLM's 311.7B routed
params are ~623 GB in BF16, so **dequant-on-load cannot serve the full model on
any box here**. A dequant-on-load vLLM plugin would therefore be throwaway
plumbing, and the size claim lives *only* in the kernel lane.

But quality does not have to wait for it. `experiments/decode_back_to_bf16.py`
decodes an artifact into a plain BF16 checkpoint through `read_unit_artifact`
— the format's own reader, the one `tests/test_kernel.py` pins the Triton
decode against with `torch.equal` — so the tensors *are* the artifact's meaning
and a KL measured on them carries to the kernel lane: both lanes serve the same
W4A16 contract, the kernel just decodes later and in registers. Two caveats,
both stated: the output is BF16-resident and proves **nothing** about size; and
the bf16 cast of an fp32 reconstruction adds a ~2^-9 rounding the kernel lane
need not have — ~0.2% of the error energy, and in the direction that makes this
arm slightly worse than the kernel lane rather than better.

`experiments/assert_render_export_identity.py` checks the seam nothing
structurally enforces — that the render PrismaQuant prices equals the bytes the
exporter wrote — on real exported units rather than by assumption.

## 9. Four measured facts that shape the MoE route (2026-09-02)

Everything here was measured on this date against the pinned serving image
`prismaquant/glm53-mia-sm121:487ecf187` (vllm `0.1.dev20051+g487ecf187`) and
the surgical model `/mnt/shared/models/GLM-5.3-Flash-4layer`. Each carries the
scope it was measured at, per principle 14 — none of them is a claim about GLM
proper, or about any other build.

### 9.1 The NVFP4 MoE arm is refused by this model's own config

`GLM-5.3-Flash-4layer/config.json` sets `swiglu_limit: 10.0`. The build's
`fused_moe/oracle/nvfp4.py` excludes `FLASHINFER_B12X` from
`NVFP4_BACKENDS_WITH_CLAMP`, so an explicit `--moe-backend flashinfer_b12x`
raises `ValueError` on this config, and the auto list is filtered to the
clamp-capable backends — `FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL`,
`FLASHINFER_CUTLASS`, `VLLM_CUTLASS`, `MARLIN`, `EMULATION`, `HUMMING`.

> **Measured 2026-09-02** (#6, `docs/measurements/nvfp4-moe-oracle-2026-09-02.md`,
> `experiments/nvfp4_moe_oracle_probe.py`), and three things in the paragraph
> above need widening or answering:
>
> * **"Which of those is backed on sm121" is no longer unmeasured.** Five of the
>   seven admit this device by their own `_supports_current_device()`
>   (`FLASHINFER_CUTLASS`, `VLLM_CUTLASS`, `MARLIN`, `HUMMING`, `EMULATION`);
>   `FLASHINFER_TRTLLM` and `FLASHINFER_CUTEDSL` are family-100 kernels and do
>   not. Asked to resolve on GLM-5.3-Flash's MoE dimensions with the W4A4 keys,
>   the oracle returns **`FLASHINFER_CUTLASS` / `FlashInferExperts`** — clamped
>   *and* unclamped, so the clamp does not change the answer on sm121, because
>   the backends it removes are the ones the device rejects anyway. `auto`
>   reaches it with **no flag at all**.
> * **The scope is the GLM-5.3-Flash family, not this model.** Every
>   GLM-5.3-Flash `config.json` on this box sets `swiglu_limit: 10.0` under
>   `text_config` — proper, BF16, 4layer, and the Tessera and EXL3 exports. The
>   4-layer model inherited the clamp; it did not introduce it.
> * **`FLASHINFER_B12X` is excluded from auto-selection outright**, not only by
>   the clamp filter: "intentionally excluded from auto-selection until the
>   upstream CUTLASS SM121 MMA op guard is resolved". The explicit flag is the
>   only path to it, and on any clamped config that path raises. Unclamped, it
>   resolves fine here, so the clamp is the whole blocker — and closing it means
>   `FlashInferB12xExperts` implementing the SwiGLU clamp, which it does not
>   (`flashinfer_b12x_moe.py` contains no `swiglu`, `clamp` or `limit`).
>
> Still open, and why #6 stays blocked: that is the **oracle's** resolution on a
> constructed config, not a served MoE. A `requires_serve_flags` cell needs a
> route someone has served.

**Scope: build `487ecf187`.** It had one consequence that is not scoped: the
sentence in `serving/config.py`'s MoE refusal that said NVFP4 W4A4 "needs
`--moe-backend flashinfer_b12x` on GB10" was an *asserted* runtime claim of the
kind principle 14 forbids, and on this family it was false twice over — the flag
is refused, and no flag is needed. **Corrected 2026-09-02 (#31), and the
deferral was wrong to take.** The argument for waiting was that the correction
should travel with the route's evidence; but the evidence that the sentence is
false is the measurement above, which exists now, and the sentence was
misdirecting an operator in the meantime. The refusal names no flag at all and
points at this receipt instead: which backend a build resolves is that build's
answer to give. When the route is served, the flags belong in the cell's
`requires_serve_flags` — a field a gate reads — and not in prose.

The FP8 oracle carries no analogous clamp filter, so **the TESSERA_FP8 family
(E4M3 wire, WINDOW body, CHANNEL plane -> per-channel FP8 W8A8) is the first
and only served MoE arm.** The `requires_serve_flags` value for its contract
cell will be whatever the pinned build is *seen* to need — not a backend name
copied from the NVFP4 lane.

### 9.2 The exporter could not see this model's body at all

Three faults, all at plan time, all silent (fixed; see
`tests/test_export_moe_layouts.py`):

* `quantizable` filtered on `name.startswith("model.layers.")` and `main`
  parsed the layer index as `name.split(".")[2]`. This checkpoint roots its
  decoder under a sub-model, `model.language_model.layers.N.`, so the filter
  matched **nothing**: an export would have quantized zero Linears and reported
  success. `BODY_LAYER` matches `model.<...>.layers.<N>.`; the vision tower is
  `model.visual.blocks.N.` and stays BF16 by the same rule rather than by a
  second exclusion list. Staying BF16 is not the same fact as being *named*,
  though, and the plugin reads the second one: `ignore` was assembled from
  three `BODY_LAYER`-gated sources, so every vision Linear was passed through
  and never named, and a `LinearBase` that is neither declared nor ignored is
  refused (#86). `ignored_modules` now derives the name from each tensor the
  export **writes**, body or not — one rule, no roster to keep beside it.
* Routed experts here are **unpacked per-expert 2-D**
  (`...mlp.experts.{e}.{gate,up,down}_proj.weight`, 2592 of them across layers
  1–3; layer 0 is dense). Being 2-D, nothing separated them from ordinary
  Linears, and they would have been encoded — for hours — into a checkpoint
  whose `config_groups` name modules vLLM never builds, which the plugin then
  refuses at **load**. `ROUTED_EXPERT_2D` splits them out at plan time. Its
  `\d+` segment is what distinguishes a routed expert from
  `mlp.shared_experts.gate_proj`, which is an ordinary Linear and stays
  quantizable as one.
* `len(shape) >= 3` was the whole test for "packed expert stack", and this
  model's attention carries `k_conv1d.weight [8192, 1, 4]`. Its ignore entry —
  module minus leaf — is `...self_attn`, the parent of every attention Linear
  in the layer. `PACKED_EXPERT_ND` identifies a stack by where it sits.
* **A fourth, found 2026-09-03 and fixed here.** `quantizable` tested
  `name.endswith(".weight")` before it looked at anything else, and
  transformers-5 stores a packed stack as an `nn.Parameter` on the experts
  module — so the tensor on disk is `...mlp.experts.gate_up_proj`, with no
  suffix at all. Those names were classified as **nothing**: not dense, not
  packed, not routed, so `expert_shapes` was empty on a model with 96 stacks.
  The classifier is what the PLAN is built from, so what survives here is the
  plan-time half: a `--plan-json` naming such a stack fell through to the
  generic "unknown tensor" check instead of the packed-expert refusal that says
  what is missing and why. The fix reads the name through its `.weight`
  spelling (`probe`), the same idiom `ignored_modules` uses, so one convention
  decides what a name means and the suffix decides nothing.

  The *load-time* half of this fault — the FusedMoE module never reaching
  `ignore`, so the plugin refuses a layer it cannot find there after the dense
  body has encoded for hours — was closed independently by #86, which replaced
  the three `BODY_LAYER`-gated ignore sources with one rule over the tensors
  actually written. Recorded because the two look like one bug and are not: the
  classifier decides the plan, `ignored_modules` decides the ignore list, and
  only the first is still this section's.

  Making `.weight` optional **widens** the classifier, and the widened edge is
  pinned rather than assumed. `<moe>.experts.<projection>` with no suffix is
  admitted *before* the suffix test, so its rank is no longer checked by the
  `len(shape) >= 3` branch; "nothing else in a decoder layer is called
  `experts.<projection>`" would be an assertion about every checkpoint yet to
  exist, and that is the same shape as the rank assumption the third fault
  above already retired. A bare name of rank < 3 therefore **refuses at plan
  time**, by name and rank: a packed stack stacks experts, so it carries an
  expert axis, and both available guesses are wrong in the expensive direction
  (as an expert stack it leaves the module BF16 and named in `ignore`; as a
  dense Linear it declares a module vLLM never builds). No checkpoint on this
  box triggers it — it is the guard that keeps the *next* layout from being
  filed silently.

Measured after the fix: **49 dense Linears (was 0), 2592 routed expert leaves
across layers [1, 2, 3], 0 packed stacks** on GLM-5.3-Flash-4layer; and on
`/mnt/shared/models/Qwen3.8-Flash-Next`, **96 packed stacks over 48 FusedMoE
modules (was 0)**, the remaining two being the MTP sidecar's, which `BODY_LAYER`
does not match by the same rule that keeps the vision tower out.

`experiments/moe_plan_baseline.py --diff --no-export`, master `cf5d0e6` against
this change, over the seven expert layouts it builds plus the real Qwen
checkpoint: **5 rows of 36 move, and every one of them is a classification or a
plan-time refusal.**

* `classify/<case>/packed` goes `[]` -> 96 on `Qwen3.8-Flash-Next`, `[]` -> 2 on
  `qwen38_packed_nosuffix`, `[]` -> 4 on `gptoss_packed`. Nothing else moves in
  any bucket, so no tensor changed which kind of layer owns it.
* The two packed plan refusals stop reading "plan names tensors that are not 2-D
  body weights" and start naming the stack, its shape and its orientation.
* `ignore` and `quantization_config` do **not** move, which is the point of
  running this against the new base: they moved in the pre-#86 measurement of
  the same change, and #86's one rule over the tensors written now reaches those
  modules by its own path.

The export rows (`tensors`, `manifest`, `ignore`, `quantization_config` sha256
per case) were **not** re-taken against `cf5d0e6`: they need a CPU encode per
case, and the box was at load 131. What can be said without them is mechanical
rather than measured, and is labelled as such — the exporter diff against master
is three hunks, in `PACKED_EXPERT_ND`'s comment, in `quantizable`, and in the
manifest dict in `main`; no encode, render, pack or write path is touched. The
one manifest field this adds is `routed_moe`, below.

### 9.3 A packed expert stack cannot always be oriented, and this model is the case

`[E, A, B]` is ambiguous on its face, and transformers-5 architectures genuinely
differ (`gate_up_proj` appears both as `[E, hidden, 2*inter]` and as
`[E, 2*inter, hidden]`). `packed_expert_orientation` reads the output axis off
`hidden_size`/`moe_intermediate_size` and **refuses** when the dims cannot
decide it.

That refusal is not defensive programming. This model has
`hidden_size == 2 * moe_intermediate_size == 4096`, so a packed `gate_up_proj`
would be `[E, 4096, 4096]` — square — and no comparison of dims orients it. A
default axis order there transposes every expert in silence.

The packed path's **orientation** tests are therefore synthetic: they fix the
contract, not agreement with a real checkpoint. Its **classification** tests are
not, as of 2026-09-03: `/mnt/shared/models/Qwen3.8-Flash-Next` is a packed
source on this box, and `test_the_real_packed_source_is_classified_as_experts`
reads its 96 body stacks off disk. That distinction is not pedantry — the
spelling the classifier missed (§9.2's fourth fault) is one only a real
checkpoint carried, and every synthetic packed fixture had written `.weight`.

### 9.4 The encode budget is real, and the fused path is already taken

Profiled (`torch.profiler`, one unit at the real expert shape 2048x4096, E4M3
q256=1024 -> WINDOW/CHANNEL, `window_bits = 14`):

* `fused_available()` is **true** and the fused Triton `_step` is **97.97% of
  CUDA time** (4.600 s of 4.696 s). The rate is not a missing fused path; it
  *is* the fused path.
* `verify=True` costs ~3% (5.13 s vs 4.97 s), so the verify is not a lever.
* **~5.0 s per expert unit** (gate/up 2048x4096: 4.97 s; down 4096x2048:
  5.11 s), so one MoE layer — 288 experts x 3 projections = 864 units — is
  **~72 min on one box**, or ~36 min split across sparky and sparklina.

Power, against the box (Netdata, `nvidia_smi.gpu_power_draw`, sparky): the
encode loop holds **71–73 W of GB10's ~140 W envelope**, against 6–17 W idle.
So the encoder is fused but roughly half-loaded, and utilization would have
said nothing — on GB10 it reads "a kernel is resident", not "the SMs are
working". That headroom is real and unspent: `src/tessera/encode.py` is
out of scope here, and the budget above is workable without it.

### 9.6 A wire parameter routes through the expert mapping and is then dropped by the loader

Issue #5's item 3 records `RoutedExperts.build_expert_params_mapping` as
suffix-agnostic, "so custom suffixes route fine". The mapping is; the loader is
not, and the difference is silent.

Measured on the pinned build, no GPU
(`experiments/moe_wire_loader_probe.py`,
`docs/measurements/tessera-moe-wire-loader-2026-09-03.md`): `load_weights`
rewrites the checkpoint name to the parameter's before it calls the loader, so
`...experts.0.gate_proj.wire` arrives as `...experts.w13_wire` — a string
containing neither `weight` nor `scale`, which are the substrings
`RoutedExperts.weight_loader`'s dispatch tests. It falls off the end and returns
`False`, writing nothing and raising nothing. `w13_wire`, `w2_wire` and
`w13_wire_len` all do; `w13_weight` and `w2_weight`, the same call through the
same stand-in, return `True` and are written.

The remedy is a value the route sets anyway: `load_weights` calls
`param.weight_loader`, so a wire parameter registered with a loader of its own
is loaded by that loader and the dispatch is never in the path. Driven through
`load_weights` itself, it is invoked, the name is yielded and the parameter is
written. It is also the only form that can work: a wire row is variable-length
at a declared stride, and `_load_w13` narrows by half the shard dimension and
then copies shape-for-shape, so a short blob into a padded row is a size
mismatch rather than a short write.

This constrains `create_weights` for the expert route. It says nothing about
the forward: no `ROUTES` entry, no `apply`, no `routed_moe` in
`scheme.STRUCTURES`, no `lane_eligibility` cell, and no served measurement to
justify one.

### 9.5 The serving image could not build the streamed decoder

Found by making the NVFP4 streamed tests fail instead of skip. Three hosts,
three different causes, all previously reported as one absent kernel:

| host | cause | now |
|---|---|---|
| sparky | `/usr/local/cuda` -> `/etc/alternatives/cuda` -> `cuda-13.3`, a partial install with no `bin/nvcc`; the complete `cuda-13.0` sits one directory away and matches `torch.version.cuda` | 23 passed |
| sparklina | `ninja` is in the venv's `bin`, off a non-login ssh's `PATH` | 23 passed |
| `glm53-mia-sm121:487ecf187` | nvcc and ninja both present, but no `cusparse.h` under `/usr/local/cuda`; ATen's `CUDAContextLight.h` includes it unconditionally | 128 passed, 0 skipped |

The third is the one that matters for shipping: the **streamed residency has no
pure-torch fallback by design**, so until this was fixed that route could not
build on the very image that serves it — and nothing said so, because a blanket
`skip if get_tessera_ext() is None` is green whether the kernel is absent or
merely unbuilt. `ext.toolchain_report()` now separates the two, and the tests
skip only on a genuinely absent toolchain and **fail** when the toolchain is
present and the build broke.

## 10. The rung set the contract publishes is the set the decoder reads (2026-09-02)

`runtime_contract.json` published `candidate_rungs_q256: [1024]` for
`TESSERA_E4M3_K1` and `[896]` for `TESSERA_E2M1_K2`, and nothing consumed it.
The first PrismaQuant-allocated artifact served seven rungs outside that list —
R749, R750, R934, R1006, R1083, R1107, R1262 — with a clean 112/112 census and
nothing refusing (receipt: `docs/measurements/tessera-allocated-served-2026-09-02.md`,
finding in §9). That is not a correctness bug at serve time; it is a contract
that states a set the runtime neither enforces nor is restricted by, which is
provenance nothing consumes.

**The bound is now derived from the decoder.** Each candidate rate was encoded
into a unit, packed exactly as the exporter packs, and taken through the
plugin's own load path — `parse_tessera_blob_for_scheme` then the route's
`prepare_*`. What that path accepts is what the contract states. Two different
mechanisms turn out to bound it, and they are not the same mechanism on the two
families:

| family | grid | reader range (q256) | step | what bounds it |
|---|---|---|---|---|
| `TESSERA_E4M3_K1` | `E4M3` | **[256, 2048]** | 1 | the trellis grammar's shaped domain at both ends: code rate 1..8 over an 8-bit-native alphabet. Continuous — 120 of 120 in-range probes accepted, no interior gap. |
| `TESSERA_E2M1_K2` | `E2M1x2` | **[896, 896]** | 1 | *above*, the same grammar (rate 7 of arity-2 native 8, so q256 ≤ 896); *below*, the native decoder, which serves the span-2 TCQ body only — under 896 `wire_recipe` writes a WINDOW body and the NVFP4 route refuses it by name at the same load seam. (The refusal was `ops.prepare_tessera_module`'s until the A4 whole-weight expansion was retired 2026-09-16; the A4 lane keeps it.) |

So E4M3's published set was two orders of magnitude too narrow and E2M1x2's
single point was exactly right, for a reason nobody had written down.

The fields changed shape to stop the misreading that opened the gap:

* `reader_rate_range_q256` **is** the decodable set, with `reader_rate_step_q256`
  making "continuous" a thing a gate can read rather than a thing a list
  implies, and `reader_rate_bound` naming the mechanism.
* `candidate_rungs_q256` is renamed **`attested_rungs_q256`**. It never was the
  decodable set; it is the rungs a `lane_eligibility` cell attests, and the
  validator still requires every cell's `rungs_q256` to be a subset. Decodable
  and attested are different claims and now have different names. It is
  deliberately **not** widened to cover the allocated run's rungs: those were
  censused by another artifact's receipt, not by a cell here. The old name is
  **retained as a deprecated alias** carrying the same list (the validator
  refuses it if the two disagree), so the change is additive and the `schema`
  string does not move — see the cross-repo note below.
* Each format entry now names its `grid`. `TESSERA_NVFP4` declares it holds
  `E2M1` as well as `E2M1x2`, and resolving a range by route alone would have
  handed an arity-1 checkpoint the arity-2 numbers. A (route, grid) pair the
  contract does not describe is refused, not served on another pair's figures.

**Checked against every artifact, not against the receipt.** The seven rungs
the receipt named are not the whole exported population. Reading the
`config_groups` of all 15 Tessera checkpoints on either box gives **17 distinct
`(grid, q256)` pairs**: `E2M1x2` at 896 and nothing else, and `E4M3` at
**sixteen** rungs spanning **493 … 1384** (493, 749, 750, 785, 814, 824, 909,
934, 1006, 1024, 1083, 1107, 1217, 1262, 1366, 1384). Every one is accepted by
the new gate. That is the argument for deriving the bound from the decoder
rather than from the exported history: a range widened to cover the receipt's
seven (749…1262) would refuse both R493 and R1384 — real rungs of the 3.0 and
5.0 bpp allocated arms — and the gate would have become an accommodation of
whichever checkpoints someone happened to cite.

The same scan settles the one refusal that is genuinely **new** for a grid
`ROUTES` still lists: an arity-1 `E2M1` scheme is now refused as unattested
(the contract publishes a range for `E2M1x2` only). No artifact has ever been
built on it — all 15 are `E2M1x2` or `E4M3` — so nothing that exists is refused
by it, and the alternative was to serve an arity-1 checkpoint on the arity-2
grid's numbers.

**The gate.** `validate_tessera_scheme` refuses a declared `q256` outside the
published set, naming the rung and the set. It runs at sidecar-parse time,
before a parameter exists — the same place the family, grid, body and plane
refusals already live. The seven rungs that slipped through are all inside the
E4M3 range, so the fix does not retroactively refuse an artifact that served
correctly; there is a test that says so by name.

`contract_version` 1 → **2**, with a `changelog` in the file itself. The
`schema` string stays `tessera.runtime-contract.v1`, and that is a decision, not
an oversight: see below.

**This file has a consumer in another repository, and the effect on it was
measured, not assumed.** PrismaQuant reads this exact packaged JSON through
`importlib.resources` (`prismaquant/tessera_render.py::tessera_serving_contract_path`
→ `prismaquant/gridbook_lane_eligibility.py::load_published_formats`), and its
`resolve_payload_rung` reads `reader_rate_range_q256` in production
(`gridbook_lane_eligibility.py:1025`) to decide whether a format name's rate
resolves at all. So the two halves of this change land on it differently:

* **The widened E4M3 range changes rate resolution, as it should.** Measured
  against the v2 file: `TESSERA_E4M3_K1_R749` now resolves to rate 749 where
  under `[1024, 1024]` it resolved to `None`; `R2049` still resolves to `None`;
  `TESSERA_E2M1_K2_R768` still resolves to `None`.
* **It does not widen admission, which is the point of separating the two
  names.** `tessera_lane_attested` is still `False` for
  `R1024`/`R749`/`R1262`/`E2M1_K2_R896` — the `lane_eligibility` cells'
  `rungs_q256` did not move, and PrismaQuant's Tessera serving pin is a
  fail-closed sentinel until Rob cuts a release tag. Decodable widened;
  attested did not.
* **The rename would have broken a v1 reader, so it is additive.** Dropping
  `candidate_rungs_q256` while `schema` still said `v1` is the same
  "current and wrong" fault this section exists to close, one field over. With
  the alias, PrismaQuant's Tessera-facing suite is **60 passed, 0 failed**
  against this tree (`tests/test_tessera_lane_admission.py` +
  `tests/test_tessera_formats.py`); without it, one test fails with a
  `KeyError`. The alias is dropped when the `schema` string moves to v2, which
  is a coordinated change across both repositories and not this one.

---

## 11. The exporter may not write a rung the reader does not publish (2026-09-02, #41)

Section 10 gave the *consumer* the published set. The producer never read it.
`export.wire_recipe` writes the WINDOW body over LUT16 for every `E2M1x2` unit
below the coset trellis's cap — the shipping default under q256 896, not an
exotic setting — and §10's table says that body has no served decode. So a
legal low-rate unit encoded fine, went into a checkpoint, and was refused at
**load**, after the encode, on the operator rather than on the exporter.

Reproduced before it was fixed, on a `[64, 64]` unit at `E2M1x2` q256 512:
5890 wire bytes written by `encode_linear_planes`, then

> `tessera target 'probe.E2M1x2.q512': q256=512 is outside the rungs this
> build's decoder reads for TESSERA_E2M1_K2 — [896, 896] (every integer).`

**What the wire can emit, and what of it can be served.** Enumerating
`export.recipe_table` per grid against the published ranges gives two gaps, not
the five a `wire_recipe`-only enumeration reads (three until #9 gave the `BF16`
grid its route):

| grid | emitted | encodable | published | verdict |
|---|---|---|---|---|
| `E2M1` | 1…1024 | 256…768 (TCQ, cap 3) | none | **gap** — the whole grid |
| `E2M1x2` | 1…895 WINDOW | 128…895 | — | **gap** — the sub-cap default |
| `E2M1x2` | 896 TCQ | 896 | [896, 896] | served |
| `E2M1x2` | 897…1024 TCQ | **none** | [896, 896] | not a gap: `bresenham_rate_schedule` refuses rate 8 (`max_trellis_rate = native − 1`) |
| `E4M3` | 1…255 WINDOW | **none** | [256, 2048] | not a gap: rate 0 is below the shaped domain |
| `E4M3` | 256…2048 WINDOW | 256…2048 | [256, 2048] | served, and the bounds coincide by construction |
| `BF16` | 1…4096 WINDOW | 256…4096 | [256, 4096] | served since #9 — `bf16_route` decodes the same window body, and q256 1792 is attested |

`wire_recipe` returning a recipe is not the same as the encoder building one:
`E2M1x2` above the cap and `E4M3` below 256 are refused by the grammar, so no
such wire can be written and neither is a producer/consumer mismatch.

**The gate.** `serving.scheme.refuse_unserveable_wire` — beside
`validate_tessera_scheme`, off the same `ROUTES` table and the same packaged
`runtime_contract.json`, importable without torch — is the export-time half of
the load-time rule. It resolves the route from `ROUTES`, the rate range from
`contract.reader_rate_grid` / `reader_accepts`, and the plane and body from the
route's own entry. **It hardcodes no cap**: the day a measured range widens,
the JSON changes and the gate follows it (principle 14). Its refusal names the
unit, the `(route, grid, q256)`, the published range, and that the rung is
still legal to *encode*.

**It is a serving-boundary refusal, not an encoder one.** This is principle 9's
one carve-out — a measured platform fact, the pinned runtime has no native
route for these bytes — so it sits where a checkpoint declaring
`quant_method: "tessera"` is written (`tessera.export_serving`'s
`check_recipe` -- `experiments/export_tessera_serving.py` before tessera#687
moved the exporter into the package -- on the default `(grid, q256)` and
every `--plan-json` override, before the first encode). `wire_recipe` and `encode_linear` keep their full
range: the rate-frontier work encodes sub-cap `E2M1x2` constantly and is
untouched.

**Fail closed with an explicit, stamped override.** `--allow-unserveable`
writes the wire anyway as a research artifact and stamps every refusal verbatim
into the manifest's new `serving_gate` block (`contract_version`,
`allow_unserveable`, `unserveable_overrides`). Two real workflows need it
today: `--grid E2M1`, whose route holds the grid while the contract publishes
no measured range for it (item 2 below), and sub-cap `E2M1x2`, which the rate
frontier encodes constantly. In both the `--stock-twin` is what gets served.
`--grid BF16` was the third until #9 landed its route; it now passes the gate.

**Item 2 of #41 — `ROUTES` lists `E2M1`, the contract publishes no range for
it — is kept as a deliberate disagreement**, pinned by a test rather than
resolved. They are not two statements of one fact: `ROUTES["grids"]` says what
the decoder *holds* (the NVFP4 decoder is arity-parametric — `arity` is a
runtime scalar into the span-2 decode kernel, retired as
`tessera_nvfp4_decode_span2_out` on 2026-09-16 and now
`tessera.kernel_a4`'s, and `lane_planes.build_anchor_values` reads it off the
forest's grid), while
`formats[]` says what has been *measured through* it. That is the same pair of
claims `tensor_parallel` already separates as `max_world_size` beside
`loader_axes`. Deleting `E2M1` would delete a true statement about the decoder;
publishing a range for it would invent an attestation nobody measured. So an
arity-1 wire is refused for serving until someone measures one.

`tests/test_serving_export_gate.py` enumerates the rungs from
`export.recipe_table` / `rung_ceiling` and the grids from `control.GRID_NAMES`,
so a new grid or a moved recipe boundary is a failing test rather than a
checkpoint that refuses at load. On the pre-fix tree the exporter accepted 41
`(grid, q256)` probes and the loader refused **32** of them, across all four
grids; after, it accepts 9 and the loader accepts every one.

**The other two writers of a `quant_method: "tessera"` artifact were checked
and need no second gate.** `retarget_checkpoint_to_plugin.py` already runs
`validate_tessera_scheme` on every group it rewrites (`:60`). The native
container written by `export.py::_write_config` — `tessera_config.json`, used
by `export_glm53_tessera.py` and merged by `merge_tessera_parts.py` — carries
`quant_method: "tessera"` but no `config.json`/`quantization_config`, so vLLM
never selects the plugin on it; it is a research container that reaches the
plugin only through one of the two gated scripts. The gate therefore sits on
every path into a checkpoint the plugin loads, and on no path that only
measures.

## 12. A fused module's roles share a route, not a rate (2026-09-03, #37)

Section 11 fixed a producer that wrote more than the consumer reads. This is
the mirror image: a producer that wrote **less**, on a rule the consumer never
had.

The exporter (`experiments/export_tessera_serving.py` at the time, now
`tessera.export_serving`) grouped a vLLM-fused module by `(grid, q256)`
equality — `len(recipes) == 1` — and passed any group whose members disagreed
through at source precision. Its own header stated the weaker and correct rule,
that the roles must share one *family*. One of the two was wrong, and the code
was.

**What the decoders actually do.** Every route reads a role from that role's
OWN manifest. `serving/fp8_route.py::prepare_tessera_fp8_module` calls
`prepare_window(unit.body_bits, unit.rates, unit.window_bits, unit.window_codes,
..., initial_state=...)` and `materialize_fp8(unit, parsed.forests, parsed.code)`
per role and concatenates; `serving/ops.py` keeps a `_PreparedRole` per role and
passes that role's own `rate`/`arity`/`memory`/`half` into
`_nvfp4_decode_span2_out`, writing `packed[row_offset : row_offset + n]`. The
only module-level facts either route uses are `columns` and — on NVFP4 — the
shared 16-entry global, which is carried by an **exact binade shift**
(`fused.shared_lut_global`), not by the rate. The module-level flattening was
never in the decoder: it was in the sidecar `scheme`, whose scalar `q256`
`parse_tessera_blob_for_scheme` then held every member to.

**The run.** `experiments/fused_member_rung_identity.py` encodes real
Qwen3-0.6B layer-0 `q/k/v` at three different rungs — 1024 / 900 / 1200 — packs
them into one `TSRFUSE1` container, and compares the fused decode against the
same three roles decoded as three one-member modules (the shape the exporter
writes for an unfused Linear, i.e. the path the unrelaxed code took). Both
grids:

> `PASS  4194304 tile elements and 4096 row scales identical across the fused
> and the per-role decode, at rungs [1024, 900, 1200]`

It passes on the **pre-change** tree as well, which is the point: the decoder
was always per-member, and the relaxation removed a producer-side restriction
rather than adding a consumer-side capability.

**What moved.**

* `serving/scheme.py` publishes `FUSED_MODULE_FIELDS` — which of a module's
  scheme fields are shared and which are per member — and `validate_tessera_scheme`
  now runs the reader-range gate (§11) **per role**, naming the role in its
  refusal. `q256` is one polymorphic field, an int or a per-role list, rather
  than two fields that could disagree; the normalised scheme always carries a
  monomorphic `role_q256`. An old plugin reading a new checkpoint refuses on
  `q256 must be an integer` — fail closed, not fail quiet.
* `tessera.serving_plan.module_scheme_key(grid, q256)` (re-exported through
  `tessera.export_serving`; `export_tessera_serving.module_scheme_key` before
  tessera#687 moved the rule into the package) is the grouping key:
  `(family, grid, body, scale plane)`. Body and plane are **derived** from the
  rung by `wire_recipe`, not assumed constant, because `E2M1x2` writes the
  window body below the coset cap and the TCQ body at it — two decoders, so the
  key separates them and the relaxation cannot let a mixed-body group through
  the back door.
* `runtime_contract.json` goes to **contract v6** with a `fused_module` block
  (`schema`, `container`, `fields`, `sidecar_q256`, `mixed_rung_receipt`), and
  `serving/contract.py` validates that block against the `scheme` constants the
  loader itself gates on, so the published rule and the enforced rule cannot
  drift (principle 14, the same pattern as `loader_axes`). The change is
  additive: no family, rung range, `lane_eligibility` cell, world size or
  `quant_method` moved.
* `experiments/plan_from_layer_config.py` — the PrismaQuant → plan converter —
  carried the same `(grid, q256)` rule as a second statement, and would have
  refused before the exporter ever ran. It now **imports** `module_scheme_key`
  instead of restating it. A group whose members disagree on family, grid, body
  or plane, or a group with a member the allocation never priced, is still
  refused by default and still demoted to BF16 (with the demotion recorded)
  under `--allow-fused-disagreement`.

**The chain, measured.** A seven-rung mink allocation over Qwen3-0.6B layer 0
(`q/k/v` at 1083/920/1200, `gate/up` at 1107/1000, `o` 934, `down` 749) run
through converter -> plan -> exporter -> the plugin's own load path:

* the pre-change converter refuses it -- `2 fused module(s) do not share one
  (grid, q256)`, naming both modules; the current one plans 7 Tessera units and
  demotes none;
* the exporter writes 4 modules / 7 units, `q256=[1083, 920, 1200]` and
  `q256=[1107, 1000]` in the two fused schemes;
* `check_wire_against_plan.py` against the PrismaQuant-priced sidecar:
  **charged 61415424 bits = 3.904687500 bpp, emitted 61415424 bits =
  3.904687500 bpp, manifest 3.904687500** -- exact, per unit and in total, so
  #15's rule ("the bytes served are the bytes priced") holds unchanged now that
  the members are priced at their own rungs;
* both mixed-rung modules decode element-for-element to their roles decoded
  alone (`4194304` and `6291456` tile elements, and the row scales).

**What this does NOT claim.**

* Nothing was served. `fused_module.mixed_rung_receipt` is `false` **as a
  published value**, and stays false until a mixed-rung checkpoint has a served
  KL receipt.
* No rung range widened. NVFP4 publishes `E2M1x2` as the single point 896 and
  no range at all for `E2M1`, so a *within-family* rung disagreement is still
  refused there by the reader-range gate — the relaxation bites on `E4M3`
  [256, 2048] and `BF16` [256, 4096].
* Tensor parallelism was not exercised on a mixed-rung module; `max_world_size`
  is still `[1]`.
* The routed-MoE cell is untouched.

**The PrismaQuant half, not implemented here.** PrismaQuant's group knapsack
(`PRISMAQUANT_TESSERA_GROUP_KNAPSACK`, on by default) asserts in a comment that
one rung per group is not the serving constraint. Under principle 14 that
assertion should be a read: the Minkowski fold should be gated on
`fused_module.fields["q256"] == "per_member"` from the pinned contract, so a
future runtime that makes the rate module-level turns the fold off by measured
fact rather than by anyone remembering to. That is a PrismaQuant-side change and
is deliberately left to PrismaQuant; what #37 owed was the value to read, and
v6 publishes it.

## 13. The contract publishes what the plugin loads, not only what it runs (2026-09-03, #28)

`runtime_contract.json` said what a Tessera serve **executes** — the families,
their reader-rate ranges, the `activation_contract` each route feeds the GEMM,
the cells a receipt covers — and nothing about what it **loads**. Those are
different claims, and one consumer needs the second one.

PrismaQuant's `tools/serve_fingerprint.py` mechanises §7.4 of its architecture
doc: a KL is bit-identical inside one container session and drifts 4–8× across
them, keyed on whether a lane's CUDA `.so` was resident in the serving process.
Its `EXTENSION_PATTERN` is the alternation of basenames whose residency it
records. With no table here to read, PrismaQuant mirrored ours into its own pin
file (`serving_extension_basenames`) — a claim about **this** runtime,
maintained in the other repository, which is principle 14 read backwards.
Until 2026-09-03 that pattern named no Tessera extension at all, so a serve
running Tessera's own native span-2 decode fingerprinted identically to a stock
serve: the one lane whose whole point is a custom decoder was the one lane the
identical-residency rule could not see.

**`native_extensions` is the table.** `contract_version` 6 → **7**, additive: a
v5 or v6 reader ignores a new top-level key and no value it already read has
moved. (It was authored against v5 and landed after §12 took v6; the renumber
is bookkeeping -- the block, not the integer, is the claim, and PrismaQuant's
`contract_answer()` over the v6 and v7 payloads is byte-identical.)
Three entries today: the window GEMV, and since 2026-09-28 (tessera#640,
contract v42) the two libraries of the fused routed window MoE lane,
`tessera_routed_fused_e4m3` (route `TESSERA_FP8`, lane decoder
`native_routed_fused_window`) and `tessera_routed_fused_value` (route
`TESSERA_BF16`, lane decoder `native_routed_fused_window_folded`) -- one source,
`csrc/routed_fused_window.cu`, built once per window family by
`tessera.routed_fused`, loaded from `tessera.serving.moe_route` through
`native_window_moe.PackedWindowMoeBundles.adapter`, and substituting the
compact adapter's decoder in both residencies when the build cannot compile
it. Between 2026-09-16 and 2026-09-28 the table held the window GEMV alone.
The span-2 NVFP4 decoder's entry was removed with the A4 whole-weight
expansion — nothing under `tessera.serving` loads that library any more, and a
table entry for it would publish a `.so` no serve can map.

**It is a glob, not a basename, and the rule is a value.** An entry whose
module name carries a build-identity hash has no exact basename to publish, so
a consumer must know whether the published string is a stem, a prefix or a
pattern — the entry therefore carries `module_name_prefix`, `filename_glob`,
and `match: "basename_fnmatch"`, which names the rule a gate applies. (The
retired NVFP4 loader was the hashed-name case; the surviving window name is
exact, and the field's meaning is unchanged.) Prose saying "this is a prefix"
is exactly the kind of field principle 14 refuses.

**`when_unavailable` replaces the `optional: true` the issue proposed.**
`optional` conflates "this build may not have compiled it" with "the route runs
correctly without it", and answers neither question a fingerprint asks. What a
fingerprint needs is *which serve ran instead*, and that differs by residency:
`resident` decodes once at load and may substitute
`tessera.stock.materialize_stock`, stamping `decoder: "torch_materialize_stock"`
on every route record; `streamed` decodes inside a traced forward, where that
path's data-dependent shapes cannot run, and refuses outright. So an absent
`.so` together with a streamed route record is an *impossible pair* — a stronger
check than a boolean can express. `nvfp4_route` gates on this block
(`ext.substitutes_when_unavailable`) rather than on a mode comparison of its
own, the same way `loader_axes` is checked against `sharding.ROUTE_TP_AXES`.

**The table is derived, not typed beside the code.** `module_name_prefix` is the
constant `ext._load_locked` itself passes to `cpp_extension.load`, and
`validate_serving_contract` refuses any block that is not
`tessera.serving.ext.NATIVE_EXTENSIONS`. That makes the published name the
loaded name by construction — but it cannot tell whether the list is *short*.
`tests/test_serving_native_extensions.py` answers that separately: it walks the
import graph from `tessera.serving` (function-local imports included — 40
first-party modules today) and refuses any native-load site reachable from it
that the table does not declare, reading each site's module name out of the AST
and failing rather than skipping when it cannot.

**`tessera_window_gemv` is declared, and that is checked rather than asserted.**
The criterion for an entry is that the library can be resident in a *serving*
process. `tessera.kernel_window_gemv` JIT-loads one, and since issue #10 the
streamed FP8 route reaches it from `tessera.serving.fp8_gemv`, so it is
serving-side and the table carries it (module `tessera_window_gemv`, exact
name, loaded by `tessera.serving.fp8_gemv` for `TESSERA_FP8`, both residencies
substituting the torch window decode without it).
`tests/test_serving_native_extensions.py` answers that separately: it walks the
import graph from `tessera.serving` (function-local imports included — 40
first-party modules today) and refuses any native-load site reachable from it
that the table does not declare, reading each site's module name out of the AST
and failing rather than skipping when it cannot.

---

## Contract v11 — which Linears the runtime offers a quant config at all

Every other block in this contract answers *what the plugin executes*. This one
answers a question that comes earlier and that the plugin structurally cannot
answer for itself: **is it asked about this module?**

`LinearBase.__init__` takes `UnquantizedLinearMethod()` in the
`quant_config is None` branch **without calling** `get_quant_method`
(vLLM 0.28, `model_executor/layers/linear.py:258`). A model implementation that
builds a projection with `quant_config=None` therefore takes vLLM's own BF16
method, and no quantization plugin can refuse it, warn about it, or even see the
prefix. On the pinned GLM build (`prismaquant/glm53-mia-sm121:487ecf187`) that
is four sites: MLA at `models/glm5next/nvidia/model.py:331`, the whole KDA layer
at `kda.py:171-174`, the indexer's `wk_weights_proj` at `attention.py:263`, and
the entire vision tower at `model.py:1082`. Twenty of twenty-four Linear
patterns; four are offered.

The exporter planned all of them — 2-D, under `BODY_LAYER` — so it wrote wires
into modules the runtime never routes, and deleted the `<module>.weight` each
one wanted. Unlike issue #86's case that does not even end in a refusal.

**The table is derived, not typed.** `tools/tessera_construction_census.py`
builds the model exactly as the loader does (`initialize_model` under
`set_current_vllm_config`) on the `meta` device, so nothing is read and nothing
is allocated, with a probe `QuantizationConfig` that records every `(prefix,
layer class)` vLLM offers it; then it walks `named_modules()` for every
`LinearBase` and records the class, `quant_config is None`, and whether the
probe was offered that prefix. The receipts live under
`docs/measurements/construction/`; `contract.construction_entry_from_receipt`
generates the contract rows from them and
`tests/test_serving_construction.py` re-derives and compares, the same rule
`native_extensions` follows.

**It also publishes the geometry** (tessera#377): each row's `output_sizes` is
the runtime's output partition list for that Linear, which the exporter's
roles must match by position (`sharding.plan_shard`). `contract.output_partitions`
reads it per checkpoint module and `dense_ownership.partition_members` derives
the roles — cutting one source tensor by row where the runtime builds a
`MergedColumnParallelLinear` over it (LFM2 `short_conv.in_proj`), refusing a
checkpoint whose rows disagree with the attested list. See
`docs/ARCHITECTURE.md` §4.4b.

**It also publishes the naming bridge.** A row carries the model class's
`hf_to_vllm_mapper` (unstacked) and `packed_modules_mapping` — the very tables
`configure_quant_config` hands this plugin — because a producer writing
`config_groups` in the *checkpoint's* namespace has to apply them to know which
vLLM module it named. On GLM they are load-bearing:
`model.language_model.` → `language_model.model.`, and a KDA layer's
`q/k/v/b/f_a/g_a` merge into **one** `self_attn.in_proj_qkvbfg_a`, an MLA
layer's `q_a`/`kv_a` into `fused_qkv_a_proj`, and the indexer's
`wk`/`weights_proj` into `wk_weights_proj`. So the `qkv_proj` the exporter's
fused rule produced there named nothing at all — the `absent` verdict, a
different failure from `never_offered` and reported as one.

**Qwen is the control and the reason this went unseen.** `Qwen3ForCausalLM` on
`vllm/vllm-openai:latest` offers 4 of 4 Linear patterns, and every Tessera
artifact served so far is Qwen.

**An uncensused architecture is a gap, not a clearance.** `construction_entry`
returns `None`, and the exporter treats that as a refusal (`uncensused`) rather
than as permission — the honest direction, and the fix is to run the census in
the serving image and commit the receipt.

## 14. The expert route exists, and what that does and does not settle (2026-09-04, #5)

`tessera.serving.moe_route` landed. It has been **loaded and executed on the
pinned runtime** — vLLM's real `RoutedExperts`, its real `load_weights`, the
runtime's own fused-MoE modular kernel —
`docs/measurements/tessera-moe-route-load-2026-09-04.md`,
`experiments/results/moe_route_load_probe.json`. That receipt is the
load-and-execute contract and nothing more: no artifact, no served census, no
KL, and therefore **no `routed_moe` cell in `runtime_contract.json`**.

**The unit is the projection, not the group.** §2 called the MoE cell "the
per-expert 2-D projection weight" and that is what the wire is: one
`tessera.fused` container per expert per projection, which is the granularity
the checkpoint's tensors, the runtime's shard ids (`w1`/`w3`/`w2`) and
`tessera.moe_layout`'s cells all already have. The **group** (`w13` = gate then
up, `w2` = down) is how those containers stack into the tile the kernel reads,
and the sidecar declares the groups, not a container per group.

**Per group the sidecar declares a `wire_stride`, not a `wire_bytes`.** §9.6's
finding — the blob length follows the data, so `[E, 2, nbytes]` at one stride
is bytes that fit all but one row — is why. The stride is the parameter row
width; a blob's true length rides beside it; `moe_layout.unpack_moe_wires`
refuses a stride that is not the maximum its lengths imply, and that refusal
fired on real bytes in the probe.

**§9.1 held, until tessera#492.** At the time of this section the route was
`TESSERA_FP8` only. `scheme.MOE_BUILDERS` is the one home for which families
have an expert route; it now names `TESSERA_NVFP4` too (§16), and the clamp
finding this doc recorded is what that builder relies on rather than the
reason it was absent.

**§9.6 held, and is now load-bearing.** The wire parameter carries its **own**
`weight_loader`; twelve loader calls in the probe went through
`RoutedExperts.load_weights` into `w13_wire`/`w2_wire`, which the stack's own
loader would have dropped silently.

### What is still owed

1. ~~**The exporter's write half** (2-D unpacked source).~~ **Landed
   2026-09-04 — see §15.** It was not blocked on a measurement, and it is not
   any more: contract v11's `construction` block already carried
   `offered_non_linear` naming `...mlp.experts` / `RoutedExperts` on
   GLM-5.3-Flash, and `classify_construction` now reads that list as well as
   `offered`, which is what the stack's construction gate needed.
2. ~~**The packed 3-D source layout.**~~ **Landed 2026-09-04.** The plan now
   states one closed convention: `out_first_chunked` or
   `in_first_interleaved`. Exact config-derived shapes are checked before the
   first encode, canonical per-expert wires are emitted, and the convention is
   stamped into both scheme and manifest. Missing or invented conventions are
   still refused; neither orientation nor gate/up split is guessed.
3. **A served census and KL.** §0's encode table prices it: the E4M3 whole-expert
   rate is 1.611 Mparam/s on a held box, so one GLM-5.3-Flash-4layer MoE layer
   (288 experts x 3 projections) is ~72–75 min and all three are ~3.7 h, before
   two 45 GB serves.
4. **Expert parallelism, tensor parallelism inside an expert, and `streamed`**
   are refused by name in the route, not measured.
5. **Memory at scale, and it sums across layers.** The route holds the wires
   *and* the decoded tile until `process_weights_after_loading` returns — and
   vLLM runs that hook only after *every* weight has loaded, so every MoE layer
   holds both at once. At GLM's 288 experts that is roughly 11 GB per MoE layer,
   i.e. **~33 GB transient for the 4-layer cut** and not fitting at all for full
   GLM. The probe ran at 4 experts and says nothing about it. The fix is a
   design one — decode per layer as it loads, or stream — not arithmetic.
6. **The model-level load hop.** The probe drives
   `RoutedExperts.load_weights` directly. In a serve
   `Glm5NextForConditionalGeneration.load_weights` runs first and decides what
   is delegated; §9.6 measured the *expert-params mapping* as suffix-agnostic,
   not the model loader above it. Whether a `.wire`-suffixed expert tensor
   survives that hop is unmeasured, and it is the first thing a served attempt
   would find out.
7. **The compiled forward.** Everything here is eager. The dense lanes broke
   under vLLM's compiled forward once already
   (`vllm-compiled-forward-breaks-lane-hot-paths`), and the MoE route has the
   same shapes of hazard in it.
8. **`resident` becomes process-wide.** MoE refuses `streamed` and
   `TESSERA_SERVE_MODE` is one process-level setting, so a mixed artifact is
   resident everywhere, dense modules included, until (4) lands.

## 15. The exporter's write half, and what it changed (2026-09-04, #5)

The exporter (`experiments/export_tessera_serving.py` at the time, now
`tessera.export_serving` with the old path kept as a shim) writes routed-MoE
stacks from the **unpacked** (per-expert 2-D) source layout. Until this, §14's list item 1 was
the only thing standing between the route and an artifact, and it stood: no
Tessera checkpoint could contain a `routed_moe` stack, so nothing could be
served through the route, the sidecar or the parameter layout. Receipt:
`docs/measurements/tessera-moe-export-seam-2026-09-04.md`.

**The plannable unit is the stack.** A `--plan-json` entry keyed
`<moe>.experts` gives every expert of that stack one rung — the same
per-layer expert uniformity every allocator in this house already enforces, and
the granularity vLLM builds one method at. Naming a leaf is still refused, now
with the spelling that works rather than a note that the write half does not
exist.

**One container per expert per projection**, written as
`<moe>.experts.{e}.{proj}.wire`. The suffix is what makes
`RoutedExperts.build_expert_params_mapping` route it to `w13_wire`/`w2_wire`
(§9.6), and the parameter's own `weight_loader` is what makes the write happen.

**`wire_stride` is derived, not declared.** It is the maximum over the group's
blobs, computed after the last one is written — which is also when
`config.json` is written, so nothing waits on it. There is no stride a caller
could pass that the bytes could not contradict, and `unpack_moe_wires` refuses
one that is not what the lengths imply.

**The reader is the gate at write time.** The scheme dict goes through
`scheme.validate_tessera_moe_scheme` — the exact function `TesseraConfig`
calls — before `config.json` is written.

**The construction gate reads `offered_non_linear` now.** A `RoutedExperts`
stack is not a `LinearBase`, so the census records it in that list rather than
in `offered`, and `classify_construction` reading only the first called the one
module the expert route exists for `absent`. The attestation is the census
either way (principle 14); the classifier now reads all of what it wrote. On
GLM-5.3-Flash-4layer all three stacks resolve `offered`.

**Nothing that was written before moves.** `moe_plan_baseline.py`, master
against this branch: 5 of 72 rows, four of them refusal text and the fifth the
manifest's `routed_moe` block. Every byte-deciding row — the tensor digest, the
`quantization_config`, the `ignore` list — is identical, and an unplanned stack
is untouched.

**It was run and read back, on the CPU.**
`experiments/moe_write_readback_check.py` (4 experts, 128x64, E4M3 q256=1024)
writes a stack and then reads it with the plugin's own functions:
`validate_tessera_moe_scheme` accepts the scheme, all 12 containers parse
against their declared role, `unpack_moe_wires` round-trips them byte for byte
off padded rows, and `prepare_tessera_moe_experts` decodes to
`w13 [4, 128, 128] float8_e4m3fn` / `w2 [4, 128, 64]` with `[4, 128, 1]`
scales — the stock per-channel FP8 stack, whose worst relative error against
the source experts is **0.077**, the rung's own quantization error; a
transposed, interleaved or misrouted tile would sit near `sqrt(2)`, which is
what makes this a layout check rather than a shape check. The `wire_stride`
argument stops
being an argument there: at ONE shape and ONE rung the eight `w13` blobs run
21293..21297 bytes, a 4-byte spread, and the declared stride is the max. This
is the plumbing and the reader's acceptance; the CUDA encoder and the fused-MoE
kernel are not in it.

**And loaded and executed, on the GPU, from those same written bytes.**
`experiments/moe_route_load_probe.sh` (pool `2be23f3a9e9d`, sparky GB10 sm121,
the pinned image `prismaquant/glm53-mia-sm121:487ecf187`, `rc=0` in 100.8 s)
runs two arms in one process differing only in who produced the bytes. The
EXPORTER's arm loads through `RoutedExperts.load_weights` (12 calls onto
`w13_wire`/`w2_wire`), materializes `w13_weight [4, 512, 512] float8_e4m3fn`
byte for byte equal to `materialize_stock`, and executes through vLLM's own
`TRITON`/`TritonExperts` fused-MoE kernel to `rel_l2` 0.014298978779530125 —
identical to the probe-encoded arm, while the bytes on disk differ (999596 vs
998984). What is still NOT measured is a serve: no `routed_moe` cell, no served
census, no KL.

### What §14's list looks like now

1. ~~The exporter's write half~~ — **done**, unpacked source.
2. ~~**The packed 3-D source layout**~~ — **done** under the two explicit
   conventions in §14. A producer must name one; the tensor is never asked to
   prove facts its dimensions cannot carry. LFM remains the independent
   unpacked served-route target, while Qwen supplies the packed-source
   ingestion population.
3. **A served census and KL** — **still open, and it is the whole of what is
   left.** Costed in §0 and in the receipt: ~75 min of held-box GPU per
   288-expert stack, ~3.75 h for all three, plus a GLM teacher dump on the
   pinned image, because no GLM artifact has ever been served in this repo.
   `experiments/moe_export_glm_4layer.sh` and `experiments/moe_stack_plan.py`
   are the driver and the plan builder; neither has been run on the real model.
4-8. Unchanged: expert parallelism, TP inside an expert and `streamed` are
   refused by name; memory at 288 experts is a design item; the model-level
   load hop and the compiled forward are unmeasured.

## 16. The NVFP4 expert route (2026-09-14, tessera#492)

`tessera.serving.nvfp4_moe_route` landed. `scheme.MOE_BUILDERS` now names two
families, and `TesseraConfig.get_quant_method` dispatches a `routed_moe` stack
to its family's builder off that table the way a Linear is dispatched off
`ROUTES`; a family with neither goes through the FP8 builder's front door and
is refused there by name.

**What the builder is.** From `process_weights_after_loading` onward it is
vLLM's own `ModelOptNvFp4FusedMoE`: the same parameter set (`w13_weight`
`[E, 2N, K/2]` and `w2_weight` `[E, K, N/2]` packed nibbles, group-16 ue4m3
`w13_weight_scale`/`w2_weight_scale`, per-expert `w13_weight_scale_2 [E, 2]`
and `w2_weight_scale_2 [E]`, per-expert `w13_input_scale [E, 2]` and
`w2_input_scale [E]`), the same `convert_to_nvfp4_moe_kernel_format`,
`make_nvfp4_moe_quant_config` and `make_nvfp4_moe_kernel`, over the backend
the runtime's own `select_nvfp4_moe_backend` picks for `(kNvfp4Static,
kNvfp4Dynamic)`. §9.1's clamp finding is what it relies on: under GLM's
`swiglu_limit` the pinned oracle refuses an explicit non-clamp backend and
resolves `auto` to flashinfer CUTLASS on sm121
(`experiments/results/nvfp4_moe_oracle_probe.json`). Nothing here writes a
kernel; the A side is the kernel's own group-16 quantisation under the static
scale, which is why the executed contract is `ROUTES[TESSERA_NVFP4]
["activation_contract"]` and not a claim the module makes.

**What is ours.** The decode: each expert projection's container is parsed
and verified whole, cut to the rank by `sharding.shard_parsed_roles` on the
group's plan (rows of `w13`, columns of `w2` -- the row cut is what
tessera#492's select-pad threading made decodable), and decoded once through
`stock.materialize_stock` after a per-expert `fused.shared_lut_global` join of
the gate and up globals, so the two halves of one `w13` tile sit under the ONE
`weight_scale_2` the kernel reads (the stock method warns and takes column 0
when they differ -- a silent wrong answer this route does not produce). The
global handed over is the MULTIPLIER, the reciprocal of the stock tile's
`weight_global_scale` divisor; `tests/test_serving_nvfp4_moe_route.py` pins
that numerically through `stock_dequant`, not by name. The planes are dropped
per expert as it lands, so a layer holds the wires only transiently, one
projection at a time -- §14's item 5 does not apply to this route.

**The A-side scale is a checkpoint fact.** The exporter writes
`experts.{e}.{proj}.input_global_scale` beside each wire -- capacity over amax,
the dense route's `trellis_input_global_scale` quantity, one per expert
projection from `--input-scales` -- and refuses a stack missing any, by key,
at the unit. The loader routes it onto `{group}_input_global_scale` by the
same suffix-agnostic expert mapping that routes `wire`, inverts it once into
modelopt's `input_scale`, and refuses at finalize if any is missing rather
than quantising activations at 1.0. FlashInfer's CUTLASS path collapses the
per-expert values to ONE per projection group
(`amax_for_moe_activation_quant`): it takes the max of the loader's
reciprocal, so the executed global scale is the smallest per-expert
`input_global_scale` -- the layer's largest calibrated amax -- broadcast to
every expert. The per-expert tensors are what the stock method reads and what
a per-expert price describes, so priced == served only where the per-expert
amax spread is zero.

**What is refused, by name.** Expert parallelism and EPLB (the parameter is
`[E, ...]` by global id and the stride invariant needs every blob), a
residency other than `resident`, a non-gated MoE, another family's scheme, an
expert/hidden/intermediate width that disagrees with the sidecar or is not a
whole number of 16-wide groups per rank, an expert whose gate arrived without
its up, and any stock tensor name (`experts.{e}.{proj}.weight` and friends) in
a Tessera checkpoint. A ROW cut below the kernel's select byte is NOT on this
list: this route decodes expert wires through `materialize_stock`, which reads
codes, so it serves any cut `slice_unit` makes. The `arity * 8 * span` row
requirement (the span-L select plane packs eight super-symbols to a byte and
resumes none on a half-byte) belongs to the dense NVFP4 route's native span-2
decoder, where `lane_planes.require_native_select_plane_admission` refuses it by
name; that route's resident `when_unavailable` torch fallback packs no planes
and serves the cut too.

**With `research_selected_moe`.** The block covers the stacks it serves
(`ResearchSelectedMoeConfig.applies_to`: FP8/E4M3 and BF16/BF16) and nothing
else: an NVFP4 stack beside them takes its production builder, and a block
that names no stack it serves is refused at config parse.

**What this does not settle.** A builder is a dispatch fact. `runtime_contract.json`
v27 publishes it as such -- `formats[].structures` names the structures the
plugin dispatches for a family, checked against `MOE_BUILDERS` by the
validator -- and adds **no** `routed_moe` cell for `TESSERA_E2M1_K2`: a cell is
an attestation and comes from a container receipt. v28 publishes the two
cells, `tessera_e2m1_k2_routed_moe_sm121_{decode,batch}_resident`, at q256 896,
eager and resident, from a two-rank served census of the GLM-5.3-Flash 4-layer
stub (`docs/measurements/tessera-glm53-a4-stub-tp2-served-2026-09-14.md`). An
NVFP4 stack at q896 exports without `--allow-unserveable` and names both cells
in its manifest; any other rung still needs the override, stamped in the
manifest's `serving_gate`. Both cells grade `route_only` and carry no smoke
record. Tensor parallelism was not attested at v28 (`max_world_size` stayed 1,
although that receipt is a two-rank serve); v29 raises every family to 2 on a
world-size receipt that adds a KL against a single-rank arm
(`docs/measurements/tessera-glm53-a4-stub-tp2-world-size-2026-09-15.md`,
tessera#514), still `route_only`. That receipt's A4 excess over the BF16
control is a re-draw of quantization noise, not added error: the A4 stub's KL
against the BF16 reference is the same at both worlds
(`docs/measurements/tessera-glm53-a4-stub-tp2-excess-resolved-2026-09-17.md`).
The stub's attention backend refuses graph mode, so no compiled forward is
attested.

**Contract v39 (tessera#604, second half).** v39 withdrew the two v28/v32
cells. They named `(vllm.fused_moe.modular_kernel, torch_materialize_stock)`,
and this build's NVFP4 expert stack cannot make that launch: it always builds
the native grouped A4 stacks. The same ids come back on the GLM serving image
at q256 896 only, executing `(tessera.kernel_a4.a4_span2_grouped_gemm,
native_span2_grouped)`, TP 1, eager and resident
(`docs/measurements/tessera-glm-u1-census-2026-09-26.md`). The world-size and
KL receipts above measured the materialising launch and do not carry over.
Routed E2M1 at q256 128..768 is unattested until a census of the grouped
route earns it.


## 17. Packed T4 arithmetic contract (2026-10-07, issue 1007)

Owner: kernels. Date: 2026-10-07.
This section completes the conditional algebra, not a native qualification proof.
The supported native specification does not supply the required local inequalities.
Both `arithmetic_qualified` and `native_arithmetic_qualified` remain false.
Issue 1007 remains open. No diagnostic pass releases that hold.

### 17.1 Source custody and represented operands

The reader base is `bf7d4550b9aa1d1d124aae31b78d223635822b71` in pull request 1008.
Kernels owns `_rounding_gamma`, `dense_packed_fp4_operand_magnitude`,
`derive_packed_fp4_arithmetic_bound`, and `check_packed_fp4_arithmetic`
in `experiments/t4_code/bench_geometry_e2m1.py`.
The implementation first separates input validation from scalar arithmetic.
It preserves the existing conditional coefficient and all live measurement thresholds.

The executive source head is `18d09db865e27d40f2ef4d80f70811294f767ba4`.
Its public base is `af39d2162fb6b721368672df35adee9ca308514c`.
The executive parent owns dense integration and architecture.
The routed owner owns the native WINDOW library and route intake.
The encoder owner owns the prepared WINDOW units.
The qualification owner owns the finite fixture.
This correction changes none of those executive sources or inputs.

For each output, use the same represented codes and group scales on both sides:

```text
v_a, v_b in {+0, -0, +0.5, -0.5, +1, -1, +1.5, -1.5,
             +2, -2, +3, -3, +4, -4, +6, -6}
a_k = v_a[k] * s_a[floor(k/16)]
b_k = v_b[k] * s_b[floor(k/16)]
G_a = represented activation global; G_w = represented weight global
X*_k = a_k / G_a; W*_k = b_k * G_w
S = sum_k X*_k * W*_k = (G_w/G_a) * sum_k a_k*b_k
M0 = sum_k |X*_k| * |W*_k|
rendered_x[k] = fl32(a_k/G_a)
M_upper >= max_output sum_k |rendered_x[k]| * |W*_k|
```

Here `K` is the actual column count, not a padded or model-wide count.
The diagnostic fixes `G_a=896` and requests float32 output.
A group contains sixteen columns. Finite UE4M3 scales range from zero to 448.
Their smallest positive value is `2^-9`.
Negative scales, NaN scale encodings, and nonfinite globals are outside this theorem.
The theorem permits signed values and numerical signed zero.
It does not establish bitwise equality of signed zeros.

E2M1 has at most two significant binary bits.
A finite UE4M3 scale has at most four significant binary bits.
Thus an exact scaled operand has at most six significant bits.
An exact product of two scaled operands has at most twelve significant bits.
Its nonzero magnitude lies in `[2^-20, (6*448)^2]`.
The upper endpoint is `7,225,344`.
These exact products fit float32 without overflow or underflow.
This representability result does not specify native multiplication behavior.
E2M1 value 0.5 and small UE4M3 scales have subnormal encodings.
Parallel Thread Execution leaves the native treatment of subnormal inputs unspecified.
Normal expanded products therefore do not remove the native subnormal prerequisite.

The quantization error is separate:

```text
Delta_q = S - sum_k original_x[k] * original_w[k]
dx[k] = X*_k - original_x[k]; dw[k] = W*_k - original_w[k]
Delta_q = sum_k (dx[k]*original_w[k] + original_x[k]*dw[k] + dx[k]*dw[k])
arithmetic_error_native = native_output - S
arithmetic_error_reference = reference_output - S
```

The arithmetic allowance does not bound `Delta_q`.
The common activation quantizer adds no second bfloat16 input conversion term.
Equal represented operands require equal codes, scales, group placement, and globals.
The existing byte checks and grammar owners retain those refusals.

### 17.2 Instructions, operation counts, and precision terms

The reader uses `tl.dot_scaled` in `kernel_a4_wire._a4_wire_gemm_kernel`.
It initializes one float32 accumulator and executes `K/128` source dot calls per output.
Its block width is 128, at `kernel_a4_wire.py:259-265` in the reader base.
The existing grammar requires columns divisible by 128, at lines 238-239.
The bound interface now enforces that same column domain.
A Triton call is not a documented internal addition count.
Its compiled instruction list and local arithmetic contract are not established here.

The executive library names this exact Parallel Thread Execution instruction at source lines 2249-2257:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
    m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

The displayed line break does not form part of the instruction name.
The source constant `fp4::BK=64` appears at line 2180.
Each unsplit output has `K/64` instruction calls.
The two calls at lines 2848-2849 update different output columns.
They do not double one output's reduction length.
The Parallel Thread Execution shape establishes sixty-four represented products per call.
It does not establish sixty-four standard fused multiply-add operations.

Use these precision terms:

| Term | Value | Meaning |
|---|---:|---|
| `u_fp32` | `2^-24` | Float32 unit roundoff for round-to-nearest in the normal finite domain. |
| `epsilon_fp32=e` | `2^-23` | One full relative spacing budget in that domain. |
| `u_fp64` | `2^-53` | Float64 unit roundoff for round-to-nearest. |
| `epsilon_fp64` | `2^-52` | The retained full spacing budget for the positive magnitude contraction. |
| `u_bf16` | `2^-8` | Bfloat16 unit roundoff for round-to-nearest. |
| `tau_fp32` | `2^-150` | Half the float32 subnormal spacing under gradual underflow. |
| `tau_bf16` | `2^-134` | Half the bfloat16 subnormal spacing under gradual underflow. |

These values follow from the significand widths and minimum subnormal exponents.
They are not fitted constants.
The retained coefficient uses `e`, not `u_fp32`.
A full spacing budget does not become a native guarantee merely because it exceeds unit roundoff.

Each use of the following assumptions remains conditional:

- **N1, native dot:** each scaled product and accumulation step has local relative error at most `e`.
  Each output path has at most `2*K` such errors.
  Parallel Thread Execution supplies neither statement.
- **N2, divisions:** each actual float32 division has error at most two units in the last place.
  Each exact nonzero quotient is normal and finite.
  The CUDA table describes `__fdividef`; this source does not prove that both divisions lower to it.
- **R1, reference dot:** the actual float32 matrix product has at most `2*K` local errors per output path.
  Each local relative error is at most `e`.
  Disabling tensor-float32 does not specify its complete reduction algorithm.
- **M1, magnitude dot:** the positive float64 matrix product has at most `2*K` local errors per output path.
  Each local relative error is at most `epsilon_fp64`.
  The source does not inspect that matrix product's compiled instruction list.
- **W1, weight formation:** the float32 weight global is an exact power of two.
  Each nonzero stock weight is exact, normal, and finite after its scale products.
  This is a domain restriction, not a conclusion from finite output.
- **E1, reader epilogue:** the float32 multiply has local relative error at most `e`.
  Its exact nonzero result is normal and finite.
- **D1, intermediate domain:** every counted nonzero intermediate obeys its stated local model.
  Overflow and unsupported subnormal behavior do not occur.
  The current interface does not establish this fact from scalar magnitude alone.

The source count gives these conditional error budgets:

| Path | Product and sum budget | Division budget | Final multiply budget | Total |
|---|---:|---:|---:|---:|
| Native packed reader | `2*K`, assumption N1 | `2`, assumption N2 | `1`, assumption E1 | `2*K+3` |
| Float32 reference | `2*K`, assumption R1 | `2`, assumption N2 | `0`, restriction W1 | `2*K+2` |
| Positive magnitude | `2*K`, assumption M1 | `0` | `0` | `2*K` with float64 epsilon |

The division budget follows from two units in the last place, not two source divisions on each path.
There is one normalization division on each path.
The native ratio is `fl32(G_w/896)`.
The reference activation is `fl32(a_k/896)`.
The reader stores `acc*ratio` as float32, at `kernel_a4_wire.py:209-210`.
The conditional diagnostic has no output bfloat16 conversion.

### 17.3 Complete conditional gamma derivation

Set `gamma(n)=n*e/(1-n*e)` with `n*e<1`.
For each local factor, assume `|delta_j|<=w_j*e` and let `n=sum_j w_j`.
Single-spacing errors have weight one. A two-spacing division has weight two.
The product inequality `product_j(1-w_j*e)>=1-n*e` follows by induction.
Also, `1+w_j*e<=1/(1-w_j*e)`.
These inequalities give `|product_j(1+delta_j)-1|<=gamma(n)`.
Thus the weighted division term retains its higher-order contribution.
It does not require a first-order replacement of two errors by one.

Apply that result to each represented product's path through the sum.
The triangle inequality gives an absolute error of at most `gamma(n)*M0`.
This step is valid for arbitrary signs and cancellation.
It uses the sum of absolute products, not the final output magnitude.
A zero exact sum can therefore have a nonzero allowance.
When all represented products are zero, the conditional allowance is exactly zero.

Under N2, `|rendered_x[k]|>=|X*_k|*(1-2*e)`.
Under W1, the reference weight equals `W*_k` exactly.
Under M1, the magnitude helper establishes `M_upper` for those reference operands.
Therefore `M0<=M_upper/(1-2*e)`.
Under N1, N2, E1, R1, W1, and D1, the two output errors satisfy:

```text
|native-S| <= gamma(2*K+3) * M0
|reference-S| <= gamma(2*K+2) * M0
|native-reference| <= [gamma(2*K+3)+gamma(2*K+2)] * M_upper/(1-2*e)
rtol = 0
```

This is the retained allowance, not a native qualification theorem.
No empirical safety multiplier appears.
The scalar implementation rounds each gamma quotient, sum, coefficient, and final allowance outward.
For its standard epsilon inputs, the counted integers, products, and gamma denominators are exactly representable in binary64.
The helper refuses negative or noninteger counts and nonfinite or invalid precision terms.
It also refuses a nonpositive gamma denominator.

The packed interface accepts positive `K` values divisible by 128.
Its gamma domain also requires `2*K+3<2^23`.
The largest such length is `4,194,176`.
The next aligned length, `4,194,304`, refuses even when the magnitude is zero.
This gamma restriction is not a declaration that every smaller shape has a qualified device kernel.
Existing native shape and grammar refusals remain independent.

For magnitude operands, float32-to-float64 conversion is exact.
Nonzero products lie between `2^-298` and a value below `2^256`.
They are normal and finite in float64.
For this legal `K` domain, the ideal positive sum is also finite in float64.
Under M1, let `g64=gamma(2*K, epsilon_fp64)`.
The measured positive sum is at least `(1-g64)` times the exact positive sum.
The helper divides by an outward lower value of `1-g64` and rounds the quotient upward.
It refuses `g64>=1`, a nonfinite reduction, or a nonfinite allowance.
M1 remains explicit because precision alone does not name the actual reduction depth.

### 17.4 Scale domains, subnormal values, and overflow

A finite ratio alone does not establish a finite multiplication result.
The normal relative model requires normal finite exact normalization quotients and epilogue products.
The native sum also requires a bound on all internal intermediates, not merely its stored result.
The ideal unnormalized magnitude obeys `sum_k |a_k*b_k|<=K*7,225,344`.
This bounds exact sums for any order.
It does not establish N1, the native input rules, or unspecified internal transformations.

The power-of-two weight global must keep each reference weight representable exactly.
A general finite weight global can introduce a separate weight formation error.
Such an error is absent from the retained coefficient, so that case is outside W1.
The general fused role ratio has no fixed-896 assumption.
Its division error and output domain must use its actual globals.

Under supported round-to-nearest and gradual underflow, a scalar result obeys:

```text
|round32(z)-z| <= u_fp32*|z| + tau_fp32
|round_bf16(z)-z| <= u_bf16*|z| + tau_bf16
```

These inequalities include zero, cancellation into subnormal values, and a subnormal result.
They exclude overflow.
They do not describe the unspecified native matrix instruction.
Flush-to-zero needs a different absolute term, up to the minimum normal magnitude for a flushed result.
A flushed input also changes the represented product before accumulation.
The current relative-only allowance includes neither contribution.
The source therefore records `intermediate_domain_established=false`.
Finite inputs and outputs do not upgrade that field.

Float32's maximum finite value is `(2-2^-23)*2^127`.
Bfloat16's maximum finite value is `(2-2^-7)*2^127`.
For round-to-nearest, the bfloat16 overflow threshold is `(2-2^-8)*2^127`.
A magnitude strictly below that threshold keeps the conversion finite.
A proof for other modes must use their own thresholds.
The code retains its finite-output refusal, including equal infinities.
It does not claim that this refusal proves a finite intermediate domain.

### 17.5 Fused WINDOW boundaries and actual reduction lengths

This section inventories the executive source; it changes no fused threshold or interface.
The dense ratio appears at `routed_fused_e2m1.py:631` in the executive head.
Each role has its own represented `G_w/G_a` ratio.
The routed ratios use the same role-specific construction.
The two globals need not equal the diagnostic's values.

For `S_split` split items, let `nk=K/64`.
The actual length of split `s` is:

```text
K_s = 64 * (floor((s+1)*nk/S_split) - floor(s*nk/S_split))
sum_s K_s = K
1 <= S_split <= floor(K/128)
```

Source lines 2440-2441 define the chunk ranges.
Source lines 3577-3584 require at least two chunks per split item.
The native partial error uses `2*K_s`, conditional on N1 for that split.
It must not use `2*ceil(K/S_split)` or the full `2*K` as an actual split count.
The reduction at lines 2942-2945 performs `S_split` source additions from zero per output.
The first addition is exact under an ordinary nonflushing scalar model.
The conservative source budget counts all `S_split` additions.
Under that scalar model, combine partial errors with this recurrence:

```text
E_s <= gamma(2*K_s) * sum_{k in split s} |a_k*b_k|
E_sum <= sum_s E_s + gamma(S_split)*sum_s (|exact_partial_s|+E_s)
E_ratio <= d_ratio * |G_w/G_a|
E_scaled <= |G_w/G_a|*E_sum
            + E_ratio*(|exact_sum|+E_sum)
            + u_fp32*(|G_w/G_a|+E_ratio)*(|exact_sum|+E_sum) + tau_fp32
```

Here `d_ratio` needs the actual division guarantee and its valid domain.
Use `2*epsilon_fp32` only under N2 for that actual division.
Use the additive subnormal term only under its stated scalar mode.
The unsplit case has no split-reduction term.
Both dense cases apply one `__fmul_rn` and one bfloat16 conversion per output.
Source lines 2916-2918 and 2947-2951 define those operations.

The converter `bf16_bits_rn` calls `__float2bfloat16_rn`, at source lines 506-507.
Let `E` bound the float32 output error against exact represented result `S`.
Under finite round-to-nearest conversion with gradual underflow:

```text
|bf16(native)-S| <= (1+u_bf16)*E + u_bf16*|S| + tau_bf16
|bf16(native)-bf16(reference)|
    <= E_pair + u_bf16*(|native|+|reference|) + 2*tau_bf16
```

This follows from the triangle inequality, not a fitted bfloat16 tolerance.
Adjacent bfloat16 values can differ across a midpoint for arbitrarily small input error.
A midpoint cannot justify a Lipschitz bound proportional only to `E_pair`.
The diagnostic's float32 coefficient therefore cannot certify these bfloat16 boundaries.

The routed output has these additional operations per scalar output:

| Mode or stage | Operations in the executive source |
|---|---|
| Gate/up projection, mode 1 | One ratio multiplication and one bfloat16 conversion for each role. |
| Fused activation, mode 0 | Two ratio multiplications and two bfloat16 conversions, then exact float32 widening. |
| Activation clamp | One `fminf` for gate; one `fminf` and one `fmaxf` for up. |
| SwiGLU after the clamp | One sign negation, one `expf`, one float32 addition, one division, one `__fmul_rn`, and one bfloat16 conversion. |
| Down projection, mode 2 | One ratio multiplication, an optional router-weight multiplication, and one bfloat16 conversion. |
| Token sum | `top_k` float32 source additions from zero, exact bfloat16 widening, and one bfloat16 conversion. |

Source lines 2885-2892 define mode 0.
Source lines 2903-2904 define mode 1.
Source lines 2916-2918 define mode 2.
Source lines 1901-1915 define the fixed-order token sum.
The router weight acts on the input instead when the caller selects that existing top-one path.
That path adds its weight conversion and bfloat16 input product before the shared quantizer.
The source defines it at `routed_fused_e2m1.py:492-497`.

For finite inputs, the clamp is nonexpansive.
For SwiGLU, propagate errors through the actual operations, not through a linear matrix product coefficient.
For a multiplication, input errors `E_a,E_b` contribute:

```text
E_product <= |a|*E_b + |b|*E_a + E_a*E_b
local_RN_error <= u_fp32*(|a|+E_a)*(|b|+E_b) + tau_fp32
```

For a division with `|b|>E_b`, input errors contribute:

```text
E_quotient <= E_a/(|b|-E_b) + |a|*E_b/(|b|*(|b|-E_b))
```

Then add the actual division's local error bound.
For `exp(t)` on a finite interval, the mean-value theorem uses `exp(max_interval)` as its derivative bound.
Then add the actual `expf` implementation's local error bound.
The source name does not establish its compiler mode, error bound, or overflow domain.
These scalar facts remain unproved at this use.
Negation is exact for finite float32 values under ordinary scalar semantics.
A token sum uses its actual `top_k` count, not `K` or `S_split`.
Each bfloat16 boundary needs the separate term above.

The second activation quantizer consumes the bfloat16 SwiGLU result, at executive Python lines 503-508.
Different activation codes produce different represented down operands.
That difference belongs to quantization error, not a same-operand arithmetic allowance.
The native prerequisite already blocks qualification before this nonlinear composition can certify a full routed path.
No finite random fixture supplies these missing arithmetic statements.

### 17.6 Exact missing statement and receipt scope

The Parallel Thread Execution specification's matrix section states:

> Accumulation of the intermediate values is performed with at least single precision.
> The accumulation order, rounding, and handling of subnormal inputs are unspecified.

Source: [Parallel Thread Execution matrix instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
The same section says element-wise multiplication uses specified precision.
It does not state the needed scaled-product local inequality.
Source: [CUDA mathematical function accuracy](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html).
That scalar table cannot define matrix-instruction arithmetic.

The missing native statement must cover every legal E2M1 code and finite UE4M3 scale:

1. State a local error inequality for the complete scaled product and accumulation operations.
2. State the maximum error depth for one output, including the incoming float32 accumulator and consecutive instruction calls.
3. State how subnormal codes, scales, products, sums, and results behave.
4. State an intermediate-domain guarantee or restrictions that exclude overflow and unsupported underflow.

A rigorous implementation-specific contract can supply the same facts instead.
Finite random samples, a fitted constant, or a float32 output type cannot supply them.
These are the original four missing facts, not new authority or seal gates.
The inherited failed-revision count remains one.
The inherited blocker clock remains `2026-10-06T18:25:28.755894+00:00`.

The retained receipts have these limits:

| Action prefix | Retained scope |
|---|---|
| `4ea427ad9294` | Packed reader at `91bdb523`: ten rows and seventy conditional comparisons. |
| `c6ef4e550613` | WINDOW fixture: thirty cases, sixty routed checks, and 450 dense checks. |
| `b060a2f2a98e` | Synthetic pure-seven timing, not arithmetic qualification. |
| `546f38019629`, `5001053baf49` | Historical CPU boundary and same-entry controls, not native proof. |

The scope note retains their full keys, payload digests, receipt digests, and result paths.
It is `/home/rob/fleet/inventory/kernels-1007-window-hold-scope-sol-20261007.md`.
The implementation record retains that scope and the original issue record without replacement.
It also retains new failing-before controls and the actual interface smoke.
No historical receipt covers this correction's new source head.
No source identity difference adds a refusal under development mode.
The existing code, scale, shape, finite-value, and discrepancy refusals stay intact.

### 17.7 Targeted native implementation attestation

Owner: kernels. Date: 2026-10-07. Arithmetic qualification remains false.
The sole implementation record is `kernels-t4-fp4-mma-attestation-sol-20261007`.
The conditional source remains frozen at `3fc62f102871e8a050d0e2eb5ef53c3c2f5745da`.
The original failed-revision count and blocker clock remain unchanged.

#### Method and retained evidence

The method follows Fasi, Higham, Mikaitis and Pranesh, PeerJ Computer Science 7:e330, 2021.
Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/ .
Use their targeted characterization method, not their older-device conclusions.
The suite uses no random samples and no fitted allowance.
Each test selects inputs that separate explicit arithmetic alternatives.
The implementation can refuse an unsupported model without calling the device defective.

The native instruction is:

`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`.

The probe uses the existing real fragment layout, with lane `4*g+t`:

- A register r, nibble i: row `g+8*(r&1)`, column `8*t+32*(r>>1)+i`.
- B register r, nibble i: column `g`, reduction index `8*t+32*r+i`.
- D register j: row `g+8*(j>>1)`, column `2*t+(j&1)`.
- A scale byte b: lane `(g,0)` names row g and group b.
- A scale byte b: lane `(g,1)` names row g+8 and group b.
- B scale byte b: lane `(g,0)` names column g and group b.

The suite retains these files beside each action:

- `inputs.bin`: every operand code, scale byte, and incoming accumulator bit.
- `inputs.json`: the exact binary layout, case names, and byte digest.
- `outputs.bin`: every native output bit, before any negative control.
- `references.json`: exact rational sums, output bits, and alternative model outputs.
- `negative-references.json`: the altered output and its exact failed inequality.
- `fp4_arithmetic_probe.cu`, `context.json`, and `probe.sass`: source, flags, compiler, device context, and native code.
- `memory-guard.json`: memory samples and the owned process-group termination record.
- `attestation.json`: each required result, refusal, model candidates, and qualification state.

The independent interpreter uses exact rational arithmetic and explicit binary rounding.
Its processor controls compare values with the host IEEE conversion and exact tie cases.
A processor interpreter failure supplies no native device verdict.
A layout failure supplies no arithmetic verdict.
A failed measured inequality refuses the device by name.

#### Six targeted properties

| Property | Targeted inputs | Tested premise |
|---|---|---|
| Product exactness | All signed E2M1 pairs; all finite A scale bytes against scale corners | No product narrows below its exact FP32 representation. |
| Scale order | Four distinct scale groups and small group sums below the alignment quantum | Group scales enter before the common accumulation, or through an equivalent exact group calculation. |
| Alignment width | Signed cancellation with exponent gaps from ten through forty | The cancellation cutoff supports the 36-bit alignment model. |
| Rounding mode | Signed sums, odd/even significands, ties, and above-half-spacing increments | The observed two-term rounding class belongs to the stated alternatives. |
| Subnormal handling | E2M1 half values, UE4M3 subnormal bytes, and FP32 accumulator subnormals | The measured input and output values remain present. |
| Intermediate domain | Maximum products, full atoms, seven carry bits, and cancellation permutations | The supported model has no narrow intermediate overflow or premature normalization. |

Product scaling and exact group scaling can be observationally equivalent.
The suite does not infer a physical circuit order when the represented results agree.
The report retains each compatible model instead of inventing a unique circuit.
An empty candidate set means that the current implementation model is unsupported.
It does not establish a defective GPU.

#### Explicit implementation inequalities

Let `epsilon=2^-23`, and let `p_k=a_k*b_k*sA_g*sB_g` for group `g=floor(k/16)`.
The contract uses these inequalities:

`|native_product(p_k)-p_k| = 0`.

`q=2^(E-35) <= 2^-35*max_i |x_i|`, where E is the largest summand exponent.

`|aligned(x_i)-x_i| < q` for a nonzero discarded part.

`|normalized(z)-z| <= epsilon*|z|` within the finite normal domain.

`|native_atom(C,p)-[C+sum_(k=0)^63 p_k]| <= (65*2^-35+2^-23)*S`.

Here `S=|C|+sum_k |p_k|`.
An atom has at most 65 aligned summands and one final normalization.
The coefficients count 65 alignments and one FP32 normalization.
They are not fitted multipliers.
The truncation model does not enlarge any aligned summand's magnitude.
Thus the exact aligned sum has magnitude at most S.
The 65 alignment errors contribute less than `65*2^-35*S`.
The final normalization contributes at most `epsilon*S`.
Their sum gives the atom inequality, including cancellation and exact zero.

Exact group formation does not require another error term.
Each raw E2M1 product is a multiple of one quarter and has magnitude at most 36.
A sixteen-product raw group has magnitude at most 576.
Its integer numerator needs at most twelve bits.
Two UE4M3 significands contribute at most eight further bits.
The scaled group therefore fits exactly within FP32's 24-bit precision.
The group-first model aligns at most five summands, which the 65-summand bound also covers.

The targeted results test each premise and retain its alternative outputs.
They do not prove every hardware input or every future compiler build.
Required reviewers decide whether the measured model supports the stated implementation domain.
No source approval or processor pass alone supplies that acceptance.

#### Reduction-length derivation

Let `L=K/64`, `M=sum_(k=0)^(K-1) |p_k|`, and `eta=65*2^-35+2^-23`.
The initial accumulator is zero.
Let `E_j` bound the error after j native atoms.
The atom inequality gives:

`E_j <= (1+eta)*E_(j-1)+eta*M`, with `E_0=0`.

The finite geometric sum gives:

`|native_dot-exact_dot| <= [(1+eta)^L-1]*M`.

The API evaluates that expression with exact rational arithmetic.
It rounds the reported absolute bound outward to binary64.
It does not fit a multiplier to observed errors.
The bound depends on the attested contract, FP32 precision, and actual reduction length.
The source performs K/64 native atoms, not one error per abstract source dot call.

#### Domain and actual boundaries

The domain has E2M1 codes 0 through 15 and finite unsigned UE4M3 bytes 0 through 126.
It requires a zero initial accumulator and a positive K divisible by 64.
Every nonzero scaled product is a multiple of `2^-20`.
Its magnitude is at most `36*448^2`.
A native sum from zero remains on that lattice under the supported alignment model.
Thus a nonzero raw accumulator cannot become an FP32 subnormal.
Subnormal accumulator probes remain useful controls, but the derived dot domain does not need that extension.

The domain guard requires `(1+eta)^L*M <= FP32_MAX`.
This bound excludes overflow on each prefix, not only at the final output.
Signs and cancellation do not relax that guard.
The implementation supports exact zero without an artificial absolute floor.
Signed-zero bit semantics are not part of the value inequality.
NaN scales, nonfinite values, and arbitrary initial accumulators are unsupported.

The raw instruction has no normalization division and returns FP32 directly.
Its normalization term and separate output-conversion term are therefore zero.
The optional represented output scale adds only the actual scalar multiplication term:

`E_scaled <= |r|*E_native + 2^-24*|r|*(M+E_native) + 2^-150`.

The optional bfloat16 conversion adds only its actual boundary term:

`E_bf16 <= E_input + 2^-8*(M_input+E_input) + 2^-134`.

The small absolute terms are half the minimum subnormal spacing, not empirical allowances.
These scalar boundaries use their explicit round-to-nearest contract.
They are separate from the characterized matrix instruction.
No activation quantization term belongs to arithmetic on equal represented operands.
No division term applies when a caller supplies an already represented scale.
A caller that computes a scale or changes the reference must account for those actual operations separately.

This attestation does not cover an uncharacterized library reference GEMM.
It does not qualify a complete fused network path with SwiGLU, router weights, split sums, or token sums.
Section 17.5 retains those distinct boundaries and reduction lengths.
The existing packed comparison remains a conditional diagnostic.
Its allowance and every live measurement path remain unchanged.

#### Fail-closed qualification and failure-first controls

`require_probe_contract` requires all six measured properties and the layout controls.
`require_t4_device_qualification` also requires kernels parent review and independent review.
Every emitted report keeps `arithmetic_qualified: false` before those reviews.
A required failed probe refuses T4 on the named device.
A meaningful negative control changes an exact native product output before analysis.
The original captured output remains unchanged in `outputs.bin`.
The altered result must fail the same product inequality and device gate.
The record distinguishes that injected failure from a failed device probe.

PrismaBuild executes the same entry point on the processor before a changed GPU invocation.
Each action retains fresh disk admission, declared resources, the memory guard, and completion custody.
Short GPU actions use ordinary exclusive admission at priority zero.
They request no quiet-host or measurement-class certificate.
The report adds no seal, source-identity refusal, serving default, pin, artifact change, or row admission.
