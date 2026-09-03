# Tessera: serving contract, MoE cell, and export gate

Status: **decisions**, 2026-09-01; sections 1 and 3 **superseded 2026-09-02**
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
| encode throughput, GLM expert shape | **23.3 Mparam/s** (0.36 s per 2048x4096) | `experiments/encode_throughput_glm_expert.py` |
| GLM routed-expert encode | **3.72 h one box / 1.86 h two** | same, over 311,653,564,416 params |
| encode power | 47 W of ~140 W envelope | same; ~3x headroom, not a bottleneck |

The encode campaign is an afternoon. Encode cost is **not** what gates this
work, which removes the main argument for building the serving backend first.

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
> the trellis, is what floors the window body above ~6 bpp.  The BF16 family is
> `unattested` in `runtime_contract.json` until a container receipt covers it;
> a family with no receipt publishes no `lane_eligibility` cell.
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
   enough to iterate, exercises the packed-expert cell) before GLM. GLM-5.3 is
   the campaign, not the testbed.
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
`FLASHINFER_CUTLASS`, `VLLM_CUTLASS`, `MARLIN`, `EMULATION`, `HUMMING`. Which
of those is backed on sm121 is **not measured**.

**Scope: build `487ecf187`, this model's config.** This does not contradict the
route status recorded for GLM NVFP4 MoE elsewhere, which was measured on a
config without a `swiglu_limit`, and it does not generalise to GLM proper
without re-measuring. It has one consequence that is not scoped, though: the
sentence in `serving/config.py`'s MoE refusal that says NVFP4 W4A4 "needs
`--moe-backend flashinfer_b12x` on GB10" is an *asserted* runtime claim of the
kind principle 14 forbids, and on this model it is false. It is corrected when
the MoE route lands, not before, so that the correction and its evidence travel
together.

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
  second exclusion list.
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

Measured after the fix: **49 dense Linears (was 0), 2592 routed expert leaves
across layers [1, 2, 3], 0 packed stacks.**

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

The packed path's tests are therefore **synthetic**: no packed-expert source is
at hand, so they fix the contract, not agreement with a real checkpoint.

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
| `TESSERA_E2M1_K2` | `E2M1x2` | **[896, 896]** | 1 | *above*, the same grammar (rate 7 of arity-2 native 8, so q256 ≤ 896); *below*, the native decoder, which serves the span-2 TCQ body only — under 896 `wire_recipe` writes a WINDOW body and `ops.prepare_tessera_module` refuses it by name. |

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
