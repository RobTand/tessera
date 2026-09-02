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
> and the streamed residency mode is the kernel lane's shape: the body stays
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
