# Tessera: serving contract, MoE cell, and export gate

Status: **decisions**, 2026-09-01. Written before the render mechanism so the
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
