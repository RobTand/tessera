# The first allocated Tessera artifact, served

*2026-09-02. Tessera `11b007c` (worktree
`agent-a6d34c0d5bba700a6`); the allocation from PrismaQuant worktree
`/home/rob/pq-wt/tessera-continuous` @ `5a320c7`. Model Qwen3-0.6B, 28 decoder
layers, 196 body Linears, 112 vLLM modules.*

PrismaQuant's continuous Tessera menu produced per-Linear rung assignments at
three byte budgets. This receipt takes them through plan → export → serve → KL
against the BF16 teacher. These are the first Tessera checkpoints served whose
rungs were **chosen** rather than set uniformly.

**The loop closes end to end, and the allocation loses.** At matched bytes the
allocated 4.0-bpp checkpoint reads **KL 0.3485** against the byte-matched
uniform control's **0.1746** — 2.0x worse — and it loses at all three budgets
(2.33x at 3.0, 2.88x at 5.0). The bytes are right to the bit, the accounting is
right to the bit, and the routing is 100% on the declared family.

**The loss is in the cost model, on the units the cost model priced.** A
separator pair serving only the seven Linears PrismaQuant actually measured,
everything else BF16 in both arms, reads **0.02517 allocated vs 0.01306
uniform** — **1.93x against the allocation**, which is 95% of the whole-body gap
in log terms. The one-layer coverage assumption is a real second problem (§7
shows re-weighting by each layer's own `h_trace` already inverts the verdict to
1.205x) but it is the smaller one. What failed first is the currency: this
campaign's `0.5 · h_trace · output_mse` on 256 activation rows says the
allocation is 1.13x *better* on exactly the units where serving says it is 1.93x
worse.

---

## 1. What was allocated, and what could be built

PrismaQuant priced **one decoder layer** (layer 0, 7 Linears, `--layer-stride
28`) and allocated over it. Its three `full` arms at 3.0 / 4.0 / 5.0 bpp are
each **single-family**: every unit is `TESSERA_E4M3_K1`, the FP8/W8A8 route. The
allocator never chose `TESSERA_E2M1_K2` at any of the three budgets, so the
mixed-**family** checkpoint this exercise set out to build does not exist to
build — the allocation is mixed-**rung**, not mixed-family. At 4.0:

| role | rung (q256) | body bits/weight | modules |
|---|---|---|---|
| `down_proj` | R749 | 2.973 | 28 |
| `o_proj` | R934 | 3.680 | 28 |
| `q_proj`, `k_proj`, `v_proj` | R1083 | 4.269 | 84 |
| `gate_proj`, `up_proj` | R1107 | 4.363 | 56 |

No fused-sibling disagreement arose: the allocator's DP already solves over
fused super-items, so q/k/v share R1083 and gate/up share R1107 by construction.
Nothing had to be passed through as BF16 for disagreement, and no non-Tessera
quantised rung appeared, so nothing had to be refused.

**Coverage is an extrapolation, and it is stamped as one.** Seven priced units
cannot fill 196. `plan_from_layer_config.py --cover broadcast-by-role` applies
each *role's* layer-0 assignment at every depth and writes
`coverage.extrapolated: true` into the sidecar. It is exact for **bytes** —
every Qwen3-0.6B decoder layer has identical shapes, so whole-body bpp equals
layer-0 bpp to the last digit — and it is an assumption about **quality**, which
§7 measures and finds wrong, though not as wrong as the cost model underneath
it.

## 2. The plan converter

`experiments/plan_from_layer_config.py` reads a PrismaQuant `layer_config.json`
and writes the exporter's `--plan-json` plus a provenance sidecar
(`tessera.plan_from_layer_config.v1`). It:

* maps `TESSERA_E4M3_K1` → grid `E4M3` and `TESSERA_E2M1_K2` → grid `E2M1x2`,
  cross-checking the format name against the entry's own `tessera_family` and
  `tessera_body_rate_q256` fields rather than trusting one of them;
* turns a BF16 passthrough choice into a plain BF16 module, and **refuses
  loudly, with a count**, any non-Tessera quantised rung — one checkpoint cannot
  hold an NVFP4-via-compressed-tensors module and a Tessera wire at once, and a
  silent BF16 substitution there would be a different artifact than the one
  priced;
* enforces the fused-family invariant (q/k/v, gate/up). A **per-member (mink)**
  allocation — PrismaQuant's group knapsack gives one family per fused group and
  a rate per member — lands here, and the deliberate answer is to **refuse**:
  vLLM builds one method per fused module, and no single rate for the group is
  derivable from the objective (the members' rates differ precisely because
  their sensitivities do, so min / bytes-weighted / max are taste, not
  arithmetic, and any of them moves both the bytes and the loss off the point
  the DP chose). `--allow-fused-disagreement` writes the plan anyway, and what
  it writes is the plan that will **serve**: every member of a disagreeing group
  is named `"BF16"` — which is what the exporter does with it — dropped from the
  unit table and the charged-bits total, and the demotion recorded
  (`fused_disagreements[].planned_as`, `totals.demoted_to_bf16_params`).
  Corrected 2026-09-02 (Tessera issue #15): the flag used to write the members'
  Tessera rungs and price them as Tessera while the exporter passed the module
  through, so the sidecar whose stated job is that the bytes served are the
  bytes priced under-reported a seven-unit mink plan at 3.90 bpp against ~12 bpp
  served. A whole-GROUP option name (`TESSERA_E4M3_K1_G3`) is refused by name
  for the same reason: it is not a rung and PrismaQuant is meant to have
  expanded it to its members before the assignment was written;
* prices every unit through PrismaQuant's **own** accountant
  (`prismaquant.tessera_formats.artifact_bpp`) and writes the charge as an exact
  rational, so the byte check downstream is equality, not a float comparison;
* names **every** body Linear, including the ones the allocation does not
  quantise. That last point was a bug, found by the exporter refusing an export:
  an unnamed tensor is *not* a BF16 passthrough, it takes the exporter's
  `--grid`/`--q256` default (E2M1x2 q256=896), so a seven-unit plan silently
  described a 4-bit checkpoint nobody priced. Both coverage modes now write
  `"BF16"` explicitly and `unplanned_body_linears` is an invariant zero.

`tests/test_plan_from_layer_config.py` — 19 tests, all passing on the host venv
— covers the mapping, the refusals, the fused invariant and the mink demotion,
both coverage modes,
the broadcast's two shape refusals, the BF16-naming invariant, and
(skipif-guarded on the PrismaQuant tree) that the sidecar reproduces
PrismaQuant's own `body_assignment_payload_bits_total`.

## 3. The bytes served are the bytes priced

Two independent checks, both exact.

**Per-unit accounting** (`experiments/check_wire_against_plan.py`): the
sidecar's charged bits against the export manifest's `wire_bytes * 8`, per unit,
plus rung, grid and shape.

```
units compared        196
charged (PrismaQuant) 1761722368 bits = 4.000260417 bpp
emitted (wire)        1761722368 bits = 4.000260417 bpp
manifest wire_bpp     4.000260417
VERDICT: the bytes served are the bytes priced
```

All 196 units agree individually, not merely in total. The 3.0 and 5.0
allocations pass the same check (1321320448 bits = 3.000260417 bpp; 2202124288
bits = 5.000260417 bpp), and the layer-0-only export of §7 agrees over its 7
units (62918656 bits, 4.000260417 bpp).

**Byte identity of the wire itself.** Re-encoding layer 0's seven units at this
commit and diffing against the `.tessera` blobs PrismaQuant priced at Tessera
`3d419e7` gives **identical body bits, identical scale planes, identical decoded
FP8 tiles** for all seven, totalling 7864832 bytes = 62918656 bits — exactly the
charge above. The allocator's cost model and the exporter are looking at the
same bytes, not at two renderings that happen to agree numerically. Principle 8
holds on this path.

Weights-only throughout (no `--hessian`), because the allocation was priced
weights-only. An H-aware export would have been cheaper bytes bought with a cost
the allocator never saw.

## 4. The matched control

A uniform arm at the *same bytes* is the only honest comparator, so one was
built rather than borrowed. Searching q256 ∈ [250, 2048] for the E4M3 rung whose
param-weighted `artifact_bpp` over the model's own 196 shapes is nearest each
allocation gives R750 / **R1006** / R1262 for 3.0 / 4.0 / 5.0. At 4.0 the
control is **4.000521 bpp against the allocation's 4.000260** — the control is
marginally *fatter*, so every comparison below is conservative against the
allocation by 0.00026 bpp. Both arms come from the same exporter at the same
commit, same encoder, same weights-only setting; resident footprint is 8.025 bpp
for both (the FP8 route materialises to per-channel FP8 pairs at serve time).

## 5. Served

Vanilla vLLM 0.28 (`vllm/vllm-openai:latest`) with Tessera installed as a vLLM
plugin, GB10 sm_121, `--gpu-memory-utilization 0.30`, `--max-logprobs 1024`, one
serve at a time under `experiments/serve_lock.sh`. KL is exact-support top-1024
KL-vs-BF16 against `qwen_teacher_bf16_v028.json.npz` on
`corpus_qwen_n8_s512.json`, 4088 positions — the same instrument as
`tessera-dense-reach-fix-2026-09-02.md` and `tessera-stock-lane-served-2026-09-02.md`.

### Route census

`tools/tessera_route_census.py`, `--expect-modules 112`, on this commit:

| arm | mode | compiled | modules on family | other-route | contract | decoder | symbol | verdict |
|---|---|---|---|---|---|---|---|---|
| allocated 4.0 | resident | no | **112 / 112** | 0 | `fp8_per_token_dynamic` | `torch_window` | `torch._scaled_mm` | served |
| allocated 4.0 | streamed | no | **112 / 112** | 0 | `fp8_per_token_dynamic` | `torch_window` | `torch._scaled_mm` | served |
| allocated 4.0 | resident | **yes** | **112 / 112** | 0 | `fp8_per_token_dynamic` | `torch_window` | `torch._scaled_mm` | served |
| uniform R1006 | resident | no | **112 / 112** | 0 | `fp8_per_token_dynamic` | `torch_window` | `torch._scaled_mm` | served |

Each row holds in **both prefill (M64) and decode (M1)**, with `problems: []`.
**112 of 112 modules on the declared family in every arm.** Four distinct rungs
in one checkpoint do not fragment the route: they decode through one window
decoder into one FP8 GEMM. That is the mechanical claim this exercise existed to
test, and it holds — including under `torch.compile` and in streamed residency.

### KL vs the BF16 teacher

Exact-support top-1024 KL-vs-BF16, teacher-student intersection, lower bound by
the data-processing inequality; 4088 scored positions; `confident` is the 41.8%
of positions where the teacher's top-1 mass clears the tool's threshold.

| arm | wire bpp | resident bpp | KL all | KL confident | top-1 agree | KL p99 |
|---|---|---|---|---|---|---|
| **allocated 3.0** | 3.000260 | 8.025 | **2.2574** | 2.4834 | 28.33% | 9.527 |
| uniform R750 (matched) | 3.000521 | 8.025 | **0.9700** | 0.8410 | 50.83% | 4.922 |
| **allocated 4.0**, resident, eager | 4.000260 | 8.025 | **0.3485** | 0.2634 | 68.35% | 2.196 |
| allocated 4.0, streamed, eager | 4.000260 | 8.025 | 0.3485 | 0.2634 | 68.35% | 2.196 |
| allocated 4.0, resident, compiled | 4.000260 | 8.025 | 0.3466 | 0.2601 | 69.08% | 2.168 |
| allocated 4.0, stock twin (vanilla vLLM) | 4.000260 | — | 0.3469 | 0.2622 | 69.45% | 2.197 |
| uniform R1006 (matched), resident, eager | 4.000521 | 8.025 | **0.1746** | 0.1146 | 77.47% | 1.107 |
| uniform R1006, resident, compiled | 4.000521 | 8.025 | 0.1759 | 0.1176 | 77.23% | 1.142 |
| **allocated 5.0** | 5.000260 | 8.025 | **0.1622** | 0.1163 | 79.38% | 1.302 |
| uniform R1262 (matched) | 5.000521 | 8.025 | **0.0564** | 0.0389 | 86.40% | 0.438 |
| *separator:* layer-0 allocated, rest BF16 | (7 units) | — | **0.02517** | 0.02191 | 91.61% | 0.253 |
| *separator:* layer-0 R1006, rest BF16 | (7 units) | — | **0.01306** | 0.00909 | 93.93% | 0.101 |
| *separator:* layer-0 allocated, **stock twin** (vanilla vLLM, no plugin) | (7 units) | — | 0.02533 | 0.02218 | 91.66% | 0.272 |
| *mechanism:* layer-0 R1006 with `down_proj` at R749 | (7 units) | — | 0.02261 | 0.01784 | 92.32% | 0.236 |
| *mechanism:* layer-0 allocated with `down_proj` at R1006 | (7 units) | — | 0.01554 | 0.01352 | 93.96% | 0.138 |

**The allocation loses at every budget, at matched bytes**: 2.33x at 3.0, 2.00x
at 4.0, 2.88x at 5.0 — and 1.93x on the seven units it actually priced, with
everything else BF16 in both arms. The three whole-body losses are not a
different failure from the seven-unit one; §7 takes them apart.

The layer-0 stock twin is the checkpoint whose BF16 passthrough rides vanilla
vLLM's own `ignore` list — 189 of 196 body Linears passed through, `ignore`
naming the fused modules vLLM builds. It loads and serves with **no plugin at
all**, at 0.02533 against the plugin's 0.02517 on the same wires. Whether that
list is read by module name or role name is a claim about another runtime; it is
served here, not asserted.

### Student-vs-student compares (the same instrument, one student as reference)

| pair | KL | top-1 agree | reading |
|---|---|---|---|
| allocated 4.0 resident **vs streamed** | **0.000000** | 100.00% | the two residency modes are bit-identical, as the plugin's own receipt claims |
| allocated 4.0 **vs its stock twin** | 0.025466 | 91.46% | the Tessera decode path vs the same wires materialised for vanilla vLLM: a kernel gap, the same size as the 0.4660/0.4699 gap the stock-lane receipt records, not a wire disagreement |
| allocated 4.0 eager **vs compiled** | 0.028838 | 90.75% | |
| uniform R1006 eager **vs compiled** | 0.020028 | 92.42% | |
| allocated 4.0 **vs uniform R1006** | 0.368693 | 67.34% | the two byte-matched checkpoints disagree with each other about as much as the allocated one disagrees with the teacher |

The eager-vs-compiled numbers (0.0288 / 0.0200) are larger than the single
0.0176 the plugin receipt records for a uniform E4M3 wire. Reported, not
explained: nothing here isolates whether that is the four-rung mix, the
allocation's low rungs, or run-to-run scatter in the compiled path.

## 6. Allocator-predicted versus served

Five views of the same comparison, in the order the pipeline produces them:

| view | what it measures | allocated | uniform R1006 | allocated / uniform |
|---|---|---|---|---|
| allocator's surrogate | interpolated Δloss, layer 0 (`predicted_dloss`) | 11.903 | 13.906 | **0.856** (alloc better) |
| re-measured, layer 0 | every (unit, rung) re-encoded and re-scored, no interpolation | 9.806 | 11.031 | **0.889** (alloc better) |
| **served, layer 0 only** | KL-vs-BF16 on exactly the seven units it priced, rest BF16 | **0.02517** | **0.01306** | **1.93** (alloc worse) |
| depth-aware, whole body | the same measured costs, re-weighted by each layer's own `h_trace` | 343.7 | 285.2 | **1.205** (alloc worse) |
| served, whole body | KL-vs-BF16 on the shipped bytes | **0.3485** | **0.1746** | **2.00** (alloc worse) |

Read rows 2 and 3 together: they are the *same seven Linears*, at the same
rungs, on the same bytes. The cost model says the allocation is 1.13x better
there; serving those exact units says it is 1.93x worse. **The surrogate
mis-ranks two byte-matched assignments of a single layer by a factor of 2.2, and
it mis-ranks them in the direction that chose the shipped artifact.**

That relocates the failure. It is not the interpolation: re-measuring all 21
(unit, rung) pairs moves the ratio from 0.856 to 0.889, a 4% correction, and
PrismaQuant's own trusted-cost check reports the same agreement (predicted/true
0.824 for this arm). The menu interpolator is fine. It is the **currency**. This
campaign's cost is `0.5 · h_trace · output_mse` — `cost.pkl` stamps
`currency: output_mse_under_route_activation_contract`, `nsamples: 4`,
`seqlen: 512`, `max_act_rows: 256` — the additive-Fisher L1 form, not the
KL-adjoint AURA objective. A scalar Fisher trace times a 256-row output MSE does
not rank Tessera rungs the way 4088 positions of full-distribution KL do.

Two secondary consistency notes. The allocation's re-measured layer-0 advantage
(0.889) is *smaller* than the surrogate's (0.856): the surrogate is optimistic
about its own choice, in the same direction PrismaQuant recorded. And at 5.0 the
re-measured layer-0 ratio is already 1.022, while served the gap is the widest
of the three (2.88x) — the surrogate is *more* wrong at higher rate, not less.

## 7. Where the loss is

The whole-body arms change two things at once — the allocation, and the
broadcast of a one-layer answer to 28 depths. So a second pair was built that
changes only the first: the same seven layer-0 Linears, once at the allocator's
rungs and once at the byte-matched uniform rung, with **every other body Linear
BF16 in both arms** (wire 7864832 vs 7865344 bytes, the control again marginally
fatter).

```
layer-0 allocated, rest BF16    KL 0.025170   top-1 91.61%
layer-0 R1006,     rest BF16    KL 0.013064   top-1 93.93%
                                       1.93x against the allocation
```

**The allocation loses on the seven units it priced.** Against the whole-body
2.00x, the separator's 1.93x accounts for 95% of the loss in log terms; the
broadcast to 28 depths adds the remaining 5%. The failure is local, and it is in
the cost model, not in the coverage assumption.

### The trade the surrogate bought

Re-measuring both sides on layer 0 — every (unit, rung) re-encoded and re-scored
in the allocator's own currency, `0.5 · h_trace · output_mse` — shows exactly
what the allocation spent and where:

| role | rung | body b/wt | allocated Δloss | uniform R1006 Δloss | alloc / unif |
|---|---|---|---|---|---|
| `mlp.down_proj` | R749 | 2.926 | 1.686 | 0.457 | **3.693** |
| `self_attn.o_proj` | R934 | 3.648 | 1.644 | 1.030 | 1.597 |
| `self_attn.k_proj` | R1083 | 4.230 | 0.464 | 0.521 | 0.891 |
| `self_attn.q_proj` | R1083 | 4.230 | 0.164 | 0.208 | 0.790 |
| `self_attn.v_proj` | R1083 | 4.230 | 2.356 | 3.211 | 0.734 |
| `mlp.gate_proj` | R1107 | 4.324 | 1.542 | 2.469 | 0.624 |
| `mlp.up_proj` | R1107 | 4.324 | 1.950 | 3.136 | 0.622 |
| **total** | | | **9.806** | **11.031** | **0.889** |

It bought bits for `gate`/`up`/`v` (0.62–0.73x) by starving `down_proj` from
R1006 to R749 (3.69x) and `o_proj` to R934 (1.60x). In the surrogate's ledger
that trade nets out to a 1.13x win. Served on exactly those seven units, it is a
1.93x loss. **The trade is real; the exchange rate is wrong.** A scalar Fisher
trace times a 256-row output MSE prices a deep cut and a shallow gain as if they
were commensurable, and 4088 positions of full-distribution KL say they are not.

### Which move costs what

Two more layer-0 arms move only `down_proj` and hold the other six, which
decomposes the separator (bytes deliberately unmatched — these are mechanism
arms, not controls):

| arm | KL | Δ vs uniform | share of the gap |
|---|---|---|---|
| layer-0 uniform R1006 (baseline) | 0.013064 | — | — |
| uniform, `down_proj` alone cut to R749 | 0.022609 | **+0.009545** | **79%** |
| the allocation, `down_proj` alone restored to R1006 | 0.015541 | **+0.002477** | 20% |
| the allocation | 0.025170 | +0.012106 | 100% |

**The two increments add to 99.3% of the whole.** The served KL of this
allocation is, to within 0.7%, the sum of one deep cut and everything else — so
the failure is not an interaction the surrogate could not have seen; it is two
independently mispriced moves.

And the surrogate got the *sign* of the expensive one right: it charged
`down_proj` at R749 3.69x the uniform rung, the largest penalty in its ledger.
What it got wrong is the other side of the trade. Restoring `down_proj` leaves
the six remaining moves — `o_proj` cut to R934, `q`/`k`/`v` raised to R1083,
`gate`/`up` raised to R1107 — which the surrogate scores as a **1.30x net win**
(8.120 vs 10.574 with `down_proj` removed from the table above). Served, that
same bundle is a **1.19x loss**. Bits spent above R1006 buy far less than the
output-MSE curve promises, so the allocation financed a cut it correctly knew was
expensive with gains that do not exist. That is a rate-distortion slope error in
the surrogate's own currency, and it is why an allocator running on it will keep
proposing deep cuts.

### The second problem: layer 0 is not a representative layer

This one is real, independently measurable, and *not* where the loss lives — but
it would have bitten a correct cost model too. The cost campaign priced layer 0
only (`--layer-stride 28`), while the probe covers all 196 Linears, so the depth
profile of the empirical Fisher `h_trace` is already on disk (`probe.pkl`):

| role | layer 0 | median of layers 1..27 | layer 0 / median | rank at layer 0 | rank at depth |
|---|---|---|---|---|---|
| `self_attn.v_proj` | 3.83e4 | 5827 | **6.58** | 1 | 1 |
| `self_attn.o_proj` | 1.37e4 | 2217 | **6.16** | 2 | 7 |
| `mlp.up_proj` | 8730 | 5518 | 1.58 | 3 | 2 |
| `mlp.down_proj` | 8480 | 4315 | 1.97 | 4 | 3 |
| `mlp.gate_proj` | 2758 | 2933 | 0.94 | 5 | 6 |
| `self_attn.k_proj` | 1920 | 4299 | **0.45** | 6 | 4 |
| `self_attn.q_proj` | 1506 | 2936 | **0.51** | 7 | 5 |

`o_proj` is the second most sensitive Linear in layer 0 and the *least*
sensitive role at depth; `q_proj` and `k_proj` are half as sensitive at layer 0
as elsewhere. Holding each role's measured layer-0 rate-distortion curve fixed
and letting only `h_trace` vary with depth gives a whole-body prediction of
343.7 (allocated) vs 285.2 (uniform) — **1.205x against the allocation**, where
the layer-0 view said 0.889 for it. The same re-weighting inverts the verdict at
all three budgets: layer-0 ratios 0.885 / 0.889 / 1.022 become 1.289 / 1.205 /
1.488 at 3.0 / 4.0 / 5.0. So the broadcast is an error the allocator would have
made regardless; it is simply the smaller of the two.

That last figure is itself in the currency this section has just impeached, so
read it as a direction, not a magnitude. The **served** depth effect is the
ratio moving from 1.93 (seven units) to 2.00 (196 units) — **1.04x** — against
the 1.205/0.889 = **1.36x** the re-weighting predicts. The `h_trace` depth
profile is measured; the whole-body number derived from it is not validated by
anything here.

### The durable lesson

**PrismaQuant's own spine says this outcome is the system working.** *Surrogates
generate, real KL selects.* This campaign ran the generator and shipped its
output: the cost is `h_trace × output_mse` on 256 calibration rows, selection was
`SELECTION_MODE=surrogate`, and no real-KL gate stood between the DP and the
export. Every check that could pass without serving did pass — bytes exact to
the bit, accounting exact to the bit, 112/112 modules on the declared route —
and the artifact is still 2x worse than uniform. The gate is the only thing that
would have caught it, and the gate was not run.

Concretely, for the Tessera menu: **`h_trace × output_mse` does not rank Tessera
rungs at matched bytes on dense Qwen, and the validated-surrogate frontier is not
optional for it.** The cheapest form of that gate is already sitting in this
receipt — a byte-matched uniform arm, served, next to the candidate. It cost two
serves and it inverted the answer.

## 8. Against the standing references

Same instrument, same teacher, same corpus, from
`tessera-dense-reach-fix-2026-09-02.md` and `tessera-stock-lane-served-2026-09-02.md`:

| arm | bpp (wire / resident) | KL all | top-1 |
|---|---|---|---|
| **allocated 4.0 (this receipt)** | 4.000 / 8.025 | **0.3485** | 68.4% |
| **uniform R1006, byte-matched (this receipt)** | 4.001 / 8.025 | **0.1746** | 77.5% |
| uniform E4M3 window L=14, reach-aware, prior receipt | 4.07 / 8.0 | 0.1512 | 78.1% |
| uniform E4M3 + LDLQ + full-H refit, prior receipt | 4.07 / 8.0 | 0.1046 | — |
| uniform E2M1x2 4.0 (W4A4), plugin | 4.0 / 4.0 | 0.6316 | — |
| production NVFP4 GPTQ+JSO (W4A4) | 4.5 / 4.5 | 0.511 | 62.6% |
| FP8 RTN per-channel (W8A8) | 8.0 / 8.0 | 0.0205 | 91.2% |

The uniform control built here (0.1746 at 4.0005 bpp) lands where the prior
receipt's independently built uniform arm does (0.1512 at 4.07 bpp) — slightly
worse on slightly fewer bytes, which is the expected ordering and a useful
cross-receipt check that this run's exporter and serve path reproduce the
lane's standing numbers.

The allocated arm reads better than production NVFP4 at 4.5 bpp (0.3485 vs
0.511) and than the 4-bit Tessera wire at 4.0 (0.6316) — but only on **wire**
bytes. Both of those serve W4A4 at 4.0–4.5 bpp *resident*, while every arm in
this receipt materialises to per-channel FP8 pairs and occupies **8.025 bpp** of
the card. Priced as it deploys, this is an 8-bit artifact and the comparison is
not a win; the honest reading is the one against its own byte-matched uniform
control, and there it loses. At 4.0 bpp of wire the best arm on this lane is
uniform.

## 9. What this does not establish

* **Not an H-aware result.** Everything here is weights-only, because the
  allocation was priced weights-only. The window body's LDLQ + full-H refit is
  worth 0.1512 → 0.1046 on the uniform E4M3 wire
  (`tessera-ldlq-window-served-2026-09-02.md`), so both arms here sit above
  their achievable floor. The comparison is internally matched; the absolute
  numbers are not the lane's best.
* **Not a statement about allocation as such.** It is a statement about one
  cost model: `h_trace x output_mse`, layer-0-only, 256 activation rows. An
  allocation driven by a KL-adjoint objective, or gated on real KL, is untested
  here and is what the spine prescribes. Nothing in this receipt says a
  well-costed allocation loses to uniform; it says this one does, on the units
  it priced.
* **Not mixed-family.** Every unit in every arm is `TESSERA_E4M3_K1`. The
  claim that a checkpoint can carry two Tessera *families* at once is untested
  by this run; what is tested is that it can carry four *rungs* with a clean
  census.
* **Every allocated rung is outside the attested contract.**
  `src/tessera/serving/runtime_contract.json` publishes
  `candidate_rungs_q256: [1024]` for `TESSERA_E4M3_K1` and `[896]` for
  `TESSERA_E2M1_K2`; the rungs served here are R749/R934/R1006/R1083/R1107 and
  the R750/R1262 controls. The plugin does not gate on that list, so nothing
  refused — which is the principle-14 gap: a producer chose rungs the consumer's
  own machine-readable table does not name, and the run succeeded anyway. The
  censuses in §5 are the evidence a widened `candidate_rungs_q256` could cite;
  widening it is a `src/tessera/serving/` change and was out of scope here.
* **One model, one size, one corpus.** Qwen3-0.6B, 4088 positions of
  `corpus_qwen_n8_s512.json`. No 4B scale check, no second corpus, no downstream
  task.
* **`mink` rates are not exportable.** PrismaQuant's continuous menu can name a
  per-*member* rate inside a fused group; the wire carries one `q256` per
  module, so a plan can only ever be a per-module rung. The allocations here
  happen not to need it (the DP solves over fused super-items), but a future
  allocation that does will have to be refused or rounded, and the converter
  currently has no path for it.
* **The fused-name `ignore` fix is attested on both lanes.** A BF16 passthrough
  is servable only if `ignore` names the module vLLM builds, not the role
  (commit `11b007c`). That the Tessera plugin accepts it is shown by the
  layer-0 arms; that vanilla vLLM's compressed-tensors path accepts it is shown
  by the layer-0 stock twin, served with no plugin at all. Both are claims about
  another runtime, and both were served rather than asserted.

## 10. Reproduce

Host (sparky), venv `/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`,
`PYTHONPATH=<worktree>/src`, `TMPDIR=/home/rob/tmp`:

```bash
# 1. allocation -> plan (+ provenance sidecar)
python experiments/plan_from_layer_config.py \
    /mnt/shared/tessera-runs/pq-continuous/qwen06b/alloc/lc_full_4.0.json \
    /home/rob/models/Qwen3-0.6B /home/rob/tmp/alloc-plans/plan_full_4.0_bcast.json \
    --cover broadcast-by-role --prismaquant /home/rob/pq-wt/tessera-continuous

# 2. plan -> checkpoint (weights-only; + the compressed-tensors twin)
python experiments/export_tessera_serving.py /home/rob/models/Qwen3-0.6B \
    /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0 \
    --plan-json /home/rob/tmp/alloc-plans/plan_full_4.0_bcast.json \
    --stock-twin /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0-stocktwin

# the byte-matched uniform control
python experiments/export_tessera_serving.py /home/rob/models/Qwen3-0.6B \
    /mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1006 --grid E4M3 --q256 1006

# 3. do the exported bytes weigh what the allocator charged?
python experiments/check_wire_against_plan.py \
    /home/rob/tmp/alloc-plans/plan_full_4.0_bcast.json.provenance.json \
    /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0

# 4. tests
python -m pytest tests/test_plan_from_layer_config.py -q
```

Serve box (sparklina), worktree rsynced to `/home/rob/tmp/wt-allocated`,
`TS=/home/rob/tmp/wt-allocated RUNS=/home/rob/tessera-runs/allocated
TESSERA_GPU_MEM_UTIL=0.30 PORT=8004`, one serve at a time under
`experiments/serve_lock.sh`:

```bash
# route census (the container writes into a bind-mounted host dir: /mnt/shared
# is NFS with root_squash and the container runs as root)
experiments/tessera_plugin_run.sh -e TESSERA_SERVE_MODE=resident \
  -v /mnt/shared:/mnt/shared -v "$RUNS":"$RUNS" -- \
  "python3 /work/tools/tessera_route_census.py \
     /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0 \
     $RUNS/census_alloc4-resident.json --expect-modules 112 \
     --gpu-memory-utilization 0.30 --tessera-commit <commit>"

# KL arm (add TESSERA_LANE_EAGER=0 for the compiled forward)
TESSERA_KL_NAME=alloc-serve-alloc4-resident \
TESSERA_KL_CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json \
  experiments/tessera_plugin_served.sh \
    /mnt/shared/tessera-runs/allocated/qwen3-0.6b-alloc-4.0 alloc4-resident resident
```

Drivers kept in `/home/rob/tmp/alloc-plans/`: `export_all.sh` (the three
allocations and their controls), `export_l0.sh` (the layer-0 pair),
`find_uniform.py` (the matched-rung search), `wire_identity.py` (the byte
identity of §3), `verify_uniform.py` (the control's re-measured layer-0
Δloss), `depth_predict.py` and `probe_depth.py` (§7),
`l0_roles.py` (the per-role table of §7), `make_probe_plans.py` +
`export_probe.sh` (the two down_proj mechanism arms and the layer-0 stock
twin), `chain_allocated.sh` / `chain_more.sh` / `chain_probe.sh` (the serve
chains), `kl_table2.py` (the KL table of §5, read off the compare JSONs).
Artifacts and logs: `/mnt/shared/tessera-runs/allocated/` (checkpoints, export
logs, `regret_uniform.json`), `/home/rob/tessera-runs/allocated/` on sparklina
(censuses, arm logs, KL compares), KL dumps
`/mnt/shared/tessera-kl/qwen_tessera_<arm>.json.npz`.
