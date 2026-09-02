# The first allocated Tessera artifact, served

*2026-09-02. Tessera `PLACEHOLDER_COMMIT` (worktree
`agent-a6d34c0d5bba700a6`); the allocation from PrismaQuant worktree
`/home/rob/pq-wt/tessera-continuous` @ `5a320c7`. Model Qwen3-0.6B, 28 decoder
layers, 196 body Linears, 112 vLLM modules.*

PrismaQuant's continuous Tessera menu produced per-Linear rung assignments at
three byte budgets. This receipt takes them through plan → export → serve → KL
against the BF16 teacher. These are the first Tessera checkpoints served whose
rungs were **chosen** rather than set uniformly.

**The loop closes end to end, and the allocation loses.** At matched bytes the
allocated 4.0-bpp checkpoint reads **KL 0.3485** against the byte-matched
uniform control's **0.1746** — 2.0x worse — while the surrogate that chose it,
and a from-scratch re-measurement of the same units, both said it should be
~1.13x *better*. The bytes are right to the bit, the accounting is right to the
bit, and the routing is 100% on the declared family. §7 locates the failure: the
allocation is a **layer-0 answer**, layer 0's Fisher profile is unrepresentative
of the other 27 depths, and re-weighting the same measured costs by each layer's
own `h_trace` already **inverts** the verdict to 1.205x against the allocation
before anything is served.

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
§7 shows is the assumption that fails.

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
* enforces the fused-family invariant (q/k/v, gate/up) unless
  `--allow-fused-disagreement`, which records each disagreement instead;
* prices every unit through PrismaQuant's **own** accountant
  (`prismaquant.tessera_formats.artifact_bpp`) and writes the charge as an exact
  rational, so the byte check downstream is equality, not a float comparison;
* names **every** body Linear, including the ones the allocation does not
  quantise. That last point was a bug, found by the exporter refusing an export:
  an unnamed tensor is *not* a BF16 passthrough, it takes the exporter's
  `--grid`/`--q256` default (E2M1x2 q256=896), so a seven-unit plan silently
  described a 4-bit checkpoint nobody priced. Both coverage modes now write
  `"BF16"` explicitly and `unplanned_body_linears` is an invariant zero.

`tests/test_plan_from_layer_config.py` — 17 tests, all passing on the host venv
— covers the mapping, the refusals, the fused invariant, both coverage modes,
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

All 196 units agree individually, not merely in total. The layer-0-only export
of §7 agrees the same way over its 7 units (62918656 bits, 4.000260417 bpp).

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

KL_TABLE_PLACEHOLDER

## 6. Allocator-predicted versus served

PLACEHOLDER_SURROGATE

## 7. Where the loss is

PLACEHOLDER_SEPARATOR

## 8. Against the standing references

PLACEHOLDER_REFERENCES

## 9. What this does not establish

PLACEHOLDER_LIMITS

## 10. Reproduce

PLACEHOLDER_REPRO
