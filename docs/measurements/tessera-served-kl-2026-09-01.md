# The first served Tessera KL — and it loses to NVFP4's better face

**Measured 2026-09-01.** Qwen3-0.6B, WikiText-2 test, the canonical n=8 × 512
token contract (4088 scored positions), `kl_tool.py` against a BF16 reference
serve. Both arms went through the *same* pipeline: render/decode to a plain
BF16 checkpoint, serve it in `prismaquant/glm53-mia-sm121:487ecf187`, dump
per-position logprobs, compare. Same corpus, same tokenizer, same box, same
serve flags. The quantizer is the only thing that differs.

## Result

| arm | bpp | KL≥ (all) | KL≥ (confident, n=1709) | top-1 agree |
|---|---:|---:|---:|---:|
| **Tessera** `E2M1_K2_R896` | **4.0014** | **0.192008** | 0.136155 | 75.44% |
| **NVFP4 (W4A16)** | **4.5000** | **0.174299** | 0.129343 | 78.01% |

**NVFP4 wins.** It spends 12.5% more bits and returns 9.2% lower KL. Per bit
that is roughly a wash; at face value, at these two operating points, the
uniform format is ahead.

Top-K coverage was 97.2–97.7% mean, so these bounds are tight. (Support is
top-1024, partition teacher∩student, so each number is a lower bound by the
data-processing inequality — but with <3% mean tail mass the slack is small.)

## What this does and does not settle

**It is not a matched-bpp comparison.** 4.00 vs 4.50 bpp. The interesting
question — does Tessera at 4.0 beat NVFP4 at 4.5 *as NVFP4 actually deploys* —
is still open, because:

**The NVFP4 arm is asymmetric in BOTH directions, and the two do not cancel.**

*Generous to NVFP4 on activations.* This arm is **W4A16**. On the GLM route
NVFP4 serves **W4A4**: `flashinfer_b12x` quantizes activations to FP4 too.
Measured on real GLM expert activations, that took NVFP4's functional error
from 0.0667 to 0.1092 — **+64%**. This arm pays none of that.

*Harsh to NVFP4 on weights.* `render_arm_to_bf16` renders through
`spec.quantize_dequantize`, which is **RTN**. PrismaQuant ships NVFP4 with
GPTQ + JSO, worth ~15% on the project's own held-out measurements. So this is
NVFP4's better activation face against its worst weight face — while Tessera
brought its full trellis optimiser to the same comparison.

**The deciding measurement is a real compressed-tensors NVFP4 export, rendered
with the production levers and served natively at W4A4.** It has not been run,
and until it is, neither the sign nor the size of the gap is settled.

**The screen and the gold metric agree in direction, which is reassuring** —
but the two ratios are not the same comparison. The weight-space harness had
Tessera at 0.0979 against NVFP4-W4A16's 0.0667 (**1.47×**), where that NVFP4
was *production-rendered*; the served 1.10× is against **RTN**. Same sign,
different denominators, so the narrowing is not a like-for-like measurement of
anything. What survives is the sign: this is *not* one of the project's
screen-vs-gold inversions.

The earlier headline — "Tessera/NVFP4-as-served = 0.9038" — was against
**W4A4** NVFP4 and is not contradicted by anything here. It is simply a
different comparison, and the arm that would test it is the one still missing.

## What it does establish

**The whole chain works, end to end, for the first time.** Tessera bytes →
`read_unit_artifact` → a plain checkpoint → a real vLLM serve → per-position
logprobs → a KL against BF16. Before today Tessera had never had a number from
a serving runtime. The producer/consumer seam is verified too: the render
PrismaQuant prices is bit-identical to the bytes the exporter writes (6/6
sampled GLM units).

## Method note: why not the GLM 4-layer cut

The first attempt used a 4-layer structural cut of GLM-5.3-Flash. It served,
and it produced a KL — and the KL was meaningless. **Top-1024 captured only
27% of the teacher's probability mass** (mean top-1 probability 0.021, versus
0.5–0.95 for a trained model), so 81% of the mass sat outside the compared
support and top-1 agreement was 23%. A 4-layer cut is not a language model;
its distribution is nearly flat and no top-K comparison on it is informative.
That run is retained as a **plumbing** validation only — it proved a
Tessera-derived GLM checkpoint loads and generates in Mia's image — and none
of its numbers are quality claims.

Qwen3-0.6B was used instead precisely because it is a real trained model. Its
97.7% coverage is what makes the table above worth reading.

## Artifacts

- `/mnt/shared/tessera-exports/qwen3-0.6b-r896` — 197 units, body bpp
  4.001354 (exact 582213/145504), 0.568 GiB total
- `/mnt/shared/tessera-exports/glm4layer-r896` — 2766 units, 23.677 G params,
  body bpp 4.000479 (exact 11562407/2890256), 12.251 GiB
- `/mnt/shared/tessera-kl/` — corpus contracts, both dumps per arm, serve logs


## Superseded in part, 2026-09-01

The bpp axis in the table above is right, but the *reason* Tessera sits at
4.0014 and cannot be moved is not what this document assumed. The rung is not
a rate: every rung of a family serialises to the same bytes, and the whole
serialisable set is two sizes (3.5 and 4.0 bpp). A matched-4.5 Tessera arm is
not merely unbuilt, it is unrepresentable on the current wire. See
`tessera-rate-ceiling-2026-09-01.md`, which also adds the 3.5 bpp point and
reads the two together as a rate-distortion curve.
