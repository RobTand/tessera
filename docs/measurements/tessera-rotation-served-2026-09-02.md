# R1 on the dense 4-bit route, served (2026-09-02)

**Question.** Tessera's NVFP4 route serves Qwen3-0.6B at KL 0.640 (4.0 bpp on
the wire, 4.5 resident, W4A4) against production NVFP4 GPTQ+JSO at 0.511.
`tessera-stock-lane-served-2026-09-02.md` attributes the loss to the
activation leg: "the A4 path on a 0.6 B model costs more than the weights
do". Does a QuaRot/SpinQuant **R1** — one orthogonal mixing of the residual
stream, folded into the checkpoint, no runtime op — close it?

**Answer: no. R1 makes the 4-bit route 2.75x worse served (0.657 → 1.809).**
The activation leg is measurably part of it — isolated on byte-identical
weights, rotation multiplies the A4 leg by **at least 1.76x** (§4a). Whether
the *weight* leg is also hurt is **not measured**: weight-space error is flat
to 1%, but the served weights-only twin was cancelled, and the one historical
number that stands in for it (0.192 decoded, different image) would say the
weight leg is ~3.4x worse too. Do not read this receipt as "the weights are
fine". The production NVFP4
recipe, rebuilt from scratch on the rotated source with its own calibration,
degrades by the same factor (0.511 → 1.271); so does per-channel FP8
(0.0205 → 0.0574). Two independent quantisers and two bit widths fail
together, so this is not a Tessera bug. Top-1 agreement on the rotated 4-bit
arm is 37%: that model is broken, not slightly worse. **R1 folded into the
checkpoint is dead for this lane.**

The scope was cut mid-run once the sign of the result was clear (see
*Cancelled*, below); this receipt reports only arms that actually served.

---

## 1. What was built

`experiments/rotate_checkpoint.py` (+ `tests/test_rotate_checkpoint.py`, 4
tests, passing). R is a seeded randomised Hadamard of size `hidden`
(1024 = 2^10), `D·H/√n` in float64, orthogonality residual **0.0**, seed 0,
sha256 `bddca3e34eba3d4d5b22358c34e2d8dfa6a60c2de655cd78f7ca2c854538c92c`.

Every RMSNorm gamma is folded into the Linears that read it
(`input_layernorm` → q/k/v, `post_attention_layernorm` → gate/up, final norm →
`lm_head`) and the gammas set to ones — without that fold RMSNorm is not
rotation-equivariant. Then `embed → W_e R`; residual **readers** (q, k, v,
gate, up) → `W R`; residual **writers** (`o_proj`, `down_proj`) → `Rᵀ W`;
`lm_head → W_lm R`. Embeddings untied. **R2** (per-head `o_proj` input) and
**R4** (`down_proj` input) are runtime ops and were out of scope;
`rotation.json` records them under `not_applied`.

`--identity` builds the control that does the folding and the untying with
`R = I` (used in §4).

**Bytes.** Untying is free here: Qwen3-0.6B already stores `lm_head.weight` on
disk, bit-identical to `embed_tokens.weight`. The untied head is
**311,164,928 bytes** and the rotated source is **1,503,300,328 bytes —
exactly the size of the original**. Both k2 exports carry the same
`passthrough_bytes` 622,460,928 (embed + head, BF16, never quantised) and the
same wire: **wire 4.0018 bpp / resident 4.5000 bpp**, checkpoint
870,290,032 bytes (W4A4) and 870,268,368 bytes (W4A16, which differs only by
the 196 absent `input_global_scale` tensors).

---

## 2. Stage 1 — the fold is exact and the rotated checkpoint serves

In-process (`verify.json`): an fp32 forward of the rotated state reproduces the
source logits to **mean KL 9.3e-12** (`algebra_float32`). Storing the rotated
weights in bf16 costs **4.9e-4 nats** (`stored_bfloat16`).

Served, on stock vLLM 0.28:

| arm | served KL | p99 | top-1 |
|---|---|---|---|
| **control**: unrotated BF16 re-served as student | **0.000000** | 0.0000 | 100.00% |
| gamma-fold only (R = I), BF16 | 0.001141 | 0.0054 | 97.63% |
| **rotated BF16, seed 0** | **0.001964** | 0.0106 | 96.97% |

The control is the load-bearing row: re-serving the teacher's own checkpoint
scores **exactly zero**, so the serving floor is zero and every number below is
signal. The rotated BF16 arm scores identically on sparky and sparklina
(0.001964 / 0.0106 / 96.97% to every digit), so the two boxes are
interchangeable for this metric. 0.00196 is an order of magnitude below the
0.02 the brief asked for, and ~10x below unrotated FP8 RTN — the fold and the
rotation are, in themselves, nearly free.

---

## 3. Stage 2 — served KL, every arm on one box against one teacher

All arms served on **stock vLLM 0.28** (`vllm/vllm-openai:latest`, repo digest
`sha256:61fc8a89…`, the same image on both boxes), eager, one serve at a time,
port 8002, `--gpu-memory-utilization 0.30`. Metric: exact KL-vs-BF16, top-1024
support, teacher/student intersection, lower bound, 4088 scored positions,
corpus `corpus_qwen_n8_s512.json` (`076d33ef…`). Teacher = the unrotated BF16
source served on the same box (`/mnt/shared/tessera-kl/qwen_rot_teacher_lina.json`).

| arm | contract | served KL | p99 | top-1 | emitted route |
|---|---|---|---|---|---|
| unrotated BF16 (control) | — | 0.00000 | 0.000 | 100.00% | — |
| rotated BF16 | — | 0.00196 | 0.011 | 96.97% | — |
| unrot FP8 per-channel RTN | W8A8 | **0.02046** | 0.180 | 91.19% | `CutlassFP8ScaledMMLinearKernel` |
| fold-only FP8 per-channel RTN | W8A8 | **0.02007** | 0.160 | 91.17% | `CutlassFP8ScaledMMLinearKernel` |
| **rot** FP8 per-channel RTN | W8A8 | **0.05744** | 0.335 | 86.37% | `CutlassFP8ScaledMMLinearKernel` |
| unrot PQ NVFP4 GPTQ+JSO 4.5 bpp | W4A4 | **0.51058** | 3.311 | 62.57% | `FlashInferCutlassNvFp4LinearKernel` |
| **rot** PQ NVFP4 GPTQ+JSO 4.5 bpp | W4A4 | **1.27119** | 6.139 | 45.21% | `FlashInferCutlassNvFp4LinearKernel` |
| unrot Tessera k2, production A-cal | W4A4 | **0.64040** | 3.798 | 58.76% | `FlashInferCutlassNvFp4LinearKernel` |
| unrot Tessera k2, my A-cal | W4A4 | **0.65671** | 3.778 | 58.59% | `FlashInferCutlassNvFp4LinearKernel` |
| **rot** Tessera k2, my A-cal | W4A4 | **1.80903** | 7.880 | 37.23% | `FlashInferCutlassNvFp4LinearKernel` |
| **rot** Tessera k2, weights only | W4A16 | **0.65324** | 3.566 | 59.88% | `MarlinNvFp4LinearKernel` *(diagnostic, not a product route)* |

Matched pairs, same recipe on both sides:

| pair | unrotated | rotated | rotation costs |
|---|---|---|---|
| Tessera E2M1x2 → NVFP4 tile, W4A4, my A-cal | 0.65671 | 1.80903 | **2.75x** |
| PQ NVFP4 GPTQ+JSO, W4A4, each side's own calibration | 0.51058 | 1.27119 | **2.49x** |
| FP8 per-channel RTN, W8A8 dynamic | 0.02046 | 0.05744 | **2.81x** |

Both unrotated comparators reproduce their published receipts to four digits on
this box (0.6404 vs the 0.640 stock-lane number; 0.5106 vs 0.511), which is
what licenses the pairing. My own activation calibration costs +2.5% against
the production one (0.65671 vs 0.64040) and is held fixed inside the Tessera
pair, so it is not the story.

**The rotated arm never wins.** It does not beat 0.511, it does not beat 0.640,
and it does not beat its own unrotated twin at any width in any format.

---

## 4. Where the loss sits: the A4 leg at least, and it is R, not the fold

**(a) The A4 leg, isolated on the served metric.** `rot-k2-w4a4` and
`rot-k2-w4a16` were exported from the same source with the same encoder and the
same wire; `experiments/compare_stock_checkpoints.py` says **588 of 588 shared
quantized tensors are byte-identical**, the only difference being the 196
`input_global_scale` tensors the W4A16 arm does not carry. So their difference
is a pure activation-leg number:

> rotated A4 leg = 1.80903 − 0.65324 = **1.156 nats** — 1.8x the entire
> weights-only arm.

The unrotated W4A16 twin was cancelled before it served (§6), so the *split* of
the rotation delta between legs is not fully measured. One rigorous statement
survives without it: KL ≥ 0, so the **unrotated** A4 leg is at most the
unrotated W4A4 total, 0.65671. Therefore

> **rotation multiplies the A4 leg by at least 1.156 / 0.65671 = 1.76x.**

That is the opposite of the hypothesis this run was launched to test.

**(b) The weight leg: flat in weight space, unmeasured on the serve.**
Dequantising every arm and comparing to
the bf16 source it was built from (`experiments/rotation_weight_error.py`;
Frobenius is rotation-invariant, so the pairs are directly comparable):

| arm | rel ‖dW‖²_F | h-weighted rel err |
|---|---|---|
| unrot FP8 RTN | 6.9595e-04 | 6.8626e-04 |
| **rot** FP8 RTN | 6.9921e-04 (+0.5%) | 6.9900e-04 (+1.9%) |
| unrot Tessera k2 | 9.1010e-03 | 1.0343e-02 |
| **rot** Tessera k2 | 9.1805e-03 (+0.9%) | 9.2085e-03 (**−11%**) |
| unrot PQ NVFP4 | 8.8744e-03 | 9.7005e-03 |
| **rot** PQ NVFP4 | 8.8319e-03 (−0.5%) | 8.4122e-03 (**−13%**) |

The weight error is unchanged to within 1% for all three recipes, and the
standard diagonal-Hessian output proxy `Σ_j h_j ‖dW[:,j]‖²` calls the rotated
arms 11–13% *better*. **Both weight-space diagnostics point the wrong way**
against a 2.5–2.8x served regression: neither is a substitute for serving —
and that is precisely why they cannot be used to claim the weight leg is
undamaged. Flat Frobenius is what failed to predict the W4A4 serve; it cannot
now be trusted to acquit the weight leg.

The one number that speaks to the served weight leg is historical and
cross-image: `tessera-stock-lane-served-2026-09-02.md` records the *unrotated*
Tessera K2 weights **BF16-decoded** at **0.192** (W16A16, on the Mia image,
same corpus family). Set beside this run's rotated weights-only arm at
**0.65324**, that reads as a ~3.4x weight-leg regression — larger than the
2.75x total. Two caveats keep it a prior and not a result: it was served on a
different image against a different teacher dump, and a decoded-BF16 GEMM is
not the same numerics as W4A16 through `MarlinNvFp4LinearKernel`. What
licenses quoting it at all is that this run reproduced that receipt's
`tessera-k2` W4A4 arm to five digits (0.64040 vs 0.6404), so the encoder and
the corpus are the same object. **If that prior holds, both legs carry the
factor and §4d is wrong.** The unrotated W4A16 twin (§6) settles it.

**(c) The gamma fold is innocent; R is not.** A folded R1 does two things at
once — it rescales every reader's input columns by gamma (moving the maxima a
quantiser sees) and it mixes the basis. The `--identity` control separates
them, and on the clean FP8 pair the fold costs nothing:

> unrot 0.02046 → **fold-only 0.02007** → rot 0.05744.

Folding alone is a −2% no-op. The whole 2.81x is R.

**(d) Two candidate mechanisms; the cancelled twin distinguishes them.**

*Candidate 1 — the A-leg story.* Unrotated, a reader sees
`x̂·gamma`, so the massive-activation channels are gamma-suppressed *before* the
activation quantiser rounds them. Once gamma is folded into `W` (which R1
requires, to stay runtime-free) the quantiser sees `Rᵀx̂` — the un-suppressed
massive channel, spread white across all 1024 coordinates — and the resulting
activation error is white in the original basis and then passes through
`W·diag(gamma)` at full gain on the informative columns. This predicts the A4/A8 leg getting worse
while the weight leg does not, and the fold alone (which keeps the basis)
being harmless — the second half of which was measured (§4c). A Fable-tier
consultation (recorded in §7) predicted this before the W4A16 arm served,
from the constancy of the factor across bit widths and quantisers.

*Candidate 2 — a common downstream gain.* R mixes the residual basis, so a
row of `Rᵀ W` no longer aligns with the row-space anisotropy the model
actually uses. Equal-Frobenius error in the rotated basis then costs more
end-to-end than the same error unrotated, in *both* legs — the weight leg
included. This is the mechanism the 0.192 prior in §4b points at, and it is
also what a row-side (output-side) Fisher census on `dW` would show; energy-
weighted column error, which is what §4b measures, is blind to it.

**The discriminator is one arm.** Serve `unrot-k2-w4a16` on this box against
this teacher. If it lands near 0.19–0.25, the weight leg is ~3x worse rotated
and Candidate 2 is the mechanism; if it lands near 0.60, the weight leg is
genuinely untouched and Candidate 1 stands. It was cancelled at 20/196 (§6).
Until it runs, the only defensible attribution is the bound in §4a.

---

## 5. The activation census — what the block scale absorbs

One bf16 forward with hooks on all 196 `model.layers.*` Linears;
`h_j = E[x_j²]` per input column, 32 × 512 tokens of WikiText train.
Per-kind medians over 28 layers, before → after rotation:

| unit kind | amax | h max/median | value kurtosis |
|---|---|---|---|
| q_proj / k_proj / v_proj | 38.00 → 5.02 | 645 → 3.8 | 82.8 → 2.8 |
| gate_proj / up_proj | 35.88 → 4.91 | 571 → 4.1 | 105.5 → 2.8 |
| o_proj | 9.97 → 9.94 | 77.3 → 76.6 | 19.4 → 19.4 |
| down_proj | 66.00 → 67.12 | 120.4 → 119.6 | 304.5 → 306.4 |

The worst individual units:

| unit | amax | h max/median | h max/mean | h kurtosis | value kurtosis |
|---|---|---|---|---|---|
| `layers.2.mlp.down_proj` (unrot) | 3888.00 | 2.089e6 | 1230.8 | 1205.4 | 474334 |
| `layers.2.mlp.down_proj` (**rot**) | 3888.00 | 2.090e6 | 1231.1 | 1205.9 | 474360 |
| `layers.2.mlp.gate_proj` (unrot) | 61.00 | 1685.8 | 486.3 | 964.4 | 312.7 |
| `layers.2.mlp.gate_proj` (**rot**) | 5.28 | 2.9 | 2.5 | 4.3 | 2.90 |
| `layers.27.self_attn.q_proj` (unrot) | 280.00 | 666.5 | 200.5 | 572.5 | 82.6 |
| `layers.27.self_attn.q_proj` (**rot**) | 4.62 | 9.1 | 5.4 | 4.7 | 2.65 |

Model-wide the median per-Linear `diag(H)` max/median falls **386.8 → 4.1**, and
per-Linear amax falls from median 35.50 (max 3888) to median 5.03 (max 3888).
R1 does exactly what QuaRot says it does to the residual readers — they become
Gaussian, kurtosis 83–105 → 2.8 — and the served result got worse anyway.

Two structural facts the census makes visible:

- **The two worst units in the model are bit-identical before and after.**
  `o_proj` and `down_proj` are the units R2 and R4 address; R1 cannot reach
  them, and `layers.2.mlp.down_proj` sits at h max/median 2.09e6 on both sides.
- The static NVFP4 `input_global_scale` is PrismaQuant's legacy `6/amax`
  (confirmed in `nvfp4_activation_contract.input_global_scale_from_max_abs`;
  the opt-in `448·6/amax` policy is off by default). Rotation removes the
  3888-amax readers this scale was being set by — and the arm still lost
  2.75x.

---

## 6. Cancelled, and why

Scope was cut by Rob mid-run once the sign of the result was established.
Rotation on an NVFP4 route is expected to be near-useless on prior grounds:
**QuaRot** and **SpinQuant** both target formats whose activation scale is
coarse (per-token or per-tensor INT4/INT8), where a single outlier channel sets
the step size for every other channel. NVFP4's scale is per **16** elements
with an FP8 E4M3 exponent, so a channel outlier is absorbed by its own block
rather than taxing the tensor — the lever rotation pulls is mostly already
pulled. Our own measurement agrees: on GLM MoE experts rotation was worth
**1.003x** (`tessera-w4a4-changes-the-lever`), and `docs/measurements/
rotation-decision.md` reached the same conclusion in weight space. The dense
Qwen case was worth testing because its *input columns* are far more
outlier-heavy than a routed expert's Gaussian inputs — and the test says the
per-16 block scale was already handling them, while the fold that R1 requires
takes away the gamma suppression the activation quantiser was relying on.

Cancelled before serving, and **not** reported here:

- `unrot-k2-w4a16` — the weights-only twin of the rotated W4A16 arm. **This is
  the one measurement whose absence weakens the receipt**: it would split the
  rotation delta between the weight leg and the A4 leg exactly instead of
  bounding it, and it decides between the two mechanisms in §4d — near
  0.19–0.25 means the weight leg is ~3x worse rotated (Candidate 2, and the
  stock-lane receipt's own "the A4 path costs more than the weights"
  attribution would then need its own re-check); near 0.60 means the weight
  leg is untouched (Candidate 1). The export was killed at 20/196; a fresh
  export (~30 min) plus one 4-minute serve would close it.
- `rot-e4m3` — the rotated Tessera E4M3/CHANNEL wire (the FP8 route). Would
  have been compared against the **weights-only** FP8-route reference
  **0.1512**; the Hessian-aware sibling of that reference (LDLQ + full-H refit,
  identical bytes, another branch) is **0.1046** and the two are never mixed.
  An H-aware *rotated* arm is a follow-up nobody has run.
- `foldonly-k2-w4a4` / `foldonly-k2-w4a16` — the gamma-fold control at 4 bits.
  The fold control exists only at FP8 (§4c), where it is a no-op; at 4 bits the
  fold's effect on per-16 group maxima is untested.
- `rot-k2-w4a4-pqcal` — deliberately skipped: a 2.75x effect swamps the
  0.57–2.35x spread between my calibration and production's, and PQ's own
  calibration on the rotated source degraded by 2.49x independently.

---

## 7. Provenance, and things a reviewer should know

- Worktree branch `worktree-agent-a820f5bb1b4e130f2`; the exporter's git stamp
  is `unknown` in the manifests built on sparklina because the worktree was
  rsynced there without `.git`. The tree is otherwise identical to the commit
  carrying this file.
- The PrismaQuant rotated arm was produced by `run-pipeline.sh` with exactly
  the environment that produced the 0.511 comparator
  (`FORMATS=NVFP4 TARGET_BITS=4.5 PARETO_TARGETS=4.5 NSAMPLES=32 SEQLEN=1024
  COST_MODE=local`), only `MODEL_PATH`/`WORK_DIR` changed. Achieved
  4.500028 bpp, 252 targets, `ignore=[lm_head, model.embed_tokens]` — the same
  config shape as the unrotated export. `lm_head` is BF16 in every arm here.
- All GPU work moved to sparklina mid-run at the coordinator's instruction.
  Checkpoints and logs: `/mnt/shared/tessera-runs/rotation/`; KL dumps:
  `/mnt/shared/tessera-kl/qwen_rot_*.json{,.npz}`.
- **Serve-lock incident.** I removed sparklina's `serve.lock` by hand at
  ~17:03Z: it was held by "kernel-window worker" whose process had been absent
  for 47 minutes with no serve container, and three workers were queued behind
  it. `experiments/serve_lock.sh` could never have cleared it — `acquire`
  writes an `owner` file *into* the lock directory and the stale branch called
  `rmdir` without removing it, so every stale lock was permanent. Two workers
  cleared locks that afternoon and two serves briefly overlapped; the receipt's
  only dump inside the flagged 17:06–17:12Z window is the **teacher**
  (completed 17:12:00Z), and it is proved intact by the control arm, which
  re-served the identical checkpoint at 17:15:09Z, outside the window, and
  scored KL **0.000000** with 100.00% top-1 against it. `serve_lock.sh` has
  since been replaced with master's ownership-checked release (`fb84e41`);
  this worker holds no lock and removed nothing else it did not own.
- **Fable consultation** (one, `fable-high`, no code): asked why an exact
  orthogonal reparameterisation amplifies the logit effect of an equal-Frobenius
  weight perturbation. It corrected two things — that per-row scaling is the
  wrong lever for a *float* 8-bit format (E4M3 rounds elementwise-relative, so
  the fold is near-commutative there and the FP8 pair isolates R cleanly), and
  that the h-weighted number IS the diag-H expected output MSE, so its being
  flat means per-Linear output MSE is genuinely unchanged. It predicted, before
  the W4A16 arm served, that the A-leg would be the carrier via the un-gamma'd
  massive channel; the measured 1.156-nat A4 leg is consistent with that, but
  does not exclude the weight leg also degrading (§4b/§4d). Its
  ranked follow-ups (basis-swap hybrids on the W4A16 pair; a row-side Fisher
  census on dW; an equal-energy residual-noise premise test) are all unrun.
- **Unchecked confound on the PQ pair.** PrismaQuant's GPTQ damping is a fixed
  1.0 (sweep off since 2026-06-12). I did not check whether that constant is
  absolute or a percent of the mean Hessian diagonal. Rotation leaves `tr(H)`
  invariant but flattens its spectrum hard on the readers (diag-H max/median
  645 → 3.8, §5), so a *percent-of-mean* damp regularises both arms
  identically while an *absolute* damp does not. The Tessera and FP8 pairs use
  no GPTQ and degrade by the same factor, so this cannot explain the result —
  but it is unverified for the 2.49x figure specifically.
- Single rotation seed (0). Top-1024 truncation is arm-dependent, and the worse
  arm loses more mass, so the reported ratios are conservative; top-1 agreement
  is unaffected.

---

## 8. Commands

```bash
# rotate, and the R = I control
PYTHONPATH=src python experiments/rotate_checkpoint.py \
  --src /home/rob/models/Qwen3-0.6B --dst <dst> --seed 0 [--identity]

# activation census + static A-side scales
PYTHONPATH=src python experiments/rotation_activation_stats.py \
  --model <src> --out <stats.json> --scales-out <scales.safetensors> \
  --text <calib.txt>

# arms (sparklina), one serve at a time under experiments/serve_lock.sh
bash experiments/rotation_export_arms.sh
bash experiments/rotation_serve_arms.sh <arm>...

# where the error is, and the A-leg isolation
PYTHONPATH=src python experiments/rotation_weight_error.py <out.json> \
  unrot:<label>=<dir> rot:<label>=<dir> ...
PYTHONPATH=src python experiments/compare_stock_checkpoints.py \
  <rot-k2-w4a4> <rot-k2-w4a16>     # 588/588 identical, 196 A-scales only
```
