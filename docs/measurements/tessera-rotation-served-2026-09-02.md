# R1 on the dense 4-bit route, served (2026-09-02)

**Question.** Tessera's NVFP4 route serves Qwen3-0.6B at KL 0.640 (4.0 bpp on
the wire, 4.5 resident, W4A4) against production NVFP4 GPTQ+JSO at 0.511.
`tessera-stock-lane-served-2026-09-02.md` attributes the loss to the
activation leg: "the A4 path on a 0.6 B model costs more than the weights
do". Does a QuaRot/SpinQuant **R1** — one orthogonal mixing of the residual
stream, folded into the checkpoint, no runtime op — close it?

**Answer: no, and not by a small margin. R1 makes every served arm about
2.5–2.8x worse, and it does so without changing the weight error at all.**
The 4-bit route goes 0.657 → 1.809. The production NVFP4 recipe, rebuilt from
scratch on the rotated source with its own calibration, goes 0.511 → 1.271.
Per-channel FP8 RTN, which was never in question, goes 0.0205 → 0.0574. Top-1
agreement on the rotated 4-bit arm is 37%: that model is broken, not slightly
worse. **R1-folded-into-the-checkpoint is dead for this lane.** It is not a
Tessera failure — two independent quantisers fail together by the same factor.

## What was built

`experiments/rotate_checkpoint.py` (+ `tests/test_rotate_checkpoint.py`, 4
tests). R is a seeded randomised Hadamard of size `hidden` (1024 = 2^10),
`signs * H / sqrt(n)` in float64, orthogonality residual 0.0, seed 0,
sha256 `bddca3e3…`. Every RMSNorm gamma is folded into the Linears that read it
(`input_layernorm` → q/k/v, `post_attention_layernorm` → gate/up, final norm →
`lm_head`) and the gammas set to ones — without that fold RMSNorm is not
rotation-equivariant. Then `embed → W_e R`, residual readers (q, k, v, gate,
up) → `W R`, residual writers (`o_proj`, `down_proj`) → `R^T W`, `lm_head →
W_lm R`. **R2** (per-head `o_proj` input) and **R4** (`down_proj` input) are
runtime ops and were out of scope; `rotation.json` records them under
`not_applied`.

The transform is exact: an fp32 forward of the rotated state reproduces the
source logits to **mean KL 9.3e-12** (`verify.json:algebra_float32`). Storing
the rotated weights in bf16 costs **4.9e-4 nats** in-process
(`verify.json:stored_bfloat16`) — that is the only intrinsic cost of the
rotation, and it is 3 orders below every quantised number below.

Untying the embeddings is free in bytes: Qwen3-0.6B already stores
`lm_head.weight` on disk (bit-identical to `embed_tokens.weight`), so the
rotated source is **1,503,300,328 bytes, exactly the size of the original**,
and both k2 exports weigh **870,290,032 bytes** — same wire, same bpp, same
`passthrough_bytes` 622,460,928.

## Stage 1 — the rotated BF16 checkpoint serves correctly

| arm | served KL | p99 | top-1 |
|---|---|---|---|
| control: unrotated BF16 re-served as student | **0.000000** | 0.0000 | 100.00% |
| rotated BF16, seed 0 | **0.001964** | 0.0106 | 96.97% |

The control is the important row: re-serving the teacher's own checkpoint
scores exactly zero, so **the serving floor is zero** and every number in this
receipt is signal, not noise. The rotated BF16 arm scores identically on sparky
and on sparklina (0.001964 / 0.0106 / 96.97% to every digit), so the two boxes
are interchangeable for this metric and the historical sparky receipts are
directly comparable. 0.00196 is two orders below the 0.02 gate the brief asked
for, and ~10x below unrotated FP8 RTN.

## Stage 2 — served KL, every arm on one box against one teacher

All arms served on **stock vLLM 0.28** (`vllm/vllm-openai:latest`, repo digest
`sha256:61fc8a89…` — the same image on both boxes), eager, one serve at a time,
port 8002, `--gpu-memory-utilization 0.30`. Metric: exact KL-vs-BF16,
top-1024 support, teacher/student intersection, lower bound, 4088 scored
positions, corpus `corpus_qwen_n8_s512.json` (`076d33ef…`). Teacher =
unrotated BF16 served on the same box (`qwen_rot_teacher_lina.json`).

| arm | bits | served KL | p99 | top-1 | route |
|---|---|---|---|---|---|
| unrotated BF16 (control) | 16 | 0.0000 | 0.000 | 100.00% | — |
| rotated BF16 | 16 | 0.0020 | 0.011 | 96.97% | — |
| unrot FP8 RTN per-channel | W8A8 | **0.0205** | 0.180 | 91.19% | `CutlassFP8ScaledMMLinearKernel` |
| **rot** FP8 RTN per-channel | W8A8 | **0.0574** | 0.335 | 86.37% | `CutlassFP8ScaledMMLinearKernel` |
| unrot PQ NVFP4 GPTQ+JSO 4.5 bpp | W4A4 | **0.5106** | 3.311 | 62.57% | `FlashInferCutlassNvFp4LinearKernel` |
| **rot** PQ NVFP4 GPTQ+JSO 4.5 bpp | W4A4 | **1.2712** | 6.139 | 45.21% | `FlashInferCutlassNvFp4LinearKernel` |
| unrot Tessera k2, production A-cal | W4A4 | **0.6404** | 3.798 | 58.76% | `FlashInferCutlassNvFp4LinearKernel` |
| unrot Tessera k2, my A-cal | W4A4 | **0.6567** | 3.778 | 58.59% | `FlashInferCutlassNvFp4LinearKernel` |
| **rot** Tessera k2, my A-cal | W4A4 | **1.8090** | 7.880 | 37.23% | `FlashInferCutlassNvFp4LinearKernel` |

Matched pairs, same recipe on both sides:

| pair | unrotated | rotated | rotation costs |
|---|---|---|---|
| Tessera E2M1x2 → NVFP4 tile, W4A4, my A-cal | 0.6567 | 1.8090 | **2.75x** |
| PQ NVFP4 GPTQ+JSO, W4A4, each side's own calibration | 0.5106 | 1.2712 | **2.49x** |
| FP8 per-channel RTN, W8A8 dynamic | 0.0205 | 0.0574 | **2.80x** |

Both unrotated comparators reproduce their published receipts to four digits on
this box (0.6404 vs the 0.640 stock-lane number; 0.5106 vs 0.511), which is
what licenses the pairing. My own activation calibration costs +2.5% against
the production one (0.6567 vs 0.6404) — a real but small confound, and it is
held fixed inside the Tessera pair.

**The rotated arm never wins.** It does not beat 0.511, it does not beat 0.640,
and it does not beat its own unrotated twin in any format at any width.

## Where the loss is *not*

Dequantising every arm and comparing to the bf16 source it was built from
(`experiments/rotation_weight_error.py`; Frobenius is rotation-invariant, so
the pairs are directly comparable):

| arm | rel ‖dW‖²_F | h-weighted rel err |
|---|---|---|
| unrot FP8 RTN | 6.9595e-04 | 6.8626e-04 |
| **rot** FP8 RTN | 6.9921e-04 (+0.5%) | 6.9900e-04 (+1.9%) |
| unrot Tessera k2 | 9.1010e-03 | 1.0343e-02 |
| **rot** Tessera k2 | 9.1805e-03 (+0.9%) | 9.2085e-03 (**−11%**) |
| unrot PQ NVFP4 | 8.8744e-03 | 9.7005e-03 |
| **rot** PQ NVFP4 | 8.8319e-03 (−0.5%) | 8.4122e-03 (**−13%**) |

So **the weight leg is unchanged** (≤1% in either direction, for all three
recipes), and the standard diagonal-Hessian output proxy
`Σ_j h_j ‖dW[:,j]‖²` says the rotated arms are 11–13% *better*. Both
diagnostics point the wrong way relative to a 2.5–2.8x served regression.

The activation side moved in the direction the hypothesis predicted, and it did
not help. Census from a bf16 forward, `h_j = E[x_j²]` per input column, before
→ after rotation:

| unit kind | amax | h max/median | value kurtosis |
|---|---|---|---|
| q/k/v_proj | 38.0 → 5.0 | 645 → 3.8 | 82.8 → 2.8 |
| gate/up_proj | 35.9 → 4.9 | 571 → 4.1 | 105.5 → 2.8 |
| o_proj | 9.97 → 9.94 | 77 → 77 | 19.4 → 19.4 |
| down_proj | 66.0 → 67.1 | 120 → 120 | 304 → 306 |

(per-kind medians over 28 layers; `o_proj`/`down_proj` are the units R2/R4
would have touched and are unchanged by construction). Model-wide, the median
per-Linear `diag(H)` max/median falls **386.8 → 4.1**: the residual readers
become Gaussian, exactly as QuaRot predicts. The static NVFP4
`input_global_scale = 6/amax` therefore stops being set by a 3888-amax outlier
on the readers — and the arm still lost 2.75x. Note the two worst units in the
model, `layers.*.mlp.down_proj` (`layers.2` has h max/median 2.09e6), are
**bit-identical before and after**: R1 structurally cannot touch them, that is
R4's job, and R4 is a runtime op.

## The mechanism (hypothesis, not a measurement)

The same ~2.5–2.8x appears at 8 bits and at 4 bits, with and without a 4-bit
activation leg, for RTN, for GPTQ+JSO and for a trellis encoder, while the
weight error itself is unchanged. A constant multiplicative factor that is
blind to bit width, to format and to quantiser is not a property of the
encoder; it is a property of the *network* the encoder writes into. The
rotated network converts a weight perturbation of a given size into a ~2.8x
larger logit perturbation.

The candidate mechanism: per-output-channel (and per-16-group) scaling buys
*relative* precision — each channel's absolute error is proportional to that
channel's own magnitude. Qwen's residual stream is dominated by a few
massive-activation channels that carry most of the energy and little
information, so unrotated, the small-magnitude channels receive small absolute
error. Rotating the writers (`R^T W`) mixes rows, so the error becomes uniform
in the rotated basis, which maps back to error on the informative small
channels comparable to that on the uninformative large ones. Both diagnostics
above are blind to this: Frobenius sees no basis at all, and `h_j = E[x_j²]`
weights by *energy*, which is the massive-activation-dominated measure rather
than an importance measure.

This is a hypothesis with two supporting facts (the constancy of the factor,
and the failure of both proxies) and no direct measurement. The discriminating
experiment would weight the weight error by a curvature/importance measure
rather than by energy — the AURA KL-adjoint objective, or a full-H rather than
diag-H weighting — on both sides of one matched pair. Not run.

## What this does not say

- It does not say rotations are dead for this lane in general. **R2 and R4 were
  not applied**, and the two units with the worst input geometry in the whole
  model (`down_proj`, h max/median up to 2.09e6) are precisely the ones R4
  addresses and R1 provably cannot. A negative result for R1-alone folded into
  the checkpoint is not a negative result for QuaRot with its runtime ops.
- It does not say the A4 leg is fine. It says R1 is not the way to fix it: R1
  removes the reader-side outliers (645 → 3.8) and the arm gets worse anyway,
  so whatever the A4 leg costs, it is not being paid on the reader inputs R1
  can reach.
- The FP8-route reference numbers this receipt sits beside are the
  **weights-only** encode at **0.1512** served; a Hessian-aware sibling (LDLQ +
  full-H refit, identical bytes) reaches **0.1046** on another branch. Every
  arm here is weights-only, so 0.1512 is the comparable figure; the two are
  never mixed. An H-aware *rotated* arm is a follow-up nobody has run — and
  given that a diag-H proxy already mis-ranks these arms by 11–13%, it is the
  obvious next question rather than a formality.

## Provenance and honesty notes

- Worktree HEAD at the time of the exports: see the commit that carries this
  file. The exporter's own git stamp is absent from the manifests built on
  sparklina ("fatal: not a git repository") because the worktree was rsynced
  there without `.git`; the tree is otherwise byte-identical to this commit.
- The PrismaQuant rotated arm was produced by `run-pipeline.sh` with exactly
  the environment that produced the 0.511 comparator
  (`FORMATS=NVFP4 TARGET_BITS=4.5 PARETO_TARGETS=4.5 NSAMPLES=32 SEQLEN=1024
  COST_MODE=local`), only `MODEL_PATH`/`WORK_DIR` changed; achieved
  4.500028 bpp, 252 targets, `ignore=[lm_head, model.embed_tokens]` — the same
  config shape as the unrotated export.
- All GPU work moved to sparklina mid-run at the coordinator's instruction;
  checkpoints, logs and KL dumps live under `/mnt/shared/tessera-runs/rotation/`
  and `/mnt/shared/tessera-kl/qwen_rot_*.json`.
- **A stale serve lock was cleared by hand on sparklina.** `serve.lock` was held
  by "kernel-window worker" whose process had been absent for 47 minutes with no
  serve container running, and three workers were queued behind it.
  `experiments/serve_lock.sh` could never have cleared it: `serve_lock_acquire`
  writes an `owner` file *into* the lock directory and the stale branch called
  `rmdir` without removing it, so every stale lock was permanent. Fixed here.
  A second limitation is flagged and **not** changed: the stale branch also
  requires `docker ps -q` to be empty box-wide, so an unrelated container (a
  pytest run) still blocks stale-clearing.

## Commands

```bash
# rotate (sparky, before the move)
PYTHONPATH=src python experiments/rotate_checkpoint.py \
  /home/rob/models/Qwen3-0.6B /home/rob/tessera-runs/rotation/qwen3-0.6b-rot-seed0 --seed 0

# activation census, both sides
PYTHONPATH=src python experiments/rotation_activation_stats.py <src> <stats.json> <scales.safetensors>

# arms (sparklina)
bash /mnt/shared/tessera-runs/rotation/export_arms_lina.sh
bash /mnt/shared/tessera-runs/rotation/serve_arms_lina.sh <arm>...

# where the error is
PYTHONPATH=src python experiments/rotation_weight_error.py \
  /mnt/shared/tessera-runs/rotation/weight_error.json \
  unrot:<label>=<dir> rot:<label>=<dir> ...
```
