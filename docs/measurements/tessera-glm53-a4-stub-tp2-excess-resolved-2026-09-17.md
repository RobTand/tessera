# The A4 stub's TP2 excess is re-drawn quantization noise, not added error (2026-09-17)

**Status: closes tessera#514.** The world-size receipt of 2026-09-15
(`docs/measurements/tessera-glm53-a4-stub-tp2-world-size-2026-09-15.md`, contract v29) measured
the A4 stub diverging from its own single-rank serve 4.26 times more than the BF16 stub at the
median |d logprob|, with about the same p99, and left the cause open: Tessera's sharded quantized
path adding error, or E2M1 quantization amplifying a perturbation stock vLLM's TP2 already
introduces. This page settles it from the receipt's own four payloads, without a new serve, and
then measures the mechanism on the pinned image's own quantizers. The receipt's grade stays
`route_only`; nothing in the contract moves.

The findings:

1. **Against the BF16 reference the A4 stub is at the same distance on one rank and on two.**
   Renormalized shared-support KL(BF16 TP1 || A4) is 0.07471 at TP1 and 0.07484 at TP2, ratio
   1.0017 with a 95% bootstrap interval of [0.990, 1.014] over positions; TP2 is worse at 50.0%
   of positions. The top-1024 lower bound (`kl_tool compare`) is 0.017920 on both arms.
2. **The A4 TP2-vs-TP1 delta is a re-draw of the quantization noise, not error on top of it.**
   Of the noise variance between the A4 and BF16 stubs, TP2 re-draws 18.9% (CI [18.3%, 19.4%])
   while the total stays put (variance ratio 1.0028, CI [0.994, 1.011]). The share of the A4
   delta that is added error is 0.7% (CI [-1.6%, +3.0%]), so any error the sharded path adds is
   at most about 1% of the quantization-noise variance at 95%.
3. **The mechanism is the runtime's quantizers responding to a BF16-ulp perturbation.** On the
   pinned image's `scaled_fp4_quant` with the checkpoint's static globals, the exact TP2
   partial-sum rounding flips about 1% of E2M1 codes, leaves the quantizer's error variance
   unchanged to 0.1%, and grows the perturbation's own power 72-87x. The per-token E4M3
   quantizer flips 2.5% of codes at 26x. BF16, with no quantizer, is 1x by construction. That
   is why the TP-vs-TP divergence of a quantized checkpoint is larger than a BF16 control's at
   the median while its distance to the reference does not move.

**TP2 is safe for the A4 path on this evidence:** it costs no quality against the reference on
the stub, and the mechanism is variance-preserving by construction, so a deeper body re-draws a
larger share of its noise without raising the total in expectation. The number that gates a served
world is the KL against the reference at that world, which `route_only` already defers to served
validation of the real export. Out of scope here: the RoCE `ibv_reg_mr` instability
(availability, not numerics; the sockets fabric is bit-identical), graph mode (tessera#508), MTP.

## What the receipt measured, and why it could not tell the two readings apart

The receipt compares each checkpoint with itself at another world size. On a quantized checkpoint
that comparison cannot separate two different things. A small perturbation entering a quantizer
flips the codes of the elements that sit near a decision boundary; every flip re-draws that
element's quantization error, and a re-drawn error has the same distribution as the one it
replaces. So a TP2 arm can differ from the TP1 arm by a great deal at the median while being no
further from the reference. The receipt's `excess_over_control` is a divergence between two worlds
of one checkpoint. It is not a statement about error, and it should not be read as one.

The same four `kl_tool` payloads the receipt names can answer the error question directly,
because the BF16 stub is the A4 stub's reference: same corpus contract, same tokenizer, same
positions.

## The reference reading

`experiments/tp2_reference_kl_514.py` reads the four payloads, refuses them unless their sha256
matches the committed receipt table (`experiments/results/glm53_a4_stub_tp_single_rank_kl_514.json`),
and writes `experiments/results/glm53_a4_stub_tp2_reference_kl_514.json`. Prefill 8 x 512,
4088 scored positions, top-1024 per position, eager, image
`localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378...`, tree `44d20d670`.

| Reference, arm | Shared pairs | \|d logprob\| p50 / p99 / max | Renormalized KL mean / p99 / max | Top-1024 KL lower bound mean | Top-1 agree |
|---|---|---|---|---|---|
| BF16 TP1, A4 TP1 | 3,242,429 | 0.1953 / 1.013 / 3.41 | 0.07471 / 0.2645 / 0.503 | 0.017920 | 69.55% |
| BF16 TP1, A4 TP2 | 3,241,119 | 0.1958 / 1.015 / 3.30 | 0.07484 / 0.2604 / 0.497 | 0.017920 | 68.81% |
| BF16 TP2, A4 TP1 | 3,238,844 | 0.1961 / 1.018 / 3.68 | 0.07520 / 0.2696 / 0.536 | - | 69.57% |
| BF16 TP2, A4 TP2 | 3,238,930 | 0.1962 / 1.016 / 2.97 | 0.07496 / 0.2631 / 0.450 | - | 68.88% |
| A4 TP1, A4 TP2 (the receipt) | 3,660,560 | 0.1132 / 0.667 / 3.04 | 0.02567 / 0.1770 / 0.269 | 0.00614 | 81.02% |
| BF16 TP1, BF16 TP2 (the control) | 3,982,406 | 0.0266 / 0.602 / 2.87 | 0.01167 / 0.1693 / 0.424 | 0.00274 | 92.44% |

Paired per position, KL(BF16 TP1 || A4 TP2) minus KL(BF16 TP1 || A4 TP1) has mean +0.00013 with a
standard error of 0.00048, and the ratio of means is 1.0017 (95% CI [0.990, 1.014]). Against the
BF16 TP2 arm as reference the ratio is 0.9969. The A4 stub's quality does not depend on the world
it is served at.

## The decomposition

On the ids all four dumps returned at a position (2,961,695 pairs), with q1 = A4TP1 - BF16TP1,
q2 = A4TP2 - BF16TP1, dA = A4TP2 - A4TP1 and dB = BF16TP2 - BF16TP1, two models make different
predictions. If TP2 re-draws part of the quantization noise, var(q2) = var(q1) and cov(q1, dA) =
-var(dA)/2. If TP2 adds error e independent of the noise, var(q2) = var(q1) + var(dA) and
cov(q1, dA) = 0. The share of the A4 delta that is added error is 1 + 2 cov(q1, dA) / var(dA).
Intervals are bootstraps over positions (500 draws), because the pairs of one position are not
independent.

| Quantity | Value | 95% CI |
|---|---|---|
| var(q2) / var(q1) | 1.0028 | [0.994, 1.011] |
| Added-error share of var(dA) | 0.007 | [-0.016, +0.030] |
| Re-drawn share of the noise, 1 - corr(q1, q2) | 0.189 | [0.183, 0.194] |
| corr(dA, dB) | 0.007 | [-0.007, +0.020] |
| var(dA) / var(dB) | 2.64 | [2.44, 2.87] |
| median \|dA\| / median \|dB\| | 4.33 (0.1167 / 0.0270) | - |
| p99 \|dA\| / p99 \|dB\| | 1.09 (0.688 / 0.633) | - |
| Kurtosis of dA / of dB | 7.4 / 32.8 | [7.1, 7.6] / [30.8, 35.7] |

Three of these carry the verdict on their own. The variance ratio and the added-error share say
that TP2 added nothing to the noise. The correlation of dA with dB says the A4 delta does not
track the BF16 control's delta, which is what a re-draw at the quantizers predicts and what a
shared upstream perturbation propagating linearly would not. The kurtosis contrast explains the
receipt's shape: the BF16 control's delta is a few chaotic positions on a near-flat stub (heavy
tails, small median), the A4 delta is many small flips everywhere (Gaussian-like, larger
median), and both tails are set by the stub's logit geometry, which is why the p99s agree while
the medians differ 4x.

## The mechanism on the runtime's own quantizers

`experiments/tp2_partial_sum_reroll_probe.py`, run through PrismaBuild on a GB10 inside the
pinned image (`pb-514-reroll-probe.sh`, result
`experiments/results/tp2_partial_sum_reroll_probe_514.json`). It feeds the operators the Tessera
routes bind to -- `scaled_fp4_quant` with a static global through
`native_ops.native_fp4_quant`, `dynamic_per_token_scaled_fp8_quant` through
`native_ops.native_fp8_quant` -- an activation and its TP2-rounded twin. The perturbation is the
exact TP2 arithmetic, `bf16(bf16(y1) + bf16(y2))` against `bf16(y1 + y2)` for a random split of
the fp32 pre-sum; it touches 23.2% of elements by one BF16 ulp. The activations are synthetic,
calibrated so the draw's amax matches the checkpoint's static global (983.04 for
`layers.1.mlp.shared_experts.gate_up_proj`, 28.29 for `down_proj`; `global = 6 * 448 / amax`).
512 rows, two draws (Gaussian and Student-t with 4 degrees of freedom), two units.

| Quantizer | Codes flipped | Error variance, TP2 / TP1 | Error re-drawn, 1 - corr | Power gain of the perturbation |
|---|---|---|---|---|
| `e2m1_group16_ue4m3_static` (E2M1, static global) | 0.93-1.04% | 0.9993-1.0005 | 3.1-3.4% | 72-87x |
| `fp8_per_token_dynamic` (E4M3, per-token) | 2.47-2.63% | 1.0031-1.0044 | 13.2-14.2% | 26.1-26.8x |
| `bf16_unquantized` (control) | 23.2% of elements differ by an ulp | (double rounding, 1.83x of a BF16 ulp's variance) | - | 1.0 by construction |

The ranges span the two draws and the two units; the JSON holds every cell. What the row for
each quantizer says: a one-ulp perturbation on a quarter of the elements flips about one code in
a hundred; the flips leave the quantizer's error variance where it was, to a tenth of a percent;
and the difference between the two arms' outputs carries 26x (E4M3) to 87x (E2M1) the power of the
perturbation that caused it. A TP-vs-TP comparison sees the last number and grows; a comparison
against the reference sees the middle number and does not.

The re-drawn share of one quantizer's error under a one-ulp perturbation (3%) is smaller than
the 18.9% the whole stub re-draws at the logits: layer 0's E4M3 quantizer, layer 1's E2M1
quantizers and four attention and MLP all-reduces sit in series, and each stage's re-draw is a
larger perturbation for the next. The probe establishes the signatures; it is not a model of
the stub's total.

## Why the sharded path is not the source

- The shard a rank decodes is held bit for bit to the parent's decode sliced, over both bodies
  and all three scale planes, on both axes (`tests/test_slice_unit.py`,
  `test_shard_decode_is_the_parent_decode_sliced`, `test_decode_codes_of_a_shard_is_the_parent_sliced`),
  and the serving seam cuts a real unit into the parent's own rows
  (`tests/test_serving_sharding.py`, `test_the_seam_cuts_a_real_unit_into_the_parents_own_rows`).
- The A-side static global is one scalar with no shard dimension on both trees:
  `trellis_input_global_scale` is a `(1,)` `BasevLLMParameter` at
  `src/tessera/serving/nvfp4_route.py` (`44d20d670` lines 150-152; HEAD lines 153-155, NaN
  sentinel refused), and the routed MoE loader writes each expert's `input_global_scale` as one
  scalar (`nvfp4_moe_route.py` at `44d20d670`, `_load_input_global_scale`, lines 444-473). A rank
  quantizes its slice of the activation with the same global TP1 uses on the whole.
- What is left is the perturbation every row-parallel Linear introduces at a world of two, the
  same one the BF16 control carries, entering quantizers that respond to it as measured above.

## The discriminators the issue asked for

The issue asked for discriminator A (the same E2M1 weights on the stock NVFP4 route at TP1 and
TP2), or B (`--quantization fp8` on the BF16 stub at both worlds) plus C (per-layer hidden-state
diffs). B ran to its TP1 floor (two dumps from one serve, bit-identical) and its TP2 arm died at
NCCL init on the RoCE `ibv_reg_mr` ENOMEM (`kl-fp8/tp2-fail.*` in the evidence directory,
2026-09-15T05:43Z). C was not run. A was built on 2026-09-16 as a stock-NVFP4 MoE adapter
(unpushed commit `2c5e357` in `/home/rob/tmp/tessera-514-stock-adapter-20260916`) and its TP2
serve reached `Application startup complete` on the sockets fabric, but no logprob dump was
taken from it. All three were indirect probes of one question -- does the sharded path add error
-- and the reference reading measures that quantity directly, at zero within about 1% of the
quantization-noise variance. A is superseded; the adapter is retained as bounded evidence and is
not merged.

## Scope

- One stub (4 layers, near-flat logits, top-1024 coverage about 0.26), one platform (sm_121),
  one image, eager, prefill 8 x 512. The reference reading is on the shared top-1024 support; the
  renormalized KL discards the mass outside it, as the receipt says.
- The A4 TP1 arm ran on sparklina and the TP2 rank 0 on sparky, as did the BF16 control's; the
  asymmetry is common to both and does not enter the decomposition.
- The probe's activations are synthetic; no hidden state of the stub was captured.
- The receipt's serves are on tree `44d20d670`, whose routed experts used the stock
  `materialize_stock` decode behind FlashInfer CUTLASS and whose dense NVFP4 tiles used the
  historical post-scale epilogue. The reading here is about those serves. The current tree's
  native routed path is covered by its own receipts, not by this page.

## Evidence

- Payloads: `/mnt/shared/dq-runs/glm-first-artifact-claude-20260914/tp2-equivalence-20260915/`
  (`kl-tp1-tp2/tp1.json.npz` `0539c6c2...`, `kl-tp1-tp2/tp2.json.npz` `06f7d86e...`,
  `kl-bf16/tp1.json.npz` `2df32781...`, `kl-bf16/tp2.json.npz` `58de614e...`), the sha256s the
  receipt table names and the analysis script checks.
- `experiments/results/glm53_a4_stub_tp2_reference_kl_514.json`, written by
  `experiments/tp2_reference_kl_514.py`; `tests/test_tp2_reference_kl_514.py` binds it to the
  receipt's payloads and derives its ratio.
- `experiments/results/tp2_partial_sum_reroll_probe_514.json`, written by
  `experiments/tp2_partial_sum_reroll_probe.py` under PrismaBuild on sparklina (action
  `834f72c627bb9261b7be4295e4863e20797885e2df1852bea8ee8b8bf635a1a0`, rc 0, 7.2 s; the first
  submission `254e993a7c95...` ran every cell to the same numbers and failed only on writing
  through the shared mount's root_squash).
