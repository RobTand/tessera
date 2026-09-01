# Tessera vs EXL3 at matched bpw: Tessera loses by 1.72×

**Measured 2026-09-01.** `experiments/exl3_arm_glm_experts_v2.py` (EXL3 arm,
inside Mia's image) and `prismaquant/experiments/glm53_expert_menu.py` (every
other arm). Six real routed-expert projections from GLM-5.3-Flash BF16
(layers 5/20/42, expert 0, `gate_proj`/`up_proj`), real cached activations from
`glm53-bf16-pread-probe-1469b9b-20260830`, tokens split fit/eval, **every arm
scored on the disjoint eval half**. Relative functional error on `y = X Wᵀ`.

This is the comparison the project has wanted since the EXL3 work began, and
which five earlier attempts failed to produce. It is a **negative result for
Tessera** and it changes the build order.

## The number

| arm | bpw | contract | rel_err | vs Tessera 4.0 |
|---|---:|---|---:|---:|
| Tessera `E2M1_K1` | 3.5000 | W4A16 | 0.12666 | 1.301 |
| **Tessera `E2M1_K2`** | **4.0000** | W4A16 | **0.09738** | 1.000 |
| **EXL3 K=4 (MCG)** | **4.0117** | W4A16 | **0.05653** | **0.581** |
| NVFP4 RTN | 4.5000 | W4A16 | 0.08294 | 0.852 |
| NVFP4 GPTQ+JSO | 4.5000 | W4A16 | 0.06595 | 0.677 |
| NVFP4 GPTQ+JSO | 4.5000 | W4A4 *(as served)* | 0.10806 | 1.110 |
| FP8_E4M3 GPTQ | 8.0156 | W8A8 *(as served)* | 0.03050 | 0.313 |

**EXL3 at 4.0117 bpw is 1.72× better than Tessera at 4.0000 bpw**, on the same
weights, the same held-out tokens, and 0.0117 bpw *more* budget. It also beats
PrismaQuant's full production NVFP4 render at 4.5 bpw by 1.17×, at half a bit
less. EXL3 is a strong format and the earlier documents were right to treat it,
not NVFP4, as the real comparator.

The rate matches by construction: K=4 trellis plus fp16 `suh`/`svh` diagonals is
`4 + 16/2048 + 16/4096` = **4.0117 bpw** on these shapes — the same figure
measured from Mia's safetensors headers, so the arm reproduces her artifact's
accounting exactly.

## How much of the gap is calibration, and how much is the coder

**Tessera's encoder is activation-blind.** It sees no activations at all. EXL3
builds a Hessian from the fit half and does LDL-ordered error compensation
inside the trellis search. That is a real asymmetry and it must be priced
before the gap is attributed to the format.

The same run measures what activation-awareness is worth on this data, by
running NVFP4 at 4.5 bpp W4A16 both ways:

```
NVFP4 RTN      (blind)  0.08294
NVFP4 GPTQ+JSO (aware)  0.06595      = 1.258x
```

So on this model, at this rate, on these tokens, activation-aware rendering is
worth **1.258×**. Tessera's deficit is **1.72×**. In log terms calibration
accounts for `ln(1.258)/ln(1.72) = 42%` of the gap. Credited the *full* factor,
a hypothetical activation-aware Tessera lands at 0.0774 — **still 1.37× behind
EXL3 at the same size**.

**Read that as a floor on the coder gap, not a settled split.** 1.258× is
NVFP4's response to compensation, measured on a scalar-quantized format whose
compensation is bolted on afterwards. EXL3's LDL ordering is integrated with the
trellis search, and a trellis coder may respond to a Hessian by more than a
scalar one does. If Tessera's coder responded like EXL3's, calibration could
close more than 42%. Nothing here measures that, because Tessera has no
activation-aware encoder to measure.

Tessera's arm is otherwise near its best: rotation and diagonals were measured
at ~1% on the weight screen (`tessera-vs-nvfp4-glm-experts-2026-09-01.md`), far
short of 1.72×.

## Why this arm quantizes rather than decoding Mia's artifact

`experiments/exl3_arm_glm_experts.py` read `GLM-5.3-Flash-EXL3-TR3-4bpw`
directly and got the classic garbage signature — `norm_ratio` 0.9992, `cos`
0.0011, `rel_err` ≈ √2. Right magnitude, no correlation. Chasing it established
three things worth keeping:

1. **The reader path is exactly right.** `LinearEXL3.get_weight_tensor()`
   reproduces the quantizer's *own* verified reconstruction — the formula
   `quantize_exl3` scores `block_nmse` against — to `rel 0.00044, cos 1.0000`.
   All three forward paths (`bc.run_alloc` fast path under
   `AUTO_RECONSTRUCT_THRESHOLD=144`, forced `reconstruct=True`, and >144 rows)
   agree to the digit. A wrong reconstruction cannot be blamed on the reader.
2. **It is not an expert permutation.** Mia's expert 0 reconstruction was
   correlated against all **288** BF16 experts in that layer: top |cos| 0.0014,
   and expert 0 ranks 6th of 288. It correlates with nothing, in that layer or
   in neighbouring layers.
3. **The version and codebook match.** The image ships exllamav3 **0.0.43**, the
   exact version the artifact's ABI names, with `codebook_mcg_mult =
   0xCBAC1FED` — the artifact's `mcg_multiplier_hex`. The emitted tensor set
   from a fresh quantization is byte-for-byte the same shape set the artifact
   stores (`suh` 4096, `svh` 2048, `trellis` [256,128,64] int16, `mcg` int32).

That leaves the stored trellis contents, and **the artifact says so itself**:
`exl3-mcg-storage-abi.json` carries `serving_reader_qualified: false` and an
empty `qualified_tp_sizes`, reason *"ExLlamaV3 v0.0.43 has no audited GLM-5.3
TP model load/inference receipt"*. `storage_checkpoint_verified` is true — the
**bytes** were checked, the **decode** never was.

**This is recorded as a strong indication, not a verdict.** Proving someone
else's checkpoint defective from a negative result is exactly the inference this
project distrusts, and there remain unexcluded explanations (a writer-side
transform outside the four stored suffixes; a source checkpoint we do not
have). What is settled is that *this* reader, at the version the artifact names,
does not reconstruct it — and that quantizing the same weights with EXL3's own
quantizer is both the fairer arm and the stronger one.

## What this means for the plan

**It de-prioritises the vLLM Tessera kernel-lane backend.** The backend is a
large build whose purpose is to serve a format that currently loses 1.72× to the
format it would replace, at the same size. Building it now would produce a
GLM-5.3-Flash artifact size-matched to Mia's and measurably *worse* than it.

The evidence points at two things to do first, in this order:

1. **Give Tessera an activation-aware encoder.** It is the one identified,
   quantified deficit — worth ≥1.258× and plausibly more on a trellis — and it
   is the "local question" toolkit PrismaQuant already owns (Hessians, LDL,
   GPTQ-style ordering). Nothing about the wire format has to change.
2. **Then re-run this exact harness.** It is six tensors and two commands. If
   the gap closes to near parity, the backend is worth building; if it stalls
   near 1.37×, the deficit is the coder and the rate-ceiling work
   (`tessera-rate-ceiling-2026-09-01.md`) is the honest next lever instead.

## Scope, honestly

- **A screen, not a KL.** Relative functional error ranks; it does not promote
  (principle 3). No served number here.
- **n = 6** projections, one expert per layer, three layers, `gate_proj`/
  `up_proj` only. `down_proj` is unpriced — the probe caches one input per
  packed-expert entry at hidden dim, so ~⅓ of expert parameters are excluded.
- **The EXL3 Hessian is rank-128** on a 4096-dim input (128 fit tokens). A real
  conversion calibrates on far more, so this arm is if anything *conservative*
  toward EXL3 — it won on held-out tokens despite a rank-deficient Hessian.
- **Both Tessera rows are the kernel lane**, which no runtime executes
  (principle 9). This measures a format, not an artifact.
