# The window GEMV folded the activation scale into bf16, and that fold is the size of #110 (2026-09-04)

> **STATUS: the arithmetic is settled, one real defect is fixed, and the
> "factor of forty" is explained -- it was a screen priced off the operating
> point, not evidence of a second term. #110 still needs its served re-run to
> close, and that run has not had a GPU.** The in-process leg is unambiguous:
> the lane's accumulation order is worth **1.6e-07** and does not move a single
> bf16 output word, the fold this document names was worth **1.4e-03**, and
> with it gone the lane's bf16 output matches `torch._scaled_mm`'s on
> **99.90-100%** of elements. Every other way the two served arms could have
> differed is closed by an artefact rather than an assumption (section 2b):
> same inode for the weights, prefill KL exactly `0.000000` over 4088
> positions, and serve logs identical but for the intended 112 refusals.
>
> **What changed in this pass is section 3.** The first pass priced the fold
> with a Gaussian noise model on the BF16 teacher and got KL 3.2e-04 against a
> measured 0.012111, then said honestly that one of the two readings was wrong
> and could not say which. It can now: the screen was. Pricing the fold as the
> deterministic rounding it actually is, on the served position set, in the
> served decode regime, through the served estimator, reads **KL >= 0.007160 at
> 95.70% top-1 against the served 0.012111 at 91.02%** -- a factor of 1.69, not
> 38. A matched pair says where the 38 came from, and it is not correlation:
> replacing the fold with independent Gaussians of the same rms *in this
> harness* moves the number by 1.29x, while removing the route's per-token FP8
> **activation quantiser** -- one variable, same regime, same arm, same 256
> positions, same rms to within 0.5% -- moves it **68.9x**, from 0.007788 down
> to 0.000113, landing on the first pass's independent 0.000318. The screen had
> priced a W8A8 term on a trajectory that was not carrying W8A8's own
> activation rounding.
>
> The remaining 1.69x is the emulation's **weight operating point**, and that
> is measured too rather than asserted: degrade the weights toward the served
> arms' distance from BF16 (`--weight-bits 4`, same regime, same arm, same
> chunks) and the fold reads **0.012073**, a factor of 1.48, against a served
> 0.012111 -- agreement to three digits that is a coincidence of magnitude at
> 128 positions, but the right direction and the right size.
>
> **This is still a screen** -- CPU, HF rather than vLLM, fp32 rather than bf16
> residual, RTN rather than the Tessera wire -- and the served re-run
> (`6c90ba1b`, queued and `ready` since this branch began) is confirmation now
> rather than adjudication; section 4 names the three outcomes it can have.
> Nothing below is a placeholder for a number that was taken and disliked, and
> section 3a records a mechanism I proposed for the 68.9x, tested, and could
> not confirm.

**What #110 asked.** Two serves of one checkpoint through one inode, differing
only in whether `prepare_fp8_gemv` could build, disagreed in the decode regime:
`KL >= 0.012111`, top-1 91.02%, over 256 M = 1 positions. Is that accumulation
order (document it and close), or a defect (fix it)?

**The answer.** A defect, and not accumulation order. It is the same size as
what was measured -- within a factor of 1.69 on a CPU emulation whose every
substitution points the same way -- so the remaining question is whether it is
ALL of what was measured, and only the served re-run answers that.

---

## 1. The defect: a scale folded into a bf16 operand

`fp8_gemv.streamed_apply` handed the kernel

```python
a_val = (a_q.float() * a_scale).to(torch.bfloat16)      # before
```

while the fallback calls `torch._scaled_mm(a_q, b.t(), scale_a=a_scale,
scale_b=...)`, which multiplies the fp8 codes exactly and applies both scales
in its fp32 epilogue.

The module docstring justified the fold: *"every E4M3 code is exact in bf16, so
the dequant is exact and what the GEMV computes is the route's own product up
to fp32 summation order."* The first clause is true and tested
(`test_every_legal_e4m3_byte_is_exact_in_bf16`: 254 finite bytes, all exact).
The second does not follow. A code carries four significant bits and a
per-token fp32 scale twenty-four; their product needs up to twenty-eight and
bf16 keeps eight.

**Priced** (`experiments/gemv_a_side_precision.py`). Its analytic leg is CPU
and fp64 with no kernel in it: every E4M3 code is exact in bf16 (`True`), the
rel rms of `bf16(code * a_scale)` against the exact product is **1.616e-03**,
and at the output of a K = 1024 dot product that costs **1.612e-03**. That is
not merely *of the order of* one bf16 rounding, it **is** one: over a
log-uniform magnitude a single bf16 rounding measures rms **1.659e-03** with a
worst case of **3.891e-03**, against the dtype's own bound of 2^-8 = 3.906e-03
(bf16 keeps eight significand bits). The first pass quoted "half a bf16 ulp is
1.953e-03", which is 2^-9 -- the half-ulp relative bound at the *top* of a
binade, not the bound; the artefact
`/home/rob/tessera-runs/ts110/gemv_a_side_precision.json` still carries that
field name, and the script now measures the bound instead of quoting it.

Its kernel leg runs the real lane on a GB10 and prices every arm against an
fp64 reference of the product both claim to compute, at 1024 x 1024, per M
(`/home/rob/tessera-runs/ts110/gemv_a_side_precision.json`, pool action
`084553c5`):

| M | lane as fixed, fp32 | lane with the fold, fp32 | lane vs ITSELF, twice | bf16 words identical to the fallback: folded -> fixed |
|---:|---:|---:|---:|---:|
| 1 | **1.349e-07** | 1.402e-03 | 1.709e-07 | 62.60% -> **99.90%** |
| 2 | 1.357e-07 | 1.674e-03 | 1.595e-07 | 58.45% -> **100.00%** |
| 4 | 1.326e-07 | 1.152e-03 | 1.613e-07 | 78.20% -> **99.98%** |
| 8 | 1.316e-07 | 1.579e-03 | 1.594e-07 | 58.69% -> **99.95%** |

Read the columns:

* **The accumulation order #110 asked about is 1.6e-07** -- the lane run twice
  on identical input, its `atomicAdd` reduction free to land in a different
  order -- and it changes **no** bf16 output word (100.00% identical, all four
  M). That is the magnitude the issue asked to have recorded somewhere a
  receipt can cite. It is not what was served.
* **The fold was ~10 000x that**, and the `bf16_A_only` control -- the same
  rounding applied in pure fp64, no kernel -- reproduces the folded arm to ten
  significant figures at every M, so the fold is the whole of the pre-fix
  discrepancy and the kernel contributes nothing to it.
* **`torch._scaled_mm` is not the inexact arm.** Its error against fp64
  (1.657e-03 at M = 1, bf16 output) equals the *fixed* lane's to ten figures:
  both are one bf16 rounding of the output and nothing else. #110 left open
  which arm was closer to the truth; it was the fallback, and now they are the
  same.

**The fix** hands the kernel the codes -- which *are* exact in bf16 -- and
multiplies the fp32 output by `a_scale`:

```python
a_val = a_q.float().to(torch.bfloat16).contiguous()     # after
return (_gemv_path(a_val, tensors, meta) * a_scale).to(torch.bfloat16)
```

This is the rule `src/tessera/serving/bf16_route.py` already states for the
**weight** side, in its own words -- the row scale "is applied in fp32 to the
GEMM's fp32 output, one rounding at the end", never folded, "because it is a
property of bf16's 7-bit mantissa rather than of any rate" -- and
`test_value_family_scale_is_applied_on_the_output_not_the_tile` pins it. The
same rule, the other operand. The activation contract does not move: the
route's quantiser still runs on every path and what the kernel multiplies is
its output; only where `a_scale` enters changes.

## 2. Why no existing receipt could see it

Three of them looked at the right objects, and none of them looked at this one.

* **`tests/test_kernel_window_gemv.py`** compares the kernel against
  `(tile.float() * scale) @ x` **for the `x` the kernel was handed**
  (`tessera-window-gemv-2026-09-02.md` §3). That is a statement about the
  kernel and it is true. The fold is in the caller that builds `x`.
* **"bit-exact on 196/196 reach units"** is a statement about the **decoded
  tile** against the torch decoder's, never about the GEMM over those bytes.
  Both arms of #110 decode the same bytes; `prepare_fp8_gemv` refuses to serve
  otherwise. The phrase is now qualified wherever this lane uses it.
* **`test_gemv_and_materialised_agree_within_fp32_summation_order`** did
  compare the two GEMMs -- with a tolerance of `2 * fp32_bound + 2^-7 * |ref|`.
  The second term is one bf16 ulp of the **output**, 0.78% relative: wide
  enough to absorb exactly the term the test's name denied. It also ran only at
  `m = 4`. **M = 1 -- the only shape #110's decode regime scores -- was never
  compared against the fallback at all.** It is now, and the test additionally
  requires the two arms' bf16 outputs to be bit-identical on all but 1% of
  elements.

## 2b. Everything else about the two arms is the same, checked

The fold is only the leading suspect if it is the only difference. #102's two
arms were audited from the artefacts on disk, and every other candidate is
closed by a fact rather than by an assumption:

* **The bytes are the same file.** `ts83/armA` and `ts83/armB` are not two
  copies; `model.safetensors` is inode `11665664` in both, as are `config.json`
  and `tessera_gridbook_manifest.json`. A weight difference is not merely
  unlikely, it is impossible.
* **The prefill regime read exactly `0.000000` over 4088 positions.** Prefill
  is M = 512, where both arms take `_materialised_path`. So attention, the
  norms, the embeddings, `lm_head`, the wire decode and `torch._scaled_mm`
  itself are bit-identical across arms. Whatever separates them acts **only on
  M <= 8 forwards** -- and inside this route the only thing that changes at
  M <= 8 is the branch in `streamed_apply`.
* **The serve logs differ in nothing but the intended refusal.** Normalised
  (digits and timestamps stripped, deduplicated), `serve_ts102-armA.log` and
  `serve_ts102-armB.log` differ only in the model path, the run-scoped hashes,
  and arm B's 112 `window GEMV lane did not prepare` warnings. Both chose
  `FLASH_ATTN` "out of potential backends: ['FLASH_ATTN', 'FLASHINFER',
  'TRITON_ATTN', 'FLEX_ATTENTION']", both report FlashAttention version 2, both
  ran the FlashInfer autotuner to `Saved 0 configs ... (0 new, 0 from previous
  config)`, and both emit the same single `jit_monitor` warning for
  `_topk_log_softmax_kernel`. Two hypotheses die here: the autotuner did not
  pick different kernels for the two arms, and the read-only
  `TORCH_EXTENSIONS_DIR` did **not** silently refuse some *other* JIT build in
  arm B -- the only JIT compilation either serve reports happened in both.

That leaves the `M <= GEMV_MAX_M` branch as the sole remaining candidate, and
the branch contains exactly two differences from the fallback: the fold, and
fp32 summation order. Summation order is measured at 1.6e-07 and changes no
bf16 word. So the fold is the only term of any size anywhere in the arm.

## 3. What that term is worth as a KL, measured without a noise model

A per-Linear relative error is not a KL. The first pass turned one into the
other with `experiments/gemv_a_side_propagation.py`, which injects
**independent Gaussian noise** of the measured relative size at every served
Linear's input on the BF16 teacher, and read **KL 0.000318 / 98.44% top-1** on
the served position set -- one part in thirty-eight of the measured 0.012111 /
91.02%. That number is real and reproducible, and as an answer to #110 it was
wrong. It is kept here as a control, because *why* it was wrong is the finding.

`experiments/gemv_a_side_exact_fold.py` removes the model. It runs the same
model twice, in each arm's **actual arithmetic**, over the served position set,
and takes the KL between the two logit sets:

| | what the Linear computes |
|---|---|
| arm A, the lane before the fix | `bf16(a_q * a_scale)` -> fp32 GEMM against the decoded fp8 tile -> bf16 out |
| arm B, `_scaled_mm` and the lane after it | the same `a_q` -> fp32 GEMM of the CODES -> `* a_scale * w_scale` in the epilogue -> bf16 out |

Both arms share the quantiser, the weights, the positions and every other op,
so the pair is matched by construction and one thing varies. Two properties of
the served metric are reproduced rather than approximated: the **regime** (a
clean prefill fills the KV cache -- in the serve both arms take
`_materialised_path` for it -- and each scored prefix is then ONE M = 1 forward
off that cache, so the fold touches only the scored token's own path), and the
**estimator** (the served headline is `kl_tool`'s lumped DPI lower bound over
the top-1024 intersection, not a full-vocab KL, so this reports that estimator,
imported from `kl_estimator`, beside the full-vocab number).

### 3a. The reading

Every row is 256 positions unless marked -- the served set, 8 chunks x stride 16 -- on
Qwen3-0.6B, CPU-only, `/home/rob/tessera-runs/ts110/`. The **regime** column
matters and is not decoration: `decode` is the served one (a clean prefill
fills the KV cache, then each scored prefix is one M = 1 forward), `prefill`
perturbs all 512 rows of a single forward and so also perturbs cached keys and
values the serve builds clean in both arms. Rows are only comparable within a
regime.

| what was priced | regime | KL >= (served-shape) | full-vocab KL | top-1 |
|---|---|---:|---:|---:|
| **#110, as served** | decode | **0.012111** | -- | **91.02%** |
| **the fold, exactly** | decode | **0.007160** | 0.007572 | **95.70%** |
| independent Gaussian, same rms, same point | decode | 0.007788 | 0.008105 | 96.09% |
| the same Gaussian, FP8 activation quantiser removed | decode | **0.000113** | 0.000125 | 99.22% |
| the fold, all 512 rows perturbed (over-applied) | prefill | 0.012576 | 0.013138 | 93.75% |
| independent Gaussian of the same rms | prefill | 0.009766 | 0.010261 | 95.70% |
| the first pass's screen (BF16 teacher, no quantiser) | prefill | 0.000318 | -- | 98.44% |
| the fold at a degraded WEIGHT rung (int4, 128 pos) | decode | 0.012073 | 0.012917 | 96.09% |
| control: one arm against itself | either | 0.000000 | 0.000000 | 100.00% |

Read the rows:

* **The fold is the same size as what was served, not one-fortieth of it.**
  0.007160 against 0.012111 is a factor of 1.69 in the lumped KL and 1.30 in
  amplitude, where the first pass reported 38. The top-1 disagreement it
  produces is 4.30% against a served 8.98%.
* **The missing factor was never the noise MODEL.** Replacing the fold's
  deterministic rounding with independent Gaussians of the same measured
  relative rms, at the same point, in the same harness and the same regime,
  changes the reading by **1.09x** in the served decode regime
  (0.007160 -> 0.007788) and 1.29x in prefill (0.012576 -> 0.009766) -- not
  38x, in either direction. The correlation the first pass worried about is
  worth almost nothing.
* **It was the operating point of the ACTIVATION path, and the pair that says
  so varies one thing.** Decode regime, Gaussian arm, same 256 positions, same
  relative rms to within 0.5% (1.3329e-03 against 1.3398e-03); the only
  difference is whether the route's per-token E4M3 activation quantiser is in
  the loop. Quantiser on: **0.007788 / 96.09%**. Quantiser off: **0.000113 /
  99.22%**. That is **68.9x from one variable**, and the off arm lands on the
  first pass's independent 0.000318 / 98.44%. The screen priced a W8A8 term on
  a trajectory that was not carrying W8A8's own activation rounding.
* **The first pass's operating-point control moved the wrong axis.** It
  degraded the **weights** (int4 RTN) and found 1.5x, which is why the screen
  read as adequate. The axis that mattered was the activation quantiser, and
  nothing in the first pass moved it.
* **The regime is worth 1.76x, in the direction that flatters the fold.**
  Perturbing all 512 rows reads 0.012576 and happens to sit on top of the
  served number; it is an over-application and is reported as one.

Why a step quantiser amplifies a small term at all -- **a mechanism I
proposed, tested badly, and am withdrawing.** E4M3 per-token quantisation is a
step function of local width `u`, so a relative perturbation `d` upstream of it
should flip a fraction of order `d/u` of the downstream codes by one whole step
and give variance `~d*u`: linear in `d`. That predicts halving `d` halves the
KL with the quantiser on and quarters it with it off. Two things are wrong with
it, and I found the second only after running the test:

* **The placement is wrong.** The Gaussian is injected on `a_q * a_scale` --
  *after* this Linear's own quantiser has already run. It never meets the step
  function I was arguing about; it meets the quantisers of Linears downstream,
  through attention and two norms. The mechanism may still be the right
  physics, but not at the point the harness perturbs.
* **The ON row is not a scaling pair, so it cannot test anything.** Its
  baseline (`exact_fold_gaussian_control.json`) passed no `--rel` and used the
  **measured per-Linear** relative size, of which 1.4005e-03 is only the mean
  over 196 Linears; `lin_half_acton` injected a **constant** 7.0024e-04
  everywhere. Two variables move, and a constant field that over-injects at
  sensitive Linears would raise the KL in every chunk exactly as observed. The
  four-of-four sign agreement I first read as "not noise" is evidence of that
  confound, not of a harness bug.

| pair, 128 positions, paired by chunk | full rel | half rel | ratio | predicted |
|---|---:|---:|---:|---:|
| quantiser ON -- **confounded, measured-rel vs constant-rel** | 0.010631 | 0.016021 | 0.66 | -- |
| quantiser OFF -- a real pair, both `--rel` | 0.000229 | 0.000196 | **1.17** | 4.0 |

So one open anomaly survives and it is the OFF row: with one shared generator,
so the field is literally `rel` times a fixed realisation, halving `rel` moved
the KL by 1.17x where a square law demands 4.0. I do not have an explanation
and I am not running a fifth emulation leg to find one, because it is outside
what #110 asks. **Neither row touches the two numbers this branch rests on**:
the 68.9x pair holds the measured rel fixed (1.3329e-03 against 1.3398e-03) and
varies only the quantiser flag, and the 0.007160 headline uses the
deterministic fold and passes no `--rel` at all. What survives is that the
variable was *located*. Its mechanism is not explained, and the paragraph that
claimed to explain it was mine and is withdrawn.

### 3b. What is still substituted, and which way it points

This is a screen and it says so. Three substitutions remain:

* **The weights are RTN, not the Tessera wire -- and this axis is now
  measured, and it covers the residual.** The weight-side *arithmetic* is the
  served one (per-output-row E4M3 codes plus a per-channel scale is exactly
  what the TESSERA_FP8 route decodes the window wire to), but the codes are
  RTN's, so the emulated model sits far closer to the BF16 teacher than the
  served arms' 0.436 nats. `--weight-bits 4` moves it toward the served
  operating point: decode regime, folded arm, chunks 0-3 in both, only the
  weight rung differs, **0.008159 -> 0.012073, a factor of 1.48**, at an
  unchanged 96.09% top-1 (`decode_wb4.json`). For scale, the served headline is
  0.012111. Two cautions against reading that agreement as more than it is: at
  128 positions the per-chunk ratios run 0.73 to 2.23, so 1.48 is a mean and
  not a tight one; and int4 RTN is not the Tessera wire, so landing on 0.012111
  to three digits is a coincidence of magnitude, not a reproduction. What it
  does establish is the *direction and rough size* the first pass had to guess
  at.
* **The size matches; the character does not, yet.** The weight-degraded run
  reads 0.012073 against a served 0.012111, but at **96.09% top-1 against a
  served 91.02%** -- the serve turns the same KL into roughly twice the top-1
  flips (about 5 of 128 positions here against about 11.5). At 128 positions
  that is a handful of flips and not much of a signal, but it is the wrong
  direction to leave unremarked: the emulation reproduces the *size* of the
  disagreement and not yet its *shape*. The served re-run reads both.
* **The residual stream is fp32; vLLM's is bf16.** Both arms share it, so it
  cannot manufacture a difference, but a bf16 trajectory is the noisier one and
  this axis is unmeasured.
* **HF, not vLLM.** Attention kernels, RoPE and norms differ in their last
  bits. Shared by both arms; unmeasured as an amplifier.

All three plausibly point the same way -- they make the emulation *quieter*
than the serve -- and now one of the three has a run behind it and confirms the
direction, worth 1.48x on its own. So 0.007160 reads as a floor **on the weight
axis specifically**; the other two axes are still asserted, and the served
re-run is what turns any of it into a number.

Two provenance notes for anyone reconciling the JSONs: `exact_fold_served_set.
json` and `exact_fold_gaussian_control.json` predate the `regime` and `arm_a`
fields and are both prefill/folded and prefill/gaussian respectively; and the
first pass's 0.000318 comes from `experiments/gemv_a_side_propagation.py`, a
different harness, so it is a corroborating reading and not a row of this
table.

## 4. What remains, and the state of the queue that holds it

1. **The served re-measurement. This is the only thing that closes #110.**
   `experiments/ts110_served_campaign.sh` reruns #102's exact pair off this
   tree at `TAG/PREFIX = ts110` (~6 min per arm), writing beside #102's dumps
   rather than over them. Arm B's code path is untouched by the fix, so
   armB-new against #102's armB is a free control that must read 0.000000.
   **Three outcomes, and the receipt must name which one it got:** the pair
   collapses to that control's floor, and the fold was the whole of the served
   difference; it reads about 0.005 (0.012111 minus what section 3 prices),
   and the fold was most of it with a second term left; or it is unmoved, and
   whatever separates the two arms is not this Linear's arithmetic at all.
2. **The CUDA leg HAS now executed, and it failed -- because the fix was
   never run on a GPU.** Action `6c90ba1b` placed at 05:49 on 2026-09-04 and
   its step-2 gate ran both modules with a device: **1 failed, 85 passed** in
   210 s, and the gate refused to spend the serve, which is the gate doing its
   job. The failure is this branch's own.
   `test_rate1_columns_fall_back_inside_the_decode_regime` referenced the M = 2
   GEMV path against `bf16(a_q * a_scale)` -- *that is the folded arithmetic
   #110 is about*. The fix changed the lane to hand the kernel `bf16(a_q)` and
   multiply `a_scale` into the fp32 output, so the lane and its own reference
   had disagreed by exactly the term the fix removes, from the moment the fix
   landed. The CPU suite could not see it: the test is `@requires_cuda` and was
   one of the 77 skips this receipt kept reporting as "all pass". The reference
   now follows the lane (M = 2 references `bf16(a_q)` and carries `a_scale` on
   the result with the accumulation bound scaled to match; the M = 4 row, which
   falls to `_scaled_mm`, is unchanged), and the action **auto-requeued for
   attempt 2**, which will re-run the gate against the correction.

   The two assertions the review named as unverified **passed**, at all four M:
   `same >= 0.90` in
   `test_gemv_and_materialised_agree_within_fp32_summation_order` and
   `lost > 3 * kept` in
   `test_the_lane_multiplies_the_codes_and_scales_the_output_not_the_operand`
   are both among the 85. So the specific worry is answered, and the general
   one was righter than it was put: the problem was never two thresholds, it
   was that **no CUDA test had run against the fix at all**, and the first run
   that did found a defect. Not yet verified on a GPU: the correction itself.

   For completeness, on CPU on this tree:
   `pytest tests/test_serving_fp8_gemv.py tests/test_kernel_window_gemv.py` =
   **9 passed, 77 skipped**, and
   `test_a_code_times_a_per_token_scale_is_NOT_exact_in_bf16` is one of the 9:
   it pins the arithmetic the fix rests on -- one bf16 rounding, bounded by
   2^-8, rms 1.6e-03 -- on a box with no GPU, which nothing else here could do.

3. **The kernel leg is measured at K = 1024 only, and the served K are three.**
   Qwen3-0.6B's served Linears carry K = 1024 (qkv, gate_up), 2048 (o_proj:
   16 heads x 128) and 3072 (down_proj) -- not the 4096 a reader might assume
   from the config's `intermediate_size`, which is an N. So the fixed lane's
   fp32 error (1.349e-07) and its bf16-identity fraction (99.90%) are properties
   of one of the three shapes. Step 4 of the campaign sweeps the other two; it
   runs LAST and non-fatally, so it cannot cost the serve above it.
4. **More positions** (#110's item 1) is a corpus question, not a budget one:
   the decode stride is pinned to the serve's KV block size and the contract is
   8 x 512, so more positions means a new corpus contract and a re-dumped
   decode teacher.
5. **The "look outside the GEMM" hypothesis is dead and should not be re-funded.**
   The candidate was that a read-only `TORCH_EXTENSIONS_DIR` refuses *every*
   torch JIT build in that container, not only this lane's. It does not:
   section 2b's log diff shows the one JIT compilation either serve performed
   (`_topk_log_softmax_kernel`) happened in **both** arms, and the attention
   backend and autotune outcome are identical.

### 4a. Why the serve has not run, stated from the ledger and not from a snapshot

Every GPU leg went to the PrismaBuild pool. The precision leg (`084553c5`)
waited 35 minutes as the oldest of fifteen GPU-needing items, then ran. The
served campaign (`6c90ba1b`, gpu=1 mem_gb=24) reached a worker once, died in
its first seconds on a permission fault -- `ts83/ext-A` holds root-owned JIT
lock files written by the serve container, so a host-side extension load there
gets EACCES -- was fixed with its own writable `TORCH_EXTENSIONS_DIR`, and has
been `ready` ever since. It has never served.

**The first pass reported that as a leaked reservation, and that reading does
not survive a second look.** Action `66266919` from `/home/rob/tmp/wf109` does
carry `detail.status: "failed"` and `returncode: 1` while sitting in
`pb-queue/claimed/` holding both of sparky's GPU tokens -- but that `detail` is
the record of a **previous** attempt. The same record carries
`requeued_unix` 1788503687 and a later `claimed_unix` 1788504804, its lease was
being heartbeated live, and `/home/rob/tessera-runs/ts109/campaign-eager.log`
shows rep 2 of that worktree's latency A/B starting at 08:48:55Z with a vLLM
serve on `ts83/armB` under it. The tokens were held by work that was running,
not by a corpse. That action has since completed and left `claimed/`; both GPU
tokens went straight to `87cf849b` and `ef037166`, two other agents' actions
that were ahead of `6c90ba1b` in `ready/`, and at the time of writing sparky
offers **zero** free GPU tokens and sparklina's one is held by an out-of-pool
pin (`out-of-pool-ts60-encode-sparklina`).

So the blocker is contention, not a defect, and the pool's own issue tracker
already names the structural half: `RobTand/prismabuild#5` -- a box-local
worktree pins its actions to one box, so `/home/rob/tmp/wf110` can only ever be
served by sparky however idle the rest of the fleet is. The *readability*
failure that produced the misdiagnosis is now filed as
`RobTand/prismabuild#10`: a claimed record whose `status`/`detail` describe an
attempt that already ended reads exactly like a leak to anyone auditing the
queue, and it took a live lease check and a log tail to tell the two apart. Two
readers hit it independently, which is the argument for fixing the record
rather than the readers.

Re-checked at the end of this pass, and the ledger is clean on its own terms:
`reservations/sparky/held/` contains exactly two entries, `87cf849b` and
`ef037166`, each holding one of the box's two GPU tokens, each with a lease
heartbeating within the last minute; `free/` holds 24 tokens, all `cpu-*` and
`mem_gb-*`. There is no orphaned GPU reservation. `6c90ba1b` is one of
nineteen actions in `ready/` waiting on a two-token pool.

The propagation and exact-fold legs ran **locally, on CPU, at `nice -n 19` on
three torch threads**, not through the pool. That is stated rather than
excused: `pbrun --demand cpu=2` and `--demand cpu=2,mem_gb=0` from this
checkout both queued and neither placed inside 120 s and 150 s, because a
box-local checkout is pinned to sparky (prismabuild#5) and sparky's 48 mem_gb
tokens were at that moment held in their entirety by the action above. Nothing
in either leg touches a GPU, and both ran under `CUDA_VISIBLE_DEVICES=` so that
the kernel, not a promise, enforced it.

## 5. The sign, at least, is the one the fold predicts

#110 reports the arms against the BF16 teacher as arm A `KL >= 0.436065` and
arm B `KL >= 0.432477`, and calls the direction surprising. Under this
mechanism it is the expected direction: before the fix arm A carried one
rounding arm B did not, so arm A should sit slightly further from the teacher.
At 256 positions that is a consistent sign and not evidence.
