# The window GEMV folded the activation scale into bf16, and that fold is the size of #110 (2026-09-04)

> **STATUS: both serves have landed and #110 is fully accounted for.** The
> re-run is outcome 2 of the three section 4 named in advance -- the fold was
> MOST of the served disagreement and not all of it: byte-identical bytes, same
> inode, same image, the two arms went from `KL >= 0.012111` at 91.02% top-1 to
> **`KL >= 0.005947` at 96.88%**. The replicate (section 6d) is then outcome 1
> of the two section 6b named: serving the fixed lane a SECOND time, changing
> nothing, it disagrees with **itself** by **`KL >= 0.005985` at 95.31%** --
> the same size. So the residual is not an arm-vs-arm difference at all.
>
> **The two answers #110 asked for.** It was a defect AND accumulation order,
> in roughly equal halves; the issue offered them as alternatives. And the
> magnitude it asked to be recorded: **`KL >= ~0.006` at `~95%` top-1 is this
> lane's reproducibility floor in the decode regime**, so a served decode
> difference at or below it is noise. The `0.012111` filed was above it.
>
> The rest of the branch is settled. The lane's accumulation order is worth
> **1.6e-07**, moving 0-0.10% of bf16 output words across the three served K;
> the fold this document names was worth **1.4e-03**, ten thousand times that;
> with it gone the lane's bf16 output matches `torch._scaled_mm`'s on
> **99.85-100.00%** of elements at all three served K -- which is the same
> fraction it matches ITSELF on. The correction is now green on a GPU -- **86 passed, 0
> failed** -- which it was not when this receipt was last written. Every other
> way the two served arms could have differed is closed by an artefact rather
> than an assumption (section 2b), and three fresh controls close the rest
> (section 6a): the untouched arm reproduces #102 to `KL >= 0.000000` at
> 100.00% top-1, and so does the prefill regime the fix does not touch, over
> 4088 positions.
>
> **What the CPU emulation got right, and where.** Section 3 priced the fold at
> `KL >= 0.007160` and called it a floor on the weight axis. As served, the fix
> moved arm A by `0.011845` -- a factor of 1.65 above that floor, in the
> direction section 3b predicted and within the 1.48x it measured for the
> weight operating point. The emulation was worth what it claimed to be worth.
> Section 3a still records a mechanism I proposed for the 68.9x, tested, and
> could not confirm; that paragraph stands withdrawn.


**What #110 asked.** Two serves of one checkpoint through one inode, differing
only in whether `prepare_fp8_gemv` could build, disagreed in the decode regime:
`KL >= 0.012111`, top-1 91.02%, over 256 M = 1 positions. Is that accumulation
order (document it and close), or a defect (fix it)?

**The answer, now served.** A defect, and it was worth about half of what was
measured. Removing it takes the arms from `KL >= 0.012111` / 91.02% to
`KL >= 0.005947` / 96.88% -- 2.04x on the bound, 2.88x on the top-1 flips
(8.98% of positions to 3.12%). The remaining `0.005947` is attributed in
section 6d, and it is not a difference between the arms: served a second time
with nothing changed, the fixed lane disagrees with itself by `0.005985`. So
the answer to #110 is *both* of the things it offered -- a defect on the M = 1
path, and accumulation order -- and the second is now a number a receipt can
cite rather than a supposition.

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

* **The accumulation order #110 asked about is 1.6e-07 in fp32** -- the lane
  run twice on identical input, its `atomicAdd` reduction (`window_gemv.cu:223`
  into the shared reduction buffer, `:309` into the fp32 output -- read off the
  source, not supposed) free to land in a
  different order. That is the magnitude the issue asked to have recorded
  somewhere a receipt can cite, and it is four orders below what was served.
  **The "and it changes no bf16 output word" that stood here was true of this
  run and is not true in general**: the served-shape sweep in section 6c ran
  the same control at K = 2048 and 3072 and reads **99.90-100.00%** identical,
  not 100.00%. An fp32 disagreement of 1.6e-07 straddles a bf16 rounding
  boundary about one time in a thousand, and a thousandth of the elements is
  not nothing -- section 6b is about exactly that.
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

## 4. What remained when this section was written -- kept, and marked

> **Section 6 is the newer text.** Items 1 and 2 below have since run and are
> answered there; item 3 has run and is section 6c. They are kept because the
> outcomes they name in advance are what makes section 6 a reading rather than
> a rationalisation -- item 1 named "about 0.005 ... the fold was most of it
> with a second term left" before the serve, and that is the outcome that
> landed. Items 4 and 5 still stand as written.

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
   **RAN. The second one: `KL >= 0.005947` at 96.88% top-1 (section 6).** The
   free control did read `0.000000` at 100.00%, and so did the prefill regime.
2. **RAN AND PASSED on attempt 2: 86 passed, 0 failed, 115.6 s.** The account
   below is of attempt 1 and is kept because the defect it found was real and
   was this branch's own. **The CUDA leg's first execution failed because the
   fix had never run on a GPU before that attempt.** Action `6c90ba1b` placed
   at 05:49 on 2026-09-04 and
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
   attempt 2**, which re-ran the gate against the correction and passed it
   **86/86**. A second, independent action (`eaedecfd`, `wf110_tests.sh`, the
   same two modules with nothing else in the process) reproduced that to the
   centisecond, so the green is not an artefact of the campaign's environment.

   The two assertions the review named as unverified **passed**, at all four M:
   `same >= 0.90` in
   `test_gemv_and_materialised_agree_within_fp32_summation_order` and
   `lost > 3 * kept` in
   `test_the_lane_multiplies_the_codes_and_scales_the_output_not_the_operand`
   are both among the 85. So the specific worry is answered, and the general
   one was righter than it was put: the problem was never two thresholds, it
   was that **no CUDA test had run against the fix before attempt 1**, and that
   first run found a defect. Attempt 2 and `eaedecfd` then verified the
   correction on a GPU, 86/86 both times.

   For completeness, on CPU on this tree:
   `pytest tests/test_serving_fp8_gemv.py tests/test_kernel_window_gemv.py` =
   **9 passed, 77 skipped**, and
   `test_a_code_times_a_per_token_scale_is_NOT_exact_in_bf16` is one of the 9:
   it pins the arithmetic the fix rests on -- one bf16 rounding, bounded by
   2^-8, rms 1.6e-03 -- on a box with no GPU, which nothing else here could do.

3. **RAN; section 6c has the numbers, and one of them corrects section 1.**
   **The kernel leg is measured at K = 1024 only, and the served K are three.**
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

### 4a. Why the serve had not run WHEN THIS WAS WRITTEN, from the ledger

> It has since run -- see section 6. This subsection is the contention account
> as of 05:55, and its structural point still stands: `6c90ba1b` waited about
> 22 minutes in `ready/` behind nineteen other actions on a two-token pool,
> because a box-local worktree pins its actions to one box
> (`RobTand/prismabuild#5`).

Every GPU leg went to the PrismaBuild pool. The precision leg (`084553c5`)
waited 35 minutes as the oldest of fifteen GPU-needing items, then ran. The
served campaign (`6c90ba1b`, gpu=1 mem_gb=24) reached a worker once, died in
its first seconds on a permission fault -- `ts83/ext-A` holds root-owned JIT
lock files written by the serve container, so a host-side extension load there
gets EACCES -- was fixed with its own writable `TORCH_EXTENSIONS_DIR`; at the
05:55 snapshot it was back in `ready/` and had not yet served.

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

---

## 5. The sign, at least, is the one the fold predicts

#110 reports the arms against the BF16 teacher as arm A `KL >= 0.436065` and
arm B `KL >= 0.432477`, and calls the direction surprising. Under this
mechanism it is the expected direction: before the fix arm A carried one
rounding arm B did not, so arm A should sit slightly further from the teacher.
At 256 positions that is a consistent sign and not evidence. **Section 6 turns
it into evidence:** re-served with the fix, arm A reads `0.432401` against the
teacher, landing on arm B's `0.432477` -- which itself reproduced to six
decimals -- so the gap the sign pointed at is gone, not merely the right shape.

---

## 6. The served re-run

Pool action `6c90ba1b`, attempt 2, claimed 06:11:53 on 2026-09-04 and run off
this tree at `548be80`; `done`, rc 0, 799 s. Same two arms as #102 --
`ts83/arm{A,B}`, one inode `11665664` at 846 726 118 bytes, re-checked by the
campaign at run start -- same corpus `076d33efc447`, same tokenizer
`76f13c8e6e55`, and the same container: the run resolved
`vllm/vllm-openai:latest` to `sha256:61fc8a896b0a...`, which is the digest #102
served on, so `latest` did not move under the comparison. Arm A's route trace
confirms the **fixed** lane is the one that ran -- `tessera_window_gemv::gemv`,
7196 launches at `M1` on each of the four served shapes
(`M1:N1024:K2048`, `M1:N1024:K3072`, `M1:N4096:K1024`, `M1:N6144:K1024`), with
`torch._scaled_mm` only at `M16`.

### 6a. The controls first, then the headline

Every row is the same estimator the served headline uses: `kl_tool`'s lumped
DPI lower bound over the top-1024 intersection.

| pair | regime | positions | KL >= | top-1 |
|---|---|---:|---:|---:|
| **control** -- new arm B against #102's arm B (code path untouched) | decode | 256 | **0.000000** | **100.00%** |
| **control** -- new arm A against #102's arm A, prefill (fix does not touch M = 512) | prefill | 4088 | **0.000000** | **100.00%** |
| **control** -- new arm A against new arm B, prefill (both `_materialised_path`) | prefill | 4088 | **0.000000** | **100.00%** |
| the size of the fix on the A side -- new arm A against #102's arm A | decode | 256 | 0.011845 | 91.80% |
| **#110 as filed** -- #102 arm B against #102 arm A | decode | 256 | 0.012111 | 91.02% |
| **#110 after the fix** -- new arm B against new arm A | decode | 256 | **0.005947** | **96.88%** |

Read them in that order:

* **The untouched arm reproduces itself exactly.** Arm B's code path
  (`tessera_prepared.decode()` then `torch._scaled_mm`) is not on the diff, and
  a serve a day after #102's reads `0.000000` at 100.00% against it. So the
  comparison is not measuring session drift -- which is what
  [[kl-cross-session-drift-is-zero-on-this-lane]] already said of this lane, now
  said again on the arm that matters here.
* **The regime the fix does not touch did not move**, over 4088 positions,
  either across runs or across arms. The fix is confined to the
  `M <= GEMV_MAX_M` branch by construction, and the serve agrees.
* **The fold was most of the disagreement and not all of it.** `0.012111` to
  `0.005947` is 2.04x on the bound; 8.98% of positions flipping top-1 to 3.12%
  is 2.88x on the flips. A second term of `0.005947` survives.

**And which arm was right, on the served metric rather than in fp64.** Against
the BF16 teacher, decode regime, 256 positions:

| arm | KL >= | top-1 |
|---|---:|---:|
| #110's arm A, folded | 0.436065 | 63.67% |
| #110's arm B, the fallback | 0.432477 | 62.11% |
| new arm B, the same fallback re-served | **0.432477** | 62.11% |
| **new arm A, fixed** | **0.432401** | 61.72% |

Arm B reproduces to six decimal places, and the fixed arm A lands within
`7.6e-05` of it. **Do not read that `7.6e-05` as resolution** -- section 6d
serves this same fixed lane a second time and it reads `0.434621`, so the
run-to-run spread of this column on the GEMV lane is about `2.2e-03`, thirty
times the gap just quoted. The honest statement is the coarser one: the fixed
arm A and arm B are indistinguishable here, where the folded arm A was not.
#110 called the direction of that pair surprising; under the fold it is the
expected direction, and removing the fold removes the distinguishable gap. The fp64 leg
had already said the fallback was the accurate arm (section 1); the serve now
says the same thing on the metric that decides.

### 6b. What the residual `0.005947` most likely is -- and the run that settled it

> Section 6d ran that run. The hypothesis below was confirmed: read this
> section for the reasoning and the controls, and 6d for the verdict.

Section 6c ran the fp64-referenced kernel leg at all three served K. Two of its
columns are the point:

* the fixed lane against `torch._scaled_mm`: **99.85-100.00%** of bf16 output
  words bit-identical;
* the lane against **itself**, rerun on identical input, its `atomicAdd`
  reduction free to land in a different order: **99.90-100.00%** identical.

**Those are the same size -- and at `M = 1`, which is the only shape the decode
serve runs, they are the same numbers.** Restricting both columns to `M = 1`:

| K | fixed lane vs `_scaled_mm` | lane vs ITSELF |
|---:|---:|---:|
| 1024 | 100.0000% | 99.9023% |
| 2048 | 99.9023% | 99.9023% |
| 3072 | 100.0000% | 100.0000% |

At two of the three served K the lane differs from *itself* by at least as much
as it differs from the fallback, and at the third they are equal to the fourth
decimal. After the fix, the fixed lane differs from the fallback by no more than
it differs from itself on a second run. If that is
the whole story, then the residual `0.005947` is not a difference between the
two arms at all -- it is the lane's own run-to-run nondeterminism, which an
arm-vs-arm compare cannot separate from an arm-vs-arm difference, because #102
and this campaign each served each arm exactly once.

**And the control for that hypothesis already exists, which is the part worth
saying carefully.** The kernel-level self-test above -- the same tensor, the
same process, the kernel called twice -- is NOT the right control for a served
replicate, because a serve adds vLLM's own nondeterminism on top: scheduler
batching, KV-cache block allocation order, the autotuner, the sampler path. The
sharper control is **arm B served twice**, and section 6a already has it: arm B
is `torch._scaled_mm`, and a serve a day after #102's reproduces it at
`KL >= 0.000000` / 100.00% over the same 256 decode positions. So on this lane,
in this regime, everything outside the GEMM reproduces exactly. That is what
makes a nonzero arm-A-against-arm-A replicate attributable to the lane rather
than to the serve around it -- and it is why the replicate is worth a GPU at
all.

**This is a hypothesis with a number attached, not a conclusion, and it is
falsifiable in one serve** -- and section 6d served it, and it was outcome one:
serve arm A a second time, unchanged, and compare
the two decode dumps. `experiments/ts110_replicate_armA.sh` does exactly that
and names its two outcomes in its own header -- about `0.006` and the residual
is the lane's own nondeterminism (which answers #110's item 2 with a magnitude:
*this* is what "bit-exact on the decoded tile" does not buy you on the GEMM);
about `0.000` and the lane is reproducible across serves, so the residual is a
real arm-vs-arm difference and #110 stays open on it.

Two things NOT claimed here. First, a scaling argument: the residual per-element
perturbation is roughly 8x smaller than the fold's, and a quadratic law would
put its KL ~60x below the fold's rather than at half of it -- but this harness's
own measured `rel` scaling is nearly flat (section 3a's unexplained OFF-row
reads 1.17x for a 2x change), so neither the quadratic estimate nor its failure
is evidence. Second, that `0.005947` is small: it is not, it is half of what
#110 filed, and until the replicate runs the honest statement is that the fix
removed a defect worth about half the disagreement and the rest is unattributed.

### 6c. The kernel leg at the served shapes

`experiments/gemv_a_side_precision.py` at `ROWS=1024`, `COLS` in
{1024, 2048, 3072} -- the K of Qwen3-0.6B's served Linears, read off the route
trace rather than guessed from `intermediate_size` (which is an N).
Artefacts `/home/rob/tessera-runs/ts110/precision_k{1024,2048,3072}.json`. M = 1:

| K | lane as fixed, fp32 vs fp64 | lane with the fold, fp32 | lane vs ITSELF, fp32 | bf16 identical to fallback: folded -> fixed |
|---:|---:|---:|---:|---:|
| 1024 | 1.397e-07 | 1.402e-03 | 1.640e-07 | 62.60% -> **100.00%** |
| 2048 | 1.321e-07 | 1.754e-03 | 1.697e-07 | 52.44% -> **99.90%** |
| 3072 | 1.481e-07 | 1.762e-03 | 1.815e-07 | 57.42% -> **100.00%** |

### 6d. The replicate: the residual is the lane disagreeing with itself

Pool action `6921120b`, `done`, run off this tree at `c6c9d99` on 2026-09-04.
`experiments/ts110_replicate_armA.sh` serves **arm A a second time, changing
nothing** -- the same worktree, the same `ts83/armA` bytes through inode
`11665664`, the same corpus `076d33efc447`, the same tokenizer, and the same
`vllm/vllm-openai@sha256:61fc8a896b0a...` -- and compares the two decode dumps
against each other. The route trace confirms the fixed lane ran again:
`tessera_window_gemv::gemv`, 7168 launches at `M1` on each of the four served
shapes, `torch._scaled_mm` at `M512`.

| pair | regime | positions | KL >= | p99 | max | top-1 |
|---|---|---:|---:|---:|---:|---:|
| **#110 after the fix** -- arm A (lane) vs arm B (fallback) | decode | 256 | 0.005947 | 0.042089 | 0.175416 | 96.88% |
| **the replicate** -- arm A vs arm A, second serve, nothing changed | decode | 256 | **0.005985** | 0.070210 | 0.088356 | **95.31%** |
| arm A vs arm B | prefill | 4088 | 0.000000 | 0.000000 | 0.000000 | 100.00% |
| arm A vs arm A, second serve | prefill | 4088 | **0.000000** | 0.000000 | 0.000000 | **100.00%** |

**This is outcome 1 of the two section 6b named in advance.** The fixed lane
disagrees with *itself*, across two serves that differ in nothing, by
`0.005985` -- the same size as, and very slightly larger than, the `0.005947`
it disagrees with `torch._scaled_mm` by. The residual is therefore **not a
difference between the two arms.** It is this kernel's own run-to-run spread,
and after the fold is removed the two arms agree to within it.

Three things make that attribution rather than a coincidence of magnitudes:

* **The decode regime itself reproduces exactly on the other arm** -- and this
  is the sharp control, sharper than the prefill null below it. Arm B is served
  in the *same* regime: the same decode attention kernel, the same scheduler,
  the same KV-block allocation, the same sampler, the same 256 M = 1 scored
  forwards. It differs from arm A in exactly one thing, the GEMM. Section 6a's
  first row serves arm B a second time a day after #102's and reads `0.000000`
  at 100.00%. So the decode pipeline minus this kernel is exact across serves,
  and an arm-A-against-arm-A spread has one place left to live.
* **The materialised path is exact across serves too.** Prefill -- where the
  GEMV never executes on a scored forward -- reads `0.000000` at 100.00% over
  4088 positions both across arms *and* across the two serves. This is a
  weaker control than the one above (it is a different regime, M = 512, a
  different attention kernel, one large forward rather than 256 small ones),
  but it is a second independent null and it agrees.
* **The compile cache was cold and it did not matter.** The replicate built a
  fresh `vllm-cache-ts110r-armA`, as arm B's cross-run control did before it --
  and arm B's control read `0.000000`. An empty inductor cache is therefore not
  a source of divergence here, which is what [[tessera-serves-itself]] found
  when it chased the same question.
* **It was predicted from fp64 before it was served.** Section 6c measured the
  lane against itself at `1.6e-07` relative, moving 0-0.10% of bf16 output
  words at `M = 1`; the arm-vs-arm identity at `M = 1` is the same 99.90-100.00%.
  The serve priced those moved words and got the same number twice.

**The magnitude #110 item 2 asked for, stated so a receipt can cite it:** on
this lane, in the decode regime, **`KL >= ~0.006` at `~95%` top-1 is the
reproducibility floor.** A served decode difference at or below that is noise.
Above it -- as `0.012111` was -- something real is happening, and in #110's case
it was the `a_scale` fold.

**And the answer to #110 item 3, which offered two possibilities:** it was
*both*. Half the filed disagreement was a kernel defect on the M=1 path (the
fold, fixed here); the other half is accumulation order, now measured on the
serving metric instead of assumed. The issue's own framing -- "consistent with
different accumulation order rather than a defect" -- was half right, and only
a serve could say which half.

**Teacher-KL readings across the three serves of these arms**, which is the
same fact seen from the other side:

| serve | KL >= vs BF16 | top-1 |
|---|---:|---:|
| arm A, folded (#102 and this campaign's baseline) | 0.436065 | 63.67% |
| arm B, fallback | 0.432477 | 62.11% |
| arm A, fixed -- first serve | 0.432401 | 61.72% |
| arm A, fixed -- second serve, nothing changed | 0.434621 | 63.28% |

The two identical serves of the fixed lane differ by `2.2e-03` on this column,
which is why section 6a's `7.6e-05` is corrected there rather than left to
imply a resolution this lane does not have. The fold's `3.7e-03` displacement
is above that spread; the fixed arm's distance from arm B is not.

**Scope, stated because it bounds every number above.** 256 M=1 decode
positions, one 0.6B checkpoint, one box (sparky), two serves. The floor is a
floor *measured twice*, not a distribution: a third serve could widen it. What
it already supports is the negative claim, which is the one #110 needs -- there
is no arm-vs-arm difference left that this instrument can see.

The fixed lane's fp32 error is flat in K at ~1.4e-07 and the fold's is flat at
~1.6e-03, so section 1's ratio holds at every served shape rather than at the
one it was measured on. The identity fractions across all four M and all three
K run **99.85-100.00%** for the fixed lane against the fallback and
**99.90-100.00%** for the lane against itself -- which is the observation
section 6b turns on, and the reason section 1's "changes no bf16 output word"
had to be corrected: at K = 1024 alone it read 100.00% four times out of four,
and a wider sweep says that was luck, not a property.
