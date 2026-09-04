# The window GEMV folded the activation scale into bf16 -- and that is not big enough to be #110 (2026-09-04)

> **STATUS: the arithmetic is settled and one real defect is fixed; the served
> number is NOT explained and #110 stays open.** The in-process leg ran and is
> unambiguous: the lane's accumulation order is worth **1.6e-07** and does not
> move a single bf16 output word, the fold this document names was worth
> **1.4e-03**, and with it gone the lane's bf16 output matches
> `torch._scaled_mm`'s on **99.90-100%** of elements. Every other way the two
> served arms could have differed is closed by an artefact rather than an
> assumption (§2b): same inode for the weights, prefill KL exactly `0.000000`
> over 4088 positions, and serve logs identical but for the intended 112
> refusals -- same attention backend, autotuner `Saved 0 configs` in both, the
> same single JIT compile in both. That leaves the `M <= 8` branch as the only
> place a difference can live, and inside it the fold is the only term above
> 1.6e-07. **Against that, a screen on the served model at the served position
> set puts the fold at KL 3.2e-04 / 98.44% top-1, against the measured 0.012111
> / 91.02% -- a factor of ~40.** One of those two readings is wrong and this
> session cannot say which: the screen still models a correlated bf16 rounding
> as independent Gaussian noise, and the code reading is only as good as the
> code I found. **The served re-run, which decides it in one measurement, did
> not happen** -- both of sparky's GPU tokens were held all session, latterly
> by an action from another worktree that had already *failed* and was still
> sitting in `claimed/` while the GPU idled at 13.89 W. The campaign is
> committed and queued (`6c90ba1b`). Nothing below is a placeholder for a
> number that was taken and disliked.

**What #110 asked.** Two serves of one checkpoint through one inode, differing
only in whether `prepare_fp8_gemv` could build, disagreed in the decode regime:
`KL >= 0.012111`, top-1 91.02%, over 256 M = 1 positions. Is that accumulation
order (document it and close), or a defect (fix it)?

**The answer.** Neither cleanly. There *is* a defect, it is not accumulation
order, and on the evidence available tonight it is not big enough to be the
whole of what was measured.

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
and at the output of a K = 1024 dot product that costs **1.612e-03**. Half a
bf16 ulp, for scale, is 1.953e-03.

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

## 3. What that term is worth as a KL -- and why it is not enough

A per-Linear relative error is not a KL. `experiments/gemv_a_side_propagation.py`
turns one into the other: it runs Qwen3-0.6B twice over the same corpus chunks,
once with independent Gaussian noise of a chosen relative rms injected at the
**input** of all 196 served Linears -- the fold's own injection point, since
what it rounds is every activation element -- and reports full-vocab KL and
top-1 agreement over 1022 positions. `--where output` is available as a coarser
cross-check and reads about 1.5x higher; the input-side numbers are quoted
because they are the faithful ones.

The obvious objection is that a screen calibrated on the *teacher* says nothing
about arms that sit 0.436 nats from it, so every row is taken twice: on the
BF16 teacher, and after RTN'ing every target weight to 4 bits per output row --
an operating point **worse** than the served arms (KL 0.719061 from the
teacher).

| per-Linear rel rms | KL, BF16 teacher | top-1 | KL, int4 operating point | top-1 |
|---:|---:|---:|---:|---:|
| 0 (control) | **0.000000** | **100.00%** | | |
| **1.612e-03  (the fold)** | **0.000094** | 99.22% | **0.000145** | 99.12% |
| 5.0e-03 | 0.000899 | 98.04% | | |
| 1.5e-02 | 0.008032 | 95.21% | 0.012323 | 93.54% |

KL goes as the square of the perturbation, as it should (the output-side sweep
spans 1.6e-03 to 2.5e-02 and holds the square law across it). The control row
is the harness proving it reads exactly zero when nothing changes.

**The scored position set is itself part of the metric, and it costs a factor
of 3.4.** The served decode regime does not score every position: it scores
prefix lengths 1, 17, 33, ... 497 -- 32 per chunk, the stride pinned to the
serve's KV block size -- which deliberately over-weights short prefixes where
the next token is barely determined (`topk_coverage_min = 0.458`,
`teacher_tail_mass_max = 0.547` in `mutual_decode.json`). Re-scoring the same
two chunks, the same seed and the same perturbation on exactly that set
(`--stride 16`) moves the fold's reading up and the agreement down:

| position set | KL, teacher | top-1 | KL, int4 point | top-1 |
|---|---:|---:|---:|---:|
| all 511 positions | 0.000094 | 99.22% | 0.000145 | 99.12% |
| **the served set (stride 16)** | **0.000318** | **98.44%** | **0.000155** | **98.44%** |

That is a real amplifier and the earlier all-positions row understated the fold
by 3.4x on the teacher. It is not the missing factor: the served set still
reads 0.000318 against a measured 0.012111, and 98.44% top-1 against a measured
91.02%.

**#110 measured `KL >= 0.012111` at 91.02% top-1. On the served position set
the fold reads 0.000318 on the teacher and 0.000155 at an operating point worse
than the served arms -- one part in thirty-eight to one part in seventy-eight.
The size that reproduces the measurement is ~1.5e-02, nine times the fold in
amplitude.** Degradation is worth 1.5x and the position set 3.4x; neither is
the missing factor, and together they are not.

**This is a screen, not a result, and §2b says it is now the weaker of the two
arguments.** The model is the served one (Qwen3-0.6B, 28 layers; the served
checkpoint's 112 fused modules are these 196 unfused Linears) and the positions
are now the served ones, but the *term* is still a noise MODEL -- independent
Gaussians standing in for a deterministic, input-correlated bf16 rounding. Set
against it is a direct reading of the code and the artefacts: §2b closes every
path but one, and down that path the fold is the only term larger than 1.6e-07.

So the two lines of evidence disagree by a factor of ~40, and the honest
statement is that **one of them is wrong and this session cannot say which**:
either the Gaussian model understates how far a correlated per-element rounding
travels, or the branch carries a third difference that reading it has not
found. What must not be written down is that either one has been established.

There is a sharper argument that does not need the screen at all. After the fix
the two arms' bf16 outputs are **bit-identical on 99.90-100%** of elements per
Linear, and where they differ they differ by one ulp. Two forwards agreeing
that closely at every Linear cannot part by 0.012 nats at the logits. So the
served re-run has two outcomes and both are informative: it collapses to ~0,
and the fold *was* the served difference after all (the screen's calibration on
a proxy model being the thing that was wrong); or it does not, and whatever
separates #102's two arms **is not this Linear's arithmetic**.

## 4. What remains, and why it did not run

1. **The served re-measurement.** `TESSERA_KL_ARM_TAG=ts110
   TESSERA_KL_DUMP_PREFIX=qwen_ts110 experiments/decode_regime_campaign.sh
   arms` reruns #102's exact pair off this tree (~6 min per arm). Arm B's code
   path is untouched by the fix, so armB-new vs #102's armB is a free control
   that should read 0.000000.
2. **The obvious "look outside the GEMM" hypothesis is already dead.** The
   candidate was that a read-only `TORCH_EXTENSIONS_DIR` refuses *every* torch
   JIT build in that container, not only this lane's. It does not: §2b's log
   diff shows the one JIT compilation either serve performed
   (`_topk_log_softmax_kernel`) happened in **both** arms, and the attention
   backend and autotune outcome are identical. Recorded here because a
   falsified hypothesis is worth as much as an open one, and the next reader
   should not spend the GPU slot on it.
3. **More positions** (#110's item 1) is a corpus question, not a budget one:
   the decode stride is pinned to the serve's KV block size and the contract is
   8 x 512, so more positions means a new corpus contract and a re-dumped
   decode teacher.

Both GPU legs went to the PrismaBuild pool. The precision leg (`084553c5`)
waited 35 minutes as the oldest of fifteen GPU-needing items, then ran. The
served campaign (`6c90ba1b`) reached a worker once, died in its first seconds
on an unrelated permission fault -- `ts83/ext-A` holds root-owned JIT lock files
written by the serve container, so a host-side extension load there gets EACCES
-- was fixed, and was `ready` again when the session ended. It has never served.
For the last 40 minutes of the session it could not be: both of sparky's GPU
tokens (`reservations/sparky/held/66266919.../gpu-0000`, `gpu-0001`) were held
by an action from another worktree that had **already failed** -- its queue
record carries `detail.status: "failed"`, `returncode: 1` -- and that was still
sitting in `pb-queue/claimed/` rather than being moved to `failed/` and having
its tokens released. Zero GPU tokens were free while `nvidia-smi` read 13.89 W
and 0% on an idle GPU. That is the inverse of the failure mode the pool was
built for and it is reported, not worked around; releasing another agent's
reservation is not this session's to do. A CPU-only
submission from this
checkout was refused outright -- `no live worker can run this action`, because
sparky's current offer carries no `cpu` capacity at all -- which is why the
propagation screen ran locally on four threads rather than through the pool.

## 5. The sign, at least, is the one the fold predicts

#110 reports the arms against the BF16 teacher as arm A `KL >= 0.436065` and
arm B `KL >= 0.432477`, and calls the direction surprising. Under this
mechanism it is the expected direction: before the fix arm A carried one
rounding arm B did not, so arm A should sit slightly further from the teacher.
At 256 positions that is a consistent sign and not evidence.
