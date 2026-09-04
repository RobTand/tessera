# The window GEMV folded the activation scale into bf16 -- and that is not big enough to be #110 (2026-09-04)

> **STATUS: the arithmetic is settled and one real defect is fixed; the served
> number is NOT explained and #110 stays open.** The in-process leg ran and is
> unambiguous: the lane's accumulation order is worth **1.6e-07** and does not
> move a single bf16 output word, the fold this document names was worth
> **1.4e-03**, and with it gone the lane's bf16 output matches
> `torch._scaled_mm`'s on **99.90-100%** of elements. But a calibrated
> propagation screen puts the fold at **KL 0.94e-04 on the teacher and 1.45e-04
> at an operating point worse than the served arms**, one part in eighty-four to
> one in a hundred and thirty of the `0.012111` #110 measured, and the sharper reading is worse than that:
> arms agreeing on 99.9% of their bf16 words cannot disagree by 0.012, so if
> the served re-run still reads 0.012 the cause is not this Linear's
> arithmetic at all. **The served re-run did not happen** -- sparky's two GPU
> slots were held for the whole session, latterly by one exclusive `gpu=2`
> action from another worktree, and the campaign is committed and queued
> (`6c90ba1b`). Nothing below is a placeholder for a number that was taken and
> disliked.

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

**#110 measured `KL >= 0.012111` at 91.02% top-1. The fold reads 0.000094 on
the teacher and 0.000145 at an operating point worse than the served arms --
between one part in eighty-four and one part in a hundred and thirty. The size
that reproduces the measurement is ~1.5e-02, nine times the fold in amplitude,
and at the degraded operating point it lands on KL 0.012323 against the
measured 0.012111.** Degradation is worth 1.5x, not 100x. So what #110 saw has
the size and the shape of a broad-band per-Linear difference of about one and a
half percent. The fold is real; it is not that.

**This is a screen, not a result.** It uses a noise model of the term, on a
model that is not the served one, at prefill positions. It is offered as a
magnitude argument and as a reason not to close #110, never as the lane's
number.

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
2. **If it does not collapse, look outside the GEMM.** A read-only
   `TORCH_EXTENSIONS_DIR` refuses *every* torch JIT build in that container,
   not only this lane's, and nothing in #102 establishes that the window GEMV
   was the only extension affected. That is the next hypothesis and it is
   cheap: census both arms for which extensions resolved, not only for which
   symbol the Tessera route stamped.
3. **More positions** (#110's item 1) is a corpus question, not a budget one:
   the decode stride is pinned to the serve's KV block size and the contract is
   8 x 512, so more positions means a new corpus contract and a re-dumped
   decode teacher.

Both GPU legs went to the PrismaBuild pool. The precision leg (`084553c5`)
waited 35 minutes as the oldest of fifteen GPU-needing items, then ran. The
served campaign (`6c90ba1b`) reached a worker once, died in its first seconds
on an unrelated permission fault -- `ts83/ext-A` holds root-owned JIT lock files
written by the serve container, so a host-side extension load there gets EACCES
-- was fixed, and was `ready` again, behind an exclusive `gpu=2` action from
another worktree, when the session ended. It has never served. A CPU-only
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
