# The dense 4-bit gap, and the block that closes it (2026-09-03)

**Read this first: the number in issue #12's title is two revisions stale.**
The issue says Tessera W4A4 **0.640** against PrismaQuant NVFP4 GPTQ+JSO
**0.511**, a gap of 1.254x. That was the *weights-only* wire. LDLQ plus the
`h^1.0` row-scale refit landed on 2026-09-02 and took the same bytes to
**0.5310275686796917** served (`tessera-ldlq-lut-plane-served-2026-09-02.md`),
so the live gap this document is about is **1.0400549849009113x**, not
1.254x.

| what | served KL-vs-BF16 | teacher dump | wire bpp | resident bpp |
|---|---|---|---|---|
| Tessera E2M1x2 q896, weights-only | 0.640404 | -- | 4.0018 | 4.500 |
| Tessera + LDLQ 1.0/32 + refit `h^1.0`, 2026-09-02 artifact | 0.5310275686796917 | `qwen_rot_teacher_lina` | 4.0018 | 4.500 |
| PrismaQuant NVFP4 GPTQ+JSO | 0.5105764371970046 | `qwen_teacher_bf16_v028` | -- | 4.500 |

### Which control the served delta is taken against, and why not that one

The published 0.5310275686796917 is *not* the control for the block delta, for
a reason measured rather than assumed. Two facts, in tension:

* **The metric does not drift.** The 2026-09-02 artifact re-served on
  2026-09-04 scores 0.5310275686796917 -- every digit, on the same bytes
  (`ldlq-block-serve/kl_repro_ldlqH1.json`, whose
  `provenance.artifact_path` is `ldlq-lut/ldlqH1-stock-twin`, i.e. the old
  checkpoint re-loaded). The cross-session KL warning inherited from the
  PrismaQuant era does not apply to this lane.
* **The encoder does.** A *fresh* `b=32` export at `82cdf513` -- same recipe,
  same block, same 220,301,312 wire bytes -- differs from that artifact in
  **161 of 784** quantized tensors of the served NVFP4 twin
  (`ldlq-block-serve/bytecmp_vs_ldlqH1.txt`), with per-tensor differing
  fractions up to 0.15. The same drift is visible one level up, on the wire
  itself and not only on its materialisation: hashing the two checkpoints'
  raw safetensors byte ranges, **69 of 112 `wire_bytes` blobs differ**
  (270 of 339 tensors identical overall -- the identical remainder is the
  bf16 passthrough), while their `activation_aware` blocks are equal
  (`ts12_wire_0902_vs_82cdf51.json`). Same declared recipe, different bytes:
  the config field a merge guard compares cannot see this, which is the point
  of hashing rather than reading it.

So quoting the published number as the control would put a commit's worth of
encoder change inside a delta meant to carry only the LDLQ block. The control
is the bracket's **own** re-run of `b=32` at its own commit, served in the same
session as the candidates. The #60 driver brackets the sweep `b32 -> b8 -> b4
-> b32`, so there are two control serves; both are quoted separately below and
their spread is the session's own error bar.

The two ends were originally scored against *different* BF16 teacher dumps --
0.5310275686796917 against `qwen_rot_teacher_lina`, 0.5105764371970046 against
`qwen_teacher_bf16_v028` -- which is a confound whether or not it is a large
one, so it was measured rather than waved away. Both were re-run through
`kl_tool compare` on the same corpus (`corpus_qwen_n8_s512.json`, sha
`076d33ef...`, contract `cfbddc2c...`, 4088 scored positions, prefill regime):

| compare | KL >= | top-1 agree |
|---|---|---|
| `nvfp4-prod` against the bracket's teacher (`ts12_kl_nvfp4prod_vs_lina.json`) | 0.5105764371970046 | 62.573% |
| `qwen_teacher_bf16_v028` against `qwen_rot_teacher_lina` (`ts12_kl_teacher_vs_teacher.json`) | **0.0** | **100.000%** |

The comparator re-scores to the same value to sixteen digits, because the two
teacher dumps are the same distribution position for position. The confound is
exactly zero and the ratio above is a one-teacher ratio.

### Pre-registered reading of the served candidate

Written before the candidate's KL existed, so the number cannot pick its own
verdict. Let `b32a`/`b32b` be the bracket's two control serves and `C` the
comparator on the bracket's teacher.

* candidate <= `C` -- **closed**: at equal residency the wire is at or ahead of
  the production NVFP4 encoder, and the gap is quoted as the ratio.
* `C` < candidate < min(`b32a`, `b32b`) -- **narrowed, still open**: the block
  is worth its measured amount but does not clear the comparator; the remaining
  levers and their cost are named.
* candidate >= min(`b32a`, `b32b`) -- **closed the other way**: the block's
  weight-space gain does not transfer to served KL, and the honest verdict is
  that this route is for residency, not quality, with the tercile table as the
  reason.

**The served candidate is pending at the time of writing** and is named
explicitly wherever it appears below rather than assumed. It is being produced
by `muse/ts-60-serve`, not rebuilt here; see "Arms that were not run".

## The byte accounting, shown rather than asserted

"Equal residency" on this pair is exact, not approximate. Both arms serve as
compressed-tensors NVFP4 on `FlashInferCutlassNvFp4LinearKernel`, and the file
the runtime loads is the same size to the byte:

| | bytes | bpp |
|---|---|---|
| comparator `fc45-0p6b-nvfp4/exported/model.safetensors` | **870,290,032** | 4.500 resident |
| Tessera b32 stock twin `model.safetensors` | **870,290,032** | 4.500 resident |
| Tessera wire (what the kernel lane would carry) | 220,301,312 | **4.001823** |
| Tessera on-disk wire + tables | 220,443,566 | 4.004407 |
| quantizable parameters, both arms | 440,401,920 | -- |

The wire column is not what these checkpoints hold; it is what the same units
cost on `tessera.kernel`, and the manifest states both. The comparison that
decides the issue is the resident one, where the two files are byte-for-byte
the same size and the quantised parameter count is identical. Tessera is
carrying 11% fewer bits of *information* into the same envelope, which is why
losing 1.040x there is a narrower loss than it looks.

## The lever, and why the floor mattered

Issue #60 measured that the LDLQ block size was never searched: the default 32
is a round number, and on dense Qwen attention it is a poor one. The closed
form `compensate.block_penalty(H_reg, block)` prices what a block costs against
full feedback, and it says why one constant cannot serve both populations --
at `b=32` it costs dense Qwen attention **14.7%** of full feedback (geomean
over 28 layers; the two-layer figure in `block_penalty`'s own docstring is
9.8%) and GLM experts **0.14%**, a factor of **70** on the 28-layer number.

`compensate.choose_ldl_block` resolves that from the Hessian, and tessera#95
fixed whose floor it is told. **But nothing called it.** Before this branch,
`choose_ldl_block` and `block_penalty` were reachable only from their own
tests: `ActivationSource.ldlq_block` was a bare `int`, so every unit in a
checkpoint got one constant and no export could price anything. The floor bug
#95 fixed was real and is fixed; the reason it had not bitten a shipped
artifact is that the code path it guards was not connected to one.

This branch connects it: `ActivationSource.ldlq_block` now takes either the
width it always took or a **budget**, `{"max_penalty": ratio}`, and derives
each unit's block from that unit's own Hessian at `floor=1`.
`experiments/dense4_block_budget_price.py` prices what that picks, over all 196
units of Qwen3-0.6B against the capture the served arms were encoded with. No
bytes are written: this is the price tag, computed before an arm is built.

### What a constant costs each role (`block_penalty`, geomean over 28 layers)

| role | cols | b2 | b4 | b8 | b16 | **b32** | b64 |
|---|---|---|---|---|---|---|---|
| `q_proj` | 1024 | 1.00465 | 1.00957 | 1.02934 | 1.06139 | **1.14719** | 1.18402 |
| `k_proj` | 1024 | 1.00465 | 1.00957 | 1.02934 | 1.06139 | **1.14719** | 1.18402 |
| `v_proj` | 1024 | 1.00465 | 1.00957 | 1.02934 | 1.06139 | **1.14719** | 1.18402 |
| `gate_proj` | 1024 | 1.00187 | 1.00519 | 1.02830 | 1.03880 | 1.11719 | 1.13947 |
| `up_proj` | 1024 | 1.00187 | 1.00519 | 1.02830 | 1.03880 | 1.11719 | 1.13947 |
| `down_proj` | 3072 | 1.00084 | 1.00199 | 1.00559 | 1.00975 | 1.03047 | 1.04980 |
| `o_proj` | 2048 | 1.00044 | 1.00124 | 1.00290 | 1.00655 | **1.01447** | 1.03098 |

The served default of 32 charges `q`/`k`/`v` **14.7%** of full feedback and
`o_proj` **1.4%** -- a factor of ten *inside one model*, before GLM is
mentioned. `q`, `k` and `v` are identical to five digits because they share a
Hessian, which is the predictor's blind spot stated as a number.

### What each policy costs, in the currency the encode tracks

Encode time is proportional to segments (`cols / block`, summed over units);
286,720 input columns, so the segment counts are exact geometry, not a fit.

| policy | segments | x b32 | blocks it picks |
|---|---|---|---|
| uniform b2 | 143,360 | 16.000 | 2 |
| uniform b4 | 71,680 | 8.000 | 4 |
| uniform b8 | 35,840 | 4.000 | 8 |
| uniform b16 | 17,920 | 2.000 | 16 |
| **uniform b32 (served)** | **8,960** | **1.000** | 32 |
| budget 1.01 | 56,608 | 6.318 | 1 ... 32 |
| **budget 1.02** | **30,776** | **3.435** | 2 ... 128 |
| budget 1.05 | 18,548 | 2.070 | 4 ... 512 |
| budget 1.10 | 10,826 | 1.208 | 4 ... 1024 |

### Where a budget spends it (budget 1.02, units per block)

| role | blocks chosen |
|---|---|
| `q_proj` | b2 x1, **b4 x11, b8 x16** |
| `k_proj` | b2 x1, **b4 x11, b8 x16** |
| `v_proj` | b2 x1, b4 x11, b8 x16 |
| `gate_proj` | b2 x1, b4 x8, b8 x9, b16 x10 |
| `up_proj` | b2 x1, b4 x8, b8 x9, b16 x10 |
| `down_proj` | b4 x1, b8 x4, b16 x2, b32 x4, **b64 x16**, b128 x1 |
| `o_proj` | b16 x6, **b32 x19**, b64 x3 |

This is the shape the two populations ask for, and it is the argument for a
budget over another constant: at 3.435x the b32 encode it gives `q`/`k` the b4
or b8 that the measured sweep says pays, while giving `down_proj` **b64** and
`o_proj` **b32** -- for `down_proj` that is *coarser than today's default*,
which a constant cannot be. Uniform b4 buys the same attention blocks at
**8x**, and spends 5 of every 8 of those segments on units whose own Hessian
says the feedback was not being skipped.

And the same budget crosses the model boundary the right way. On GLM-5.3
experts, `block_penalty` reads 1.00065 at b16, 1.00137 at b32, 1.00611 at b128
and 1.01295 at b256 (`ldlq-block/block_predictor.json`, L5 and L20 gate/up), so
budget 1.02 gives them **b256 or coarser** -- eight times *cheaper* than the
constant they run today, for a predicted 1.3% of full feedback which the
measured sweep put at 0.51%. One budget, one export, b4 on dense attention and
b256 on MoE experts. That is the thing a global flip to 4 cannot do, and it is
the reason this is wired as a budget rather than a new number.

**What the price table does not buy: affordability.** Every column of it is
expensive because the per-segment cost is launch-bound -- see the filed
principle-7 finding below. A budget makes small blocks *cheaper than a uniform
small block*; it does not make them cheap.

## The screens: what the block is worth in weight space

Two censuses, both at 4.00 bpp, both with the comparator arm joined
bit-identically (`A-arm mismatches: 0` over the shared units), from
tessera#60's data on `/mnt/shared/tessera-runs/ldlq-block/`.

### Below 16, on the six H-matched attention units (`dense4_below16.json`)

Held-out output-space error, `out`; the b32 control was run first and last and
is identical on all six units (`drift_same: true`), and every arm is at the
same bpp to seven digits.

| arm | vs b32 (geomean, n=6) | vs NVFP4 GPTQ+JSO (geomean, n=6) |
|---|---|---|
| b32 (the served default) | 1.0000 | **1.0421** |
| b16 | 0.9569 | 0.9972 |
| b8 | 0.9373 | 0.9768 |
| b4 | 0.9240 | **0.9629** |
| **b2** | **0.9196** | **0.9583** |
| b1 | 0.9223 | 0.9611 |

`b2` is an interior optimum and `b1` -- full sequential compensation, the
comparator's own granularity -- is **worse**. Every one of the winning blocks
is below 16, which is the floor `choose_ldl_block` would have inherited from
the stitching path had #95 not fixed it.

### Model-wide at b8, 49 units, 7 roles x 7 layers (`census_block_*.json`)

`C/A` is Tessera's H-weighted weight-leg error over the comparator's, so below
1.0 is a Tessera win. This is the table that matters, because it says the lever
lands hardest exactly where the issue said the gap lives.

| role | n | C/A at b32 | C/A at b8 | b8/b32 |
|---|---|---|---|---|
| **`k_proj`** | 7 | **1.0858** | **0.9947** | **0.9161** |
| **`q_proj`** | 7 | **1.0748** | **0.9881** | **0.9193** |
| `gate_proj` | 7 | 0.9680 | 0.9038 | 0.9337 |
| `up_proj` | 7 | 0.9360 | 0.8919 | 0.9528 |
| `v_proj` | 7 | 0.9579 | 0.9191 | 0.9595 |
| `down_proj` | 7 | 0.9860 | 0.9716 | 0.9855 |
| `o_proj` | 7 | 0.8301 | 0.8162 | 0.9833 |
| **all** | **49** | **0.9736** | **0.9246** | **0.9496** |

Two readings:

* **The two roles the issue names are the two the block buys most.** `k_proj`
  0.9161 and `q_proj` 0.9193, against `o_proj` 0.9833 and `down_proj` 0.9855 --
  the roles that already win buy least. The lever is role-targeted by
  accident of the Hessian rather than by design, which is the ordering the
  mechanism receipt predicted from `comp/gptq`.
* **At b8 no role is above 1.0.** `k_proj` and `q_proj` cross from 1.0858 and
  1.0748 to 0.9947 and 0.9881. In weight space the dense-4 deficit is gone.

### Does the predictor that drives the budget agree with the measurement?

The budget mode stands or falls on `block_penalty` ranking units the way the
encoder does, so the two are joined on the same 49 units: predicted
`penalty(b8)/penalty(b32)` against measured `C/A(b8) / C/A(b32)`.

| role | n | predicted b8/b32 | measured b8/b32 | predicted - measured |
|---|---|---|---|---|
| `k_proj` | 7 | 0.8909 | 0.9161 | -0.0252 |
| **`v_proj`** | 7 | **0.8915** | **0.9595** | **-0.0680** |
| `q_proj` | 7 | 0.8982 | 0.9193 | -0.0211 |
| `up_proj` | 7 | 0.9149 | 0.9528 | -0.0380 |
| `gate_proj` | 7 | 0.9152 | 0.9337 | -0.0185 |
| `down_proj` | 7 | 0.9487 | 0.9855 | -0.0367 |
| `o_proj` | 7 | 0.9904 | 0.9833 | +0.0071 |
| all | 49 | 0.9208 | 0.9496 | -- |

Per-unit Spearman **+0.637**. The predictor gets the material separation right
-- attention and `gate`/`up` buy 7-10%, `o_proj` and `down_proj` buy 1-2% --
and is uniformly optimistic about the size, which is the constant-`||eta||`
assumption its own docstring names.

**Its largest single error is `v_proj`, and it is the blind spot, not noise.**
Predicted 0.8915 against measured 0.9595: the predictor ranks `v_proj` with
`q`/`k` because it *is* `q`/`k` as far as `H` is concerned, and the encoder
does not. Quantified here at 6.8 points on the census, it is the same fact the
mechanism receipt measured as 5.5% against 2.0%. A budget therefore buys
`v_proj` a block it does not need, and the honest description of the mode is
that it spends encode time where feedback is being skipped -- not where it
pays most.

**This is a screen and the issue is not closed by it.** The prediction it
licenses, written down before the serve so it cannot be fitted afterwards: if
the weight-leg ratio carried proportionally, `1.040 x 0.9496 = 0.988` -- the
4.0-bpp wire would edge past NVFP4 GPTQ+JSO at equal residency. The bridge from
a weight-leg census to a served scalar held to half a percent once
(1.035x census against 1.040x served, `tessera-dense4-residual-mechanism`),
which is one agreement, not a derivation.

## The served leg

**Not yet landed at the time of this commit.** The bracket
`b32 -> b8 -> b4 -> b32` is being encoded by `muse/ts-60-serve` at `82cdf513`;
at the last read `b8` stood at `[120/196]` after 15,380 s and `b4` at
`[60/196]` after 14,226 s, so the first control-plus-candidate pair is hours
out and the second is most of a day out. This section is filled from
`/mnt/shared/tessera-runs/ldlq-block-serve/kl_*.json` when they land, read
against the zones pre-registered at the top of this document.

What is settled without them, and does not change when they arrive:

* the gap the issue is about is **1.0400549849009113x**, not the 1.254x in
  its title; both ends are quoted at full precision from their own compare
  files, and the two teacher dumps behind them were scored against each other
  and agree exactly (KL 0.0, top-1 100%), so it is a one-teacher ratio;
* the control for the delta is the bracket's own `b32`, not the published
  0.5310275686796917, because a fresh export of the same recipe at the same
  block moves 161 of 784 tensors;
* the lever is priced across 196 units and two models, and the mechanism that
  spends it is wired, tested and byte-neutral at a stated block.


## Arms that were not run, and why

* **A duplicate b4/b8 export.** The served arms this issue needs were already
  encoding on `muse/ts-60-serve` when this branch opened -- b32 landed at
  23:00 UTC, b4 and b8 were mid-flight -- and re-running either would have cost
  a second 4-15 hour core for a number already being produced. The bytes are
  read, not rebuilt.

  What that costs in scope is measured, not waved at. The bracket was encoded
  at `82cdf513`, and the *preceding* range moved bytes -- the same recipe at
  the same block differs from the 2026-09-02 artifact in 161 of 784 twin
  tensors and 69 of 112 wire blobs -- so the forward range was not assumed to
  be different. A fresh `b=32` export at this checkout, with the served arm's
  verbatim arguments, reproduces that arm's layer-0 wire **exactly**: 4 of 4
  `wire_bytes` blobs and all 123 shared tensors identical, `activation_aware`
  equal (`ts12_byte_identity.json`). So `82cdf513..HEAD` -- the intervening
  merges *and* this branch's own commit together -- moves no bytes on the
  units it was checked on, and the bracket's absolute numbers, not only its
  delta, are readable at this commit. The check's own limit: `--layers 1`, so
  it covers 4 of 112 modules; it is layer 0's wire, not the whole
  checkpoint's. The
  bracket is in any case read as an *internally matched* pair -- control and
  candidates from one tree, one session, one teacher, so the delta between them
  is a fact about the block whatever the surrounding commits did.
* **Rotation.** Off the lever list by decision: 2.5-2.8x worse served on this
  model (`rotation-hurts-block-scaled-formats`).
* **A learned codebook.** Kernel-lane only under the FP4-native constraint.
* **The reach-aware per-row start**, which took the 8-bit wire from 0.470 to
  0.151. `encode_unit` calls `initial_channel_scale` only when the plane is
  `CHANNEL`, and `wire_recipe(E2M1x2, 896)` returns `LUT`. It cannot apply
  here, and this is the third document to say so.

## The ruling on the floor, and the byte-identity proof it owes

The change is a **caller-side** one: `choose_ldl_block` is told `floor=1` from
`ActivationSource.block_for` (the `encode_unit(ldl=...)` path's floor, per
tessera#95), a budget is opt-in, and `DEFAULT_LDLQ_BLOCK` is untouched. **No
schema minor bump**: the wire is unchanged, the default is unchanged, and
`activation_aware.ldlq_block` keeps its old type and value whenever a block is
stated. What that owes, per the acceptance criteria, is a proof that unchanged
arguments move no bytes -- not a reading of the diff.

Two exports settle it, both with the served arm's verbatim arguments
(`--grid E2M1x2 --q256 896 --ldlq-sigma 1.0 --ldlq-block 32
--refit-metric h^1.0 --layers 1`) and both hashed on the raw safetensors byte
ranges:

| pair | tensors compared | `wire_bytes` blobs | differing | `activation_aware` | verdict |
|---|---|---|---|---|---|
| `HEAD~1` vs `HEAD` -- this commit alone (`ts12_byte_identity_pair.json`) | 312 | 4 | **0** | equal | **BYTE-IDENTICAL** |
| `HEAD` vs the bracket's arm at `82cdf513` (`ts12_byte_identity.json`) | 123 shared | 4 | **0** | equal | **BYTE-IDENTICAL** |

The first isolates the commit: `HEAD~1..HEAD` is exactly this change, so a
byte-identical pair across it is the claim, not an argument for it. The second
is wider than needed and is the more useful of the two: it says the whole range
`82cdf513..HEAD` -- three merges into `encode.py`, `window_viterbi.py`,
`compensate.py` and `scale_channel.py`, plus this commit -- reproduces the
served arm's layer-0 wire exactly.

Both are `--layers 1`, i.e. 4 of 112 modules. That is the honest limit: seven
units settle a per-unit byte claim in minutes where the whole model costs
hours, and the *preceding* range shows what a real encoder change looks like
here (69 of 112 wire blobs moving), so this is not a check with no power to
fail. It is stated as layer 0's wire, not as the checkpoint's.

## The tests, and what each of them would fail without

`tests/test_ldlq_block_budget.py`, 20 cases, all on the PrismaBuild pool
(`20 passed in 0.83s`, sparky). They split into two halves on purpose.

**Four are regression guards** and pass on the pre-change tree as well -- that
is what they are for: `the_default_is_still_the_measured_constant`,
`a_stated_block_writes_the_int_it_always_wrote`,
`for_unit_at_a_stated_block_is_bit_identical_to_the_old_expression`, and
`a_stated_block_below_one_is_still_refused`. A guard that went red on the old
tree would be describing a behaviour change the branch is claiming not to make.

**The other sixteen fail there**, measured, not predicted: `16 failed, 4
passed in 0.79s` with the test file run out of the pre-change checkout
(`ts12-pre`, whose `export.py:276` still reads `ldlq_block: int =
DEFAULT_LDLQ_BLOCK`). The failure modes, tallied from that run:

| failure | count | what it says |
|---|---|---|
| `TypeError: '<' not supported between instances of 'dict' and 'int'` | 14 | the old `if self.ldlq_block < 1` meets a budget spec |
| `AttributeError: 'ActivationSource' object has no attribute 'block_for'` | 1 | the derivation does not exist there |
| `Failed: DID NOT RAISE tessera.errors.GrammarError` | 1 | `ldlq_block=1.02` is **silently accepted** by the old field |

Two of these are worth more than a count. The refusal tests fail on the *type
error* rather than on their expected `GrammarError`, which is a real failure
and not the one their name implies -- worth saying rather than reporting a bare
"would fail". And the single `DID NOT RAISE` is a small find of its own: on the
pre-change tree a float `ldlq_block=1.02` passes construction, because
`1.02 < 1` is `False`, and would reach the encoder as a block width. The new
validation refuses it by name.

One pre-registered expectation was wrong and is corrected rather than quietly
dropped: the split was predicted 5/15, on the guess that
`the_segment_count_a_budget_implies_is_what_the_encode_time_tracks` was pure
`block_penalty` arithmetic. It is not -- it builds a budget-bearing
`ActivationSource` first, so it fails on the old tree too. The split is 4/16.

The counterfactual run itself is recorded below under its own heading, because
the obvious way to take it does not work in this suite.

**No regressions in the neighbourhood.** The ten test files that touch the
changed code -- `test_compensate`, `test_export`,
`test_export_ignore_completeness`, `test_export_moe_layouts`,
`test_export_plan_cache`, `test_ldlq_block_budget`, `test_ldlq_lut_plane`,
`test_ldlq_window`, `test_merge_guard`, `test_serving_export_gate` -- run
**181 passed, 1 skipped** on the branch (sparky, 173.77 s).

## What remains unmeasured

* **GLM.** Every number above is dense Qwen3-0.6B. The E2M1 route's wins are
  on GLM's experts and must not be paid for out of them; the promotion gate for
  any default change to the LUT plane includes a six-expert geomean no worse
  than 1.00x. The budget mode is designed so that gate is easy to pass -- on
  GLM's Hessians the same budget picks a coarse block -- but "designed to" is
  not "measured to", and no GLM arm was run on this branch.
* **The derived arm itself has no served number.** What is served below is a
  *uniform* small block. The derived mode's value over a uniform one is encode
  time, not quality, and that claim rests on the price table, not on a serve.
* **`block_penalty` cannot see role.** `q_proj`, `k_proj` and `v_proj` of one
  layer have bit-identical Hessians, and the block is worth 5.5% on q/k against
  2.0% on v. A budget provisions all three alike. It spends encode time where
  feedback is being *skipped*, which is not the same claim as where it pays.
* **The encode cost of a small block is a principle-7 problem, and it is not
  fixed here.** See below.

## Filed, not fixed: the LDLQ segment cost is launch-bound

The LDLQ pass costs a roughly fixed amount per *segment* (`cols / block`),
nearly independent of the width inside it -- 0.694 s/segment measured on the
CHANNEL plane, and Qwen3-0.6B's b32 export took 12,871 s for 8,960 segments.
That is why a global b4 is a ~15-hour encode on a 0.6 B model, and why "flip
the default to 4" is the wrong shape of fix regardless of what it buys. A
per-segment cost invariant to the work inside the segment is the signature of a
launch-bound loop, which principle 7 calls a bug rather than a price. Fixing it
would make the whole axis cheap and would change what a budget can afford, so
it is worth an issue of its own; it needs before-and-after profiles from both
instruments and is out of scope here.

## Filed, not fixed: a `PYTHONPATH` counterfactual does not bind in this suite

The obvious way to show a test would fail without the change is to run it with
`PYTHONPATH` pointed at the pre-change source. Done that way here it reported
**20 passed** -- against a tree in which the field under test is still a bare
`int`. `tests/conftest.py` opens with
`sys.path.insert(0, Path(__file__).resolve().parents[1] / "src")`, so the
checkout that owns the *test file* wins position 0 and the environment variable
is silently ignored (the venv's editable `.pth` is a plain path entry and loses
to both). A counterfactual has to move the test file into the old checkout, not
the old source onto the path.

Recorded because the failure mode is silent and green: it produces a passing
run that looks like evidence for the opposite of what it tested. The
counterfactual quoted above is the one run with the test file under
`ts12-pre/tests/`, where `parents[1]/src` resolves to the pre-change tree.

## Provenance

| what | where |
|---|---|
| the 2026-09-02 artifact's served KL, bytes and corpus | `docs/measurements/tessera-ldlq-lut-plane-served-2026-09-02.md` |
| that artifact re-served, and the fresh export's 161/784 byte difference | `/mnt/shared/tessera-runs/ldlq-block-serve/kl_repro_ldlqH1.json`, `bytecmp_vs_ldlqH1.txt` |
| the comparator at full precision, and its teacher | `/home/rob/tessera-runs/stock/kl_nvfp4-prod.json` |
| the mechanism, the role table and the `comp/gptq` ordering | `docs/measurements/tessera-dense4-residual-mechanism-2026-09-03.md` |
| the below-16 sweep | `/mnt/shared/tessera-runs/ldlq-block/dense4_below16.json` (tessera#60) |
| the 49-unit b8/b32 censuses | `/mnt/shared/tessera-runs/ldlq-block/census_block_{8,32}.json` (tessera#60) |
| the served arms | `/mnt/shared/tessera-runs/ldlq-block-serve/` (`muse/ts-60-serve`, exported at `82cdf513`) |
| the derived-block price table | `experiments/dense4_block_budget_price.py` |
| the byte-identity check, and the two pairs it produced | `experiments/dense4_block_byte_identity.py`; `experiments/dense4_block_byte_identity_pair.sh`; results `/mnt/shared/tessera-runs/ldlq-block/ts12_byte_identity{,_pair}.json` |
| the wire drift between the 2026-09-02 artifact and `82cdf513` | `/mnt/shared/tessera-runs/ldlq-block/ts12_wire_0902_vs_82cdf51.json` (69 of 112 `wire_bytes` blobs) |
| the comparator re-scored on the bracket's teacher, and the teacher-vs-teacher zero | `experiments/dense4_comparator_one_teacher.sh`; `/mnt/shared/tessera-runs/ldlq-block/ts12_kl_{nvfp4prod_vs_lina,teacher_vs_teacher}.json` |
| the bracket reader that resolves the pre-registered zones | `experiments/dense4_read_bracket.py` |
| the pre-change counterfactual (`16 failed, 4 passed`) | test file run out of `ts12-pre`, pool action `80c1cd3fada2` on sparky |
| the impacted-file suite (`181 passed, 1 skipped`) | pool action `ae6350752e26` on sparky, 173.77 s |
| the Hessian | `/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt`, wikitext-2 **train**, 16,384 fit tokens, `text_sha256 a5c5fd09...`, `fit_ids_sha256 229c6f72...` |
| the KL corpus | `/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json` -- the **Qwen** contract, wikitext-2 **test**, disjoint from the capture; 8 x 512, 4,088 scored positions |
