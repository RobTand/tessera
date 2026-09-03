# The BF16 recipe's `(L, sigma)`, and where the 4-bit residual actually sits (2026-09-02)

Two issues, one session. **#18**: the BF16 route's `(L, sigma)` was stated
rather than searched, and its GLM expert evidence was one tensor. **#12**: the
dense 4-bit route loses to NVFP4 GPTQ+JSO at equal residency by 1.254x.

**Claim (#18, sigma).** `BF16_CHANNEL_SIGMA` is a **gauge, not a knob**. On four
dense Qwen Linears, dyadic shifts over a 16x range decode **bit-identically**;
the file hash moves and the tensor does not. There is no dyadic value left to
find. The identical shift on E4M3 costs +5 to +19% at x2 and +70 to +98% at x4.

**Claim (#18, L).** At 4 bpp the default `L = 14` is on both the `wt` and the
`h` frontier and nothing here argues with it. **At 8 bpp `wt` is the wrong gate
for L**: `wt` decreases monotonically in L on 4 of 4 units at both rungs, so a
`wt`-gated sweep always answers "deeper", while `h` is non-monotone on 4 of 4 at
R=8 and its optimum is unit-dependent (L = 8, 10, 16, 16 on the four). In the
geomean L=10 is 10.8% better on `h` than the default for 0.127 bpp *less* -- but
two of the four units keep L=14 and L=16 on their own `h` frontier, so that is a
geomean statement and not a verdict on the constant. Weight space, four dense
units, proposal only -- there is no BF16 lane to serve on.

**Claim (#18, reach).** The `window_sigma`/`channel_sigma` **ratio** is the real
knob (the constant alone is the gauge above), and **the right value is
rung-dependent**. At 4 bpp the default ratio is the `wt` optimum on 4 of 4
units. At 8 bpp it is not: `0.707` -- reach 5.66 row-RMS instead of 4.00 -- wins
on `wt` on 4 of 4 (geomean **0.813x at identical bytes**) and on `h` on 3 of 4
(geomean 0.742x). Spending it needs a wire change, because `BF16_RECIPE` leaves
`window_sigma=None` and so pins the ratio to 1 by construction.

> **Spent, 2026-09-03 (#48).** `BF16_RECIPE` now carries an explicit
> `window_sigma` and `wire_recipe` scales it per rung as `sqrt(max(R, 4)/4)`.
> The `0.813x` above reproduces through the built path to within the
> non-dyadic gauge tolerance (0.02%), the rule is the interior optimum on four
> rungs it was never fitted to, and below the reference it is floored because
> the law loses there. Not a wire, schema or contract change — the table is on
> the ALPHABET plane and a reader never rebuilds it. Still weight space:
> `docs/measurements/tessera-bf16-reach-recipe-2026-09-03.md`.

**Claim (#18, GLM).** The one-tensor result holds on six experts. BF16 at R=8
beats **EXL3 K=8** on the weight leg 6/6 (geomean `wt` **0.867x**) and is level
in out-space (**1.004x**), at 8.0352 bpp against 8.0117 -- nearly matched, not
matched. Against the E4M3 wire it is **4.23x** better for +0.0156 bpp.

**Claim (#18, transfer).** The dense L conflict **does not transfer to GLM
experts.** On two expert `gate_proj` tensors every L is on the `wt`, `h` *and*
`out` frontier at both rungs -- deeper is monotonically better on all three --
because on experts `h` tracks `wt` to four or five digits. Not because `H` is
isotropic there (its diagonal spans 3.9x and 21.7x); because the encoder's
per-column relative error is, which is what makes the two metrics the same
number. The L question is a dense-model question. At `L=16` the BF16 route also beats EXL3 K=8 in out-space
(0.902x / 0.922x at +1.5% bytes), so the "level in out-space" above is a
property of the `L=14` constant, not of the route.

**Claim (#12).** The 1.254x premise **does not reproduce**. It is **1.039x**
served on master, moved by the LDLQ-on-LUT work at byte-identical bytes -- and
*not* by the reach-aware per-row start, which is provably not on that path. The
residual is a **weight-leg** residual: on held-out rows the weight leg alone is
**1.035x**, against 1.039x served. And it **concentrates**: over all 196
Linears, Tessera at 4.0 bpp beats NVFP4 at 4.5 on `hq` on five of seven roles
and loses only `q_proj` (1.065) and `k_proj` (1.098) -- 1.080 and 1.114 once
each role carries the discount measured on its own role. LDLQ + the H-solved
refit is worth **1.30x** on the weight leg on every one of the seven.

**Retracted within this document:** an intermediate reading of the 4-bit census
said the residual had left the weight leg entirely. It had not; the census
metric was in-sample for one arm. See "The census metric was not a control".

Harnesses: `experiments/bf16_l_sigma_sweep.py`,
`experiments/bf16_route_weight_space.py`, `experiments/dense4_residual_census.py`,
`experiments/dense4_census_out_check.py`. Byte proof:
`audit_byte_baseline.py --diff` is `0 changed of 36` across the batch on the
unamended harness and `0 changed of 40` across each documentation commit.

## #18, part 1: sigma is a gauge, and the file hash was the wrong invariant

With `window_sigma` left at `None` the window table is built at
`sigma = channel_sigma` (`encode.py`: `table_sigma = window_sigma; if
table_sigma is None and scale_plane is CHANNEL: table_sigma = channel_sigma`),
so both ends of the scale move together. bf16 is closed under x2 and
nearest-value snapping commutes with x2, so a **dyadic** shift should leave the
decoded tensor alone.

The first draft of the harness hashed the artifact and would have reported that
prediction **wrong**. The shift is *written down*: the ALPHABET plane holds the
doubled table and the fp32 global halves. The bytes move; the tensor does not.
**The invariant is the tensor, not the file** -- and a byte-equality test is the
wrong test for a claim about meaning.

`model.layers.2.self_attn.k_proj`, `--stage gauge`, L=14, q1024:

| arm | bpp | wt | file sha | tensor sha |
|---|---|---|---|---|
| x0.25 / x0.5 / x1 / x2 / x4 (dyadic) | 4.265625 | 0.06975480 | *five distinct* | `280d3d566a73a203` |
| x0.75 / x1.5 / x3 (odd) | 4.265625 | 0.06975998 | *three distinct* | `09858e59e66db87b` |

Two orbits, one per odd part of the multiplier, 0.0074% apart in `wt`. The
same shape on all four units. Every arm's repeat control was byte-identical.

The contrast is what makes this a property of the **grid** and not of the
mechanism. Same L, same units, E4M3, `wt` relative to each unit's own x1:

| unit | x0.25 | x0.5 | **x1** | x1.5 | x2 | x3 | x4 |
|---|---|---|---|---|---|---|---|
| `2.mlp.down_proj` | 1.0000 | 1.0000 | **1.0000** | 1.0330 | 1.1896 | 1.5865 | 1.9817 |
| `2.self_attn.q_proj` | 1.0000 | 1.0000 | **1.0000** | 0.9748 | 1.0587 | 1.3712 | 1.7101 |
| `2.self_attn.k_proj` | 1.0000 | 1.0000 | **1.0000** | 0.9680 | 1.0499 | 1.3589 | 1.6950 |
| `14.mlp.gate_proj` | 1.0000 | 1.0000 | **1.0000** | 1.0032 | 1.1048 | 1.4366 | 1.7909 |

E4M3's table runs past 448 and the gauge breaks upward while staying exact
downward. The default sits on a **one-sided cliff** with all of its margin
below -- filed as issue #36, because that is the wire we ship and it was found
by accident.

## #18, part 1: L is the axis, and `wt` is the wrong gate for it

`--stage dense-l`, four dense Qwen Linears, `window_sigma` tracking
`channel_sigma`, geomean, every arm priced at the bytes it wrote:

| rung | L | bpp | wt | h |
|---|---|---|---|---|
| R=4 | 8 | 4.01514 | 0.08573 | 0.08990 |
| R=4 | 10 | 4.02148 | 0.07792 | 0.07846 |
| R=4 | 12 | 4.04688 | 0.07288 | 0.07554 |
| R=4 | **14 (default)** | 4.14844 | 0.06936 | 0.07120 |
| R=4 | 16 | 4.55469 | 0.06735 | 0.07038 |
| R=8 | 8 | 8.01514 | 0.00940 | 0.01071 |
| R=8 | **10** | 8.02148 | 0.00842 | **0.01059** |
| R=8 | 12 | 8.04688 | 0.00724 | 0.01144 |
| R=8 | **14 (default)** | 8.14844 | 0.00661 | 0.01173 |
| R=8 | 16 | 8.55469 | 0.00605 | 0.01163 |

Lower-left hull of (bpp, error) -- the matched-bytes reading:

```
frontier(wt) R1024 L=8,10,12,14,16   R2048 L=8,10,12,14,16
frontier(h)  R1024 L=8,10,12,14,16   R2048 L=8,10
```

**On `wt` every L is on the frontier at both rungs; on the *geomean* `h` the
R=8 frontier stops at L=10.** L=12, 14 and 16 are strictly dominated there --
more bytes *and* more H-weighted error. The default L=14 costs +0.127 bpp over
L=10 and is 10.8% worse on `h`.

**That is a geomean statement, and the four units disagree with it.** Per unit
at R=8, from the same `dense_l.json`:

| unit | best-`h` L | `h` frontier | best-`wt` L |
|---|---|---|---|
| `2.mlp.down_proj` | 16 | 8, 14, 16 | 16 |
| `2.self_attn.q_proj` | **8** | 8 | 16 |
| `2.self_attn.k_proj` | 16 | 8, 10, 14, 16 | 16 |
| `14.mlp.gate_proj` | 10 | 8, 10 | 16 |

Two of the four keep L=14 *and* L=16 on their own `h` frontier; one wants the
shallowest table in the sweep. So "L=14 is off the frontier at 8 bpp" is true of
the geomean and false on half the units, and this sweep does not have the
sample to say which L any individual unit should carry.

What *is* uniform across all four is the disagreement between the two axes:
**`wt` decreases monotonically in L on every unit at both rungs, and `h` is
non-monotone in L on every unit at R=8.** A sweep gated on `wt` therefore always
answers "deeper", at every unit, at every rung -- which is exactly what makes it
the wrong gate.

Which is the same inversion issue #18 already names: "an alphabet worth 4.3x on
plain error and nothing on the H-weighted columns is a reach problem, not an
alphabet one". Swept on L, the two axes disagree in sign at 8 bpp. `wt` improves
monotonically with L because a deeper table resolves the bulk more finely; `h`
gets worse because the extra depth buys reach the H-heavy columns do not want.

At 4 bpp there is no such conflict: L=16 minimises both metrics on three of the
four units, `down_proj` ties L=14 with L=16 on `h` to five digits, and the
default L=14 is on both frontiers everywhere.

**Proposal, not a flip.** This is weight space on four dense units. Principle 3
says a screen does not promote, and there is no BF16 lane to serve on, so
`BF16_WINDOW_BITS` stays at 14 and the finding is recorded. What it does say is
that whoever builds the BF16 lane should carry `L` as a per-rung -- and, on this
evidence, possibly per-unit -- choice rather than one constant, and should gate
it on `h` or a serve, **never on `wt`**, which is monotone and will always
answer "deeper".

## #18, part 1: reach is the mechanism, and the default is right at 4 bpp and wrong at 8

`--stage reach` pins the table (`window_sigma = 1.0`) and moves only
`channel_sigma`, so the ratio -- the table's spread against the row's -- is the
sole variable, and `channel_sigma` *is* that ratio for the duration of this
stage. Reach in row-RMS units is `4.0 / channel_sigma`. Four dense Qwen
Linears, both rungs, every arm at **identical bytes within its unit and rung**,
control repeated in place:

| rung | best `wt` | best `h` | 0.707 vs default, `wt` | 0.707 vs default, `h` |
|---|---|---|---|---|
| R=4 | **1.0 on 4 of 4** | 1.414, 1.0, 0.707, 1.0 | 1.0265 (worse) | 1.0272 (worse) |
| R=8 | **0.707 on 4 of 4** | 1.414, 0.5, 0.707, 0.5 | **0.8127** | **0.7416** |

(units in the per-unit columns are `2.mlp.down_proj`, `2.self_attn.q_proj`,
`2.self_attn.k_proj`, `14.mlp.gate_proj`, in that order; geomeans over the four.)

**At 4 bpp the default is the `wt` optimum on every unit**, and it is a genuine
interior optimum -- moving the ratio 2x either way costs 1.4% to 24%. The
single-unit table that started this section, `2.mlp.down_proj` at R=4:

| channel_sigma | bpp | wt | h | reach (row RMS) | rows over reach |
|---|---|---|---|---|---|
| 0.25 | 4.08854 | 0.15362 | 0.11940 | 16.00 | 0.4% |
| 0.5 | 4.08854 | 0.09108 | 0.07902 | 8.00 | 6.0% |
| 0.707 | 4.08854 | 0.07639 | 0.06630 | 5.66 | 14.3% |
| **1.0 (default)** | 4.08854 | **0.07373** | 0.05833 | 4.00 | 54.9% |
| 1.414 | 4.08854 | 0.07479 | **0.05754** | 2.83 | 100% |
| 2.0 | 4.08854 | 0.07479 | 0.05754 | 2.00 | 100% |

**At 8 bpp the default is off the optimum, uniformly and by a lot.** Every one
of the four units prefers `channel_sigma = 0.707` -- reach 5.66 row-RMS instead
of 4.00 -- on `wt`, by a geomean of **0.813x at identical bytes**, and three of
the four prefer it on `h` too (geomean 0.742x; per-unit 1.053, 0.714, 0.534,
0.753). Two of them want to go further still: `q_proj` and `gate_proj` both
minimise `h` at `channel_sigma = 0.5`, reach 8.0 row-RMS, with 0.0% and 0.3% of
rows clipped.

Which is the same shape as the L result and points the same way: **the right
reach is rung-dependent.** A 4-bit table has to buy resolution by giving up
range; an 8-bit table does not, and paying the 4-bit price at 8 bits costs about
19% of the weight error for nothing.

Three corrections to what a single unit suggested, kept here because the
single-unit reading was in an earlier draft of this document:

* **"`h` wants more clipping than `wt` does" is `down_proj`'s behaviour, not a
  rule.** `down_proj` wants more clipping at both rungs (best `h` at 1.414);
  `q_proj` and `gate_proj` want *much less* at R=8 (best `h` at 0.5, the least
  clipping in the sweep); `k_proj` sits between. The direction is unit-dependent
  and the two attention projections lean the opposite way from the MLP
  down-projection -- the same split that shows up in #12's per-role census.
* **"More than half the rows already exceed reach at the default" is
  `down_proj`'s 54.9%.** The other three are 29.4%, 23.4% and 26.0%. The
  qualitative point survives -- clipping a quarter to a half of the rows is the
  *good* setting at 4 bits, so "rows over reach" is a diagnostic and not a
  defect -- but the number is not a constant.
* **The 1.414 and 2.0 arms produce identical bytes on `down_proj` only.** There
  every row is clipped at both settings and the reach-aware start lands them in
  the same place; on the other three the shas differ, because 99.8-99.95% of
  rows are clipped at 1.414 and 100% at 2.0, and the handful of unclipped rows
  still move.

**And none of this is spendable through `BF16_CHANNEL_SIGMA`.** Part 1 of this
section is why: with `window_sigma=None` -- which is what `BF16_RECIPE` carries
(`DEFAULT_WINDOW_SIGMA`) -- the table is built at `sigma = channel_sigma`, both
ends move together, the ratio stays pinned at 1, and the constant is a gauge.
This stage could move the ratio only because it pinned `window_sigma = 1.0`
explicitly. So the finding is a **wire proposal, not a constant to retune**: to
spend it the BF16 recipe needs an explicit reach term -- a `window_sigma` per
rung, or equivalently a reach constant -- and `wire_recipe` would have to carry
it the way it already carries `window_bits`. Filed as
[#48](https://github.com/RobTand/tessera/issues/48), with the per-rung `L`
question folded into it. Nothing changes here: weight space, four dense units,
no BF16 lane to serve on, principle 3.

## #18, part 2: six GLM experts, R=8

The abandoned run, re-run after the R=8 dispatch fix. The issue's cost estimate
held: **1442 s (E4M3) + 969 s (BF16) per tensor became 134-164 s and 134-151 s**,
so six tensors took ~28 minutes instead of ~4 hours. Geomean:

| arm | bpp | wt | h | out |
|---|---|---|---|---|
| EXL3 K=8 | 8.0117 | 0.00608 | 0.00608 | 0.00475 |
| **bf16 q2048** | **8.0352** | **0.00527** | **0.00527** | **0.00477** |
| e4m3 q2048 | 8.0195 | 0.02231 | 0.02231 | 0.02017 |
| FP8 RTN LS-refit | 8.0078 | 0.02080 | 0.02080 | 0.01881 |

* **The bytes are nearly matched, not matched**: +0.0234 bpp for the
  2^14 x 2-byte table amortised over a 2048x4096 expert, +0.29%. On a
  1024x1024 dense Linear the same table is 0.25 bpp and the comparison would
  not be fair.
* **BF16 wins the weight leg 6/6** by 10-15% (`wt` 0.849-0.895x, geomean
  0.867x). The variance is EXL3's: BF16 sits in 0.00523-0.00531 on every
  tensor while K8 ranges 0.00588-0.00623.
* **Out-space is level**: 1.004x geomean, 0.980-1.031x per tensor, three each
  way.
* **Against the 8-bit alphabets it is not close**: 4.23x better than the E4M3
  wire, 3.95x better than FP8 RTN LS-refit.

The alphabet-floor claim now rests on six experts rather than one. Still a
weight-space screen; there is no BF16 lane and nothing here is promotable.

## #18, part 1: the L conflict does not transfer to GLM experts

`--stage glm-l` ran the same L sweep on two GLM-5.3 expert `gate_proj`
tensors (2048x4096, layers 5 and 42), scored on 1024 **held-out** activation
rows, with EXL3 K=4/6/8 reconstructions in the same table as the reference.
Nothing here is fit to `H` -- no LDLQ, no refit -- so `wt`, `h` and `out` are
all out-of-sample and the arms are directly comparable.

| unit | arm | bpp | wt | h | out |
|---|---|---|---|---|---|
| L5 | EXL3 K=4 | 4.0117 | 0.08816 | 0.08814 | 0.07237 |
| L5 | R=4 L=8 | 4.0044 | 0.08663 | 0.08663 | 0.08364 |
| L5 | R=4 **L=14 (default)** | 4.0352 | 0.06942 | 0.06941 | 0.06693 |
| L5 | R=4 L=16 | 4.1289 | **0.06754** | **0.06754** | **0.06500** |
| L5 | EXL3 K=8 | 8.0117 | 0.00623 | 0.00623 | 0.00511 |
| L5 | R=8 **L=14 (default)** | 8.0352 | 0.00531 | 0.00531 | 0.00512 |
| L5 | R=8 L=16 | 8.1289 | **0.00478** | **0.00478** | **0.00461** |
| L42 | EXL3 K=8 | 8.0117 | 0.00600 | 0.00600 | 0.00359 |
| L42 | R=8 **L=14 (default)** | 8.0352 | 0.00527 | 0.00527 | 0.00370 |
| L42 | R=8 L=16 | 8.1289 | **0.00473** | **0.00473** | **0.00331** |

**Every L is on the `wt`, `h` *and* `out` frontier, on both units, at both
rungs.** Error falls monotonically with L on all three metrics. The dense-Qwen
conflict -- `wt` monotone, `h` non-monotone, the two axes disagreeing in sign at
R=8 -- **does not appear here at all**.

The mechanism is visible in the table: on GLM experts `h` and `wt` agree to four
or five significant digits (0.07786 vs 0.07786, 0.00531 vs 0.00531). **It is not
that the Hessian is isotropic** -- measured on the same 1024 eval rows the
diagonal `H_ii` spans **3.86x** max/min on L5 and **21.70x** on L42
(std/mean 0.181 and 0.377), which is not isotropic by any reading. With a
diagonal `H`,

```
h^2 / wt^2 = [ sum_i H_ii e_i / sum_i H_ii w_i ] / [ sum_i e_i / sum_i w_i ]
```

is exactly 1 whenever the per-column relative error `e_i / w_i` is uniform,
*however much `H` varies*. So the term that has to be near-constant is the
encoder's per-column relative error -- which is what a column-blind CHANNEL
plane produces on weights with no outlier columns, and precisely what dense Qwen
breaks (the reach-aware start exists because a quarter of Qwen's rows exceed the
window table's reach). **So the L question is a dense-model question** -- the
dense result cannot be transferred to experts, and the expert result cannot be
transferred to dense.

Two things fall out that are worth recording even though nothing is promoted:

* **L=14 is not the frontier point on GLM experts either**, but in the opposite
  direction from dense: `L=16` is better on all three metrics at both rungs, for
  +0.094 bpp. On dense Qwen at R=8 the geomean wanted *shallower*.
* **At L=16 the BF16 route beats EXL3 K=8 in out-space**, 0.902x on L5 and
  0.922x on L42, at 8.129 bpp against 8.012 (+1.5% bytes). The six-expert result
  reported earlier in this document -- level in out-space, 1.004x -- was measured
  at the default L=14, and this says that "level" is a property of the constant
  rather than of the route.

Controls: 4 of 4 repeat arms byte-identical *and* tensor-identical, wall drift
0.91x to 1.15x between the first and repeated baseline.

## #12: the premise, and the mechanism it was attributed to

Two findings, and only the second one moves the number.

**The reach-aware per-row start cannot have moved this.** It lives in
`initial_channel_scale`, which `encode_unit` calls only inside
`if scale_plane is ScalePlaneKind.CHANNEL`. `wire_recipe(E2M1x2, 896)` is
`TCQ` over `LUT`. Monkeypatch the function to raise, then encode both wires:

```
E2M1x2 q896 (LUT plane): encoded 4.2500 bpp, initial_channel_scale calls = 0  -> reach fix NOT on this path
E4M3 q1024 (CHANNEL plane): raised as expected -> initial_channel_scale was called
```

**What did move it** is the LDLQ-on-LUT work
(`tessera-ldlq-lut-plane-served-2026-09-02.md`): served KL 0.640404 -> 0.531028
at **220,301,312 wire bytes either way**, so the gap to NVFP4 GPTQ+JSO at
4.5 bpp is **1.039x on 11% fewer bits**, not 1.254x. All three of the issue's
"still open" items are addressed: LDLQ landed, LDLQ *is* the GPTQ-equivalent
compensation, and the per-16 block plane now has its scales solved against `H`.

## The census metric was not a control

`dense4_residual_census.py` scores every arm on `sqrt(E H E^T / W H W^T)` with
the **fit-row** Hessian, and its early units read as a Tessera win on the
weight leg. That reading was wrong and the error is the one this project keeps
paying for: arm C's LDLQ factor and block-scale refit were built from that same
`H`, so C is scored **in-sample**, while production NVFP4 GPTQ+JSO was
calibrated on PrismaQuant's own activations elsewhere and is scored
**out-of-sample**. Two treatments, no control.

`dense4_census_out_check.py` measures the size of it on the six Linears that
carry **held-out** activation rows (`x_eval_qwen06b.pt`, the eval slice,
disjoint from the fit slice `H` was accumulated on; neither arm saw them):

| unit | C/A on `hq` | C/A on `out` | `out`/`hq` |
|---|---|---|---|
| `0.self_attn.q_proj` | 1.0251 | 1.0395 | **1.0140** |
| `1.self_attn.k_proj` | 1.1184 | 1.1346 | **1.0145** |
| `13.mlp.down_proj` | 0.9190 | 0.9750 | 1.0609 |
| `14.mlp.gate_proj` | 0.9633 | 0.9874 | 1.0251 |
| `2.mlp.down_proj` | 1.2858 | 1.2740 | 0.9908 |
| `27.self_attn.o_proj` | 0.7627 | 0.8511 | 1.1159 |
| **geomean** | **0.9992** | **1.0353** | **1.0361** |

**The census metric flatters the arm fit to it by 3.6% in the geomean**, on five
of six units, and the sixth agrees within 1%. Every `hq` number the census
produces has to carry that discount before it is read as a statement about
held-out behaviour. All six repeat controls were bit-exact.

**The discount is not uniform, and its spread is wider than itself: 0.99x to
1.12x.** That matters because it is not a constant you may multiply through a
per-role table -- `o_proj`, whose 0.763 is the largest win in the census, has
the largest discount (1.116) and so wins by less than `hq` says; `q_proj` and
`k_proj`, which carry the loss, have the *smallest* (1.014) and so lose by
almost exactly what `hq` says. The per-role numbers below are therefore quoted
raw, with the discount given as a range and the two roles that matter given
their own measured value.

## What the residual is, once the metric is honest

| measurement | scope | Tessera / NVFP4 |
|---|---|---|
| served KL, whole model, 4.0018 vs 4.5 bpp | 196 Linears | **1.039x** |
| weight leg, held-out out-space, same wire | 6 Linears | **1.035x** |
| weight leg, fit-row H quadratic | 6 Linears | 0.999x (in-sample for C) |

The served gap and the out-of-sample weight-leg gap agree to within half a
percent. Six units against a whole-model KL is suggestive rather than proof,
but the reading is clear: **the shared A4 activation leg contributes essentially
nothing to the difference, and the residual is a weight-leg residual.** So the
weight leg is the right place to keep hunting, and #35 -- a Gauss-Seidel sweep
on the LUT-plane block-scale refit, the one untried optimiser -- is the right
next lever rather than a footnote.

Per-unit variance is the other half of the story: `C/A out` ranges **0.851 to
1.274** across six units. The 4-bit route wins `27.self_attn.o_proj` by 15% and
loses `2.mlp.down_proj` by 27%.

## And it does concentrate: the residual is q_proj and k_proj

`dense4_residual_census.py` across **all 196** Qwen3-0.6B Linears, three arms
per unit, one process, 5821 s. The weights-only arm was re-run on the first four
units at the end as the drift control and came back **bit-exact 4 of 4** across
that whole span:

| role | n | A `hq` (NVFP4 4.5) | C `hq` (Tessera 4.0) | B/A | **C/A** |
|---|---|---|---|---|---|
| `o_proj` | 28 | 0.07269 | 0.06121 | 1.3217 | **0.8421** |
| `down_proj` | 28 | 0.07604 | 0.06699 | 1.1755 | **0.8810** |
| `up_proj` | 28 | 0.07158 | 0.06736 | 1.2323 | **0.9411** |
| `v_proj` | 28 | 0.07626 | 0.07286 | 1.2178 | **0.9554** |
| `gate_proj` | 28 | 0.04640 | 0.04534 | 1.3862 | **0.9772** |
| `q_proj` | 28 | 0.04931 | 0.05250 | 1.3680 | **1.0648** |
| `k_proj` | 28 | 0.04945 | 0.05427 | 1.4045 | **1.0976** |
| **all** | 196 | | | **1.2981** | **0.9619** |

The partial cuts taken along the way are worth recording because they are the
reason not to quote a partial census: `C/A` read 0.9544 at n=51, 0.9754 at n=94
and 0.9619 at n=196 -- a 2% band, not converging monotonically. **The role
ordering was stable in all three cuts and `q_proj`/`k_proj` were the only losers
in all three**, which is the part that was worth reporting early; the geomean was
not.

Two readings.

**The residual is not spread; it is `q_proj` and `k_proj`.** On `hq` Tessera at
4.0 bpp beats production NVFP4 at 4.5 on five of seven roles by 2-16% and loses
the two attention projections by 6-10%. The in-sample discount that turns those
into held-out statements is a **range, 0.99x to 1.12x**, and multiplying the
geomean through would be the same borrowing error the census itself fell into,
so each role below carries its *own* measured discount where the out-check had a
unit of that role, and the geomean only where it did not:

| role | `hq` C/A | discount | out-space C/A | discount source |
|---|---|---|---|---|
| `o_proj` | 0.8421 | 1.1159 | **0.940** | own (n=1) |
| `down_proj` | 0.8810 | 1.0253 | **0.903** | own (n=2) |
| `up_proj` | 0.9411 | 1.0361 | **0.975** | geomean |
| `v_proj` | 0.9554 | 1.0361 | **0.990** | geomean |
| `gate_proj` | 0.9772 | 1.0251 | **1.002** | own (n=1) |
| `q_proj` | 1.0648 | 1.0140 | **1.080** | own (n=1) |
| `k_proj` | 1.0976 | 1.0145 | **1.114** | own (n=1) |
| **all** | 0.9619 | 1.0361 | **~0.997** | geomean |

**Two roles carry the loss.** Every other role wins, and `q`/`k` lose by 8% and
11% out-of-sample -- which is a smaller loss than the `hq` cut at n=51 suggested
(1.116 / 1.145 there, after a discount I had borrowed rather than measured).

The three routes to the same number do **not** agree as tightly as the n=94 cut
suggested, and that is worth stating rather than smoothing:

| route | scope | Tessera / NVFP4 |
|---|---|---|
| served KL, whole model | 196 Linears | **1.039x** |
| weight leg, direct held-out out-space | 6 Linears | **1.035x** |
| weight-leg census, `hq` x measured discount | 196 Linears | **~0.997x** |

The first two are direct measurements and they agree to half a percent. The
third is a 196-unit `hq` census multiplied by a discount estimated on six units,
which are not a random sample -- they are the units that happened to
have held-out activation rows cached -- so its 4% disagreement with the other
two is an artefact of the discount's own sampling error, not a third opinion.
**Read the residual as 1.03x-1.04x, weight-leg, concentrated in `q_proj` and
`k_proj`**, and treat the census as the *shape* of it rather than the size.

That shape is what a per-role decision exploits. `q_proj` and `k_proj` are fused
siblings on the serving side and must share a format, which makes them one
decision rather than two -- and PrismaQuant's allocator exists to make exactly
that decision. A uniform-format comparison is the wrong frame for a route whose
error is this role-dependent.

**LDLQ + the H-solved refit is worth 1.30x on the weight leg**, uniformly:
`B/A` 1.2981 against `C/A` 0.9619. The lever is not a tail effect -- it is 1.18x
to 1.40x on every one of the seven roles, and it is largest exactly where the
weights-only wire was furthest behind. It is also the steadiest number the
census produced: 1.3155 at n=51, 1.3403 at n=94, 1.2981 at n=196.

Caveats: `hq` (so read every number with the 0.99x-1.12x discount above, and
only `q_proj`/`k_proj` with a discount measured on their own role), and weight
leg only. Full JSON at
`/mnt/shared/tessera-runs/ldlq-lut/dense4_residual_census.json`.

## Method

Every stage runs its arms **back to back in one process on one box**, with the
default arm first and **repeated last**; a disagreement between the two
baselines is printed as `!! DIFFER` or `!! FAILED` rather than absorbed into a
ratio. Every arm is priced at the bytes it actually wrote (`ExportedUnit.bpp`),
never at its label; NVFP4 is priced from its own export -- packed nibbles, a
per-16 E4M3 block scale, an fp32 global -- and never with a plane borrowed from
Tessera. Comparing two treatments and calling one a control is the error that
cost this project 19.2x, and the repeat control is the cheap guard against it.

**Scope, stated so it is not borrowed.** All of #18 is weight space; there is no
BF16 lane and no served number. The dense L and reach results are four dense
Qwen Linears; the GLM result is six experts. The #12 census is the weight leg
only -- both arms deploy the same A4 leg -- and the served numbers it is
compared against are the gate.

## What a GPU run must still confirm

* Any change to `BF16_WINDOW_BITS`, or a per-rung `L`, or a per-rung reach
  ratio, needs a served A/B at matched bytes in **one** vLLM session against the
  current default -- KL drifts 4-8x across sessions.
* `--stage glm-l` answered the transfer question in weight space -- it does not
  transfer, `h` tracks `wt` to four digits on GLM experts -- but the *dense*
  result it fails to transfer is still unserved. Both need the same A/B.
* Whether the rung-dependent reach result survives more than four units, and
  whether the unit-dependent `h` direction tracks role the way #12's census
  does. Both of the reach findings that are uniform (the `wt` optimum at each
  rung) are uniform over n=4.
* Whether the in-sample discount measured on six units holds across all 196 --
  and in particular that the 1.4% measured on the two `q_proj`/`k_proj` units in
  that set is not an artefact of n=2. This is the one loose end the finished
  census did not close: it is why the discounted census (~0.997x) and the direct
  out-space measurement (1.035x) disagree by 4%.
* A `q_proj`/`k_proj`-only served A/B, since a per-role decision is what the
  concentration argues for and no served number yet separates the roles.
