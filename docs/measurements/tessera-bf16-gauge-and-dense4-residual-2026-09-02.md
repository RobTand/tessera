# The BF16 recipe's `(L, sigma)`, and where the 4-bit residual actually sits (2026-09-02)

Two issues, one session. **#18**: the BF16 route's `(L, sigma)` was stated
rather than searched, and its GLM expert evidence was one tensor. **#12**: the
dense 4-bit route loses to NVFP4 GPTQ+JSO at equal residency by 1.254x.

> **Read the banner before the claims.** The `#18` claims below were written
> from two **one-axis** sweeps -- one over `L`, one over the reach ratio -- and
> a one-axis sweep over `L` prices a wider window table at **nothing**. The
> joint grid at the end of this document byte-matches every cell, and three of
> the statements below do not survive it: the `L` claim at 4 bpp, the `transfer`
> claim's out-space comparison, and the "no such conflict at 4 bpp" line in the
> dense-`L` section. Each is marked in place. Where they disagree, **the
> byte-matched sections are the reading**; the claims here are kept because the
> shape of the mistake is the finding.

**Claim (#18, sigma).** `BF16_CHANNEL_SIGMA` is a **gauge, not a knob**. On four
dense Qwen Linears, dyadic shifts over a 16x range decode **bit-identically**;
the file hash moves and the tensor does not. There is no dyadic value left to
find. The identical shift on E4M3 costs +5 to +19% at x2 and +70 to +98% at x4.

**Claim (#18, L).** At 4 bpp the default `L = 14` is on both the `wt` and the
`h` frontier of *this* sweep -- **superseded**, and by the harder direction:
byte-matched, `L = 12` beats it on **8 of 8** dense units at R=4 (0.9333x `wt`,
0.8916x `h`) at bytes the two provably share, and `L = 16`, which this sweep
liked, is the worst of the three widths there. What follows is what an unpriced
`L` axis says. **At 8 bpp `wt` is the wrong gate
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

**Superseded, both halves.** "Every L is on all three frontiers, deeper is
monotonically better" and the `L=16`-vs-K8 out-space number are the same
unpriced comparison: the wider table is paid for with `+1.5%` bytes and nothing
is given back. Byte-matched on six experts, deeper is *not* free at R=4 --
`L=16 r=1` is 1.0105x `out` against the shipped pair at equal bytes, and the
shipped `(14, 1.0)` is the best of all twelve cells. The direction survives at
R=8, where `L=16 r=1.25` wins 6/6 at 0.9001x byte-matched; the `+1.5%`
comparison against EXL3 does not survive at all, and no matched
BF16-vs-K8 number at `L=16` exists yet.

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

**And that is the sweep's own artifact, not a property of 4 bpp.** Nothing on
this axis charges for the table, so "deeper is at least as good" is close to
tautological. Byte-matched on eight dense units (below), `L=16` is the **worst**
of the three widths at R=4 -- **0 of 8** wins and **1.1882x** on `h` -- and the
shipped `L=14` is beaten 8 of 8 by `L=12`. The conflict at 4 bpp is not absent
here; it is hidden by a free table.

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
## #18, part 1: the *pair*, byte-matched -- what two one-axis sweeps could not say

Everything above sweeps `L` at ratio 1 (`--stage dense-l`, `glm-l`) or the
ratio at `L=14` (`--stage reach`).  Neither says whether the two axes interact,
and both price a wider table by *slope* -- "L=16 costs +0.094 bpp, and the rate
axis buys 1.903x error per bpp, so compare against that" -- rather than by
building the arm that spends the same bytes.  `--stage pair-dense` /
`pair-glm` close both gaps:

* **The grid is joint.** L in {12, 14, 16} x ratio in {1.0, 1.25, sqrt(2), 1.75},
  every cell run, at three rungs (R = 4, 6, 8 bits/weight).
* **The axes are named in every row.**  Ratio 1.0 is `window_sigma=None` --
  the value the recipe stores, so the reference arm exercises the shipped path
  rather than a re-spelling of it.  Every other ratio is
  `window_sigma = ratio * channel_sigma` with `channel_sigma` pinned at the
  shipped 1.0: this is the **ratio** axis (reach), never the tracking axis, and
  it costs no bytes.
* **The reference is built, not priced.**  `q256` accepts any positive integer
  (`grammar.py`), so for each `L` the harness solves for the rung at which the
  shipped `(L=14, r=1)` recipe spends *exactly* the candidate's bytes, encodes
  that arm, and asserts `|bpp_candidate - bpp_ref| < 1e-9` before any ratio is
  formed.  A row that does not find an exact-bpp reference is dropped, loudly.
* **Per-unit rows and a win count, then the geomean.**  Never the geomean alone
  -- that is what #65 closed.
* **Drift control first and last** at every (unit, rung): the shipped pair is
  encoded again as the last arm and asserted byte-identical *and*
  tensor-identical to the first.

**Pre-registered before the numbers** (committed in `c35f56c`, ahead of any
GPU run):

* Gate: `h` on dense Qwen, `out` on GLM experts.  `wt` is **disqualified in
  advance** as a gate -- it is monotone in `L` by construction, so it cannot
  distinguish "deeper table helps" from "deeper table costs bytes".
* ADOPT-WORTHY = a strict majority of per-unit wins **and** a geomean below
  1.00 **and** the six-expert GLM cross-check no worse than 1.00x.  All three,
  or nothing is proposed.
* CONFIRMED = the shipped `(14, 1.0)` is on the frontier at matched bytes.
* Anything else is INCONCLUSIVE and is reported as such.
* **No default flips out of this stage regardless of the outcome.**  Both axes
  change bytes and `encoder_profile_id`, and there is no BF16 serving lane, so
  every number here is a weight-space (dense: H-weighted) screen, not a
  promotion.  House principle 3.

**Scope of every number in this section**: grid `bf16`, weight space (dense `h`
is the captured activation second moment, GLM `out` is held-out-row output
error), no LDLQ, `scale_refit=4`, ratio axis (not tracking).  Populations and
counts are stated per table.

### The six GLM experts, joint grid, byte-matched

`--stage pair-glm --layers 5 20 42 --projs gate_proj up_proj --experts 0
--rungs 1024 1536 2048`, 2048x4096 tensors, `out` scored on 1024 **held-out**
capture rows, ratio axis, weight space. `pair_glm.json`, encoded on sparklina.

Every ratio below is against the **byte-matched** shipped pair: for `L=12` the
reference is the shipped `(14, r=1)` recipe run at `R1018/R1530/R2042`, for
`L=16` at `R1048/R1560/R2072`, so the two arms weigh the same to 1e-9. All
**12 of 12** cells present at every (unit, rung); **18 of 18** repeat controls
byte- *and* tensor-identical.

| rung | arm | wins | win@1% | `wt` geo | `h` geo | `out` geo | bpp |
|---|---|---|---|---|---|---|---|
| R=4 | L=14 r=1 **(shipped)** | - | - | 1.0000 | 1.0000 | **1.0000** | 4.0352 |
| R=4 | L=16 r=1 | 1/6 | 0/6 | 1.0079 | 1.0081 | 1.0029 | 4.1289 |
| R=4 | L=12 r=1 | 0/6 | 0/6 | 1.0152 | 1.0149 | 1.0134 | 4.0117 |
| R=4 | L=14 r=1.25 | 0/6 | 0/6 | 1.0142 | 1.0141 | 1.0135 | 4.0352 |
| R=4 | L=14 r=1.75 | 0/6 | 0/6 | 1.1886 | 1.1883 | 1.1882 | 4.0352 |
| R=6 | L=14 r=1 **(shipped)** | - | - | 1.0000 | 1.0000 | 1.0000 | 6.0352 |
| R=6 | **L=16 r=1** | **6/6** | **6/6** | 0.9806 | 0.9806 | **0.9801** | 6.1289 |
| R=6 | L=16 r=1.25 | 6/6 | 6/6 | 0.9875 | 0.9873 | 0.9855 | 6.1289 |
| R=6 | L=14 r=1.25 | 5/6 | 1/6 | 0.9931 | 0.9929 | 0.9940 | 6.0352 |
| R=8 | L=14 r=1 **(shipped)** | - | - | 1.0000 | 1.0000 | 1.0000 | 8.0352 |
| R=8 | **L=16 r=1.25** | **6/6** | **6/6** | 0.9037 | 0.9033 | **0.9056** | 8.1289 |
| R=8 | L=16 r=1 | 6/6 | 6/6 | 0.9331 | 0.9330 | 0.9332 | 8.1289 |
| R=8 | L=14 r=1.25 | 6/6 | 6/6 | 0.9408 | 0.9404 | 0.9375 | 8.0352 |
| R=8 | L=16 r=1.41 | 6/6 | 6/6 | 0.9417 | 0.9413 | 0.9405 | 8.1289 |

Per-unit `out` for the two arms that matter, in file order (`L20.gate`,
`L20.up`, `L42.gate`, `L42.up`, `L5.gate`, `L5.up`):

* R=6 `L=16 r=1`: 0.982 0.975 0.980 0.984 0.979 0.981 -- six of six, spread 0.9%.
* R=8 `L=16 r=1.25`: 0.907 0.911 0.901 0.901 0.896 0.918 -- six of six, spread 2.2%.
* R=4 `L=16 r=1`: 1.007 1.002 1.011 0.990 1.005 1.003 -- one of six, and that
  one by 1.0%.

**The shipped `(14, 1.0)` is the best cell of twelve at R=4 and is not on the
frontier at R=6 or R=8.** At R=4 no alternative wins a majority of units on
any axis; at R=6 and R=8 `L=16` wins 6/6 with a winning geomean, byte-matched.

### The two axes are not separable, and the reason is in the encoder

`separable?` in the reader means: is the best ratio the same at every `L`, and
the best `L` the same at every ratio. R=4 yes/yes, **R=6 no/no** (best ratio is
1.25 at L=12 and L=14 but 1.0 at L=16), R=8 yes/yes. So separability is itself
rung-dependent, and a sweep of one axis at a fixed value of the other cannot
be trusted to have found the joint optimum -- which is what the two earlier
one-axis stages did.

The mechanism is not subtle once the diagnostic is printed. At ratio 1 the
table's reach is **not** a constant of the recipe; it moves with `L`:

| L | reach (row-RMS) | rows over reach (`L5.e0.gate_proj`) | same, across all six | bpp at R=4 |
|---|---|---|---|---|
| 12 | 3.672 | 0.868 | 0.680-0.868 | 4.0117 |
| **14** | **4.000** | **0.515** | **0.374-0.515** | **4.0352** |
| 16 | 4.312 | 0.146 | 0.125-0.157 | 4.1289 |

The third column is **one** unit (`L5.e0.gate_proj`, which happens to be the
highest of the six at `L=12` and is not at `L=16`); the fourth is the range over
all six, so the column cannot be read as a maximum.

A wider table has more quantiles, and its extreme quantiles sit further out.
So `L` is not a resolution knob with a reach knob beside it: **`L` buys
resolution *and* reach at once, and only `L` costs bytes.** Two axes that
move the same physical quantity cannot be searched one at a time.

That also names the one comparison this grid cannot make.  `L=16` at ratio 1
has reach 4.312, which is the reach `L=14` would have at ratio 1.078 -- a
value the grid does not contain (its ratios are 1.0, 1.25, sqrt(2), 1.75).
So every `L` comparison here moves resolution *and* reach together, and the
grid cannot say which of the two the winning cell is buying.  A matched-reach
arm (`L=16 r=1`, vs `L=14 r=1.078`) would separate them; it is a follow-up,
not a correction, because the shipped path is ratio 1 and the verdict below
is about the pair as it would actually be spelled.

The second thing the diagnostic says is that **rows-over-reach is not the
objective**. At the shipped pair, 37-52% of GLM expert rows exceed the window
table's reach; ratio 1.25 takes that to 0.2-0.9% -- and at R=4 it makes the
error *worse* on 6 of 6 units. Clipping half the rows is the right trade when
codes are scarce and the wrong one when they are not: the ratio's optimum
walks from 1.0 at R=4 to 1.25 at R=8. Any rule that minimised `over` would
have picked 1.75 everywhere, which is 19% worse at R=4 and 6% worse at R=8.

### Eight dense Qwen Linears, same grid -- and the opposite answer

`--stage pair-dense`, the same eight units the `dense-l` sweep used, gate `h`
(the captured activation second moment), same three rungs, same byte matching.
`pair_dense.json`, encoded on sparky through the PrismaBuild pool. **12 of 12**
cells at every (unit, rung); **24 of 24** controls byte- *and*
tensor-identical.

| rung | best pair at matched bytes | wins | `wt` geo | `h` geo | shipped `h` |
|---|---|---|---|---|---|
| R=4 | **L=12 r=1** | **8/8** (8 at >1%) | 0.9333 | **0.8916** | 1.0000 |
| R=6 | **L=12 r=1.25** | 7/8 | 0.9164 | **0.8879** | 1.0000 |
| R=8 | **L=12 r=1.41** | 5/8 | 0.8760 | **0.8395** | 1.0000 |

and `L=16`, which won 6/6 on the experts, is the *worst* width here: 0/8 wins
and **1.1882x** at R=4, 1.1720x at R=6.

**The two populations want opposite directions from the same default.** GLM
experts want a *deeper* table (L=16, 6/6, 0.98-0.91x); dense Qwen wants a
*shallower* one (L=12, 8/8, 0.89x). The shipped `L=14` is on neither
frontier except at R=4 on the experts, where it is exactly right.

Two reasons, both measurable and both in the tables above:

* **The table's byte price is a per-unit quantity, not a constant.** A
  2^L x 2-byte table amortised over a 2048x4096 expert costs 0.031 bpp at
  L=14; over the dense set it costs 0.089 bpp, and over a 1024x1024
  `k_proj` 0.25 bpp. Byte-matching prices that honestly for the first time
  -- the earlier stages compared `L=16` against the shipped rung and paid for
  the wider table with nothing.
* **Dense Qwen has the outlier rows the experts do not.** At the shipped pair,
  rows over reach span **0.175-0.844** across the eight dense units against
  **0.374-0.515** across the six experts, and the dense spread is the point:
  one recipe is serving units whose reach demand differs by 5x.

### The axis disqualified in advance was disqualified for the wrong reason

`wt` was ruled out as a gate before the run because it is monotone in `L` --
a wider table can only fit the weights better.  That is true *unmatched*.
**Byte matching retires the premise**, because at matched bytes the wider
table is paid for out of the rung.  All three figures below are at ratio 1,
L = 12 / 14 / 16:

* dense Qwen, R=4: `wt` **rises** with `L` (0.9333 / 1.0000 / 1.1724);
* six GLM expert-0 units, R=8: `wt` **falls** (1.1154 / 1.0000 / 0.9331);
* dense Qwen, R=8: `wt` is **non-monotone within one population**
  (1.0351 / 1.0000 / 1.0743).

So `wt` at matched bytes is a real axis rather than a foregone one.  It is
**not**, however, a second gate that happens to agree.  Stated exactly, over
the six (population, rung) points:

* **The gate's winning cell is also below 1.00 on `wt` wherever it is below
  1.00 on the gate** -- 5 of 6 points; the sixth is GLM R=4, where the winner
  *is* the reference (1.0000 on both).
* **The best `L` under `wt` is the gate's best `L` at 5 of 6 points.**  The
  exception is dense R=8: the gate `h` picks `L=12 r=sqrt(2)` (0.8395) and
  `wt` picks `L=14 r=sqrt(2)` (0.8557).  Both beat the shipped pair; they
  disagree about which width.
* **Cell by cell they do disagree in sign, and only on dense** -- one cell at
  R=4 (`L=14 r=1.25`: `wt` 0.9967, `h` 1.0279), one at R=6, four at R=8
  (e.g. `L=12 r=1`: `wt` 1.0351, `h` 0.9565).  On the GLM experts there is no
  sign disagreement at any rung, on any cell.

The verdict below therefore does not rest on the choice of gate -- ADOPT-WORTHY
fails under `wt` too, and for the same reason, that the two populations want
opposite `L` -- but "the metrics agree" would be too strong a way to say it.

### The reading, against the criteria registered before the run

* **ADOPT-WORTHY** required a strict majority of per-unit wins **and** a
  geomean below 1.00 **and** the six-expert GLM cross-check no worse than
  1.00x. **Nothing clears all three.** `L=12` clears the first two on dense
  (8/8, 0.8916x at R=4) and fails the cross-check (1.0134x on the experts).
  `L=16 r=1.25` clears the first two on the experts (6/6, 0.9056x at R=8) and
  fails on dense (0/8, 1.1882x at R=4). So, as pre-registered: **nothing is
  proposed, and the default does not move.**
* **CONFIRMED, and only here:** at R=4 on GLM experts the shipped
  `(L=14, ratio 1.0)` is the best of all twelve cells. That is the rung the
  4-bit wire ships at, and on that population the inherited pair is not merely
  defensible, it is optimal on the grid searched.
* **The rest is INCONCLUSIVE by design, not by accident.** Above R=4, and on
  dense at every rung, the default is beaten -- but every candidate that beats
  it changes bytes and therefore `encoder_profile_id`, there is no BF16
  serving lane, and these are weight-space and H-weighted screens. Under
  principle 3 that is a screen, not a result.

What the search does close is the framing in the issue. `(L, sigma)` is not
"two numbers nobody looked at": `sigma` is a gauge at the dyadic multipliers
(part 1, above), the ratio is a real axis that costs no bytes, `L` is a real
axis that does, and **the two are entangled through reach**, so the pair had
to be searched jointly and now has been. The finding is not that the constant
is wrong; it is that **a single constant is the wrong shape** -- the frontier
moves with the rung and with the population, and `_window_bits_for` already
has the rate in hand (`max(14, R)`) if Rob ever wants to spend it.

### What this did not measure

* **No served number.** Weight space and H-weighted columns only; there is no
  BF16 lane, so nothing here is promotable and nothing is proposed.
* **Three rungs, not five.** R = 4, 6, 8. R = 5 and 7 (q256 1280, 1792) are
  the obvious fill-in and would say whether the R=4 optimum walks or jumps.
* **The expert-index widening has finished, and it reproduces the expert-0
  pattern exactly.** `pair_glm_experts.json` (`--experts 1 2`, layers 5/20/42,
  `gate_proj`, R = 4 and 8) is **6 of 6** units, each with 12 of 12 cells at
  both rungs, and **12 of 12** controls byte- *and* tensor-identical. **R=4:
  the shipped `(14, 1.0)` is the best of all twelve cells, 0/6 wins for every
  other arm** (nearest is `L=16 r=1` at 1.0105x, then `L=12 r=1` at 1.0178x).
  **R=8: `L=16 r=1.25` wins 6/6 at 0.9001x** `out`, against 0.9056x on the six
  expert-0 units -- the same arm, the same margin to within 0.6%. So the R=4
  CONFIRMED reading stands on **twelve** expert units drawn from three layers
  and three expert indices, and the R=8 result on twelve as well. The re-read
  is
  `PYTHONPATH=experiments python experiments/pair_report.py /mnt/shared/tessera-runs/reach/pair_glm_experts.json`.
* **The dense pair artifact is being overwritten as this lands, and that is a
  tooling defect, not a doubt about the numbers.** `pair_dense.json` is a
  single fixed path with no per-run identity, so a re-run of the same sweep
  truncates the completed file it is reproducing: at the time of writing it
  holds 4 of its 8 requested units, mid-re-derivation, with its `args` still
  naming all eight. The eight-unit numbers above were read off the complete
  file; a reader arriving now would see a file that looks finished and is not.
  `pair_report.py` now checks the unit set against `args` and refuses rather
  than summarising, which is the half of this that belongs in the repo -- the
  other half is a per-action result path, which is PrismaBuild's.
* **Wall times in these runs are not a measurement.** Three sweeps shared two
  boxes with a dozen other agents' jobs; the controls are byte- and
  tensor-identity, which is unaffected, and the seconds column is not a claim.

## #18, part 1: the bundle split -- entries and reach, priced apart

The joint grid above ends by naming the comparison it cannot make.  At ratio
1 a wider window table has more entries **and** its outermost entry sits
further out, so `L=16 r=1` is not "the shipped recipe with more resolution":
it is the shipped recipe with more resolution *and* a reach of 4.3125 where
the shipped one reaches 4.0.  Every `L` number above prices that bundle.  It
matters which half is being bought, because the two halves have different
prices: **entries cost bytes** (the table is `2^L x 2` on the ALPHABET plane,
and the sweep byte-matches it against the shipped pair built at the rung that
spends the same bytes), while **reach costs nothing at all** -- a spread is a
constant in the recipe and the artifact is the same size at every value of
it.

`experiments/matched_reach.py` builds the arms that split them.  For a target
reach it searches for the spread at which a given width realises *exactly*
that reach, and asserts the realised value before returning it.  A search,
not a division: a table's entries are grid values, so its reach is the
**snapped** outermost quantile and therefore a step function of the spread.
On the shipped BF16 grid the naive `target / own_reach` lands on the wrong
step -- at `L=14`, asking for `L=12`'s reach of 3.671875 by that route
delivers 3.6875 -- and a receipt computed that way would report a 0.4% miss
as an exact match.  The three widths' own reaches, from the code:

| L | own reach (row-RMS at the shipped `channel_sigma = 1.0`) |
|---|---|
| 12 | 3.671875 |
| 14 | **4.0** |
| 16 | 4.3125 |

The rows of `experiments/bf16_matched_reach_run.sh` each encode one width at
all three of those reaches, which turns the one-dimensional `L` axis into a
factorial whose **rows are entry counts** (byte-matched by the sweep) and
whose **columns are realised reaches** (free).  The `L=14` row is the cheap
and decisive one: it is the shipped table at another width's reach, at zero
byte cost and no change to the table's size.

**The reading was registered in the run script's header before any number
existed.**  For a (population, rung), `A` is the landed byte-matched effect
of the L-arm on its gate and `B` the effect of the `L=14` arm at that width's
reach; `recovered = log B / log A`.  At or above 0.5 the majority of the L
win is reach and is free; at or below 0.15 it is entry count and costs bytes;
between, both halves are real.  A *negative* fraction is a different fact and
is reported as one rather than as a band: the two axes moved the gate in
opposite directions, so no part of one is recoverable from the other.  Both
orientations of that occur -- an L win the spread alone reverses
(`B > 1 > A`), and an L arm that hurts while the spread alone helps
(`B < 1 < A`, which the landed dense `L=16` arms at 1.1655x-1.1882x can
produce).

**Where that reading applies was registered with it.**  `recovered` is a
ratio of two logarithms, so wherever `A` sits near 1 it is dominated by its
own third digit: the landed GLM `R=4` arms are 1.0029x and 1.0134x, and
splitting a 0.3% effect into halves would report precision the quantity does
not have.  So the fraction is computed and printed for every cell and the
*verdict* is read only where `|log A| >= log 1.01` -- where the byte-matched
L arm moves its gate by at least 1%.  Applied to the grids already landed
that admits all six dense cells (0.8916x to 1.1882x) and five of the six GLM
ones, excluding exactly one, `R=4 L=16` at 1.0029x.  It is a reporting
convention and is labelled as one, not a noise floor: the encoder is
deterministic and the repeat control is byte identity, so there is no noise
here to clear -- only conditioning.  It lives in
`matched_reach_report.read_split`, with the threshold pinned by
`tests/test_matched_reach_report.py` against those same landed numbers,
because a threshold that lives only in a comment is one the report can
quietly ignore.

The physical check comes free and is reported: at a matched reach the same
rows are clipped, so `rows_over_reach` at `(14, r*)` must equal it at
`(L, 1.0)` on every unit.  The controls are the sweep's own in-process repeat
plus a **cross-run** one these separate processes make possible and the
in-process repeat cannot give: the shipped `(L=14, r=1)` baseline is
re-encoded in every row and must be byte- *and* tensor-identical to the same
arm in the landed `pair_glm.json` / `pair_dense.json`.
`experiments/matched_reach_report.py` refuses to summarise without it.

That physical check, though, is a check on the *diagnostic*: both sides
compute `rows_over_reach` from the same helper, so it can confirm the
arithmetic and is silent about whether the **encoder** built the table the
helper describes.  What makes a matched arm an arm is a separate identity,
and it is four constants, verified in the code rather than argued:
`bf16_l_sigma_sweep` passes `window_sigma = ratio * BF16_CHANNEL_SIGMA`
(`None` at ratio 1.0) and `encode_unit`'s CHANNEL branch sets
`table_sigma = window_sigma`, falling back to `channel_sigma` only when it is
`None` (`src/tessera/encode.py:2378-2385`); the seed is
`BF16_RECIPE.window_seed = DEFAULT_WINDOW_SEED = 0`
(`src/tessera/export.py:146,714`), which is the helper's default; and `half`
is read *only* on the `sigma is None` branch of `_window_points_cpu`
(`src/tessera/encode.py:739-750`), which this path never takes, so that
argument cannot move the table here at all.  The grid is `BF16_GRID`, the
recipe's own.  `tests/test_matched_reach.py` pins all four, including by
building each table at `half` 8, 16 and 32 and asserting one reach.

**No numbers yet, and that is the state of this section.**  The apparatus,
the ratios and the reading are here; the two `L=14` rows -- the cheap and
decisive ones -- were submitted to the PrismaBuild pool on 2026-09-04
(action keys `9f7abf6f...` for `glm-14` and `8b60e5a1...` for `dense-14`) and
were still queued behind other agents' work when this was written, with
sparklina's GPU held out of the pool until ~07:00 and sparky draining roughly
one item every two and a half minutes.  The records outlive the session that
submitted them, so the rows will run; what they will not do is run tonight.  Nothing below this line
should be read as measured until those files exist at
`/mnt/shared/tessera-runs/reach/matched/`, and
`experiments/matched_reach_report.py` is what turns them into the table above.
The costs, from the landed runs' own per-arm wall clocks: the `L=14` row is
about 12 minutes of GPU on the six GLM experts and 10 on the eight dense
Linears (three rungs each); the `L=16` row, which carries the expensive
width, is about 38 minutes on the experts; the `L=12` row about 11.
