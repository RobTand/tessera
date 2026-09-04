# The dense 4-bit residual is the comparator's compensation, not Tessera's plane (2026-09-03)

**Disposition of issue #12: live, unchanged, and re-scoped a second time.** The
served number is still **1.039x** -- Tessera W4A4 **0.531** at 4.0018 bpp
against PrismaQuant NVFP4 GPTQ+JSO **0.511** at 4.5 bpp, both W4A4, both on the
same corpus and teacher, 220,301,312 wire bytes either way
(`tessera-ldlq-lut-plane-served-2026-09-02`, lines 642-645). Nothing has moved
it: `DEFAULT_REFIT_OBJECTIVE["lut16"]` is still `h^1.0`, the objective that was
served, and `git log b877c7f..HEAD -- src/tessera/encode.py src/tessera/export.py`
is empty. The #35 Gauss-Seidel arm improves both roles in weight space
(`q_proj` 0.9819x, `k_proj` 0.9320x) but it is a screen with no serve behind
it.

> **Superseded in part 2026-09-04 (#103).** The clause that used to follow --
> "`refit_gauss_seidel` ... is **not** a field on `export.ActivationSource`,
> so no exporter can set it and `tessera_config.json` has no field for the
> merge guard to compare" -- is no longer true: it is a field, it rides into
> the config, and the merge guard compares it. The *screen* half of the
> sentence stands; only the export path moved.

What this document adds is the mechanism the issue asked for. It is not the one
the thread expected, and two of the natural hypotheses are measured down rather
than argued down.

## The residual factors exactly, and the factor that orders it is not the plane

Join the plane census's NVFP4-RTN arm to the residual census's three arms and
the census ratio is a product of four measured ratios on the same unit and the
same Hessian:

    C/A  =  plane  x  gptq  /  (body x comp)

| factor | definition | what it prices |
|---|---|---|
| `plane` | `P_lut16 / P_e4m3` | Tessera's sixteen per-unit E4M3 scale entries against NVFP4's one E4M3 byte per 16, alphabet and rounding held fixed |
| `gptq` | `P_e4m3 / A` | what GPTQ+JSO buys NVFP4 over its own RTN |
| `body` | `P_lut16 / B` | what the trellis buys over E2M1 nearest rounding on the same plane |
| `comp` | `B / C` | what LDLQ 1.0/32 plus the `h^1.0` block-scale refit buys |

It is an identity, not a model: `max |log(C/A) - log(plane x gptq / (body x
comp))| = 4.4e-16` over all 196 Linears, and each arm's aggregate is joined back
to the census it came from (`A_check` 196/196, and 5/5 per unit in the per-row
run below).

| role | n | C/A | plane | gptq | body | comp | comp/gptq | A `hq` | btwRow sd | oct |
|---|---|---|---|---|---|---|---|---|---|---|
| `o_proj` | 28 | 0.8421 | 1.0317 | 1.2900 | 1.0070 | 1.5696 | 1.2167 | 0.07269 | 0.291 | 5.61 |
| `down_proj` | 28 | 0.8810 | 1.0524 | 1.1282 | 1.0101 | 1.3343 | 1.1827 | 0.07604 | 0.259 | 5.98 |
| `up_proj` | 28 | 0.9411 | 1.0107 | 1.2356 | 1.0134 | 1.3095 | 1.0597 | 0.07158 | 0.220 | 5.35 |
| `v_proj` | 28 | 0.9554 | 1.0100 | 1.2170 | 1.0094 | 1.2746 | 1.0473 | 0.07626 | 0.195 | 4.93 |
| `gate_proj` | 28 | 0.9772 | 1.0239 | 1.3722 | 1.0135 | 1.4185 | 1.0338 | 0.04640 | 0.347 | 5.57 |
| **`q_proj`** | 28 | **1.0648** | 1.0356 | 1.3244 | 1.0025 | 1.2848 | **0.9701** | 0.04931 | 0.567 | 6.49 |
| **`k_proj`** | 28 | **1.0976** | 1.0443 | 1.3515 | 1.0049 | 1.2796 | **0.9468** | 0.04945 | 0.479 | 6.14 |
| all | 196 | 0.9619 | 1.0297 | 1.2716 | 1.0087 | 1.3495 | 1.0613 | | | |

Three readings, in descending order of how much they change what to do next.

**The ordering is `comp/gptq` and nothing else.** Per unit against `log(C/A)`,
Spearman: `comp/gptq` **-0.949**, `gptq` +0.626, `comp` -0.495, `plane` +0.191,
`body` -0.299. Per role the `comp/gptq` ordering reproduces the `C/A` ordering
exactly, and `q_proj` and `k_proj` are the only two roles below 1.0. A role
loses when the comparator's compensation removes more error than Tessera's
does, and for no other reason that this decomposition can see.

**The trellis body is worth 1.0087x at this rate**, geomean over 196 units, sd
of `log body` **0.0093**. At the E2M1x2 cap the tuple trellis is within 1% of
plain E2M1 nearest rounding on the same plane. That is consistent with
`tessera-trellis-codebook-geometry` ("0.0-0.3% on the E2M1x2 cap, closed for
Tessera-4"), measured here on a 196-unit set rather than on the cap's codebook
geometry, and it means no share of this residual is recoverable from the body.

**The plane is real but it is a quarter, and it is not what orders the roles.**
`plane` costs 1.0297x overall and 1.0356x/1.0443x on `q`/`k` -- but its largest
value is on `down_proj` (1.0524x), which wins by 12%. Tessera's sixteen-entry
scale table is 4 bits per 16 cheaper than NVFP4's byte, most of the 4.00-vs-4.50
gap, and it costs about 3% of weight-leg error to buy that. That is a priced
trade, not a defect.

## The matched pairs: same Hessian, same shape, and only one factor moves

`q_proj`, `k_proj` and `v_proj` of one layer read the same hidden state, so
their fit Hessians are bit-identical -- checked per layer, 28 of 28, not assumed
-- and `k_proj` and `v_proj` are the identical shape at the identical rate on
the identical alphabet. Whatever separates them is a property of `W`.

| factor | `k_proj / v_proj` | layers where k > v | `q_proj / v_proj` | layers where q > v |
|---|---|---|---|---|
| `plane` | 1.0339 | 23 / 28 | 1.0253 | 21 / 28 |
| **`gptq`** | **1.1105** | **28 / 28** | **1.0882** | **28 / 28** |
| `body` | 0.9955 | 8 / 28 | 0.9932 | 4 / 28 |
| `comp` | 1.0040 | 13 / 28 | 1.0080 | 17 / 28 |
| C/A | 1.1488 | 28 / 28 | 1.1144 | 28 / 28 |

In logs, the `k`-over-`v` gap is **76% the `gptq` leg, 24% the `plane` leg, and
nothing at all the body or Tessera's own compensation** (0.1048 + 0.0333 +
0.0045 - 0.0040 = 0.1386 = log 1.1488). The `q`-over-`v` split is 78% / 23%.

So the sentence the issue has been carrying -- that `q_proj` and `k_proj` are
where Tessera's compensation is weak -- does not survive the control. Tessera's
compensation is **flat** across the triple: it removes the same fraction of
error on `k` as on `v`, on 15 of 28 layers slightly more. What varies is the
comparator: GPTQ+JSO removes 11.1% more on `k` than on `v`, on every single
layer. The role-level appearance of a weak `comp` on `q`/`k` (1.28 against
`o_proj`'s 1.57) is a between-role confound with the Hessian, and the
H-matched control removes it.

## The shape of it: one side's compensation scales with the headroom and the other's does not

`P_exact` -- E2M1 nearest rounding on an exact fp32 scale plane -- is a
difficulty axis that appears in neither compensation leg, so correlating a leg
against it is not shared-denominator arithmetic. Over the 196 units:

| | Spearman against `log P_exact` |
|---|---|
| `gptq` (NVFP4's GPTQ+JSO) | **-0.523** |
| `comp` (Tessera's LDLQ + refit) | **+0.026** |
| `plane` | -0.264 |

| difficulty tercile | `P_exact` | `gptq` | `comp` | `plane` | C/A |
|---|---|---|---|---|---|
| easiest third | 0.06092 | **1.3621** | 1.3443 | 1.0396 | **1.0452** |
| middle third | 0.07857 | 1.2460 | 1.3462 | 1.0336 | 0.9482 |
| hardest third | 0.09780 | **1.2123** | 1.3579 | 1.0162 | **0.8989** |

**The comparator's compensation converts an easy unit into a larger fractional
gain; Tessera's removes the same fraction whatever the unit.** Across the three
terciles `comp` reads 1.3443, 1.3462, 1.3579 -- flat to three digits -- while
`gptq` falls from 1.3621 to 1.2123. C/A follows the difference, from 1.0452 on
the easy third to 0.8989 on the hard third.

And `q_proj` and `k_proj` are among the easiest roles in the model:

| role | `P_exact` | `gptq` | `comp` | C/A |
|---|---|---|---|---|
| `gate_proj` | 0.06319 | 1.3722 | **1.4185** | 0.9772 |
| **`q_proj`** | 0.06484 | 1.3244 | 1.2848 | **1.0648** |
| **`k_proj`** | 0.06619 | 1.3515 | 1.2796 | **1.0976** |
| `down_proj` | 0.08440 | 1.1282 | 1.3343 | 0.8810 |
| `up_proj` | 0.08770 | 1.2356 | 1.3095 | 0.9411 |
| `v_proj` | 0.09199 | 1.2170 | 1.2746 | 0.9554 |
| `o_proj` | 0.09288 | 1.2900 | 1.5696 | 0.8421 |

Difficulty alone is not the answer, and `gate_proj` is the unit that says so:
it is the **easiest** role of the seven and it still wins, because its `comp`
sits at 1.4185 rather than `q`/`k`'s 1.28. Two conditions have to coincide for
a role to lose -- the unit is easy enough that the comparator's compensation is
near its best, and Tessera's compensation happens to sit at the low end of its
own role range. `q_proj` and `k_proj` are the only two roles where both hold.

`comp` is not a pure function of the Hessian either. `gate_proj` and `up_proj`
share a bit-identical `H` and their `comp` differs by **1.0833x on 28 of 28
layers**, which is what keeps `gate_proj` winning against the same 1.1105x
`gptq` disadvantage that sinks `k_proj`.

## Two hypotheses, measured down

**Rotation is not on the list** and was not tried: `rotation-hurts-block-scaled-formats`
has it 2.5-2.8x worse served on this model.

**The reach-aware per-row start cannot be it.** `encode_unit` calls
`initial_channel_scale` only when `scale_plane is ScalePlaneKind.CHANNEL`, and
`wire_recipe(E2M1x2, 896)` returns `scale_plane=ScalePlaneKind.LUT`. The thread
established this already; it is repeated here because it is the first thing
anyone reaching for the 8-bit route's 0.470 -> 0.151 result will try.

**QK-norm makes a head's scale a gauge, and it explains 2-3%, not the residual.**
The checkpoint has `q_norm.weight` and `k_norm.weight` on every layer and no
`v_norm` -- the one structural difference between `k_proj` and `v_proj`. A
per-head RMSNorm over `head_dim` makes each head's overall magnitude an *exact*
gauge freedom, so the weighting the next op applies is equal per head, not
proportional to head energy, and the derived aggregate is

    hq_head = sqrt( mean_j  E_j H E_j^T / W_j H W_j^T )

the RMS over heads of the per-head relative error. Measured on layers 0-1, arms
joined 5/5 to both censuses, control bit-identical:

| role | out-norm | btw-head sd log2 | within-head sd log2 | C/A plain | C/A head |
|---|---|---|---|---|---|
| `q_proj` | yes | 0.226 | 0.493 | 1.0759 | **1.0421** |
| `k_proj` | yes | 0.169 | 0.445 | 1.0909 | **1.0717** |
| `v_proj` | no | 0.127 | 0.183 | 0.9125 | 0.9135 (control) |

The direction is right and the control does not move (0.1%), so the effect is
real. But it is worth 3.1% on `q_proj` and 1.8% on `k_proj`, and the reason is
in the same table: on `q`/`k` the row-scale spread is **within-head** (0.49,
0.45) rather than between-head (0.23, 0.17), and only the between-head part is
a gauge. The head gauge is a correction to the metric, not the mechanism.

Adjacent and unmeasured: `q_norm.weight` and `k_norm.weight` are learned
*per-dim* gains with sd `log2` of **1.435** and **2.595** across the 28 layers,
and the census metric ignores them entirely. A row's amax anti-correlates with
its gain (median Spearman -0.167 on `q_proj`, 28 of 28 layers negative; -0.178
on `k_proj`, 26 of 28), so the dims that dominate `hq` are mildly the ones the
norm turns back down. Whether a `g^2`-weighted aggregate changes the role
ordering is not measured here. Filed as its own issue.

## Where the difference lives, and what fixing it is worth

PENDING_BUCKETS

## The lever the mechanism names, and it is already exportable

If the deficit is that Tessera's compensation is coarser than the comparator's
on these two roles, then narrowing the block that quantises together must buy
more on `q`/`k` than on `v_proj` -- and if it buys the same everywhere, the
diagnosis is wrong. `block_ldl`'s own docstring says the columns inside a block
"see no correction from each other" and that "smaller blocks therefore
compensate more". The served default is `block=32`; production GPTQ compensates
column by column.

Swept on the H-matched triple in layers 0 and 1, one arm changing and it is
`ldl_block`, `refit_metric` fixed at the served `h^1.0`, `out` on the held-out
eval slice, the default arm run first and again last and bit-identical on all
six units, bpp identical to four decimals on every arm:

| unit | b16 | **b32 (default)** | b64 | b128 | b16/b32 |
|---|---|---|---|---|---|
| `L0.q_proj` | 0.04166 | **0.04375** | 0.04625 | 0.04674 | **0.9522** |
| `L0.k_proj` | 0.03880 | **0.04192** | 0.04343 | 0.04470 | **0.9256** |
| `L0.v_proj` | 0.07289 | **0.07442** | 0.07654 | 0.07774 | 0.9794 |
| `L1.q_proj` | 0.05241 | **0.05501** | 0.05593 | 0.05650 | **0.9527** |
| `L1.k_proj` | 0.04328 | **0.04567** | 0.04602 | 0.04681 | **0.9477** |
| `L1.v_proj` | 0.06828 | **0.06969** | 0.07031 | 0.07121 | 0.9798 |

The prediction holds. `out` is monotone in block size on all six units, and
halving the block is worth **5.5% on `q`/`k` against 2.0% on `v_proj`** -- the
2.8x ratio the diagnosis asked for. Carried onto the census `hq` ratios, the
`b16` geomeans are `q_proj` 0.9497 and `k_proj` 0.9348, which would move the
role table to `q_proj` **1.011** and `k_proj` **1.026** from 1.065 and 1.098.
That is most of the residual, and it is free in bytes.

Two things make this lever cheap in a way `refit_gauss_seidel` is not:

* `ldlq_block` is already a field on `export.ActivationSource`, already written
  into the `activation_aware` config block, and already compared by the merge
  guard (`tests/test_merge_guard.py:172`). A serve arm needs no plumbing.
* The dial has never been searched. `tessera-ldlq-lut-plane-served-2026-09-02`
  records that on GLM's 4096-column experts `ldl_block=128` is "flat to
  marginally better" at 9.0x encode against 35.1x -- the opposite end of the
  same axis, on a different model. A default of 32 that is beaten by 16 on every
  dense unit measured and matched by 128 on every GLM expert measured is a round
  number, not a derived one. Filed as its own issue.

Extrapolating a two-layer ratio onto 28 layers is the weak step, and no arm
below `block=16` was run, so the floor of the axis is unknown. The end of the
axis is `block=1`, which is full sequential compensation and the comparator's
own granularity.

## What this does not license

* **Nothing here is served.** Weight leg, `hq` and one held-out out-space slice.
  The gate for any change to the LUT-plane default remains a serve at
  byte-identical bytes against the same teacher in the same vLLM session as a
  re-run of the current default, plus a GLM six-expert geomean no worse than
  1.00x -- the E2M1 route's wins are on GLM's experts and must not be paid for
  out of them.
* **Arm C is scored in-sample on `H`.** `dense4_out_check.json` measured the
  discount on six units at 0.9908x to 1.1159x, geomean 1.0361, and 1.014x on
  the two roles that carry the gap. Read every `hq` ratio with that spread.
* **The census aggregate and the served scalar are different objects.** 196
  Linears of weight-leg error against one whole-model KL, bridged only by a
  six-unit out-of-sample check that agreed to half a percent (1.035x weight leg
  against 1.039x served). That agreement is suggestive, not a derivation.
* **Both A-side numbers are W4A4.** The activation leg is the same on both arms
  and this document prices none of it. A `W?A8` figure from the 8-bit route does
  not belong in any table above.
* **One model, one wire, one rate.** Qwen3-0.6B dense, E2M1x2 at `q256=896`,
  `sigma=1.0`. No GLM arm was run.

## Provenance

| what | where |
|---|---|
| the identity and the role table | `experiments/dense4_decomp_report.py` over `dense4_plane_census.json` + `dense4_residual_census.json` |
| the plane arms and the field statistics | `experiments/dense4_plane_census.py` -> `/mnt/shared/tessera-runs/ldlq-lut/dense4_plane_census.json` (196 units, control 4/4 bit-identical) |
| the census arms | `experiments/dense4_residual_census.py` -> `dense4_residual_census.json` (196 units) |
| the per-row legs, buckets and oracle | `experiments/dense4_perrow_legs.py` -> `dense4_perrow_legs.json` |
| the QK-norm head aggregate | `experiments/dense4_qknorm_gauge.py` |
| the block sweep | `experiments/dense4_reach_sweep.py` -> `dense4_reach_sweep.json` |
| H and the held-out rows | `h_full_qwen06b.pt`, `x_eval_qwen06b.pt` (wikitext-2 train; fit slice 16384 tokens, eval slice 8192, disjoint; the KL corpus is the test split) |
| the comparator | `/home/rob/dq-runs/fc45-0p6b-nvfp4/exported`, one `config_groups` entry over all 252 targets, so its GPTQ+JSO settings are role-uniform |

`dense4_plane_census.py`, `dense4_plane_mechanism.py` and `dense4_reach_sweep.py`
are a prior session's uncommitted work, carried in verbatim from
`/home/rob/tmp/ts12-mechanism`; that session produced the data on `/mnt/shared`
and did not report. `dense4_plane_mechanism.py` is unrun: its hypothesis is that
the plane carries the residual, which the completed plane census does not
support.
