# The per-unit reach lever: a 2.8% oracle on the shipping grid, and no rule that reaches it (2026-09-03)

**Verdict.** Issue #80 is real -- `channel_sigma`'s optimum *is* per-unit -- and
it is **not worth shipping on E4M3**. The per-unit ceiling at the wire's own
rung is an **oracle** 0.9723 of the shipped default's h-weighted weight error,
the oracle is chosen with the answer in hand across 15 arms on 8 units, and the
one predictor derivable from what the encoder already knows about the unit
picks the wrong arm on 8 of 8 units at that rung and would land at **1.0805 --
worse than doing nothing**. On BF16 the same knob is a gauge up to the grid's dyadic
residue class and is worth 0.01% at that rung (0.65% at R2048); BF16's large
per-unit effect lives on the other axis and is already recorded as #48.

Nothing in this document changes the encoder. No default moved, no digest
moved, no wire byte moved.

## Scope of every number below

Weight space, h-weighted (`sqrt(sum_ij h_j e_ij^2 / sum_ij h_j w_ij^2)`), diagonal
Hessian from `/mnt/shared/tessera-runs/bf16/refs/h_diag.pt`. Eight dense
Qwen3-0.6B Linears, no GLM experts. Grid E4M3 and BF16, window body, CHANNEL
plane, L=14, `window_seed=0`. Rungs R1024 (4 b/wt -- **the rate the E4M3 wire
ships at**, 4.07-4.08 bpp) and R2048 (8 b/wt, the Tessera-8 research rung).
Every arm is priced at the bytes it wrote and the default arm was run first and
repeated last, byte-identical both times. Data:
`/mnt/shared/tessera-runs/reach/reach_{e4m3,bf16}_{spread,ratio,wide}.json`,
harness `experiments/bf16_l_sigma_sweep.py --stage reach` at `430ca5e`. No
served KL, no PPL: this is a weight-space screen and it is not a result.

## What the knob actually is: two axes, one clamp

`channel_sigma` and `window_sigma` are usually discussed as one "spread". In the
code they are two different things, and the sweeps confirm the reading:

* **`channel_sigma` multiplier `m`** (at `window_sigma=None`). `encode.py:1700`
  makes the table track the channel scale, so the table's nominal spread and the
  row's move together. The codebook in row-RMS units would be invariant -- and on
  BF16 it is, to a rounding: exactly at dyadic multipliers, and within 0.05% at
  R1024 / 1.75% at R2048 elsewhere. On E4M3 it is not, for one reason: the table's outermost
  entry is snapped onto the grid, and the grid stops at 448. So `m` is a **pure
  reach clamp**: realised `reach_rms` = 384/sigma0 = 4.0773 while the top entry
  fits, then 448/(m*sigma0) once it does not -- 3.8055 at m=1.25, 3.1712 at 1.5,
  2.3784 at 2.
* **`window_sigma/channel_sigma` ratio `rho`.** The table models a Gaussian
  `rho` times wider than the row's, so the codebook in row units is `rho` times
  wider -- until the same 448 stops it. `reach_rms` climbs 4.0773 -> 4.7568 at
  rho=1.25 and never moves again. That saturation is issue #84 and it caps this
  axis at **1.167x**.

Both axes therefore reduce to one physical quantity, the body's reach in
row-RMS units, but they reach it from opposite sides and they are not
interchangeable: `m` narrows without changing the table's modelled sigma, `rho`
widens *and* re-models. E4M3's whole accessible range is `reach_rms` in
(0, 4.7568]; the shipped default sits at 4.0773.

**Direction, because the issue's language inverts.** "A spread 1.5x wider" in
#80 is `m=1.5`, which is **0.78x the reach**, not 1.5x. The three units the
issue says want it "widest" want the *narrowest* reach on the sweep.

## E4M3, R1024 -- the shipping rung

h error relative to the shipped default (`m=1, rho=1`); lower is better.

| unit | m0.5 | m0.75 | rho0.5 | rho0.75 | **default** | rho1.25 | rho1.5 | rho2 | rho4 | m1.25 | m1.5 | m2 | argmin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L2.self_attn.k_proj` | 1.000 | 0.999 | 1.122 | 1.114 | 1.000 | 0.911 | 0.934 | 1.086 | 1.843 | 0.963 | **0.904** | 0.936 | m1.5 |
| `L2.self_attn.q_proj` | 1.000 | 1.001 | 1.042 | 1.042 | 1.000 | 0.980 | 1.027 | 1.222 | 2.117 | 0.977 | **0.946** | 1.042 | m1.5 |
| `L14.mlp.gate_proj` | 1.000 | 1.001 | 1.048 | 1.050 | 1.000 | 0.981 | 1.013 | 1.173 | 1.983 | 0.981 | **0.952** | 1.009 | m1.5 |
| `L27.self_attn.o_proj` | 1.000 | 1.000 | 1.031 | 1.031 | 1.000 | 1.031 | 1.121 | 1.354 | 2.344 | 1.000 | 1.023 | 1.163 | default |
| `L14.self_attn.v_proj` | 1.000 | 1.000 | 1.010 | 1.008 | 1.000 | 1.087 | 1.224 | 1.536 | 2.681 | 1.002 | 1.046 | 1.247 | default |
| `L2.mlp.up_proj` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.098 | 1.242 | 1.565 | 2.754 | 1.006 | 1.051 | 1.259 | default |
| `L14.mlp.down_proj` | 1.000 | 1.000 | 1.002 | 1.002 | 1.000 | 1.026 | 1.118 | 1.362 | 2.360 | 1.008 | 1.077 | 1.291 | default |
| `L2.mlp.down_proj` | 1.000 | 1.001 | 0.982 | 0.985 | 1.000 | 1.098 | 1.226 | 1.474 | 2.249 | 1.036 | **1.151** | 1.371 | rho0.5 |
| **geomean** | 1.0000 | 1.0004 | 1.0287 | 1.0283 | 1.0000 | 1.0244 | 1.1080 | 1.3363 | 2.2722 | 0.9965 | 1.0159 | 1.1555 | **oracle 0.9723** |
| `reach_rms` | 4.0773 | 4.0773 | 2.0386 | 3.0580 | 4.0773 | 4.7568 | 4.7568 | 4.7568 | 4.7568 | 3.8055 | 3.1712 | 2.3784 | |

Three readings.

1. **The split is real and it is asymmetric.** Three units take 5-10% off their
   h error at `m=1.5`; the same arm costs `L2.mlp.down_proj` 15%. A rule that
   misclassifies the heavy-tailed unit gives back more than the light-tailed
   ones win, which sets the bar any rule has to clear.
2. **The oracle is 0.9723.** That is the most a per-unit reach rule can be worth
   at the shipping rung on these units, with the answer already known.
3. **The geomean is a compromise nobody chose, exactly as #80 says** -- but it
   is a *cheap* compromise: no arm beats the default by more than 0.35% on the
   geomean (`m=1.25` at 0.9965).

## E4M3, R2048 -- the 8-bit research rung

| unit | m0.5 | m0.75 | rho0.5 | rho0.75 | **default** | rho1.25 | rho1.75 | m1.25 | m1.5 | m2 | argmin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L2.self_attn.k_proj` | 1.000 | 1.013 | 1.044 | 1.057 | 1.000 | 0.959 | 0.956 | 0.965 | 0.945 | **0.943** | m2 |
| `L2.self_attn.q_proj` | 1.000 | 1.014 | 1.033 | 1.036 | 1.000 | 0.954 | **0.930** | 0.967 | 0.933 | 0.931 | rho1.75 |
| `L14.mlp.gate_proj` | 1.000 | 1.001 | 1.017 | 1.016 | 1.000 | 0.942 | 0.887 | 0.954 | 0.890 | **0.883** | m2 |
| `L27.self_attn.o_proj` | 1.000 | 1.003 | 1.005 | 1.005 | 1.000 | 0.999 | 0.999 | 1.001 | 1.003 | 1.003 | rho1.5 |
| `L14.self_attn.v_proj` | 1.000 | 1.002 | 1.010 | 1.008 | 1.000 | 1.002 | 1.009 | 0.999 | **0.998** | 1.003 | m1.5 |
| `L2.mlp.up_proj` | 1.000 | 1.005 | 1.002 | 1.006 | 1.000 | 1.003 | 1.010 | 1.002 | **1.000** | 1.002 | m1.5 |
| `L14.mlp.down_proj` | 1.000 | 0.999 | 1.001 | 1.000 | 1.000 | 0.983 | 0.973 | 0.981 | 0.970 | **0.970** | m2 |
| `L2.mlp.down_proj` | 1.000 | **1.367** | 0.995 | **1.362** | 1.000 | 1.333 | 1.411 | 1.313 | 1.341 | 1.411 | rho0.5 |
| **geomean** | 1.0000 | 1.0447 | 1.0132 | 1.0557 | 1.0000 | 1.0158 | 1.0122 | 1.0176 | 1.0027 | 1.0086 | **oracle 0.9640** |

At R2048 the split is cleaner -- everything except `L2.mlp.down_proj` wants a
narrower reach -- but two of those numbers are not a reach effect at all. See
"the non-dyadic table" below.

## BF16 -- the spread axis is a gauge up to the grid's residue class

`channel_sigma` multipliers on BF16, h error relative to `m=1`, `reach_rms`
constant at 4.0000 throughout (the BF16 table's outermost entry scales exactly
with sigma, so this axis never touches a clamp):

| unit | m0.5 | m0.75 | m1.25 | m1.5 | m2 | | m0.5 | m0.75 | m1.25 | m1.5 | m2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | *R1024* | | | | | | *R2048* | | | | |
| `L2.self_attn.k_proj` | 1.0000 | 1.0005 | 0.9997 | 1.0005 | 1.0000 | | 1.0000 | 0.9874 | 0.9877 | 0.9874 | 1.0000 |
| `L2.self_attn.q_proj` | 1.0000 | 1.0000 | 0.9998 | 1.0000 | 1.0000 | | 1.0000 | 0.9961 | 0.9947 | 0.9961 | 1.0000 |
| `L14.mlp.gate_proj` | 1.0000 | 1.0000 | 1.0001 | 1.0000 | 1.0000 | | 1.0000 | 1.0004 | 0.9945 | 1.0004 | 1.0000 |
| `L27.self_attn.o_proj` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | | 1.0000 | 0.9996 | 0.9985 | 0.9996 | 1.0000 |
| `L14.self_attn.v_proj` | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | | 1.0000 | 0.9994 | 0.9979 | 0.9994 | 1.0000 |
| `L2.mlp.up_proj` | 1.0000 | 0.9999 | 1.0000 | 0.9999 | 1.0000 | | 1.0000 | 0.9999 | 0.9995 | 0.9999 | 1.0000 |
| `L14.mlp.down_proj` | 1.0000 | 0.9999 | 0.9999 | 0.9999 | 1.0000 | | 1.0000 | 0.9954 | 0.9933 | 0.9954 | 1.0000 |
| `L2.mlp.down_proj` | 1.0000 | 0.9998 | 1.0000 | 0.9998 | 1.0000 | | 1.0000 | 0.9825 | 0.9955 | 0.9825 | 1.0000 |
| **geomean** | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | | 1.0000 | 0.9951 | 0.9952 | 0.9951 | 1.0000 |
| **oracle** | | | | | **0.9999** | | | | | | **0.9935** |

**Dyadic multipliers are an exact gauge** -- `m=0.5` and `m=2` are 1.0000 on all
eight units at both rungs -- because BF16 is closed under scaling by two away
from the exponent extremes and nearest-value snapping commutes with it, so the
codebook in row-RMS units is the same object. `m=0.75` and `m=1.5` give
column-for-column identical numbers for the same reason (they differ by a factor
of two). What is left is the **residue class**: a non-dyadic multiplier snaps the
table's quantiles onto different grid points, which is worth at most 0.05% per
unit at R1024 and 0.02-1.75% per unit at R2048.

**So the per-unit lever this issue is about is worth 0.01% on BF16 at R1024 and
0.65% at R2048** -- under 1% at both rungs, oracle, before any rule.

BF16's large per-unit effect is on the `rho` axis, where the reach is unbounded:
a per-unit oracle of **0.7602** at R2048 (`k_proj` 0.498 at rho1.5, `gate_proj`
0.461 at rho2, `q_proj` 0.558 at rho2) against 0.9875 at R1024. That is #48's
14-15% finding read per unit, on the 16-bit route, which has no serving lane and
no served KL. It is recorded here, not acted on.

## The derived predictor, and why it fails

A per-unit value found by sweeping each unit is a lookup table. The rule has to
come from something the encoder can compute from the unit itself, so the
candidate is the model the window table is *built* on: at each position the
trellis may land on the `2^R` entries its `R` new bits reach, and those entries
are a permutation of the table's quantiles, so one position is the nearest of
`K = 2^R` draws from the table's own empirical distribution (Tseng et al.'s
random Gaussian codebook). For a target `t`,

    D(t) = E[ min_K (t - c)^2 ] = int_0^inf 2x (1 - F(t+x) + F(t-x))^K dx

and the unit's predicted h error is `sqrt(sum_rj h_j s_r^2 D(w_rj / s_r) /
sum_rj h_j w_rj^2)` with `s_r` the row scale `initial_channel_scale` actually
assigns under that arm -- the production function called, not modelled, so the
reach-aware per-row start is inside the prediction. No fitted constant, no
per-unit search: `(w, h, codebook, R)` in, an arm out.
`experiments/reach_predictor_check.py` scores it against the arms already on
disk.

| rung | argmin agreement | geomean if the rule is followed | oracle |
|---|---|---|---|
| R1024 (shipping) | **0/8** | **1.0805** | 0.9723 |
| R2048 | 3/8 | 1.0221 | 0.9640 |

It fails in one direction on every unit: it over-credits a narrow reach, because
it prices one position and the window body's advantage is the path. On
`L2.mlp.down_proj` at R1024 it predicts 0.710 for `m=2` where the encoder
measures 1.371 -- the catastrophic misclassification, on the one unit that
cannot afford it. Rank correlation is weakly positive on 7 of 8 units
(Spearman 0.43-0.74) and negative on that one (-0.53), which is the worst
possible combination: enough signal to look promising, wrong where it costs.

**No cheaper statistic works either.** The issue's own first candidate is the
`over` fraction the reach stage already computes. It is not predictive: at
R1024 `k_proj` (over = 0.202) is helped 10% by `m=1.5` and `o_proj` (over =
0.208) is hurt 2%; `up_proj` and `v_proj` have the *lowest* over (0.143, 0.145)
and are both hurt. `max_z`, per-row kurtosis and h-weighted kurtosis separate no
better, except that h-weighted kurtosis does isolate `L2.mlp.down_proj` (1140
against <= 8.1 for every other unit). Using that as the sole gate -- "narrow
unless the h-weighted kurtosis says massive-activation" -- gives 0.9661 at
R2048 (near the 0.9640 oracle) and **0.9983 at R1024**, which is 0.17% at the
rung that ships. Eight units cannot fit anything more than that, and the issue
says so itself.

## Recommendation

Do not ship a per-unit reach rule. Close #80 with this measurement.

* The ceiling at the shipping rung is 2.8% of a weight-space screen, which is
  well inside the region where weight-space and served KL have repeatedly
  disagreed in this repo, and it is an oracle rather than a rule.
* The only derived rule available is net negative (1.0805).
* The cost is not zero even though the bytes are: a per-unit spread is a
  per-unit `encoder_profile_id` and a per-unit schema minor 5, so a checkpoint
  stops having one encoder identity. That is a real price for 2.8% of a screen.

What is worth carrying forward from the sweep is not the lever:

* **The non-dyadic table.** At R2048, `L2.mlp.down_proj` costs 1.36x under
  *every* arm whose table sigma is a non-dyadic multiple of sigma0, and 1.00x
  under every dyadic one -- independently of reach, and reproduced across the
  two arm families that share only the table sigma (m0.75 -> 1.367 at
  `reach_rms` 4.0773, rho0.75 -> 1.362 at `reach_rms` 3.0580). Its `wt` moves
  0.3% while its `h` moves 37%, the codebook's own one-step RTN distortion moves
  1%, and clipped energy falls rather than rises. Filed separately; it means the
  R2048 column for that unit is not measuring reach.
* **The reach start lands with round-to-nearest.** `initial_channel_scale`
  computes `reach * rms / amax` as a *lower bound* on a row's scale and lands it
  with `land_channel_scale`, which rounds to nearest; 166 of 1024 rows of
  `L2.mlp.down_proj` land about 3e-4 below the bound and clip the loud weight
  the bound was raised for. `land_at_least` exists for exactly this and is used
  only by the refit's floor. The lost energy is small here (1.9e-8 of the
  h-weighted budget) but the code says a thing it does not do. Filed separately.
