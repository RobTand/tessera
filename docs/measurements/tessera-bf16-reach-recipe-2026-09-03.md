# The BF16 wire learns its own reach, and the rule is `sqrt(R)` (2026-09-03)

Issue **#48**. The BF16 recipe left `window_sigma` unset, which ties the window
table's spread to the row scale and so pins the body's **reach** — how many
row-RMS the largest table entry can express — to one value at every rung.
Issue #18 measured that value optimal at R=4 and 19% off at R=8, and could only
measure it by pinning `window_sigma` *outside* the recipe. This is the build,
and the re-measurement of the thing that was built.

**Claim (the rule).** The optimal reach is `reach*(R) = reach*(4) · sqrt(R/4)`,
floored at the reference rung. The **exponent is derived** — the loading factor
of a quantizer over a Gaussian source balances a granular term in `A² 2^(-2R)`
against the tail past `A`, giving the classical `A* ~ sqrt(2 ln N)` and, with
`N = 2^R`, a square root in the rate. The **amplitude is calibrated**, at the
one rung where the measurement says the pinned value is already optimal. It is
not derived: the classical constant is `sqrt(2 ln 2) = 1.177` and the measured
one is `4.05 / sqrt(4) = 2.03`, 1.7× larger, in the direction this mechanism's
overload predicts — a row that passes the reach is *rescaled whole*, not
clipped sample by sample, so overload is dearer here than the tail integral
says and the optimum sits further out. That factor is a calibration and is
labelled as one.

**Claim (the law predicts) — weights-only encode.** R=5, 6 and 7 were never
fitted. At each the law's value is the **interior optimum on `wt`, on 4 of 4
dense Qwen Linears** — both brackets (the law halved and doubled in reach) lose.
Against the pinned wire the built recipe is **0.975× / 0.943× / 0.910× / 0.813×
on `wt` at R=5/6/7/8**, on 4 of 4, at identical bytes; **0.979× / 0.930× /
0.864× / 0.741× on `h`**, on 3 of 4. The R=8 number is #18's `0.8127×`,
reproduced through the built path rather than through the proxy that found it.
*Second pass, 2026-09-03:* the R=7 and R=8 cells reproduce exactly
(`reach_recipe_repro_r78.json`: pinned/built 1.0986 / 1.2302 on `wt`, 1.1575 /
1.3488 on `h`, gauge twin 0.9988 / 0.9998). **But this is the weights-only
encode, not the one an export runs, and the `h` here is `h_diag.pt`, not the
production capture.** Under the production encode the picture changes — see
"The production encode and the matched control" below: the pinned wire loses on
3 of 4, not 4 of 4, the `wt` optimum is not interior, and the R=4 calibration
point is not the production optimum.

**Claim (the law also fails, and where).** Below the reference the law asks for
*less* reach and is wrong: the pinned value is better by 0.04% (R=1), 0.6%
(R=2) and 1.7% (R=3) on `wt`, on 4 of 4, and at R=3 it beats the law's brackets
on both sides. Two different things are happening, and only one has a
mechanism. At R=1 and R=2 the per-row clamp saturates — `initial_channel_scale`
scales any row whose largest weight would fall outside the reach until it fits,
and at the law's spread that holds **99.8–100.0%** of the rows. Once every row
is clamped the row scale comes from its amax however wide the table is, the
reach is a gauge again, and there is nothing left to buy: both brackets read
1.0000 at R=1, and the loss is only 0.04–0.6%. At R=3 the clamp does **not**
saturate (it holds 0.66–0.96 of the rows against the pinned wire's 0.23–0.55)
and the law loses by the most of the three. Decomposed per row
(`r3_rows.json`, second pass) it is **not a clamp effect**: on 3 of 4 units the
rows clamped under *neither* reach lose the most per row (4.6–5.5%), the rows
that change clamp status 3.6–4.7%, the rows clamped under both 2.3–2.5%; on
`down_proj`, where 55% of rows are already clamped at the pinned reach, the
delta is 55% in the rows that change status and only 0.7% in total. A narrower
table costs every row on these units — a spread effect the overload derivation
does not price. So the optimum does not fall below the reference reach on these
rows, and the mechanism is the table's spread against ordinary rows, not the
tail. Either way the recipe **floors at the reference rung**,
and the floored recipe writes the pinned wire's bytes at R ≤ 4, verified unit
for unit.

**This is an encode-side screen on four dense Linears. It is not a serve.**
The BF16 lane exists (`TESSERA_BF16_K1`, contract v7, two sm_121 dense cells
`backed_with_serve_flag`) and was attested at q256 1792 on the *pinned* wire;
nothing here has been served, so nothing here is promotable under principle 3.
The next gate is stated at the bottom.

Harness: `experiments/bf16_reach_recipe.py`. The receipt's original table was
made with the first cut of that harness (`--rungs 512 1024 1536 2048
--gauge-twin`, sqrt(2) brackets, `wt` + `h` from `h_diag.pt`); the current
harness adds `--production`, `--eval-x`, `--weights-only` and a quarter-octave
grid and was used for the second pass.
Raw: `/mnt/shared/tessera-runs/bf16/qsweep/reach_recipe.json` (R=2,4,6,8),
`reach_recipe_odd.json` (R=1,3,5,7), `reach_recipe_repro_r78.json` (second-pass
reproduction of R=7/8), `reach_production.json` (production encode),
`reach_control.json` (the matched control), `r3_rows.json` (per-row
decomposition at R=3 and R=5).
Parent receipt: `docs/measurements/tessera-bf16-gauge-and-dense4-residual-2026-09-02.md`.

---

## What was pinned, and why it could not be spent

`encode_unit`'s CHANNEL branch resolves the table's spread and the reach start
from `window_sigma`, falling back to `channel_sigma` when it is unset
(`src/tessera/encode.py:1382`, `:1434-1436`). `BF16_RECIPE` left it unset, so
both ends of the scale moved together and the ratio that *is* the reach was 1
by construction. That is also why `BF16_CHANNEL_SIGMA` is a **gauge**: bf16 is
closed under ×2 and nearest-value snapping commutes with ×2, so a dyadic change
to it moves the file and not the tensor (#18). The two constants are different
things and only one of them can express reach.

## The rule

```
window_sigma(R) = BF16_WINDOW_SIGMA · sqrt(max(R, 4) / 4)
```

with `BF16_WINDOW_SIGMA = 1.0 = BF16_CHANNEL_SIGMA`, so the value *is* the
ratio and the reference rung reproduces the pinned wire exactly. `R` is the
rung's rate rounded to whole bits — nearest, not the ceiling `_window_bits_for`
takes, because a table narrower than the rate is a hard constraint and this is
not: rounding costs at most `sqrt(4.5/4) = 1.06` of the continuous law near the
reference and keeps `recipe_table` at one range per rate (15 rows for BF16)
instead of one per q256.

| R | window_sigma | reach, row-RMS |
|---|---|---|
| 1–4 | 1.0000 | 4.05 |
| 5 | 1.1180 | 4.53 |
| 6 | 1.2247 | 4.96 |
| 7 | 1.3229 | 5.36 |
| 8 | 1.4142 | 5.73 |
| 16 | 2.0000 | 8.10 |

## The measurement

Four dense Qwen Linears (`2.mlp.down_proj`, `2.self_attn.q_proj`,
`2.self_attn.k_proj`, `14.mlp.gate_proj`), one process per run, every arm of a
(unit, rung) writing **identical bytes** — checked, not assumed — and the built
arm run first and repeated last, byte- *and* tensor-identical on 32 of 32
controls. Geomeans over the four units, each arm relative to the law's arm, so
**a number above 1.000 is an arm the law beats**. `wins` counts the units where
the arm is worse than the law.

| R | law σ | arm | `wt` geo | `h` geo | `wt` worse on |
|---|---|---|---|---|---|
| 1 | 0.500 | pinned 1.0 | **0.9996** | 1.0007 | 0/4 |
| 1 | | law/√2 | 1.0000 | 1.0006 | 2/4 |
| 1 | | law·√2 | 1.0000 | 1.0007 | 1/4 |
| 2 | 0.707 | pinned 1.0 | **0.9944** | 0.9993 | 0/4 |
| 2 | | law/√2 | 1.0000 | 1.0002 | 2/4 |
| 3 | 0.866 | pinned 1.0 | **0.9835** | 0.9884 | 0/4 |
| 3 | | law/√2 | 1.0060 | 1.0050 | 4/4 |
| 3 | | law·√2 | 0.9864 | 1.0026 | 0/4 |
| **4** | **1.000** | **pinned 1.0 (= the law)** | **1.0000** | **1.0000** | **0/4** |
| 4 | | law/√2 | 1.0437 | 1.0413 | 4/4 |
| 4 | | law·√2 | 1.0265 | 1.0272 | 4/4 |
| 5 | 1.118 | pinned 1.0 | 1.0261 | 1.0212 | 4/4 |
| 5 | | law/√2 | 1.0940 | 1.0877 | 4/4 |
| 5 | | law·√2 | 1.0761 | 1.0569 | 4/4 |
| 6 | 1.225 | pinned 1.0 | 1.0603 | 1.0758 | 4/4 |
| 6 | | law/√2 | 1.1367 | 1.1389 | 4/4 |
| 6 | | law·√2 | 1.1143 | 1.0242 | 4/4 |
| 7 | 1.323 | pinned 1.0 | 1.0986 | 1.1575 | 4/4 |
| 7 | | law/√2 | 1.1520 | 1.2240 | 4/4 |
| 7 | | law·√2 | 1.1343 | 0.9672 | 4/4 |
| 8 | 1.414 | pinned 1.0 | 1.2302 | 1.3488 | 4/4 |
| 8 | | law·√2 | 1.1272 | 0.8613 | 4/4 |

(At R=2 and R=8 one bracket coincides with the pinned arm and is not run twice
under two names. The R=1–3 rows were measured before the floor was added, which
is what the floor is made of: the "law" column there is the *unfloored*
`sqrt(R/4)`, and the shipping recipe at those rungs is the `pinned` row.)

### Four readings

* **The exponent survives four rungs it was not fitted to.** R=5, 6 and 7 have
  the law strictly inside its own bracket on `wt` on 4 of 4 units, and R=8
  reproduces #18. Two points spaced by √2 could not separate `sqrt(R)` from
  `sqrt(R+c)` at all. Six place the optimum strictly inside a ×√2 bracket
  around `sqrt(R/4)` at every rung from 5 to 8, which excludes any exponent
  that would move the reach by ≥ √2 across that range — linear in `R` is out.
  It does **not** pin `c` to zero: at R=8 a reach of 4.9 (`c ≈ 4`) sits between
  the sampled 4.0 and 5.66 and was never measured.
* **`h` follows but by less, and not everywhere.** The law wins `h` on 3 of 4
  units at every rung above the reference; the fourth is `2.mlp.down_proj`,
  which #18 already identified as the unit that prefers *more* clipping on the
  Hessian-weighted metric. At R=7 and R=8 the wide bracket beats the law on `h`
  (0.967, 0.861) while losing badly on `wt` — the two metrics disagree in sign
  there, exactly as #18 found for `L`. **`wt` is what the law is fitted to and
  `h` is not a free win.**
* **The reference rung is a re-parameterisation, not a change.** At R=4 the
  built arm and the explicit pinned arm are the same file on 4 of 4 units, and
  after the floor so are R=1, 2 and 3 — 16 of 16 (unit, rung) pairs at R ≤ 4,
  byte-identical.
* **The gauge twin bounds how exactly this could reproduce.** #18 spent the
  ratio as `(σ_table 1.0, σ_row 0.707)`; the recipe spends it as `(1.414, 1.0)`.
  Those are the same reach related by a **non-dyadic** ×√2, so by #18's own
  gauge result they are different orbits. Measured, they agree to within
  **0.12% at every rung** (0.9988–1.0000 on `wt`, the worst at R=7) and to
  **0.02% at R=8**, where the reproduction claim is made. The recipe reproduces the
  sweep to the tolerance a re-parameterisation allows and no closer, which is
  the right amount.

## The production encode and the matched control (second pass, 2026-09-03)

Everything above is a **weights-only** encode (`encode_linear_planes` with no
Hessian: no LDLQ, no metric-aware refit). An export runs
`ActivationSource.from_capture` defaults — LDLQ σ=1.0 block=32 and the refit
objective `{channel: hessian, lut16: h^1.0, s6b: plain}` — and `for_unit()`
passes only encode kwargs, so a run with and without it differs in the encode
alone. Both runs below use the same four units, the same quarter-octave grid
(`law · 2^(k/4)`, k = −4…4 at R=4, −4…4 elsewhere), the same scorer, the
production capture `/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt` for `h`,
and `/mnt/shared/tessera-runs/bf16/refs/x_eval_dense4.pt` (wikitext-2 train,
held out from the fit rows) for `out`, the pre-registered deciding axis.
Bytes identical within every (unit, rung); the built arm repeated in place and
byte-identical in every block.

**Pinned wire ÷ built recipe** (geomean over 4 units; > 1 means the built
recipe wins; "n/4" = units where the pinned wire is worse):

| R | encode | `wt` | `h` | `out` |
|---|---|---|---|---|
| 5 | control (weights-only) | 1.0261 (4/4) | 1.0221 (3/4) | 1.0343 (3/4) |
| 5 | **production** | 1.0206 (3/4) | **0.9736** (3/4) | 1.0154 (2/4) |
| 6 | control | 1.0603 (4/4) | 1.0813 (3/4) | 1.0587 (3/4) |
| 6 | **production** | 1.0415 (3/4) | 0.9994 (3/4) | 1.0540 (3/4) |
| 7 | control | 1.0986 (4/4) | 1.1627 (3/4) | 1.2081 (4/4) |
| 7 | **production** | 1.0523 (3/4) | 1.1543 (3/4) | 1.0928 (3/4) |
| 8 | control | 1.2302 (4/4) | 1.3650 (3/4) | 1.3716 (3/4) |
| 8 | **production** | 1.1457 (3/4) | 1.3060 (3/4) | 1.2307 (3/4) |

**Where the optimum sits, as a multiple of the law's reach** (log-log parabola
through the best grid point and its neighbours; MATERIAL = beyond
`sqrt(4.5/4) = 1.0607`, the harness's pre-registered tolerance):

| R | encode | `wt` | `h` | `out` |
|---|---|---|---|---|
| 4 | control | 1.138 (MATERIAL) | 1.094 | 1.132 |
| 4 | **production** | 1.063 (MATERIAL) | 1.044 | **1.079 (MATERIAL)** |
| 5 | control | 1.062 (MATERIAL) | 1.088 | 1.110 |
| 5 | **production** | 0.969 | 1.002 | 0.981 |
| 6 | control | 1.011 | 1.085 | 1.087 |
| 6 | **production** | 0.930 (MATERIAL) | 0.988 | 0.931 |
| 7 | control | 0.985 | 1.123 | 1.093 |
| 7 | **production** | 0.853 (edge) | 1.028 | 0.973 |
| 8 | control | 1.020 | 1.102 | 1.102 |
| 8 | **production** | 0.903 (MATERIAL) | 1.042 | 1.015 |

(The control's `h`/`out` optima at R=5–8 are within tolerance by the harness's
rule but 1–3 of 4 units sit on the grid's upper edge, so those are lower
bounds.)

### What the control attributes

* **The objective flip is the encode's.** Same scorer, same grid, same units:
  the weights-only encode puts the `out` optimum **1.09–1.13× above** the law at
  every rung; the production encode puts it **0.93–1.08×** — the encode moves
  the optimum down by 0.87–0.93× at every rung. LDLQ + the Hessian refit want
  less reach than a plain nearest-value encode does. The law happens to sit
  between the two.
* **`wt` is not a quality axis under the production encode.** On `down_proj` at
  R=4 the production `wt` is 0.353 against 0.074 weights-only while `out`
  improves 3.1× (0.0251 → 0.0081): LDLQ spends weight error to buy output
  error. The `MATERIAL` flags on `wt` at R=6 and R=8 under production are on
  that axis and do not decide anything. `out` is the pre-registered decider,
  and on `out` the law is inside its own tolerance at R=5–8.
* **The R=4 calibration point does not hold under the production encode.** The
  amplitude was taken from "the rung where the pinned value is already
  optimal". That was a sqrt(2)-bracket finding: on the quarter-octave grid the
  R=4 `out` optimum is 1.132× the pinned reach under the weights-only encode
  (MATERIAL by the rule, 0 edge) and 1.079× under production (MATERIAL). The
  gain on offer at R=4 is 1.1–1.9% of `out` (`law·2^(1/4)` reads 0.9893
  production / 0.9811 control). Small, but it is the anchor of the whole rule.
* **The 4-of-4 is 3-of-4, and the fourth regresses.** Under production
  `2.mlp.down_proj` prefers the pinned reach (or less) at every rung on `out`:
  the law is **1.02× / 1.11× / 1.20× / 1.23× worse** than the pinned wire on
  that unit at R=5/6/7/8 (0.00538 vs 0.00527, 0.00361 vs 0.00326, 0.00224 vs
  0.00186, 0.00121 vs 0.00098). Its `out` optimum is 0.50–0.89× the law at
  every rung (reach ≈ 3.1–3.6 row-RMS). `q_proj` and `k_proj` want the
  opposite — 1.16–1.48× the law at R=7–8 (`q_proj` R=8: `law·2^(1/4)` is
  0.00455 vs the built 0.00589, 23% better). **The per-unit optima span
  0.5×–1.5× of the rule.** A scalar per rung is the average of units that
  disagree by 3×; the geomean win hides a 10–23% per-unit regression on one of
  four. Filed as issue #54.
* **What the control does *not* change.** The pinned wire loses at R=7 and R=8
  on both encodes on 3 of 4 units and on every axis; the direction of the term
  is right above the reference. What it changes is the confidence in the
  amplitude, the "interior optimum" wording, and the claim that the reference
  rung is calibrated where the measurement says it is optimal.

## Is this a wire change?

**No — and this is the load-bearing part.** The window table travels on the
ALPHABET plane and the reader takes it off the plane; it never rebuilds it
(`encode.window_table`'s docstring, `unit_artifact._read_window_unit`). So:

* **Format**: unchanged. `encoder_profile_id` binds `window_bits`, not the
  spread, because the table itself is covered by the payload digest.
* **Schema**: unchanged. `WireRecipe.to_config` already writes `sigma` and
  `from_config` already accepts a float or null, so no minor moves; every
  existing artifact stays readable, and an artifact written now is readable by
  any reader that already understands the window body. `wire.recipes` records
  the value per rung range, so a checkpoint replays at *its own* spread
  (`encode_settings_from_config`) rather than at whatever the module currently
  defaults to.
* **Merge guard**: covered. `wire.recipes` is compared as a table when every
  part carries it (`experiments/merge_tessera_parts.py:68`), and `body.sigma`
  is in both `SHARED` (`:52`) and `PROJECTED_BY_TABLE` (`:75`), so two parts
  built at different spreads cannot merge silently.
* **Runtime contract**: unchanged by this change, and **`contract_version` is
  7**, not 5 as the first cut of this receipt and commit `aa4a551` said (v6
  added `fused_module`, v7 `native_extensions`; both landed on master before
  this branch was rebased). Nothing in `src/tessera/serving/runtime_contract.json`
  mentions the spread (`grep -c sigma` = 0), the decodable rate range does not
  move, and no published rung changes. **But** the contract attests
  `TESSERA_BF16_K1` at `attested_rungs_q256: [1792]` — R=7 — on the strength of
  `tessera-bf16-route-served-2026-09-02.md`, which was served on the pinned
  wire. The *route* attestation stands (the decoder takes the table off the
  ALPHABET plane, so the bytes decode by the same kernel); the *KL receipt*
  behind it describes bytes a fresh export at 1792 no longer writes. The
  contract carries no encoder identity next to the rung, so it cannot say so
  itself. Filed as issue #55; not fixed here.

What *does* change is bytes, at BF16 rungs away from R=4. The byte audit was
extended to say so: `experiments/audit_byte_baseline.py` carried BF16 only at
q256 1024 — the reference rung — so it would have reported "0 changed" for a
change that moves every other rung. With a BF16 R=8 row added, the audit on the
rebased tree reads **2 changed of 55** (25 encodes = 20 shape + 5 value, 8
release rows, 22 decodes; the first cut of this receipt said "of 20", the count
of shape rows in the tree it was run on), both of them `bf16-2048-256c/{none,
scale}`, and every E2M1, E2M1x2, E4M3 and BF16-at-R=4 digest identical. The
second pass found the closure was one row with no guard and one rule it could
not see: `_reach_rate_for` rounds half up, and no row sat at an exact half
rate, so banker's rounding would have moved no digest. A `bf16-1152-256c` row
(R=4.5) was added: pre-#48 bytes vs now reads **4 changed of 57** (the 1152
and 2048 pairs), and a banker's-rounding mutation of `_reach_rate_for`
against the current baseline reads **2 changed of 57**, exactly the 1152
pair — the row sees the rule and nothing else does. Then
`tests/test_audit_byte_baseline.py::test_the_bf16_rows_reach_off_the_reference_rung`
now refuses a BF16 matrix with no off-reference row and no half-rate row;
`tests/test_bf16_route.py::test_the_reach_rate_rounds_half_up_and_is_bounded`
pins the rounding rule directly.

## What this does not claim

* **Not a serve.** Encode-side error metrics, four dense Linears of one small
  model (Qwen3-0.6B). No vLLM, no KL, no PPL. A BF16 lane exists and the A/B is
  runnable; it has not been run.
* **Not established under the production encode as stated above.** The
  "interior optimum on 4 of 4" and "calibrated where the pinned value is
  optimal" claims are weights-only findings. Under the encode an export runs
  the pinned wire loses on 3 of 4, `down_proj` regresses 2–23% on `out`, and
  the reference-rung optimum is 1.08× above the pinned reach. The law is within
  its own tolerance of the production `out` optimum at R=5–8 and not at R=4.
* **Not a claim that licenses a default change.** The term is on by default
  for BF16 above R=4 (`BF16_RECIPE.window_sigma = 1.0` plus the rule), which is
  what #48 asked to build; the *value* it carries is a screen on two encodes
  that disagree about where the optimum is by 0.87–0.93×, with one of four
  units regressing. Under principle 3 nothing here promotes; whether a
  screen-grade default is acceptable on a lane whose attested rung it moves is
  Rob's call, not this receipt's.
* **Not a claim about experts.** Everything here is dense Qwen. #18 found the
  `L` question answers *oppositely* on GLM experts, and the same could be true
  of reach; the reference rung's amplitude in particular is calibrated on rows
  whose peak-to-RMS is a dense-model property.
* **Not a claim that the floor is derived.** The clamp saturates below reach
  ≈3.5 on *these* rows, which covers R=1 and R=2; at R=3 it does not saturate
  and the law still loses, and the per-row decomposition says the loss is in
  the unclamped rows — a spread effect, not the mechanism the derivation
  prices. That boundary is in any
  case a property of the weights' peak-to-RMS, not of the rate, so the floor is
  placed at the rung the recipe is calibrated on because that is the one place
  both sides are measured.
* **Not a claim that the exponent is exact.** `sqrt(2 ln N)` is the *leading*
  term of the optimal loading factor; the classical result carries slowly
  varying corrections in `ln ln N / ln N`, which over R=4…8 would bend the
  curve slightly below a pure square root. The bracket here resolves a factor
  of √2 in reach, and a correction that small sits inside it. What is
  established is that the optimum moves with the rate, moves as a square root
  to within ±√2 at six rungs, and is not the constant it was pinned at.
* **Not a claim about `L`.** #48 offered to fold the per-rung `L` question in.
  It is not folded in: the measurement gives no rule for it — the dense optimum
  is unit-dependent (8, 10, 16, 16), the expert optimum is monotone-deeper, and
  `wt` is the wrong gate for it in both regimes. No `L` value moved.
* **Not a claim on `h`.** The rule is fitted to `wt`. `h` agrees on 3 of 4
  units and the wide bracket beats the law on `h` at R=7 and R=8; under the
  production encode the pinned wire is *better* on `h` at R=5 (0.9736) and
  level at R=6.
* **Not a claim about the rows' peak-to-RMS transferring.** The reach is a
  ratio to row RMS; the units here have 1024–3072 columns and clamp fractions
  0.23–0.55 at the pinned reach. GLM experts (#18) answered the `L` question
  oppositely; no reach measurement exists there, and the per-unit spread found
  here (0.5×–1.5× of the rule on four dense units) says a single rule is not
  even the right shape for dense Qwen.
* **Not a re-attestation.** The BF16 route's serving receipt (contract v5 when
  it was cut, now carried at v7; q256 1792, dense sm_121) was measured on the
  pinned wire. R=7 is a rung the recipe now moves, so that receipt's *bytes*
  are no longer what a fresh export at that rung produces. The route is
  unchanged; the artifact it was measured on is not the artifact a rebuild
  would write.

## The next gate

A served A/B at matched bytes in one vLLM session, on the BF16 lane
(`TESSERA_SERVE_MODE`, the route the contract attests at 1792), at R=7 — the
attested rung, where the production-encode `out` delta is 1.09× geomean with
one unit 1.20× the other way — against the same model exported at the pinned
spread (`window_sigma=BF16_CHANNEL_SIGMA` at every rung). Until that exists
this is a screen, and a screen is not a result. The served A/B also decides
whether the contract's 1792 attestation is re-cut on the new bytes or the term
is held off that rung.
