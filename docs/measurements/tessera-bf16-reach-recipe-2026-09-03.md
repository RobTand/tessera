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

**Claim (the law predicts).** R=5, 6 and 7 were never fitted. At each the law's
value is the **interior optimum on `wt`, on 4 of 4 dense Qwen Linears** — both
brackets (the law halved and doubled in reach) lose. Against the pinned wire the
built recipe is **0.975× / 0.943× / 0.910× / 0.813× on `wt` at R=5/6/7/8**, on
4 of 4, at identical bytes; **0.979× / 0.930× / 0.864× / 0.741× on `h`**, on
3 of 4. The R=8 number is #18's `0.8127×`, reproduced through the built path
rather than through the proxy that found it.

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
and the law loses by the most of the three. There the honest statement is that
the optimum does not fall below the reference reach on these rows, with no
mechanism established. Either way the recipe **floors at the reference rung**,
and the floored recipe writes the pinned wire's bytes at R ≤ 4, verified unit
for unit.

**This is a weight-space screen on four dense Linears. It is not a serve.**
There is no BF16 lane serving today; nothing here is promotable under principle
3. The next gate is stated at the bottom.

Harness: `experiments/bf16_reach_recipe.py`.
Raw: `/mnt/shared/tessera-runs/bf16/qsweep/reach_recipe.json` (R=2,4,6,8) and
`reach_recipe_odd.json` (R=1,3,5,7).
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
  part carries it, and `body.sigma` is in `PROJECTED_BY_TABLE`, so two parts
  built at different spreads cannot merge silently.
* **Runtime contract**: unchanged, `contract_version` stays 5. Nothing in
  `src/tessera/serving/runtime_contract.json` mentions the spread, the
  decodable rate range does not move, and no published rung changes. The
  serving routes take the table off the unit like any other reader.

What *does* change is bytes, at BF16 rungs away from R=4. The byte audit was
extended to say so: `experiments/audit_byte_baseline.py` carried BF16 only at
q256 1024 — the reference rung — so it would have reported "0 changed" for a
change that moves every other rung. With a BF16 R=8 row added, the audit reads
**2 changed of 20**, both of them that row, and every E2M1, E2M1x2, E4M3 and
BF16-at-R=4 digest identical.

## What this does not claim

* **Not a serve.** Weight space, four dense Linears of one small model. No
  vLLM, no KL, no PPL. There is no BF16 lane to serve on.
* **Not a claim about experts.** Everything here is dense Qwen. #18 found the
  `L` question answers *oppositely* on GLM experts, and the same could be true
  of reach; the reference rung's amplitude in particular is calibrated on rows
  whose peak-to-RMS is a dense-model property.
* **Not a claim that the floor is derived.** The clamp saturates below reach
  ≈3.5 on *these* rows, which covers R=1 and R=2; at R=3 it does not saturate
  and the law still loses, with no mechanism offered. That boundary is in any
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
  units and the wide bracket beats the law on `h` at R=7 and R=8.
* **Not a re-attestation.** The BF16 route's serving receipt (contract v5,
  q256 1792, dense sm_121) was measured on the pinned wire. R=7 is a rung the
  recipe now moves, so that receipt's *bytes* are no longer what a fresh export
  at that rung produces. The route is unchanged; the artifact it was measured
  on is not the artifact a rebuild would write.

## The next gate

A served A/B at matched bytes in one vLLM session, on a BF16 lane, at a rung
above the reference — R=7 or R=8, where the weight-space delta is largest —
against the same units encoded at the pinned spread. Until that exists this is
a screen, and a screen is not a result.
