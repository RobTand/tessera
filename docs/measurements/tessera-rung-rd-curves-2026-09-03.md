# The rung rate-distortion curve, and the four costs that try to rank it

*2026-09-03. Tessera worktree `/home/rob/tmp/ts-rung-rd` on branch
`claude/rung-rd-curves`, parent `780817a`. Model Qwen3-0.6B. The allocation
under audit is PrismaQuant worktree `/home/rob/pq-wt/tessera-continuous`
@ `5a320c7`, priced at `tessera_commit 3d419e7`. Harness:
`experiments/rung_rd_curves_2026-09-03/`.*

Issue #4 asks for one thing: a cost that ranks **rungs of one format at matched
bytes**. Issue #1 showed the shipped additive-Fisher currency does not — an
allocated Tessera checkpoint served 2.00x worse than a byte-matched uniform arm
at 4.0 bpp. This receipt measures the curve that cost was trying to approximate,
and scores the four candidates #4 names against it.

**The headline is not "one of them wins". It is that at 4.0 bpp there is almost
nothing to win.** Over the seven Linears the campaign priced, the best possible
byte-matched re-allocation — an oracle given the measured table itself — reads
**0.941x** the uniform control with **P(worse) = 0.075**: not significant at 95%.
The shipped cost, on the same axis, lost **1.970x**. So the currency's problem at
this budget is not that it ranks a real ordering badly; it is that it *invents*
an ordering where the achievable spread is instrument noise, and then spends
real bytes on it.

**The mechanism is a plateau the surrogate cannot see.** Taking `q_proj`,
`k_proj` or `gate_proj` from R1006 to **BF16** — deleting their weight
quantization entirely — changes end KL by `+4.2e-4 ± 4.1`, `+2.4e-4 ± 5.1`,
`+1.1e-4 ± 3.2`. Zero, or slightly negative. Over that same interval
`output_mse` keeps falling ~2.6x per bit and the cost keeps offering to buy it.

**Below the knee the axis is real and the costs work.** At a 3.0-bpp budget the
oracle is **0.748x** and AURA at its production settings reaches **0.780x**,
both at P = 0.000. The rung axis is worth allocating over where the curves are
steep; at 4.0 bpp on these units it is flat, and a cost that does not know that
is strictly a liability.

Nothing here changes a default. No format, lever or lane was promoted.

---

## 1. The seven units, recovered not guessed

`/mnt/shared/tessera-runs/pq-continuous/qwen06b/alloc/lc_full_4.0.json` — the
allocator's own layer config for the 4.0-bpp arm that §7 of
`tessera-allocated-served-2026-09-02.md` served — names them directly:

| unit | rung in the 4.0 arm | `h_trace` (probe.pkl) | wire bytes @R1006 |
|---|---|---|---|
| `model.layers.0.self_attn.q_proj` | R1083 | 1505.7 | 1050624 |
| `model.layers.0.self_attn.k_proj` | R1083 | 1919.6 | 533504 |
| `model.layers.0.self_attn.v_proj` | R1083 | 38309.0 | 533504 |
| `model.layers.0.self_attn.o_proj` | R934 | 13653.0 | 1048576 |
| `model.layers.0.mlp.gate_proj` | R1107 | 2758.2 | 1567744 |
| `model.layers.0.mlp.up_proj` | R1107 | 8730.5 | 1567744 |
| `model.layers.0.mlp.down_proj` | R749 | 8480.3 | 1563648 |

All seven are `TESSERA_E4M3_K1` — the E4M3 window body over a CHANNEL scale
plane, decoded to a per-channel FP8 pair and served W8A8. The exact seven were
recovered; no substitution was needed.

The geometry is the receipt's own separator: **one unit moves, the other six hold
at R1006, every other body Linear stays BF16 in both arms.** That is what
isolates the cost model from the broadcast-to-28-layers coverage assumption.

## 2. The proxy, and its measured agreement with the serve

Serving ~130 arms is not affordable. The quantity under test is local — the
seven modules' arithmetic — so the harness runs exactly that arithmetic
in-process against a local BF16 HF teacher:

* weight side: `export.encode_linear_planes` at HEAD, then `decode.materialize_fp8`
  — the same pair `serving/fp8_route.py` hands the GEMM;
* activation side: the route's declared contract `fp8_per_token_dynamic` —
  per-token absmax / 448 with vLLM's `min_scaling_factor = 1/(448·512)` floor,
  saturating cast to E4M3;
* GEMM: `torch._scaled_mm`, rowwise A and B scales, bf16 out — the call
  `TesseraFp8LinearMethod.apply` makes.

**Encoder identity is a gate, not an assumption.** `encoder_identity.py`
re-encodes all 17 `.tessera` blobs in the campaign's `verify_cache/wire` at HEAD:
container bytes differ by 14–18 B (metadata growth over 153 commits), but
**packed planes and decoded FP8 tiles are identical for all 17**. The wire this
harness builds is the wire that was priced and served.

**Agreement, measured on the four arms the receipt served:**

| arm | served (top-1024 KL) | proxy `kl_top1024` | proxy `kl_full` | served top-1 | proxy top-1 |
|---|---|---|---|---|---|
| uniform R1006 | 0.013064 | 0.013344 (+2.1%) | 0.013727 | 93.93% | 93.93% |
| the allocation | 0.025170 | 0.026352 (+4.7%) | 0.027044 | 91.61% | 91.49% |
| allocation, `down_proj` restored | 0.015541 | 0.015609 (+0.4%) | 0.016030 | — | 93.32% |
| uniform, `down_proj` alone cut to R749 | 0.022609 | 0.022290 (−1.4%) | 0.022875 | — | 92.39% |

Ratios against the uniform control — the quantity every conclusion below is
stated in — reproduce to 2.5–3.7%: served **1.927 / 1.190 / 1.731**, proxy
(`kl_full`) **1.970 / 1.168 / 1.666**. The HF-vs-vLLM difference cancels because
both arms of every comparison share the same teacher.

**One honest gap.** The serve's two-bundle decomposition summed to 99.3% of the
jointly measured allocation damage; this proxy's sums to **84.8%** (and it
attributes 69% of the damage to `down_proj` where the serve attributed 79%). The
proxy is slightly *more* super-additive than the serve. It changes no ordering —
the 1.97x reproduces — but it is why **every offline pick below was rebuilt and
measured as one joint arm rather than summed**.

## 3. The rate-distortion table

Measured ΔKL vs the uniform-R1006 arm, in units of 1e-4, `kl_full`, 4088 scored
positions, paired-bootstrap SE (2000 resamples) underneath. Metric: exact
full-vocabulary KL(BF16 teacher ‖ student). Weights-only: no Hessian, no LDLQ,
exporter-default `scale_refit`, which is how the audited blobs were built.

| unit | R320 | R512 | R640 | R749 | R826 | R900 | R934 | R970 | R1006 | R1044 | R1083 | R1107 | R1150 | R1200 | R1262 | R1340 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `q_proj` | +479.3 | +98.6 | +35.6 | +26.4 | +29.2 | +21.4 | +16.8 | +4.7 | 0 | +5.0 | −1.8 | +0.2 | +7.0 | +3.3 | −2.3 | −0.7 |
| ±SE | 34.7 | 26.6 | 5.5 | 3.6 | 5.1 | 6.6 | 5.4 | 3.3 | — | 2.8 | 2.8 | 3.4 | 3.6 | 3.7 | 4.0 | 3.7 |
| `k_proj` | +319.9 | +90.5 | +30.6 | +22.6 | +11.8 | +6.9 | +10.8 | +3.8 | 0 | +2.6 | −3.6 | +1.9 | +2.4 | +8.5 | −4.0 | −3.1 |
| ±SE | 27.6 | 8.8 | 5.2 | 4.5 | 4.3 | 4.2 | 3.3 | 3.5 | — | 3.0 | 2.0 | 2.4 | 2.1 | 3.9 | 3.3 | 3.1 |
| `v_proj` | +1809.2 | +420.2 | +228.4 | +95.5 | +60.0 | +27.2 | +20.1 | +10.5 | 0 | +0.7 | −9.8 | −12.3 | −1.8 | −18.8 | **−31.4** | −27.2 |
| ±SE | 62.3 | 23.9 | 22.8 | 8.7 | 8.2 | 5.8 | 6.3 | 3.9 | — | 4.2 | 3.7 | 4.4 | 4.8 | 4.7 | 5.5 | 5.2 |
| `o_proj` | +2057.8 | +526.2 | +330.4 | +87.8 | +45.0 | +34.7 | +36.8 | +30.5 | 0 | −4.9 | +5.5 | −8.3 | −12.5 | −10.2 | −13.5 | **−22.4** |
| ±SE | 58.2 | 26.5 | 27.4 | 6.0 | 6.1 | 4.7 | 7.3 | 5.5 | — | 3.2 | 4.6 | 4.3 | 3.1 | 3.4 | 4.8 | 3.5 |
| `gate_proj` | +512.7 | +126.1 | +34.1 | +28.5 | +18.1 | +14.6 | +12.8 | +7.6 | 0 | +6.7 | +0.1 | +9.4 | +1.9 | +3.0 | +1.7 | +2.6 |
| ±SE | 32.7 | 14.7 | 6.7 | 7.1 | 8.3 | 3.7 | 3.1 | 2.7 | — | 3.7 | 2.7 | 3.4 | 2.4 | 2.7 | 3.4 | 3.0 |
| `up_proj` | +380.8 | +90.2 | +46.0 | +21.2 | +18.5 | +3.8 | +4.2 | +1.4 | 0 | −4.5 | −9.5 | −9.6 | −6.0 | −14.1 | −14.8 | **−17.9** |
| ±SE | 20.7 | 8.3 | 5.5 | 3.6 | 4.4 | 2.5 | 2.4 | 2.8 | — | 1.9 | 2.7 | 3.1 | 3.4 | 2.5 | 2.6 | 3.2 |
| `down_proj` | +683.7 | +231.3 | +144.0 | +91.5 | +49.0 | +44.4 | +13.5 | +14.4 | 0 | +2.0 | +2.7 | −0.6 | −0.8 | −4.5 | −5.1 | **−10.3** |
| ±SE | 28.4 | 18.5 | 7.9 | 9.5 | 4.7 | 6.1 | 3.3 | 2.6 | — | 2.8 | 2.1 | 2.5 | 2.3 | 2.0 | 2.6 | 2.5 |

Baseline (all seven at R1006): `kl_full` **0.013727**, top-1 93.93%.
Two floors on the same arms: **W16A8** (weights unquantized, the route's A-side
only) **0.002370**; **W8A8 per-channel FP8 RTN** **0.003413**.

Below R1006 every curve is steep, smooth and monotone within noise. Above R1006,
`q/k/gate` never clear 2 SE in either direction, and only `v/o/up/down` retain a
real gain by R1200–R1340.

## 4. Why: the headroom arms

Fourteen arms, one per unit per bound, other six at R1006. The R1006→BF16 column
is *everything* any finer rung could ever buy for that unit; the FP8-RTN column
is the E4M3 wire's own rate-to-infinity asymptote. ΔKL in 1e-4, same bootstrap.

| unit | →FP8 RTN | ±SE | →BF16 | ±SE | best measured rung ≤R1340 |
|---|---|---|---|---|---|
| `q_proj` | +4.0 | 3.9 | **+4.2** | 4.1 | −2.3 (R1262, 0.6 SE) |
| `k_proj` | −0.3 | 3.0 | **+2.4** | 5.1 | −4.0 (R1262, 1.2 SE) |
| `v_proj` | −32.0 | 5.4 | **−38.5** | 5.6 | −31.4 (R1262) |
| `o_proj` | −27.1 | 3.6 | **−28.7** | 3.0 | −22.4 (R1340) |
| `gate_proj` | +4.2 | 2.6 | **+1.1** | 3.2 | −0.0 |
| `up_proj` | −18.7 | 4.1 | **−18.9** | 3.4 | −17.9 (R1340) |
| `down_proj` | −20.1 | 4.9 | **−18.5** | 4.1 | −10.3 (R1340) |

All seven at BF16 reads −113.6e-4; all seven at FP8 RTN −103.1e-4. The four
units with real headroom sum to −104.5e-4, 92% of the whole-arm figure.

**Three of the seven units have zero headroom at R1006** — and `v/o/up` are
already within 8–20% of their own ceiling by R1262–R1340. Any cost built from
weight or output error will keep pricing gains there, because the error it
measures really does keep falling; what stops falling is the model's loss.

## 5. The four candidates

Every pick below was **rebuilt as one joint arm and measured**, not summed. The
knapsack solves over the same fused serving units the allocator does
(q/k/v one rung, gate/up one rung) at a byte floor matching
`assert_byte_matched`'s 0.1% slack. Three budgets: the uniform arms at R749,
R1006 (the 4.0-bpp point) and R1262.

### 5.1 The 4.0-bpp budget (uniform R1006 = 0.013727, 7865344 B)

| arm | `kl_full` | vs uniform | ΔKL ±SE (1e-4) | P(worse) |
|---|---|---|---|---|
| **oracle on the measured table** | 0.012922 | **0.941x** | −8.1 ± 5.3, CI95 [−18.0, +2.8] | 0.075 |
| **c1 AURA**, production dtype (fp32), 16 or 64 probes | 0.012999 | **0.947x** | −7.3 ± 6.3, CI95 [−19.8, +4.9] | 0.117 |
| c2 empirical, 9 anchors + log-interp | 0.013593 | 0.990x | — | 0.408 |
| §3a best per-unit rescale of L1 | 0.013599 | 0.991x | — | 0.439 |
| c2 empirical, 5 anchors + log-interp | 0.013703 | 0.998x | — | 0.506 |
| c1 AURA at bf16/16 probes (**my deviation**, see §5.4) | 0.019031 | 1.386x | — | 1.000 |
| **c4 L1 re-measured at 1×/10×/80× activation rows — identical pick** | 0.021466 | 1.564x | — | 1.000 |
| **L1 as shipped = the served allocation** | 0.027044 | **1.970x** | +133.2 ± 16.8 | 1.000 |

### 5.2 The 3.0-bpp budget (uniform R749 = 0.051095) — the axis is real here

| arm | `kl_full` | vs uniform | P(worse) |
|---|---|---|---|
| oracle | 0.038225 | **0.748x** | 0.000 |
| c2 empirical, 5 anchors | 0.039387 | 0.771x | 0.000 |
| c1 AURA (fp32, 64 probes) | 0.039835 | **0.780x** | 0.000 |
| c1 AURA (bf16, 16 probes) | 0.040076 | 0.784x | 0.000 |
| L1 as shipped | 0.099684 | 1.951x | 1.000 |

### 5.3 The 5.0-bpp budget (uniform R1262 = 0.006367) — nothing to win, and they lose

| arm | `kl_full` | vs uniform | P(worse) |
|---|---|---|---|
| oracle | = uniform | 1.000 | — |
| c1 AURA (fp32, 64 probes) | = uniform | 1.000 | — |
| c2 empirical, 5 anchors | 0.007235 | 1.136x | 0.999 |
| c2 empirical, 7 anchors | 0.008358 | 1.313x | 1.000 |
| c1 AURA (bf16, 16 probes) | 0.008771 | 1.378x | 0.999 |

### 5.4 Candidate by candidate

**c4 — more samples at the top of the curve. FAILS. Cheapest, ran first, and it
is not resolution.** The reproduction of the campaign's currency is exact
(`down_proj` R826: 0.000275226 here vs 0.000275242 in `cost.pkl`). Seed spread
at the shipped 4 samples / 256 rows is max/min **1.02–1.16**, an order below the
2.2x effect. At **40 sequences / 20480 rows — 80x the activation rows** — the
knapsack pick is **byte-identical to the 4/256 pick** and measures **1.564x
worse than uniform**. Resolution is excluded as the explanation.

**c1 — AURA / KL-adjoint. The only candidate that reaches the oracle, and only
where the oracle has something to give.** At its production settings
(`aura_cost.py --dtype` default `float32`, `n_probes` 16 or 64, `token_scope=all`,
`seed_base=7000`, calib n=4 × 256 train) it measures **0.947x at 4.0 bpp
(P = 0.117, indistinguishable from uniform, and from the oracle's 0.941x)** and
**0.780x at 3.0 bpp (P = 0.000)**. It also signs the shipped allocation
correctly where L1 does not (§5.5).

*The deviation, recorded.* My first AURA run resolved the model in **bf16**;
`aura_cost.py` defaults to fp32. The advisor flagged it. Correcting the dtype
moved the 4.0-bpp result from **1.386x (a significant loss)** to **0.947x
(parity)**; probe count 16→64 changed nothing at that budget but fixed the
5.0-bpp budget (1.511x additive → picks uniform). **AURA on seven units is
dtype- and probe-sensitive**; anyone re-running it should use fp32 and ≥64
probes, and should not read a 16-probe bf16 result as AURA's.

**c2 — empirical unit-KL per rung. Scored honestly, it is parity, not a fix.**
Scored against itself the empirical table is exact and measures nothing, so the
shippable form was built: a few measured anchors plus the campaign's own
log-linear interpolant. At 4.0 bpp the 5- and 9-anchor surfaces measure **0.998x
and 0.990x** (P ≈ 0.4–0.5) — parity with uniform, which is all the budget has to
give. At 3.0 bpp the 5-anchor surface is **0.771x**, near-oracle. At 5.0 bpp it
**loses, 1.136x and 1.313x**: interpolating log-KL across the plateau inherits
exactly the slope error it was meant to replace. An empirical surface needs
dense anchors *above* the knee, where measurement is hardest and the signal is
smallest.

**c3 — refit-aware pricing. The premise does not hold for this run, and the
correlation is absent.** `tessera-allocated-served-2026-09-02.md` proved the
priced blobs and the served bytes were the same objects, both the refit render;
there is no un-refit/refit mismatch to blame. Measured anyway
(`refit_tracking.py`, a full `--scale-refit 0` re-sweep at 6 rungs):

* pricing on the **un-refit** render picks the **identical assignment** as the
  refit render (1.487x additive either way) — the refit gain in `output_mse` is
  1.01–1.23x and nearly rung-independent, so it cancels in the marginal ranking;
* the L1 per-rung residual **does not track** refit gain: Spearman **−0.20**,
  Pearson −0.08, n=21 — if anything the sign is backwards;
* curve *shape* survives the refit (per-unit Spearman on-vs-off 0.60–1.00 for
  six units; `down_proj` 0.09, which is its above-R1006 noise, not the refit).

**Not measured, named as untested:** LDLQ and the exact full-H refit change the
encode itself, so the rung curve under them would have to be re-swept. This
receipt says nothing about that half of candidate 3.

### 5.5 The exchange table, and the named inversions

What each cost thought the shipped allocation's seven moves were worth, against
what they measured (ΔKL in 1e-4, single-unit):

| unit | move | Δbytes | ΔL1 (shipped currency) | ΔAURA (fp32, 64) | measured ΔKL |
|---|---|---|---|---|---|
| `q_proj` | R1006→R1083 | +78848 | −6.52e−2 | −1.71e−4 | −1.8 ± 2.8 |
| `k_proj` | R1006→R1083 | +39424 | −1.28e−1 | −1.18e−4 | −3.6 ± 2.0 |
| `v_proj` | R1006→R1083 | +39424 | −1.01e+0 | −3.43e−3 | −9.8 ± 3.7 |
| `o_proj` | R1006→R934 | −73728 | +4.12e−1 | +5.47e−4 | +36.8 ± 7.3 |
| `gate_proj` | R1006→R1107 | +155136 | −1.13e+0 | −2.98e−4 | +9.4 ± 3.4 |
| `up_proj` | R1006→R1107 | +155136 | −1.30e+0 | −6.19e−4 | −9.6 ± 3.1 |
| `down_proj` | R1006→R749 | −394752 | +1.22e+0 | +6.50e−3 | +91.5 ± 9.5 |
| **total** | | **−512** | **−2.00 (a net gain)** | **+2.41e−3 (a net loss)** | **+112.9 additive / +133.2 joint** |

L1 signs the whole allocation as a **gain**; AURA at production settings signs it
as a **loss**. That single sign is the difference between the two candidates.

Specific inversions — steps a candidate prices as a gain that measured flat or
worse. Every one is above R1006:

* **L1 as shipped** (26 inversions): `gate_proj` R1083→R1107 priced +2.28e−1 of
  gain, measured **+9.3e−4 (2.8 SE, a loss)**; `o_proj` R1044→R1083 priced
  +1.55e−1, measured **+10.4e−4 (2.3 SE)**; `v_proj` R1107→R1150 priced +4.00e−1,
  measured **+10.4e−4 (2.2 SE)**.
* **L1 at 80x the rows** (24): the same three steps, same sign, smaller
  magnitudes — resolution moves the number, not the verdict.
* **AURA fp32/64** (19): `v_proj` R1107→R1150, `gate_proj` R1083→R1107 and
  R1006→R1044, `k_proj` R1083→R1107 and R1150→R1200.
* **empirical 5-anchor** (16): `v_proj` R1107→R1150 and R1262→R1340,
  `o_proj` R1044→R1083, `gate_proj`/`q_proj` R1006→R1044.

**On rank correlation.** Within-unit Spearman is vacuous here — every candidate
is monotone in rate and so is the truth below the knee, so all seven units score
0.9+. Pooled *marginal-KL-per-byte* Spearman is the honest version, and it is a
weak discriminator: L1 0.529–0.553, AURA 0.487–0.511, empirical-interp
0.688–0.790. AURA scores *below* L1 on it and re-allocates far better, because
what matters is not the ranking of all steps but the sign of the few big ones.
Read the re-allocation tables, not the correlation.

**§3a — weights or curves?** Re-fitting one log-scalar per unit to the L1 curves
(i.e. allowing `h_trace` to be arbitrarily wrong while keeping the curve shape)
reaches 1.029/1.051 additive and **0.991–0.998 measured, CI straddling zero**. It
recovers parity, not a win. The exchange rate is wrong *and* the curve shape is
wrong; fixing only the first buys nothing.

## 6. Honest negatives

* **No candidate orders the seven units' rungs correctly at 4.0 bpp**, and the
  reason is not that the candidates are bad — it is that above R1006 there is no
  ordering to recover. The oracle itself is not significant (P = 0.075).
* **Candidate 4 does not fix it.** Stated plainly, with an 80x-rows sweep as
  evidence, and it produced a byte-identical pick.
* **Candidate 3's premise does not describe this run**, and the correlation it
  proposes is absent (Spearman −0.20).
* **Candidate 2 is not free.** Its interpolated form loses at 5.0 bpp.
* No fifth idea was invented to rescue the result.
* The proxy's additivity (84.8%) is worse than the serve's (99.3%), and it
  under-attributes `down_proj` (69% vs 79%). Every headline number above is a
  jointly measured arm, not a sum, for exactly this reason.

## 7. What this does and does not license

**Does not:** promote anything. No default was changed, no lever was wired, no
artifact was exported or served. AURA's 0.947x / 0.780x are proxy numbers on
seven layer-0 Linears of one 0.6B model — a *screen*, not a serving result
(principle 3). A served A/B is the gate, and the promotion decision is Rob's.

**Does say, for issue #4's closure criterion:** "BEAT uniform at 4.0 bpp" may
not be achievable on these units at all, because the measured ceiling for
byte-matched rung re-allocation there is 0.941x with P(worse) = 0.075. The
defensible closure at that budget is the *gate* — refuse to ship an allocation
that cannot beat its byte-matched uniform control — not a better cost. Where
the axis has slope (3.0 bpp), AURA is the candidate worth a served test.

## 8. Scope and provenance

* **Model** `/home/rob/models/Qwen3-0.6B`, layer 0 only, seven Linears; all other
  body Linears BF16 in every arm.
* **Format** `TESSERA_E4M3_K1` only — E4M3 grid, window body (`span 1`,
  `window_bits 14`), CHANNEL scale plane, served W8A8. Nothing here is measured
  on the E2M1/E2M1x2 4-bit lane, on MoE, or on a dense model other than Qwen.
* **Encode** weights-only: no Hessian, no LDLQ, exporter-default `scale_refit`
  (a `--scale-refit 0` counterfactual sweep exists for §5.4-c3 only).
* **Corpus** `/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json`,
  `contract_sha256 cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d`,
  8 × 512, **4088 scored positions**, Qwen tokenizer identity
  `76f13c8e…`. One corpus; no held-out second draw.
* **Statistics** paired bootstrap over the 4088 per-position KLs, 2000 resamples,
  common random numbers across arms.
* **Cost inputs** `h_trace` from the campaign's `probe.pkl`; the shipped
  `output_mse` from its `cost.pkl` (`nsamples 4, seqlen 512, max_act_rows 256,
  layer_stride 28`); re-measurements reproduce
  `production_weight_cache._local_forward_render_score` on wikitext-2-raw-v1
  **train** windows, the campaign's own calibration.
* **No performance claim is made**, so no before/after profile is attached
  (principle 15 applies to cost/speed claims; every claim here is a KL claim).

Artifacts: `experiments/rung_rd_curves_2026-09-03/{encoder_identity,rd_table,
output_mse_resolution,aura_rungs,score_candidates,refit_tracking,verify_picks}.py`
and `experiments/rung_rd_curves_2026-09-03/results/*.json`.

## 9. Consultations

Two `advisor` calls, both acted on and both verified against code or data before
use. The first, before any substantive work: gave the encoder-identity gate, the
A-side emulation constants, the correction that per-unit Spearman is vacuous
(score pooled marginal-per-byte and knapsack re-allocation instead), the §3a
ordering, the candidate-4 pre-screen, the candidate-3 premise caveat, and the
instruction to read `aura_cost.py` rather than reconstruct AURA from memory. The
second, before writing this receipt: the per-unit headroom arms of §4 (which
turned the plateau from an observation into a measurement), the AURA dtype check
that reversed §5.4-c1, and the framing of §6. No Fable consultation was needed.
