# The BF16 recipe's `(L, sigma)`, and where the 4-bit residual actually sits (2026-09-02)

Two issues, one session. **#18**: the BF16 route's `(L, sigma)` was stated
rather than searched, and its GLM expert evidence was one tensor. **#12**: the
dense 4-bit route loses to NVFP4 GPTQ+JSO at equal residency by 1.254x.

**Claim (#18, sigma).** `BF16_CHANNEL_SIGMA` is a **gauge, not a knob**. On four
dense Qwen Linears, dyadic shifts over a 16x range decode **bit-identically**;
the file hash moves and the tensor does not. There is no dyadic value left to
find. The identical shift on E4M3 costs +5 to +19% at x2 and +70 to +98% at x4.

**Claim (#18, L).** At 4 bpp the default `L = 14` is on both the `wt` and the
`h` frontier and nothing here argues with it. **At 8 bpp it is off the `h`
frontier in both axes**: `L = 10` is 10.8% better on `h` for 0.127 bpp *less*,
and L=12/14/16 are strictly dominated. Weight-space, four dense units,
proposal only -- there is no BF16 lane to serve on.

**Claim (#18, GLM).** The one-tensor result holds on six experts. BF16 at R=8
beats **EXL3 K=8** on the weight leg 6/6 (geomean `wt` **0.867x**) and is level
in out-space (**1.004x**), at 8.0352 bpp against 8.0117 -- nearly matched, not
matched. Against the E4M3 wire it is **4.23x** better for +0.0156 bpp.

**Claim (#12).** The 1.254x premise **does not reproduce**. It is **1.039x**
served on master, moved by the LDLQ-on-LUT work at byte-identical bytes -- and
*not* by the reach-aware per-row start, which is provably not on that path. The
residual is a **weight-leg** residual: on held-out rows the weight leg alone is
**1.035x**, against 1.039x served.

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

## #18, part 1: L is the axis, and at 8 bpp the default is off the h frontier

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

**On `wt` every L is on the frontier at both rungs; on `h` the R=8 frontier
stops at L=10.** L=12, 14 and 16 are strictly dominated there -- more bytes
*and* more H-weighted error. The default L=14 costs +0.127 bpp over L=10 and is
10.8% worse on `h`.

Which is the same inversion issue #18 already names: "an alphabet worth 4.3x on
plain error and nothing on the H-weighted columns is a reach problem, not an
alphabet one". Swept on L, the two axes disagree in sign at 8 bpp. `wt` improves
monotonically with L because a deeper table resolves the bulk more finely; `h`
gets worse because the extra depth buys reach the H-heavy columns do not want.

At 4 bpp the default is on both frontiers and nothing here argues with it.

**Proposal, not a flip.** This is weight space on four dense units. Principle 3
says a screen does not promote, and there is no BF16 lane to serve on, so
`BF16_WINDOW_BITS` stays at 14 and the finding is recorded. What it does say is
that whoever builds the BF16 lane should carry `L` as a per-rung choice rather
than one constant, and should gate it on `h` or a serve, never on `wt`.

## #18, part 1: reach is the mechanism, measured directly

`--stage reach` pins the table (`window_sigma = 1.0`) and moves only
`channel_sigma`, so the ratio -- the table's spread against the row's -- is the
sole variable. `model.layers.2.mlp.down_proj`, R=4:

| channel_sigma | bpp | wt | h | reach (row RMS) | rows over reach |
|---|---|---|---|---|---|
| 0.25 | 4.08854 | 0.15362 | 0.11940 | 16.00 | 0.4% |
| 0.5 | 4.08854 | 0.09108 | 0.07902 | 8.00 | 6.0% |
| 0.707 | 4.08854 | 0.07639 | 0.06630 | 5.66 | 14.3% |
| **1.0 (default)** | 4.08854 | **0.07373** | 0.05833 | 4.00 | 54.9% |
| 1.414 | 4.08854 | 0.07479 | **0.05754** | 2.83 | 100% |
| 2.0 | 4.08854 | 0.07479 | 0.05754 | 2.00 | 100% |

Three things worth keeping:

* **The `wt` optimum is at the default**, and it is a genuine interior optimum
  -- 2x either way costs 1.4% to 24%.
* **More than half the rows already exceed reach at the default**, and that is
  the *good* setting. Clipping the extremes to resolve the bulk is the right
  trade at 4 bits, which is why "rows over reach" is a diagnostic and not a
  defect.
* **`h` wants slightly more clipping than `wt` does** (1.414 beats 1.0 by 1.4%),
  the same direction as the L result: the H-weighted objective prefers
  resolution over range. The `1.414` and `2.0` arms produce **identical bytes**
  because every row is clipped and the reach-aware start lands them all at the
  same place -- the clamp saturating, not a coincidence.

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

| unit | C/A on `hq` | C/A on `out` |
|---|---|---|
| `0.self_attn.q_proj` | 1.0251 | 1.0395 |
| `1.self_attn.k_proj` | 1.1184 | 1.1346 |
| `13.mlp.down_proj` | 0.9190 | 0.9750 |
| `14.mlp.gate_proj` | 0.9633 | 0.9874 |
| `2.mlp.down_proj` | 1.2858 | 1.2740 |
| `27.self_attn.o_proj` | 0.7627 | 0.8511 |
| **geomean** | **0.9992** | **1.0353** |

**The census metric flatters the arm fit to it by 3.6%**, on five of six units,
and the sixth agrees within 1%. Every `hq` number the census produces has to
carry that discount before it is read as a statement about held-out behaviour.
All six repeat controls were bit-exact.

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
loses `2.mlp.down_proj` by 27%. If that concentrates by role, a per-role format
choice is worth more than any further encoder work -- which is the argument
PrismaQuant's allocator makes, and it is measurable here.

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

* Any change to `BF16_WINDOW_BITS`, or a per-rung `L`, needs a served A/B at
  matched bytes in **one** vLLM session against the current default -- KL drifts
  4-8x across sessions.
* Whether the R=8 `h`-frontier result transfers from dense Qwen to GLM experts
  (`--stage glm-l`).
* The full 196-Linear per-role decomposition of the 4-bit residual, and whether
  the 3.6% in-sample discount measured on six units holds across all of them.
