# The activation-aware Tessera encoder: built, measured, and it is not the gap

**Measured 2026-09-01.** `experiments/tessera_compensated_glm_experts.py`,
`experiments/compensation_block_and_coder.py`,
`experiments/tessera_lever_cross_glm_experts.py`,
`experiments/tessera_scale_headroom_glm_experts.py`, and
`experiments/exl3_rate_sweep.py`.  Mechanism in `src/tessera/compensate.py`,
tests in `tests/test_compensate.py`.  Six GLM-5.3-Flash routed-expert
projections (layers 5/20/42, gate and up, expert 0), real cached activations,
tokens split fit/eval, **every arm scored on the disjoint eval half**.  The
Hessian is built from the fit half only.

`exl3-head-to-head-2026-09-01.md` ended with an order: build the encoder, then
re-run the harness, because activation-awareness was the one identified,
quantified deficit and it needed no wire change.  That was done.  **The encoder
works, the response is real, and it is not where the gap lives.**

---

## What was built, and why not the other thing

**Importance weighting is a provable no-op on this encoder**, so a probe built
on it would have returned a meaningless null.  Tessera's trellis runs *down a
column* — over output features — and treats columns (input features) as
independent.  Every position in one column therefore shares one weight
`H[j,j]`, `viterbi_columns` keeps `cost` as `[cols, states]` and takes a
per-column `min`, and `picked`/`choice`/the completion `argmin`/the release
order are all per-column or per-position.  A positive per-column scalar leaves
every one of those argmins unchanged: the anchors, the completion bits, the
scale bytes and the release order come out **bit-identical**.  Structural, not
measured.

So what was built is **LDLQ** — EXL3's own scheduling.  Block-LDL of the
regularised Hessian, input-feature blocks quantised last to first, each block's
residual pushed into the blocks not yet quantised.  It is the same family as
GPTQ, which is what PrismaQuant's NVFP4 render uses.

**It needs no wire change and no encoder change.**  Tessera's columns are
independent inside the Viterbi and its scale groups are within-row spans, so an
aligned column slice encodes **bit-identically** to the corresponding span of a
whole-matrix encode — tested at arity 1, arity 2, and under `R_IN_ONLY`
rotation.  Compensation is therefore preprocessing: it computes a modified
target and the ordinary encoder runs on it.  Every compensated arm asserts that
its stitched reconstruction re-encodes from its own returned target, so what
was scored is what an exporter would build.

---

## Result 1 — the response is real and it is 1.088×

| arm | bpp | rel_err | vs baseline |
|---|---:|---:|---:|
| plain (the head-to-head baseline) | 4.0000 | **0.09738** | 1.000× |
| plain + LDLQ | 4.0000 | 0.08952 | **1.088×** |
| `R_IN_ONLY` rotation | 4.0000 | 0.09862 | 0.987× |
| rotation + LDLQ | 4.0000 | 0.09056 | 1.075× |

The baseline arm reproduces 0.09738 exactly, which is the harness check.

Acceptance, fixed before the run, was **~0.077** — the 1.258× that PrismaQuant's
NVFP4 render gains from GPTQ+JSO.  1.088× is materially short of it, and the
head-to-head named that branch: *the trellis responds worse than a scalar format,
the coder question reopens.*  **Two controls exclude that reading.**

## Result 2 — both controls pass, and the trellis is not the problem

Same LDLQ loop, same Hessian, five block sizes, two coders — one substitution:

| LDL block | Tessera | gain | NVFP4 RTN | gain | **gain ratio** |
|---:|---:|---:|---:|---:|---:|
| none | 0.09738 | 1.000× | 0.08294 | 1.000× | — |
| 32 | 0.09071 | 1.074× | 0.07709 | 1.076× | 1.002× |
| 64 | 0.09076 | 1.073× | 0.07718 | 1.075× | 1.002× |
| 128 | 0.09119 | 1.068× | 0.07782 | 1.066× | 0.998× |
| 256 | 0.08952 | 1.088× | 0.07654 | 1.084× | 0.996× |
| 1024 | 0.08931 | 1.090× | 0.07619 | 1.089× | 0.998× |

**The gain ratio is 1.00× at every block size.**  A trellis coder and a scalar
coder respond to error feedback identically on these weights.  The "responds
worse" branch is closed.

**Block size is not the limit either.**  The spread is 1.068–1.090 and it is
*non-monotonic* — coarser blocks are very slightly better — so EXL3's block of
16 would not have helped.  Both coders show the same non-monotonicity, which
points at something they share: compensation inflates a group's `amax`, and
both formats set a per-group scale from it.

**What the controls do show** is that this LDLQ realises **1.076×** where
PrismaQuant's GPTQ+JSO gets **1.258×** on the same coder, the same weights and
the same activations.  The difference is not error feedback — it is
`static_act_order` and JSO's in-loop scale search.  Both are portable to
Tessera and neither touches the wire.

## Result 3 — every cheap lever Tessera already owns, crossed

`d` = rank-1 diagonals (segment 2a, EXL3's `suh`/`svh` by construction,
`+0.0117` bpp).  `r` = `R_IN_ONLY` rotation.  `q` = LDLQ at block 256.
The diagonals are fit **once on the whole rotated matrix** and sliced, because
`sv` is per output row and a slice-local fit is one no whole-matrix encode
could reproduce.

| arm | bpp | rel_err | vs plain | vs EXL3 |
|---|---:|---:|---:|---:|
| `d-q` | 4.0117 | **0.08888** | **1.096×** | 1.572× |
| `--q` | 4.0000 | 0.08952 | 1.088× | 1.584× |
| `drq` | 4.0117 | 0.08974 | 1.085× | 1.587× |
| `-rq` | 4.0000 | 0.09056 | 1.075× | 1.602× |
| `d--` | 4.0117 | 0.09689 | 1.005× | 1.714× |
| `---` | 4.0000 | 0.09738 | 1.000× | 1.723× |
| `dr-` | 4.0117 | 0.09769 | 0.997× | 1.728× |
| `-r-` | 4.0000 | 0.09862 | 0.987× | 1.745× |

`d-q` at **4.0117 bpp is EXL3 K=4's rate to four decimals**, so that row is a
matched-size comparison with no rounding in it: **1.572×**.

Two readings worth keeping.  **Rotation hurts** (0.987× alone, and it costs the
`drq` arm against `--q`).  Tessera can only rotate the *input* side — S7 makes
two-sided rotation a weight-space measurement state, not a serving branch — so
this arm is structurally half of EXL3's incoherence processing, and half of it
is worse than none.  **Diagonals buy almost nothing** (1.005×), which says the
rank-1 magnitude field the per-32 scale cannot see is already small on these
weights.

## Result 4 — the scale rule is already the best global choice

`_pack_scales` lands a group's `amax` exactly on the top of the grid.  That
guarantees no clipping; it does not minimise error, and principle 2 says to ask
the objective.  A global headroom multiplier is the crude form of JSO's
question, and it is encoder-side only — whatever scale is chosen is written
into the segment-2b bytes, so every headroom is already a legal artifact.

| headroom | rel_err | vs the rule |
|---:|---:|---:|
| 0.55 | 0.12773 | 0.762× |
| 0.65 | 0.10276 | 0.948× |
| 0.75 | 0.12331 | 0.790× |
| 0.85 | 0.11794 | 0.826× |
| 0.95 | 0.09960 | 0.978× |
| **1.00 (the `amax` rule)** | **0.09738** | 1.000× |
| 1.10 | 0.11445 | 0.851× |
| 1.25 | 0.15427 | 0.631× |

**The rule wins.**  As a *global* multiplier this lever is closed: `amax/peak`
is already the best of eight.  Two things are worth keeping anyway.  The curve
is **jagged, not unimodal** — 0.65 beats both 0.75 and 0.85 — because the
effective scale snaps through the E8M0 base plus a 4-bit refinement, so a
smooth multiplier moves it in jumps.  Any future per-group search has to search
the *stored words*, not a continuous scale.  And a global multiplier is not
JSO: JSO chooses per group, inside the loop, against an activation-weighted
objective.  What is closed is the crude form.

---

## Result 5 — the decisive one: at matched *payload* bits the gap is 1.142×

`quantize_exl3` takes `K` as a parameter, so EXL3 has a rate axis and Tessera's
two serialisable rungs can be met on it.  **K=3 has 3.0 payload bits, exactly
Tessera's `E2M1_K1` rung**, so this is a matched-payload comparison with no
extrapolation in it.

| coder | payload bits | total bpw | rel_err |
|---|---:|---:|---:|
| EXL3 K=2 | 2.0 | 2.0117 | 0.22082 |
| **EXL3 K=3** | **3.0** | 3.0117 | **0.11089** |
| **Tessera `E2M1_K1`** | **3.0** | **3.5000** | **0.12666** |
| Tessera `E2M1_K2` | 3.5 | 4.0000 | 0.09738 |
| EXL3 K=4 | 4.0 | 4.0117 | 0.05653 |

**At matched payload bits, EXL3 leads by 1.142× — not 1.72×.**

The K=4 row re-measures `exl3_arm_glm_experts_v2.py`'s 0.05653 to five decimals,
which is the check that the isolation below worked.

> **A trap, recorded because it nearly became a result.** Running K=2, 3 and 4
> in one process gives a correct number for the first K and `rel_err` **55–65**
> for the rest — at *unchanged* reader drift, so the reader faithfully
> reproduces a quantizer that produced garbage. Something in `quantize_exl3`'s
> call path is stateful across `K`. `exl3_rate_sweep.py` therefore runs **one K
> per process**. A drift assertion checks that the bytes mean what the
> quantizer meant; it cannot check that the quantizer meant anything.

### What the two comparisons say together

| axis | EXL3 | Tessera | gap |
|---|---:|---:|---:|
| matched **bytes** (~4.01 bpp) | 0.05653 | 0.08888 (`d-q`) | **1.572×** |
| matched **payload** (3.0 bits) | 0.11089 | 0.12666 | **1.142×** |

The difference between those two rows is **where the budget goes**:

    Tessera E2M1_K2 @ 4.0000 bpp = 3.5 payload + 0.5000 scale plane  (12.5%)
    EXL3    K=4     @ 4.0117 bpw = 4.0 payload + 0.0117 suh/svh      ( 0.3%)

Tessera spends an eighth of its budget on scale metadata, and body plus
completion sum to the grid's cap, so it **cannot** move that saving into the
payload.  Confirmed by the accountant: `terminal_rate` prices `E2M1_K1` at
3.500000 bpp as 3.000000 payload plus exactly 0.500000 of scale plane, and the
registry prices `E2M1_K2` at 4.0.

There is a second, smaller term.  EXL3's rate-distortion slope over its own
curve is **1.99× and 1.96× per payload bit**; Tessera's 3.0→3.5 step is
**1.30× per half bit = 1.69× per bit**.  So the coder gap *widens* with rate —
1.142× at 3.0 payload, and extrapolating Tessera's own slope to 4.0 payload
would put it at 0.0749 against EXL3's 0.05653, a gap of 1.32×.  Treat that as
indicative only: the 3.0→3.5 step is also a **family change** (arity 1 → arity
2), and `tessera-project-scope` already records that below cap the scalar
trellis beats the k-tuple by 28%, so the step is not a pure rate move.

---

## Result 6 — buying the scale plane back does not lift the curve

`group` and `half` are manifest geometry, already on the wire, so the scale
plane's cost is a parameter and not a constant.  Sizes below are the
**accountant's**: `encode_linear` builds the real artifact, verifies its own
round-trip, and returns `exact_bytes`.

| geometry | scale bpp | total bpp | rel_err | vs `g32/h16` |
|---|---:|---:|---:|---:|
| `g32/h16` (today) | 0.5000 | 4.0005 | 0.09738 | 1.000× |
| `g32/h16` + diagonals | 0.5000 | 4.0122 | 0.09689 | 1.005× |
| `g64/h32` | 0.2500 | 3.7505 | 0.10276 | 0.948× |
| `g64/h32` + diagonals | 0.2500 | 3.7622 | 0.10137 | 0.961× |
| `g128/h64` | 0.1250 | 3.6255 | 0.10723 | 0.908× |
| `g128/h64` + diagonals | 0.1250 | 3.6372 | 0.10519 | 0.926× |
| `g256/h128` | 0.0625 | 3.5630 | 0.11093 | 0.878× |
| `g256/h128` + diagonals | 0.0625 | 3.5747 | 0.10905 | 0.893× |

*The accountant's totals carry ~0.0005 bpp of container overhead that the
formula-derived `bpp` columns in Results 3–4 omit, and it prices the rank-1
diagonal planes itself — the `+d` rows differ from their `−d` rows by exactly
16·(rows+cols)/params = 0.011719 bpp and nothing more.  A first version of this
table added that term on top of `exact_bytes` and over-charged every `+d` row by
0.0117 bpp; the `rel_err` column was never affected and the corrected sizes make
the diagonal arms marginally cheaper than first published, not dearer.*

So the scale plane can be cut from 0.5000 bpp to 0.0625 — down to EXL3's order
of overhead — for **1.139× error and 0.4375 bpp saved**.  That is a real,
serialisable rate-distortion trade at points that exist today.

**And it does not help.**  Tessera at 3.5630 bpp scores 0.11093; **EXL3 at
3.0117 bpp scores 0.11089** — the same quality, half a bit cheaper.  Cutting the
scale plane moves Tessera *along its own curve*, it does not lift it onto
EXL3's.

The slope says why.  Over this clean within-family sweep (no arity change,
payload fixed at 3.5 bits, only the scale plane moving) Tessera buys
**1.347× per bpp**.  EXL3 buys **1.96× per payload bit** across its own curve.
Scale bits are cheaper bits than payload bits, which is not a surprise — but
Tessera cannot buy payload bits at all above 3.5, so the cheap axis is the only
one it has.

Diagonals do behave as predicted: they matter more as the group scale coarsens
(1.005× at `g32/h16`, 1.017× at `g256/h128`), because a rank-1 magnitude field
is what a coarse per-group scale most obviously fails to see.  The effect is
still small.

---

## The curves, side by side

| bpp | EXL3 | Tessera (best arm at that size) | gap |
|---:|---:|---:|---:|
| 4.0117 | **0.05653** | 0.08888 (`d-q`, LDLQ + diagonals) | **1.572×** |
| 3.5747 | ~0.0759 (interpolated) | 0.10905 (`g256/h128`+d) | ~1.44× |
| 3.0117 | **0.11089** | — (Tessera cannot reach 3.01) | — |
| 3.0 payload bits | 0.11089 | 0.12666 (`E2M1_K1`) | **1.142×** |

**EXL3's curve is below Tessera's at every size Tessera can reach, and the gap
widens with rate** — 1.142× at 3.0 payload bits, ~1.44× near 3.57 bpp, 1.572×
at 4.0117.  That is EXL3's steeper slope, not a single missing lever.

---

## What this changes

**The encoder-first order from the head-to-head is complete, and it was the
right order** — it produced the measurements that redirect the work, which is
what a gate is for.  But the answer is not the one the order was hoping for.

1. **Activation-awareness is built, and it is worth 1.088× — not 1.258×.**  The
   mechanism is real and the response is real.  It is not the gap.
2. **The coder is not "less responsive to calibration".**  Trellis and scalar
   respond identically (1.00× ratio at five block sizes).  The head-to-head's
   feared branch is closed, and its 1.37× "ceiling on the coder gap" is now a
   loose bound: at matched payload bits the two formats are **1.142×** apart.
3. **The scale plane is expensive but not misallocated.**  It costs an eighth
   of the budget against EXL3's 0.3%, and buying it back is measurable, cheap
   and serialisable today — and it lands Tessera *below* EXL3's curve, not on
   it.  So this is a **size** lever, not a quality one.
4. **Three levers are now closed by measurement**, all of which looked
   plausible before: `R_IN_ONLY` rotation (0.987×, and Tessera cannot rotate
   the output side — S7 makes two-sided a weight-space state, so half of
   incoherence processing is worse than none), the global scale-headroom
   multiplier (loses to the `amax` rule it would replace), and finer LDL blocks
   (non-monotonic; EXL3's 16 would not have helped).
5. **What is actually left is the rate-distortion slope**, and that is a
   statement about the format, not about a missing pass.  Two open items speak
   to it and neither has been evaluated here: raising the grid's cap above 3.5
   payload bits, and the free-grid question — `tessera-project-scope` records
   grid and trellis as substitutes worth **1.78 dB at zero redundancy**
   (≈1.22×), and the shipping rungs sit at exactly zero redundancy
   (`completion=0`).  That number is inherited, not re-measured here.
6. **Two encoder levers stay open and need no wire change:**
   `static_act_order`, and a per-group scale search against an
   activation-weighted objective — the two pieces of GPTQ+JSO this LDLQ does
   not implement, together worth the difference between 1.076× and 1.258× on a
   scalar coder.  Note from Result 4 that such a search must enumerate the
   **stored scale words**; the effective scale snaps through E8M0 plus a 4-bit
   refinement, so a continuous multiplier is the wrong search space.

**Consequence for the standing goal.**  A GLM-5.3-Flash artifact size-matched to
Mia's, with routed experts in Tessera, is buildable at 4.0117 bpp and would
score **0.08888** where hers scores **0.05653** on this screen.  That is the
honest state: the size target is reachable and the quality is not, and no lever
measured today closes it.

**The kernel-lane backend stays de-prioritised**, for a changed reason.  The
head-to-head de-prioritised it because Tessera lost 1.72× and the deficit was
unlocated.  It is now located — the rate-distortion slope of the grid-plus-
completion construction — and that is a format question, not a backend one.

## Scope

Six projections of one expert in three layers of one model.  `down_proj` is
unpriced — the probe caches one input per packed-expert entry at hidden dim.
Functional error on cached activations, not a served KL: **principle 3 says this
selects nothing.**  It is a screen, and it is the same screen every arm in this
series ran on, which is what makes the arms comparable to each other.  The
interpolated EXL3 row in the side-by-side table is log-linear between two
measured points and is marked as such; every other number is measured.
