# Where Tessera stands, 2026-09-01

One page, written to be read before the next build decision. Every number here
is measured and every measurement is linked. Nothing in this file is a plan.

## The standing goal is evidence-blocked

> *"use prismabuild to serve a glm 5.3 flash version that's size matched to MIA
> and inherits its mtp and vision head handling. Everything eligible to be
> quantized can go through pq and be exported in tessera format."*

The size target is real and reachable: Mia's body excluding vision and MTP is
**158.783 GiB**, and Tessera's `E2M1_K2` rung at **4.0000 bpp** clears her
4.2989 bpw expert target with room. The blocker is not size. It is that the
format which hits the size currently **loses 1.72× in quality to the format it
would replace, at the same size** — measured on real GLM routed experts, held
out, against EXL3 quantized fresh by its own quantizer
(`docs/measurements/exl3-head-to-head-2026-09-01.md`).

Shipping the artifact today would mean shipping something size-matched to Mia's
and measurably worse than it. That is the whole finding.

> **Re-measured 2026-09-01 (late):** the 1.72× was against a probe-split EXL3
> number. EXL3 quantised fresh with a 7168-row Hessian and scored on the same
> held-out capture rows is 0.0679; the default wire is **1.176× behind on the
> weight leg, 1.070× under W4A4** (1.017× in weight space, but against
> LDLQ-inflated EXL3 — ~1.22× behind it uncompensated)
> (`docs/measurements/tessera8-targets-2026-09-01.md`). The size target is
> still reachable and the quality gap is now small enough to be a served-KL
> question rather than a format question.

## What the routed-expert menu actually is

Six GLM projections, layers 5/20/42, gate and up, real cached activations,
tokens split fit/eval, every arm scored on the disjoint eval half. Relative
functional error, lower is better:

| arm | bpp | contract | rel_err |
|---|---:|---|---:|
| EXL3 (fresh quantize) | 4.0117 | W4A16 | **0.05653** |
| NVFP4, GPTQ+JSO | 4.5 | W4A16 | 0.06595 |
| NVFP4, RTN | 4.5 | W4A16 | 0.08294 |
| **Tessera K2** | **4.0000** | W4A16 | **0.09738** |
| NVFP4, GPTQ+JSO | 4.5 | **W4A4** (what vLLM serves) | 0.10806 |
| Tessera K1 | 3.5000 | W4A16 | 0.12666 |
| FP8_E4M3 | 8.0156 | W8A8 | 0.03050 |

Two things follow, and they cut opposite ways:

- **NVFP4 at 4.5 bpp is Pareto-dominated by Tessera at 4.0** on this route.
  `flashinfer_b12x` serves GLM MoE as W4A4, and the activation leg costs +64%.
  Tessera is both smaller and better than the format it was built to beat.
  (`docs/measurements/glm53-expert-menu` table, memory
  `glm-expert-menu-nvfp4-is-dominated`.)
- **EXL3 is not NVFP4.** It is W4A16, it is 1.72× better than Tessera at
  matched size, and it is what Mia actually shipped. The comparison that
  matters is the one we lose.

## Where the 1.72× comes from — MEASURED, 2026-09-01

**Superseded by measurement.** The bound below was a transplanted estimate; the
encoder it was waiting on has since been built and the decomposition is now
measured. See `docs/measurements/tessera-activation-aware-encoder-2026-09-01.md`.

- **Activation-awareness is worth 1.088×**, not the 1.258× acceptance — but the
  trellis is *not* less responsive than a scalar coder (same LDLQ loop, five
  block sizes, NVFP4-RTN vs Tessera → **gain ratio 1.00×**).
- **At matched *payload* bits the formats are 1.142× apart**, not 1.72×:
  EXL3 K=3 at 3.0 payload scores 0.11089, Tessera `E2M1_K1` scores 0.12666.
  Tessera @4.0000 bpp spends 3.5 payload + **0.5 on its scale plane**; EXL3
  @4.0117 spends 4.0 + 0.0117.
- **Buying the scale plane back does not lift the curve.** 0.5000 → 0.0625 bpp
  of scale costs 1.139× and saves 0.4375 bpp; Tessera at 3.5630 bpp scores
  0.11093 where **EXL3 at 3.0117 scores 0.11089**.
- **The gap is the rate-distortion slope:** EXL3 buys 1.96× per payload bit,
  Tessera 1.347× per bpp on a clean within-family sweep. It widens with rate —
  1.142× → ~1.44× → 1.572×.
- **Closed by measurement:** `R_IN_ONLY` rotation (0.987×, hurts), the global
  scale-headroom multiplier (loses to the `amax` rule), finer LDL blocks
  (non-monotonic), diagonal-Hessian importance weighting (a *provable* no-op).
- **Still open:** the free-grid question (≈1.22× at zero redundancy,
  inherited not re-measured) — and, under the FP4-native constraint below, it
  is out of scope for the default; `static_act_order`.
- **Built since (2026-09-01, later the same day):** the per-group scale search
  over the **stored scale words** is `scale_refit` — see the next section.

**The best Tessera arm at Mia's exact rate (4.0117 bpp) scores 0.08888 where
EXL3 scores 0.05653.** The size target is reachable; the quality is not.

## The FP4-native lever battery — MEASURED, 2026-09-01

Rob's constraint, same day: Tessera must natively use NVIDIA's 4-bit tensor
cores. Tessera-4's decoded tile already does (E2M1 codes × a per-16
E4M3-representable S6b scale); the learned codebook is the one lever that
does not, and it stays kernel-lane research. Every FP4-native lever was then
priced on the **real K-grouped S6b plane** (every earlier number used an
N-grouped fp32 amax plane), six routed experts, held-out, against
EXL3@A4 projected. Full tables: `docs/measurements/tessera-fp4-native-levers-2026-09-01.md`.

- **The plane's VALUES are the lever, and it ships.** LS refit of each
  half's scale to the trellis's codes, landed on the stored S6b words,
  alternated with the trellis and ending on a refit: **1.084× over the
  artifact plane at the same four Viterbi passes, default-on** (`cf82b00`,
  `61df165`), no wire change, the profile id untouched. EXL3@A4 gap
  1.253× → ~1.199×. Encode time ~4× the amax plane's; `scale_refit=1` is a
  free 1.044×. The merged 151.487 GiB export was built at refit 0. This is a
  per-tensor screen (six tensors, weight leg, 128 held-out tokens);
  promotion needs the served refit-0-vs-4 A/B, queued.
- **The plane cannot be deleted or thinned.** Rank-1 (row × 16-block)
  field 0.77×, 0.83× with its bits re-spent on L=8; E8M0-only 0.84×;
  E8M0 + L=2 at 4.0 bpp 0.90×. The 0.5 bpp buys per-column magnitude
  structure and the per-16 hardware scale is the right carrier for it.
- **Extra rate has one FP4-native home: Wei L=2 in the trellis** (+0.25 bpp,
  1.104×, gap → 1.142×), a wire change. **At exactly 4.0 bpp, one E4M3
  per 32 duplicated into both per-16 slots + L=2 is 1.047× over the shipping
  encoder** — the strongest same-size wire candidate.
- **LDLQ's regulariser was the unswept knob:** σ=1.0 is 1.137× (S6b) /
  1.178× (flat E4M3) where EXL3's 0.025 gave 1.083×; stacked with the refit,
  1.134× / 1.149×, gap → 1.131× / 1.125×. Still a screen — adjacent 128-token
  halves — until the 16-document capture running on lina scores it.
- **Closed under the constraint:** global headroom (again), H16-weighted LS
  (+1%, needs activations), group-local LDLQ, every plane-thinning form.
- **Format against format** (weight leg alone vs EXL3 K=4, the 1.72× above):
  artifact plane 1.722× → default encoder **1.590× at exactly 4.0 bpp**, no
  wire change; flat E4M3 plane 1.565×; per-32 E4M3 + L=2 1.519× at 4.0;
  per-16 E4M3 + L=2 1.409× at 4.25; LDLQ stack 1.396× (screen).
- **The rate/plane frontier is mapped** (plane granularity × Wei L, every
  cell refit): at 4.0 bpp the point is **per-32 E4M3 + L=2 (1.047×)**; the
  flat E4M3 plane beats S6b at every L (1.7–2.9%); above 4.0, per-32 + L=4
  (4.125, 1.074×) then per-16 + L=2 (4.25, 1.129×). A Wei-L wire change
  keeps the embedded completion axis and the served tile; it changes the
  load-time materialiser (stock lane) and the kernel lane's decoder.

<details><summary>The superseded bound, kept for the record</summary>

### Where the 1.72× comes from — bounded, not settled

NVFP4 RTN (activation-blind) 0.08294 → GPTQ+JSO (activation-aware) 0.06595 is a
**1.258×** response to compensation. Transplanting that response to Tessera
credits calibration with `ln(1.258)/ln(1.72)` = **42%** of the gap and leaves
**1.37×** for the coder.

That 1.37× is a **ceiling under one assumption**, not a floor: if a trellis
responds to a Hessian at least as well as a scalar quantizer does — plausible,
since EXL3's LDL ordering is integrated with its trellis search rather than
bolted on after — then calibration's share is at least 42%. If a trellis
responds *worse*, the coder gap is larger and the format itself is the problem.
Nothing here measures Tessera's response, because Tessera has no
activation-aware encoder to measure. Do not quote 1.37× as the coder gap.

</details>

## The rate axis is two points, not a band

> **⚠ SUPERSEDED 2026-09-01, later the same day. This section is false on
> `HEAD`.** It was written at 05:51, before `a96064b` (07:43) fixed the flat
> ladder and `a4de134` (08:08) admitted `E4M3` to the serialisable set. Both the
> heading and the claim that there is "nothing above 4.0" are wrong. The rate
> axis is continuous and the serialisable set is three families, not two:
>
> | Family | Ladder, at completion 0 | Top rung |
> |---|---|---|
> | `E2M1_K1` | 1.5 to 3.5 bpp | 3.5000 |
> | `E2M1_K2` | 1.0 to 4.0 bpp | 4.0000 |
> | `E4M3_K1` | 1.5 to 7.5 bpp | 7.5000 |
>
> What survives: every measurement taken at a **top rung** is unaffected,
> because at the cap the completion capacity is zero and the bugs are inert.
> That covers the EXL3 head-to-head, the matched-payload 1.142x, and "Tessera
> 4.0 beats NVFP4 4.5 as served". Any sub-cap point measured before `a96064b`
> is void.
>
> See `docs/handovers/tessera-handover-2026-09-01-evening.md` for the audit,
> and `experiments/results/tessera_rate_grid.json` for the measured ladder
> these rows are read from.

<details><summary>The superseded section, kept for the record</summary>

Every rung of a Tessera family serialises to the **same bytes** — a column at
rate `R` writes `R` body bits *and* `cap − R` completion bits. The serialisable
set is `E2M1_K1` at **3.5000** and `E2M1_K2` at **4.0000**, and there is nothing
between them and nothing above 4.0. The rung is a quality knob at fixed size;
every sub-top rung is strictly dominated
(`docs/measurements/tessera-rate-ceiling-2026-09-01.md`).

</details>

This refuted the bit-trade's gain leg. The trade still wins — freeing 8.123 GiB
by pricing attention and `lm_head` at FP8 buys FP8 on ~2.6 of 45 expert layers,
gain/cost **7.7×** — but its mechanism is "promote the layers that need it", not
"raise everyone's rate" (`docs/measurements/glm53-bit-trade-2026-09-01.md`).

## The decision chain

**2026-09-02 (frontier) — the production encoder on the six experts, every
(grid, body, plane, rate) arm at equal bytes** (`docs/tessera-one-format.md`
§4). Headline: **E4M3 window body over the CHANNEL plane, L=12, no LDLQ:
0.985× EXL3 K4 at 4.0 bpp, 0.957× EXL3 K3 at 3.0 bpp, 1.016× at 5.0**,
on the true wire, decoded bit-exactly by the kernel lane. As served (W8A8
on the FP8 tensor core) it is 1.047× EXL3's W4A16 at 4.0 bytes and 0.973×
at 3.0. The window body also takes the E2M1x2 sub-cap ladder from 1.36–1.43×
behind EXL3 to 1.06–1.10×; at the E2M1x2 cap the coset trellis still wins
at L=12 (1.170× vs 1.244×). Gates for flipping the E4M3 recipe: kernel
decode (done, incl. CHANNEL), quality on the wire (done), encoder
throughput (fused Viterbi in flight), and PrismaQuant's byte accountant
pricing a shape-dependent recipe (the seam refuses rather than floors).

1. Full-model allocation deferred — an allocation cannot be built or exported
   today (`export_native_compressed.py` has zero Tessera references,
   `tessera_allocator.py` sets `producer_eligible: False`,
   `_TESSERA_SERVING_LANE_EXISTS = False`).
2. Gate set before running: *if Tessera loses badly to EXL3 at matched bpw, the
   kernel-lane backend is premature.*
3. Gate run. **1.72× is losing badly.**
4. Therefore: encoder first. The backend and the rate-ceiling work are both
   gated on the same harness re-run.

## Next, in order (rewritten 23:40 UTC after the wire build)

The index-plane measurement (`docs/measurements/tessera-index-plane-2026-09-01.md`)
overturned the "same-size wire changes cap near 1.05×" reading: that number
was a per-32 loading loss, not a limit. Halving the plane's *bytes* at per-16
granularity is lossless, and the freed quarter-bit on Wei's span-2 partition
is the same-size lever the limits doc said did not exist. **It is built and
default-on** (schema minor 1, `docs/measurements/tessera-wire-default-2026-09-01.md`):
the production encoder measures **1.125× over today's default at 4.0 bpp**
on the six GLM experts, and the W4A4 gap to EXL3@A4 is 1.205× → **1.137×**.
The 151 GiB export on disk (refit 0, span 1, S6b) is 1.22× behind it.

**2026-09-02 — the window body (schema minor 2) is built, not yet default.**
The bitshift trellis on the tile (`docs/measurements/tessera-window-body-2026-09-02.md`):
a position's code is a table lookup on the last L bits of its column's
stream, the 2^L table rides the ALPHABET plane, no forest, no completion
axis. Below the E2M1x2 cap it is 1.3× better than the coset trellis at the
same bytes (3.5 bpp); on E4M3 under a per-channel plane, L=14 is 1.2× better
than the convolutional trellis and 1.07× better than EXL3 K4 in output space
at 4.0 bpp — and W8A8 on the FP8 tensor core is level with EXL3's W4A16 at
the same bytes. All of that is six Gaussian-input GLM-5.3-Flash routed experts
in a weight-leg screen with no served arm; the same wire served on dense Qwen
loses 23× to FP8 RTN at equal residency (7.4× after the reach fix, later on
this page), because the CHANNEL plane does not see outlier input columns. At the E2M1x2 cap the structured coset table stays better
until L≥14–16. `BodyKind.WINDOW` / `window_bits` are manifest fields bound
into the profile id; `encode_linear(body=BodyKind.WINDOW, window_bits=L)`
writes it; `DEFAULT_BODY` stays TCQ until (a) a window GEMV exists in the
kernel lane (`pack_unit_for_kernel` refuses the body today) and (b) the
encoder is faster than the O(2^L) reference (~150 s per 2048×4096 tensor at
L=14). The per-channel scale plane the E4M3 headline used is not yet a
`ScalePlaneKind`; it is the next wire addition, and it is also the layout
the served W8A8 path consumes.

**2026-09-02 (later) — the cohesive view, and the per-channel plane.**
Rob: *"homogenize things where possible between 4-bit and 8-bit so that we
have a grand cohesive view of things instead of piecewise."* The view is
`docs/tessera-one-format.md`: five axes (grid, body, rate, scale plane,
route), the 4-bit and 8-bit tiles two points of the first, EXL3, Gridbook
and NVFP4 placed in the same grammar. Built for it (schema minor 3,
`docs/schema/prismaquant.tessera.v1.md` §1c): **`ScalePlaneKind.CHANNEL`**
— one fp16 per output row on the DIAG_SV plane times an fp32 global, no
block planes, the layout the FP8 tensor core consumes and the plane the
8-bit headline was measured under; `decode.materialize_fp8` yields the
stock per-channel FP8 pair, so E4M3 has a stock lane exactly as E2M1 has
NVFP4. **`export.wire_recipe(grid, q256)`** replaces three global defaults
as the one statement of which body/plane a grid ships (today `TCQ_RECIPE`
everywhere; the E4M3 and sub-cap flips are one line each, gated on the two
workers). The window body's rate cap is now the grid's whole width. The
pinned six-tensor re-run makes the headline **0.938× of EXL3 K4 at L=14**,
and on the true wire the E4M3 plane is worth 1.14× at equal bytes
(`tessera-window-body-2026-09-02.md`). `experiments/tessera_frontier.py`
scores every (grid, body, plane, rate) point through the production
encoder on one protocol; its first run is in progress. In flight: the fused
window Viterbi and the kernel-lane window decode (two worktree workers),
and the PrismaQuant seam (recipe-aware pricing, activation contract per
grid × plane).

1. ~~**Kernel lane: span-2 decode.**~~ **Done** (`docs/measurements/tessera-kernel-span2-2026-09-01.md`):
   the tuple GEMV decodes the minor-1 wire bit-exactly at the wire's own
   4.0 b/wt (the LUT plane is read as nibbles, not materialised), and with
   the per-unit values in subset order it is as fast as span 1 at the same
   launch shape (0.0664 vs 0.0673 ms) and 11% faster at its default
   (0.0524 vs 0.0589 ms, 75 W of ~140). Scalar-lane and prefill-GEMM span-2
   decodes are still open; the tuple family is what ships.
   **Also landed:** the scale-weighted trellis
   (`docs/measurements/tessera-trellis-weighting-2026-09-01.md`), exporter
   default, +0.8% at the default wire; W4A4 vs EXL3@A4 is now **1.133×**
   *projected* — and **1.070× measured** once EXL3 was quantised fresh and
   scored on the same held-out capture rows (K=4 out-space 0.0679, not the
   probe's 0.0565; weight leg 1.176×, weight space 1.017× against
   LDLQ-inflated EXL3 and ~1.22× against it uncompensated;
   `docs/measurements/tessera8-targets-2026-09-01.md` §3). Every "1.72×"
   and every ratio built on 0.0565 in this file is superseded by that.
2. **Re-drain GLM on the new wire** (Rob's call; the merged export is a
   different artifact under minor 1) and the **served A/B** — new default vs
   the refit-0 export — on the `tessera-served-kl-2026-09-01` harness. That
   is the promotion gate for both the refit and the wire.
3. **LDLQ with a real Hessian.** Out-of-document verdict is in
   (`tessera_ldlq_generalisation.json`: σ=1.0 gives 1.081×/1.105× on the two
   held-out folds, 1.52× on the adjacent-halves control; σ=0.025 is harmful;
   gain still rising at 7k rows). Large capture (≥64k tokens, per-expert
   routed tokens shrunk towards the shared H), then the LDLQ encode inside
   PrismaQuant's render path. Stacks with the new wire (~1.10× more, screen).
4. **The E4M3 payload grid — Rob's mandate (2026-09-01 22:50 UTC), analysis
   DONE** (`docs/measurements/tessera8-targets-2026-09-01.md`): *"perform the
   same optimization on the 8-bit format + kernels … two targets: the
   theoretical, and how exl3 would perform with 8-bit activations"*.
   - **Target 1:** the E4M3 alphabet floor (per-channel FP8 RTN, 8 bpp) is
     **0.0189** A16 / **0.0306** W8A8; no E4M3 tile at any rate goes below.
     The served A8 leg alone is 0.0241. Trellis-mechanism gap to Shannon on
     Gaussian: 1.29× at R=4 with the corrected codebook geometry (was 1.39×),
     vs EXL3's 1.07×.
   - **Target 2:** EXL3 K6@A8 = 0.0299 already sits on that floor and
     K8@A8 = 0.0246 is **1.25× better** than any 8-bit E4M3 tile — *"beat
     EXL3 with W8A8"* is closed above ~5 bpp by the alphabet. At **4–5 bpp**
     a per-channel, no-plane Tessera-8 (LM+midpoints anchors snapped to
     E4M3, Ungerboeck code, LS row scale) is **1.13× behind EXL3 under W?A8**
     (level with EXL3-after-LDLQ in weight space, ~1.2× behind it
     uncompensated), its decoded tile a stock per-channel FP8 tensor on every
     vendor — and **at 4.0 bytes as served it beats Tessera-4: W8A8 0.0816
     vs W4A4 0.1176, 1.44×** (the A4 leg is 3.6× the A8 leg), at half the
     FP4 MMA rate with no native FP4 path needed. The cross-platform case,
     with LDLQ (1.06–1.09×, no wire change) unspent. The floor binds EXL3 on
     the FP8 tensor core too; its W?A8 numbers assume its own decode kernel.
   - **Gridbook (Rob's ask), format for format, both sides activation-blind
     (no imatrix, no LDLQ):** Tessera-4 at 4.0 beats FP8-CB K32 by 1.09× on
     the weight leg; Tessera-8 beats FP8-CB by 1.12× at 4 and 5 bpw and
     loses 1.25× at 6; FP4-CB K24 beats Tessera-4's sub-cap ladder by 1.09×
     at 3.28 bpp. As each deploys, FP8-CB K32 (W8A8) beats Tessera-4 (W4A4)
     by 1.30× at 4.0 bytes, and Tessera-8 per-channel beats both.
   - **Found on the way:** below the cap the doubled Lloyd-Max codebook is
     worse than scalar RTN at R=5 and the conv code is non-Ungerboeck; on
     the E2M1×2 cap the code is worth 0.0–0.3 % (closed), below the cap it
     is the whole game. The E4M3 family's S6b plane was its other defect:
     the LUT plane saves 0.25 bpp *and* improves error 1.13–1.33× at
     R=4–6, leaving the family 1.07× / 1.05× (out / A8) behind EXL3 at
     4.25 bpp — kernel-lane only; a per-16 plane on an FP8 weight is not a
     tensor-core scale layout. Next: fix the sub-cap builder anchors (E4M3 and
     E2M1×2), LDLQ on the per-channel arm, then an 8-bit kernel lane
     (`build_code_lut` is still R=3). `conv_generators` is now in the config
     and the merge guard; a config replay resolves missing keys to their
     legacy meaning (`encode_settings_from_config`, witnessed byte-identical
     on two GLM units of the 151 GiB export).
5. Held: scalar-lane LUT split; delete partA/partB (ask first — note they
   can no longer re-merge as they stand: the merge guard refuses any part
   lacking a SHARED key, and the parts predate `conv_generators`, written
   only since `efa1b9e`; the 151 GiB merged artifact is their only
   mergeable form); ladder probe dispatch; box chores.

## 2026-09-02 (follow-ups: compensation, sub-4 EXL3 rungs, Gridbook as it ships)

`experiments/tessera_vs_exl3_followups.py` (37 arms × 6 experts, same
held-out rows; EXL3 K4 reproduces `tessera8_targets.json` bit-identically;
EXL3 K=2/3 added so sub-4 ratios are bracketed, not extrapolated).
LDLQ σ=3 on the default TCQ wire: 1.169× → 1.102× vs EXL3 (lower bound);
on scalar per-channel Tessera-8: 1.147× → 1.059× at R=4, 1.184× → 1.094×
at R=5 (1.01–1.02× under EXL3@A4). Gridbook: the production imatrix is
worth ≤0.5% on every rung; its gated LDLQ regresses every rung (K32 out
0.0869 → 0.1069; the gate's hold-out is half of its own fit rows and the
fit is rank-deficient at 1% damping) — production runs it off, so the
imatrix rows are Gridbook as it ships: FP8-CB 1.28×/1.33×/1.35× at 4/5/6,
FP4-CB 1.26–1.27× at 2.3–3.3, all behind the window body over CHANNEL
(0.957×/0.985×/1.016×/1.240× at 3/4/5/6). Open: LDLQ on the window body.
Full suite on the merged tree: 471 passed.

## 2026-09-02 (the flip)

Both mechanical gates closed and the recipe flipped, default, not opt-in.
The fused window Viterbi (`src/tessera/window_viterbi.py`, `ee9bdf2`) is
15× at L=12, 26× at L=14 and 26× at L=16 on a 2048×4096 tensor, bit-exact
(identical states and the identical sse float; the artifact bytes are the
reference's under every plane), with `torch.profiler` before/after and
Netdata power (22× the work per joule at 47% of the envelope). The config
carries the recipe per rung (`wire.recipes`, `dccbfd3`), PrismaQuant prices
a shape-dependent recipe exactly or refuses (`def11bd`), and `wire_recipe`
now returns: E4M3 → window over CHANNEL at L=14 on every rung (L=14 wire
arms, six experts, `experiments/results/tessera_frontier_L14.json`: **0.940×
EXL3 K4 at 4.0 bpp and 0.947× K5 at 5.0** in output space, 0.946×/0.964×
against EXL3 at the same 8-bit activation, 1.005×/1.18× as served — W8A8
against EXL3's W4A16; per expert 0.92–0.96× on all six);
E2M1x2 → window over LUT16 at L=12 below the cap (q256 < 896), the coset
trellis at the cap (the L=14 window at the cap is 1.227× — per expert
1.21–1.26× — against the trellis's 1.170×); E2M1 → the coset trellis, unmeasured
under the window body. Not shippable yet: `materialize_fp8` is a function,
not an exporter writing compressed-tensors; no serving lane is attested;
units are not TP-shardable; LDLQ on the window body is unmeasured.

**2026-09-02 (the stock lane, served).** Four of the five "not shippable"
items above are closed for the *materialised* form and the fifth is fixed.
`tessera.stock` (`406e951`) writes an E2M1/E2M1x2 unit over a LUT plane as
the compressed-tensors NVFP4 triple and an E4M3 unit over the CHANNEL plane
as the per-channel FP8 pair, bit-exact against the bytes-only reader in
vLLM's own arithmetic; fused groups share one `weight_global_scale` by an
exact binade shift (52 of 140 members on Qwen3-0.6B, none refused);
`experiments/export_stock_compressed.py` writes a checkpoint vanilla
`vllm/vllm-openai` v0.28.0 loads with no plugin, and served it on native
kernels for both formats (`FlashInferCutlassNvFp4LinearKernel`,
`CutlassFP8ScaledMMLinearKernel`); the tensors are ordinary NVFP4/FP8 and
inherit those formats' TP sharding (inherited, not measured: every serve here was TP=1); the window Viterbi's graph capture is thread-local behind a
lock, six threads at once bit-exact. The served A/B on Qwen3-0.6B
(`docs/measurements/tessera-stock-lane-served-2026-09-02.md`) is the honest
part: **as an NVFP4 encoder at 4.5 resident, Tessera loses to production
GPTQ+JSO NVFP4 by 1.254× KL (0.640 vs 0.511, W4A4, same kernel, same input
scales); as an FP8 encoder at 8.0 resident, Tessera-8's 4.07-bpp wire is 23×
worse than FP8 RTN (0.470 vs 0.020).** The inversion against the frontier is
activation structure the encoder never sees: Tessera-8 has 1.6× less plain
weight error than production NVFP4 and more Hessian-weighted error, carried
by `layers.2.mlp.down_proj`, whose input channel at 2.2 million × the median
second moment gets nothing from a per-row scale (1.68e-1 weighted vs 7e-3
plain). GLM's expert inputs are Gaussian; dense k/down inputs are not.
Still not shippable at its own rate: 4.0 bpp exists on the kernel lane only,
which stock vLLM does not run. Next, in order: activation-aware encoding on
the CHANNEL plane (LDLQ / `col_weights`, the render leg refuses them today),
then column smoothing folded into the preceding layer.

## 2026-09-02 (the serving path: a Tessera family inside Gridbook's trellis lane)

The stock lane measured the wire's *values* as served and lost at the stock
formats' residency; the wire's own 4.0 bpp had no runtime. It now has one in
build: Gridbook (`/home/rob/gridbook`, an out-of-tree vLLM plugin -- the
sanctioned CB lane) already carried a trellis lane scaffold (one
self-describing `wire_bytes` blob per Linear, a `family` scheme, E2M1 W4A4
and E4M3 W8A8 lane classes decoding to the stock tile for `torch._scaled_mm`,
a shared decode pool, resident/streamed modes, contract cells with receipts),
and Tessera's `build_unit_artifact` defaults to `ContainerClass.GRIDBOOK`.
So the wire-rate product is a **`TESSERA_NVFP4` family in that lane**, not a
fourth lane: the blob is decoded to the same NVFP4 tile the stock lane
materialised (byte-identical, so its served numbers are the stock arm's
0.640), the checkpoint holds the wire (4.0018 bpp on Qwen3-0.6B, 4.0044 on
disk with the per-unit tables and the container framing), resident mode holds
4.5 bpp in memory and streamed mode the wire plus one shared tile.

Built today (Tessera `a386407`, `06da06e`; Gridbook worktree
`/home/rob/gb-tessera-family` branch `tessera/family` on the release-0.9.1 +
emit-route base, `f108ef6`): `parse_unit_artifact` (the reader's verified
planes before reconstruction), `lane_planes.py` (the packers, Triton-free, so
Gridbook imports them and no parser is vendored), `fused.py` (one container
per vLLM-fused module; roles decoded into row slices; LUT tables moved onto
one global by an exact binade shift, refused otherwise -- agrees with
`stock.share_global` on Qwen), `export_gridbook_tessera.py`,
`gridbook_lane_served.sh`; Gridbook `tessera_scheme.py`, `tessera_ops.py`,
`tessera_nvfp4_lane.py`, `config.py` dispatch, `docs/TESSERA-LANE.md`, tests
(scheme, dispatch, lane numerics against `materialize_stock`).

Proven by a real vLLM 0.28 serve of the exported checkpoint: config parsed,
all 112 modules (fused qkv / gate_up included) dispatched to the lane,
weights loaded under the fused names, every blob parsed against its scheme,
every fused group's shared global exact, planes packed -- the load path
fails only at the native span-2 decoder, which is in flight
(`csrc/tessera_nvfp4.cu`, opus-high worker, oracle = `materialize_stock`
byte-for-byte). Container facts: `vllm/vllm-openai:latest` is 0.28.0 /
torch 2.13, has nvcc but not `cusparse.h`; linking only the *missing* headers
from the `nvidia/cu13/include` wheel dir (Gridbook's own Dockerfile rule)
makes its JIT builds work, and Gridbook's E2M1 lane + r256 decoder tests
pass there (`/home/rob/tessera-runs/gbfam/gbrun.sh`).

Next, in order: (1) the decoder lands → serve → KL must reproduce the stock
arm's 0.640 within the kernel's nondeterminism floor (the acceptance; a
different number means a decode defect, not a result); (2) a contract cell +
receipt (schema bump) so the lane stops resolving `unattested`, then
PrismaQuant's `_TESSERA_SERVING_LANE_EXISTS`; (3) the window body decoder
(sub-cap rates) and the E4M3/CHANNEL family (FP8 W8A8 route), each behind the
same oracle; (4) the routed-MoE cell, which is where Tessera 4.0 beats NVFP4
4.5 (GLM experts) and the product earns its rung. Dense quality at 4.0 is
still 1.25× behind production NVFP4 at 4.5 on Qwen; the allocator decides
where the rung is worth its bytes, and the activation-aware CHANNEL lead is
unchanged.

## 2026-09-02 (served and attested: both residency modes, eager and under the compiled forward)

The lane is real. The Qwen3-0.6B Tessera checkpoint (4.0018 bpp on the
wire) serves through Gridbook's `TESSERA_NVFP4` family on vanilla vLLM
0.28.0, and its numbers are the stock arm's: served KL-vs-BF16 0.6316 in
both residency modes eager (stock NVFP4 kernel on the same tile 0.6404),
0.6271 in both modes under vLLM's default compiled forward with CUDA graphs
(stock 0.6220), the two modes bit-identical to each other in both regimes.
Every cross-kernel mutual (0.245-0.257) sits where the stock kernel's own
eager-vs-compiled mutual sits (0.247); the fp64 per-Linear reference says
every module computes its input times its tile. Route census: 112 / 112
modules on `torch._scaled_mm` in both modes, eager and compiled. Full
numbers, floors and files: `docs/measurements/tessera-gridbook-lane-served-2026-09-02.md`.

Attested: Gridbook runtime contract v13 carries the family
`TESSERA_E2M1_K2` (rung 896 q256, the E2M1x2 cap, the one rate the receipt
covers) with two `device_qualified` sm_121 dense cells,
`backed_with_serve_flag` on the two lane env flags, TP=1. PrismaQuant's
admission (`tessera_render.tessera_lane_attested`) is now a lookup in the
pinned contract's cells, not a constant; it flips to True when PrismaQuant
re-pins to a Gridbook release that packages v13. Cutting that release is
Rob's call (the pin scripts refuse unreleased commits), so today the rung is
priceable and serveable but not yet exportable through PrismaQuant.

Six findings on the way to the compiled forward, all in Gridbook
`tessera/family` (`a1bcd06`, `5b176eb`, `c9219a4`, `fe5b8f8`, `11d3a20`,
`5f70798`; the contract is `4fbc543`): the route record specialised the token
dimension; the fingerprint guard compared `data_ptr`; the extension was
looked up lazily behind a lock; the decode called a pybind symbol directly
(now a custom op); the streamed decode mutated a per-device pool every
layer aliased, which Inductor's functionalisation turned into an illegal
memory access (now a functional op that owns its tile; the Tessera lane
holds no pool); and vLLM's compile cache is keyed without the residency
mode, so a resident load after a streamed one in one `~/.cache/vllm` took
the streamed AOT-compiled forward and died at the first forward (the other
order was not run; every lane now folds its mode and
release into `VllmConfig.additional_config`, the one hash input a plugin
reaches). Every lane receipt before today was eager-only; the trellis
siblings keep their pools and their streamed compiled mode is untested.

Scope of what is attested: one rung, dense Linears, TP=1, the E2M1x2 cap
wire decoded to the stock NVFP4 tile. Not in the lane: the window body
(sub-cap rates), the E4M3/CHANNEL family (the FP8 W8A8 route), routed MoE
experts (where Tessera 4.0 beats NVFP4 4.5 on GLM and the rung earns its
place), an 8-bit family. Next, in order: (1) the Gridbook release and the
PrismaQuant re-pin (Rob); (2) the window-body decoder and the E4M3/CHANNEL
family behind the same byte-exact oracle; (3) the routed-MoE cell; (4) the
trellis siblings' streamed mode under compile, on the functional-decode
pattern.

## 2026-09-02 (the product: one lane, two families, one checkpoint)

The stop-hook goal -- one shippable product uniting the 4-bit and 8-bit
Tessera wires -- is now a served thing rather than a plan. Gridbook's Tessera
lane has two routes behind one flag pair (`GRIDBOOK_TESSERA=1`,
`GRIDBOOK_TESSERA_MODE=resident|streamed`): `TESSERA_NVFP4` (E2M1x2 cap wire
-> the stock NVFP4 tile, W4A4) and `TESSERA_FP8` (E4M3 default wire, window
L = 14 over the CHANNEL plane -> the per-channel FP8 pair, W8A8). The
checkpoint's per-module scheme picks the route, so one checkpoint carries
both and one serve executes each on its own tensor-core path. Everything
below is Qwen3-0.6B on vanilla vLLM 0.28.0, GB10, both residency modes,
eager and under the default compiled forward
(`docs/measurements/tessera-gridbook-fp8-lane-served-2026-09-02.md`):

- **FP8 route.** The fresh E4M3 encode is byte-identical to the stock-lane
  checkpoint of 2026-09-02 (392 / 392), so that receipt is the comparator:
  lane 0.4660 eager / 0.4669 compiled against 0.4699; mutual lane-vs-stock
  0.021 inside the lane's own eager-vs-compiled 0.027; modes bit-identical;
  model memory 0.73 GiB resident, 0.55 GiB streamed (stock 0.74). The
  streamed decode is pure torch over the packed window streams -- no custom
  op, no pool -- and the compiled forward traces it.
- **E2M1 route under the unified flags:** 0.6316, unchanged, mutual 0.
- **The mixed checkpoint** (28 `down_proj` on NVFP4, 84 modules on FP8,
  4.06 bpp wire, 7.32 resident): lane 0.6772 eager / 0.6733 compiled
  against its compressed-tensors twin's 0.6741 on vLLM's own two kernels;
  mutual 0.103 inside 0.118; all four censuses 28 + 84 on the declared
  routes. Module for module it carries exactly the two uniform checkpoints'
  bytes, and its KL is worse than either (0.640 / 0.470): KL does not add,
  and the split was a route exercise, not an allocation.
- **Contract v14** (`gridbook.runtime-contract.v14`): rows `TESSERA_E2M1_K2`
  and `TESSERA_E4M3_K1`, two `device_qualified` sm_121 cells each,
  `backed_with_serve_flag` on the one flag pair. PrismaQuant's admission
  test covers the v14 shape; the answer stays False until the serving pin
  lands on a Gridbook release that packages v14 (Rob's call).

What this attests is faithfulness, not quality: on this dense model the
4.07-bpp E4M3/CHANNEL wire is 23x behind 8.0-bpp FP8 RTN (4 bits of code
against 8; production NVFP4 at 4.5 bpp is 25x behind the same arm -- and
Tessera-8's 0.470 is quoted at the wire's 4.07 while it *serves* at 8.0
resident on this route, so it is not a 4-bit point on that table) and
the 4.0-bpp E2M1x2 wire is 1.25x behind production NVFP4 at 4.5 under W4A4
(`tessera-stock-lane-served-2026-09-02.md`). The
lane makes whatever the encoder produces and the allocator chooses
shippable at the wire's bytes; the wire's quality on dense models is the
encoder's open problem (the CHANNEL plane's outlier blindness), and the
Tessera-8 wins on Gaussian-input GLM experts are where the 8-bit route
earns its place. Not in the lane: sub-cap E2M1x2 rates (the window decoder
is plane-agnostic by construction, but decoding an E2M1x2 window body
through it is untested and the NVFP4 route does not wire it), routed MoE
experts, TP > 1, and the trellis siblings still hold their own pools. One
lead from the three arms, an inference and not a decomposition: moving 84
of 112 modules from E2M1x2 to E4M3 moved KL 0.640 -> 0.677, so `down_proj`
carries the damage in every arm and the other modules' family barely
moves this model. Next: the Gridbook release and PrismaQuant
re-pin; the exporter codec and lane spec on the PrismaQuant side so an
allocation over `TESSERA_*` rungs ships from there; the routed-MoE cell.

## 2026-09-02 (later): the dense failure was the encoder's source model, fixed at the same wire

The 4.07-bpp E4M3/CHANNEL wire on Qwen3-0.6B serves at **KL 0.151** (top-1
78.1%) against 0.470 before, same teacher, image and corpus. That arm is 4.07
wire / **8.0 resident** W8A8: 3.4x better than production NVFP4 GPTQ+JSO at 4.5
wire / 4.5 resident W4A4 (0.511) *across a residency and an A-side*, and 7.4x
behind FP8 RTN at 8.0 / 8.0 (0.0205), where it was 23x -- the second being the
comparison at equal bytes, which Tessera loses on both legs (7.4x at 8-bit here,
1.254x at 4-bit under W4A4). The mechanism: the window table reaches 4.08
sigma0 and a quarter of Qwen's rows (59% of `down_proj`) carry a larger weight,
which clipped in the Hessian-dominant columns; `initial_channel_scale` now
starts such rows at the sigma that puts their max on the reach (the fp16 row
scale the plane already stores), so wire, table, decoder and lane are
untouched. H-weighted census 0.0872 -> 0.0765 (NVFP4 0.0955), 192/196 tensors
ahead of NVFP4; six GLM experts unmoved (0.998x). Branch-metric h-weighting is
a no-op by construction (columns are independent trellises); h enters through
the refit (h^0.75 beats NVFP4 on plain and weighted error on the worst unit)
and LDLQ. Receipt: `docs/measurements/tessera-dense-reach-fix-2026-09-02.md`.
The "23x / 25x / best ~4-bit point" framing above is the pre-fix table.

## 2026-09-02 (the housing): Tessera serves itself

Rob: *"i want all of the serving machinery housed within the Tessera plugin.
If there's infra that we need from gridbook, move it over."*  Done.
`tessera.serving` is Tessera's **own** out-of-tree vLLM plugin -- entry point
`tessera = "tessera.serving:register"` in the `vllm.general_plugins` group,
registering `quant_method: "tessera"`, selected by the checkpoint with **no
enable flag** and one operator knob `TESSERA_SERVE_MODE=resident|streamed`.
Both routes moved (`TESSERA_NVFP4` W4A4, `TESSERA_FP8` W8A8), with the streamed
window decode, the span-2 CUDA decoder, the compile-identity hook, the route
telemetry, the census tool, the tests and a Tessera-owned
`runtime_contract.json`.  Nothing under `src/tessera` imports `gridbook`
(an `ast` test on the import graph, not the substring).  Gridbook withdrew the
lane, the `GRIDBOOK_TESSERA[_MODE]` flag pair and the `TESSERA_*` contract rows
at **contract v15** -- v13/v14 were never released, so it is a withdrawal, not
a break.  PrismaQuant admits the lane through
`prismaquant/tessera_serving_runtime_pin.json` and is **fail-closed** until a
Tessera release tag exists (verified: the table parses, 2 families / 4 cells,
`tessera_lane_attested` False on both rungs).  Cutting that tag is Rob's call.

The plugin requirement is a **field**, not prose: every eligibility cell
carries `requires_plugin: "tessera"` beside `route_status` and
`requires_serve_flags`, refused on both sides if absent, because "this artifact
needs software vLLM does not ship" is a claim about a runtime (principle 14).

**MoE is designed everywhere and served nowhere**: the dispatch matches vLLM's
routed-experts layer before `LinearBase` and *raises* unless the checkpoint
named that prefix in `ignore` (returning `None` would silently hand it
`UnquantizedFusedMoEMethod`); the exporter separates rank-3 packed expert
stacks from 2-D weights so an MoE checkpoint cannot export as "fully
quantized" with BF16 experts; and there is **no `routed_moe` cell** --
`structures: ["dense"]` in a field a gate can read, because no served expert
measurement exists.

**TP is designed in from the start** (Rob, same day): the artifact is
TP-agnostic and the exporter never encodes per rank; a rank cuts at load.
`serving/sharding.py` derives the axis from the sizes vLLM asks for
(`out_size*tp == rows` -> `"row"`; `in_size*tp == columns` -> `"column"`, the
vocabulary `tessera.layout.can_shard` already speaks), never from a class name;
`_shard_unit_for_rank` is the seam (identity at `tp_size == 1`, and above it
`tessera.layout.slice_unit` by name).  The two families part company on the
**initial state** a sliced unit carries: the window family threads it through
the packed plane's existing L-bit pad, which *is* `state_{-1}`, so that family
shards with no kernel change; the span-2 family **refuses**, because its
window's reversed bit order makes a threaded start state unwritten and
untested, and a decoder starting every row at the pinned zero state would
decode a sliced unit wrongly and quietly.  `max_world_size: [1]` is in the
contract: what is built is the seam, not the cut.  *(Corrected 2026-09-02, #7:
the cut is built too — `config` no longer refuses every world above one, and
`sharding.require_axis_supported` refuses the one axis a route cannot start,
published as `tensor_parallel.units[].loader_axes`.  `max_world_size` stays 1
because it is an attestation and no multi-rank serve has been run.)*

**The move is exact, and the one arm that was not is understood.**  Twelve
served arms (three checkpoints x two residency modes x eager/compiled) against
the Gridbook lane's own dumps on the **same inodes** -- `model.safetensors`
hardlinked, only `quant_method` and a new `structure` key differing in
`config.json`.  Route census **112/112** on the declared routes in all four
mode x regime combinations, `other_route_modules: 0`.  Mutual KL: **eleven of
twelve arms at exactly 0.000000 with 100% top-1**.  The twelfth
(`k2-resident-graph`) read 0.017591 -- and it is inductor, not the move: the
identical sources recompiled from an **empty** compile cache serve logits
bit-equal to Gridbook (0.000000 / 100%) and differ from the chain's own build
by 0.017117.  Two builds of one graph, two sets of kernels.  Replaying the
artifact does not test this (the second serve loads the same AOT key and
reproduces it exactly); recompiling does.  For scale, compilation moves that
arm by 0.2442 (plugin) / 0.2445 (Gridbook) against its own eager dump, which
the two runtimes reproduce to three digits, and **eager is 0.000000 across both
runtimes and both residency modes**.

Receipt (censuses, mutual KL vs the Gridbook lane on the same inodes, and the
six sites where the unit slicer lands):
`docs/measurements/tessera-serving-plugin-2026-09-02.md`.
## 2026-09-02 (the third family): a 16-bit route, because the ceiling was the alphabet

The window body stops improving above ~6 bpp because the **E4M3 alphabet** runs
out of values, not because the trellis runs out of shaping. Snap the identical
table to bf16 instead and the same trellis keeps halving: on six GLM routed
experts, output space, geomean, the BF16 window is **0.797x** of the E4M3
window at R=6 and **0.433x** at R=7, and **0.828x of E4M3 one whole rung
above it** — the 16-bit route at R buys more than the 8-bit route at R+1, on
every one of the six. Against EXL3 it holds **0.932-0.955x across R=4..7**,
where E4M3 inverts at 6 (1.198x) and collapses at 7 (2.318x). At R=8 on
L5.gate_proj it is **0.231x of E4M3 and 1.002x of EXL3 K8** — it lands *on*
EXL3's 8-bit point. On six dense Qwen Linears (H-weighted) the same shape:
0.784x at R=6, 0.511x at R=7, and **0.518x of a full FP8 RTN tile at 0.9 bpp
less**. Below R=6 the alphabet is free and costs 0.016 bpp (0.049 on small
dense Linears): the two arms are within 1%, so the menu keeps both.

At **8 bpp** on those six dense Linears the prediction that started this
reproduces on a different model, harness and H: **BF16 0.00783 · E4M3 0.02280
· FP8 RTN 0.02341**, against W1's predicted 0.0079 / 0.0234 / 0.0238 — every
arm **within 3%**, through the real wire. E4M3 spends its alphabet (0.02744 -> 0.02375 ->
0.02280 over R=6,7,8; the last bit buys 4%) while BF16 keeps taking ~1.6-1.8x
a bit. **1% more bytes than a full FP8 tile, 3.0x less error**, and BF16 at R=7 is
1.9x better than E4M3 at R=8 a whole bit cheaper. One caveat kept in view:
`layers.2.mlp.down_proj` still never crosses on the H-weighted axis through
R=8 (1.048x) while reaching 0.23x on plain Frobenius — a reach problem, not an
alphabet one, and the reason (L, sigma) for this grid is stated rather than
searched.

Built, not just measured: `BF16_GRID` (65 536 codes, the code *is* the bf16 bit
pattern, `payload_bits=16`), `BF16_RECIPE` (window, span 1, CHANNEL, L=14),
`materialize_bf16` / `bf16_route.stream_bf16` (both return the *pair* -- tile
plus row scale -- and `*_folded` is the twin's single tile), a pure-torch
streamed decoder,
`--grid BF16` in the exporter with `--stock-twin`, 29 tests. Qwen3-0.6B is
exported at R=6 and R=7 (wire 6.129 / 7.129 bpp) with plain-BF16 twins that a
stock `from_pretrained` loads and generates from; all 196 units are bitwise
`materialize_bf16_folded` of the wire and the twins are structurally the source
checkpoint (311/311 tensors). Two format-level fixes fell out: the ALPHABET
plane now carries the grid's code width (schema §1e, no minor — the grid is
recovered by digest, so an old reader fails closed), and
`calculator.terminal_rate` was charging one byte per table entry for every
grid.

**The route does not fold its row scale, and the twin's fold is priced as the
twin's.** A CHANNEL scale is an output-row factor, so it commutes with the
matmul: the route runs the stock BF16 GEMM on the exact code tile and scales
the output in fp32 (4.04e-7 relative) instead of folding (1.65e-3). Measured on
the six experts the fold is a rate-independent 0.0013-0.0018 on *every* arm
including EXL3's and RTN's — 2.0% of the error at R=4, **15.4% at R=7** — so a
served number taken on the twin is a ceiling, not the route's value. **How big
a ceiling is now measured, and it is small.** The share composes in quadrature,
so 15.4% is a 2.4% gap in squared error, not a 15% one; on the dense Qwen units
it is ~1.2%; and *served* at R=7 the twin's KL is 1.0011x the route's on `all`
and 0.9961x on `confident` — below what the corpus resolves. Issue #45 carries
the high-rung arm that would show it.

Receipt, with the hand-offs for the plugin, the fused kernel's 32 KiB bf16
table, and PrismaQuant's three changes:
`docs/measurements/tessera-bf16-route-2026-09-02.md`.

**The plugin hand-off is taken up (issue #9).** `TESSERA_BF16` is a third
`ROUTES` entry and `serving/bf16_route.py` is its route module: W16A16, the
packed window planes decoded to a bf16 tile in both residency modes,
`torch.mm(..., out_dtype=torch.float32)` and the row scale as an fp32 epilogue,
cross-checked element for element against `tessera.decode.materialize_bf16` at
load. The contract publishes the family (`TESSERA_BF16_K1`, reader rate range
[256, 4096] derived by loading 25 rungs, TP `max_world_size 1`).

**And it is served (contract v5).** Four route censuses on the pinned image --
`resident`/`streamed` crossed with the eager and the compiled forward -- each
record **112 of 112** declared modules on the `TESSERA_BF16` route, in both the
prefill and the decode shape, zero problems, all four emitting the identical
greedy continuation. Served KL-vs-BF16 at `q256 = 1792` (wire 7.129 bpp) is
**0.004923** and **bit-identical between the two residency modes**, against
0.004929 for the folded stock twin vanilla vLLM serves from the same encode.
`attested_rungs_q256` moves `[] -> [1792]` and two `sm_121` dense cells appear,
`decode` and `batch`, `bf16_unquantized` on `torch.mm`. Nothing wider: one
rung, one platform, dense only, no routed-MoE cell, `max_world_size 1` -- every
other rung in the reader's range still resolves `unattested`, which is the rule
holding rather than bending. Footprint as the runtime reports it: `resident`
loads 1.12 GiB, `streamed` 0.71 GiB, the 0.41 GiB being the wire. Receipt:
`docs/measurements/tessera-bf16-route-served-2026-09-02.md`. Still not
selectable from PrismaQuant: `ANCHOR_BUDGET_BITS` refuses `payload_bits >= 16`
on a premise that is TCQ's alone (a window body has no forest).
