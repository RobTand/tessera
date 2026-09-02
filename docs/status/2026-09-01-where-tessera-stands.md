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
the same bytes. At the E2M1x2 cap the structured coset table stays better
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
