# Tessera as one format: the grammar, what is unified, and what is not

**Date:** 2026-09-02. **Status:** the cohesive view Rob asked for
(*"homogenize things where possible between 4-bit and 8-bit so that we have
a grand cohesive view of things instead of piecewise"*). Every number is
measured and linked; the frontier table is filled by
`experiments/tessera_frontier.py` on the production encoder and says so
where a cell is still the experiment's protocol rather than the wire's.

## 1. The grammar

A Tessera unit is one point in a five-axis space, and the 4-bit and 8-bit
formats are the same construction at two points of the first axis:

| axis | what it is | values today | wire |
|---|---|---|---|
| **grid** | the hardware tile × tuple dimension: the set a code decodes to | `E2M1` (16), `E2M1x2` (256 pairs), `E4M3` (256) | `PayloadGrid(values, arity)`, digest in the profile id |
| **body** | the shaping machine: how a column's bits become codes | `TCQ` (rate-½ conv code, 64 states, 4 subsets, span 2), `WINDOW` (bitshift trellis, `2^L` states, stored table) | `BodyKind`, minor 2 |
| **rate** | bits per position, continuous at 1/256 | q256 root, Bresenham per column; TCQ caps at `payload_bits−1`, WINDOW at `payload_bits` | manifest root + schedule |
| **scale plane** | the magnitude field the tile is multiplied by | `S6B` (E8M0/32 + nibble/16), `LUT` (nibble/16 → 16-entry E4M3 table), `CHANNEL` (fp16/row × global) | minor 1 / minor 3 |
| **route** | which MMA executes the decoded tile, and with what activation contract | NVFP4 MMA W4A4 (E2M1 + per-16 plane), FP8 MMA W8A8 (E4M3 + CHANNEL), dequant W?A16 (kernel lane) | not wire: a property of (grid, plane) the spec must carry |

Read in coset-code terms (Forney): every format above is a subset of the
tile product `T^n`. The index rule is the body; the two ways to buy shaping
gain are *memory* (states shared along a column) and *dimension* (tuples).
Where the neighbours sit:

| format | grid | body | plane | memory | dimension |
|---|---|---|---|---|---|
| NVFP4 / FP8 RTN | E2M1 / E4M3 | product code | per-16 E4M3 / per-row | 0 | 1 |
| Gridbook FP4-CB / FP8-CB | E2M1 / E4M3 | fitted vector codebook (product of two 4-tuples) | per-16 / **per-row fp32** | 0 | 8 |
| EXL3 | Gaussian alphabet (not a tile) | bitshift trellis, `2^16` states | rank-1 su/sv | 16 | 1 |
| Tessera TCQ | E2M1x2 / E4M3 | conv code over 4 subsets | per-16 LUT | 6 | 2 / 1 |
| Tessera WINDOW | E2M1x2 / E4M3 | bitshift trellis on the tile | per-16 LUT / per-row | 12–16 | 2 / 1 |

Gridbook's FP8-CB and Tessera-8 over the CHANNEL plane now carry the
**same plane** and decode by the **same operation** (index → table → tile
bytes); they differ in the body's corner — zero memory and eight
dimensions against deep memory and one. Measured on the same rows at 4.0
bpp, the memory corner wins: FP8-CB K32 is 1.28× behind EXL3 in output
space where the window body at L=14 is 0.94× of it
(`tessera8-targets-2026-09-01.md` §4, `tessera-window-body-2026-09-02.md`).

## 2. What is one code today

* **Grid.** `alphabet.PayloadGrid`; the 4-bit and 8-bit tiles differ only
  in `values`/`arity`. The forest builder, the window table, the calculator
  and the reader are grid-generic.
* **Body.** `encode_unit` dispatches on `BodyKind`; both bodies share the
  scale-refit loop, the trellis weighting, the release path and the
  serialiser. The reader resolves either from bytes alone.
* **Rate.** One q256 axis; `export._plan_for` applies the body's cap.
* **Scale plane.** `ScalePlaneKind` with three kinds, one `unit_scale_field`
  every decoder path reads (`decode.py`), one refit discipline (landed on
  the stored word: S6b words, LUT bytes, fp16 rows).
* **Recipe.** `export.wire_recipe(grid, q256)` is the single statement of
  which point of the grammar the exporter writes for a grid; the exporter,
  the config, the merge guard and (via the seam) PrismaQuant read it. Today
  every grid resolves to `TCQ_RECIPE` (span 2 over LUT16).
* **Accounting.** `calculator.terminal_rate` prices every combination
  (window table inline, CHANNEL rows inline) and agrees with the built
  artifact to the bit (`tests/test_channel_plane.py`,
  `tests/test_window_body.py`).
* **Materialisers.** `materialize_nvfp4` (E2M1 + block plane → the stock
  NVFP4 pair) and `materialize_fp8` (E4M3 + CHANNEL → the stock per-channel
  FP8 pair): both tiles have a stock lane that a runtime which has never
  heard of Tessera serves.

## 3. What is still piecewise, and the plan for each

| seam | state | plan |
|---|---|---|
| **Kernel lane** | decodes the span-2 TCQ wire over LUT16 (bit-exact, 0.0524 ms) **and the window body over LUT16, S6b and CHANNEL** (bit-exact against `read_unit_artifact`; `f274b8d`, `8489634`). The window GEMV costs the same as span-2 through L=14 and 1.83–1.86× at L=16, where the per-unit 2^L table stops fitting (`experiments/results/tessera_kernel_window_table_sweep.json`) | an FP8-tile GEMM path (decode to E4M3 bytes + per-row scale, feed the FP8 MMA); for L=16, shared-memory table residency or one table per layer |
| **Encoder speed** | reference window Viterbi ~30 s / 150 s / 13 min per 2048×4096 at L=12/14/16 | fused Triton Viterbi in flight (worker `a812f218`), bit-exact against the reference, accepted on profiler + power evidence |
| **PrismaQuant spec** | reads `wire_recipe` (PrismaQuant `b02d8b2`): every accountant takes the recipe; the route is derived from (grid base, plane) with the activation quantiser and capability floor taken by reference from the NVFP4 / FP8_E4M3 registry rows; family bounds follow the body's cap. Shape-dependent terms (CHANNEL rows, window table) are priced exactly where a shape exists and **refused** in the shape-free `FormatSpec` rate | shape-aware `bits_for_shape` on the spec and in every byte gate, so a shape-dependent recipe synthesizes (in flight); then the E4M3 flip |
| **Measurement** | four harnesses, three protocols (per-channel and per-16 fp32 arms re-implement the encoder) | `experiments/tessera_frontier.py`: every (grid, body, plane, rate) point through the production encoder, one JSON, one table, EXL3/Gridbook/NVFP4/FP8 comparators at matched bpp |
| **Per-grid defaults** | `wire_recipe(grid, q256)`: E4M3 → window over CHANNEL, L=14, every rung; E2M1x2 → window over LUT16, L=12, below its cap (q256 < 896) and the coset trellis at it; E2M1 → the coset trellis. Flipped 2026-09-02 once the kernel lane decoded the body and the fused Viterbi encoded it. | E2M1 under the window body, and L=14 below the E2M1x2 cap, are measurements to run |

## 4. The frontier

Six GLM-5.3-Flash routed experts (L5/L20/L42 × gate/up), last 1024 capture
rows held out, geometric means. `out` = weight-only output-space error,
`a4`/`a8` = executed error under the served NVFP4 / FP8 activation
quantisers. Ratios are against EXL3 at the same bpp (log-linear between its
rungs). **Protocol column:** *wire* = production `encode_linear` →
`read_unit_artifact`; *exp* = the experiment's own encoder (identical
construction, its own plane code).

| bpp | arm | plane | protocol | out | vs EXL3 out | served | vs EXL3 at the same activation | vs EXL3 W4A16 |
|---|---|---|---|---|---|---|---|---|
| 2.5 | E2M1x2 TCQ span 2 (the wire before the flip) | LUT16 | wire | 0.2635 | 1.401× | a4 0.2763 | 1.317× | 1.469× |
| 2.5 | E2M1x2 window L=12 | LUT16 | wire | 0.1982 | **1.056×** | a4 0.2155 | 1.029× | 1.149× |
| 3.0 | E2M1x2 TCQ span 2 (the wire before the flip) | LUT16 | wire | 0.1817 | 1.357× | a4 0.2004 | 1.261× | 1.497× |
| 3.0 | E2M1x2 window L=12 | LUT16 | wire | 0.1417 | **1.061×** | a4 0.1654 | 1.042× | 1.238× |
| 3.0 | E4M3 window L=12 | CHANNEL | wire | 0.1274 | **0.957×** | a8 0.1296 | **0.958×** | **0.973×** |
| 3.5 | E2M1x2 TCQ span 2 (the wire before the flip) | LUT16 | wire | 0.1364 | 1.431× | a4 0.1608 | 1.220× | 1.687× |
| 3.5 | E2M1x2 window L=12 | LUT16 | wire | 0.1044 | **1.098×** | a4 0.1350 | 1.025× | 1.420× |
| 4.0 | E2M1x2 TCQ span 2 (at its cap: the wire) | LUT16 | wire | 0.0793 | **1.170×** | a4 0.1169 | 1.067× | 1.723× |
| 4.0 | E2M1x2 window L=12 | LUT16 | wire | 0.0842 | 1.244× | a4 0.1202 | 1.099× | 1.776× |
| 4.0 | E2M1x2 window L=14 | LUT16 | wire | 0.0824 | 1.227× | a4 0.1189 | 1.091× | 1.771× |
| 4.0 | E4M3 TCQ span 2 (the wire before the flip) | LUT16 | wire | 0.0812 | 1.197× | a8 0.0847 | 1.177× | 1.248× |
| 4.0 | E4M3 TCQ span 2 | CHANNEL | wire | 0.0957 | 1.414× | a8 0.0986 | 1.374× | 1.457× |
| 4.0 | E4M3 window L=12 | LUT16 | wire | 0.0757 | **1.119×** | a8 0.0794 | 1.106× | 1.173× |
| 4.0 | E4M3 window L=14 | LUT16 | wire | 0.0733 | 1.091× | a8 0.0770 | 1.081× | 1.147× |
| 4.0 | E4M3 window L=12 | CHANNEL | wire | 0.0665 | **0.985×** | a8 0.0707 | **0.987×** | **1.047×** |
| 4.0 | E4M3 window L=14 | CHANNEL | exp, pinned | 0.0632 | 0.938× | a8 0.0676 | 0.946× | 1.003× |
| 4.0 | E4M3 window L=14 | CHANNEL | **wire, the default** | 0.0629 | **0.940×** | a8 0.0673 | **0.946×** | **1.005×** |
| 4.5 | NVFP4 GPTQ+JSO | per-16 | exp | — | — | a4 0.1188 | — | 1.53× (at 4.0 bytes) |
| 5.0 | E4M3 TCQ span 2 (the wire before the flip) | LUT16 | wire | 0.0425 | 1.232× | a8 0.0488 | 1.161× | 1.413× |
| 5.0 | E4M3 window L=12 | LUT16 | wire | 0.0406 | 1.177× | a8 0.0471 | 1.122× | 1.367× |
| 5.0 | E4M3 window L=12 | CHANNEL | wire | 0.0349 | **1.016×** | a8 0.0423 | 1.011× | 1.232× |
| 5.0 | E4M3 window L=14 | CHANNEL | **wire, the default** | 0.0323 | **0.947×** | a8 0.0402 | **0.964×** | 1.179× |
| 6.0 | E4M3 window L=12 | CHANNEL | wire | 0.0218 | 1.240× | a8 0.0324 | 1.090× | 1.842× |
| 8.0 | FP8 per-channel RTN (E4M3 floor) | per-row | wire | 0.0215 | (EXL3 K8 0.0048) | a8 0.0322 | 1.318× | — |

The *wire* rows are `experiments/tessera_frontier.py` on the six experts
(`experiments/results/tessera_frontier.{json,log,stdout}`, run 2026-09-02
on sparklina; the tree was the 23:12 rsync of the working tree that became
Tessera `1675c24` — sparklina holds no git checkout, which is why the JSON
records `git: unknown`; the kernel-lane commits after it do not touch the
encoder), window arms at L=12, `scale_refit=4` (the default), no LDLQ.
EXL3 rows are the reference quantiser's reconstructions on the same rows
(K=2..8; K4 0.06736, K5 0.03429, geometric means). "Served" is the
executed error under the route's activation quantiser (a4 = NVFP4, a8 =
FP8); "vs EXL3 at the same activation" divides it by EXL3's own error
under that same quantiser (the hypothetical EXL3@A4 / EXL3@A8 baselines),
and "vs EXL3 W4A16" divides it by EXL3's weight-only error, which is what
EXL3 executes. The L=14 `exp, pinned` row is the pinned-tile experiment
(`tessera_bitshift_tile`); the L=14 `wire` rows are
`experiments/results/tessera_frontier_L14.json` — the same six experts,
the same harness and the same EXL3 reference rows, run on sparklina from a
tree rsynced before the fused-Viterbi merge, `8489634` (the record says `git: unknown`
because sparklina has no checkout; the reference encoder took 108–138 s per
arm, and the fused kernel is bit-exact with it). Ratios are geometric means
of the six per-tensor ratios against EXL3 interpolated to the arm's bpp.

**What the table says.** (i) Below the cap the window body owns the
E2M1x2 ladder on the true wire, not just the tile: 1.06–1.10× behind EXL3
at 2.5–3.5 bpp where the coset trellis is 1.36–1.43×. At 4.0 the coset
trellis at its cap (1.170×) beats the window at L=12 (1.244×) and at L=14
(1.227×, per expert 1.21–1.26×): the tuple sweep's "L=14 levels them at
equal payload" did not survive the wire's LUT16 plane, so the cap stays on
the coset trellis and the window owns only the rungs below it. (ii) On E4M3
the plane and the body compound: the wire before the flip (coset trellis over
LUT16) is 1.20× behind EXL3 K4; the window over the same plane 1.12×; the
window over the CHANNEL plane **0.985× at 4.0 and 0.957× at 3.0** — the
production encoder, at L=12, before LDLQ, decoded bit-exactly by the kernel
lane — level at 5.0 (1.016×) and 1.24× behind at 6.0, where the E4M3 floor
(per-channel RTN 0.0215 at 8 bpp against EXL3 K6 0.0175 at 6) caps every
8-bit tile. The coset trellis over the CHANNEL plane is the worst E4M3 arm
(1.41×): the sub-cap E4M3 forest builder, not the plane. (iii) As served,
E4M3 + CHANNEL + window executes W8A8 at 0.973× EXL3's W4A16 number at
3.0 bytes on disk and 1.047× at 4.0 — the best executed contract we have;
under EXL3's own activation quantiser (EXL3@A8, the standing baseline's
8-bit form) it is 0.958× and 0.987× — all at L=12. At L=14, the default
the flip ships, the 4.0-byte row is **level with EXL3's weight-only number
as served (1.005×)** and 0.946× under EXL3@A8; the 5.0-byte row is 1.18×
as served and 0.964× under EXL3@A8. The 3.0-byte row (L=12) is the only
executed arm strictly under EXL3's weight-only number.
"Bytes" here are bytes on disk: the stock per-channel FP8 route reached
through `materialize_fp8` holds 8 bits per weight resident after load, and
holding 4 bits resident under W8A8 needs the decode-to-FP8-tile GEMM the
kernel lane lists as a plan (§3), which on a unified-memory box is the
difference that matters. Every E2M1 arm pays the NVFP4 activation leg (a4
≈ 1.4–1.5× its own weight-only error at 4.0), which is why W4A4 loses as
served at every rate and why W8A8 is the 4.0-byte recipe. (iv) Gridbook's
corner of the grammar loses to the memory corner at every rate measured;
its rows are below, from the follow-ups harness.

**Compensation and the Gridbook rows** (`experiments/tessera_vs_exl3_followups.py`,
`experiments/results/tessera_vs_exl3_followups.{json,log}`, 37 arms × 6
tensors on the same held-out rows; arithmetic mean over tensors, ratios
per tensor against EXL3 log-linear between rungs, EXL3 K=2/3 added so
nothing sub-4 is extrapolated; the EXL3 K4 re-score reproduces
`tessera8_targets.json` bit-identically). LDLQ here is `compensate.py`'s
column-sequential feedback with H from the first 7168 capture rows,
`regularize_hessian` σ ∈ {1, 3}, block 32 — on the *default* TCQ wire
and on the scalar per-channel Tessera-8, not yet on the window body.

| bpp | arm (follow-ups harness) | out | vs EXL3 out (min–max) | served | vs EXL3 at the same activation |
|---|---|---|---|---|---|
| 4.000 | Tessera-4 wire at the cap (TCQ span 2, LUT16, refit 4) | 0.0798 | 1.169× (1.153–1.202) | a4 0.1176 | 1.066× |
| 4.000 | Tessera-4 wire at the cap + LDLQ σ=3 (whole-unit re-encode) | 0.0754 | **1.102× (1.096–1.110)** | a4 0.1146 | 1.039× |
| 4.008 | Tessera-8 R=4 per-channel scalar (no body) | 0.0780 | 1.147× (1.136–1.165) | a8 0.0816 | 1.132× |
| 4.008 | Tessera-8 R=4 per-channel + LDLQ σ=3 (row scale frozen) | 0.0721 | **1.059× (1.046–1.067)** | a8 0.0760 | 1.052× |
| 5.008 | Tessera-8 R=5 per-channel scalar (no body) | 0.0410 | 1.184× (1.169–1.203) | a8 0.0476 | 1.127× |
| 5.008 | Tessera-8 R=5 per-channel + LDLQ σ=3 (row scale frozen) | 0.0379 | **1.094× (1.075–1.102)** | a8 0.0449 | 1.064× |
| 2.281 | Gridbook FP4-CB K16 +imatrix | 0.2770 | 1.262× (1.240–1.308) | a4 0.2893 | 1.214× |
| 2.781 | Gridbook FP4-CB K20 +imatrix | 0.1968 | 1.258× (1.246–1.285) | a4 0.2143 | 1.186× |
| 3.281 | Gridbook FP4-CB K24 +imatrix | 0.1419 | 1.274× (1.256–1.295) | a4 0.1659 | 1.153× |
| 4.008 | Gridbook FP8-CB K32 +imatrix | 0.0869 | 1.279× (1.267–1.300) | a8 0.0902 | 1.251× |
| 4.008 | Gridbook FP8-CB K32 +imatrix +LDLQ (as the gate ships it) | 0.1069 | 1.582× (1.267–1.767) | a8 0.1096 | 1.528× |
| 5.008 | Gridbook FP8-CB K40 +imatrix | 0.0460 | 1.329× (1.310–1.354) | a8 0.0519 | 1.231× |
| 6.008 | Gridbook FP8-CB K48 +imatrix | 0.0240 | 1.353× (1.304–1.401) | a8 0.0340 | 1.136× |

σ=3 beats σ=1 on every leg of every trellis arm (σ=0.025, EXL3's value on
its rotated basis, was already measured harmful on ours). On the TCQ wire
the honest arm is the whole-unit re-encode of the compensated target,
because the LUT16 plane is fit over whatever the encoder is handed and a
32-column slice encode is not the wire; that makes 1.169× → 1.102× a lower
bound on the gain (the stitched diagnostic, a different wire with 128
LUTs, reaches 1.082×). On per-channel Tessera-8 the columns are
independent given the frozen row scale (slice-exactness asserted to 0.0),
so 1.147× → 1.059× at R=4 and 1.184× → 1.094× at R=5 are the protocol's
own numbers; under EXL3@A4 both LDLQ'd per-channel arms sit at 1.01–1.02×.
Gridbook's production imatrix (`moe_imatrix.py` E[x²] per column) is
worth ≤0.5% on every rung, so the earlier "as Gridbook ships it would be
better than this" caveat is null; its LDLQ arm *as the gate ships it*
regresses every rung (K32 out 0.0869 → 0.1069) because the gate's
hold-out is a random half of the same 7168 fit rows — it certifies fit
error, not generalisation — and the fit is 3584 rows of a 4096-column
Hessian at 1% damping: +53% weight error, the regime
`tessera-ldlq-regulariser` already measured harmful. Production runs
`PRISMAQUANT_CB_LDLQ=0`, so the imatrix rows are Gridbook as it ships.
Against the window body over CHANNEL (0.957× / 0.985× / 1.016× / 1.240× at
3/4/5/6 bpp, before compensation) Gridbook FP8-CB is 1.28× / 1.33× / 1.35×
at 4/5/6 and FP4-CB 1.26–1.27× at 2.3–3.3; the earlier note that Gridbook
wins at 6 and below 4 was measured against the scalar Tessera-8 and is
superseded by the window body. Composition of LDLQ with the window body
is the open arm: measured on neither tile.

## 5. The product the allocator sees

One family name per grid, `TESSERA_<base>_K<arity>_R<q256>`, on a
continuous ladder — `E2M1_K2` 1.0–4.0 bpp, `E4M3_K1` 1.5–7.5 (8.0 under
the window body) — with the recipe resolved from `(grid, rung)`, the bytes
priced by the calculator, and the activation contract carried per (grid,
plane): an allocator comparing an E2M1x2 rung with an E4M3 rung at the same
bytes is comparing W4A4 against W8A8, and at 4.0 bytes those differ 1.44×
as served (`tessera8-targets`). The cost model prices what the route
executes (PrismaQuant principles 8/9/14), never the weight leg alone.

The recipe is per rung, and the checkpoint says so. `wire_recipe(grid,
q256)` is the one function; the exporter resolves it per unit (the caller's
explicit overrides apply on top, for every rung alike), and the config
records the whole table as contiguous `q256` ranges under `wire.recipes`
(body, span, plane, window table parameters, modelled spreads). The flat
`body` / `scale.plane` / `trellis.span` keys are a projection of that table
kept for readers of the earlier configs and read `per-rung` when the table
varies, so a reader that does not know the table cannot mistake one body
for the other. `encode_settings_from_config(config, q256)` replays a unit
at its own rung's meaning and refuses to guess when the table varies and no
rung is named; the merge guard compares the table across parts whenever the
parts carry it. The flip itself landed the same day, once both mechanical
gates had closed: the kernel lane decodes the window body bit-exactly over
every plane, and the fused Viterbi (`window_viterbi.py`, 15× at L=12, 26×
at L=14, bit-exact, profiled and powered per principle 15) encodes it at
1.1 s per 2048×4096 pass at L=14 — the reference took 30 s. `wire_recipe`
now returns the window body over CHANNEL at L=14 on E4M3 at every rung, the
window over LUT16 at L=12 on E2M1x2 below its cap, and the coset trellis at
the E2M1x2 cap and on E2M1.

## 6. What this does not unify, on purpose

* **The TCQ body is not a window table.** A span-2 super-symbol's state
  would be 2R+7 bits wide; the structured coset table still wins at the
  E2M1x2 cap; both bodies stay, per unit.
* **Gridbook's wire is not Tessera's wire.** Its 8-tuple codebook would
  need k-vector table entries (a VALUES plane). Same grammar, same plane,
  same decode form — one wire only where Gridbook still wins after the
  frontier run.
* **Rotation** stays a `RotationState` nothing selects: dead on expert
  inputs on both tiles (Gaussian, kurtosis 3.0–3.2, block size not the
  cause); one open weight-space cell on high-kurtosis attention rows under
  the per-channel plane, to be measured as a frontier arm, not assumed.
