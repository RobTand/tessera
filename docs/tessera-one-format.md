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
| **Per-grid defaults** | `wire_recipe` returns the TCQ recipe for every grid | flip E4M3 to window over CHANNEL and E2M1x2 sub-cap to window over LUT16 when the two workers land; E2M1x2 at the cap stays TCQ |

## 4. The frontier

Six GLM-5.3-Flash routed experts (L5/L20/L42 × gate/up), last 1024 capture
rows held out, geometric means. `out` = weight-only output-space error,
`a4`/`a8` = executed error under the served NVFP4 / FP8 activation
quantisers. Ratios are against EXL3 at the same bpp (log-linear between its
rungs). **Protocol column:** *wire* = production `encode_linear` →
`read_unit_artifact`; *exp* = the experiment's own encoder (identical
construction, its own plane code).

| bpp | arm | plane | protocol | out | vs EXL3 out | as served | served vs EXL3 W4A16 |
|---|---|---|---|---|---|---|---|
| 2.5 | E2M1x2 TCQ span 2 (default wire) | LUT16 | wire | 0.2635 | 1.401× | a4 0.2763 | 1.47× |
| 2.5 | E2M1x2 window L=12 | LUT16 | wire | 0.1982 | **1.056×** | a4 0.2155 | 1.15× |
| 3.0 | E2M1x2 TCQ span 2 (default wire) | LUT16 | wire | 0.1817 | 1.357× | a4 0.2004 | 1.50× |
| 3.0 | E2M1x2 window L=12 | LUT16 | wire | 0.1417 | **1.061×** | a4 0.1654 | 1.24× |
| 3.0 | E4M3 window L=12 | CHANNEL | wire | 0.1274 | **0.957×** | a8 0.1296 | **0.973×** |
| 3.5 | E2M1x2 TCQ span 2 (default wire) | LUT16 | wire | 0.1364 | 1.431× | a4 0.1608 | 1.69× |
| 3.5 | E2M1x2 window L=12 | LUT16 | wire | 0.1044 | **1.098×** | a4 0.1350 | 1.42× |
| 4.0 | E2M1x2 TCQ span 2 (default wire, at its cap) | LUT16 | wire | 0.0794 | **1.170×** | a4 0.1169 | 1.72× |
| 4.0 | E2M1x2 window L=12 | LUT16 | wire | 0.0842 | 1.244× | a4 0.1202 | 1.78× |
| 4.0 | E4M3 TCQ span 2 (default wire) | LUT16 | wire | 0.0812 | 1.197× | a8 0.0847 | 1.25× |
| 4.0 | E4M3 TCQ span 2 | CHANNEL | wire | 0.0957 | 1.414× | a8 0.0986 | 1.46× |
| 4.0 | E4M3 window L=12 | LUT16 | wire | 0.0757 | 1.119× | a8 0.0794 | 1.17× |
| 4.0 | E4M3 window L=12 | CHANNEL | wire | 0.0665 | **0.985×** | a8 0.0707 | **1.047×** |
| 4.0 | E4M3 window L=14 | CHANNEL | exp, pinned | 0.0632 | 0.938× | a8 0.0676 | 1.003× |
| 4.0 | Gridbook FP8-CB K32 | per-row | exp (provisional) | 0.0862 | 1.28× | a8 0.0904 | 1.33× |
| 4.5 | NVFP4 GPTQ+JSO | per-16 | exp | — | — | a4 0.1188 | 1.53× (at 4.0 bytes) |
| 5.0 | E4M3 TCQ span 2 (default wire) | LUT16 | wire | 0.0425 | 1.232× | a8 0.0488 | 1.41× |
| 5.0 | E4M3 window L=12 | LUT16 | wire | 0.0406 | 1.177× | a8 0.0471 | 1.37× |
| 5.0 | E4M3 window L=12 | CHANNEL | wire | 0.0349 | **1.016×** | a8 0.0423 | 1.23× |
| 6.0 | E4M3 window L=12 | CHANNEL | wire | 0.0218 | 1.240× | a8 0.0324 | 1.84× |
| 8.0 | FP8 per-channel RTN (E4M3 floor) | per-row | wire | 0.0215 | (EXL3 K8 0.0048) | a8 0.0322 | — |

The *wire* rows are `experiments/tessera_frontier.py` on the six experts
(`experiments/results/tessera_frontier.{json,log,stdout}`, run 2026-09-02
on sparklina at Tessera `40ae011`, window arms at L=12, `scale_refit=1`,
no LDLQ). EXL3 rows are the reference quantiser's reconstructions on the
same rows (K=2..8; K4 0.06736, K5 0.03429). The L=14 wire arms at 4.0 and
5.0 are running; the L=14 CHANNEL row above is the pinned-tile experiment.

**What the table says.** (i) Below the cap the window body owns the
E2M1x2 ladder on the true wire, not just the tile: 1.06–1.10× behind EXL3
at 2.5–3.5 bpp where the coset trellis is 1.36–1.43×. At 4.0 the coset
trellis at its cap (1.170×) beats the L=12 window (1.244×); the six-expert
tuple sweep says L=14 levels them at equal payload, and the wire pays the
window no label bits, so the L=14 wire arm decides the cap. (ii) On E4M3
the plane and the body compound: the default wire (coset trellis over
LUT16) is 1.20× behind EXL3 K4; the window over the same plane 1.12×; the
window over the CHANNEL plane **0.985× at 4.0 and 0.957× at 3.0** — the
production encoder, at L=12, before LDLQ, decoded bit-exactly by the kernel
lane — level at 5.0 (1.016×) and 1.24× behind at 6.0, where the E4M3 floor
(per-channel RTN 0.0215 at 8 bpp against EXL3 K6 0.0175 at 6) caps every
8-bit tile. The coset trellis over the CHANNEL plane is the worst E4M3 arm
(1.41×): the sub-cap E4M3 forest builder, not the plane. (iii) As served,
E4M3 + CHANNEL + window at 4.0 bytes executes W8A8 at 1.047× EXL3's W4A16
(0.973× at 3.0 bytes): the best executed contract we have, and the only
one at or under EXL3's weight-only number. Every E2M1 arm pays the NVFP4
activation leg (a4 ≈ 1.4–1.5× its own weight-only error at 4.0), which is
why W4A4 loses as served at every rate and why W8A8 is the 4.0-byte
recipe. (iv) Gridbook's corner of the grammar loses to the memory corner
at 4.0 (1.28× vs 0.985×); its 6-bpp and ≤3.3-bpp claims are re-derived from
the follow-up worker's finished file, not from this run's copies.

## 5. The product the allocator sees

One family name per grid, `TESSERA_<base>_K<arity>_R<q256>`, on a
continuous ladder — `E2M1_K2` 1.0–4.0 bpp, `E4M3_K1` 1.5–7.5 (8.0 under
the window body) — with the recipe resolved from `(grid, rung)`, the bytes
priced by the calculator, and the activation contract carried per (grid,
plane): an allocator comparing an E2M1x2 rung with an E4M3 rung at the same
bytes is comparing W4A4 against W8A8, and at 4.0 bytes those differ 1.44×
as served (`tessera8-targets`). The cost model prices what the route
executes (PrismaQuant principles 8/9/14), never the weight leg alone.

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
