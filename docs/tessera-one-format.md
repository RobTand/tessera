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
| **PrismaQuant spec** | `synthesize_tessera_spec` calls every rung weight-only at sm80; `wire_overhead_q256` prices no window table and no CHANNEL plane; `render_tessera_weight` hardcodes the TCQ body | read `wire_recipe`; carry the activation contract per (grid, plane): E2M1 + block plane = W4A4 on sm120, E4M3 + CHANNEL = W8A8 on sm89; price the table and the rows |
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
| 3.0 | E2M1x2 window L=16 (K=5) | per-16 fp32 | exp | 0.1738 (L5.gate) | — | a4 | — |
| 3.5 | E2M1x2 window L=16 (K=6) | per-16 fp32 | exp | 1.34× better than TCQ | — | a4 | — |
| 4.0 | E2M1x2 TCQ span 2 (default wire) | LUT16 | wire | 0.0841 (L5.gate) | 1.176× | a4 | 1.070× (W4A4) |
| 4.0 | E4M3 TCQ span 2 | LUT16 | wire | 0.0812 | 1.206× | a8 0.0847 | 1.257× |
| 4.0 | E4M3 window L=12 | LUT16 | wire | 0.0757 | 1.124× | a8 0.0794 | 1.179× |
| 4.0 | E4M3 window L=12 | CHANNEL | exp, pinned | 0.0664 | **0.986×** | a8 0.0706 | 1.048× |
| 4.0 | E4M3 window L=14 | CHANNEL | exp, pinned | 0.0632 | **0.938×** | a8 0.0676 | **1.003×** |
| 4.0 | Gridbook FP8-CB K32 | per-row | exp | 0.0862 | 1.28× | a8 0.0904 | 1.33× |
| 4.5 | NVFP4 GPTQ+JSO | per-16 | exp | — | — | a4 0.1188 | 1.53× (at 4.0 bytes) |
| 5.0 | E4M3 window L=14 | CHANNEL | exp, pinned | 0.0339 | **0.989×** | a8 0.0415 | 1.21× |
| 8.0 | FP8 per-channel RTN (E4M3 floor) | per-row | exp | 0.0189 | (EXL3 K8 0.0051) | a8 0.0306 | — |

The production-encoder rows for every cell above, plus the E2M1x2 window
arms on the true wire and the E4M3 3.0/6.0 rungs, are written by
`experiments/tessera_frontier.py` into `experiments/results/tessera_frontier.{json,log}`
and folded in here when the run lands.

**What the table says.** (i) Below 4.0 the window body owns the E2M1x2
ladder; at the cap the coset trellis does. (ii) On E4M3 the plane is worth
as much as the body: the same window over the LUT16 plane is 1.14× behind
the same window over the CHANNEL plane at equal bytes. (iii) At 4.0 bytes
as served, E4M3 + CHANNEL + window is the best executed contract we have
and is level with EXL3's W4A16 on an FP8 tensor core; at 5.0 and above the
FP8 activation leg is the floor and EXL3's W4A16 pulls ahead as served.
(iv) Gridbook's corner of the grammar loses to the memory corner at 4.0;
its remaining claims (6 bpp on E4M3, ≤3.3 bpp on E2M1) are the frontier
run's remaining cells.

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
