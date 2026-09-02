# The window body: a bitshift trellis on the hardware tile

**Date:** 2026-09-02. **Scripts:** `experiments/tessera_bitshift_tile.py`
(E4M3, per-channel protocol; results `experiments/results/tessera_bitshift_tile.json`,
`tessera_bitshift_tile_computed.json` for the hash-table arm,
`tessera_bitshift_tile_pinned.json` for the pinned-start re-run,
`tessera_bitshift_tile_L16_R{4,5}.json` for the L=16 arms run on sparklina)
and `experiments/tessera_bitshift_tuple.py` (E2M1x2, per-16 protocol;
`tessera_bitshift_tuple.json`). **Tensors:** six GLM-5.3-Flash routed
experts (L5/20/42 × gate/up, expert 0, 2048×4096). **Eval:** the last 1024
rows (two documents) of the 16-document pread capture, held out; the served
activation quantisers' global scales are fit on the first 14. **Metric:**
output-space weight leg `|x(ŵ−w)ᵀ|/|xwᵀ|` (`out`), plus the executed error
under the served NVFP4 activation quantiser (`a4`) and the vLLM FP8 dynamic
per-token quantiser (`a8`); geometric means over tensors. No LDLQ on any
Tessera arm; the EXL3 arms are the reference quantiser's own output on the
same rows (`exl3_reference_quantise.py`, LDLQ'd, W4A16).

## Why this was run

`tessera-theoretical-limits-2026-09-01.md` put the encoder at 1.155× the
E2M1x2 tile bound and closed the weight-space levers on the shipping wire;
`tessera-rate-plane-frontier` and `tessera-index-plane` spent the plane's
redundancy; `tessera8-targets` left Tessera-8 1.13× behind EXL3 under
W?A8. What remained was the trellis itself. EXL3's advantage is not its
lattice — on this protocol its Gaussian-oracle ratio is 1.068 at K=4 — it
is the size of its trellis state: QTIP's bitshift trellis (Tseng et al.
2024) uses `2^L` states with `L = 16`, where Tessera's rate-1/2
convolutional code has 64. Tessera's constraint is that every
reconstruction must be a code on the E2M1/E4M3 tile so the 4-bit and 8-bit
tensor-core paths execute it natively (`tessera-fp4-native-constraint`).
The question: does a large-state trellis keep its gain when its table is
snapped to the tile?

**The construction.** A column's stream is `R` bits per position. The
state at position `t` is the last `L` bits of the stream (`state_t =
((state_{t−1} << R) | bits_t) mod 2^L`, `state_{−1} = 0`), and the code at
`t` is `TABLE[state_t]` — a `2^L`-entry table of grid codes. The Viterbi
is exact: every state has `2^R` predecessors sharing its low `L−R` bits,
so one step is a `[2^R, 2^(L−R)]` minimum plus a branch cost per state.
The table is a seeded permutation of equal-mass quantiles of the modelled
source, snapped to the nearest grid code (pairs at arity 2). No
convolutional code, no anchor forest, no completion axis: the shaping is
the `L − R` bits of history every position shares with its predecessors.

## Result

**At 4.0 bpp on E4M3 under a per-channel plane the window body beats EXL3
K4 in output space, and executed W8A8 is level with EXL3's W4A16 at the
same bytes.** Six-tensor geomeans, free start, table not charged (see the
caveats for both):

| arm (per-channel E4M3, LS refit ×2) | bpp | wt | out | a4 | a8 |
|---|---|---|---|---|---|
| EXL3 K=4 (LDLQ, W4A16) | 4.012 | 0.08629 | **0.06736** | 0.10912 | 0.07146 |
| Tessera-8 R=4, 4-subset conv trellis (today) | 4.008 | 0.08556 | 0.07748 | 0.11562 | 0.08108 |
| window L=10 | 4.013 | 0.08024 | 0.07274 | 0.11242 | 0.07656 |
| window L=12 | 4.014 | 0.07294 | 0.06599 | 0.10822 | 0.07020 |
| window L=14 | 4.015 | 0.06931 | **0.06271** | 0.10629 | **0.06712** |
| EXL3 K=5 | 5.012 | 0.04387 | **0.03429** | 0.09256 | 0.04179 |
| Tessera-8 R=5 today | 5.008 | 0.04504 | 0.04071 | 0.09509 | 0.04721 |
| window L=12 | 5.014 | 0.03980 | 0.03607 | 0.09316 | 0.04329 |
| window L=14 | 5.015 | 0.03726 | **0.03372** | 0.09228 | 0.04135 |

Ratios at 4.0 bpp: window L=14 is **1.235× better than today's trellis**
and **1.074× better than EXL3 K4** on the weight leg; as served, Tessera-8
W8A8 (`a8` 0.06712) against EXL3 W4A16 (`out` 0.06736) is **1.004×** —
level, at the same bytes, on the FP8 tensor core instead of a dequant
kernel. At 5.0 bpp the weight leg beats EXL3 K5 by 1.017×, but under A8
EXL3's W4A16 is 1.206× better as served: the activation leg dominates at 5
bits and the FP8 activation error is the floor (`tessera8-targets`).

**L=16** (sparklina, four of six tensors so far, free start): R=4 `out`
L5.gate 0.06484 (today 0.08298, EXL3 0.07237), L5.up 0.06686 (0.08526,
0.07465), L20.gate 0.06227 (0.07896, 0.06925), L20.up 0.06519 (0.08303);
R=5 L5.gate 0.03489, L5.up 0.03600, L20.gate 0.03304, L20.up 0.03507.
About 3% over L=14 per tensor, for four times the table and the encode.

### Three caveats, each measured

1. **Start state.** The sweep's Viterbi started free (any initial state,
   charged `L/rows` bpp). The wire pins state 0, as the decoder assumes.
   Pinned re-run (`--pinned-start`, table charged in `bpp`): L5.gate R=4
   L=14 `out` 0.06734 / `a8` 0.07200 at 4.023 bpp; L5.up 0.06963 / 0.07448;
   L5.gate R=5 L=14 0.03633 / 0.04438. The per-tensor pinned/free ratio and
   the six-tensor pinned geomeans are **pending** the run's end and are
   appended below; the headline is quoted free-start until then.
2. **Table cost.** The `2^L` table is charged on the ALPHABET plane, per
   unit: 0.0039 bpp at L=12, 0.0156 at L=14, 0.0625 at L=16 on a 2048×4096
   unit. Against EXL3's rate–distortion slope (~3.3% per 0.05 bpp) the L=16
   margin shrinks by ~4%; L=14's by ~1%. The pinned `bpp` column carries it.
3. **Scale plane.** The E4M3 arms use one fp32 scale per output row (LS
   refit): the layout the served FP8 W8A8 path consumes, and **not yet a
   Tessera `ScalePlaneKind`**. On the wire's per-16 LUT plane a 256×1024
   Gaussian smoke had window L=12 level with the TCQ body at R=4 (0.0680
   vs 0.0683): the amax-bounded per-16 source is where the anchor forest
   already fits well, and the window's gain is a property of the plane as
   much as of the trellis. The per-channel plane is the next wire addition.

## The Gaussian oracle: what the tile costs a large-state trellis

Ratio of RMS error to the Shannon bound `2^−R` on a 2048×512 standard
Gaussian, each arm at its best global scale (EXL3 K4 1.068, K5 1.085 on
this protocol; `tessera_bitshift_tile.json["gaussian"]`):

| R=4 arm | ratio | R=5 arm | ratio |
|---|---|---|---|
| today: 4-subset conv trellis, free alphabet / E4M3-snapped | 1.294 / 1.348 | | 1.370 / 1.427 |
| window, free table, L=10 / 12 / 14 / 16 | 1.274 / 1.163 / 1.106 / **1.071** | | 1.409 / 1.232 / 1.138 / 1.090 |
| window, **E4M3-snapped** table, L=10 / 12 / 14 / 16 | 1.276 / 1.164 / 1.107 / **1.074** | | 1.442 / 1.270 / 1.192 / 1.155 |
| window, 2^(R+1)-value (LM-midpoint) table by hash, L=16 | 1.347 | | 1.421 |
| window, sorted table (no permutation) | 2.42 | | 3.09 |
| window, **computed** 1MAD-hash table, L=12 / 14 / 16 | 1.370 / 1.228 / 1.185 | | 1.601 / 1.419 / 1.311 |

Snapping to E4M3 costs nothing at R=4 (1.071 → 1.074) and ~6% at R=5: the
tile is not the tax, the state size is. A computed table (QTIP's 1MAD,
which a kernel could evaluate without storage) is 0.11 worse at L=16 than
a stored one, so the wire form is a stored table, priced. A small alphabet
by hash (the "LM-mid" arm) is no better than today's trellis: the gain
needs distinct values per state, not a bigger index into a small set.

On the **E2M1x2 tuple grid** (`tessera_bitshift_tuple.json["gaussian"]`,
ratio to `2^−(K/2)`), the snapped table is a different story:

| K (bpp at the cap) | coset trellis (Larsen6 / Ung6) | window free L=12 / 14 / 16 | window **E2M1x2-snapped** L=10 / 12 / 14 / 16 |
|---|---|---|---|
| 7 (4.0) | 1.409 / 1.405 | 1.228 / 1.139 / 1.090 | 1.791 / 1.681 / 1.640 / 1.623 |
| 6 (3.5) | 1.365 / 1.370 | 1.165 / 1.106 / 1.072 | 1.374 / 1.296 / 1.259 / 1.227 |
| 5 (3.0) | 1.399 / 1.405 | 1.125 / 1.082 / 1.057 | 1.202 / 1.137 / 1.098 / 1.077 |

At the cap (K=7, 128 of 256 tuples per state class) a random snapped table
is *worse* than the structured coset partition on a Gaussian: 16 E2M1
values per coordinate is too coarse a tile for random points to shape
well. Below the cap the snapped window wins by 1.1–1.3×. The best table
sigma is 2.0–2.3 grid units (peak 6); the free table prefers 1.4.

## E2M1x2 on the experts (per-16 fp32 amax/6 scale, LS refit ×2)

`out` per tensor (`tessera_bitshift_tuple.log`; the sweep is still
running on L20.up/L42 — six-tensor geomeans are appended when it finishes):

| tensor | K | coset Larsen6 / Ung6 | window L=12 | L=14 | L=16 |
|---|---|---|---|---|---|
| L5.gate | 7 (4.0) | 0.09114 / 0.09099 | 0.09472 | 0.09099 | 0.08950 |
| L5.up | 7 | 0.09434 / 0.09378 | 0.09769 | 0.09440 | 0.09201 |
| L20.gate | 7 | 0.08728 / 0.08696 | 0.09050 | 0.08719 | — |
| L5.gate | 6 (3.5) | 0.16611 / 0.16754 | 0.12880 | 0.12414 | **0.12179** |
| L5.up | 6 | 0.17187 / 0.17245 | 0.13289 | 0.12821 | **0.12556** |
| L5.gate | 5 (3.0) | 0.22741 / 0.22824 | 0.18215 | 0.17701 | **0.17379** |
| L5.up | 5 | 0.23493 / 0.23665 | 0.18855 | 0.18295 | **0.17948** |

The oracle's verdict holds on real weights: **at the E2M1x2 cap the window
body does not win** (L=12 loses 4%; L=14 is 0.0 to −0.7% per tensor before
its 0.016 bpp table is charged, a slight loss after; L=16 wins 1.7% before
its 0.0625 bpp table, which EXL3's slope prices at ~4%), and **below the
cap it is 1.3× better** —
1.34× at 3.5 bpp and 1.31× at 3.0 bpp at L=16, 1.29×/1.25× at L=12. At
3.5 bpp the window beats Gridbook FP4-CB K24 (0.1425 mean) by 1.17×; at
the cap the coset trellis on the span-2 wire stays the shipping body.

## What the wire now carries (schema minor 2)

`docs/schema/prismaquant.tessera.v1.md` §1b. `BodyKind.WINDOW` and
`window_bits` are manifest fields after the minor-1 fields; the table is
the ALPHABET plane (exactly `2^L` bytes, range-checked against the grid by
the reader), DESCENDANT and COMPLETION are empty, span is 1, and the
profile id binds `body:window,L=` in place of the convolutional code and
the forest rule. `encode.viterbi_window` is the exact encoder (pinned
start, per-position weights, chunked), `decode.replay_window` the
closed-form decoder (`⌈L/R⌉` shifted ORs), `encode.window_table` the
table builder (source quantiles per coordinate, seeded permutation,
nearest-code snap; `sigma=None` models the per-16 amax-bounded source,
a number a plain Gaussian in grid units for a per-channel plane).
`encode_linear(body=BodyKind.WINDOW, window_bits=L, window_seed=,
window_sigma=)` writes it; the config records `body:{kind, window_bits,
seed, sigma}` and the merge guard compares all four. Every TCQ artifact is
byte-identical and keeps its minor. `tests/test_window_body.py` holds the
Viterbi to the exhaustive search, the replay to the encoder's states, the
accountant to the bytes, and the reader to fail-closed.

**Not the default.** `DEFAULT_BODY` stays TCQ, and one mechanical gate is
left: the kernel lane has no window GEMV (`pack_unit_for_kernel` refuses
the body; the decode is a table lookup per position on an `L`-bit window
of the packed column, with the table in shared memory — 64 KB at L=16).
The *encoder* gate is closed — the section below is the measurement that
closed it.

## The encoder, fused (2026-09-02)

`encode.viterbi_window` keeps its definition and gains a machine:
`src/tessera/window_viterbi.py`, selected by `impl="auto"|"reference"|
"fused"` (auto is the fused path on CUDA inputs with Triton present, the
reference everywhere else). The reference spends a step moving eight
materialised `[2^L, cols]` intermediates — a view, a min, a
repeat_interleave, a subtract, a square, a sum, an add — through memory to
do one front's worth of arithmetic. The fused path does the class minimum,
the traceback write and the branch cost in **one** kernel, one front in and
one front out; sizes the column batch so both fronts stay in L2 across the
whole step loop; walks the traceback in one kernel instead of a gather per
step; and captures the batch as a **CUDA graph**, since every batch is the
same `2 + steps` launches with the same pointers and only a device-side
descriptor moving.

**Identity, not tolerance.** The states are the reference's states and the
`sse` is the reference's float, on all twelve rows below (same tensor per
row) and on 35 CUDA-gated cases in `tests/test_window_viterbi_fast.py`
spanning L ∈ {5,6,7,8,9,10,12,14,16}, R ∈ {2,3,4,5,6,7} with R ≤ L, both
arities, with and without per-position weights, chunk widths that divide
nothing, and 1–4096 columns. Three things the kernels do deliberately buy
it: the class minimum is an unrolled scan with a strict `<`, so the winner
is the *first* minimal index as `torch.min(dim=0)` returns; the epilogue is
left to torch on the assembled front, so the chunk's summation order is the
reference's; and every multiply that feeds an add is inline-asm `mul.f32`.
That last one is not paranoia — `enable_fp_fusion=False` (ptxas
`--fmad=false`) does **not** stop Blackwell's NVPTX backend packing the
arithmetic into `mul.rn.f32x2`/`add.rn.f32x2` and contracting the packed
pair; measured, the fused path returned exactly `fma(d0,d0,e1)` where the
reference returns `(d0*d0)+e1`, one ulp apart on a third of the elements at
arity 2. A trellis is a chain of decisions, so one ulp is not a rounding
detail: it flips a state and every state after it.

One 2048×4096 fp32 target, chunk 512, E4M3-shaped table at arity 1 (256
distinct values) and an E2M1 pair table at arity 2; reference and fused
timed **back to back on the same tensor**, which is the only honest
comparison on a shared board (`--impl both`):

| L | R | arity | reference (s) | fused (s) | speedup | ref W | fused W |
|---|---|---|---|---|---|---|---|
| 12 | 4 | 1 | 6.72 | 0.45 | **15.0×** | 49 | 46 |
| 12 | 4 | 2 | 5.15 | 0.45 | **11.5×** | 52 | 50 |
| 12 | 5 | 1 | 6.50 | 0.69 | **9.4×** | 54 | 59 |
| 12 | 5 | 2 | 4.04 | 0.34 | **11.8×** | 67 | 63 |
| 14 | 4 | 1 | 29.65 | 1.14 | **26.0×** | 54 | 72 |
| 14 | 4 | 2 | 22.32 | 1.16 | **19.2×** | 54 | 58 |
| 14 | 5 | 1 | 29.18 | 2.03 | **14.4×** | 55 | 93 |
| 14 | 5 | 2 | 22.30 | 1.29 | **17.2×** | 56 | 72 |
| 16 | 4 | 1 | 129.39 | 5.06 | **25.6×** | 52 | 68 |
| 16 | 4 | 2 | 96.35 | 4.95 | **19.5×** | 53 | 69 |
| 16 | 5 | 1 | 156.31 | 8.51 | **18.4×** | 52 | 83 |
| 16 | 5 | 2 | 97.00 | 4.94 | **19.6×** | 52 | 82 |

Whole sweep 605 s → 31 s, **19.5×**, every
`sse` identical
(`experiments/results/window_viterbi_bench_paired.jsonl`; states *and* sse
identity re-checked at L=16, R=4 against a full reference run in
`window_viterbi_bench_check.jsonl`). Contention is the reason for the
back-to-back protocol: the same reference config measured 428 s beside two
other GPU jobs and 129–166 s beside none — a "before" number is worthless
without the board state, and the first baselines here
(`window_viterbi_bench.jsonl`) carry a neighbour list that a narrower
`pgrep` had missed.

**Per principle 15 the profiles are the claim, not the seconds.**
`torch.profiler`, same box, R=4 arity 1
(`experiments/results/window_viterbi_profile_{reference,fused}.txt`):

* reference L=12 — 196,648 launches, self CUDA **4.970 s** over six op
  kinds at ~16.4 k calls each, and `Command Buffer Full` 58.5% of CPU: the
  launch queue is the wall. L=14 — self CUDA **29.132 s**, `Command Buffer
  Full` 86.4%.
* fused L=12 — self CUDA **0.344 s**, 96.8% of it in `_step` (67,584
  launches at 4.92 µs). L=14 — **1.288 s**, 98.7% `_step`. L=16 —
  **5.139 s**, 99.3% `_step` at 4.85 µs.
* board time 4.970 → 0.344 s at L=12 (14.5×) and 29.132 → 1.288 s at L=14
  (22.6×), and wall now sits within 19%/8%/6% of board time at L=12/14/16,
  where before the graph it was 2× board time at every L.

**Power against the envelope, not utilisation.** Netdata
`nvidia_smi.gpu_power_draw` over two 150 s windows of L=16 R=4 arity 1
(`experiments/results/window_viterbi_power_windows.jsonl`): fused **65.8 W**
mean (60–69) at 4.618 s a tensor, reference **52.3 W** mean (52–54) at
127.6 s. The fused path draws a quarter more power and finishes 27.6×
sooner: 304 J a tensor against 6674 J, **22× the work for a joule**, at 47%
of the ~140 W envelope against 37% — so the headroom is still there, and
`gpu_utilization` said 96% on both sides throughout.

**Rejected here, with the numbers, so nobody pays twice.**

* A **register-resident column kernel** (one block per column, the whole
  `2^L` front in registers for every step, only the traceback byte moving)
  is exact and lost: 1.156 s against 0.980 s at L=12 R=4, 2048×4096,
  identical states and sse. `tl.min(..., return_indices)` and the reshape
  that shifts the front are block-wide barriers twice a step where the
  tiled kernel's scan is thread-local — holding the front costs more than
  moving it through L2.
* **Widening the column batch** past L2: the working-set sweep is a knee at
  L2/6 ≈ 4 MB and a cliff after it — L=16 runs 5.34 s at 4 MB, 6.23 s at
  8 MB and 14.59 s at 16 MB, identical `sse` at every point. The cache is
  doing the work, and the budget is a measurement knob
  (`TESSERA_WINDOW_L2_BYTES`), never a correctness one.
* An earlier reading that **CUDA graphs do not help here** was wrong and is
  retracted: it compared a `cudaEvent` span (which measures the stream,
  gaps included) against wall. The profiler's per-kernel time is what
  exposes the gap, and graphs are worth ~2×: the same twelve configs
  eager-fused and graph-fused, same board, are
  `window_viterbi_bench_warm.jsonl` against `window_viterbi_bench_graph.jsonl`
  (L=12 R=4 a1 0.671 → 0.488 s, L=14 2.737 → 1.342 s, L=16 10.884 →
  5.297 s), and the uncontended reference-only rows are
  `window_viterbi_bench_uncontended.jsonl`.

**Encoder throughput only — no quality is measured in this section**, and
the numbers are per 2048×4096 tensor, not per model. The recipe target
stays **L=14 per-channel on E4M3**: the kernel lane's own sweep prices the
per-unit 2^L table at nothing through L=14 and at **1.83–1.86×** at L=16
(span-2 0.204 ms, window L=12 0.199, L=14 0.208, L=16 0.375, identical
bytes on every other plane —
`.claude/worktrees/agent-ac41ec3897043fa75/experiments/results/tessera_kernel_window_table_sweep.json`),
so any L=16 quality reading must be stated *before the table*.

## What is not established

* The E4M3 headline is under a per-channel plane that the wire does not
  yet spell; on the per-16 LUT plane the gain is smaller (level at L=12 on
  a Gaussian smoke). Measure the window body on the true wire — E2M1x2 at
  K=5/6 over the LUT16 plane, E4M3 at R=4/5 over LUT16 and over the
  per-channel plane once it exists — before any default moves.
* Mixed rates over one table (a Bresenham unit mixing K=4/5 columns) were
  not measured; the arms above are single-rate. The refit should absorb it;
  that is a prediction.
* No LDLQ on the window arms. On the conv-trellis arms LDLQ was worth
  1.05–1.07× (`tessera-ldlq-regulariser`); the two levers should compose.
* Gridbook comparators from the follow-up worker: "+imatrix" rows are
  within 0.1% of bare (weight-only quantisers barely move on an imatrix);
  its "+imatrix +LDLQ(gated)" rows are worse than bare by 1.3–1.4× and are
  not a valid comparator until its gate is understood.

## Next

1. `ScalePlaneKind.CHANNEL` (one scale per output row, fp16 or E4M3×global):
   the served W8A8 layout, and the plane the headline was measured under.
2. Window GEMV in the kernel lane; then the default flips **per grid**:
   E4M3 (window) and E2M1x2 below the cap (window), E2M1x2 at the cap
   (coset trellis, span 2) — a per-unit choice the wire already expresses.
3. ~~A fast encoder~~ — done 2026-09-02 (see *The encoder, fused*):
   9.4–26× on the 2048×4096 sweep, bit-exact, profiled before and after.
   What is left on that axis is a survivor-limited (M-algorithm) encoder,
   which would trade exactness for rate and therefore needs a quality A/B,
   not a clock; and the encoder on a *full expert layer* rather than one
   tensor.
4. LDLQ on the window body; the held-out served A/B.
