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

**L=16** (sparklina, all six tensors, free start, table not charged): R=4
`out` 0.06079 / `a8` 0.06532 at 4.016 bpp — 0.909× of EXL3 K4 before its
0.0625 bpp table, which EXL3's slope prices at ~4%; R=5 `out` 0.03264 /
`a8` 0.04048. About 3% over L=14 per tensor, for four times the table and
the encode, and most of it given back to the table.

### Three caveats, each measured

1. **Start state.** The sweep's Viterbi started free (any initial state,
   charged `L/rows` bpp). The wire pins state 0, as the decoder assumes.
   Pinned re-run (`--pinned-start`, table charged in `bpp`): L5.gate R=4
   L=14 `out` 0.06734 / `a8` 0.07200 at 4.023 bpp; L5.up 0.06963 / 0.07448;
   L5.gate R=5 L=14 0.03633 / 0.04438. **Pinned six-tensor geomeans, table
   charged** (`tessera_bitshift_tile_pinned.json`, run complete):

   | arm | bpp | out | a8 | vs EXL3 out | as served, a8 vs EXL3 W4A16 |
   |---|---|---|---|---|---|
   | window L=12 pinned, R=4 | 4.012 | 0.06640 | 0.07059 | **0.986×** | 1.048× |
   | window L=14 pinned, R=4 | 4.023 | 0.06321 | 0.06759 | **0.938×** | 1.003× |
   | window L=12 pinned, R=5 | 5.012 | 0.03632 | 0.04350 | 1.059× | — |
   | window L=14 pinned, R=5 | 5.023 | 0.03392 | 0.04151 | **0.989×** | — |

   The pinned start costs 0.6–0.8% against the free start: the headline
   is now quoted pinned — **0.938× of EXL3 K4 at L=14**, W8A8 level with
   EXL3's W4A16 at the same bytes.
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

**Not the default.** `DEFAULT_BODY` stays TCQ. One gate is now closed and
one is open. The kernel lane **does** decode the body (see "Kernel lane"
below). What remains is the reference encoder, which is O(2^L) per
position: ~30 s at L=12, ~150 s at L=14, ~13 min at L=16 per 2048×4096
tensor on GB10, which is days per MoE model. A Triton Viterbi with the
cost front resident (fits at L≤14) or an M-algorithm with measured
survivors is the next encoder step, and per principle 15 its acceptance is
profiler evidence, not a bench number.

## Kernel lane (2026-09-02)

**Claim.** The Triton GEMV decodes a window body bit-exactly at the wire's
own bytes, over both grids, both scale planes and mixed rate schedules, and
at the shipping shape it is **the span-2 trellis kernel's equal up to
L=14 while reading fewer bytes, and 1.85× its cost at L=16** — the
per-unit `2^L` table is the whole of that, measured with the encoder taken
out of the run. `pack_unit_for_kernel` dispatches on `BodyKind`; the span-2
trellis path is untouched, byte for byte and kernel for kernel.

**The decode.** No replay tables and no halo argument: a state *is* the
last `L` bits of the column's stream, so the kernel reads an `L`-bit field
out of the plane and indexes the unit's own `2^L` ALPHABET table, then the
grid's value table (`kernel.build_window_values`) — the same shared /
per-unit seam the trellis lane cuts, with the halves the other way round.
Fusing them would be `2^L × arity` floats: 512 KB per unit at L=16, arity
2, against 64 KB for the table. `pack_window_planes` writes the plane
column-major MSB-first with an `L`-bit zero pad per column (the pad *is*
`state_{−1} = 0`, as `SELECT_PAD` is for the trellis lane) and a per-column
bit offset, because a mixed schedule gives every column its own stride —
the case the span-2 lane refuses outright.

**Method** (principle 15). `experiments/tessera_kernel_window_bench.py`,
same tensor and protocol as the span-2 bench: one GLM routed expert
(`layers.20.mlp.experts.0.gate_proj`, 2048×4096), ≥20 s timed loops,
CUDA-event ms/call, a `torch.profiler` pass for self CUDA time, and
`nvidia-smi` power sampled once a second with each arm's epoch window in
the JSON. Results `experiments/results/tessera_kernel_window_bench.json`,
log `...bench.log`, power `...power.csv`. Envelope ~140 W. A second script,
`experiments/tessera_kernel_window_table_sweep.py`
(`...window_table_sweep.json`), times synthetic units with no encoder in the
run, which is what isolates `L`.

**The box was not idle.** Three other Tessera GPU jobs were resident for the
whole run — `tessera_vs_exl3_followups.py` (PID 1103204, from 20:18),
`tessera_bitshift_tuple.py --chunk 256` (PID 1123072, from 20:34) and
`window_viterbi_bench.py` (PID 1346756, from 22:41) — and the last of those
plus a `pytest` run *started while the bench was running*, at 22:41 and 23:04.
(The counts the two logs print — "6 other tessera process(es)", "concurrent
GPU jobs: 9" — count the shell wrappers too; `ps` showed three python jobs on
the GPU, the PIDs above.)
The bf16 anchor reads 0.4323 ms against the 0.0161 this tensor takes on an
idle box: absolute ms below are inflated ~27× and are **not** comparable to
the span-2 lane's published 0.0524 ms. The comparison that survives is
arm-against-arm inside this run, and even that is weakened by the two jobs
that arrived mid-run, which is why the second table exists.

| arm | ms/call | kernel self CUDA | bytes/call | b/wt | power mean/max |
|---|---|---|---|---|---|
| bf16 torch GEMV (anchor) | 0.4323 | — | 16.78 MB | 16.0 | 46.1 / 48.5 W |
| span2 trellis E2M1×2 R=7 | 0.2097 | 0.1496 | 4.20 MB | 3.75+0.25 | 54.5 / 56.8 W |
| window E2M1×2 R=7 L=12 | 0.2082 | 0.1468 | 3.94 MB + 4 KB table | 3.51+0.25 | 56.6 / 58.7 W |
| window E2M1×2 R=7 L=14 | 0.2134 | 0.1498 | 3.96 MB + 16 KB table | 3.51+0.25 | 57.4 / 59.9 W |
| window E2M1×2 R=7 L=16 | 0.5062 | 0.2110 | 4.01 MB + 64 KB table | 3.51+0.25 | 53.2 / 53.5 W |
| window E4M3 R=4 L=12 | 0.2095 | 0.1482 | 4.47 MB + 4 KB table | 4.01+0.25 | 65.4 / 68.3 W |
| window E4M3 R=4 L=14 | 0.2139 | 0.1953 | 4.48 MB + 16 KB table | 4.01+0.25 | 65.8 / 68.7 W |
| window E4M3 R=4 L=16 | 0.3826 | 0.1743 | 4.53 MB + 64 KB table | 4.01+0.25 | 57.3 / 58.9 W |

Every arm draws 46–69 W of a ~140 W envelope, the bandwidth-bound signature
the span-2 lane already has (`gpu_utilization` is non-diagnostic on GB10 —
principle 15); the window body does not change what the lane is limited by.
At L ≤ 14 **neither instrument separates the arms in this run**: ms/call is
flat at 0.208–0.214 because contention floors it, and the kernel row is one
short profiler pass on the same contended box — so the lone 0.1953 on E4M3
L=14 is that pass's noise, not a slower kernel (the repeated sweep below puts
the arm at 1.02× span-2). L=16 is the one width that breaks the floor, in
both grids, which contention alone does not explain. The sweep below is what
ranks all of them.

**The L=16 cost is the table, and it is not the encoder's heat.** Each arm
above is timed right after its own 16–474 s encode, so a wall number could be
carrying whatever the encode left behind. `tessera_kernel_window_table_sweep.py`
removes the encoder: synthetic units at the same 2048×4096 shape (real
forests, real LUTs, real plane layouts, random body bits), all resident before
anything is timed, two passes. Across L the plane bytes, the scale plane and
the value table are *identical* — the rate is fixed, so only the `2^L` ALPHABET
table changes size — which makes this an attribution and not a correlation.

| arm | table | ms/call pass 0 | pass 1 | vs span-2 |
|---|---|---|---|---|
| span2 E2M1×2 R=7 | — | 0.2038 | 0.2036 | 1.00× |
| window E2M1×2 R=7 L=10 | 1 KB | 0.1970 | 0.1972 | 0.97× |
| window E2M1×2 R=7 L=12 | 4 KB | 0.1992 | 0.1975 | 0.97× |
| window E2M1×2 R=7 L=14 | 16 KB | 0.2059 | 0.2082 | 1.01× |
| window E2M1×2 R=7 L=16 | 64 KB | 0.3755 | 0.3704 | 1.83× |
| window E2M1×2 R=7 L=18 | 256 KB | 0.6011 | 0.6013 | 2.95× |
| window E4M3 R=4 L=10 | 1 KB | 0.1992 | 0.1991 | 0.98× |
| window E4M3 R=4 L=12 | 4 KB | 0.2026 | 0.2036 | 1.00× |
| window E4M3 R=4 L=14 | 16 KB | 0.2083 | 0.2071 | 1.02× |
| window E4M3 R=4 L=16 | 64 KB | 0.3795 | 0.3790 | 1.86× |
| window E4M3 R=4 L=18 | 256 KB | 0.8439 | 0.8447 | 4.14× |

ms/call reproduces to under 2% across passes; the profiler's self-CUDA row
does not (its pass is a handful of calls, and on this box it scattered ±30%,
including one 0.057 ms read on an arm that read 0.151 ms the other pass). On
a contended box the 6 s CUDA-event loop is the instrument that ranks and the
profiler row is the one that attributes — the two are not interchangeable,
which is the same lesson the synchronising `rates.max()` taught below.

**So the lane's answer is per-L, not one number.** At L ≤ 14 the window GEMV
is the span-2 kernel's equal or a shade under it (0.97–1.02×) *while reading
fewer bytes* — 3.51 b/wt of body against 3.75 at the E2M1×2 cap. At L=16 the
64 KB per-unit table falls out of whatever the SM was holding and it costs
1.83–1.86×; at L=18, 3–4×. Two things could move that, and they are not the
same kind of work: holding the table in shared memory is lane work, while
sharing one table across a layer's units is an encoder-and-wire choice, since
the table is per-unit on the wire today. As the lane stands, though, **a
window body wider than L=14 is a decode the kernel pays for**, and the widths
this document's own results favour (L ≥ 14–16, §Result) land on the wrong
side of that. It is a lane finding the encoder work should hear, not a defect
in this decode.

**Bit-exactness.** `tests/test_kernel_window.py` (57 cases): one-hot
columns through the kernel equal `read_unit_artifact(blob)` by
`torch.equal` — the comparison is against the **bytes**, not the encoder's
tensors — over E2M1×2 with a mixed {5,6,7} schedule at L ∈ {10,12,14,16}
and E4M3 at R ∈ {4,5} and L ∈ {12,14,16}, at 256×512, at 2048×4096, and at
200×768 (100 codes, so both tail masks run). Every one of those cells runs
under the LUT16 plane; the S6b plane runs on four of them (E2M1×2 mixed at
L=10 and L=14, E4M3 R=4 at L=12, R=5 at L=14), because the plane is a
`constexpr` branch in the kernel and does not interact with `L` — that is
an argument, not a measurement, and the coverage is what it is. The bench re-checks one-hot exactness at 2048×4096 for
every arm before it times it, which is where the L=14/16 big-shape cases
are covered — encoding one costs 2.5–5.5 minutes. A hand-made plane (no
encoder) pins the window read at L ∈ {12,18,20}.

**What the profiler changed.** Two findings, both invisible to ms/call
alone or to the kernel row alone:

* The first cut computed the window, the state and the code once per
  *output row* rather than once per *code*, and read four bytes per code.
  At the E2M1×2 cap that is 64 byte loads per lane per column for a result
  bit-identical across the arity axis: on a 3 s smoke it read 0.163 ms
  self CUDA against the span-2 kernel's 0.105 — **1.55× the work for fewer
  bytes**. Blocking as `[LANES, VEC, arity]` and reading each half-lane's
  windows out of one int64 (16 loads) closed it. (Those two numbers are
  smoke-length and off the table below, which is the 20 s run.)
* A launch-shape check that read `int(rates.max())` off the device tensor
  synchronised every call: 2.6 ms/call against a ~0.1 ms kernel. The
  profiler's kernel row was *unchanged* and only ms/call moved, which is
  the case for keeping both instruments.

## What is not established

* ~~The E4M3 headline is under a per-channel plane that the wire does not
  yet spell~~ — it does now (schema minor 3, `ScalePlaneKind.CHANNEL`,
  `docs/schema/prismaquant.tessera.v1.md` §1c). **On the true wire**
  (`experiments/tessera_window_wire.py`, production `encode_linear` →
  `read_unit_artifact`, six-tensor geomeans, `tessera_window_wire_e4m3.json`):
  E4M3 window L=12 over the LUT16 plane at 4.0 bpp (q960) is `out` 0.07573
  against the span-2 TCQ default's 0.08123 (1.073× better, 25 s vs 4 s per
  tensor) and at 5.0 bpp 0.04056 vs 0.04252 (1.048×). Against the same body
  over the per-channel plane at the same bytes (0.06640 pinned L=12), the
  block plane costs **1.14×** on E4M3: on this tile the plane is worth as
  much as the body, and the CHANNEL plane is the E4M3 recipe's. The E2M1x2
  true-wire arms (K=5/6/7 over LUT16, mixed rates) run in
  `experiments/tessera_frontier.py`.
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

1. ~~`ScalePlaneKind.CHANNEL`~~ **Done** (schema minor 3): one fp16 per
   output row on the DIAG_SV plane times an fp32 global, the served W8A8
   layout (`decode.materialize_fp8`), refit landed on the stored word.
2. ~~Window GEMV in the kernel lane~~ — landed, see "Kernel lane" above;
   a window unit over the CHANNEL plane decodes in the same lane (identity
   block plane, row scale as an epilogue in the reader's own fp32
   expression; `tests/test_channel_plane.py`, bit-exact on one-hot
   columns). The default can flip **per grid** on the encoder's schedule
   alone —
   E4M3 (window) and E2M1x2 below the cap (window), E2M1x2 at the cap
   (coset trellis, span 2), a per-unit choice the wire already expresses —
   **but only up to L=14**: at L=16 the per-unit table costs the GEMV
   1.85×, and L≥14 is exactly where the encoder wins. Sharing one table
   across a layer's units, or holding it in shared memory, is the lane work
   that would remove the constraint. The other gate is still the O(2^L)
   encoder.
3. A fast encoder (Triton or survivor-limited), timed on a full expert
   layer under `torch.profiler` and Netdata before it is trusted.
4. LDLQ on the window body; the held-out served A/B.
