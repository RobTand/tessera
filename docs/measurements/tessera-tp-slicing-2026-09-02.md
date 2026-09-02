# Slicing a Tessera unit into the shard a rank loads

**Date** 2026-09-02 · **Branch** `worktree-agent-a5d4cf4818e8e77ba` off
`3d419e7`, commits `dee75d9`, `1ce9b70`, `d8cc017` and the one this line lands
in ·
**Design** [`docs/design/tensor-parallel.md`](../design/tensor-parallel.md) ·
**Schema** `docs/schema/prismaquant.tessera.v1.md` §1d (minor 4)

The question: can a rank load exactly its shard of a Tessera unit from bytes
that never knew the TP degree? The answer is yes, and this is what it costs.

Environment for every command below:

```
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
export PYTHONPATH=src
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python   # torch 2.11+cu130
```

One GB10, shared with six other workers throughout, so treat every absolute
time as an upper bound and every *ratio measured within one run* as the claim.

---

## 1. Correctness: the shard is the parent's decode, sliced

```
$PY -m pytest tests/test_slice_unit.py -q
# 75 passed, 2 skipped in 43.25s

$PY -m pytest tests/ -q
# 616 passed, 2 skipped in 1122.87s
```

The two skips are correct refusals, not gaps: the released unit is 512 columns
and a released unit cuts columns only on 256-column superblocks, so tp=4 and
tp=8 along columns do not exist for it. `can_shard` says so, and the test
skips on its answer.

Coverage, per the brief:

| body / plane | case | releases | axes exercised |
|---|---|---|---|
| WINDOW + CHANNEL | `e4m3-window-channel` (the shipped E4M3 wire) | 0 | rows, columns, tp ∈ {2,4,8} |
| WINDOW + LUT | `e2m1x2-subcap-window-lut` | 0 | rows, columns, tp ∈ {2,4,8} |
| TCQ + LUT | `e2m1x2-cap-tcq-lut` (span 2) | 0 | rows, columns, tp ∈ {2,4,8} |
| TCQ + LUT | `e2m1-tcq-lut-release` | 96 positions | rows tp ∈ {2,4,8}, columns tp=2 |
| TCQ + S6B | `_s6b_unit` | 0 | the block-scale plane at 32 |

plus seven real units from the shipped checkpoint
`/home/rob/tessera-runs/gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook`
(q/k/v, o_proj, gate/up, down_proj), sliced on both axes and re-parsed.

What the property test asserts, for every case × axis × tp: the shard's decode
equals `decode(parent)[r0:r1, c0:c1]` **bit for bit** (integer codes and the
dequantised tensor), the shard survives a wire round-trip, and a shard of a
shard equals the direct slice.

On top of the decode, the two materialisers a serving route actually calls:
`stock.materialize_stock` on every case (the NVFP4 triple's packed nibble pairs
and per-16 scale bytes; the FP8 pair's byte plane and per-row scale) and
`decode.materialize_fp8` on the E4M3 case, each compared against the parent's
materialisation sliced, on both axes. Decoding correctly is not sufficient for
these: the packed layouts are a second layout over the same values, and a
column cut landing off a nibble pair or off a scale group would decode right
and pack wrong.

**Nothing at row offset 0 moved.** Two golden-byte tests hold the parent bytes
against hashes recorded from an archive of `src/` and `tests/` taken at
`3d419e7` (`/home/rob/tmp/tp-slice/head_src`, scripts `head_golden.py`,
`head_unit_golden.py`):

```
e4m3-window-channel        21159 B  bca30ebbc1687d1a753525ceb1148edd1469504d2cd2c18cbf9aa5f052dd802c
e2m1-tcq-lut-release        8398 B  1840b3f9dfe3929d9aa86006f207ad427658135d2d5c95fa5dc0b6ad0e532f31
e2m1x2-subcap-window-lut    8059 B  ae0e675dc7b0fbf40a5cee88f22baf5bb74b650d07b6aaa30f6008a443196ec5
layout (512,8,32,8)                 cb39f35e686e8858485917c81ab4c53c3b1464d83559ac49dcbbda24fdd783a3
layout (640,16,64,16)               0e88573aaa13af73d8fc13d8dab0a16b5f52257f70085493ee5f20bd30b73787
```

The second pair is the encoder-free `conftest.make_artifact` path — layout,
manifest, container and nothing else, which is exactly the code this schema
minor touches. Both survive, because a whole unit still writes the nine-plane
canonical order it always wrote.

Independently: all 196 units of the shipped checkpoint parse, slice at
tp ∈ {2,4,8} on both axes, and re-serialise, with zero mismatches
(`/home/rob/tmp/tp-slice/check_all.py`).

**One existing test changed, deliberately.**
`test_review_1a.py::test_every_plane_kind_has_a_normative_width` asserted that
every `PlaneKind` carries a schema-fixed element width. INITIAL_STATE cannot:
its width is `window_bits` (14 on the shipped wire) under a window body and
the convolutional code's memory (6) under the coset trellis, which is a
property of the encoder profile, not of the schema — a normative entry would
have to be wrong for one of the two bodies. The test now asserts the exception
is *exactly one*, and the width is bound twice elsewhere instead: `Manifest`
refuses a descriptor whose `element_bits` differs from the `state_bits` its
shard record declares (and, under a window body, from `window_bits`), and
`parse_unit_artifact` re-checks it against the body once the profile id has
resolved one. Both refusals are tested.

---

## 2. Time: what a rank pays at load

`/home/rob/tmp/tp-slice/measure_tp.py`, over the same 196 units. *before* is
the same script against the previous commit's `layout.py` (the
unbounded-prefix replay), run back to back with *after* on the same box.

| | before | after |
|---|---|---|
| parse (bytes → planes) | 42.93 ms/unit | 40.39 ms/unit |
| decode (planes → weights) | 32.98 ms/unit | 32.90 ms/unit |
| slice **one** shard, tp=2 last rank | 14.86 ms | 14.51 ms |
| slice **one** shard, tp=4 last rank | 15.42 ms | 14.39 ms |
| slice **one** shard, tp=8 last rank | 16.91 ms | 14.32 ms |
| slice **all** tp=2 shards | 14.92 ms | 14.52 ms |
| slice **all** tp=4 shards | 35.31 ms | 33.62 ms |

A rank's own cut is roughly a third of the parse it follows, and a rank cuts
one shard, not `tp` of them. The absolute win from the bounded tail is small
here because these units are 1024–2048 rows; what changes is the *shape*:
before, one shard costs more the further down the unit it sits
(14.86 → 15.42 → 16.91 ms as tp rises); after, it is flat.

**How much of this table to believe.** Every number lands near a multiple of
~4.9 ms, and "slice all tp=2 shards" (14.52) is not measurably more than
"slice one" (14.51) even though it does a second cut. That is host-device sync
granularity under GPU time-slicing with six other workers, and a min-of-3
picks the luckiest alignment. So read the *shapes* as the claims -- flat versus
growing in the prefix, a column cut cheaper than a row cut, a cut cheaper than
the parse it follows -- and read "36%" or "3.0x" as approximate. A precise
per-rank load budget needs an unshared box.

The scaling itself, on one 12288×256 E4M3/WINDOW/CHANNEL unit — 12288 rows is
a GLM-5.3-Flash expert `w2`, and the narrow column count keeps the encode
affordable (`/home/rob/tmp/tp-slice/measure_scale.py`):

| cut | rows above it | before | after |
|---|---|---|---|
| rank 0, any tp (stores no state) | 0 | 4.76 ms | 4.86 ms |
| last rank, tp=2 | 6144 | 15.13 ms | 14.60 ms |
| last rank, tp=4 | 9216 | 15.85 ms | 14.73 ms |
| last rank, tp=8 | 10752 | 16.18 ms | 14.72 ms |
| last rank, tp=16 | 11520 | **21.29 ms** | **14.71 ms** |

Before: the cut cost O(rows above it) — the last rank of a tp=16 split replayed
fifteen sixteenths of the unit to recover fourteen bits per column. After: flat
to within 0.8% across a 2× range of prefix length, because the replay runs over
`ceil(L/R)` positions (window) or `memory + 1` super-symbols (TCQ) and no more.
Rank 0 is 4.8 ms either way — it stores no state, so it never enters the path.

Where the remaining ~10 ms goes: it is the state computation's own fixed cost
(one bounded replay per rate group, plus the plane), not the prefix. It is
launch-overhead-bound — a spot `nvidia-smi` sample during these runs read 35 W
of a ~140 W envelope, which is a sample and not a series — and batching the
per-rate-group replays is the obvious thing to cut if a rank's load time ever
matters. It has not been cut, because 14 ms against a
40 ms parse and a 33 ms decode is not where load time goes.

TCQ and window bodies at 512×512 (`/home/rob/tmp/tp-slice/measure_tcq.py`),
including the E2M1x2 cap unit encoded from a Qwen3-0.6B `o_proj` tile through
`export.wire_recipe`:

```
e2m1x2-cap-tcq        512x512  TCQ span 2 / LUT      granularity rows=4 cols=16
  parse 20.91 ms   row cut 14.58 ms (70% of parse)   column cut 4.87 ms (23%)
e2m1-tcq-lut-release  512x512  TCQ span 2 / LUT, 2048 releases   rows=2 cols=256
  parse 58.64 ms   row cut 29.32 ms (50% of parse)   column cut 14.62 ms (25%)
e2m1x2-subcap-window  512x512  WINDOW / LUT          granularity rows=2 cols=16
  parse 14.56 ms   row cut 14.58 ms (100% of parse)  column cut 4.89 ms (34%)
```

A **column** cut is 3–4× cheaper than a row cut on every body, because it
stores no state at all.

---

## 3. Bytes: what the mechanism costs on the wire

The INITIAL_STATE plane is `state_bits` bits per column, on shards below row 0
only — rank 0 starts at row 0 and carries none. So a row cut costs
`(tp − 1) × state_bits / rows` bpp, independent of the column count, and a
**column cut costs nothing**.

Measured, `model.layers.0.mlp.down_proj` (1024×3072, E4M3 wire, L=14, parent
plane region 1 591 296 B = 4.0469 bpp), each shard written as its own artifact:

| tp | total bytes | × parent | state plane | duplicated ALPHABET table |
|---|---|---|---|---|
| 2 | 1 613 056 | 1.0137× | 5 376 B (+0.0137 bpp) | 16 384 B (+0.0417 bpp) |
| 4 | 1 656 576 | 1.0410× | 16 128 B (+0.0410 bpp) | 49 152 B (+0.1250 bpp) |
| 8 | 1 743 616 | 1.0957× | 37 632 B (+0.0957 bpp) | 114 688 B (+0.2917 bpp) |

Read that table carefully. The **state plane** is the mechanism's cost, and it
is small. The **table** column is the cost of *materialising* each shard as a
standalone artifact, which duplicates the per-unit 2^L-byte window table per
shard — and which nothing on the serving path pays, because a rank slices in
memory from the one artifact on disk. (The totals include a third, smaller
term: each shard carries its own 24-byte header, manifest and digests.)

Projection, at the shipped E4M3 wire (`state_bits = L = 14`) and at the
E2M1x2 cap (`state_bits = memory = 6`):

| shape | rows | wire bpp | tp=2 | tp=4 | tp=8 |
|---|---|---|---|---|---|
| Qwen3-0.6B `q_proj` | 2048 | 4.047 | 0.0068 (0.17%) | 0.0205 (0.51%) | 0.0479 (1.18%) |
| Qwen3-0.6B `down_proj` | 1024 | 4.047 | 0.0137 (0.34%) | 0.0410 (1.01%) | 0.0957 (2.37%) |
| Qwen3-4B `gate_up_proj` | 19456 | 4.070 | 0.0007 (0.02%) | 0.0022 (0.05%) | 0.0050 (0.12%) |
| Qwen3-4B `down_proj` | 2560 | 4.070 | 0.0055 (0.13%) | 0.0164 (0.40%) | 0.0383 (0.94%) |
| GLM-5.3-Flash expert `w1` 4096×12288 | 4096 | 4.070 | 0.0034 (0.08%) | 0.0103 (0.25%) | 0.0239 (0.59%) |
| GLM-5.3-Flash expert `w2` 12288×4096 | 12288 | 4.070 | 0.0011 (0.03%) | 0.0034 (0.08%) | 0.0080 (0.20%) |
| GLM expert `w1`, TCQ span 2 | 4096 | 4.000 | 0.0015 (0.04%) | 0.0044 (0.11%) | 0.0103 (0.26%) |

At the shapes this exists for — the GLM expert bodies — the cost of being
tensor-parallel is **0.03–0.6% of the wire**, paid only by the ranks that are
not rank 0, and only on a row cut. On a 512×512 unit the same plane is
0.0117–0.0703 bpp: the same fraction at a smaller scale.

For comparison, the mechanism this replaces — one re-encoded artifact per TP
degree — costs 100% of the wire per additional degree, plus the encode.

---

## 4. What this does not show

* **Nothing here was served.** The claim is "the bytes slice exactly", which
  the tests hold at the tensor level, through the wire, and through the window
  kernel lane. A two-box TP=2 serve, and KL-vs-BF16 against the same artifact
  on one rank, is the gate that would turn this into "the model serves
  tensor-parallel". It has not been run.
* **The plugin loader does not exist yet.** `docs/design/tensor-parallel.md`
  is its contract; W2 owns the code.
* **The span-2 kernel lane refuses a shard** rather than pack one against the
  pinned zero. The window lane — the shipping E4M3 wire — takes it with no
  kernel change, and `test_slice_unit` runs `tessera_gemv_window` on a tp=2
  shard against the reference decode.
* **No encoder change was made or needed**, which was the constraint. The
  encoder still pins the start state to zero; slicing is entirely a layout,
  manifest and decode-side mechanism.

## 5. Provenance

* Base commit `3d419e7`; the work is `dee75d9` plus the commit this receipt
  lands in.
* Scratch scripts, golden blobs and the archived base source:
  `/home/rob/tmp/tp-slice/`.
* **Fable consultations: none.** The advisor's read was that the shift-register
  algebra is simple enough that the oracle test *is* the proof, and to escalate
  only if the oracle disagreed. It does not: `test_slice_unit` checks the
  computed TCQ state against `TCQ.decode`'s scalar walk and the window state
  against a per-column Python walk, on every case.
