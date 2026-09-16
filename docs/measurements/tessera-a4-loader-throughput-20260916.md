# Native A4 loader throughput: repeated-work elimination

Measured 2026-09-16 in the pinned image
`localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb`
with the real `TesseraNvFp4MoEMethod._load_wire` intake over real wires from
`merged-4c384e60`, TP2 rank-local geometry, PyTorch CUDA allocator capped at
8 GiB of the actual device total, container 16 GiB with no swap, under the
deployed `max_split_size_mb=20` loader context.  No full model was constructed
and no server was started.  This measures loader throughput and per-layer
memory; it is not a full-model footprint or serve claim.

## Causal finding

The f5 in-process profile (worker main thread, both ranks; sleeping threads
excluded) put 97 % of the sampled second in `_load_wire`, and none of it
decoded a wire — it re-derived layer-constant facts per wire: two serialised
SHA-256 passes (`verify_plane_region` 4.36 / 4.25 s), `_check_rate`'s linear
legal-rate scan (2.14 / 2.23 s, 3.28 M calls), `_steps_of`'s per-column
`body_bits` walk (0.85 / 0.72 s), 4096 `completion_capacity` calls per wire
(1.19 / 1.02 s), the encoder-profile search per wire (0.56 / 0.52 s), and
`_plane_words` rebuilt once per packer (0.20 / 0.23 s).

## Changes

- `container.verify_plane_region`: for regions ≥ 1 MiB the per-plane checks
  (content digests included) run on one short-lived worker thread while the
  caller computes the whole-region payload digest.  `plane_ranges` is
  evaluated inside that same work, so a malformed manifest cannot raise
  before the payload comparison; the payload digest is still checked before
  any plane-level failure is re-raised, the worker is joined on every exit,
  and every check and message is unchanged.
- `grammar._check_rate`: membership keeps the **original** domain semantics —
  equality against each integer, so an integral-valued float is accepted and
  a fractional one refused.  The default cap uses a `range` derived from
  `LEGAL_RATES` (only when that tuple is the contiguous range; the tuple
  itself remains the fallback), a non-default cap uses `range(1, cap + 1)`,
  and an unhashable rate still raises `TypeError` before any membership test.
- Geometry-keyed derivations take an optional **caller-owned** memo
  (`_ExpertIntake._memo`, one per loaded layer, never module-global):
  `_resolve_tcq_profile`, `validate_rate_schedule`,
  `completion_limit_from_elements`, `completion_widths`,
  `slicing._steps_of`, `lane_planes.require_no_completion_plane`.  Keys are
  `(... , len(rates), hash(rates))` with the full schedule stored and compared
  on a hit; the first single-slot version thrashed between the gate/up and
  down geometries and was caught by profiling.
- `kernel_wire`: the byte-reversed word view is built once per wire and
  handed to all three span-2 packers (`words=`); `_nibble_at` selects the
  nibble with one shift.
- `_load_wire` docstring corrected (it claimed a decoded stock tile); the
  `A4ExpertAxis` docstring now states precisely that a written expert's planes
  are never held twice while a pending `w13` half coexists with the
  destination buffers during intake.

## Measured before/after, same instrument and inputs

Probe timers: `loader_seconds` wraps only the `_load_wire` call;
`loop_seconds` includes the probe's per-callback accounting; `wire_bytes_read`
is the sum of the actual per-wire `tensor_bytes` (projections differ, so it is
not `count x constant`).  Units are MiB = 2**20 and MiB/s = bytes / 2**20 / s.

| geometry | before (`a0d85cc`) | after (`4d512bf`) | loader speedup |
|---|---|---|---|
| 64 experts / 192 wires, rank 0 | 805,557,824 B, 2.396 s loader = 320.6 MiB/s (loop 2.858 s) | same bytes, 1.427 s = 538.4 MiB/s (loop 1.885 s) | **1.68x** (loop 1.52x) |
| 288 experts / 864 wires, rank 0, real finalize | 3,625,010,208 B, 9.483 s = 364.5 MiB/s (loop 11.757 s) | 5.175 s = **668.1 MiB/s** (loop 7.477 s) | **1.83x** (loop 1.57x) |
| 288 experts / 864 wires, rank 1, real finalize | 9.308 s = 371.4 MiB/s | 5.852 s = **590.8 MiB/s** | **1.59x** |

Loader-frame profile, same probe (192 wires): `_load_wire` 3.774 → 1.845 s,
`parse_unit_metadata` 2.205 → 0.940 s, `verify_plane_region` 0.682 → 0.422 s,
`prepare_span2_compact` 1.394 → 0.723 s, `_check_rate` 3.28 M → 0.67 M calls,
`completion_capacity` 1.97 M → 12.7 k calls.  The profile's own accounting
overhead (`_recurse_add_to_result`, `_walk_tensors`) is reported separately in
the receipts and is excluded from `loader_seconds`.  Every run follows the
same warm-up: expert 0 absorbs the Triton JIT, then the run rebaselines.

An earlier revision of this document quoted 12 s → 7 s from whole-second
UTC callback timestamps; those figures were 1-second-quantized.  The table
above is the same-instrument measurement, and the coarse form should not be
quoted as a speedup.

## Memory (unchanged by this commit)

Full one-expert-layer, rank 0 and rank 1: allocated 1,818,922,496 B,
reserved 1,847,590,912 B, 17 segments, retained packed planes 1,815,281,280 B
per rank, `reserved/allocated = 1.016`, oracle PASS.  Both-host time series
(2 s cadence, Netdata `mem.available` plus `/proc/meminfo`
`MemAvailable`/`Cached`/`Shmem`/`Mlocked`) covering each run window are in
`loader-memory/receipts/full-layer-r{0,1}-perf2/netdata-series.txt`
(44 samples per run; 11+ peer samples each).

## Correctness gate

- Red/green through PrismaBuild: with the review-found `1 <= rate <= cap`
  predicate temporarily restored, exactly the two domain tests fail
  (`loader-memory/receipts/pb-red-rate-domain.txt`, 2 failed / 24 passed);
  with the fix in place the same suite is **26 passed, 0 skipped**
  (`pb-green-throughput.txt`, action
  `a34224fabbd80f8ce2299719714fed238594b577c3dcc83187a1a66fb8bc1d84`,
  CAS result `757302ad4c50e9f041667ef22b732c84429c5e6af19c13cad9f1b474f84e3c72`,
  sparklina GB10).
- `tests/test_loader_throughput_contract.py`: the historical rate predicate is
  spelled out in the test and must agree with `_check_rate` over ints, bools,
  floats, Fractions and an unhashable list, at the default and a non-default
  cap; every corpus artifact parses identically with and without a shared
  memo, and a mutation is refused identically after a warm memo; on a real
  ≥ 1 MiB expert wire the parallel path keeps payload-before-plane error
  precedence, still compares content digests, joins its thread on every exit,
  and keeps malformed-input error classes (`foreign magic`, header size,
  truncation).
- Both TP cuts: 2-expert real-axis finish and full 288-expert production
  finalize compared field-by-field against the materialising reference with
  the `fused.shared_lut_global` gate/up join; oracle PASS and scratch-reuse
  invariance hold (`receipts/perf-gate2-r{0,1}`, `full-layer-r{0,1}-perf2`).

## Remaining limitations

- `Manifest.decode` still calls `bresenham_rate_schedule` twice per wire
  (~7 % of the remaining probe load time); untouched.
- The probe's intake order is file/key order for one layer, so its memory
  figure is not a production-order peak bound (see
  `loader-memory/ASTRA-F5-ORDER-FINDING.md`); per that finding, writing each
  projection directly into its final destination and deferring only the small
  LUT/global reconciliation is the next structural step, not part of this
  change.
- Full-model memory acceptance and any serve claim remain with the full-run
  owner (`FULL-LOAD-MEMORY-ACCEPTANCE.md`).

## Standing note

One ten-line local CPU equivalence check of the rate predicate was run outside
PrismaBuild while drafting the fix; the same matrix is asserted inside the PB
suite above, which is the evidence that stands.  The task-local PB action
payload is untracked in the worktree and retained at
`loader-memory/pb-staging-tests.sh`.
