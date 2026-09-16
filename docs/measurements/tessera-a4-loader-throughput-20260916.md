# Native A4 loader throughput: repeated-work elimination

Measured 2026-09-16 in the pinned image
`localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb`
with the real `TesseraNvFp4MoEMethod._load_wire` intake over real wires from
`merged-4c384e60`, TP2 rank-local geometry, PyTorch CUDA allocator capped at
8 GiB of the actual device total, container 16 GiB with no swap.  No full
model was constructed and no server was started.

## Causal finding

The f5 loader profile (both hosts, in-process `py-spy` on the worker's main
thread) put 97 % of the sampled second in `_load_wire`, and inside it:

| item | head (sparky) | peer | nature |
|---|---|---|---|
| `verify_plane_region` (two SHA-256 passes) | 4.36 s | 4.25 s | mandatory digests, serialised |
| `_check_rate` | 2.14 s | 2.23 s | a linear scan of the legal-rate tuple per column |
| `_steps_of` (`body_bits` per column) | 0.85 s | 0.72 s | per-wire re-derivation of one geometry fact |
| `completion_limit_from_elements` / `completion_widths` | 1.19 s | 1.02 s | 4096 `completion_capacity` calls per wire, same answer every wire |
| `encoder_profile_id` search | 0.56 s | 0.52 s | same (code, grid) pair resolved per wire |
| `_plane_words` | 0.20 s | 0.23 s | byte-reversal recomputed three times per wire (once per packer) |

None of it was wire decoding: all of it was **per-wire re-derivation of
layer-constant facts, re-scanning of constant domains, or two hash streams
that do not have to run one after the other**.

## Changes (this commit)

- `container.verify_plane_region`: for regions ≥ 1 MiB the per-plane checks
  (including the content digests) run on one short-lived worker thread while
  the whole-region payload digest is computed on the calling thread.  The
  checks, their precedence (payload digest first) and every message are
  unchanged; `hashlib` releases the GIL, so the two SHA-256 streams overlap.
- `grammar._check_rate`: the default cap's domain is a frozenset constant and
  a non-default cap is an integer-range test — no tuple rebuilt or scanned per
  column.
- Geometry-keyed derivations (`_resolve_tcq_profile`, `validate_rate_schedule`,
  `completion_limit_from_elements`, `completion_widths`,
  `slicing._steps_of`, `lane_planes.require_no_completion_plane`) take an
  optional **caller-owned** memo dict (`_ExpertIntake._memo`, one per loaded
  layer, never module-global).  Keys are `(... , len(rates), hash(rates))` with
  the full schedule stored and compared on a hit, so a digest collision cannot
  share an entry and a different wire always re-derives.  The first version of
  this memo was single-slot and thrashed between the gate/up and down
  geometries (96 of 192 calls re-derived); it is a keyed table now.
- `kernel_wire`: the byte-reversed word view is computed once per wire and
  handed to all three span-2 packers (`words=`), instead of each packer
  rebuilding it.
- `_nibble_at`: one shift selects the nibble, no `where` and no second mask
  tensor.
- `_load_wire` docstring corrected (it claimed a decoded stock tile).

## Measured after (same inputs, exact same allocator context)

In-process probe, 64 experts = 192 wires, TP2 rank 0:

| | load | wires/s | MB/s |
|---|---|---|---|
| before (`22957de`) | 5.018 s | 38.3 | 153.1 |
| after | 3.021 s | 63.6 | 254.3 |
| | | | **1.66x** |

Full one-expert-layer load with the production
`process_weights_after_loading` (288 experts = 864 wires):

| rank | before | after | speedup |
|---|---|---|---|
| sparky (rank 0) | 12.00 s = 72.0 wires/s = 302.1 MB/s | 7.00 s = 123.4 wires/s = 517.9 MB/s | 1.71x |
| sparklina (rank 1) | 11.00 s = 78.5 wires/s = 329.5 MB/s | 7.00 s = 123.4 wires/s = 517.9 MB/s | 1.57x |

Loader-frame profiles (192 wires, same probe): `_load_wire` 3.774 → 1.845 s,
`parse_unit_metadata` 2.205 → 0.940 s, `verify_plane_region` 0.682 → 0.422 s,
`prepare_span2_compact` 1.394 → 0.723 s, `_check_rate` 3.28 M calls → 0.67 M
calls (0.565 → 0.029 s), `completion_capacity` 1.97 M → 12.7 k calls.

Memory is unchanged by this commit: full-layer rank 0 and rank 1 each
1,818,922,496 B allocated, 1,847,590,912 B reserved, 17 segments, retained
packed planes 1,815,281,280 B — the same figures as the staging/preallocation
fix, still ~1.016x reserved/allocated.  Both-host Netdata/`meminfo` snapshots
around each full-layer run are in
`loader-memory/receipts/full-layer-r{0,1}-perf/netdata-{before,after}.txt`.

## Correctness gate

- Both TP cuts, 2-expert geometry: real axes finished; every finalized slot
  `torch.equal` to the materialising reference reader with the loader's own
  `fused.shared_lut_global` gate/up join; scratch-reuse invariance holds
  (`loader-memory/receipts/perf-gate2-r{0,1}`).
- Both TP cuts, full 288-expert layer: production finalize ran; 2 sampled
  experts x 3 projections compared the same way; oracle PASS.
- PrismaBuild run of the standalone suites
  (`tests/test_native_a4_loader_staging.py` plus the migrated
  `tests/test_serving_nvfp4_moe_route.py` device cases, which carry the
  corruption/refusal/load-order/TP checks): **19 passed, 0 skipped, 13
  device-allocated**, action
  `a39417a5c2d88bc5fcca677e548a4b1417948cbd7f4ac75474b375628ec390f7`
  (durable receipt: `loader-memory/receipts/pb-throughput-gate.txt`;
  an earlier identical run was `acb57def77d7`).

## Remaining measured opportunity (not taken here)

`Manifest.decode` still calls `bresenham_rate_schedule` twice per wire
(0.21 s of the 3.0 s probe, ~7 %), and the first digest pass is still one
full SHA-256 over the region.  Both are bounded, local follow-ups; neither
blocks the current gain.

## Standing note

This is a loader-throughput measurement, not a full-model footprint or
end-to-end serve claim.  Full reload remains with the f5/f6 owner.
