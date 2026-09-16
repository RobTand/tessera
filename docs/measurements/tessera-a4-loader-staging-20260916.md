# Native A4 loader staging: the `max_split_size_mb` slab amplification

Measured 2026-09-16 in the pinned image
`localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb`
with the real `TesseraNvFp4MoEMethod._load_wire` intake over real wires from
`merged-4c384e60`, TP2 rank-local geometry, PyTorch CUDA allocator capped at
8 GiB of the actual device total and the container bounded at 16 GiB memory
with no swap.  No full model was constructed and no server was started.
This document is a measurement of the loader's staging, not a full-model
resident-footprint proof; `f4` under its own owner measures that end to end.

## The runtime context the loader runs in

`vllm/v1/worker/gpu_worker.py` wraps `model_runner.load_model` in
`_scoped_allocator_max_split(max_split_size_mb=20)`; CuMem weight pools are not
active by default (`enable_cumem_allocator=False`), so this is the only
allocator-state difference between a bare probe and the real load.

## Measured amplification (same inputs, only that setting toggled)

All byte figures below are exact; MiB is `2**20`, GiB is `2**30`.

| run | live (unique storages) | allocated | max allocated | reserved = max reserved | segments | inactive 20.0 MiB slabs |
|---|---|---|---|---|---|---|
| 16 experts, default context | 100,873,344 B = 96.2 MiB | 102,510,592 B = 97.8 MiB | 112,602,112 B = 107.4 MiB | 140,509,184 B = 134.0 MiB | 22 | 1 |
| 16 experts, `max_split_size_mb:20` | 100,873,344 B = 96.2 MiB | 102,510,592 B = 97.8 MiB | 112,602,112 B = 107.4 MiB | **1,189,085,184 B = 1134.0 MiB** | 72 | 51 |
| 64 experts, default context | 403,493,376 B = 384.8 MiB | 411,079,680 B = 392.0 MiB | 421,170,176 B = 401.7 MiB | 459,276,288 B = 438.0 MiB | 75 | — |
| 64 experts, `max_split_size_mb:20` | 403,493,376 B = 384.8 MiB | 411,079,680 B = 392.0 MiB | 421,170,176 B = 401.7 MiB | **4,800,380,928 B = 4578.0 MiB** | 282 | — |

The active block tables are identical in both contexts: the whole difference
is dead 20.0 MiB slabs, each created by a ~3.75 MB request (the per-wire
`_plane_words` reversal buffer, and the packed BODY transfer before staging)
and not reused under that setting.  This is a demonstrated allocator
amplification consistent with f3's early pressure; it is not a forensic proof
that it is the sole contributor there.

## The fix (approved source `22957de`)

`compact_prep._plane_u8` and `kernel_bits._plane_words` fill a
**caller-owned reusable buffer** when the loader hands one in
(`_ExpertIntake._scratch`, one dict per loaded layer, never module-global);
without `scratch` both keep their original single-allocation path.
`serving.native_a4.A4ExpertAxis` allocates its stacked planes once on the
first `put`, copies each expert into its own slot, drops the per-expert unit,
and `finish()` returns those buffers with no stacking copy.

## Measured after the fix (exact same context, same inputs)

The axis buffers are allocated on the first `put`, so a *partial* run already
shows the destination allocation; the separate post-`real_finalize` figures
are distinguished below.

| run | geometry, stage | live / retained unique planes | allocated | max allocated | reserved = max reserved | segments |
|---|---|---|---|---|---|---|
| `p16-r0-split20-fix3` | 16 experts, axes buffered (288 slots) | 1,827,340,000 B = 1742.7 MiB | 1,830,977,024 B = 1746.2 MiB | 1,840,301,568 B = 1755.1 MiB | 1,847,590,912 B = 1762.0 MiB | 17 |
| `p64-r0-split20-fix2` (staging only, per-unit axis) | 64 experts | 415,552,096 B = 396.3 MiB | 423,138,816 B = 403.5 MiB | 428,255,744 B = 408.4 MiB | 457,179,136 B = 436.0 MiB | 74 |
| `full-layer-r0` | 288 experts, **after real finalize** | 1,815,281,280 B = 1731.2 MiB (gate 605,388,672 + up 605,388,672 + down 604,503,936) | 1,818,922,496 B = 1734.7 MiB | 1,840,301,568 B = 1755.1 MiB | 1,847,590,912 B = 1762.0 MiB = 1.720703125 GiB | 17 |
| `full-layer-r1` | 288 experts, **after real finalize** | same as r0 | 1,818,922,496 B = 1734.7 MiB | 1,840,334,336 B = 1755.1 MiB | 1,847,590,912 B = 1762.0 MiB = 1.720703125 GiB | 17 |

`process_weights_after_loading` ran for real over the fully loaded method on
both TP cuts (all 864 wires and input scales present); zero inactive 20.0 MiB
slabs remain.  The reserved/allocated ratio after finalize is 1.016.

Per-expert-layer logical planes: 1,815,281,280 B = 1,731.19 MiB = 1.69061 GiB
per rank.  All 42 routed expert groups in the artifact declare the identical
geometry (288 experts; `w13` 4096x4096, `w2` 4096x2048, q256 896, E2M1x2,
gate/up 2048 rows each; hidden 4096, TP2 rank-local intermediate 1024), so the
**pure packed stack total is 1,815,281,280 x 42 = 76,241,813,760 B =
71.005722 GiB per rank**.  That is the packed plane bytes alone: it does not
include JIT/context allocations, runtime buffers, KV cache or OS headroom,
and it is not a full-model footprint claim.

## Independent numerical gate

- 2-expert geometry: the real axes were finished; every finalized slot was
  compared field by field (`torch.equal` on select/label/point/nibbles/
  lut_bytes/label_lut/code_nibbles and the geometry scalars) against the
  materialising reference reader (`parse_tessera_expert_blob` +
  `shard_parsed_roles` + `prepare_span2_planes`), with the loader's own
  `fused.shared_lut_global` gate/up join applied to the reference halves.
- 288-expert geometry: after loading all 288 experts, **2 sampled experts x 3
  projections** per TP cut were compared the same way against the finalized
  `tessera_a4_*_stack` slots.  This is a sampled gate, not an all-288 oracle.
- Scratch reuse leaves an expert already written byte-identical.
- A failed comparison exits non-zero with `status: FAIL`.
- Standalone regressions through PrismaBuild (action
  `c766d26e47c20cb10ec5519aced112a4cecdf7b211abfe6ce7d68b6f631f2e51`,
  CAS `d778f36af6aebd53063342e7959c2b081ae87edbcde67d1082e8b89cd7de594c`,
  sparklina GB10): **19 passed, 0 skipped, 13 tests device-allocated** —
  `tests/test_native_a4_loader_staging.py` plus the migrated
  `tests/test_serving_nvfp4_moe_route.py` device cases including the stock
  `scaled_fp4_quant` + `_scaled_mm` apply oracle.

## Payload accounting and the full-model estimate (not a measurement)

Root-computed from the artifact manifest (global = both ranks): raw protected
expert weights 13.5 GiB global (TP2 sharded); other weights 14.070247 GiB
global; dense wires 0.703290 GiB global; other misc 0.078752 GiB; vision
1.049837 GiB excluded by `--language-model-only`.  The full-model weight
estimate is **85-88 GiB/rank plus 6 GiB KV and runtime/OS headroom**; f4
measures the actual figure.

## Receipts and evidence

- Probe receipts: `loader-memory/receipts/{p16-r0-*,p64-r0-*,gate2-r*,full-layer-r*}/`
  (`summary.json` + per-callback `callbacks.jsonl` + CUDA snapshots).
- The PB action payload for the standalone tests is the untracked
  `pb-staging-tests.sh` in this worktree; it is retained as bounded evidence.
- The approved serving source for the upcoming `f4` freeze is this branch's
  `22957de`.

## Standing exception

One standalone CPU `torch.flip(out=...)` API probe was run outside
PrismaBuild during the diagnosis; it is recorded here rather than papered
over.  Every later standalone test went through PrismaBuild; the actual
vLLM-method probes are exempt under the vLLM carve-out.
