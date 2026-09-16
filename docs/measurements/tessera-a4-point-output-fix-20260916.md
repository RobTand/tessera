# Native A4 point-plane output allocation: direct-to-destination intake

Measured 2026-09-16 in the pinned image
`localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb`
with the real `TesseraNvFp4MoEMethod._load_wire` intake over real wires from
`merged-4c384e60`, TP2 rank-local geometry, PyTorch CUDA allocator capped at
8 GiB of the actual device total, container 16 GiB with no swap, under the
deployed `max_split_size_mb=20` context.  Probe runs also preallocate a 2 GiB
parameter-shaped pool so the allocator's composition resembles the full
constructor's, which is the condition under which the ml19 runtime diagnostic
showed the churn.  No full model was constructed and no server was started.

## Finding it fixes (ml19 runtime diagnostic, root-owned)

At layer 6 the real run held **288 pooled blocks of exactly 20,971,520 B whose
last request was 1,572,864 B** -- the point plane
(`kernel_wire.pack_span2_point_cuda`'s `torch.zeros(cols*steps_per_col*wid//8)`),
one per expert projection.  Reserved 20.545 GiB against 14.887 GiB allocated;
the standalone probe without the constructor's 7.95 GiB parameter pool pooled
only 2 such blocks, which is why the earlier probes did not see it.  The
per-wire output was allocated, copied into the axis by `put`, and dropped.

## Change

`prepare_span2_compact` takes an optional `out_factory(field, size, dtype)`;
when the loader supplies one, the packers write select/label/point straight
into the expert's preallocated axis slot and the LUT/label-LUT/code tables are
written into their slots too.  `A4ExpertAxis` grows
`destination`/`bind_geometry`/`set_lut_bytes`/`set_global`/`mark_filled`; the
intake keeps only the 16-byte LUT table and its scalar global in `pending`
until the mate arrives, then writes the joined table and shared global into
both slots.  Arbitrary wire order, the refusals, and the allocating readers
(dense route, tests, `put`) are unchanged.

## Measured (same probe, same conditions, same instrument)

| run | point-plane inactive slabs | reserved | oracle |
|---|---|---|---|
| 16 experts, 2 GiB pool, before (`a0d85cc`) | 1 block (9.17 MiB) | 3762.0 MiB | — |
| 16 experts, 2 GiB pool, after | **0** | 3762.0 MiB | — |
| 288 experts / 864 wires, 2 GiB pool, after, real finalize, rank 0 | **0** | 3762.0 MiB (17 segments) | PASS |
| same, rank 1 | **0** | 3762.0 MiB | PASS |
| ml19 runtime, real pool (root-owned evidence) | 288 blocks / 5.6 GiB | 20.545 GiB | — |

The 288-expert after-run: allocated 3,746.2 MiB (2 GiB pool + 1,731.2 MiB
final planes + scratch), `reserved - allocated` 15.8 MiB, 37 segments peak,
oracle PASS against the materialising reference with the
`fused.shared_lut_global` gate/up join, on both TP cuts.  Loader throughput
669-732 MiB/s of actual wire bytes (unchanged-to-slightly-better versus the
accepted throughput commit; no perf claim here).

Interleaved pending-order smoke (two experts pending at once: e0 gate, e0
down, e1 gate, e0 up, e1 down, e1 up) on both TP cuts: stored axes equal the
reference after both joins, zero slabs (`receipts/pt-interleave-r{0,1}`).

## Tests

PrismaBuild, pinned image on sparky/GB10: **30 passed, 0 skipped, 14
device-allocated** -- the loader-staging suite (including the new
direct-destination axis regression: two different real wires, direct path
versus allocating path, `torch.equal` on every field and `finish` returning
the axis buffers), the migrated route cases with the stock apply oracle, and
the container/grammar contract tests.  Action
`1f4654cae14a705fbd7ba61612148bb974583c027edaa3b9210720aa97652fcb`
(`loader-memory/receipts/pb-point-output-fix.txt`).

## Remaining limitations

- The bounded probe cannot reproduce the real constructor's 7.95 GiB
  parameter pool under the 8 GiB cap; the acceptance metric is the same
  snapshot class (inactive blocks whose last request is the point-plane size)
  and the real-pool baseline is the ml19 evidence.
- The repeated two-node diagnostic (frozen source, cap 32 GiB allocator /
  48 GiB container / 24 GiB headroom) is the launcher owner's run; this
  document does not claim it.
- No full-model launch, and no claim that the bounded numbers extrapolate to
  a safe full run.
