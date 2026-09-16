# Native A4 loader staging: the `max_split_size_mb` slab amplification

Measured 2026-09-16 in the pinned image
`localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb`
with the real `TesseraNvFp4MoEMethod._load_wire` intake over real wires from
`merged-4c384e60`, TP2 rank-local geometry, PyTorch CUDA allocator capped at
8 GiB of the actual device total and the container bounded at 16 GiB memory
with no swap.  No full model was constructed and no server was started.

## The runtime context the loader runs in

`vllm/v1/worker/gpu_worker.py` wraps `model_runner.load_model` in
`_scoped_allocator_max_split(max_split_size_mb=20)`; CuMem weight pools are not
active by default (`enable_cumem_allocator=False`), so this is the only
allocator-state difference between a bare probe and the real load.

## Measured amplification (same inputs, only that setting toggled)

| run | live | allocated | reserved | segments | inactive 20 MiB slabs |
|---|---|---|---|---|---|
| 16 experts, default context | 96.2 MiB | 97.8 MiB | 134.0 MiB | 22 | 1 |
| 16 experts, `max_split_size_mb:20` | 96.2 MiB | 97.8 MiB | **1134.0 MiB** | 72 | 51 |
| 64 experts, default context | 384.8 MiB | 392.0 MiB | 438.0 MiB | 75 | — |
| 64 experts, `max_split_size_mb:20` | 384.8 MiB | 392.0 MiB | **4578.0 MiB** | 282 | — |

The active block tables are identical in both contexts: the whole difference
is dead 20 MiB slabs, each created by a ~3.75 MB request (the per-wire
`_plane_words` reversal buffer, and the packed BODY transfer before staging)
and never reused under that setting.  At ~one slab per wire, the full
42-layer routed intake would have churned roughly 17 GiB of reserved CUDA
memory per layer — the scale that matches the full-run pressure on both
Spark hosts.

## The fix (this commit)

`compact_prep._plane_u8` and `kernel_bits._plane_words` now fill a
**caller-owned reusable buffer** when the loader hands one in
(`_ExpertIntake._scratch`, one dict per loaded layer, never module-global);
without `scratch` both keep their original single-allocation path.
`serving.native_a4.A4ExpertAxis` allocates its stacked planes once on the
first `put`, copies each expert into its own slot, drops the per-expert unit,
and `finish()` returns those buffers with no stacking copy.

## Measured after the fix (exact same context, same inputs)

| run | geometry | allocated | reserved | segments | inactive 20 MiB |
|---|---|---|---|---|---|
| `p16-r0-split20-fix3` | 16 experts | 1746.2 MiB | 1762.0 MiB | 17 | 0 |
| `full-layer-r0` | 288 experts, production finalize | 1818.9 MiB | 1847.6 MiB | 17 | 0 |
| `full-layer-r1` | 288 experts, production finalize | 1840.3 MiB | 1847.6 MiB | 17 | 0 |

The 16-expert figure includes the 288-slot preallocation the serving layer
needs; the full-layer numbers are the corrected residency: ~1.8 GiB reserved
per rank per expert layer with `reserved/allocated = 1.016`, versus ~17 GiB
of would-be churn.  `process_weights_after_loading` ran for real over the
fully loaded method (every expert and input scale present) on both TP cuts.

## Independent numerical gate

On both TP cuts: a 2-expert geometry finished its real axes and a full
288-expert layer ran the production finalize; each finalized slot was
compared field by field (`torch.equal` on select/label/point/nibbles/
lut_bytes/label_lut/code_nibbles and the geometry scalars) against the
materialising reference reader (`parse_tessera_expert_blob` +
`shard_parsed_roles` + `prepare_span2_planes`), with the loader's own
`fused.shared_lut_global` gate/up join applied to the reference halves.
Reusing a scratch buffer across experts leaves an expert already written
byte-identical.  A failing comparison exits non-zero with `status: FAIL`.

## The standing exception

One standalone CPU experiment (`torch.flip(out=...)` API probing) was run
outside PrismaBuild during this diagnosis; it is recorded here rather than
papered over.  Every later standalone test is submitted through PrismaBuild;
the actual vLLM-method probes above are exempt under the vLLM carve-out.
