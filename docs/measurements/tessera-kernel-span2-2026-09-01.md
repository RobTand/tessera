# The kernel lane decodes the minor-1 wire at no cost — span-2 tuple GEMV (2026-09-01)

**Claim.** The Triton tuple GEMV now decodes a span-2 body over a LUT scale
plane — the shipping wire since `616f5e8` — bit-exactly, reading the wire's
own 4.0 b/wt (3.75 of body, 0.25 of scale nibbles; the LUT plane is *not*
materialised to E4M3 bytes), and at the same launch shape it is as fast as
or faster than the span-1 kernel it replaces. Rob's constraint was
*"saving bits just to re-spend them in memory is bad — update the kernels to
compensate"*: the resident bytes equal the on-disk bytes, and the decode pays
nothing for the label plane.

## Method (principle 15)

`experiments/tessera_kernel_span_bench.py`. One GLM routed expert
(`layers.20.mlp.experts.0.gate_proj`, 2048×4096, the shipping shape) encoded
at the E2M1×2 cap (R=7). Each arm runs a timed loop of ≥20 s so the Netdata
`nvidia_smi.gpu_power_draw` series (10-s samples on sparky) has a window to
read; the in-process number is CUDA-event ms per call, attributed by a
`torch.profiler` pass to the kernel's self CUDA time. The planes (4.2 MB per
call) stay L2-resident across the loop, so every number here is *decode*
time, not DRAM time — the right quantity for a before/after on the decoder,
the wrong one for a serving-throughput claim. The bf16 anchor is a torch
GEMV over the same weight (16.8 MB, also L2-resident: 1.04 TB/s).

Results: `experiments/results/tessera_kernel_span_bench.json` (20-s runs),
`tessera_kernel_shape_sweep.json` (4-s runs, six launch shapes). Envelope
~140 W (`CLAUDE.md` §4.15).

## Before / after

| arm | launch (lanes, split-K) | ms/call | self CUDA | bytes/call | power |
|---|---|---:|---:|---:|---:|
| bf16 torch GEMV (anchor) | — | 0.0161 | — | 16.8 MB | 100 W |
| span-1 tuple GEMV, **before** (`d51edce`) | (64, 32) | 0.0674 | 0.0658 | 4.20 MB (3.5 body + 0.5 E4M3 bytes) | 59 W |
| span-2 | (64, 32) | *refused* (`pack_kernel_planes`) | | | |
| span-2, first cut (`subset_lut` as a dependent load) | (64, 32) | 0.0751 | 0.0737 | 4.20 MB | — |
| span-1, after | (64, 32) | 0.0673 | 0.0667 | 4.20 MB | 58 W |
| **span-2, after** (values in subset order) | (64, 32) | **0.0664** | 0.0650 | 4.20 MB (3.75 body + 0.25 nibbles) | 60 W |
| span-1, after | (64, 128) | 0.0589 | 0.0584 | 4.20 MB | 68–74 W |
| **span-2, after (default shape)** | (64, 128) | **0.0524** | 0.0511 | 4.20 MB | 75 W |

Bit-exactness: one-hot columns through the kernel equal `reconstruct_unit`
by `torch.equal` on 256×512 units over both the E2M1 and the free-16 tuple
grids (`tests/test_kernel.py::test_span2_one_hot_gemv_is_bit_exact`); on the
real expert the relative error against the reference decode is 3.4e-07 for
both kernels — the fp32 atomic-add order noise of split-K, as at span 1.

Shape sweep (4-s runs, ms/call, span-1 / span-2): (64,32) 0.0669/0.0665 ·
(64,64) 0.0685/0.0644 · (64,128) 0.0587/0.0519 · (32,64) 0.0609/0.0586 ·
(32,128) 0.0589/0.0517 · (16,128) 0.0620/0.0673. Both kernels are
latency-bound (58–60 W at 64 programs in flight; 68–75 W at 256): the
2048-row expert makes only two program rows, so split-K is what fills the
48 SMs. The span-2 wrapper defaults to split-K 128; the span-1 wrapper is
left at 32 as the legacy lane, with the sweep on record.

## What changed in the decode

- **Select plane: one bit per pair.** The pair index of a lane, `base/2`,
  is a multiple of 4 but not of 8, so the window's sub-byte offset is no
  longer a constant: it is computed per lane (`pm = p % 8`) and the shift is
  a tensor. A column of pairs must be a multiple of 8 pairs (16 codes);
  the plane ends in 8 bytes of slack so the 3-byte window read on the last
  pair of the last column stays inside the tensor (the span-1 kernel's
  equivalent read is one byte past its plane on the last column — latent,
  not changed here).
- **Label plane: two bits per pair,** MSB-first; a lane's four pairs are one
  byte, aligned because `base` and `steps` are multiples of 8.
- **Point plane: unchanged** — the span-1 point plane, byte for byte.
- **Label derivation in the kernel.** `label_lut[window]` (128 entries,
  512 B, shared) gives the pair's super-label; position 1 takes the stored
  label, position 0 the super-label minus it mod 4 (`trellis.py`,
  `decode._replay_span`).
- **The anchor index is arithmetic.** The first cut looked the anchor up in a
  `(label, point)` table — a second dependent load per code — and was 12%
  slower than span 1. The four subsets partition the anchors, so permuting
  the per-unit value table into subset order (`build_subset_values`) makes
  the value of `(label, point)` sit at `(label·POINTS + point)·arity`: the
  dependent-load depth per code is the span-1 kernel's (window bytes → one
  small table → the value), and the small table is 512 B where the span-1
  fused index table is 16 KB. That is where the 12% went, and a little more.
- **Scale plane at wire size.** A nibble per (row, 16-column group), two
  per byte, into a 16-entry fp32 table (`pack_scale_nibbles`,
  `lut_scale_table`); the unit's global scale stays a scalar multiply on the
  accumulator, as before.

## Limits

- The E2M1×2 tuple family only. The scalar R=3 lane (`tessera_gemv`,
  `tessera_dequant`, the prefill `tessera_gemm`) is span-1; a span-2 arity-1
  unit has no kernel path yet. Prefill for the tuple family goes through
  `materialize_nvfp4` (the reference decoder) as it did before.
- An S6b plane at span 2 has no kernel path (`pack_unit_for_kernel`
  refuses); the shipping wire is LUT.
- L2-resident numbers. A model-scale decode streams planes from DRAM, where
  the 4× fewer bytes than bf16 matter more and the 3.3× decode-time gap to
  the bf16 anchor matters less; that number needs a many-expert rotation,
  not this bench.

Built: `src/tessera/kernel.py` (`pack_kernel_planes(span=2)`,
`pack_scale_nibbles`, `lut_scale_table`, `build_span2_luts`,
`build_subset_values`, `_tuple_gemv_span2_kernel`, `tessera_gemv_tuple_span2`,
`pack_unit_for_kernel`, `gemv_from_packed`); tests in `tests/test_kernel.py`.
