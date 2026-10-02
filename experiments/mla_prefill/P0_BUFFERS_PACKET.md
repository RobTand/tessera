# MLA pass-buffer schedule (T8): QA packet

Experiment-private, default-off. This is a **source-level ownership proof plus a
proposed CPU compile gate**, not an accepted native result. No speed claim is made here.

## Provenance

* base: Tessera `82e674041680c3b74fdf381f868c9a16f4f8f576`, kernel
  `1e7d34d2a4d112f7cd4bb76953da158d54236eab2d17aed6bc14129fcb66672d`
* Earlier source `62dc8d0fd676...` was compiled directly in Docker, outside
  PrismaBuild. **Those artifacts are EXCLUDED FROM QUALIFICATION.** They
  predate the mutant and subsequent source corrections, have no PB receipt,
  and cannot establish current register use, SASS identity or correctness.
  They remain at `/home/rob/tmp/astra-resume-20261002/t8_performance/mla-pass-buffers-p0-local/`
  as bounded evidence of the policy violation, without being deleted or relabeled.
* Root transferred implementation to Astra after that disclosure. The corrected
  build uses the existing shared loader recipe through PB and retains final
  DSOs for subsequent numerical work; it does not reuse the excluded object files.

## What changes

`src/tessera/serving/csrc/mla_prefill_mg.cu`, guarded by one compile-time
selector `TESSERA_MLA_P0_BUFFERS`:

* `0` (default) is the current qualified L0 schedule. The `#else` branch is
  textually byte-identical to the shipped region; only the shared closing brace
  moved outside the `#if/#else`. Generated default-code identity still needs
  the qualified final build; source equality alone is not that measurement.
* `1` reuses FlashInfer's two WFP8 parities (`sm.w_fp8()`, `2 x 2560 = 5120`
  bytes, `smem_layout.cuh` `SMEM_W_FP8_MG`) as the pass0/pass1 storage of a
  **single** V chunk, so one publication barrier covers both residual passes.

No new cache, resolver, registry, numeric reference, serving flag, runtime fork,
or duplicated kernel body. The stock `tessera_mla_prefill_mg_copy` kernel and the
production C ABI are untouched; the selector is experiment-private build
selection, not an ABI/`configured[2]` change.

## Schedule (region, per key tile)

| | L0 (default) | P0 buffers |
|---|---|---|
| per vc | 2 pub + 1 retire | 1 pub |
| total (4 vc) | 8 pub + 4 retire = 12 | 4 pub + 3 retire = 7 |

```text
for vc in 0..3:
  if vc>0: all-math retirement barrier          # 3 total
  for wp in 0..1: for g in 0..1:
    same si,wn, quantize_weight_quad_for_pass(...,wp) -> parity wp, group g
  all-math publication barrier                  # 4 total (one per vc)
  xv_acc = 0                                    # same scope/value as L0
  for wp in 0..1: for g in 0..1: for nt in 0..1:
    same pv_fp8_d2_16x8 on parity wp into same xv_acc[g][nt]
  same acc_o += xv_acc * vc_scale fold
```

The next key tile's existing math wait after QK (`bar_sync_t<MATH>` before the
`w_head_sc_all` reset) still precedes reuse of the W parities, and the last
tile's finalization barriers are unchanged.

## Changed-expression / consumer map

The changed numerical loop is inside the `for (int vc ...)` region. Selector
and bound assertions, the builder and experiment wrapper also change. QK,
softmax, cross-warp max, normalizer, `KvFree`/`BulkReady`, the
epilogue (`kv_buf(0)` staging), LSE, and the mask/order IO path are untouched.

| expression | L0 | P0 | operand source (identical) |
|---|---|---|---|
| `si0/si1` | `1.f/vc_sc[gid]`, `1.f/vc_sc[gid+8]` | same | `w_head_sc_all`, immutable across the loop |
| `wn00..wn11` | `w*vsc*si` | same | `p[g][*]`, `vsc_cache[vc][*]` |
| `quantize_weight_quad_for_pass` | pass `wpass` | pass `wp` | pure; recomputes residual from `wn`, never reads prior shared bytes |
| W store address | `wfp8_parity + g*WFP8_GRP_SIZE`, `wfp8_parity = w_fp8()+(vc&1)*STRIDE` | `w_fp8()+wp*STRIDE+g*WFP8_GRP_SIZE` | same 2560-byte layout, same `row*(BI+16)+key` |
| `pv_fp8_d2_16x8` | same helper | same | synchronous ldmatrix A + `d2_load_b_fp8` B + mma.sync |
| `acc_o` fold | `acc_o += xv_acc*vc_scale` | same | same association |

W writers/readers, exhaustively: writers are the two per-wp `for g` store loops;
readers are `pv_fp8_d2_16x8` only. The IO warps never touch WFP8: KV buffers
precede it; their mask and `tile_order` follow `Layout::TOTAL`. `q_rope`
aliases `OFF_W_FP8` through `OFF_SCRATCH`, but this instantiation has
`D_ROPE=0` and Q setup finishes before the loop.

Retirement paths covered: (a) between V chunks, the top-of-loop barrier at
`vc>0`; (b) between key tiles, the existing MATH barriers after QK/softmax;
(c) last tile, the unchanged finalization barriers.

## Static assertions added

* exactly two 2560-byte W parities: `Layout::SMEM_W_FP8_MG == 5120`
* closed scope: `NUM_HEADS==32`, `PAGE_BLOCK==64`, `GROUPS==2`, `Layout::N_HG==2`,
  `BI==64`, `QUANT_TILE==128`, `D_NOPE/QUANT_TILE==4`
* `D_NOPE==512`, `D_ROPE==0`, `ARBITRARY_FP32`, `SCALE_IN_KV_SMEM`, no V RoPE

## CPU compile gate (no GPU)

`experiments/mla_prefill/p0_compile_row.sh` runs inside the pinned 5be image
(`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705...`, nvcc 13.0.88,
`-use_fast_math -O3 sm_121a`), no `--gpus`.

The wrapper calls `MlaPrefillBuild` through `p0_build_only.py`, exactly the
owner used by `MlaPrefillLibrary`. Source, selectors, resolved compiler,
canonical include roots, flags and build ID have one owner. It builds final
baseline/candidate/wrong-pass DSOs once, retaining each manifest, build.ninja,
SASS and resource report. The image root is read-only and include roots must
remain inside its FlashInfer package; only checkout and output are mounted.
No raw-nvcc recipe or hardcoded alternate include tree remains.

Retained reuse verifies current source/selection/flags/compiler/header/image,
build recipe and DSO length/hash before any native load or build. Explicit
`require_retained=True` refuses a missing build rather than compiling on the
GPU worker. Default selection cannot adopt a candidate manifest. The runtime
SM121 gate remains; the CPU builder never calls a kernel.

Qualified final codegen, numerical and speed results are **pending**.

## GPU protocol (proposed; NOT authorized here)

Bounded, same-window old/new arms; no sweep.

* Host: one GB10 box, exclusive GPU, CPU-only room for the launcher; netdata and
  raw power/clock series on both boxes, before/after in-process profile, exact
  kernel census.
* Correctness: reuse the existing edge/pooled/random/all-mask/hole/tile-boundary/
  signed-zero/extreme-scale `mask_gate.py` cases plus `512/2048/2049` shapes,
  requiring exact output AND LSE. A deterministic wrong-pass-buffer mutant must
  fail; add a selector + buffer-identity control.
* Timing: first paired `2048@8192`, plus `2048@2048` control; existing ABBA arms.
  Energy/work-J stays HOLD until clock/cadence alignment is established.
* Fallthrough preserved: decode, mixed, graph/capture, unsupported shapes.

## Reuse map

* source/consumer proof: `EVIDENCE.json` `source_consumer_review` (Tessera
  `82e6740416`, kernel `1e7d34d2...`), `smem_layout.cuh`, `warp_tiles.cuh`.
* harnesses: `d1_bench.py`, `mask_gate.py`, `mask_abba.py`, `d1_row.sh`,
  `mask_row.sh`, `sass_compare.py` (all reused, not rewritten).
* prior receipts under `/mnt/shared/tessera-measurements/mla-mask-recovery-20261002/`
  and `/home/rob/tmp/claude-campaign-20260926/tmp/sol-mla-recovery-20261002/`
  are reused only for their exact identities.
