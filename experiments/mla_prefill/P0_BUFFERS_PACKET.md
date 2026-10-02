# MLA pass-buffer schedule (T8): QA packet

Experiment-private, default-off. The source proof, retained CPU builds and
bounded native operator gate are complete. The measured reductions below apply
only to the two pooled-mask resident operator cells, not full-model serving.

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
  moved outside the `#if/#else`. The retained baseline is the comparator below;
  source equality does not establish binary identity to an older shipped DSO.
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

The qualified retained builds and bounded operator results are recorded below.

## GPU protocol (completed once under lead authorization)

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
* Serving fallthrough is unchanged in source; this operator experiment does
  not exercise the vLLM decode, mixed or graph/capture dispatcher.

## Reuse map

* source/consumer proof: `EVIDENCE.json` `source_consumer_review` (Tessera
  `82e6740416`, kernel `1e7d34d2...`), `smem_layout.cuh`, `warp_tiles.cuh`.
* harnesses: `d1_bench.py`, `mask_gate.py`, `mask_abba.py`, `d1_row.sh`,
  `mask_row.sh`, `sass_compare.py` (all reused, not rewritten).
* prior receipts under `/mnt/shared/tessera-measurements/mla-mask-recovery-20261002/`
  and `/home/rob/tmp/claude-campaign-20260926/tmp/sol-mla-recovery-20261002/`
  are reused only for their exact identities.

## Accepted operator result — 2026-10-02

Frozen GPU source: `04e640554eae4f97de97c7fb8454b70188181e21`; CUDA SHA-256
`bb45f827fe34fd64222ca27db85ee1a92a6a792b05d2cc506c74b419e36c17ed`.
The CUDA source and flags equal the retained CPU-build source `1ae2e51482`.
The three PB builds (`1cf9b68af1c5`, `e77a9ed86ecb`, `4b041c2c3534`) use
image `5be13705`, nvcc 13.0.88 and SM121a, with no GPU visible. Each finished
rc0 in about 7 seconds and below 866 MB peak memory. No GPU rebuild occurred.

Final codegen: baseline/candidate have 3424/3072 instructions and 30/25
BAR.SYNC instructions, respectively; both use 168 registers, zero stack/local
memory and no LDL/STL. The copy kernel's 3048 instruction/control words are
identical across all three builds.

One PB GPU action `0c77dbd5ddccc866630e2c7e477cb67e435be1e1fd2315710a7331b03dca19aa`
finished rc0 in 566.41 seconds on Sparky's GB10. Its full canonical receipt is
`4d333730fe8853798d61bd44ee63cf5b1258804a0ca93dfe8a8e40b145ea09ed`.
CPU affinity was [5,6], native threads 1, exclusive GPU, aggregate reservation
16 GiB and GPU subset 4 GiB; cgroup peak was 1,395,040,256 bytes. The latter
is a cgroup measurement, not a complete unified-memory/GPU peak.

All 12 pooled/random served-shape cells and 5 edge cells have raw bit-exact
candidate output **and LSE** versus stock; the unchanged copy also passes.
Both wrong-pass mutant cells fail causally (2,129,868 differing BF16 output
words in total), with exact LSE and unchanged-copy output/LSE. The numerical
reference is stock FlashInfer. The performance comparator is retained old L0:
legacy JSON arm `stock` means `retained_l0`, and `l0` means `pass_buffers`.

| Pooled-mask cell | Old L0 median | Pass-buffer median | Less operator time |
|---|---:|---:|---:|
| 2048 queries, 8192 context | 8.018737 ms | 6.497623 ms | 18.9695% |
| 2048 queries, 2048 context | 4.210287 ms | 3.641979 ms | 13.4981% |

Each cell used two ABBA cycles (eight arms), conventional medians and resident
CUDA graph timing. Actual arms were 32.429–33.674 s and 31.990–32.361 s,
respectively. The separate eager Torch profiles contain three calls of
`tessera_mla_prefill_mg_l0` per arm: old/new means 7581.785/6189.434 us and
3999.691/3573.141 us. The profiles retain the same kernel identity; the
separate codegen report above establishes no local-memory spills.

Raw power and both-host Netdata CPU/power/pressure series are retained. Netdata
power's native cadence is 10 s; its returned 1 s buckets are not 1 s sensor
samples. Power and clocks vary across arms (candidate clocks are lower), and
sensor coverage does not establish energy accuracy. **Energy/work-J is HOLD**;
there is no clock normalization or utilization-based saturation claim.
The wrapper omitted the descriptive `MLA_PLACEMENT` environment from Docker,
so the raw JSON says `unqualified`; the sealed PB receipt independently binds
actual exclusive GPU placement. Historical raw output is unchanged.

No random-mask performance, vLLM dispatch/capture behavior, model quality,
full-model prefill/decode speed, T4/T16 qualification, default promotion or
runtime-pin change is established here. Common attention may apply to other
weight precisions only under the same stock geometry, FP8 KV and resolver
contract, with separate qualification.

Durable raw results are under
`/mnt/shared/tessera-measurements/mla-pass-buffers-20261002/gpu-v1/`.
Each paired directory contains `abba.json`, both before/after Torch traces and
`both-host-netdata.json`; matrix/edges/mutant contain their gate reports.
Retained DSOs and manifests are in the adjacent `build-v1/` directory:

* baseline SHA `30206f1f622dd0b09c90dd22f3daa0bd2db840fd4e93b4dff6ded6f786406f5a`
* candidate SHA `a88b11fca6e702fd659ed001ed85d336de86a0c4be790f63df4a5741a32d956d`
* mutant SHA `94623ee4bbaa48456713b62bf7dee3f6222f833efd7dc01abe2a9e4d7444e402`

The source-bundle, full-request CAS, resources and independently recomputed
results are recorded under
`/home/rob/tmp/astra-resume-20261002/t8_performance/attention_comm/` in
`NATIVE-BUILD-CAS.json`, `NATIVE-BUILD-ARTIFACTS.json`,
`BUILDER-CPU-PROOF.json`, `GPU-SOURCE-LEAD-REVIEW.json`,
`GPU-RESULT-LEAD-REVIEW.json`, `GPU-COMPLETION.json` and
`GPU-FINAL-ARTIFACTS.json`. CPU controls cover 17 distinct cases, zero skips
or missing collection, with causal failures before the retention fixes.

## Default-off serving integration

Commit `f7c8928947` selects `p0_buffers=True` through the existing cached
`library_for_device` owner, behind `TESSERA_RESEARCH_MLA_MASK_SKIP` only.
Dispatch telemetry identifies `mg_mask_skip_pass_buffers`; stock fallback
still identifies `stock_mg`. The generic loader's default remains old L0,
and the native source, builder, ABI, retained DSOs and contract are unchanged.
The actual runtime fixture now uses that production factory and refuses any
compiler call; the existing Torch cache must resolve to the retained candidate.

CPU causal RED `0ec24cd9a71f` fails both new selector/telemetry assertions,
with the stock control passing. GREEN `420ac63f6708` runs the whole existing
registration boundary file in the pinned vLLM image with CUDA disabled:
15 passed, zero skips or uncollected modules, two pytest workers/native1,
16.77 s action elapsed. Canonical receipt:
`dddc60a2f8d321596910d53bc116b0ebe8115ce3a8387e46a601985ea551f101`.
This CPU population does not qualify actual native vLLM execution. The two
runtime GPU cases and the full-model serve remain separate, pending gates.

The unrelated discovered placement forwarding omission is corrected in
`42b7c2851c`; it changes future descriptive metadata, not the historical
operator receipt or results. The wrapper syntax and new runtime fixture were
checked through CPU PB action `42bbab6f8d4e` without launching CUDA.
