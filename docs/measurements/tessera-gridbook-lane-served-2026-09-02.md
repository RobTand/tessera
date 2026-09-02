# The Tessera wire served by Gridbook's lane (2026-09-02)

**Result.** Gridbook's `TESSERA_NVFP4` lane serves the 4.0-bpp Tessera wire
(E2M1x2, q256 = 896, TCQ span-2 over the LUT16 scale plane) on vanilla vLLM
0.28.0 and reproduces the stock NVFP4 arm's served KL within the kernel's own
floor. Served KL-vs-BF16 lower bound, Qwen3-0.6B, WikiText 8 × 512:

| arm | mode | bytes on disk (bpp) | resident (bpp) | KL ≥ | top-1 |
|---|---|---|---|---|---|
| stock NVFP4 checkpoint of the same encode | eager | 4.50 | 4.50 | 0.6404 | 58.8% |
| Gridbook lane, `resident` | eager | 4.00 | 4.50 | 0.6316 | 58.4% |
| Gridbook lane, `streamed` | eager | 4.00 | ~4.00 + transient tile | 0.6316 | 58.4% |
| stock NVFP4 | compiled + CUDA graphs | 4.50 | 4.50 | 0.6220 | 59.5% |
| Gridbook lane, `resident` | compiled + CUDA graphs | 4.00 | 4.50 | 0.6271 | 59.2% |
| Gridbook lane, `streamed` | compiled + CUDA graphs | 4.00 | ~4.00 + transient tile | 0.6271 | 59.2% |

The acceptance was "the served KL reproduces the stock arm's 0.640 within the
kernel's nondeterminism floor". The floor is measured, not assumed, on the
criterion's own axis: the stock kernel against itself, same bytes, same GEMM,
eager versus compiled, moves KL-vs-BF16 by 0.018 and reads 0.247 mutual KL
(top-1 70.4%). The lane differs from the stock arm by 0.009 on that axis
(0.257 mutual eager-vs-eager, 0.248 compiled-vs-compiled), inside the
kernel's own spread. The two residency modes are bit-identical to each other,
eager and compiled (mutual KL 0.0000, top-1 100%): the streamed mode decodes
the same tile the resident mode holds.

The lane is not better than the stock arm. A −0.009 on a number whose 1σ is
0.010 is a draw; see the noise section for how that σ was established.

## What was measured, and against what

* **Checkpoint.** `experiments/export_gridbook_tessera.py` on Qwen3-0.6B at
  the wire default (E2M1x2 q896, TCQ, LUT16): 112 vLLM-fused modules, 196
  units, `wire_bytes` 220 301 312 (4.0018 bpp over 440 401 920 quantised
  params), 4.0044 bpp on disk with the per-module manifest, 4.5 bpp resident
  in `resident` mode (the decoded NVFP4 tile plus the blocked scale plane).
  Embeddings, `lm_head` and norms pass through in BF16 (622 460 928 bytes);
  the checkpoint is 842 943 478 bytes. The stock comparator is the same
  encode materialised through `tessera.stock.materialize_stock` as a
  compressed-tensors NVFP4 checkpoint (4.5 bpp on disk), served by vLLM's
  own `FlashInferCutlassNvFp4LinearKernel`
  (`tessera-stock-lane-served-2026-09-02.md`).
* **Serve.** `vllm/vllm-openai:latest` = v0.28.0, Gridbook `pip -e`'d at
  container start (`/home/rob/tessera-runs/gbfam/gbrun.sh`), branch
  `tessera/family`; the serves and the eager census are on commit
  `fe5b8f8`, the compiled census on `11d3a20` (a census-tool change over
  the same lane code). Earlier arms of the same numbers were taken on
  `a1bcd06`, `5b176eb` and `c9219a4` as the compile fixes landed; every
  resident number reproduced exactly across them.
  `experiments/gridbook_lane_served.sh <ckpt> <arm> <mode>` serves with
  `GRIDBOOK_TESSERA_NVFP4=1 GRIDBOOK_TESSERA_NVFP4_MODE=<mode>`, eager by
  default, compiled with `TESSERA_LANE_EAGER=0`; `serve_and_dump_kl.sh`
  takes `TESSERA_KL_EAGER=0` for the stock arms.
* **KL.** `/home/rob/dq-runs/kl_tool.py` dump/compare on the Qwen contract
  `/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json` against the
  image-matched BF16 teacher dump. The number quoted is the lower bound
  (`ALL KL >=`) the tool derives from the served top-k logprobs; the
  incident that produced the per-tokenizer corpus contract is in
  `kl-corpus-contract-is-per-tokenizer`.

## The lane executes what the stock arm executes

`tools/tessera_route_census.py` (Gridbook, commit `43bd94c`; takes
`--gridbook-commit` since `a1bcd06`) loads the checkpoint in-process, walks
every Linear's route record after a prefill and a decode step, and refuses
on any module that did not reach the lane. Both modes, eager: verdict
`served`, 112 of 112 modules on `torch._scaled_mm` under policy
`TESSERA_NVFP4:<mode>` and contract `e2m1_group16_ue4m3_static`, no other
route, no problems. Decode records carry M=1 at the four module shapes
(N1024:K2048, N1024:K3072, N4096:K1024, N6144:K1024); prefill records carry
the 64-token M. Under `--compiled` (vLLM's default compiled forward,
in-process) the record is written from the trace and carries `M*`, so the
census attests dispatch rather than shape there: 112 of 112 in both steps,
the same generated continuation as eager. Receipts:
`/home/rob/tessera-runs/gbfam/route_census_{resident,streamed}[_compiled]_<commit>.json`.

One census did not pass on the first run, and the reason is a product
finding rather than a lane defect. vLLM keys its compile caches (the
backbone directory and, under `VLLM_USE_AOT_COMPILE`, the 0.28 default, the
AOT-compiled forward) by `VllmConfig.compute_hash()` and the contents of the
files Dynamo traced; a Gridbook residency mode is invisible to both, so
`resident` and `streamed` shared one AOT key. The resident compiled census,
run after the streamed one in one container's cache, loaded the streamed
function (vLLM disables its guards on load) and died at the first forward
on `'NoneType' object has no attribute '_PreparedTesseraModule__roles'`,
the prepared module resident mode releases after decoding. The chain that
ran it piped the census through `tail` and reported `exit=0`; the receipt
was never written. Re-run alone in a fresh cache it served
(`route_census_resident_compiled_11d3a20.json`). The fix is Gridbook's
`compile_identity.py` (`5f70798`; the contract v13 attestation is `4fbc543`): every lane builder folds its mode and the Gridbook
release into `VllmConfig.additional_config`, the one hash input a plugin
can reach, before any cache key is computed. Verification: both compiled
censuses back to back in one cache
(`route_census_{streamed,resident}_compiled_shared_11d3a20+identity.json`),
each on its own AOT key, both `served`. The chain7 serves were never
exposed: each ran in its own container with a fresh cache and traced itself
(their logs show a Dynamo transform and a full compile, no AOT load).

## Faithfulness at the module level (fp64 reference, real inputs)

For every one of the 112 modules, on activations captured from the served
model, the lane's output and the stock kernel's output were each compared
against an fp64 reference of the same NVFP4 tile
(`/home/rob/tessera-runs/gbfam/bench/gemm_real_all.txt`):

| | stock kernel | Gridbook lane |
|---|---|---|
| max relative error vs fp64, over 112 modules | 1.687e-3 | 2.599e-3 |
| median ratio lane / stock | | 1.42 |

The lane's local error is ~1.4× the stock kernel's, uniformly, with one
inversion (`layers.2.mlp.down_proj`, 0.34×, the module whose input column
carries a 2.2 M× second-moment outlier; recorded, not explained). The
propagated difference between the two serves equals the total difference at
every module: over the 111 modules with a nonzero propagated term,
max |propagated − total| / total = 1.09e-2 (at `layers.2.mlp.down_proj`),
so no module computes something other than its input times its tile. Layer
0's `qkv_proj` sees identical inputs by construction and shows the cleanest
local number: 2.8e-3 relative between the two kernels' outputs. The lead for
the 1.4× is the lane's fp32 `out_dtype` and fp32 epilogue multiply against
the stock kernel's fused bf16 epilogue; untested.

## The floor, three ways

1. **Same bytes, same kernel, eager versus compiled.** The stock arm served
   twice: KL-vs-BF16 0.6404 eager, 0.6220 compiled (−0.018); mutual KL
   between the two serves 0.2473, top-1 agreement 70.4%. Nothing changed but
   vLLM's elementwise fusion. The lane's eager-vs-compiled mutual is 0.2445
   (70.0%) in both residency modes; lane-compiled vs stock-compiled 0.2484
   (70.3%); lane-eager vs stock-eager 0.2567 (70.7%). Every cross-kernel
   number sits where the same-kernel number sits.
2. **bf16-level perturbation on the criterion's axis.** Multiplicative
   `(1 + ε·N(0,1))` noise on every Linear output in the served model
   (`experiments/lane_numerics/noise_control.py`, `hidden_kl_band.py`).
   Mutual KL between the perturbed and unperturbed model is 0.278–0.288,
   flat in ε from 1e-3 to 5e-3. On the KL-vs-BF16 axis the 8-chunk exact
   mean (full-vocab, not the served lower bound) is 0.6887 for the stock
   tile and 0.6787 for the lane; the per-chunk delta between ε = 0 and
   bf16-level ε has std 0.0288 (stock) / 0.0261 (lane), i.e. a 1σ of 0.0102
   / 0.0092 on the 8-chunk mean. The three-ε band of the mean was ±0.002,
   which is cancellation across three draws, not the floor; the per-chunk σ
   is.
3. **Kernel swap.** `kl_stock-vs-cudnn.json`: the stock arm under two
   cuBLAS/cuDNN selections is bit-identical (KL 0, top-1 100%), so there is
   no kernel-swap floor to subtract on the stock side.

Chunk 0 alone reads stock 0.573 / lane 0.529 exact; that is a single-chunk
draw inside the observed per-chunk range (−0.059 … +0.053) and is not
headlined. The unit of account is the 8-chunk mean: 0.6887 ± 0.010 versus
0.6787 ± 0.009.

## Graph mode: the lanes could not start under vLLM's default serve

Every Gridbook lane receipt before this one was taken with `--enforce-eager`,
and the 2026-08-29 sibling receipt does not record its compile mode. vLLM
0.28's default traces the forward with Dynamo (`VLLM_COMPILE`,
`cudagraph_mode FULL_AND_PIECEWISE`, dynamic token dim) and the lanes broke
the trace three ways, none of them numerics:

1. the route record's shape string called `int()` on the token dimension,
   specialising it (`ConstraintViolationError` on `input_ids.size()[0]`);
   the E2M1 and E4M3 trellis lanes carried the same line. Fixed by
   `route_shape()` (`M*` while compiling).
2. the prepared module's fingerprint guard compared `data_ptr`
   (`Unsupported: Data pointer comparison`); skipped while compiling in
   `tessera_ops` and `trellis_ops`, eager keeps it.
3. the streamed mode's first decode resolved its CUDA extension behind a
   lock inside the traced forward (`Unsupported: Unsupported context
   manager`); resolved at preparation now.
4. the decode called the extension's pybind symbol directly, which Dynamo
   marks as skipped (`Unsupported: Attempted to call function marked as
   skipped`); it is now the registered custom op
   `prismaquant::tessera_nvfp4_decode_span2_out`, the pattern the trellis
   sibling's decode already used.

5. with those four fixed the streamed mode compiled and died at runtime in
   an Inductor-generated copy kernel (`CUDA_LAUNCH_BLOCKING=1`: an illegal
   memory access in `triton_poi_fused_3`, before the first GEMM). Every
   streamed layer decoded into one per-device pool that all 112 layers
   aliased; a compiled forward functionalises a mutation of overlapping
   graph inputs by cloning the mutated view, running the op on the clone
   and copying it back into the pool input (the generated code:
   `tessera_nvfp4_decode_span2_out_default => as_strided, clone, slice`,
   then a combo copy kernel writing the pool inputs `arg0_1`/`arg1_1`), and
   it was that copy-back that faulted: the decode's own launch check
   (`C10_CUDA_KERNEL_LAUNCH_CHECK`) would have raised first had the decode
   kernel faulted. Why the copy-back addressed out of range is torch's
   business; what is ours is that the pattern is fragile and wasteful (two
   full-tile copies per layer per forward). The forward's decode is now a
   functional custom op (`prismaquant::tessera_nvfp4_decode_module`) that
   owns the tile it returns, and the Tessera lane holds no shared pool.

Gridbook `a1bcd06` (1, 2), `5b176eb` (3), `c9219a4` (4) and `fe5b8f8` (5), with tests that trace the guarded paths through
`torch.compile(fullgraph=True)`. The compiled serves above are the receipt
that the fixes hold. Each failure was found only by serving: the resident
mode never decodes inside the forward, so it passed after (1) and (2) while
the streamed mode needed all five. The trellis siblings keep their shared
pools and mutating decode ops; their streamed compiled mode is untested and
carries the same exposure.

## Scope

* One rung: the E2M1x2 cap wire (q256 = 896). Below the cap the default
  Tessera body is the window body, which this lane does not decode; the
  Gridbook contract row (`TESSERA_E2M1_K2`, v13) therefore names one rate.
* One model, dense, TP = 1, `sm_121`. Routed MoE, where Tessera 4.0 beats
  NVFP4 4.5 on GLM experts, is not in the lane.
* The 8-bit family (E4M3 / CHANNEL, an FP8 W8A8 route) is not in the lane.
* The lane's numbers ARE the stock arm's numbers; at equal resident bytes the
  Tessera 4.0 wire still loses to production GPTQ + JSO NVFP4 at 4.5 by
  1.25× on this model (`tessera-stock-lane-served-2026-09-02.md`). What the
  lane buys is the bytes on disk and, in `streamed` mode, in memory.

## Files

`/home/rob/tessera-runs/gbfam/`: `qwen3-0.6b-tessera-k2-gridbook/` (the
checkpoint), `export_gridbook_full.log`, `route_census_*.json`,
`serve_qwen_gridbook_k2-*.log`, `kl_gridbook_k2-*.json`, `kl_lane-*.json`,
`kl_stock-graph-vs-lane-graph.json`, `bench/gemm_real_all.txt`,
`bench/noise_{teacher,stock,gridbook}.npz`, `bench/band.log`,
`lane_tests_and_band.log`, `tests_census_<commit>.log`, `chain{3,4,5}.log`.
`/home/rob/tessera-runs/stock/`: the stock arms. Dumps under
`/mnt/shared/tessera-kl/qwen_*.json.npz`.
