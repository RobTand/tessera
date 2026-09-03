# The Tessera E4M3 wire served by Gridbook's FP8 route, and one checkpoint carrying both families (2026-09-02)

**Result.** Gridbook's Tessera lane now has two routes behind one flag pair,
and both are served. The `TESSERA_FP8` route serves the 4.07-bpp Tessera E4M3
wire (E4M3 grid, q256 = 1024, the window body at L = 14 over the CHANNEL
scale plane -- the format's own default for that grid) on vanilla vLLM 0.28.0
as the per-channel FP8 pair, W8A8 on `torch._scaled_mm` under
`fp8_per_token_dynamic`, and reproduces the stock FP8 arm's served KL. The
fresh encode is byte-identical to the stock checkpoint the stock-lane
receipt was taken on (392 of 392 quantised tensors), so that receipt is the
comparator outright. Served KL-vs-BF16 lower bound, Qwen3-0.6B, WikiText
8 x 512:

| arm | mode | bytes on disk (bpp) | model memory as served | KL >= | confident KL >= | top-1 |
|---|---|---|---|---|---|---|
| stock FP8 checkpoint of the same encode (`tessera-e8`, stock lane, `CutlassFP8ScaledMMLinearKernel`) | eager | 8.03 | 0.74 GiB | 0.4699 | 0.3894 | 63.23% |
| Gridbook lane, `resident` | eager | 4.07 | 0.73 GiB | 0.4660 | 0.3845 | 63.23% |
| Gridbook lane, `streamed` | eager | 4.07 | 0.55 GiB | 0.4660 | 0.3845 | 63.23% |
| Gridbook lane, `resident` | compiled + CUDA graphs | 4.07 | 0.73 GiB | 0.4669 | 0.3868 | 62.67% |
| Gridbook lane, `streamed` | compiled + CUDA graphs | 4.07 | 0.55 GiB | 0.4669 | 0.3868 | 62.67% |

The two residency modes are bit-identical to each other in both regimes
(mutual KL 0.0000, top-1 100%, eager, 0.0000, top-1 100%, compiled): the streamed
mode decodes, inside the forward, the same E4M3 bytes the resident mode
holds. The lane against the stock arm reads 0.0211 mutual KL
(top-1 91.9%); the lane against itself, eager versus
compiled, same bytes, same GEMM, reads 0.0269 (top-1
90.3%). The cross-kernel number sits where the same-kernel
number sits, and the -0.004 on KL-vs-BF16 is a draw.

**What this attests, and what it does not.** Faithfulness: the route serves
the wire's bytes, both modes, both regimes. Not quality: the 4.07-bpp
Tessera-8 wire reads 0.47 on this dense model against 0.0205 for per-channel
FP8 round-to-nearest at 8.0 bpp on the same route
(`tessera-stock-lane-served-2026-09-02.md`), a 23x gap that is 4 bits of
code against 8; production GPTQ+JSO NVFP4 at 4.5 bpp W4A4 reads 0.511 on
the same table -- but Tessera-8 reaches that row at **8.0 bpp resident**
against NVFP4's 4.5, so it is not a 4-bit point on this route at all: the
wire's 4.07 is a disk number on the `resident` route. The `streamed` route
does hold it in memory -- 0.55 GiB against 0.73, same KL (:17-20), which is
what this lane buys (:219) -- but it decodes inside the forward and still
computes W8A8, so what survives at either residency is the A-side: a W8A8
contract against NVFP4's W4A4. The kernel lane that would compute over the
wire itself (`tessera-window-kernel-2026-09-02.md`) has no served KL yet.
Priced as it deploys on this route
this is an 8-bit arm, and its comparator at that residency is FP8 RTN, which
it loses to. What is bad is narrower: the CHANNEL plane's blindness to
outlier input columns, and the greedy continuation of "The capital of France
is" is " 111111111111111" on the lane exactly as it is on the stock arm. The
route is the product's 8-bit half; the wire it carries is the encoder's
business.

## What was measured, and against what

* **Checkpoint.** `experiments/export_gridbook_tessera.py --grid E4M3 --q256
  1024` on Qwen3-0.6B: 112 vLLM-fused modules, 196 units, `wire_bytes`
  224 100 352 (4.0708 bpp over 440 401 920 quantised params; the 0.07 over
  4.0 is the CHANNEL plane's fp16 row scale and the per-unit window table),
  4.0734 bpp on disk with the per-module manifest. Resident in `resident`
  mode: 441 778 176 bytes, 8.025 bpp -- one E4M3 byte per weight plus one
  fp32 per row (344 064 rows), which is what the stock FP8 pair holds.
  Embeddings, `lm_head` and norms pass through in BF16 (622 460 928 bytes);
  the checkpoint is 846 726 118 bytes.
* **The comparator.** `--stock-twin` wrote the compressed-tensors
  materialisation of the same wires in the same run;
  `experiments/compare_stock_checkpoints.py` found it identical to
  `/home/rob/tessera-runs/stock/qwen3-0.6b-tessera-e4m3-q1024-fp8`, the
  checkpoint behind the stock-lane receipt, on every one of the 392 shared
  quantised tensors (`twin_vs_stock_e8.log`). The E4M3 encoder is
  deterministic across runs, so the earlier receipt (KL 0.4699, top-1
  63.23%, model memory 0.74 GiB, eager, `CutlassFP8ScaledMMLinearKernel`) is
  the comparator and the twin was not served again.
* **Serve.** `vllm/vllm-openai:latest` = v0.28.0, Gridbook `pip -e`'d at
  container start (`/home/rob/tessera-runs/gbfam/gbrun.sh`), branch
  `tessera/family`, working tree over `f419ef6` (the receipts record
  `gridbook_commit: f419ef6+fp8`; the code that ran is the code committed as
  `980fa1e`, documentation aside).
  `experiments/gridbook_lane_served.sh <ckpt> <arm> <mode>` serves with
  `GRIDBOOK_TESSERA=1 GRIDBOOK_TESSERA_MODE=<mode>`, eager by default,
  compiled (vLLM's default `VLLM_COMPILE`, `cudagraph_mode
  FULL_AND_PIECEWISE`) with `TESSERA_LANE_EAGER=0`.
* **KL.** `/home/rob/dq-runs/kl_tool.py` dump/compare on the Qwen contract
  `/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json` against the
  image-matched BF16 teacher dump; the number quoted is the lower bound
  (`ALL KL >=`) the tool derives from the served top-k logprobs; "confident"
  is its n = 1709 subset (42% of positions).
* **Model memory** is vLLM's own load report (`Model loading took N GiB`).
  Resident holds the FP8 pair and reads what the stock arm reads (0.73 vs
  0.74 GiB); streamed holds the packed window streams plus per-unit tables
  and reads 0.55 GiB, 0.18 GiB less, against 0.20 GiB by byte count (the
  pair less the wire); the remainder is the prepared tables, gather indices
  and the allocator's rounding, and was not separated.

## The route executes what the stock arm executes

`tools/tessera_route_census.py` loads the checkpoint in-process, runs one
prefill and one decode step, and refuses on any module that did not reach
the route the checkpoint declares for it (`declared_families`, from the
config's scheme per module). Both modes, eager: verdict `served`, 112 of 112
modules on `torch._scaled_mm` under policy `TESSERA_FP8:<mode>` and contract
`fp8_per_token_dynamic`, no other route, no problems. Decode records carry
M = 1 at the four module shapes (N1024:K2048, N1024:K3072, N4096:K1024,
N6144:K1024); prefill records carry the 64-token M. Under `--compiled` the
record is written from the trace and carries `M*`, so the census attests
dispatch rather than shape: 112 of 112 in both steps. The generated
continuation is the same string in all eight censuses of the day (both
families, both modes, eager and compiled). Receipts:
`/home/rob/tessera-runs/gbfam/route_census_e4m3_{resident,streamed}[_compiled]_f419ef6+fp8.json`.
The resident eager census was run twice: the first attempt died in vLLM's
memory profiler (`Initial free memory 111.06 GiB, current free memory 112.2
GiB`) because a host test suite was sharing the GPU at that moment; the
second, alone, is the receipt.

The E2M1 arm (`TESSERA_E2M1_K2`, the 4-bit receipt of
`tessera-gridbook-lane-served-2026-09-02.md`) was re-served under the unified
flags before the contract's cells were changed to name them: served KL
0.6316, unchanged (top-1 58.41%), mutual KL 0.0000 (top-1 100%) against the v13-flag
serve of the same checkpoint; all four of its censuses re-run
(`route_census_k2_*_f419ef6+fp8.json`, 112 / 112 on
`e2m1_group16_ue4m3_static` in every one).

## What the streamed mode does inside the forward

The FP8 route's streamed mode is pure torch over the wire's packed bits
(`gridbook/tessera_window.py`): the window streams stay in the layout
`tessera.lane_planes.pack_window_planes` wrote (per column, L zero pad bits
then steps x R bits MSB-first, columns byte-aligned), regrouped by rate so
one gather pattern serves every column of a rate; position t's L-bit window
starts at bit (t + 1) R and is read with one four-byte gather, a shift and
a mask into an int32 word (the window never reaches a sign-extended bit,
since shift + L <= 32), then one table gather whose table has the grid's
native byte map folded in; the inverse permutation restores column order.
No unpack, no replay, no `nonzero`, static shapes, a fresh tile per forward
-- the pattern the 4-bit route needed five fixes to reach under vLLM's
compiled forward. At preparation every role is decoded both ways and the
route refuses on any byte that differs from
`tessera.decode.materialize_fp8`; the tests do the same against Tessera's
replay on uniform and mixed-rate encodes. Host microbenchmark, one
3072 x 1024 tile: 1.96 ms eager, 0.23 ms under `torch.compile(fullgraph=True)`;
served, the decode is one more fused region of the compiled forward and was
not timed separately. That is a measurement of what the decode costs, not
a throughput claim for the mode.

## The mixed checkpoint: both families in one serve

The product deliverable is one checkpoint carrying both families, served by
one lane in one process. `mixed_plan.json` puts the 28 `mlp.down_proj`
modules on the NVFP4 route (E2M1x2 q896, TCQ, LUT16 -- the 4-bit wire) and
the other 84 modules (`qkv_proj`, `o_proj`, `gate_up_proj`) on the FP8 route
(E4M3 q1024, window L = 14, CHANNEL); `export_gridbook_tessera.py
--plan-json` wrote it in one run with its stock twin.

* **Bytes.** Wire 223 598 592 bytes, 4.0617 bpp over 440 401 920 params:
  4.0013 over the 88 080 384 NVFP4 params (28 units) and 4.0768 over the
  352 321 536 FP8 params (168 units). Resident in `resident` mode
  403 128 320 bytes, 7.323 bpp (4.5 on the NVFP4 modules, 8.029 on the FP8
  ones). On disk 4.0643 bpp with the manifest; checkpoint 846 228 018 bytes.
* **The union, byte for byte.** Against the uniform E4M3 twin, every one of
  the 336 FP8 tensors is identical; against the uniform E2M1x2 stock
  checkpoint (`stock/qwen3-0.6b-tessera-k2-q896-nvfp4`), every one of the
  112 NVFP4 tensors is identical (`compare_stock_checkpoints.py`; the tensors
  the tool reports as differing are the `weight_scale` keys the two formats
  share by name). The encoder is per-module and deterministic, so the mixed
  checkpoint is exactly the two uniform checkpoints' modules, module by
  module.
* **The twin on the stock lane.** The compressed-tensors materialisation of
  the same wires (NVFP4 triples with `input_global_scale` for `down_proj`,
  FP8 pairs elsewhere; config groups `nvfp4-pack-quantized` and
  `float-quantized`) is served by vanilla vLLM with no plugin on
  `FlashInferCutlassNvFp4LinearKernel` and `CutlassFP8ScaledMMLinearKernel`,
  model memory 0.72 GiB (`serve_qwen_stock_tessera-mixed-twin.log`).

| arm | mode | model memory as served | KL >= | confident KL >= | top-1 |
|---|---|---|---|---|---|
| stock twin of the same wires (compressed-tensors, both formats on vLLM's own kernels) | eager | 0.72 GiB | 0.6741 | 0.5609 | 57.53% |
| Gridbook lane, `resident` | eager | 0.70 GiB | 0.6772 | 0.5563 | 57.09% |
| Gridbook lane, `streamed` | eager | 0.55 GiB | 0.6772 | 0.5563 | 57.09% |
| Gridbook lane, `resident` | compiled + CUDA graphs | 0.70 GiB | 0.6733 | 0.5735 | 57.02% |
| Gridbook lane, `streamed` | compiled + CUDA graphs | 0.55 GiB | 0.6733 | 0.5735 | 57.02% |

Mutual KL lane-vs-twin 0.1030 (top-1 81.7%); lane eager-vs-compiled, same
bytes, same GEMMs, 0.1180 (81.0%); the two modes 0.0000 (100%) in both
regimes. The same pattern as both uniform arms: the cross-kernel difference
sits inside the same-kernel one, and the streamed mode is the resident mode's
bytes decoded in the forward. The mutuals are larger than the FP8 arm's
(0.02-0.03) and smaller than the 4-bit arm's (0.25): the E2M1x2 modules
carry the kernel-level spread.

* **Route census**, all four (`route_census_mixed_{resident,streamed}[_compiled]_f419ef6+fp8.json`):
  verdict `served`; in every step 28 modules on `TESSERA_NVFP4:<mode>` under
  `e2m1_group16_ue4m3_static` and 84 on `TESSERA_FP8:<mode>` under
  `fp8_per_token_dynamic`, all on `torch._scaled_mm`, matching the
  checkpoint's `declared_families` (`{TESSERA_FP8: 84, TESSERA_NVFP4: 28}`);
  M = 1 / M = 64 eager, `M*` compiled; no other route, no problems.
* **Greedy continuation** of "The capital of France is": eager " the largest
  in the world, but it is also the most important in the world", compiled
  " the capital of France is the capital of France is the capital of France
  is the" -- an argmax flipping at a near-tie in a 0.67-KL model; recorded,
  not explained (the twin's continuation was not captured).

**What the split is, and is not.** It is a demonstration that one lane
executes two tensor-core contracts from one checkpoint; nothing chose it on
cost. Its KL is worse than either uniform arm (0.640 all-E2M1x2, 0.470
all-E4M3) although every module carries exactly the bytes it carries there:
KL does not add across modules. The three arms together carry a lead, an
inference rather than a measured decomposition: moving 84 of the 112
modules from E2M1x2 to E4M3 -- the family that reads 0.470 when every
module is on it -- moved KL from 0.640 to 0.677, so the 84 modules' family
barely moves this model and `down_proj` carries the damage in every arm,
consistent with the outlier-column note in the stock-lane receipt. A
per-module decomposition would settle it; nothing here measured one. The allocation is PrismaQuant's job -- the
rungs are priced there by name (`TESSERA_<grid>_K<arity>_R<q256>`) and
admission waits on the serving pin -- and the lane is what makes whatever it
chooses shippable.

## Scope

* One rate per family: the E2M1x2 cap (q256 = 896, TCQ) on the NVFP4 route
  and the E4M3 default (q256 = 1024, window L = 14) on the FP8 route; the
  contract rows name those rates and no other. Sub-cap E2M1x2 rates (window
  body over the LUT16 plane) are not in the NVFP4 route; the window decoder
  is plane-agnostic by construction, but decoding an E2M1x2 window body
  through it is untested.
* One model, dense, TP = 1, `sm_121`. Routed MoE experts -- where Tessera 4.0
  beats NVFP4 4.5 and where every Tessera-8 win was measured -- are not in
  the lane.
* Faithfulness, not quality: every number above is the stock arm's number
  for the same bytes. On this dense model the 4.07-bpp E4M3/CHANNEL wire
  loses to 8.0-bpp FP8 RTN by 23x (4 bits of code against 8; production
  NVFP4 at 4.5 bpp is 25x behind the same FP8 arm) and the 4.0-bpp E2M1x2
  wire loses to production NVFP4 at 4.5 bpp by 1.25x under W4A4
  (`tessera-stock-lane-served-2026-09-02.md`); what the lane buys is the
  bytes on disk and, in `streamed` mode, in memory.
* The FP8 arms have no stock compiled comparator (the stock-lane receipt is
  eager-only), so the FP8 floor is the lane's own eager-versus-compiled
  mutual; the 4-bit receipt established the stock kernel's own spread on
  that axis (0.018 on KL-vs-BF16, 0.247 mutual).

## Files

`/home/rob/tessera-runs/gbfam/`: `qwen3-0.6b-tessera-e4m3-gridbook/` (the
E4M3 checkpoint), `qwen3-0.6b-tessera-e4m3-stock-twin/`,
`export_gridbook_e4m3.log`, `twin_vs_stock_e8.log`,
`qwen3-0.6b-tessera-mixed-gridbook/`, `qwen3-0.6b-tessera-mixed-stock-twin/`,
`mixed_plan.json`, `export_gridbook_mixed.log`,
`route_census_{e4m3,k2,mixed}_*_f419ef6+fp8.json`, `census_*.log`,
`serve_e8-*.out`, `serve_mixed-*.out`, `serve_k2-resident-v14.out`,
`serve_qwen_gridbook_*.log`, `kl_gridbook_*.json`, `kl_e8_*.json`,
`kl_mixed_*.json`, `kl_k2_v13-vs-v14.json`, `chain11.{sh,log}`,
`chain11_suite.log`, `chain12.{sh,log}`, `chain12_suite.log`.
`/home/rob/tessera-runs/stock/`: the stock arms. Dumps under
`/mnt/shared/tessera-kl/qwen_*.json.npz`.
