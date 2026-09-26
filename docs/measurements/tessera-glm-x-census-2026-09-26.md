# The window routes on the GLM serving image: a TP1 eager census (2026-09-26)

**Result.** One route census of a GLM-5.3-Flash stub (one dense layer and three
MoE layers) served on the GLM serving image recorded **all nine** declared
Tessera modules as served, in **both** the decode and the batch regime, with
`verdict: served` and `problems: []`. Each module ran on its family's own
activation contract and on the launch pair below:

| family | structure | rungs (`q256`) | launch |
|---|---|---|---|
| `TESSERA_E4M3_K1` | dense | 832, 1024, 1088 | `tessera::window_gemm_dense` / `native_window_gemm` |
| `TESSERA_BF16_K1` | dense | 832, 1024, 1088 | `tessera::window_gemm_dense` / `native_window_gemm_folded` |
| `TESSERA_E4M3_K1` | routed_moe | 896 | `tessera.native_window_moe.NativeWindowMoE.__call__` / `native_window_moe_compact` |
| `TESSERA_BF16_K1` | routed_moe | 1024 | `tessera.native_window_moe.NativeWindowMoE.__call__` / `native_window_moe_compact_folded` |

Contract **v38** mints eight cells from this receipt,
`tessera_{e4m3_k1,bf16_k1}_{dense,routed_moe}_sm121_{decode,batch}_resident`.
Three of the four pairs leave `scheme.EXPERIMENTAL_LAUNCHES` in the same change;
the fourth, the epilogue dense GEMM, was already attested at v34. Issue
tessera#604 (first half).

**Nothing wider is attested.** The census ran eager only, at TP 1, resident
only. Every cell is grade `route_only`, because no KL arm was run, and smoke is
`not_recorded`. No timing or throughput is claimed; see section 4. The cells
cover only the rungs the stub carried.

---

## 1. What was served

| | |
|---|---|
| image | `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`. The launcher resolved this reference from docker's `RepoDigests` (issue #132). vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, torch `2.13.0+cu130`, python 3.12.3. |
| device | `NVIDIA GB10`, compute capability `[12, 1]`, platform token `sm_121` (sparky) |
| tessera | `1a5eb99e7`, the head of tessera#620, which lets the census name the engine's attention, KV cache and MoE backends (tessera#618) |
| artifact | `/mnt/shared/tessera-runs/moe/glm53-4layer-x-picks-20260925/merged`. It is a GLM-5.3-Flash stub: hidden 4096, MoE intermediate 2048, 288 routed experts, 4 layers. It holds 9 Tessera modules and 1,738 encoded units, 6,949,273,600 wire bytes. Every wire is `WINDOW` / `CHANNEL` / span 1 / `window_bits 14`, which is `served_recipe`'s wire at each rung and each structure. |
| `config.json` | committed as `experiments/results/glm53_x_stub_config.json`, sha256 `d93f81564431df80d9c6e8a89370a14ce43ec997db4febfd0663178645980401` (the receipt's `checkpoint_sidecars` records the same digest) |
| receipt | committed byte for byte as `experiments/results/glm53_x_stub_tp1_eager_census.json`, sha256 `d10738cd5692588a4afd4b8f3aeeb33924b99a8e144f84bb125624c660a34edc`. The run log `census.log` (sha256 `02feeb9a…`) is kept with a copy of the receipt beside the build box's run directory, not in this tree. |
| wall clock | 2026-09-26 00:56:41Z to 01:01:20Z, `elapsed_s` 268 |

### The serve configuration

The census ran through `tools/tessera_route_census.py` inside
`experiments/tessera_plugin_run.sh`, with these settings:

- Environment: `TESSERA_SERVE_MODE=resident` and `TESSERA_RESEARCH_GLM53_NOPE=1`.
- Engine: `--attention-backend CUSTOM`, `--kv-cache-dtype fp8_ds_mla`,
  `--moe-backend triton`, `--kernel-config '{"enable_flashinfer_autotune": false}'`
  and `--trust-remote-code`.
- Budget: eager, `--gpu-memory-utilization 0.45`,
  `--kv-cache-memory-bytes 4294967296` and `--max-model-len 4096`.
- Forwards: a 64-token prefill and a one-row decode.

The receipt records every engine setting above in its `engine_backends` block,
and records the two environment variables in its `env` block.

**Why the cells name only the residency flag.** The NoPE switch and the
attention and KV-cache settings select the model's attention backend. The MoE
backend setting selects vLLM's stock fused-MoE kernel. No Tessera route reads
any of them: the dense window GEMM is one launch with no fallback, and the
compact expert lane never consults vLLM's MoE backend oracle (`moe_route`
calls `select_fp8_moe_backend` only on the materialising branch, which this
build cannot take). A serve without these settings would therefore not move a
Tessera module to a different launch. The cells follow the routed E2M1 cells'
precedent and require only `TESSERA_SERVE_MODE=resident`. This document and
the receipt carry the rest.

This census did not measure whether a serve without the settings loads at all.

## 2. Per module

Both phases recorded the same symbol and decoder for every module (the last
column). The shapes are the census's own M1 decode and M64 batch forwards.

| module | route | structure | `q256` | launch | decode shape | batch shape | same pair both phases |
|---|---|---|---|---|---|---|---|
| `layers.0.mlp.down_proj` | TESSERA_FP8 | dense | 832 | `tessera::window_gemm_dense` / `native_window_gemm` | M1:N4096:K12288 | M64:N4096:K12288 | yes |
| `layers.0.mlp.gate_up_proj` | TESSERA_BF16 | dense | 1088 | `tessera::window_gemm_dense` / `native_window_gemm_folded` | M1:N24576:K4096 | M64:N24576:K4096 | yes |
| `layers.1.mlp.experts` | TESSERA_BF16 | routed_moe | 1024 | `NativeWindowMoE.__call__` / `native_window_moe_compact_folded` | M1:N4096:K4096 | M64:N4096:K4096 | yes |
| `layers.1.mlp.shared_experts.down_proj` | TESSERA_BF16 | dense | 1088 | `tessera::window_gemm_dense` / `native_window_gemm_folded` | M1:N4096:K2048 | M64:N4096:K2048 | yes |
| `layers.1.mlp.shared_experts.gate_up_proj` | TESSERA_FP8 | dense | 1024 | `tessera::window_gemm_dense` / `native_window_gemm` | M1:N4096:K4096 | M64:N4096:K4096 | yes |
| `layers.2.mlp.experts` | TESSERA_FP8 | routed_moe | 896 | `NativeWindowMoE.__call__` / `native_window_moe_compact` | M1:N4096:K4096 | M64:N4096:K4096 | yes |
| `layers.2.mlp.shared_experts.down_proj` | TESSERA_BF16 | dense | 1024 | `tessera::window_gemm_dense` / `native_window_gemm_folded` | M1:N4096:K2048 | M64:N4096:K2048 | yes |
| `layers.2.mlp.shared_experts.gate_up_proj` | TESSERA_BF16 | dense | 832 | `tessera::window_gemm_dense` / `native_window_gemm_folded` | M1:N4096:K4096 | M64:N4096:K4096 | yes |
| `layers.3.mlp.shared_experts.down_proj` | TESSERA_FP8 | dense | 1088 | `tessera::window_gemm_dense` / `native_window_gemm` | M1:N4096:K2048 | M64:N4096:K2048 | yes |

`decoder_coverage` reads `{native_window_gemm: 3, native_window_gemm_folded: 4,
native_window_moe_compact: 1, native_window_moe_compact_folded: 1}` in both
phases, over 9 modules each. `other_route_modules` is 0 in both phases.

## 3. The #104 void-census guard

This arm passed no `--require-decoder`, so the receipt's
`required_decoder_coverage` is empty and nothing in the run itself would have
refused a phase in which a decoder took zero modules. The guard is taken after
the fact instead, in two places:

- `decoder_coverage` above is non-zero for every one of the four decoders in
  both phases.
- `tests/test_glm_x_census_cells.py` replays the census tool's own
  `all_structure_agreement` over the committed receipt against the packaged
  table. It requires all 18 module-phase records to be covered by a cell, and
  it requires the join to agree. With the eight cells removed from the table,
  the same replay covers nothing.

The next census of these routes should pass `--require-decoder` for each
decoder it expects.

## 4. What this receipt is not

- **Not a timing measurement.** The census ran on sparky 11 s after the
  scheduler claimed an unrelated measurement row on the same box. No field in
  this receipt records timing or throughput, and no cell carries one. The
  contention cannot have moved a route: a route is a dispatch decision, not a
  rate.
- **Not compiled.** The GLM NoPE attention backend is eager-only on this image
  (tessera#508).
- **Not TP > 1.** The contract's tensor-parallel bounds do not move. A TP2
  census on this image is the next receipt.
- **Not streamed.** Routed stacks are resident-only, and the dense modules
  were served resident only.
- **Not quality.** No KL arm and no greedy smoke were run.

## 5. What the contract withdraws with it

Contract v38 withdraws `tessera_e4m3_k1_routed_moe_sm121_{decode,batch}_resident`
on `eugr/spark-vllm@sha256:0afec8d4…` at `q256 1024`. Those cells named
`(vllm.fused_moe.modular_kernel, torch_materialize_stock)`. This build cannot
make that launch for an FP8 expert stack: `moe_route.compact_window_lane`
answers True for FP8 whenever `scheme.parse_compact_tessera_expert_blob` is
defined, and this build defines it. The launch leaves `scheme.ROUTE_LAUNCHES`
in the same change.

This is an evidence downgrade, and the changelog states it as one. The
withdrawn batch cell carried a top-1024 KL lower bound
(`docs/measurements/tessera-lfm-campaign-2026-09-04.md`). Both withdrawn cells
carried the recorded greedy smoke
(`docs/measurements/moe-smoke-recorded-2026-09-05.md`). Both were measured on
the materialised FP8 arithmetic. The measurements stay in the tree and are not
retracted. What changes is the claim that this build serves the function they
measured.
