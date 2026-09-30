# E4M3-instruction fused pairs: the censuses behind contract v47

Status: measured 2026-09-30. Contract v47 moves the two launch pairs of the
E4M3 family's own tensor-core instruction (library
`tessera_routed_fused_mma_e4m3`, `mma.sync.m16n8k32.e4m3.e4m3.f32`) out of
`scheme.EXPERIMENTAL_LAUNCHES`:

- `(tessera.routed_fused.FusedRoutedWindowMoE.__call__, native_routed_fused_window_e4m3mma)`
- `(tessera::fused_window_dense, native_fused_window_dense_e4m3mma)`

They have been the E4M3 family's default dispatch since `c4fc615002`
(`routed_fused.E4M3_MMA_DEFAULT = "e4m3"`).

## Why all six E4M3 cells move together

Leaving `EXPERIMENTAL_LAUNCHES` is per pair. `contract._validate_cell_executes`
derives each cell's `executes` from `scheme.ROUTE_LAUNCHES`, narrowed by
structure, regime, residency and the lanes each rung reaches
(`contract._lanes_a_rung_reaches`). It has no image axis. The library's
`lane.requires` publishes `column_rates` and `column_rates_routed_moe` 1..8, so
every rung of every E4M3 cell reaches it, and both pairs enter all six E4M3
cells at once:

| Cell | Image | Rungs (q256) | Receipt |
|---|---|---|---|
| `tessera_e4m3_k1_dense_sm121_decode` | `vllm/vllm-openai@sha256:61fc8a89...` (pinned serve image) | 1024 | qwen3-0.6b, both residencies |
| `tessera_e4m3_k1_dense_sm121_batch` | same | 1024 | same |
| `tessera_e4m3_k1_dense_sm121_decode_resident` | `spark-vllm-nccl230@sha256:f8dbe1a0...` | 832..1088 | u1 stub B |
| `tessera_e4m3_k1_dense_sm121_batch_resident` | same | 832..1088 | same |
| `tessera_e4m3_k1_routed_moe_sm121_decode_resident` | same | 832..1088 | same |
| `tessera_e4m3_k1_routed_moe_sm121_batch_resident` | same | 832..1088 | same |

`census.cell_launch_agreement` keys cells on their image, so each cell's image
was censused on the instruction. No cell rung changes.

## The censuses

All three ran directly with `experiments/routed_fused_census.sh` on sparky
(vLLM work is exempt from PrismaBuild), TP 1, eager, from a clean checkout of
`bc4b372bc6` (the contract v47 commit). The E4M3 instruction was the dispatch
by default; `TESSERA_FUSED_E4M3_MMA` was unset.

1. `qwen3-0.6b-uniform-R1024` on the pinned serve image, once per residency,
   `--require-decoder native_fused_window_dense_e4m3mma --expect-modules 112
   --no-manifest-lanes`. Both runs recorded all 112 modules on
   `native_fused_window_dense_e4m3mma` in both phases, verdict `served`,
   `problems: []`. Each phase joined 112 of 112 records to
   `tessera_e4m3_k1_dense_sm121_decode` (decode) and `..._batch` (prefill).
   Receipts:
   `experiments/results/qwen3_0_6b_uniform_r1024_e4m3mma_resident_eager_census.json`
   (sha256 `db2add28...`) and
   `experiments/results/qwen3_0_6b_uniform_r1024_e4m3mma_streamed_eager_census.json`
   (sha256 `c6bf18b6...`), beside the unchanged
   `qwen3_0_6b_uniform_r1024_config.json`. Replayed by
   `tests/test_dense_fused_census_cells.py`.
2. u1 stub B on the GLM serving image, resident, with the v45 serve settings
   (`--attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend triton
   --kernel-config '{"enable_flashinfer_autotune": false}' --trust-remote-code
   --gpu-memory-utilization 0.45 --kv-cache-memory-bytes 4294967296
   --max-model-len 4096 --expect-modules 21`) and `--require-lane
   tessera_routed_fused_mma_e4m3 --require-lane tessera_routed_fused_value`.
   Verdict `served`, `problems: []`, both required lanes engaged. In both
   phases:

   | Modules | Launch |
   |---|---|
   | Routed E4M3 (4 stacks) | `FusedRoutedWindowMoE.__call__` / `native_routed_fused_window_e4m3mma` |
   | Routed BF16 (1 stack) | `FusedRoutedWindowMoE.__call__` / `native_routed_fused_window_folded` |
   | Dense E4M3 (8) | `tessera::fused_window_dense` / `native_fused_window_dense_e4m3mma` |
   | Dense BF16 (8) | `tessera::fused_window_dense` / `native_fused_window_dense_folded` |

   Receipt: `experiments/results/glm53_u1_stub_b_e4m3mma_tp1_eager_census.json`
   (sha256 `eb4c6ee9...`) with `glm53_u1_stub_b_e4m3mma_config.json`, which is
   byte-identical to the v45 receipt's config. Replayed by
   `tests/test_glm_u1_census_cells.py`, which also checks the rungs the E4M3
   modules carried.

The v47 changelog prose was edited after these runs, to replace path
spellings the portability test refuses. The cells, the scheme and the
dispatch code are the ones the censuses ran on.

## The fail-before

The same qwen3-0.6b census, resident, ran first from `b2887ce1ac`, whose table
is still v46. It recorded all 112 modules on the E4M3-instruction dense pair in
both phases and the verdict was `REFUSED`, with 224 cell-agreement problems,
one per module per phase. Each named that pair as one the covering cell does
not publish. The run log is kept on the build box at
`/home/rob/tmp/claude-campaign-20260926/tmp/e4m3cells/failbefore-v46-20260930T135049Z/`.
`tests/test_dense_fused_census_cells.py::test_the_join_fails_on_the_table_before_v47`
replays the same failure as a mutation of the packaged table.

## Supporting evidence, not a cell

A TP2 resident serve of the GLM-5.3 A8S artifact with
`TESSERA_FUSED_E4M3_MMA=e4m3`, on staged source `18748d31` (`c4fc615002` minus
the default flip), ran on image
`spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`.
Its rank-1 route traces (`TESSERA_ROUTE_TRACE`) recorded all 56 E4M3 modules
(14 dense and 42 routed stacks, all q256 1024) on the E4M3-instruction pairs,
in the decode regime (M 1..8) and the batch regime. No module ran a
16-bit-library pair. Trace files:
`/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-A8SE-20260930T1253Z/route/tr3-rank1.json`
and `.../route/latency-rank1.json`. No cell names that image, and v47 adds
none.

## What stays

The 16-bit library's pairs stay attested beside the new ones:
`TESSERA_FUSED_E4M3_MMA=f16` still selects them. `scheme.EXPERIMENTAL_LAUNCHES`
is empty again.
