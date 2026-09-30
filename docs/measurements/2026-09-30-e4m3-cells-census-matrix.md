# E4M3-instruction fused pairs: which cells a census can earn

Status: open. Written 2026-09-30 against `c4fc615002` (PR #746), where the
E4M3 instruction became the E4M3 family's default
(`routed_fused.E4M3_MMA_DEFAULT = "e4m3"`) while its two launch pairs stayed
in `scheme.EXPERIMENTAL_LAUNCHES`:

- `(tessera.routed_fused.FusedRoutedWindowMoE.__call__, native_routed_fused_window_e4m3mma)`
- `(tessera::fused_window_dense, native_fused_window_dense_e4m3mma)`

This note records why contract v47 cannot yet move them out, and the
censuses that would let it.

## The mechanism

Leaving `EXPERIMENTAL_LAUNCHES` is per pair. `contract._validate_cell_executes`
derives each cell's `executes` from `scheme.ROUTE_LAUNCHES`, narrowed by
structure, regime, residency and the lanes each rung reaches
(`contract._lanes_a_rung_reaches`). It has no image axis. So once a pair
leaves the set, the validator demands it in every cell whose rungs reach the
pair's lane, on every image.

The lane is `tessera_routed_fused_mma_e4m3`. Its `lane.requires` publishes
`column_rates` [1..8] and `column_rates_routed_moe` [1..8], so every rung of
every E4M3 cell reaches it. That is six cells:

| Cell | Image | Rungs (q256) |
|---|---|---|
| `tessera_e4m3_k1_dense_sm121_decode` | `vllm/vllm-openai@sha256:61fc8a89...` (pinned serve image) | 1024 |
| `tessera_e4m3_k1_dense_sm121_batch` | same | 1024 |
| `tessera_e4m3_k1_dense_sm121_decode_resident` | `spark-vllm-nccl230@sha256:f8dbe1a0...` | 832..1088 |
| `tessera_e4m3_k1_dense_sm121_batch_resident` | same | 832..1088 |
| `tessera_e4m3_k1_routed_moe_sm121_decode_resident` | same | 832..1088 |
| `tessera_e4m3_k1_routed_moe_sm121_batch_resident` | same | 832..1088 |

`census.cell_launch_agreement` keys cells on their image, so an attestation's
scope is its image digest.

## The receipt that exists

The GLM-5.3 A8S artifact
(`/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported`,
56 served E4M3 modules: 14 dense and 42 routed stacks, every one at q256
1024) was served TP2, resident, with `TESSERA_FUSED_E4M3_MMA=e4m3` on staged
source `18748d31` (`c4fc615002` minus the default flip). Run directory:
`/home/rob/tmp/claude-campaign-20260926/pact/u4/runs/u4-A8SE-20260930T1253Z/`.
Route traces (`TESSERA_ROUTE_TRACE`, per module, per M, per rank) land in
`route/tr3-rank{0,1}.json` (batch shapes) and `route/latency-rank{0,1}.json`
(decode shapes M 1..8 and batch shapes), collected to
`/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/A8SE-nightly-20260930/`.

It runs on `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`,
which is not the digest of any E4M3 cell. It covers rung 1024 only. It is a
route trace, not a `tools/tessera_route_census.py` receipt, so the replay
tests (`tests/test_glm_u1_census_cells.py`, `tests/test_dense_fused_census_cells.py`)
cannot read it as it stands. It therefore earns none of the six cells.

What it could earn, if the release image should carry cells: four new cells on
`5be13705`, E4M3 dense and routed_moe, decode and batch, resident, rung 1024,
provided both ranks' traces show the two E4M3-instruction decoders on all 56
modules in both regimes. That needs an image axis in the derivation (below) or
the six existing cells to be earned first.

## The censuses each cell needs

1. `tessera_e4m3_k1_dense_sm121_{decode,batch}` on the pinned serve image
   `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`:
   rerun the v43 census of `qwen3-0.6b-uniform-R1024`
   (`experiments/results/qwen3_0_6b_uniform_r1024_config.json`) with the
   E4M3 instruction as the dispatch (`c4fc615002`, or
   `TESSERA_FUSED_E4M3_MMA=e4m3`), TP 1, eager, once per residency
   (`TESSERA_SERVE_MODE=resident` and `=streamed`),
   `--require-decoder native_fused_window_dense_e4m3mma --expect-modules 112`.
   The receipts mirror `qwen3_0_6b_uniform_r1024_fused_{resident,streamed}_eager_census.json`.
2. `tessera_e4m3_k1_{dense,routed_moe}_sm121_{decode,batch}_resident` on
   `spark-vllm-nccl230@sha256:f8dbe1a0...`: rerun the v45 census of the u1
   stub B (`experiments/results/glm53_u1_stub_b_fused_mixed_config.json`,
   receipt `glm53_u1_stub_b_fused_mixed_tp1_eager_census.json`) with the E4M3
   instruction as the dispatch, TP 1, eager, resident. Stub B carries E4M3
   routed stacks at q256 896, 928, 1024 and 1088 and dense modules from 832 to
   1088, the rung spread these cells already cite.

Both are vLLM serves. Neither can run while a u4 TP2 window holds the Sparks.

## A design option, not built

`requires_serve_flags` already narrows the derivation by residency
(`TESSERA_SERVE_MODE`). A second serve-flag axis keyed on
`TESSERA_FUSED_E4M3_MMA` would let a cell state which library its receipt
served: cells on `5be13705` could carry `=e4m3` and derive the
E4M3-instruction pairs, while the six existing cells carry `=f16` and keep
deriving what their receipts saw. That is a change to `scheme.route_launches`
and to the cell schema, and it is the owner's call.

## Consequence of the default flip today

With `c4fc615002` the default serve dispatches the E4M3-instruction pairs.
On either cell image, `census.cell_launch_agreement` would raise a problem for those
modules: each executed a pair outside the covering cell's `executes`. Land the
censuses above before, or with, the default flip.
