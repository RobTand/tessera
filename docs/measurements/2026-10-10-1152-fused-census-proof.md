# Fused census proof for R1152 and the priced pairs (tessera#1142)

Status: filed 2026-10-10. Owner: tessera#1142. Parent: tessera#690.
Scope: CPU proof only. It starts no GPU serve. It edits no cell.

## Pre-fix gap

R1152 was admitted but uncensused. The two base routed E4M3 cells
covered rungs 768 through 1279 through run tables `[3]`, `[3, 4]`,
`[4]` and `[4, 5]`, so `cell_covers_rung` admitted R1152 through
table `[4, 5]`. No census rung named R1152. The `rungs_q256` list of
each base cell ended at 1088. The `e4m3mma` executes entry of each
base cell ended at 1088. No committed receipt served R1152. The only
proof for table `[4, 5]` was the R1088 stack of stub B.

## Priced pairs

The pair list comes from the packaged rule, not from a roster. It
holds the two-rate run tables of `TESSERA_E4M3_K1` that the `e4m3mma`
routed lane reaches and that a base cell covers: `[3, 4]` and
`[4, 5]`. Each other rule pair has no covered rung and the export
gate refuses it. The allocator can price no pair outside this list
on this lane at contract v66.

## Launch pairs

`routed_fused.fused_launch_for_rates` maps each pair to its launch.
Values below hold at default build flags (`MMA8_A_RING` off):

| Pair | Slot | Gate/up stages | Gate/up smem | Down stages | Down smem |
|---|---|---|---|---|---|
| `[3, 4]` | 8 | 3 | 53,456 | 3 | 36,880 |
| `[4, 5]` | 12 | 3 | 56,528 | 3 | 39,952 |

Both launches fit the sm_121 per-block opt-in limit of 101,376 bytes.
R1152 takes the same launch as held R1088: table `[4, 5]`, slot 12,
identical shared-memory bytes in both launches.

## Stub B mapping

The held `e4m3mma` receipt
(`experiments/results/glm53_u1_stub_b_e4m3mma_tp1_eager_census.json`)
records four E4M3 routed modules as served on
`tessera.routed_fused.FusedRoutedWindowMoE.__call__` with decoder
`native_routed_fused_window_e4m3mma` in decode and in prefill. Those
modules carry tables `[3, 4]`, `[4]` and `[4, 5]`. Each priced pair
therefore maps to a fused launch pair that stub B served.

## What this note does not claim

It claims no new serve. The GPU gate for tessera#690
(`2026-10-09-690-gpu-gate.md`) stays closed, so a fresh R1152 serve
remains a follow-up. It moves no cell, no rung, no pin and no lane.
Contract v66 stays byte-identical. R1024 cells and R768 v66 cells
pass unchanged. E2M1 stays parked under tessera#1149.

## Receipts

- CPU proof: `tests/test_1152_fused_census_proof.py` (fails before
  the fix at collection with `ImportError: cannot import name
  'fused_launch_for_rates'`; passes after it).
- Unchanged scope: `tests/test_t8_routed_cells.py` on CPU.
