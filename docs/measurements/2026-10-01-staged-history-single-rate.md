# Staged stream history: single-rate launches only

Refs #750.

## Change

`routed_fused_window.cu` stages each half's stream history word into shared
memory (`PREV_STAGED`, tessera#763) on single-rate launches only. The kernel
switch is `STAGE_PREV = PREV_STAGED && !TWO`. Two-run launches (`TWO`: R832,
R960, R1088 and every other pair of adjacent rates) go back to the register
path that master used before #763.

- The 768 B region stays laid out on every launch of the E4M3-instruction
  library, so `SMEM_FIXED_MMA8` and the launch sizes do not change.
  Occupancy is one 512-thread block per SM either way.
- Output is bitwise by construction: the two-run launches run the same
  register path that was bitwise before #763, and the single-rate launches
  do not change.
- No contract, rung, route or `executes` entry changes.

## Why

PrismaBuild row `8a5a23e0` timed #763 against its parent on the T8R release
artifact. Single-rate launches gained 7-15%. Every two-run launch lost 1-9%,
routed and dense alike, and NCU puts the loss on one full-block barrier. See
[the staged stream history](2026-09-30-staged-stream-history.md), section
"Two-run, dense and shared-expert launches: slower".

## Measurement

Pending: an `ab_arms.sh` row, master against this change, on the same
artifact and groups as `8a5a23e0`. Acceptance:

- R1024 routed and the R1024 dense and shared-expert groups: 1.00 in both
  passes.
- R832 and R1088 routed at M = 2048: at or below about 0.94 of master.
- Two-run launches at M = 1: at or below about 0.98 of master.
- Two-run dense and shared-expert groups: at or below about 0.98 of master.
- Every row bitwise in both passes.
