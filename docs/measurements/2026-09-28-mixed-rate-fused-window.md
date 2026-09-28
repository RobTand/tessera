# The fused window kernel at every rate (contract v45), measured

Issue: RobTand/tessera#694 (item 2 of #690: the fused window kernel's run
table generalised to mixed rates). Tree: the `claude/tessera-fused-mixed-rate`
branch over `a5ffd2dc20` (master after #693). Boxes: sparklina (GB10, sm_121,
48 SMs) for every GPU row; dl380g10 for the CPU suite. Image X
(`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`, vLLM
0.28.1rc1.dev397, torch 2.13.0+cu130) for oracle, profile, NCU, bench, tests
and the stub-B census; the pinned `vllm/vllm-openai@sha256:61fc8a89...` (vLLM
0.28.0, the same torch) for the two `qwen3-0.6b-uniform-R1024` censuses. Every
row ran through PrismaBuild at priority -10; keys are in the receipts table at
the end.

## What the change is

The persistent fused window kernel (`serving/csrc/routed_fused_window.cu`;
the routed identity of #640, contract v42, and the dense identity of #692,
v43) always carried a run table -- a column block is a pair of runs `(r_lo,
n_lo, r_hi, n_hi)` and `decode_rows<FP8, R>` is instantiated for R in 1..8 --
but its word ring was sized for rate 4 and its shared-memory layout was fixed,
so both fused lanes published `column_rates = [4]` and every mixed-rate GLM
rung kept the compact adapter (routed) or the Triton window GEMM (dense).

Since v45 the word stages are sized per launch. `Params::slot_words` carries
`slot_words_for_rate(r) = 2r + 2 * (r odd)` words per column and 64-row half,
rounded up to a multiple of 4 for the larger rate of the pair
(`routed_fused.slot_words_for_pair`); a `Layout<MODE>` template places the
decode tables, the B and A stages, scales, descriptors and the claim counter
ahead of the word ring; the launch requests `smem_bytes(mode, slot) =
SMEM_FIXED[mode] + WORD_STAGES * 2 * BK * slot * 4` dynamically (91,216 B
fixed for the two-table gate/up modes, 58,448 B for down and dense). The two
extra words at an odd rate are the copy path: a column's words start 16-byte
aligned and a 64-row half at rate r is 8r bytes, so an odd rate's half is
8-byte aligned at odd half indices; the producer copies every half in 16-byte
`cp.async.cg` pieces, from the aligned word pair before it when the half is
misaligned (one 8-byte tail when it is not), and the decoder reads the half
from the slot's third word there. The decoder loads a word past a lane's
eight fields only where a field reaches into it (`u + 8R > 32`, `> 64`), so
no launch reads past a half.

The device decides the rates. sm_121 grants 101,376 B per block
(`cudaDevAttrMaxSharedMemoryPerBlockOptin`): the two-table gate/up launch
holds slot 8 (97,360 B; rates 1-4) and slot 12 (100,432 B; rates 5 and 6) and
not slot 16 (103,504 B; rates 7 and 8), while the one-table down and dense
launches hold every slot (70,736 B at 16). `ROUTED_LANE_RATES` is derived
from exactly that inequality, `(1, 2, 3, 4, 5, 6)`, and the dense identity,
which runs each role in its own one-table launch, reaches 1..8. Both fused
`native_extensions` entries therefore publish `lane.requires.column_rates =
[1..8]` and a new structure-scoped field `column_rates_routed_moe = [1..6]`;
`scheme.decide_lane_requirements` decides the latter only over a `routed_moe`
structure fact and refuses by name without one, the export plan gate and
`_lanes_a_rung_reaches` pass the cell's structure, and the contract validator
holds the field to an ascending subset of `column_rates`. A v44 reader
refuses the new field (fail closed). One correctness fix rode along: the
previous window word was loaded for the first 8-row group only, but a field's
14-bit window reaches 13 bits before it, so at rate 1 every group whose window
starts inside the half's first word (`8 * j * rate < 32`) read a stale word;
every such group now loads it.

What each rate rests on. The real-expert oracle and the profile cover the
E4M3 routed rungs a GLM-5.3-Flash layer can be cut at, rates 3, 4 and 5
(R832, R928, R960, R1024, R1088), and the dense oracle covers stub B's role
shapes at rates 3, 4 and 5 (832, 880, 960, 1024, 1088). Routed rates 1, 2
and 6 and dense rates 1, 2, 6, 7 and 8 rest on the GPU tests' synthetic-wire
cases, which run the same launches on the device against the derived bound
and the Triton lane (`test_routed_fused_window.py` `Q256_CASES` 256..1536;
`test_dense_fused_window.py` `Q256_CASES` 256..2048 including 1792 and 2048,
and `test_dense_forward_at_every_rate_is_within_the_derived_bound` at 2048),
not on a served artifact; no PACT-panel artifact exists at R768, R1152 or
R2048.

The E4M3 routed rungs this unit measures are R832 (rates 3/4, 192/64
columns per 256), R928 (3/4, 96/160), R960 (3/4, 64/192), R1024 (4, the #640
lane) and R1088 (4/5, 192/64). The criterion from #690 item 2 is that E4M3
routed stacks at R832-R1088 run fused within 1.5x of R1024's fused time at the
same M on the same layer and expert set.

## Oracle: routed E4M3 at five rungs

`experiments/routed_pair_oracle.py --mode oracle --rung <R>`, one PB row per
rung on sparklina. Layer 3 of GLM-5.3-Flash, 16 of 288 experts loaded, top-8
routing over the loaded set, M in {1, 3, 64, 512, 2048}. Every stage (gate,
up, activation, down) is compared against an fp64 reference with the derived
per-element bound of #693 (`gamma(K, 2^-23) Sigma` accumulation, the two E4M3
epilogue multiplies, the bf16 output); the end-to-end forward is compared
against the staged composition and against a repeat of itself; the launch
pair is read off the route's `emit_route` record after every apply.

PENDING: the after-fix oracle rows (R832, R928, R960, R1024, R1088).

The first-cut kernel (8-byte copies, see "Tried and rejected") passed the same
oracle at R832 and R928: stage max|d|/bound 0.30-0.54 (activation 0.98-0.99,
the shared bf16 rounding), end-to-end <= 0.0031, repeat diff 0, `fused` pair
recorded in every case (rows `0ad71616...` and `4404963a...`, `measure-20260928T175919Z/`).

## Oracle: dense, GLM role shapes

`experiments/dense_fused_oracle.py` on stub B's role shapes at the rungs it
carries (832, 880, 960, 1024, 1088), M in {1, 3, 64, 512, 2048}, TP1 and both
TP2 ranks, both families; the derived bound of #693 for the fused and Triton
lanes and the row-ulp criterion between them.

PENDING: the after-fix dense oracle row.

## GPU tests (image X)

`experiments/routed_fused_tests_action.sh` over `test_routed_fused_window.py`,
`test_lane_reachability.py`, `test_serving_contract.py`,
`test_contract_platform_axis.py`, `test_serving_export_gate.py`,
`test_dense_fused_window.py`, `test_dense_fused_census_cells.py`,
`test_glm_u1_census_cells.py`, `test_step4_route_qualification.py` and
`test_native_window_moe.py` on sparklina.

PENDING: row `3af798fa...` (tests-5, the final kernel).

## Profile: torch.profiler plus in-process power, eager, one routed layer

`experiments/routed_pair_oracle.py --mode profile --rung <R>`, one exclusive
PB row per rung (`--exclusive --gpu-capacity 0`, see "PrismaBuild findings"),
layer 3, all 288 experts, M in {1, 8, 64, 512, 2048}. Three legs per M:
**stock** (vLLM's `TritonExperts` on materialised bf16 weights), **compact**
(the Tessera adapter the routed cells attested before #640), **fused** (this
lane). Wall time is CUDA events over 200 forwards; kernel time is the
profiler's self device time per forward over 20 profiled iterations; power is
`nvmlDeviceGetPowerUsage` at 10 Hz over a 20 s window of back-to-back
forwards against the 140 W envelope, and the Netdata
`nvidia_smi.gpu_..._power_draw` series over the same window is recorded
beside it. Work per joule (forwards per joule) ranks the legs;
`gpu_utilization` is not read.

PENDING: the after-fix profile rows at R832, R928, R960, R1024, R1088 and the
1.5x table (fused kernel us at R / fused kernel us at R1024, per M).

## NCU

`--mode ncu` rows at R832 (slot 8, rate-3 and rate-4 decoders) and R1088
(slot 12, rate-4 and rate-5 decoders), plus the dense row, with the
`LaunchStats Occupancy SpeedOfLight MemoryWorkloadAnalysis
MemoryWorkloadAnalysis_Tables WarpStateStats SchedulerStats
InstructionStats` sections, before (first cut) and after the 16-byte copy fix.

PENDING: rows `d6c029b9...`, `ec67eee5...`, `2260466e...` (first cut) and
their after-fix counterparts.

## Route census: the widened predicate earns nothing new and re-earns everything

Three census rows on sparklina: the u1 stub B on image X (TP1, eager,
resident, the v38 serve settings, 21 modules) and `qwen3-0.6b-uniform-R1024`
on the pinned image in both residencies (112 modules,
`--require-decoder native_fused_window_dense`).

PENDING: the three receipts. Expected from the predicate: every routed stack
of stub B (E4M3 at q256 896, 928, 1024, 1088; BF16 at 1024) records the fused
pair in both phases, every dense module whose rows are a multiple of 128 and
columns a multiple of 32 records the fused dense identity at its rung (832,
880, 960, 1024, 1088), and the four E4M3 window cells are re-measured
(`remeasured_at_v45`) with unchanged `executes`.

## PACT bench: TP2 per-rank shapes

`tools/pact_tradeoff/bench_linears.py --ms 512,2048,8192,1,2,3,4,5,6,7,8
--warmup 10 --iters 30` on sparklina, before (the #693 tree,
`tessera-a5ffd2dc-master`) and after (this tree), both as exclusive rows.
The panel's mixed-rate E4M3 groups are the lane's cells; its q256 1024 groups
are the #640/#692 control and its NVFP4 groups the untouched control. The
panel has no R768, R1152 or R2048 artifact, so those rates are covered by the
oracle and the GPU tests only.

PENDING: `bench-before2` (`028e2b44...`) and `bench-after3`.

The before run is `bench-before-20260928T162411Z` (row `681306ec...`, 163 s,
no overlapping GPU row). An earlier after-run (`bench-after-20260928T175819Z`, row `8214a2ea...`) shared the
GPU with two oracle rows and read 1.6-3.85x slower at R1024 than the before
run; it is void, not a regression (see "PrismaBuild findings").

## Tried and rejected

- **8-byte copies for the odd-rate halves (the first cut).** A 64-row half at
  an odd rate is 8-byte aligned at odd half indices, and the first cut copied
  every odd-rate half in 8-byte `cp.async.ca` pieces. At R832 (rates 3/4)
  the fused kernel measured 1111 + 505 us (gate/up + down) per M = 1 forward
  against 829 us at R1024 -- 1.95x, outside the criterion -- and 0.55-0.67 of
  the stock path at every M, at 57-70 W (0.41-0.50 of the envelope) where the
  R1024 lane draws 72-81 W (`measure-20260928T175919Z/routed-R832-profile`, row `9fb4de06...`,
  a non-exclusive row that overlapped one CPU row). Replaced by 16-byte
  `cp.async.cg` pieces from the aligned word pair before the half (commit
  `72e96fb1dc`), which is the shipped kernel; its effect is the before/after
  pair in the profile and NCU sections.
- **Reading past a half.** The first cut's decoder loaded the next one or two
  words unconditionally, so a half at the ring's end could read the next
  column's words; at rate 6 (slot 12, 48 bytes per half exactly) that read
  was outside the slot. The loads are now predicated on a field reaching into
  the word, and rate 6 became reachable (the slot rule's two extra words are
  for the copy path, not the read).
- **Narrowing `column_rates_routed_moe` to the rates that meet the 1.5x
  criterion.** Refused: the field is the set the launch REACHES on the target
  (derived from the shared-memory inequality), not the set that is fast; a
  time criterion is reported, never encoded as a predicate.

## PrismaBuild findings

- PrismaBuild admits `--demand gpu=1` rows concurrently on one GB10 when
  memory allows, so two rows can share the GPU; `--measurement` is placement
  and attestation, not isolation. Every timing row of this unit (profile, NCU,
  bench) is submitted `--exclusive --gpu-capacity 0`. The #692 profile row
  (`f6cf7108...`) shared 5.6 of its 13.8 minutes with row `e74203c1...`, so
  part of that table is suspect; the mixed-rate profile at R1024 re-measures
  the same layer exclusively.
- Both Sparks drained for the `u4-A8-20260928T1809Z` TP2 MTP census window
  while this unit's rows were queued; the drain was legitimate and the rows
  waited.

## CPU suite (PrismaBuild, untagged -> dl380g10)

PENDING: the full-suite rows. Targeted rows so far: cpu5 (contract and
reachability files, both shards rc 0), cpu6 (3 failed / 125 passed -- three
stale test-side expectations naming rate 6 where the fixed kernel reaches
it; fixed in `e9e8fbe492`), cpu7 (`test_lane_reachability.py`,
`test_contract_platform_axis.py`, `test_serving_contract.py`: 90 passed / 1
skipped and 92 passed, rows `71717f76...`, `8eefa99e...`), cpu8
(`test_contract_platform_axis.py` with the `remeasured_at_v45` assertion: 21
passed, row `8dfb9688...`).

## Receipts

PENDING.
