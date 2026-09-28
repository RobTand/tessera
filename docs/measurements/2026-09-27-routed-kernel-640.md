# Tessera #640 — final investigation report; performance work blocked

## Disposition / decision for Claude

**Not fixed. Every family still misses both M512 targets.** No performance kernel
change was promoted. Grid, small-tile and 32-bit-decoder experiments were reverted.
The remaining source change in `src/tessera/window_gemm_grouped.py` is annotation-only.

**Claude decision requested:** authorize a new fused, warp-specialized routed-MoE
CUDA/CUTLASS path with explicitly new launch identities and the corresponding v39
cell work, or direct continued tuning within the current identities. The proposed
next path changes an attested execution identity; the brief requires escalation
before merge. I have **not** proved that every same-identity optimization is
exhausted. I have not implemented or renamed a fused kernel, changed a contract,
changed a default, loosened a bound, or run a serve.

No PR was opened or merged. **Merged SHAs: none.** Issue #640 must stay open.
The branch is `astra/tessera-640-routed-kernels`, based on `09d6559d7`.
The branch started with contract v38; origin/master subsequently gained v39 in
`52bc86ea1`. No contract update was borrowed into this measurement.

All 12 submitted PB actions are terminal and consumed: 9 successful, 3 failed
(the expected pre-fix regression, a wrapper-gate regression subsequently fixed,
and the decoder prototype's initial compilation failure). Successful CAS payload
hashes were verified. Failed actions have terminal/log evidence, not success CAS
receipts. There is no pending PB action or CI run from this work; no PR means no
PR CI was launched.

## Diagnosis (recorded before kernel edits in `DIAGNOSIS.md`)

- Baseline native M512 latency: E4M3 **162.867 ms**, BF16 **161.628 ms**, E2M1
  **109.284 ms**, against stock **34.224 / 70.215 / 18.448 ms**.
- On 48-SM GB10, grouped kernels use 256 threads/CTA and 169–197 registers/thread;
  achieved occupancy is ~16.4–16.6%. Native GPU power is only 45–53 W of 140 W.
- At M512, the grid reserves 64 M-blocks per expert although the mean expert load
  is 4096/288 = 14.22 routes. Gate/up/down launch **2,359,296 CTAs** in total,
  many of which only evaluate the inactive-block branch. Low useful occupancy
  is real, but a large grid is not useful work.
- Tensor activity is only ~4.65–5.60% for E4M3/BF16 and ~1.75–1.95% for E2M1.
  The capture's compute-memory proxy is ~13.47–29.20% of sustained peak; it is
  **not a reliable standalone GB10 DRAM-bandwidth measurement**.
- Native has three grouped GEMMs per forward. The actual stock trace has **two
  GEMMs**, not one CUDA kernel. Native grouped work accounts for ~98% of device
  time. Merely removing host launches cannot explain away the 90–129 ms gap.
  Exact host-launch overhead was not separately isolated.
- Existing code already reaches `tl.dot` for window FP8/BF16 and `tl.dot_scaled`
  for E2M1. Tensor-core existence is not tensor-core saturation.

### Interventions and what they disproved

1. **Device-side expert M loop:** removes inactive M-grid scheduling while
   retaining arithmetic, identities and baseline tiles. M512 improved only
   modestly (E4M3 157.583, BF16 149.829, E2M1 108.117 ms); all targets missed.
   This was an early screen: all nine oracle cases ran, but timing/torch/Netdata
   covered M512 only and **standalone NCU was not run for this intermediate**.
   It is not a complete promotion receipt.
2. **Smaller tiles:** combined that loop with experimental BM16/BN32, four
   warps, one stage; BK unchanged (64 window, 128 A4). Occupancy rose to
   ~25–33%, yet M512 worsened to **241.795 / 242.510 / 251.451 ms**.
   BF16 power rose to 76 W, with less work/J. Occupancy/power alone are not
   success criteria. These experimental constants were never proposed as a
   shipping heuristic or an autotuning policy.
3. **32-bit funnel-shift decoder:** preserved index bits and existing identities,
   leaving the original grid/tiles intact. The initial uint32-mask expression
   failed compilation (`Scalar 4294967295 is out of range for type int32`);
   the corrected run passed every oracle case. M512 was **164.594 / 160.000 /
   111.131 ms**. Window gate register use rose (E4M3 196, BF16 201), while down
   register counts fell (167, 163); occupancy stayed ~16.6%. No compelling
   performance improvement. E2M1 was an unchanged control in this trial.

The small-tile and corrected-decoder trials have complete nine-case
Torch+NCU+Netdata evidence. The performance target remains unmet; fusion and
warp-specialized kernel implementation are **not delivered**.

## Measurement contract and limitations

Timing: one isolated PB GB10 action at a time, eager TP1, GLM layer 3, all 288
experts, top-8, synthetic N(0, 0.5²), seed 604, M=1/64/512. Image:
`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Stock and native use the same loaded wires in each paired action. No served KL,
full-model throughput, PACT receipt, bpp change or new serving qualification is
claimed. The numerical oracle uses its original 16-expert subset and reductions,
not the all-288 timing population.

CUDA events give latency. Work/J is replay forwards/s divided by that window's
Netdata mean GPU watts, **not a hardware energy-counter integral**. Raw fractional
UTC bounds are in the window appendix and each `profiles.json`; Netdata queries
round bounds outward to whole seconds. Power returned only two coarse samples per
window; ~20-second NVML sampling is also retained as a cross-check (e.g. baseline
E4M3 M512 198 samples, mean 52.698 W vs Netdata 52.5 W). Do not overinterpret small
energy deltas. Utilization percentages were not used as evidence.

Baseline timing/torch/Netdata ran on sparklina; baseline NCU ran on sparky, the
same GB10 model, **not the same physical device**. Complete subsequent trials ran
on sparklina. NCU replay timings are not substituted for event timings.

Across native windows, mean box CPU ranges were baseline 9.25–10.62%, grid
9.94–10.37%, small tiles 8.72–14.15%, funnel 6.54–9.78%. Maximum reported CPU /
memory / IO PSI percentages across those windows were respectively:
**0.913 / 1.299 / 3.46**, **0.52 / 0 / 2.27**, **1.26 / 0 / 2.02**,
**0.19 / 4.26 / 4.360**. These are whole-box observations, not proof of absent
external load. `telemetry-summary.json` and the original series retain every leg.

## Before/after measurements

“Native” is the candidate for that table; “stock” is its contemporaneous control.
Watt fractions use the 140 W envelope. Missing intermediate measurements are
explicit, never filled with a different run.

### baseline-profile

| Family | M | Native ms | Native fwd/J | Native W (%140) | Stock ms | Stock fwd/J | Stock W |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4M3 | 1 | 2.898 | 5.2923 | 65.0 (46.4%) | 0.902 | 32.5895 | 34.0 |
| E4M3 | 64 | 117.303 | 0.1579 | 54.0 (38.6%) | 27.147 | 1.0811 | 34.0 |
| E4M3 | 512 | 162.867 | 0.1170 | 52.5 (37.5%) | 34.224 | 0.6806 | 43.0 |
| BF16 | 1 | 2.737 | 5.4477 | 67.0 (47.9%) | 2.137 | 15.9949 | 29.0 |
| BF16 | 64 | 123.642 | 0.1555 | 52.0 (37.1%) | 57.855 | 0.6178 | 28.0 |
| BF16 | 512 | 161.628 | 0.1167 | 53.0 (37.9%) | 70.215 | 0.4896 | 29.0 |
| E2M1 | 1 | 1.343 | 10.4300 | 71.0 (50.7%) | 0.559 | 34.4727 | 52.0 |
| E2M1 | 64 | 82.666 | 0.2603 | 46.5 (33.2%) | 14.821 | 1.2747 | 53.0 |
| E2M1 | 512 | 109.284 | 0.2035 | 45.0 (32.1%) | 18.448 | 1.0222 | 53.0 |

### grid-loop

| Family | M | Native ms | Native fwd/J | Native W (%140) | Stock ms | Stock fwd/J | Stock W |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4M3 | 1 | not measured | — | — | — | — | — |
| E4M3 | 64 | not measured | — | — | — | — | — |
| E4M3 | 512 | 157.583 | 0.1231 | 51.5 (36.8%) | 33.787 | 0.7188 | 41.0 |
| BF16 | 1 | not measured | — | — | — | — | — |
| BF16 | 64 | not measured | — | — | — | — | — |
| BF16 | 512 | 149.829 | 0.1258 | 53.0 (37.9%) | 72.889 | 0.4899 | 28.0 |
| E2M1 | 1 | not measured | — | — | — | — | — |
| E2M1 | 64 | not measured | — | — | — | — | — |
| E2M1 | 512 | 108.117 | 0.2370 | 39.0 (27.9%) | 18.474 | 1.0614 | 51.0 |

### small-tile

| Family | M | Native ms | Native fwd/J | Native W (%140) | Stock ms | Stock fwd/J | Stock W |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4M3 | 1 | 3.004 | 4.2695 | 77.5 (55.4%) | 0.909 | 32.0625 | 34.0 |
| E4M3 | 64 | 150.689 | 0.1086 | 61.0 (43.6%) | 27.304 | 1.0784 | 34.0 |
| E4M3 | 512 | 241.795 | 0.0678 | 61.0 (43.6%) | 34.139 | 0.6808 | 43.0 |
| BF16 | 1 | 3.304 | 3.6462 | 82.0 (58.6%) | 2.095 | 15.7360 | 30.0 |
| BF16 | 64 | 150.706 | 0.0873 | 76.0 (54.3%) | 56.428 | 0.6104 | 29.0 |
| BF16 | 512 | 242.510 | 0.0543 | 76.0 (54.3%) | 70.841 | 0.4712 | 30.0 |
| E2M1 | 1 | 1.986 | 8.5883 | 58.5 (41.8%) | 0.542 | 34.7846 | 53.0 |
| E2M1 | 64 | 161.754 | 0.1931 | 32.0 (22.9%) | 14.820 | 1.2750 | 53.0 |
| E2M1 | 512 | 251.451 | 0.1205 | 33.0 (23.6%) | 18.435 | 1.0204 | 53.0 |

### funnel32

| Family | M | Native ms | Native fwd/J | Native W (%140) | Stock ms | Stock fwd/J | Stock W |
|---|---:|---:|---:|---:|---:|---:|---:|
| E4M3 | 1 | 2.989 | 5.0181 | 66.5 (47.5%) | 0.902 | 30.6639 | 36.0 |
| E4M3 | 64 | 119.321 | 0.1457 | 57.5 (41.1%) | 27.049 | 1.0844 | 34.0 |
| E4M3 | 512 | 164.594 | 0.1075 | 56.5 (40.4%) | 34.067 | 0.6521 | 45.0 |
| BF16 | 1 | 2.756 | 5.1030 | 71.0 (50.7%) | 2.194 | 14.5659 | 31.0 |
| BF16 | 64 | 118.577 | 0.1441 | 58.5 (41.8%) | 56.659 | 0.5855 | 30.0 |
| BF16 | 512 | 160.000 | 0.1078 | 58.0 (41.4%) | 71.196 | 0.4530 | 31.0 |
| E2M1 | 1 | 1.349 | 9.9637 | 74.0 (52.9%) | 0.549 | 33.0955 | 55.0 |
| E2M1 | 64 | 83.491 | 0.2495 | 48.0 (34.3%) | 14.835 | 1.2426 | 54.5 |
| E2M1 | 512 | 111.131 | 0.1874 | 48.0 (34.3%) | 18.468 | 0.9848 | 55.0 |

## Nsight Compute after/before

Gate and down at M512 are shown below; the 27-kernel raw tables also include up
and both smaller M values. Tensor activity uses elapsed cycles, occupancy active
warps; compute-memory is the explicitly limited GB10 proxy described above.

| Run | Family | Projection, M512 | Registers/thread | Occupancy % | Tensor activity % | Compute-memory %peak |
|---|---|---|---:|---:|---:|---:|
| baseline-ncu | E4M3 | gate | 194.00 | 16.60 | 5.60 | 29.20 |
| baseline-ncu | E4M3 | down | 179.00 | 16.54 | 4.65 | 20.08 |
| baseline-ncu | BF16 | gate | 184.00 | 16.61 | 5.53 | 26.90 |
| baseline-ncu | BF16 | down | 169.00 | 16.53 | 4.71 | 17.43 |
| baseline-ncu | E2M1 | gate | 197.00 | 16.48 | 1.95 | 14.73 |
| baseline-ncu | E2M1 | down | 197.00 | 16.36 | 1.75 | 13.47 |
| small-tile | E4M3 | gate | 128.00 | 33.23 | 1.17 | 31.82 |
| small-tile | E4M3 | down | 146.00 | 24.94 | 1.07 | 21.01 |
| small-tile | BF16 | gate | 128.00 | 33.21 | 1.20 | 69.72 |
| small-tile | BF16 | down | 128.00 | 33.22 | 1.08 | 64.42 |
| small-tile | E2M1 | gate | 100.00 | 33.25 | 0.29 | 13.07 |
| small-tile | E2M1 | down | 100.00 | 33.28 | 0.27 | 12.29 |
| funnel32 | E4M3 | gate | 196.00 | 16.61 | 5.31 | 27.74 |
| funnel32 | E4M3 | down | 167.00 | 16.54 | 4.61 | 19.90 |
| funnel32 | BF16 | gate | 201.00 | 16.60 | 5.59 | 27.58 |
| funnel32 | BF16 | down | 163.00 | 16.53 | 4.71 | 17.44 |
| funnel32 | E2M1 | gate | 197.00 | 16.45 | 2.06 | 15.53 |
| funnel32 | E2M1 | down | 197.00 | 16.31 | 1.91 | 14.70 |

## Numerical gate

`oracle-comparison.json` is the machine-readable comparison to the supplied
historical `oracle.final.json`. Every completed trial and the final retained tree
passed **all 9 family/M cases with zero teacher-forced-stage violations and the
expected launch pairs**. Bounds, acceptance criteria and arithmetic were not
loosened. All teacher-forced-stage and end-to-end error statistics match the
historical run exactly.

The brief's blanket E2M1 bit-exact statement is broader than the supplied JSON:
end-to-end max absolute differences were already **0**, **3.0517578125e-05** and
**0.0009765625** at M1, M64 and M512. These remain unchanged; the originally
bit-exact M1 remains bit-exact. E2M1 M64 repeat-apply differences were already
nonzero (historical **1.4901161193847656e-08**); trial observations range from 0 to
**2.9802322387695312e-08**, consistent with the existing FP32 route reduction.
Consequently not every repeatability diagnostic is byte-identical to history.
The original E2M1 propagated end-to-end bound is explicitly non-discriminating;
its teacher-forced stages and launch-pair checks remain the original gate, not a
new claim about served quality.

Oracle receipts:
- `grid-loop/oracle.json`: PB `2b38229146441a9e016db88f24b4bf5422f5c3696ed64cca6c44864206856f29`.
- `small-tile/oracle.json`: PB `766b56f278fc14ac6c7513979f368aa1e4eda90d548836d5549556c04a864d37`.
- `funnel32/oracle.json`: PB `bc4128628c219f7e416976f1a5d7a9b5722ab7b6000c46da1f55144f0fca8b68`.
- **Final retained tree**, `final-validation/oracle.json`: PB
  `1ec6fd461956bf7d031d413538abd6e6cd7125729506aefad480b95951804984`,
  receipt SHA256 `589638b071def72ab3fff54c2809edc83e502db1052b6cd8259d706397af93be`,
  stdout CAS `d1132af35e87b5248fd1f2e62ace387a9e0fbf6de7bd35e08647f861592982ec`.

The failed first funnel attempt stopped before timing/NCU because the oracle
failed on a compilation error. It is recorded as failure, not a numerical pass.

## Tests, failures caught, and final review

- New fail-closed profile test shown failing first: PB
  `9e0535d14f6ad66de641db8ba76a524f7d932ca0f1eef5d2db34cf91640c67f5`,
  3 failed, CPU-only, 2 xdist workers. Exact failure:
  `AssertionError: caught profile failure must produce a failing process exit`
  / `assert 0 != 0` at `tests/test_routed_pair_profile_exit.py:45`.
  After fixing `run_profile`, PB
  `85f8e2d6b76ca0a26ace9b8b770a3c78846ef160bcd2d1a0352a76d5b5b64946`
  passed all 3, no skips.
- Impact selector (`impacted.json`) narrowed the harness commit to 90 files.
  PB `a3036b3d471c3904c339410c375770b786bdb2570b79ca6b19bda426b11e1415`
  ran with 8 work-stealing workers on dl380g10: **2274 passed, 1 failed,
  86 skipped; 0 uncollected modules; no CUDA device and 0 CUDA allocations**.
  This is a RED run, not a green full-suite claim. The submission client had
  earlier timed out; its rc75 JSON was superseded by the worker's terminal rc1
  and actual test log. Exact failure:
  `AssertionError: routed_pair_oracle.sh starts a container without gating its image`
  at `tests/test_runtime_image_pin.py:481`.
- Fixed that wrapper by calling the shared `runtime_image_require`, requiring
  an explicit immutable image and propagating its declaration. Only the failed
  file was run on pristine master `52bc86ea1`: PB
  `87122dc3c7d07b01c279a56854da322eee0d71f877b865ef8304d3825a8c876c`,
  **36 passed**, CPU, 2 workers, no skips/uncollected modules. On the branch,
  that file plus the new regression: PB
  `69c56356704696c7c7bc43cab28fc312809e0cb02ced4efa4e9ea2ca958401d9`,
  **39 passed**, same CPU population, no skips/uncollected modules.
  The other 89 files were not needlessly rerun; the failed file was verified.
- Final GPU action above ran the oracle, compileall for touched Python modules,
  and `tests/test_window_gemm_grouped.py` plus the new regression with
  `-n 2 --dist=worksteal --durations=20 --strict-cuda`: **14 passed, 0 failed,
  0 skipped, 0 uncollected modules; 11 tests allocated on CUDA**. Population:
  torch **2.13.0+cu130**, one **NVIDIA GB10**. This is targeted coverage, not
  the coordinator's full CUDA suite. `final-validation/final.surface.json`
  and `final-validation/stdout-summary.txt` are retained. Test dependencies
  were scoped to the admitted output directory (pytest 8.4.2, xdist 3.8.0).
- Active LSP review: the incorrect optional-quantizer annotations were fixed.
  Remaining host-LSP findings are image-only vLLM imports and Triton constexpr /
  launch-option modeling false positives; actual imports and JIT execution
  passed in the pinned image. No claim that host LSP is entirely clean.
- Self-review compared the copied oracle with its original: numeric functions
  and tolerances unchanged; only type assertions/initialization, telemetry,
  trace export, honoring `--m`, and fail-closed profile status were added.
  `git diff --check` passed. No wire, table layout, serving cell or kernel
  arithmetic change remains. No full suite was run; that remains the
  coordinator's integration responsibility.

### CPU skip population, verbatim

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 86 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
tessera surface: skip reasons, verbatim --
       43  encoder is a GPU job
        8  the encoder is a GPU job
        7  could not import 'prismaquant.native_moe_panel': No module named 'prismaquant'
        6  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        6  box artifact absent: the PrismaQuant checkout whose pricing this suite is pinned against -- /home/rob/prismaquant/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_DIR; documented default /home/rob/prismaquant)
        3  needs a CUDA device
        2  box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch -- /home/rob/pq-wt/tessera-continuous/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_WORKTREE; documented default /home/rob/pq-wt/tessera-continuous)
        2  needs CUDA
        2  the route prepares wires with CUDA packers
        1  box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)
        1  E2M1 publishes no reader range
        1  needs a device
        1  needs a CUDA device: the positive arm builds the extension
        1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/tsplugin/vllm-cache-fresh/torch_compile_cache/torch_aot_compile/15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
tessera surface: 0 test(s) allocated on the device
tessera surface: skipped for evidence this box does not hold --
        6  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        1  box artifact absent: checkpoints, served censuses and serve logs this box produced -- /home/rob/tessera-runs/tsplugin/vllm-cache-fresh/torch_compile_cache/torch_aot_compile/15957ad9e7a72f1d7539f792e4d4cee6e704e2e99696f07e909c209f30f5ddec is not on this box (set TESSERA_RUNS_DIR; documented default /home/rob/tessera-runs)
        1  box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)
        6  box artifact absent: the PrismaQuant checkout whose pricing this suite is pinned against -- /home/rob/prismaquant/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_DIR; documented default /home/rob/prismaquant)
        2  box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch -- /home/rob/pq-wt/tessera-continuous/prismaquant/tessera_formats.py is not on this box (set TESSERA_PRISMAQUANT_WORKTREE; documented default /home/rob/pq-wt/tessera-continuous)
tessera surface: population written to /home/rob/tmp/claude-campaign-20260926/pi/kernel-640/impacted.surface.shard-0.json sha256:33f015dd5e438f4a44a8b94adde4b4ff19b01e426d6fff779a2897c297e6a425
```

## Commits delivered locally / side findings

- `85e58ea94` — fix stale architecture prose claiming no BF16 routed cell exists.
- `df965c9b7` — preserve oracle/profile/NCU harness, CAS-returning PB action and
  fail-closed regression. Profile mode previously swallowed family failures.
- `6fdd2ccfc` — route the copied wrapper through the existing image verifier;
  caught by the selected tests, not left as a warning.
- `74e87fe7a` — annotation-only correction: prepared grouped quantizer accepts
  `None`, as its existing builders and runtime already support; remove redundant
  annotation quotes. Pre-fix Pyright error was
  `Argument of type "str | None" cannot be assigned to parameter "quantizer" of type "str"`.
  Final GPU tests exercise this same module. No new numerical test was needed
  for an annotation correction; the error and CUDA verification are recorded.

All side findings were fixed on the branch; none was silently deferred as a
product defect. Experimental patches remain in scratch (`small-tile.patch`,
`funnel32.patch`) and immutable PB snapshots, not the working tree. There is no
new runtime kernel identity hiding behind an old name.

## Next implementation decision

Recommended next experiment, subject to Claude approval: explicitly named fused
paired gate/up plus activation and a separately synchronized down phase, with
warp-specialized decode/MMA overlap and a hardware-derived scheduling/config set.
Use stock's measured two-GEMM structure as the comparator rather than pretending
it is one launch. This is a proposal, **not** a claim it reaches the target.
The oracle, all-nine paired timings, torch traces, NCU and windowed Netdata must
be rerun before any promotion. CUDA/CUTLASS does not relax that obligation.

No changes to PrismaQuant, its pin, PACT, MTP, #399, production defaults, or ship
criteria were made. The only delegated work was one read-only kernel scout on
`zai/glm-5.3-flash`; no model fallback or concurrent children.

## Complete PB receipt index

Successful entries have verified payload hashes and copied receipts in
`terminal-records/`; `receipts.json` preserves full metadata. Failures retain
their terminal records and original logs.

| Submission log | PB action key | Terminal | Success CAS |
|---|---|---|---|
| baseline-ncu.pb.log | `5236da907f3304c8840fdbeaa4aac5de8504330c121afa713528d0a39b05c0fd` | done | verified |
| baseline-profile.pb.log | `c83ba555953fae3f6b00bc3b178381ed3a70b3e441a37f02e4944c8f28025495` | done | verified |
| final-validation.pb.log | `1ec6fd461956bf7d031d413538abd6e6cd7125729506aefad480b95951804984` | done | verified |
| funnel32-retry.pb.log | `bc4128628c219f7e416976f1a5d7a9b5722ab7b6000c46da1f55144f0fca8b68` | done | verified |
| funnel32.pb.log | `71d0e057c9080eeb854d05bebcde3d2942a00687c6225460724e2c6489ff0a35` | failed | absent (failed action) |
| grid-loop.submit.log | `2b38229146441a9e016db88f24b4bf5422f5c3696ed64cca6c44864206856f29` | done | verified |
| impacted.pb.log | `a3036b3d471c3904c339410c375770b786bdb2570b79ca6b19bda426b11e1415` | failed | absent (failed action) |
| profile-exit-after.log | `85f8e2d6b76ca0a26ace9b8b770a3c78846ef160bcd2d1a0352a76d5b5b64946` | done | verified |
| profile-exit-before.log | `9e0535d14f6ad66de641db8ba76a524f7d932ca0f1eef5d2db34cf91640c67f5` | failed | absent (failed action) |
| small-tile.pb.log | `766b56f278fc14ac6c7513979f368aa1e4eda90d548836d5549556c04a864d37` | done | verified |
| wrapper-fixed.log | `69c56356704696c7c7bc43cab28fc312809e0cb02ced4efa4e9ea2ca958401d9` | done | verified |
| wrapper-pristine.log | `87122dc3c7d07b01c279a56854da322eee0d71f877b865ef8304d3825a8c876c` | done | verified |

## Exact measurement windows (UTC Unix seconds)

Raw Netdata series accompany every leg in the corresponding `profiles.json`.
The local files live under
`/home/rob/tmp/claude-campaign-20260926/pi/kernel-640/`.
Queries round these fractional bounds outward to whole seconds; the returned
power sample counts expose their limited resolution.

| Run | Family | M | Leg | UTC Unix start | UTC Unix end | Netdata power samples |
|---|---|---:|---|---:|---:|---:|
| baseline-profile | e4m3 | 1 | native | 1790471641.262311 | 1790471661.311672 | 1 |
| baseline-profile | e4m3 | 1 | stock | 1790471666.595367 | 1790471686.637883 | 2 |
| baseline-profile | e4m3 | 64 | native | 1790471697.805694 | 1790471719.740173 | 2 |
| baseline-profile | e4m3 | 64 | stock | 1790471726.001577 | 1790471746.895070 | 2 |
| baseline-profile | e4m3 | 512 | native | 1790471759.879286 | 1790471780.714065 | 2 |
| baseline-profile | e4m3 | 512 | stock | 1790471787.291283 | 1790471809.159145 | 2 |
| baseline-profile | bf16 | 1 | native | 1790471852.057831 | 1790471872.104740 | 1 |
| baseline-profile | bf16 | 1 | stock | 1790471877.648836 | 1790471897.655207 | 2 |
| baseline-profile | bf16 | 64 | native | 1790471908.861018 | 1790471930.996355 | 2 |
| baseline-profile | bf16 | 64 | stock | 1790471938.633771 | 1790471960.831496 | 2 |
| baseline-profile | bf16 | 512 | native | 1790471973.465043 | 1790471994.160137 | 1 |
| baseline-profile | bf16 | 512 | stock | 1790472002.356719 | 1790472024.892844 | 1 |
| baseline-profile | e2m1 | 1 | native | 1790472061.390984 | 1790472081.430706 | 1 |
| baseline-profile | e2m1 | 1 | stock | 1790472086.633850 | 1790472106.655808 | 2 |
| baseline-profile | e2m1 | 64 | native | 1790472115.873989 | 1790472137.024083 | 2 |
| baseline-profile | e2m1 | 64 | stock | 1790472142.737442 | 1790472163.577799 | 1 |
| baseline-profile | e2m1 | 512 | native | 1790472173.721451 | 1790472194.682942 | 1 |
| baseline-profile | e2m1 | 512 | stock | 1790472200.558482 | 1790472220.640249 | 1 |
| grid-loop | e4m3 | 512 | native | 1790472806.607429 | 1790472826.795774 | 2 |
| grid-loop | e4m3 | 512 | stock | 1790472833.370805 | 1790472855.085798 | 1 |
| grid-loop | bf16 | 512 | native | 1790472902.210632 | 1790472924.711035 | 1 |
| grid-loop | bf16 | 512 | stock | 1790472933.027629 | 1790472956.354886 | 2 |
| grid-loop | e2m1 | 512 | native | 1790472993.780230 | 1790473014.552626 | 1 |
| grid-loop | e2m1 | 512 | stock | 1790473020.425805 | 1790473040.525800 | 1 |
| small-tile | e4m3 | 1 | native | 1790473557.664539 | 1790473577.716529 | 2 |
| small-tile | e4m3 | 1 | stock | 1790473583.000746 | 1790473603.020493 | 1 |
| small-tile | e4m3 | 64 | native | 1790473615.308223 | 1790473637.791376 | 2 |
| small-tile | e4m3 | 64 | stock | 1790473644.063121 | 1790473665.009423 | 1 |
| small-tile | e4m3 | 512 | native | 1790473681.614809 | 1790473705.807393 | 1 |
| small-tile | e4m3 | 512 | stock | 1790473712.383779 | 1790473734.247272 | 1 |
| small-tile | bf16 | 1 | native | 1790473776.184996 | 1790473796.239223 | 2 |
| small-tile | bf16 | 1 | stock | 1790473801.773401 | 1790473821.837842 | 1 |
| small-tile | bf16 | 64 | native | 1790473833.923022 | 1790473856.387072 | 2 |
| small-tile | bf16 | 64 | stock | 1790473863.960691 | 1790473885.654424 | 1 |
| small-tile | bf16 | 512 | native | 1790473901.987037 | 1790473925.985908 | 2 |
| small-tile | bf16 | 512 | stock | 1790473934.209525 | 1790473956.848240 | 2 |
| small-tile | e2m1 | 1 | native | 1790473989.611971 | 1790474009.615422 | 2 |
| small-tile | e2m1 | 1 | stock | 1790474014.810661 | 1790474034.824943 | 1 |
| small-tile | e2m1 | 64 | native | 1790474047.778371 | 1790474068.493955 | 2 |
| small-tile | e2m1 | 64 | stock | 1790474074.206084 | 1790474095.042599 | 1 |
| small-tile | e2m1 | 512 | native | 1790474111.719182 | 1790474139.123401 | 2 |
| small-tile | e2m1 | 512 | stock | 1790474145.002671 | 1790474165.120082 | 1 |
| funnel32 | e4m3 | 1 | native | 1790475289.551918 | 1790475309.602503 | 2 |
| funnel32 | e4m3 | 1 | stock | 1790475314.884962 | 1790475334.886772 | 1 |
| funnel32 | e4m3 | 64 | native | 1790475345.735256 | 1790475367.691581 | 2 |
| funnel32 | e4m3 | 64 | stock | 1790475373.946380 | 1790475394.776781 | 1 |
| funnel32 | e4m3 | 512 | native | 1790475407.836693 | 1790475428.917135 | 2 |
| funnel32 | e4m3 | 512 | stock | 1790475435.487666 | 1790475457.296209 | 2 |
| funnel32 | bf16 | 1 | native | 1790475499.726715 | 1790475519.773022 | 2 |
| funnel32 | bf16 | 1 | stock | 1790475525.329719 | 1790475545.456226 | 1 |
| funnel32 | bf16 | 64 | native | 1790475556.064187 | 1790475578.017350 | 2 |
| funnel32 | bf16 | 64 | stock | 1790475585.606563 | 1790475607.468775 | 2 |
| funnel32 | bf16 | 512 | native | 1790475620.023696 | 1790475640.494005 | 1 |
| funnel32 | bf16 | 512 | stock | 1790475648.736469 | 1790475671.524609 | 2 |
| funnel32 | e2m1 | 1 | native | 1790475704.570511 | 1790475724.610800 | 1 |
| funnel32 | e2m1 | 1 | stock | 1790475729.815256 | 1790475749.821277 | 2 |
| funnel32 | e2m1 | 64 | native | 1790475759.073513 | 1790475780.449517 | 2 |
| funnel32 | e2m1 | 64 | stock | 1790475786.162939 | 1790475806.953265 | 2 |
| funnel32 | e2m1 | 512 | native | 1790475817.183503 | 1790475838.522951 | 2 |
| funnel32 | e2m1 | 512 | stock | 1790475844.402352 | 1790475864.488981 | 1 |


## Same-session correction: exact numerical-report equality

The sentence in Numerical gate claiming all end-to-end statistics match history
exactly is too broad. Teacher-forced-stage statistics match throughout; complete
end-to-end statistics match on the final retained tree and grid-loop trial.
Small-tile and funnel trials have 11 differing E2M1 M64 elements, versus 10 in
history; their maximum error and row-ULP statistics are unchanged. These
intermediate reports are not byte-identical. Every case still has zero bound
violations. The delivered RESULT.md contains the corrected wording. This
append-only correction preserves the dated measurement history.
