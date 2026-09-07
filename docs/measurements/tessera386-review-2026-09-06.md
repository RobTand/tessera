# PR 386 review and paired encoder evidence — 2026-09-06

This completes the timing comparison left open in the [original handoff](tessera385-batched-ldlq-2026-09-06.md). The measured E2M1_K2 eight-expert batch is **4.176× faster** than the pre-change encoder, with identical per-unit artifact bytes. BF16_K1 changes by only 1.009×; that is negligible. This report covers the B8 encoder workload below, not B16/B32, dense-layer throughput, a served model, or the PrismaQuant caller. Paired in-process evidence comprises full-width BF16 and explicitly narrower E2M1 profiles; the full-width before E2M1 capture exceeded its memory limit.

## Source, workload, and method

All evidence is retained at `/mnt/shared/tessera-measurements/tessera386-review-20260906/` (abbreviated `R` below). `comparison.json` contains source/input hashes, the cross-arm byte comparison verdict, wall/CPU/memory records, and sampled energy calculations. `timing-receipt-audit.json` independently verifies all four successful terminal exit codes, CAS payload hashes/sizes, canonical receipt hashes/action IDs, and immutable source-bundle hashes/sizes.

Before is pristine `f8b87ce4` with the same benchmark harness copied into the checkout. After uses the production tree at `637ae0ed` (unchanged through review/harness head `f08eec5c`). The harness SHA-256 is `38dbfd229917458cc1ee128400dd1a7c1929788ddaacc52ee0c53d423ae9410b`; each receipt records and rechecks its source and inputs before completing. Both before processes have package source digest `bdf210cb57c9a23005aca4df174df1f1af60b448cd094153c6c1534c90ff473f`, and both after processes have `081b47f5cbcd172c0709eb17114c4783166be0b4b51c0a7a90a696ce60de7128`.

The workload is the first eight captured LFM layer-18 expert `w1` units, each `[1792,2048]`, totaling 29,360,128 quantizable parameters. Both arms use the same captured routed-row Hessians and encoder arguments, with BF16_K1@1792 and E2M1_K2@896. Calibration is WikiText-2 raw train, 32 samples, sequence length 512, seed 0, 16,384 fit positions. Inputs live at `/mnt/shared/tessera-measurements/tessera385-2026-09-06/units/`: `units.json` SHA-256 `39a6c2fdbb53dbcf37c5b0468e796d1b7629783667872fca6bb3f7ed7dde0e7f`; `units.safetensors` SHA-256 `8a79b042d880006782b60e45cd4560bfe8f564f4ad5ff2f55999a8d6a99ff80f`.

PrismaBuild admitted the A1, B1, B2, A2 processes as isolated measurements on sparky (NVIDIA GB10, driver 595.84, Python 3.12.3, PyTorch 2.11.0+cu130, CUDA 13). Each reserved four CPUs with OMP/MKL/OpenBLAS threads bounded to four. A1/B1 reserved 64 GiB; B2/A2 reserved 16 GiB after observing RSS around 2.5 GiB and CUDA reservations below 2.4 GiB. These caps were nonbinding. Each family has one complete warmup and one measured pass per fresh process. The two repetitions per source are a repeatability check, not a confidence interval or a long-lived cache endurance test.

Commands use the admitted Python environment and `PYTHONPATH=src`:

```text
experiments/tessera385_bench.py --out-dir R/ARM --run-label ARM
  --arms unbatched|batched --expert-count 8 --batch-sizes 8
  --families BF16_K1@1792,E2M1_K2@896 --warmup-repeats 1 --repeats 1
```

Exact action manifests and execution logs are in the CAS request and terminal paths named by `timing-receipt-audit.json`.

## Paired timing and bytes

| Arm | PB action prefix | BF16_K1 seconds | E2M1_K2 seconds |
|---|---|---:|---:|
| A1, before | `21ee9d8a9a09` | 30.519 | 95.540 |
| B1, after B8 | `763e2dfdacd0` | 30.335 | 22.949 |
| B2, after B8 | `067d8d620d94` | 30.221 | 23.005 |
| A2, before | `8cacace094a7` | 30.600 | 96.359 |
| Before mean | | 30.560 | 95.950 |
| After mean | | 30.278 | 22.977 |
| Before / after | | 1.009× | 4.176× |

All 16 family/unit blobs are byte-identical across all four arms, including the pre-change implementation. The encoder fixture ID is unchanged: `03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519`. This establishes equal artifact bytes and hence unchanged artifact size for these inputs; it is not a new model-level KL, served quality, or bpp measurement.

Steady phase peak CUDA allocation grows from 0.641 to 2.150 GB for BF16 and 0.450 to 1.700 GB for E2M1 (decimal units). Process RSS remains around 2.0/2.5 GB. Batch compute chunks are bounded, but caller-supplied input lists span the whole call; the eight-entry plan cache bounds entries, not arbitrary bytes. These measurements follow only one warmup and do not qualify a heterogeneous long-lived workload's retained-plan memory.

## Power and whole-host evidence

Per-unit blob hashes are retained in each `ARM/results.json`. Every arm retains raw Netdata series for **both sparky and sparklina** in `ARM/netdata-both-hosts.json`: CPU, load, RAM, available memory, I/O, and GPU power. Netdata's GPU chart period is ten seconds, too coarse for a strong energy claim about a 23-second phase. For A2/B2, `broker-power-sparky.jsonl` additionally records the fleet broker's approximately one-second power samples; the integration requires complete attributed samples for the exact active action, no foreign processes, and bracketing of the full phase. There are 22–90 interior samples per phase, with maximum gap 1.078 seconds. A1/B1 are not covered by this observer. Both telemetry sources and integration scripts are retained under `R`. `host-summary.json` records measured-phase whole-host user+system CPU at 7.51–17.91% on sparky and 1.94–48.91% on sparklina; sparky I/O wait is 0.26–0.48%. Those are whole-box observations, including unrelated work on the other host, not process attribution.

| A2 → B2 | Mean GPU W | Estimated GPU J | Params / J before → after | Work / J ratio |
|---|---:|---:|---:|---:|
| BF16_K1 | 91.07 → 90.31 | 2786.91 → 2729.37 | 10,535 → 10,757 | 1.021× |
| E2M1_K2 | 26.84 → 40.15 | 2586.31 → 923.76 | 11,352 → 31,783 | 2.800× |

Energy is a sampled estimate for one paired repetition, not board-integrated or whole-system energy. E2M1 after is only 28.7% of the roughly 140 W envelope; the improvement does **not** establish GPU saturation. GPU utilization percentage is not used to diagnose saturation. CPU time closely tracks wall time in both arms; in-process profiles must resolve where that activity goes.

## Paired in-process profiles

The full before action `bf8cce9104ff` preserved its BF16 CUDA-only table, then hit the enforced 64 GiB memory limit during E2M1 capture (terminal exit 137, `memory_limit_oom`). It has no successful CAS receipt. Its failed terminal/logs and checkpointed table are retained under `R/profile-A`; the failure is not a successful full profile. The after action `8152ae15d2f0` completed with both tables, terminal exit 0, and independently verified CAS payload/canonical receipt hashes. Its full E2M1 table records 8,620,474 device events and 21.072 seconds total device time; there is no complete full-width before E2M1 table to compare it with. Both use eight full `[1792,2048]` units, one warmup, four bounded CPU threads, and measurement isolation.

| Full BF16 CUDA activity | Before | After B8 |
|---|---:|---:|
| `_step` launches | 3,670,016 | 3,670,016 |
| `_step` device seconds | 27.623 | 28.027 |
| `_traceback` launches | 2,048 | 256 |
| `_traceback` device seconds | 0.647 | 0.252 |
| Total recorded device seconds | 28.906 | 28.884 |
| `cudaStreamSynchronize` calls | 4,336 | 1,000 |
| Recorded synchronization CPU seconds | 28.272 | 26.548 |

The unchanged BF16 step-launch count and dominant step time explain the negligible timing delta: batching reduces traceback and synchronization calls, but does not reduce this window step workload. This is attribution from the paired profiles, not an inference from utilization.

To keep E2M1 profiling bounded after the full-capture OOM, actions `960b3260975a` / `c6eb1cb782e0` capture the same eight experts with only the leading 64 input columns and their matching Hessian principal submatrices. Both therefore profile `[1792,64]`, B8 versus eight unbatched units, one warmup, four bounded CPU threads, 24 GiB, with `--profile --only-profile --profile-experts 8 --profile-batch 8 --profile-input-columns 64`. Slicing happens before activation-source preparation. This measures a narrower representative operation sequence; it neither replaces nor rescales the full-width wall, energy, or byte results. Both bounded actions completed with terminal exit 0, verified CAS payload sizes/hashes and canonical receipt hashes, matching package/input identities, and identical sliced-input blobs. `R/profile-comparison.json` records the audit and table hashes. An intermediate 256-column pair (`232df844eb34` / `a41c64be712e`) was withdrawn after the completed full-head table revealed 8.62 million device events and roughly 50 GB aggregation memory: that new evidence made the 32 GiB intermediate reservation unsuitable. The before intermediate capture had started; the after never started. Both withdrawals and cleanup are retained as superseded evidence, not successful profiles.

| E2M1 CUDA activity, eight `[1792,64]` units | Before | After B8 |
|---|---:|---:|
| Total recorded device events | 2,226,919 | 419,471 |
| Total recorded device seconds | 2.935 | 0.648 |
| `cudaGraphLaunch` calls | 64 | 8 |
| `cudaStreamSynchronize` calls | 18,354 | 18,242 |
| Recorded synchronization CPU seconds | 2.277 | 0.441 |

On this narrower workload, batching reduces device events and graph submissions while leaving the synchronization call count almost unchanged. This supports the batching mechanism and preserves the separate host-sync work; these ratios are not substituted for the full-width ABBA timing delta. The profiler's full-head E2M1 aggregation also consumed substantial CPU time and roughly 50 GB RSS, so profile wall/RSS must not be confused with encoder timing/RSS.

Manifests are in `R/profile-campaign.json` and `R/profile-bounded-campaign.json`. CPU operator tracing is disabled; CUDA runtime events, including synchronization calls, are nevertheless present and compared directly. These are not full Python/aten CPU profiles. Profiler phase wall times include finalization overhead and are not speed measurements. Both-host Netdata is retained separately for profiler actions. The bounded harness change is `f08eec5c`; PB CPU compilation action `247c56b3f0d4` passed with its CAS payload and canonical receipt verified.

## Validation and review fixes

The full independently audited test evidence is `R/test-validation.{md,json}`. The initial eight-action impacted run is retained as **red: 3,630 passed, 2 failed, 25 skipped, 1 xfailed**, with 494 observed CUDA-allocating tests (a floor) and zero modules not collected. Four actions exited 0; three exited 4 at an absent-artifact gate; one exited 1 for two assertions. All eight effective source attestations agree, and successful CAS payloads and canonical receipt hashes were verified. PB partitioned the 184 impacted files through its own shard interface; each action ran four xdist workers with native threads bounded to one.

Repairs have separate, overlapping populations; they are not added to the broad count:

| Repair/check | Mode | PB prefix | Result |
|---|---|---|---|
| Shared-setting presence mismatch, before | CPU serial | `0e957aadcdce` | 6 failed |
| Same six regressions, after fix | CPU serial | `f71a876d5814` | 6 passed |
| Issue-reference snapshot refresh | CPU, 2 workers | `5abbb5e0649e` | 8 passed |
| Cold LUT refusal on pristine base | Strict CUDA serial | `c3b31450d312` | 1 failed, 1 allocation |
| Cold LUT refusal after fixture-order repair | Strict CUDA serial | `418d8417288a` | 1 passed, 1 allocation |
| Supplied artifact and PrismaQuant dependencies | CPU, 4 workers | `d432b9bcf10e` | 163 passed |
| Supplied KL instrument | CPU, 4 workers | `09061bc3806c` | 29 passed |
| Final report issue references and refreshed snapshot | CPU, 2 workers | `f0bdb06d33b6` | 3 passed |

GitHub’s `pure` check also passed on pushed source/harness head `f08eec5c`; its captured status is `R/github-ci-f08eec5c.json`. All successful targeted repairs above have zero skips and zero modules not collected. An earlier artifact repair submitted in strict CUDA mode (`36d18789e52b`) passed 163 tests but correctly exited 4 for zero CUDA allocations; it remains failed evidence. The CPU rerun is the valid check. The complete base LUT file passed 55 tests (`04d4ae07c5d9`) despite its cold case failing alone, demonstrating the pre-existing fixture-order dependence.

Original skips comprise 15 absent-evidence/input cases, three missing-vLLM cases, two requiring two CUDA devices, and five inapplicable format partitions/ranges. Supplied-input CPU repairs do not establish vLLM or two-device coverage. The available xfail record does not contain its reason. Worker skip counts are nonadditive; controller populations anchored to terminal/CAS logs are authoritative. No deduplicated aggregate-green claim is made.

Review fixed a production bug in sparse `per_unit` dictionaries: explicit shared settings could disagree silently with omitted settings. Presence is now compared by key before resolving the shared encoder fixture, with six demonstrated regressions. Two separate fixes refresh the issue snapshot and resolve the encoder fixture before a test temporarily changes LUT landing mode. Prose now states the plan cache's actual memory bound and the current uniform-control forest accounting. The still-permissive uniform-control assertion was filed as Tessera issue 388 because its live triage work belongs to the parent agent; no accounting assertion was changed here.

## Retained work and qualification boundary

PR2 host-sync work at `4e9de57`, worktree `/home/rob/tessera-runs/tessera385-pr2`, is retained unchanged and unmerged. Its production diff was reviewed: ordered device fp64 sums, grouped scalar readbacks, masked coupled sweeps, and speculative LUT scoring are plausible arithmetic-preserving changes. Speculative scoring can increase work after accepted updates. It needs its own byte comparison and before/after timing, profiles, and power evidence. None of the PR 386 measurements includes PR2.

This encoder change does not change the wire format, fixture identity, serving plugin contract, or serving admission. No new serving qualification or PrismaQuant end-to-end result is claimed. The original 32-expert and dense experiments remain historical evidence; B8 results do not certify those unmeasured after cases. Issue 385 remains open: its B32 ≥5× target on sparklina and its later host-sync/front-residency priorities are not qualified by this sparky B8 evidence.
