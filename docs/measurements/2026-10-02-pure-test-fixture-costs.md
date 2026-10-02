# Pure test fixture costs — #864

Two test-only changes remove repeated work: the impacted-selector module reuses
one analysis of its read-only checkout, and the grid-roster module constructs
grids when their parametrized cases run. Production graph construction,
numerics, kernels, recipes, runtime-contract bytes and serving gates are
unchanged. This records CPU test costs; it does not qualify a serving lane or
predict a full-CI speedup.

## Workload and instruments

The retained pure CI run `37052569471`, Python 3.12.14/pytest 9.1.1, reports
2749 passed/162 skipped in 596.86s. Its job took 624s. It has no per-test
durations or equivalent CPU profile. The diagnostic therefore starts from exact
master `5133f84625c4deb34fc5812b3132cb40611aac8d` and uses an owned, torch- and
numpy-absent Python 3.12.14 environment, pytest 9.1.1 and xdist 3.8.0.

All execution was admitted through published PrismaBuild, priority -10 and
native threads1, with CUDA hidden. DL380 has two Xeon Gold 6230 sockets,
40 physical cores/80 logical CPUs. PB preserved its assigned affinity. The full
diagnostic reserved CPU64/memory32GiB; matched causal runs reserved
CPU1/memory2GiB and received preferred physical cores. The two touched modules
were independent PB file shards, CPU1/memory2GiB each. Keeping each module in
one process avoids multiplying its shared checkout-analysis work.

An external, sealed stdlib cProfile pytest plugin records collection and test
calls in each pytest process. It writes outside the source checkout. Subprocess
children are excluded; cProfile elapsed times include waiting and are not CPU
seconds. PB's cgroup CPU accounting and aligned Netdata CPU/load/RAM/I/O/pressure
series on DL380, Sparky and Sparklina provide the separate process/host views.
No GPU energy or production-runtime claim is made.

## Diagnostic and negative results

The corrected directory diagnostic action
`0f4b8aa608ebe5f66d2de4dd95b53063ca0b8e4e2a116a2456fd921998d8b601`
ran `python -m pytest tests -q -p no:cacheprovider -p pure_cost_profile -n64
--dist worksteal --durations=80 --surface-json=...`. It completed in 537.27s:
2752 passed, 1 failed, 157 skipped, no collection errors, and 142 dependency
excluded modules. Its 1776-file source identity stayed verified at entry/end;
all 64 workers agreed. This is a cost diagnostic, not a green suite receipt.
xdist being installed allows five cases skipped by the serial CI environment.

The single failure is a measurement interaction: `full_engine_worker.py:507`
starts its own cProfile, which Python 3.12 refuses while another profiler is
active. Only that node was checked without the diagnostic profiler:
`2914b700042df0295bcbf8a7ff2342c499519ecc0562d13031336e2894034bc1`,
1 passed, exit0. No whole-suite repeat was used to clear it.

The largest call-time tail was repeated checkout analysis:
439.42s for the leaf-selection test, 334.33s for opaque paths, and about
114–122s for each other real-checkout graph query. Across 65 process profiles,
`import_graph` carried 1786.82s cumulative elapsed time. Grid-roster import
also fitted Lloyd grids 128 times, carrying 1831.01s cumulative elapsed time.
These overlapping process totals are not additive CPU time. PB measured
6318.53 cgroup CPU-seconds, peak resident scope memory 12.02GB, and a host
CPU mean/peak of 19.10%/86.99%: a busy startup followed by a narrow test tail.

Earlier attempts are retained as negative evidence:

- An eight-shard Python 3.14 exploratory fanout used explicit file buckets,
  overriding Tessera's directory `collect_ignore` exclusions. Its errors are
  not equivalent population evidence. This exposed PB #1458.
- PB `--profile sample` creates untracked `.prismabuild-profile/container-route`
  and `exit_status` before the child. Probe
  `99306bba26507428330070a65aedb211c8ccd0b6caf98816070ce8bb024314e7`
  confirms source identity correctly refuses that dirty checkout. PB #1459
  owns the instrumentation repair. No dirty-source gate was changed here.
- Interpreter provisioning required a current scoped uv installer; the
  unsuccessful installer invocations and withdrawn interpreter-presence probe
  remain in the evidence directory. Final measurements use exact 3.12.14.

## Matched causal checks

The first check calls the existing graph owner twice on the same unchanged
checkout and counts actual source reads. Before the fixture change it fails
`assert 2 <= 1` at `tests/test_impacted_tests.py:50`. The after run passes.
Both use the same test node, interpreter, profiler, native-thread bounds and
CPU/memory demand.

| Measure | Before | After |
|---|---:|---:|
| `import_graph` calls | 2 | 1 |
| `_read_python_source` calls | 1840 | 920 |
| Graph cumulative profiled elapsed seconds | 250.54 | 103.63 |
| Test call seconds | 251.22 | 104.05 |
| PB cgroup CPU-seconds | 267.43 | 113.63 |
| PB action elapsed seconds | 268.74 | 114.10 |
| DL380 mean host CPU busy (PB Netdata window) | 33.72% | 5.12% |

Before action:
`c47ebc9ad73415c23f703779bb2511f2fc32482d1e66519863653b71ef581bcd`.
After action:
`abaee352c60554323f14ae43abc0f993740bad01c1b8e8fac487aaad865ca415`.
The assigned physical cores were 29 and 1 respectively. The graph work is
halved. Host load differed, so the observed time delta is
not an isolated estimate of intrinsic speed or of hosted CI wall time.

The second check imports the grid-roster module in a child that refuses any
Lloyd fit during import. Before the change it fails at the eager
`lloyd_max_grid(16)` parametrization; after the change it passes. The existing
five buildability cases retain their IDs and exact constructor arguments:
E2M1, E4M3, E2M1x2, free-scalar-16 and free-tuple-1024. Fits still run when their
cases execute.

| Measure | Before | After |
|---|---:|---:|
| Parent collection `lloyd_max_grid` calls | 2 | 0 |
| Their cumulative profiled elapsed seconds | 13.96 | 0 |
| Pytest session seconds | 16.91 | 2.74 |
| PB cgroup CPU-seconds | 25.41 | 7.40 |
| DL380 mean host CPU busy (PB Netdata window) | 12.42% | 3.92% |

Before action:
`5695272bf1ff236c56d8ab6d76c8c14edd5ec54cdee7dd2b518f4f64ea5b6fb0`.
After action:
`4e968b5dc12e8ee087d51230140aa6ed21daff01543fe5e9684ef8767af55543`.
The assigned physical cores were 9 and 0 respectively. This eliminates
collection-time fits; its time delta carries the same host-load
and profiling limits.

## Validation and integrity

Complete touched modules, with graph-copy/guard isolation and temporary-repo
freshness regressions:

| PB action | Population | CAS receipt |
|---|---|---|
| `b848fc0b9a508a72e4690abe8f15833272b361eb943e624a6538fb020c8bc60f` | 107 passed | `3c786ae01004fc2d1b6285c810a53400d7324f3fc7fee5bdc66a8938236573d6` |
| `2b6e2b59bfd9d0a3be44342155f92a01109477154dcefb72ba26efb303dc8770` | 7 passed, 2 skipped | `a6b7bf15f90228bfdee61bc65352721ba4a37ab07b9925bdc914878df8f8a21a` |

All 116 collected nodes ran and reconciled, with no failures, collection
errors, duplicate nodes or missing files. Both skips state
`could not import 'torch': No module named 'torch'`; the two export/recipe
controls therefore remain unexercised in this pure environment.

Action `d3267f60aa7992727de0b287a5584eb0092ded2e98a0bf0f3a3752e9e779354e`
compiled both touched modules. Its initial selector invocation correctly
returned full for an undeclared generated closure stamp. Using the existing
`merge_suite.SOURCE_VERIFIER` declaration (`pbsnapshot.py verify`), action
`07d9f1e17cc68055c134fe9916ae5f979805faa4303701ee3641f250c04762d1`
narrows the exact `5133f846...HEAD` comparison to 303 files, with no full-forcing
path or unreadable source. That broad selection is retained for coordinator
acceptance; it was not rerun here. The touched-module receipt is bounded
validation, not proof of all 303 selected files or the CUDA surface.

Evidence root:
`/home/rob/tmp/astra-resume-20261002/t8_performance/tessera-test-speed-sol/`.
Raw 69 profiles and surface files:
`/mnt/shared/tessera-measurements/ts-pure-test-speed-20261002/`.
`profile-bindings.json` verifies every profile's length/hash and the sealed
plugin digest. `source-binding.json` verifies seven transported snapshot inputs
and 42 source-blob comparisons, including unchanged graph/numerical owners and
the packaged runtime contract. Successful result claims are independently
checked against their CAS receipts and hashed payloads; failed actions retain
hashed terminal logs rather than successful CAS receipts. Full worker
attestations were not independently audited.
