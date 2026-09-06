# Exact scalar search removes the encoder identity startup bottleneck

Tessera #364 reported 76.19 seconds of CPU behavioral identity work before a
real PrismaQuant 96-projection campaign's first anchor. An isolated profile
located the expensive operation: the default CHANNEL spread calculation for
BF16. Its nearest-value search allocated a roughly 2 GiB distance matrix at
each of 40 candidate spreads. The BF16 fixture paid this while constructing its
artifact's profile, even though its encoding uses an explicit spread.

`scale_channel._default_sigma` now searches the insertion point in the already
sorted scalar alphabet and compares its two neighbours. Those are the same
float64 subtractions and absolute distances the exhaustive minimum selected.
The Gaussian samples, 40 candidate spreads, relative-error reduction and
first-winning-spread rule are unchanged. There is no fixture omission, new
identity cache, GPU substitution or changed default.

## Controlled before and after

Both arms used PrismaBuild `--measurement`, one admitted physical performance
CPU (assigned affinity `[5]`) and 8 GiB on sparky, a GB10 host. GPU visibility
was disabled; native Torch/OMP/MKL/OpenBLAS threads were one. Python 3.12.3,
Torch 2.11.0+cu130, Linux 6.17.0-1032-nvidia, aarch64, glibc 2.39. Before was
base `828ba1f`; after changed only the nearest-distance expression in
`src/tessera/scale_channel.py`. The receipts preserve the actual parentless PB
snapshot IDs rather than calling them identical source trees.

The principal comparison profiles eight normal `encode_linear` calls in a
fresh process: E4M3, q256=1024, 16×128 CPU weights from the deterministic encoder
fixture, rolling the rows by each unit index. Imports and fixture-weight
construction occur before timing; the first encode computes the cold identity.
The profiler runs on the main thread, and Netdata is queried after timing.

| Measurement | Before | After |
|---|---:|---:|
| BF16 default-spread calculation | 80.796 s | 0.0833 s |
| Cold behavioral identity, profiler cumulative | 86.658 s | 5.892 s |
| First normal encode, including identity | 87.222 s | 6.452 s |
| Eight-unit batch, including startup | 91.185 s | 10.330 s |
| Remaining seven encodes | 3.963 s | 3.878 s |
| PB scope CPU time, complete action | 92.239 s | 11.425 s |
| PB scope peak memory | 4.379 GiB | 0.405 GiB |
| Process maximum RSS | 4.563 GiB | 0.597 GiB |
| Process system CPU time | 72.079 s | 0.143 s |

The cold identity is 14.7× faster; this bounded batch is 8.83× faster.
The nearly unchanged remaining seven encodes show that the improvement is
startup work. This is a CPU preparation measurement, **not** GPU encoding
throughput, a served-quality result or a rerun of the frozen 96-point campaign.
All times include cProfile overhead; there is one matched batch per arm.

The host view agrees with one busy CPU: Netdata's whole-host CPU activity
averaged 7.48% before and 6.79% after across the aligned measurement windows.
Its GPU power samples remained at 4 W in both windows. GPU power is an idle
observation here, not CPU package power or a whole-system energy measurement;
no work-per-joule improvement is claimed. Full `/proc/stat` endpoints, Netdata
samples and PB scope counters are retained with the artifacts.

## Exactness and validation

Both arms produced identity
`220ee0aaed5fd628f6fe92c02b08cbdf90b6e26b76313fa505a9b32fecbf973c`.
Every one of the 13 encoded fixture contribution hashes matched, as did all
eight complete normal artifact hashes. The observed default spread values for
E2M1, E2M1x2, E4M3 and BF16 also matched exactly. These are measured values,
not new constants or test expectations to maintain.

An initial identity-only profile independently measured 85.207 → 6.071 s,
with identical contributions. Its concurrent Python host sampler caused
cProfile to attribute waits across threads; that profile is retained as
bounded diagnostic evidence, and its individual nested function totals are
not the hotspot proof. The subsequent single-thread batch profiles above
resolve that instrumentation limitation.

The import-graph selector returned 180 files. PrismaBuild action
`c9ea2770123262f372857fe5f617515dd10bebb8e65fcfae1502efc0d13bb993` ran them on
sparklina CPU, 12 xdist workers with `--dist worksteal --durations 20`, 48 GiB,
native threads one: **3029 passed, 1 failed, 544 skipped, 0 uncollected**.
The single failure was the new `#364` architecture reference missing from the
older offline issue snapshot. All numerical, byte-baseline audit and identity
mutation/refusal witnesses passed. The snapshot was refreshed; the affected
file then passed **3/3, no skips or uncollected modules**, on dl380g10 CPU with
2 workers (action `da2f18a87f3f16cd2b7675127c4f6c945fcc2de43013d34f25b7c32c0ba3e333`,
exit zero, CAS verified). Its pristine-base comparison also passed 3/3 on
sparky's no-Torch interpreter; that interpreter reports 70 globally uncollected
Torch modules, none belonging to the three selected documentation tests.

The first run remains recorded as red rather than being rewritten as a green
whole run. Its skip reasons are preserved verbatim in the JSON receipt; they
include CUDA paths, missing box artifacts and unavailable vLLM. No CUDA surface
coverage is claimed. Combined integration is the coordinator's separate run.
No numerical behavioral bug is being introduced or repaired: the before/after
performance and exact-output comparison qualify this computation-preserving
optimization, using the existing mutation and refusal tests.

## Artifacts and reproduction

Artifacts live under `/mnt/shared/tessera-runs/identity-startup/`:
`before/` and `after/` each contain `profile.json`, `profile.pstats`, `batch.json`
and `batch.pstats`. `instruments/` preserves both exact Python measurement
programs. Each PB action seals its program in the command, so the command and
source can also be recovered from the immutable action/CAS input.
The adjacent JSON receipt records artifact hashes, source snapshots, complete
PB action keys, CAS payload verification and the exact output comparison.

| Action | PB action key |
|---|---|
| Initial before profile | `c8a3e326233193e8bf734dbc5e8b667b7c98fee6813105e3a74029c77da13bbc` |
| Initial after profile | `9321392ea6b9fe108002b3365c8d00bd15272cee81c2596cedc2b7dbfa1be6a5` |
| Principal before batch | `f5e9b816b80a4755ffdd5017f9bc1e25b6655504cb5d287f51f0b4a3e4ff165e` |
| Principal after batch | `561c1b1d3ecec4bbd84049f645fca8529713f74b529f39382cdf1a098bbde4e0` |

All four actions exited zero. Their terminal logs and actual CAS payload
hashes were checked; the files are not accepted merely because submissions
were acknowledged.
