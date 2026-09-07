# Reference-prefix finalization, 2026-09-07

The bounded original-wire reference prefix completed under PB
`c098dbb9b4c21ccf2e00403f4b03b82d713b2de0b67e16049b27bb20ca34b787`
on Sparklina at source `78f9c763`: 4 CPUs, 64 GiB physical memory and a 48 GiB
GPU subset, native measurement cache unchanged. The run retained startup,
one native invocation and finalization. All missing full-roster invocations
remain explicit errors. Fixed resources and timings remain null.

Raw artifacts are under
`/mnt/shared/tessera-native376-resource/full-engine-reference-prefix-r2/resources/`.
The compact capture is 411,348,981 B (SHA-256
`943ed5f6b2a277210239f9c08618841ff28ad2f676755a043021acf05ecacbe0`);
its expanded canonical identity is
`fdd8b6235246d253282992521daa5c32cb284c62f32f98a22e8925579d3edbcf`.
The exact timestamp-revision join passed with 789,593 revised fields. The
FlashInfer native-owner rule failed its mapped-library identity check and
assigned no owners. This is a retained incomplete result, not memory closure.

Finalization recorded 9.704 s compact encoding/write and 211.862 s ledger
analysis. Logical retained history-row payload was 1,548,332,985 B, sharing
47,333,175 B of unique chunks (61,825,092 B of Python byte objects), plus
19,429,736 B of row-cache containers. The dictionary held 947 frame values
and 10,554,587 encoded bytes. These component counts exclude other Python
objects and allocator overhead. Worker high-water RSS was 7,613,714,432 B;
PB aggregate peak was 9,561,751,552 B, with no OOM.

PB wall time was 575.1 s and aggregate CPU time 588.4 s. Launcher/container
exit 0 and scope cleanup were checked; the exact container was already removed
when a subsequent stop/attach command arrived, so no stop mutation occurred.
CAS receipt `79065715f14ba5ea06cbf8a0c4a4359a5a2fbf6d84dcf13c80dac7c6f2b7df03`
and its payload were independently rehashed. Both-host Netdata, the deterministic
worker cProfile and the bounded read-only stack observation are retained.

The single stack landed in CUDA API name normalization. The completed profile
superseded that hypothesis: API argument analysis accounted for 3.945 s and
normalization for 1.287 s. Canonical codec traversal recorded 518.6 million
`parts` calls and 28.62 million JSON calls. The shared codec now groups flat
allocator/API fields and coordinate arrays in C JSON encoding while keeping
frame bytes separately shared. It preserves sorted-key canonical bytes,
integer/float/boolean/signed-zero distinctions and exact frame expansion.
No API witness, history field, native owner refusal or completeness check is
removed. Full resource capture remains deferred pending retained-capture replay
and the next bounded qualification.

Compile and 110 worker/resource/native-owner/codec tests passed through PB
`e3875b52fbce6e64c2e2914bed252a8c7efde5de3badef1b88f90b34bb0df0cd`
on x86 Torch 2.11 CPU, 6 workers, zero skipped or missing. Added cases cover
frame fields at every sorted-key position, flat arrays and nested frame fields.
Exit 0, cleanup and receipt/payload hashes were verified; receipt
`84d8025238392221d5d63843445c81d8c1030a8d6e394d20aeaf74f3133c334a`.

## Exact retained-prefix CPU replay

PB `9d2b2d5d0174b5603036baf28eafdf95198ab25759a6d55e8751d8eafe196c0d`
ran one isolated before/after analyzer pair on Sparky, 1 CPU/16 GiB with GPU
access disabled and native threads bounded to one. Each arm used a fresh
process and the same 411,348,981-byte raw prefix. Both reproduced the entire
persisted ledger exactly, including its canonical capture hash, history join,
all refusal messages and null resources. Profiled analysis fell from
205.822 s to 96.077 s. `parts` calls fell from 497.5 million to 94.7 million;
JSON calls fell from 25.30 million to 5.51 million. In this clean CPU profile,
canonical digest time fell from 125.333 s to 45.803 s and exact history
verification from 69.450 s to 39.171 s.

Independent process high-water observations were 2,821,906,432 B before and
2,818,056,192 B after; no material memory improvement is claimed. PB aggregate
peak was 3,257,720,832 B with no OOM. Exit 0, cleanup and CAS receipt/payload
hashes were checked; receipt
`572dd27d91cb14b9be15770370942896401139311729b6d5b43e9a402f4bfa4c`.
Evidence is retained under
`/mnt/shared/tessera-native376-resource/reference-prefix-analysis-replay-r1/`:
result SHA-256 `8828a79f9cef831a5a85d0b91757019d6defac3b3b285bdfc3957fbf2e6ddacd`,
before/after profiles and both-host Netdata (10 series, zero missing;
index `4d43eb4be9511f0d01f98a00149cde7158b816592e43f017f3131e218cb0cae8`).
This is one CPU analyzer replay pair, not a full-engine speedup measurement.

The retained prefix has 219,106 Torch history rows and 1,441,057 API events.
Most history exists at model load, before the first observed native invocation.
The 789,593 retained timestamp revisions remain a separate checkpoint-growth
cost; a complete-run size/time budget must account for them. The complete
capture and timing pass have not been restarted at this checkpoint.

## Conditional full-run capacity estimate and direct launch

PB CPU artifact inspection
`262595d29337ce7a4f59d680536737f5744ecdb796cb890ed77c38059babd104`
measured the parsed Python 3.12.3 component sizes. The largest prefix checkpoint
occupied 134,141,307 Python bytes and 27,609,131 canonical compact bytes.
Charging this observed maximum to all 165 planned checkpoints gives 22.13 GB
of checkpoint objects. Conservatively allowing three simultaneous checkpoint
representations gives 66.40 GB, with approximately 4.86 GB of capture wire
bytes plus model, collector and other state. The planning estimate was roughly
94 GB including a growth allowance; it is conditional on native history/API
and checkpoint growth, not a hard bound or a measured full-run peak.

The inspection artifact is
`/mnt/shared/tessera-native376-resource/reference-prefix-growth-audit-r1.json`,
SHA-256 `7bf6ad9b5f24b7ec431494e98e3e201ff4d691cba04d53e880417de10119d932`.
Exit 0, cleanup and receipt/payload hashes were checked; CAS receipt
`13b139406b818ecb2034fe3802b13fc221674a8b486388df43f8648ea58e0418`.

A fresh Lina observation found 124.06 GB available physical memory and no GPU
process or container. At 17:25:36 UTC, the full 76-invocation resource capture
started at source `3340b253`, directly under the user's expanded vLLM exemption.
It has no PB submission or GPU quota. The owned launcher retains the exact
container identity, source/image/calibration/cache identities, sampled cgroup
memory/CPU counters and host MemAvailable, and both-host Netdata. Its host CPU
affinity is unrestricted with native-library threads bounded to one; this
differs from the earlier PB CPU mask and does not establish an engine speedup.
The result is outstanding at this checkpoint. Artifacts and launch log are at
`/mnt/shared/tessera-native376-resource/full-engine-reference-r3/` and the sibling
`full-engine-reference-r3-launcher.log`. Complete resource closure and timings
remain unmeasured; the native-owner refusal is unchanged.

While the immutable remote `3340b253` full capture ran, PR #400 was reconciled
with master `a29dbec6`. Both architecture sections/provenance records were
retained. The collector resolution preserves the shared `current_context_id`
method; its only delta from the captured branch is the incoming explanatory
SONAME comment. Compilation plus all full-engine, native-resource and native-MoE
receipt CPU checks passed: 311 tests, zero skipped or missing, x86 Torch 2.11
CPU with 8 workers, PB
`ddd2dfb86f629873f3a2c9ce3a7506fb44ab4628036845c755657014d6588d72`.
Exit 0, cleanup and receipt/payload hashes were checked; receipt
`e91eb130bfcfa3ff0adcfd45a5d17a4071f9aa1dadd8339d5fb903b225c4b8b5`.
This integration check does not change the source identity of the running
experiment or claim GPU coverage for the CPU suite.
