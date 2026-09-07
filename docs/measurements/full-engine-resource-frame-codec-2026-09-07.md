# Exact resource-history row reuse and frame compaction, 2026-09-07

The all-native reference R2 attempt remained too slow after removal of the
full-snapshot round trip. PB `4f5efbc246d362c3a100dccd2a6c936fb64f2732428f9c2b112ddf033f2a87ba`
on Sparklina started at 16:00:36 UTC. A bounded read-only stack observation at
16:07:52 found `unit:10:end`; the pre-stop observation at 16:09:48 found
`unit:17:begin`, with `serialized_history_prefix_bytes = 1555360175`. The latter
establishes 17 completed native invocations and approximately 41 completed
checkpoints by the recorder's ordering, not a persisted complete capture.
There are 76 planned native invocations. The exact owned container was stopped
under the user's fix/stop/rerun instruction; child exit 137, PB return 1 and
completed scope cleanup are retained. Aggregate PB peak was 10,121,695,232 B
within the 64 GiB reservation. No complete memory receipt or price was produced.
All sealed native cache entries remained unchanged.

R2 evidence is under
`/mnt/shared/tessera-native376-resource/full-engine-reference-r2/resources/`:
pre-stop stack/log/inspect, launcher result and both-host Netdata (10 series,
zero missing). The after-stack sample used the same admitted CPU mask 5–8,
15 s at 49 Hz, 734 samples and zero errors; its SHA-256 is
`c94abcf59012488fa414ecd81d660817ee5b7563bb4e59bcb0e0acf83c11d55f`.
Exact-prefix JSON encoding remained 11.163 of 14.980 sampled seconds. This is
observer CPU attribution, not a model throughput or power-efficiency claim.

## Representation and preserved contracts

`full_engine_snapshot_codec.py` extends the existing raw recorder. Exact frame
arrays share one value and canonical byte string. Unchanged history rows reuse
canonical chunks; revised/new rows are encoded again. Scalar chunks are combined
and hash updates are batched without changing their concatenated bytes. Integer,
float, boolean and signed-zero distinctions remain exact. Logical prefix size,
encoded/reused row counts, actual retained chunk/container sizes, process RSS
and high-water observations are explicit; none substitutes for PB aggregate peak.

The old finalization also encoded the entire expanded capture in one buffer,
then encoded it again for its digest. Capture v2 now writes a hash-bound frame
dictionary and explicit references. Expansion reproduces the exact legacy v1
JSON values; damaged/missing/ambiguous references are refused. The ledger's
`capture_sha256` is the streamed expanded v1 canonical digest, and the receipt's
file SHA binds the actual v2 bytes. Legacy v1 reads remain supported. Finalization
size, encoding/write time and ledger-analysis time are retained separately.
This is an observer encoding change, not a serving wire or allocator change.

`--qualify-first-native-prefix` retains startup and one native invocation,
closes collection, and allows the rest of the stock request to run unobserved.
Its workload identity declares this scope; all missing invocation errors and
incomplete admission remain visible. It preserves ordinary full-mode identities
and budgets. cProfile covers the first native observation and finalization;
startup checkpoints carry their own observer-time/memory measurements. This
bounded qualification cannot substitute for a complete capture.

## Replay evidence and negative result

Two pre-integration CPU checks failed as expected in PB `81c0b4e57dfd7ccf831b63cba83412e807887022f7a9942a943560092be396ca`:
row-encoding counters were absent and the compact schema was unsupported.
Initial codec/resource tests passed 60 tests. Compile plus resource, codec,
worker, timing and reference checks passed 123 tests on x86 Torch 2.11 CPU,
zero skipped or missing, in PB
`6ab242161b6d9a950fe1c182bfe5150f862d813c8c91c2f92debe55176af94e3`.
Exit 0, cleanup, CAS receipt and payload hashes were independently verified;
receipt `3dffcecf9de40aaf938d2e621bb92ba83d640216c1b98563c11b1c8bf4e7370c`.

The first row-codec replay (`ec41c2abd29d94b70771233fac064238e7321aa7ce0dbb5d1f01347a22475091`)
was a negative performance result: exact hashes and ledger parity passed, but
profiled observer time grew from 9.058 to 15.333 s. cProfile found 10.3 s in the
recursive Python frame-key builder and 6.4 million small hash updates. That
implementation was superseded before any GPU submission. Its artifacts remain
under `/mnt/shared/tessera-native376-resource/resource-observer-replay-r2/`.

The corrected replay, PB
`6404d9aa5e2c09929c921eb9462dc822a1436347592721eb4b40de2acbde054a`,
used the actual R4 returned values: 21 snapshots, 10,662 history rows, one
before/after cProfile pair on Sparky with 1 CPU/8 GiB and GPU access disabled.
Observer time was 9.603 s before and 4.846 s after. Serialization/write time was
1.483 s versus 0.279 s, with 265,463,951 versus 47,374,674 bytes. Checkpoint and
final-snapshot hashes and the complete ledger, including its incomplete result
and null fixed resources, were identical after exact expansion. The final
logical row payload was 134,485,888 B, sharing 11,152,962 B of unique chunks;
cache containers occupied 959,128 B (other object/allocator overhead is excluded
and process/PB memory is separate). These are a single CPU replay pair, not a
full-engine speedup estimate or a reference-model capacity qualification.

Exit 0, cleanup and CAS receipt/payload hashes were independently checked;
receipt `7a9373f38d85310f27e2b4fae42ce3bb3c805fa21d6818d334b3a4b3d036184f`.
Evidence is in `/mnt/shared/tessera-native376-resource/resource-observer-replay-r3/`:
`result.json` SHA-256 `ee6ea46eaa283918afc4ecb19601d94e25172a75c9a3720080d04857f85f61fd`,
before/after observer and serialization profiles, and both-host Netdata
(10 series, zero missing; index SHA-256
`554a85f08dcedf8d324b9abf4f35e8b64f2b474ef721d383144cd2ba07081bfb`).
Reference-specific prefix growth, finalization size/time and aggregate peak must
be checked before another full capture; they are unmeasured at this checkpoint.

After adding deterministic cProfile output to the prefix path, PB
`f919e3b6fcf9b3f709ba2027490023880d199c6afd7aa7caf0c2697e53d7f6cc`
compiled the worker/launcher and passed all 34 worker tests on x86 Torch 2.11
CPU, zero skipped or missing. Exit, cleanup and CAS receipt/payload hashes were
verified; receipt `14acc588fd44d89465c4f7087c97b5a289083bc5a754c592415ae8002da9227f`.

## Bounded reference prefix R1: retained failure, 16:49 UTC

PB `0d7badee5f0ef4cc84770cff0216b82757ee9314d60964ea6574a2dc1edb5d5a`
ran on Sparklina with 4 CPUs, 64 GiB physical memory and a 48 GiB GPU subset.
The first native invocation completed, but the required native ownership
library identity failed validation before recorder finalization. The worker
therefore produced no raw capture. The retained profile and both-host Netdata
(10 series, zero missing) are under
`/mnt/shared/tessera-native376-resource/full-engine-reference-prefix-r1/resources/`.
The launcher and container exited 1; PB cleanup was complete and the exact
container was independently confirmed removed. PB aggregate peak was
5,662,146,560 B with no OOM. These observations do not qualify finalization.

The worker now retains invalid native ownership evidence and an explicit
capture error, then finalizes. The ledger revalidates the unchanged evidence
before assigning any native ownership and still refuses it. This does not
accept an absent or changed library. Two CPU regressions reproduced the lost
capture before the fix in PB
`6ee7ccc264eec0d2abc1cd3ccb9409eef368178334ee3d5d21a35cc0184fdd39`.
Compile and 104 worker/resource/native-owner/codec tests passed after the fix in
PB `c91dd62167486c0bc49092d229850e042c7886ec92b6c0bea45f3e40df4c8ce3`
on x86 Torch 2.11 CPU, 6 workers, zero skipped or missing. Exit 0, cleanup and
CAS receipt/payload hashes were checked; receipt
`6203c931a93f99aacd7a902e327f29a1b0c49de4d71e7247162cf0657f1e7105`.
