# Full-engine resource observer serialization cost, 2026-09-07

The all-native original-wire reference resource pass did not finish. PB action
`1806dd5f8e59faf953f27d3e0ff8e0f9e82cb2d4d4edaa59f0c0d9511eee8f9c`
ran on Sparklina with 4 CPUs, 64 GiB aggregate memory and a 48 GiB GPU subset.
The exact owned container was stopped after the authorized bounded profile;
its exit was 137, PB failed with return code 1, and scope cleanup completed.
No complete capture or fixed price was produced. The selected export proof was
`588d84dfdd48da1635c1b078fb3b60597c45a11987be9f975c31b19028585b1c`;
actual resolved KV capacity assertions passed, and all sealed native-cache
entries remained byte-identical.

Evidence is retained under
`/mnt/shared/tessera-native376-resource/full-engine-reference-r1/resources/`.
`profile-diagnosis.json` binds the target, exact profiler argv, CPU affinity,
capability and output hashes. A privileged debug exec, confined to the admitted
container and its assigned CPUs 5–8, sampled worker PID 127 at 49 Hz for 15 s:
734 samples, zero errors. Unprivileged root/uid1000 attempts were denied and
produced no profile. The target package/container configuration was unchanged.
The Python leaf `json.encoder.iterencode` accounted for 8.510 of 14.980 sampled
seconds (56.8%). History comparison, encoding helpers and hashing were also
visible. This is observer cost in that interval, not engine throughput.
Both-host Netdata contains 10 series, zero missing, index SHA-256
`411b6ca6d7da42c656279ecb61fe23273a1bd89a625566e88de8e89eea9d3c66`.

The shared recorder serialized and decoded the entire returned Torch snapshot,
then separately serialized its history again for the exact prefix hash at
every native boundary. It now retains Torch's fresh Python snapshot directly,
encodes the exact canonical history once, and skips per-field difference
construction for equal history rows. Complete raw capture serialization still
occurs at finish. Existing prefix hashes, revision values, owners and all
closure refusals are preserved. `serialized_snapshot_bytes` is now null;
`serialized_history_prefix_bytes` records the bytes actually encoded, without
estimating Python resident memory.

PB red `4411c804581ade4bfd13ba881a7d225568b146760b709cf8a893b8bef06a813d`
failed on the observed redundant full-snapshot encoding. Green
`085093ece42faa0537d2e0f8bc866b08fdd2d711d38d290bff84352f4aaf6267`
passed all 39 targeted pure-CPU resource tests, zero skips; this interpreter has
no Torch and the repository reported 72 unrelated modules not collected.
Exit 0, cleanup and CAS receipt/payload hashes were independently verified;
receipt `065d08aa669a9ab6993cf125a75760c9d66a7f0a2127e0cd8ed1c9f0371f4fcf`.

A CPU replay used the actual R4 returned values (21 snapshots, 10,662 history
rows) and the old recorder from `81999058`. PB measurement
`1110db835f3f590c82440a97a9cc13e22d3d0d04cbca0bff1d5dc57ba5841f0b`
on Sparky reserved 1 CPU/8 GiB, disabled GPU access, and recorded cProfile for
one before/after pair. Profiled wall times were 26.268 s and 9.478 s. This single
pair diagnoses redundant CPU work; it is not an engine speedup estimate.
Checkpoint and final-snapshot canonical hashes matched exactly between arms,
and replaying the full ledger produced an exactly equal result, including its
incomplete status and null fixed resources. Receipt/payload hashes, exit 0 and
cleanup were independently verified; CAS receipt
`0d9d9dc2cd59f0b06089aa81e8379f9da8b15c5fc26a090460ef1c4b8837bdd6`.

Replay evidence is retained under
`/mnt/shared/tessera-native376-resource/resource-observer-replay-r1/`:
`result.json` SHA-256 `99a319aa193e11c2b3da20cdae0cf760798b7d7500a90927de4bff418bdd1631`,
`before.pstats`, `after.pstats`, and both-host Netdata (10 series, zero missing;
index `c2721bdeb7ea57dddd1029cc478576cc98a530c64beff8f6ecfc9c10a5f1173e`).
The full reference after-profile, resource closure and timing partition remain
unmeasured at this checkpoint.
