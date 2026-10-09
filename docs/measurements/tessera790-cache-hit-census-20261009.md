# tessera#790 cache-hit census rows (2026-10-09)

These retained producer rows do not complete the census acceptance.
Rows A-C used snapshot `PYTHONPATH` with producer venv
`pq-b770d-producer-cb2a150f7-gb10-20261005`, which no longer exists.
They requested stack `model.language_model.layers.10.mlp.experts`, grid E4M3, q256 1024.
Source `/mnt/shared/models/GLM-5.3-Flash-BF16`: 120 shards,
642652070880 bytes, and 7 auxiliary files (24319052 bytes).
The cache receipts name a 300-second quiescence requirement.
Shared cache `/mnt/shared/tessera-measurements/ig790-source-digest-cache`.

## Rows

| Row | Action | Host | Elapsed | Command | Receipt |
|-----|--------|------|---------|---------|---------|
| A baseline | `6ad9ee90...` | sparklina | 1981 s | `tools-ig790-row.sh PY /tmp/ig790-rowA3` (no flag) | ABSENT, 120 files |
| B fill | `6c04a798...` | sparky | 1681 s | `tools-ig790-row.sh PY /tmp/ig790-rowB2 --source-digest-cache <shared>` | hashed 120, cached 0 |
| C hit | `9fd60a4f...` | sparklina | 60 s | `tools-ig790-row.sh PY /tmp/ig790-rowC --source-digest-cache <shared>` | cached 120, hashed 0, stat-bound |

Producer snapshot parents for A-C are `572539ad224c9ca3f36b9754051f73aed9ca99b4`.
Their raw snapshot commits differ. A common parent does not prove equivalent source.
Row A child read_bytes ends at 642784002048 (~598.6 GiB scope growth across
samples; scope total 643.1 GB). Row C scope reads 50.0 MB.
GPU power after each row: A 4.46 W, B 4.84 W, C 4.54 W, all 0 pct.
The action-duration difference between A and C is 1921.18 s.
This is not a measured producer-phase or census idle-time difference.
The shell sample waits affect both action durations.

## py-spy (`--profile sample:10`, speedscope CAS blobs)

| Row | Blob prefix | Total samples | Historical aggregate hash weight (seconds) |
|-----|-------------|---------------|--------------------------------------------|
| A | `26272af9` | 40893 | 3936.6 |
| B | `f0c8f383` | 41220 | 3933.2 |
| C | `b5c01651` | 30 | 1.6, including the imported `sha256_file` symbol |

The C receipt reports cached 120 and hashed 0.
[INFERENCE] Its brief cache-lookup profiles are consistent with no shard-body hash.
The producer code still reads headers, config, and auxiliary files.
The 50.0 MB scope count supports a reduced read volume, but does not establish per-file read coverage.

These weights aggregate process and thread profiles. They are not wall-clock durations.
Rows A and B report substantial sampler delay.
Row A reports up to 1375.92 s of delay near its end.
Do not use these profiles to calculate phase duration or saved GPU idle time.

## Host disk I/O (dl380g10 Netdata, retained history)

`storage_pool/shared` (the source filesystem) lives on dl380g10 itself
(`storage_pool/shared` on `/mnt/shared`; pool `storage_pool` raidz1 over
sdb/sdc/sdd/sde + NVMe cache). The GB10 workers read it over NFS, so the
serving reads appear as `nfsd.io` on this host and as pool-disk reads on
sdb-sde. Rows A and C ran on sparklina; row B ran on sparky. Both workers
serve from this host, so windows below bracket each row's claim-to-finish.

| Window (UTC) | `nfsd.io` read | Pool disks sdb+sdc+sdd+sde read | `system.io` reads |
|--------------|----------------|----------------------------------|-------------------|
| A 07:17:22-07:50:25 (1983 s) | 655779 MiB (625.4 GiB), mean 338637 KiB/s | 240844 MiB (235.2 GiB) | 610243 MiB |
| B 07:58:22-08:26:24 (1682 s) | 655905 MiB (625.5 GiB), mean 399314 KiB/s | 131590 MiB (128.5 GiB) | 669845 MiB |
| C 08:28:34-08:29:35 (61 s) | 53.9 MiB, mean 905 KiB/s | 303 MiB (0.3 GiB) | 424 MiB |

The reported physical disk total for B is 45.4 percent lower than A, not 80 percent.
Filesystem caches and other consumers can affect this total.
The retained host totals do not separate other consumers' physical ZFS reads.
Process scope counters do not provide that separation.
No quiet-pool fence or per-consumer disk trace exists for these historical windows.
This acceptance criterion remains unmet.

## Limits

- Per-file header/config/aux read split is derived from code path
  (`quantizable` header scan + `config.json` + 7 aux files ~24 MB inside the
  50.0 MB scope read), not from per-file counters.
- Row E read four tensors and printed hashes. It did not compare those hashes with expected values.
- No historical row proves caller byte integrity.
- Row A took 1981.87 s, which exceeds this attempt's 1800-second limit.
- No new full baseline can run without an approved duration exception.
- Historical projection files stayed in worker `/tmp`. This attempt recovered them separately, as described below.
- Historical GPU power is a box-window observation, not a census workload observation.

## Row D: PQ bridge cache-hit (2026-10-09T09:21:43Z, sparklina)

Action `16386001...` (take 3; takes 1-2 failed rc1, retained below) ran the
PQ `request_expert_projection` bridge from a shared copy of PQ worktree
`pq-790-bridge` at `b597fc363b` (reported PQ `origin/main`, with RobTand/prismaquant#2239;
copy sha `f43d75540517` matches the worktree file) through
`tools-ig790-pqbridge-row.sh` with caller python `pb-cpu` and
`PYTHONPATH=$PWD/src` (this snapshot, so the producer probe resolves
`tessera.producer_plan`). Same source, stack, and warm shared cache as row C.
Elapsed 20.5 s, child rc 0, `PQ-ELAPSED` 2.1 s for the bridge and producer together.
The shell's 20-second sample wait accounts for most of the remaining action time.
Producer receipt: cached 120, hashed 0,
stat-bound. Caller receipt: schema
`prismaquant.source_digest_cache_use.v1`, used true, reason names the
advertised option and the handed cache. GPU power after: 4.66 W, 0 pct.
py-spy `--profile sample:10` blob `778a8262...` (37 samples). Child
read_bytes sampled NA (child already reaped at the 20 s sample).

Takes 1-2: `76d8b64e...` rc1 (`No module named 'prismaquant'`; worker lacks
`/home/rob/pq-wt`, fixed with the shared copy) and `59dadb90...` rc1 in
0.5 s (`OUT: unbound variable` from a bad script edit, fixed and
syntax-checked). An x86 probe of the same script (`e6472c04...`, dl380g10,
rc0) returned the same receipts with `PQ-ELAPSED` 6.9 s.

## Takeover correction: retained artifacts

This attempt did not run a new GLM baseline or a new GPU job.
It corrected the launcher defects and recovered the original artifacts.
The issue remains open. The complete matched census measurement still lacks the required observations.

### Archive packet

Action `822529ffd810fa8eb4729bd7b772bb024c17e0c60e1f0060ec497243e63df2c1` ran a CPU-only archive extraction.
It verified each of the eight historical payload digests.
It retained exact sealed commands, environments, snapshot identities, CAS receipts, payload text, and resource observations.

- Packet: [/mnt/shared/tessera-measurements/ig790-a4/evidence.json](file:///mnt/shared/tessera-measurements/ig790-a4/evidence.json).
- Packet SHA256: `b1a5f29b16e41067b5538e3439daf40c0abe5dff0ddedfa5c3e9435952a31dd6`.
- Packet size: 606106 bytes.
- Archive action payload SHA256: `974f24af7492f89e59726b80ec9bacd88b54ed4aa35c498b33e8191505f50132`.

### Recovered source identities

Action `b459f97342d4244c48788588bb96ef8adeab95822ad97ae4dabc23d2b3889c51` recovered the five original projection files through SSH.
It did not read checkpoint tensor bodies or run a producer.
All five recovered `source` objects agree.
Each object contains 120 shard digests, 38770 tensor mappings, the config digest, and seven auxiliary digests.

- Packet: [/mnt/shared/tessera-measurements/ig790-a4/recovered-projections.json](file:///mnt/shared/tessera-measurements/ig790-a4/recovered-projections.json).
- Packet SHA256: `702e32d8be6c5a7cdcb2f1bf1c9726b0d82b7a8d1e828854f035e023d4ee7839`.
- Packet size: 25353520 bytes.
- Recovery action payload SHA256: `0c02025edc74bbbcb0d23630042523718ef58be57c3ca1f7ce3c6a2e81c66cb2`.
- Canonical source JSON SHA256: `8cb44968de5893362c81ddee238f649f4351933113aa856703f09c1b10ed680a`.
- Canonical encoding: `json.dumps(source, sort_keys=True, separators=(',', ':')).encode()`.
- Config SHA256: `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`.

These files were mutable worker artifacts until this recovery.
The historical actions did not bind their full projection digests at execution time.
The recovery therefore preserves evidence, but does not replace contemporaneous source custody.

| Row | Original worker path | Recovered file SHA256 |
|-----|----------------------|-----------------------|
| A | sparklina `/tmp/ig790-rowA3/projection.json` | `fbf3cbf474a7d7ce817d87fffce484f41efaebbea6bce730433fcd2efaa4bffe` |
| B | sparky `/tmp/ig790-rowB2/projection.json` | `cbecfbfa0dd7c6681539c584f6a6f865d638832304cfe428e159fc93c0c856d9` |
| C | sparklina `/tmp/ig790-rowC/projection.json` | `4ade055b2478fec7b799a35bd734caa0bb21c27afecb462a3cdaf3c337466fef` |
| D | sparklina `/tmp/ig790-pqbridge-hit3/pq-projection.json` | `70141b8e65fa38b7c7d56af2f497b46c26a2531bba4d925448f99cd1381e9515` |
| E | sparklina `/tmp/ig790-pqverify-hit2/pq-projection.json` | `70141b8e65fa38b7c7d56af2f497b46c26a2531bba4d925448f99cd1381e9515` |

The recovered packet retains complete producer receipts for B-E and caller receipts for D-E.
The C-E receipts retain each shard's writer identity and quiescence requirement.
The archive packet also retains the current cache entries.
Current cache entries are not historical projection exports.

### Exact historical commands and versions

The following commands are the sealed action commands, not new submissions.
The archive packet retains each command's full PrismaBuild environment and demand.

```bash
# A
bash tools-ig790-row.sh /home/rob/venvs/pq-b770d-producer-cb2a150f7-gb10-20261005/bin/python /tmp/ig790-rowA3
# B
bash tools-ig790-row.sh /home/rob/venvs/pq-b770d-producer-cb2a150f7-gb10-20261005/bin/python /tmp/ig790-rowB2 --source-digest-cache /mnt/shared/tessera-measurements/ig790-source-digest-cache
# C
bash tools-ig790-row.sh /home/rob/venvs/pq-b770d-producer-cb2a150f7-gb10-20261005/bin/python /tmp/ig790-rowC --source-digest-cache /mnt/shared/tessera-measurements/ig790-source-digest-cache
# D
bash tools-ig790-pqbridge-row.sh /home/rob/venvs/pb-cpu/bin/python /tmp/ig790-pqbridge-hit3
# E
bash tools-ig790-verify-row.sh /home/rob/venvs/pb-cpu/bin/python /tmp/ig790-pqverify-hit2
```

Rows A-C were producer-only rows. They did not exercise the PrismaQuant caller.
Rows D-E used the shared PrismaQuant package, with reported revision `b597fc363b`.
The bridge file SHA256 is `f43d75540517fb5a7cabad600cca9c244fa722678de57a23d61f3865a412c5b3`.
The caller authentication file SHA256 is `648ca4df4ea6dd7ece764acc94df04ce2a44f13705a666639e0d6a2d59a0d2b8`.
The scripts use the action snapshot's `src` through `PYTHONPATH`.
Thus the snapshot source, not the venv name alone, identifies the producer.
The current smoke uses `pb-cpu` as both caller and producer interpreter.

| Row | Raw snapshot commit | Snapshot input SHA256 |
|-----|---------------------|-----------------------|
| A | `abc2493d51ad7596b342852853b0b0c8ab4b6537` | `4e8acae8a47ddf96a2b97df8cf283c0759b9a2caf911bbc6239bb659cd388a59` |
| B | `2f97d013e99858199009bb5b2efcaeaf75f690bf` | `c7739954d78a8600c34292c078167207b48d9d41c4c340c2dd29a9293740be1e` |
| C | `21e69ff49d243c13155bd5b91c63cbb7dfbf49cb` | `6c308ef6e60aa1b0908d553a1ef32b52155f1476866f8260e2723643113ace62` |
| D | `723533fe42adab13bfa68eeae83722eda884f468` | `d9b893a2618c0aa4cf116dac0fe908861e3c2089a4245e3d4f27d10d79c2de0d` |
| E | `3d40018da608effe9831431a3a34ca1676d8f80e` | `8950cd6dcc4b8007441cfec43f15c7bc123119d9732049d6c7fd5f877e73d131` |

The four producer modules show no Git path changes between `572539ad224c9ca3f36b9754051f73aed9ca99b4` and review head `23e70cd926281fd2d00e8a5adea2a6b843fb91ef`.
Those modules are `producer_plan.py`, `serving_parts.py`, `source_digest_cache.py`, and `export_serving.py`.
This path comparison does not qualify whole-snapshot source equivalence.
No action-owner audit of the differing closure stamps ran in this attempt.

### Full action keys and scope observations

All five historical action receipts report exit status zero.
Rows D-E also print child status zero.
Both scripts formerly concealed failed child status. The corrected scripts now return that status.

| Row | Full action key | Action elapsed (s) | Scope `read_bytes` | Scope `rchar` | Mean box GPU power (W) |
|-----|-----------------|--------------------|--------------------|---------------|------------------------|
| A | `6ad9ee906f5e20729a009473710fba7c336adfda4b1bd6cab455949c72905966` | 1981.87 | 643124400128 | 643061945741 | 4.4965 |
| B | `6c04a798ae3f054b54852c0af128eecc676239ed639374a2bc3896a55787bdd2` | 1681.36 | 642652311552 | 643012048515 | 4.7984 |
| C | `9fd60a4f8cebde0cb48212e9ab5b983987e7eb94cf52fab0566f58c0402dd695` | 60.68 | 50032640 | 277117022 | 4.5909 |
| D | `163860015b529196f968aa321b7609c4791dbace2cb91892058534f2f3ddee41` | 20.52 | 7213056 | 417609656 | 4.6958 |
| E | `d4168b50d55661786beeb19162337b4d0b216f91538a4f27d693067c525ac539` | 80.81 | 8192 | 110036090 | 4.6018 |

These `/proc` counters cover each PrismaBuild resource scope, not only the producer.
They separate action reads from unobserved external process reads, but not physical ZFS traffic by consumer.
The counters have different semantics: `read_bytes` reports storage accounting, while `rchar` reports read-call bytes.
Memory-mapped tensor reads can appear in the child samples but not in the final scope read-call count.
Therefore E's 8192-byte scope count does not mean that its tensor reads used only 8192 bytes.

The scripts and resource receipts do not identify the producer phase's final timestamps.
The A-C action-duration difference cannot establish the required census idle-GPU time difference.
The historical 104 GiB demand is a reservation, not a current measured workload peak.
The archive packet retains the original cgroup memory peaks and separate box memory observations.

### Row E: read evidence, not integrity evidence

Row E started at `2026-10-09T09:44:44Z` on sparklina.
It reported producer cached 120, hashed 0, mode `stat-bound`, and caller use `true`.
It read 67108864 tensor bytes across four units.
Its child samples reported `read_bytes` of 729088, 17506304, and 40570880 at 20, 40, and 60 seconds.
Its original script printed truncated tensor hashes without a comparison.
This row does not prove byte integrity, and its four-unit scope does not cover the full projection.

### Full payload and profile digests

Each digest resolves below `/mnt/shared/prismabuild-fleet/cas/blobs/<first-two-characters>/<digest>`.
The archive packet retains these paths and their complete CAS receipts.

| Row | Payload SHA256 | py-spy profile SHA256 |
|-----|----------------|----------------------|
| A | `5d3d5f2b5c60c42d0ab8a9e69c30ecf9c1c64473f9c3d79ac59b598a021c4105` | `26272af9d01cf23fbf5c7ebe79b510c0afc91d1e45dd39ae2cd96ae830c04915` |
| B | `68e2056a4ebe6faf0809545237cd43a6184c692320c1781b9d5258fc33819d6f` | `f0c8f383c7d923fdc85f9d2367c2934730c97315750e060166a67b5800a4620a` |
| C | `049d101d739e5a1c0e71b6e66634fbaa01c0e8cfd1657dd8175dd9c7f04bb9c2` | `b5c01651ad00689c04c0c816cd37608482c35ae5e4969fcc1e97c3f9c54f043d` |
| D | `9481ce0a073e0e1007f3960822da36b5bdf9ac748d7431a058f4e5df628ca4ae` | `778a8262a8d3929fda6798d067283287e1634a4c13a39139a0026e5476436d87` |
| E | `f38b1808acb484cee3676c06e84c38e0cd371af6a22c7d1328e3fda6424cb2d6` | `4d128111819c9a7012be5c1e86877709543c5a4c6e613f6db59bb21ecd2b5f22` |

### ZFS host query retention

[Selected Netdata response fields](tessera790-netdata-selected-20261009.json) retain exact query arguments and the returned disk averages for A and C.
The file records the selected storage tiers, intervals, dimensions, and data flags.
The A query used tier 1, with 60-second samples. The C query used tier 0, with one-second samples.
These queries confirm that the historical disk data remain available.
They do not identify other pool consumers or reconstruct a quiet interval.
No traffic separation claim follows from subtraction of host totals.

### CPU controls and actual caller comparison

Both child-exit regression cases failed before the fix in action `85ff84c899d1e7a3ee739ede3dcb4c9387821e48f3e6747d72b90ce2d877f534`.
The failure line was `assert 0 == 1` after each script printed `ROW-END rc=1`.
The temporary changed-byte smoke failed before the fix in action `735b69fab58be315e29d1382fc99243e6bf351ae851958459cd5da46ad5a9ae9`.
Its failure line was `Failed: DID NOT RAISE RuntimeError`.
The fixture kept the tensor shapes and shard size unchanged.

The first corrective test action, `c19377d32fd044afdba428b4b769c86e268e2a3cf1e0e552b3506092f59a50b7`, failed with one failure and five passes.
The caller's default dev mode suppressed its own seal refusal; the explicit digest comparison still stopped the smoke with `AssertionError`.
The corrected launcher selects `PRISMAQUANT_DEV_MODE=0` for this integrity measurement.
The helper also raises a byte-mismatch error without dependence on Python assertions.
It uses `CaptureSourceAuthentication.recording` and `owner.safe_open` for every projected unit.
It preserves the caller's held descriptors and full-shard digest comparison.

Action `2ceb3596da23eb35a0fb1d09055e8db72219b583d0dc32e10b533706c78b2179` passed all seven selected checks.
These checks cover child status, changed-byte rejection, the actual small-checkpoint launcher, and document issue references.
Mode: x86, two xdist workers, `--dist worksteal`, one native math thread per worker, and `--durations=10`.
Population: 7 passed, 0 failed, 0 skipped, and 0 uncollected modules.
Verbatim device reason: `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`.
No test allocated on a CUDA device. No skip reasons exist.
Payload SHA256: `558496d83eddf173dbe2713f304c8c5c41e0b316da7b21de7637e28cea047e04`.

Standalone action `a62229a381361c5e285a171488e387f6a0af312667925a409f60d97fd316c153` ran the corrected launcher on a real fixture checkpoint.
It used the real producer subprocess and real expert geometry, with CUDA hidden.
It compared the expected and actual full-shard digest and read every one of the six projected units.
Both digest values were `f6e3c6143ff664a17b683db352244dceb2f02be4159e2631acc3df60693bc205`.
The receipt reported `fresh_descriptor_sha256`, 197304 shard bytes hashed, six payload reads, and 196608 tensor bytes.
The producer receipt reported hashed 1, cached 0, and recorded `false`, because the new fixture was not quiescent.
The caller receipt reported cache use `true`.
This smoke proves the corrected small-checkpoint path, not the real GLM census.
Payload SHA256: `ad276b9c89aef8f29df45de03c1db707fffd49de97111809a243edfed77b08a3`.

The archive extraction verified the prior fixture controls without new test execution:

| File | Action key | Passed | Payload SHA256 |
|------|------------|--------|----------------|
| `tests/test_tessera_producer_plan.py` | `3f80286a975f3dcec8286a73b455745e35064353b19c76a529ff96790782bba2` | 8 | `4adbb66a0e4d47f509320b0184b8dd023d03ea75952170a92d44d9a733987c7a` |
| `tests/test_source_digest_cache.py` | `972d8f3b41b61e9908a52abf533c0e5d88bc211dd3e8308d8bf6f587932082fc` | 16 | `7e048fc322dfee6d315f300c482674f4498a65704c259254f08d4e88281ea9ab` |
| `tests/test_producer_plan_public.py` | `1a932c449ef527ed56f73c4e41cea6cab75bf64fc486cad889c8550f6e77d97d` | 4 | `ae207a12eb1a152fdd1039e62656a1e8581e09d697e1850cf073596076a52277` |

Each historical control reports 0 skips, 0 uncollected modules, and 0 CUDA allocations.
Each reports `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`.
The producer and cache controls cover unchanged reuse, changed-shard rehash, restored-mtime rewrites, and absent-flag byte identity.
These controls do not substitute for the absent matched production census comparison.

### Remaining acceptance and authority

The baseline and hit must use the same real caller scope and host conditions.
They must retain phase timestamps, full source custody, independent caller comparisons, and separated physical pool traffic.
The historical observations do not supply those facts.
The hosted `pure` result at the corrected head has not yet been observed.
The issue cannot close from this report.

The baseline took 1981.87 seconds under an earlier 3600-second action limit.
The present brief limits every job to 1800 seconds and forbids a repeat of an unsplittable longer measurement.
A complete one-call source seal cannot resume across separate jobs without a different measurement.
Request an approved duration exception and a pool observation window before a new matched census pair.
Do not reinterpret the action-duration delta as a reduced acceptance criterion.

### Direct CAS links

- A: [payload](file:///mnt/shared/prismabuild-fleet/cas/blobs/5d/5d3d5f2b5c60c42d0ab8a9e69c30ecf9c1c64473f9c3d79ac59b598a021c4105), [profile](file:///mnt/shared/prismabuild-fleet/cas/blobs/26/26272af9d01cf23fbf5c7ebe79b510c0afc91d1e45dd39ae2cd96ae830c04915).
- B: [payload](file:///mnt/shared/prismabuild-fleet/cas/blobs/68/68e2056a4ebe6faf0809545237cd43a6184c692320c1781b9d5258fc33819d6f), [profile](file:///mnt/shared/prismabuild-fleet/cas/blobs/f0/f0c8f383c7d923fdc85f9d2367c2934730c97315750e060166a67b5800a4620a).
- C: [payload](file:///mnt/shared/prismabuild-fleet/cas/blobs/04/049d101d739e5a1c0e71b6e66634fbaa01c0e8cfd1657dd8175dd9c7f04bb9c2), [profile](file:///mnt/shared/prismabuild-fleet/cas/blobs/b5/b5c01651ad00689c04c0c816cd37608482c35ae5e4969fcc1e97c3f9c54f043d).
- D: [payload](file:///mnt/shared/prismabuild-fleet/cas/blobs/94/9481ce0a073e0e1007f3960822da36b5bdf9ac748d7431a058f4e5df628ca4ae), [profile](file:///mnt/shared/prismabuild-fleet/cas/blobs/77/778a8262a8d3929fda6798d067283287e1634a4c13a39139a0026e5476436d87).
- E: [payload](file:///mnt/shared/prismabuild-fleet/cas/blobs/f3/f38b1808acb484cee3676c06e84c38e0cd371af6a22c7d1328e3fda6424cb2d6), [profile](file:///mnt/shared/prismabuild-fleet/cas/blobs/4d/4d128111819c9a7012be5c1e86877709543c5a4c6e613f6db59bb21ecd2b5f22).

