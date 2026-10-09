# tessera#790 cache-hit census rows (2026-10-09)

Matched GLM-5.3-Flash-BF16 rows through PrismaBuild. Producer venv
`pq-b770d-producer-cb2a150f7-gb10-20261005` (since removed) plus snapshot
`PYTHONPATH` to this checkout. Stack
`model.language_model.layers.10.mlp.experts`, grid E4M3, q256 1024.
Source `/mnt/shared/models/GLM-5.3-Flash-BF16`: 120 shards,
642652070880 bytes, 7 aux files (24319052 bytes), quiescent ~38 days.
Shared cache `/mnt/shared/tessera-measurements/ig790-source-digest-cache`.

## Rows

| Row | Action | Host | Elapsed | Command | Receipt |
|-----|--------|------|---------|---------|---------|
| A baseline | `6ad9ee90...` | sparklina | 1981 s | `tools-ig790-row.sh PY /tmp/ig790-rowA3` (no flag) | ABSENT, 120 files |
| B fill | `6c04a798...` | sparky | 1681 s | `tools-ig790-row.sh PY /tmp/ig790-rowB2 --source-digest-cache <shared>` | hashed 120, cached 0 |
| C hit | `9fd60a4f...` | sparklina | 60 s | `tools-ig790-row.sh PY /tmp/ig790-rowC --source-digest-cache <shared>` | cached 120, hashed 0, stat-bound |

Producer snapshot parents are all `572539ad`; snapshot commits differ
(`abc2493d`, `2f97d013`, `21e69ff4`) because each pbrun snapshot is parentless.
Row A child read_bytes ends at 642784002048 (~598.6 GiB scope growth across
samples; scope total 643.1 GB). Row C scope reads 50.0 MB.
GPU power after each row: A 4.46 W, B 4.84 W, C 4.54 W, all 0 pct.
Saved producer-phase time C vs A: 1921 s (~32 min).

## py-spy (`--profile sample:10`, speedscope CAS blobs)

| Row | Blob | Samples | `sha256*` stack weight | Top leaf |
|-----|------|---------|------------------------|----------|
| A | `26272af9...` | 4089.3 | 3936.6 (`sha256_file` serving_parts.py:113/114) | `sha256_file` 2432.8 |
| B | `f0c8f383...` | 4122.0 | 3933.2 | `sha256_file` 2463.3 |
| C | `b5c01651...` | 3.0 | 1.6 transient (`SourceDigestCache.sha256` import `sha256_file` symbol) | `open` 1.0 |

Row C holds no sustained shard-body hash: 8 `source-sha256_*` threads exist
but each contributes <= 3 samples of cache-lookup work (`fingerprint`
open+fstat, entry read). Header/config/aux reads stay: `quantizable`
(header scan), `config.json`, and 7 aux files total ~24 MB, inside the
50.0 MB scope read.

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

Pool-disk totals sit below `nfsd.io` because the pool's NVMe cache and ARC
absorb repeat reads; row B (second full read, ~40 min after A) served
~80 pct less disk than row A for identical bytes. The comparison that
matters is row C vs rows A/B: `nfsd.io` falls from ~625 GiB to 54 MiB
(4 orders), pool disks from 128-235 GiB to 0.3 GiB. No other pool consumer
was isolated: `system.io` covers all host I/O, and per-consumer splits were
not recorded, so residual traffic (row C `system.io` 424 MiB vs `nfsd.io`
54 MiB) is unattributed host I/O, not producer reads.

## Limits

- Rows call the producer CLI direct, not the PQ `request_expert_projection`
  bridge: no `source_digest_cache_use` receipt exists for these rows. The
  caller handoff (PQ #2239, merge `e9b65f45`, on PQ `origin/main`) was read,
  not executed: this checkout cannot run another repository's code.
- Per-file header/config/aux read split is derived from code path
  (`quantizable` header scan + `config.json` + 7 aux files ~24 MB inside the
  50.0 MB scope read), not from per-file counters.
- Caller-side independent byte verification was not run.
