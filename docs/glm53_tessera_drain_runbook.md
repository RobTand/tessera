# GLM-5.3 Tessera export: drain → merge runbook (2026-09-01)

Written mid-run so the sequence survives a context reset. Steps are ordered
because two of them are irreversible.

## 0. Confirm the drain

    cd /home/rob/prismabuild && PYTHONPATH=src python3 tools/fleet/tessera_status.py

Expect `shards 120/120`, `failed 0`. The monitor also emits
`ALL 120 SHARDS ENCODED` and exits.

## 1. Merge (safe; writes a new directory)

    cd /home/rob/tessera
    PYTHONPATH=src python3 experiments/merge_tessera_parts.py \
      /mnt/shared/models/GLM-5.3-Flash-Tessera-E2M1K2-20260901-parts/shard-* \
      --out /mnt/shared/models/GLM-5.3-Flash-Tessera-E2M1K2-20260901 \
      --move

`--move` needs one filesystem; both are on /mnt/shared. The merge enforces:
no shard claimed twice, the union covers the source's 120 shards exactly,
every `weight_map` entry resolves, and all 21 SHARED encoding fields agree
across parts. That last check was repaired in `317c882` -- it was comparing
5 of 13 fields and silently skipping the eight that do not exist, including
`grid.digest` and `conv_memory`, the two that detect encoder drift.

Expect body bpp 4.0005 and **151.5 GiB** on disk. Do NOT expect ~163 GiB --
an earlier revision of this line conflated our expected size with Mia's EXL3
reference (163.560 GiB), which is a different artifact. Ours is 12.1 GiB
smaller at a comparable rate; that is a result, not a shortfall.

## 2. Cross-check BEFORE deleting anything  [DONE 2026-09-01]

Merged clean: 120 shards, 38,770 tensors, 4.0005 bpp, 151.487 GiB. The repaired
guard resolved 21/21 SHARED fields with zero disagreements across all 120 parts
-- no encoder drift. Shards 00001 and 00036 (partA) and 00085 (partB) verified
byte-identical old-vs-new, joining shard 61. Four independent witnesses; step 2
passes and step 3 is unblocked pending the announcement it requires.

Two part directories from the 2026-08-31 run survive as an independent
witness. Shard 61 was already verified byte-identical between the old and
new encoders (sha `f90ec57f...`, 1,342,636,295 B). Spot-check 2-3 more
old-vs-new shard hashes before step 3.

## 3. Delete the superseded parts (IRREVERSIBLE -- announce first)

Only after step 2 passes. The old partA/partB dirs are unmergeable on their
own (both were killed before writing an index or config), so they have no
value except as this cross-check.

## 4. Deploy the encoder fix, THEN dispatch the probe

Order matters. `run_local_action` re-verifies the code closure at execution
and again at the CAS commit point, and the closure covers every
`tessera/**/*.py`. Syncing before the queue is empty fails the in-flight
shards.

    rsync -a --exclude .git --exclude __pycache__ \
      /home/rob/tessera/src/tessera/ \
      /mnt/shared/prismabuild-fleet/checkout/tessera/src/tessera/

Then the rate-band probe (one action per shard, band-prices 2.0-3.5 bpp
from one encode per Linear):

    cd /home/rob/prismabuild
    PYTHONPATH=src python3 tools/fleet/dispatch_tessera_ladder.py --shards 1-120

## 5. Worker loops

The 8 loops now running hold the PRE-ledger `pool.py` and declare no
capacity. Do **not** start ledger-aware workers alongside them -- a new
worker would see an empty ledger and oversubscribe the box. Wait for the
current loops to exit on max-idle, then:

    PYTHONPATH=src python3 tools/fleet/worker_loop.py \
      --gpu-slots 4 --mem-gb 96 --timeout-s 7200

4 slots per box was measured: one exporter holds ~8 GB, four concurrent took
a GB10 from 116 GB free to 55 GB, and gave 1.38x throughput per box (188s
solo vs ~545s for four) -- about 2.8x across the fleet.

## Known-open, not blocking

* The E4M3 sub-cap menu rungs are handicapped by their partition
  (1.95-2.32x off best-subset). Live on the priced menu, not on this
  artifact, which is uniform q896 = rung 7 where no partition exists.
* Everything measured today is weight-space `rel_err`. The serving metric
  is what promotes; weight-space has inverted on this project before.
