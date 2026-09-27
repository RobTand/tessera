# Compact serving manifests — Tessera #635

## Scope

Only serialization of new serving manifests changes: one writer, compact JSON,
existing insertion order. Main, stock-twin, merged-part and fresh residency-refresh
outputs use it. Config/index JSON stays indented. No wire, schema, contract cell,
kernel, serving gate, or PrismaQuant pin changes. Existing artifacts are not
rewritten or re-bound.

## Real GLM byte measurement

PrismaBuild action `ae50d2ce177e78283d81f74e5748ce4cfdeb263c529bb890687819cbc2140eb5`
finished on dl380g10, exit 0, CPU 1 / memory 2 GiB, no GPU. CAS result
`edda6ff535aa7a035e9531137d74e241175ee21181ccd2fab61982ffbcf8bcb7`, receipt
`15aa53a94afbe4e924dcf4c51c9c5ac7ac5d8b715bac19e89c955fc506df145c`.

Source: `/mnt/shared/tessera-runs/moe/glm53-body-mtp-bf16-r1024-20260926/exported`.
Read its manifest, wrote a **new temporary** compact manifest with the production
helper, read it back and verified parsed equality and insertion order. Recorded
stat fences for all 128 files before/after; the source roster and stamps stayed
unchanged. No source/payload digests were recomputed. Temporary output was deleted.

| Quantity | Bytes |
| --- | ---: |
| Original manifest | 42,445,119 |
| New temporary compact manifest | 29,341,610 |
| Measured serialization saving | 13,103,509 |
| Existing artifact, all files | 175,643,087,583 |
| Projected fresh artifact with only that replacement | 175,629,984,074 |
| Historical strict all-file budget | 175,642,157,752 |
| Projected headroom | 12,173,678 |

The last artifact size/headroom are arithmetic projections, **not a fresh export
or a promotion receipt**. The existing artifact remains 929,831 bytes over budget.
No serving, KL, GPU, runtime-speed or wire-byte claim was measured.

## Digest/read compatibility audit

- `experiments/full_engine_artifact.py` hashes actual disk bytes (streamed with
  a stat fence in attested mode). `experiments/capture_full_engine_resources.py`
  carries that digest to `experiments/full_engine_worker.py`, which hashes the
  served file before parsing. Both old and newly written files retain exact-byte
  identity; an old binding cannot be reused for a compact rewrite.
- `src/tessera/cached_unit.py` and `experiments/glm_routed_owner_inputs.py`
  compare canonical **cached-unit document** digests. Their canonicalization
  stays unchanged. Bound child/authority/policy files separately bind raw bytes.
- `tools/tessera_route_census.py`, `experiments/ts113_sparklina_campaign.sh`,
  `experiments/ts5_lfm_teacher_bound.py`, and the source/selected-cache handoff
  record disk bytes, not an assumed pretty representation. Runtime/core manifest
  hashes elsewhere name vLLM metadata, not this file.
- PrismaQuant's `prismaquant/tessera_route_receipt.py` and
  `prismaquant/shipcard_cli.py` bind the exact supplied file text and independently
  hash disk bytes. Read-only audit at PQ `141b08c20f1`; no PQ edit is needed.
- JSON readers, including the streamed artifact reader, accept both layouts.
  `experiments/ts104_chain.sh` was the whitespace-sensitive exception; lane
  display now parses JSON. The test executes only that line, never the campaign.
- `experiments/refresh_native_resident_manifest.py` already creates a fresh output
  and records the original source digest. It now shares the compact writer; this
  task did not run refresh on any real artifact.

## Regression receipts

All runs: dl380g10, Python 3.14.4, torch 2.11.0+cpu, no CUDA; two xdist workers per
PB shard, worksteal, native threads 1, CPU 2 / memory 6 GiB, priority -10,
`--timeout-s 1200 --pytest-args '["--durations=15"]'`. Interpreter:
`/home/rob/venvs/pq-pb461728e4-tessera-4c4ff1c2/bin/python`.
All three post-fix shards: exit 0, 0 skipped, 0 uncollected modules, 0 tests
allocated on device. These passes do not cover the CUDA-gated surface.

| File | Before (PB key) | Failure before fix | After (PB key; count) |
| --- | --- | --- | --- |
| `tests/test_serving_manifest_json.py` | `26491db33ea1d2a3e3eea9766424dd6dc116518fe383779361b30f22f302f80b` | `AttributeError: module 'tessera.serving_parts' has no attribute 'write_serving_manifest'`; lane display `assert 1 == 0`; refresh byte index 1 newline vs quote | `2018cf2b4cfc991877dc8151405a8a204e5353e9bb0136a21219f29e9f1f7e80`; 3 passed |
| `tests/test_serving_parts.py` | `a6cc2c5803ea88a5b5b848ffa8e142e58ffee71a25bb265f928cbb815c8d4294` | Both old-fixture styles: byte index 1 newline vs quote | `23d474967e877946ef96aadeb9b9efa6166ffc060a7f336f08d8fc5d06eb99f5`; 42 passed |
| `tests/test_refit_trailing.py` | `59271a9884b28f85b2a5524229c5f20ab89501985236d283d2cb23463c11473f` | Exported main/twin compact assertion: byte index 1 newline vs quote | `5e716070a4ae3a0b10785889fcbc81e886e0a63163637e56d9ff90c63dc95c0e`; 9 passed |

Post-fix aggregate: 54 passed; each file's collected/ran/outcome counts reconcile,
no missing files or reconciliation problems. PB terminal records and CAS receipts
were inspected. Local receipt files are `.pi/pre-fix.json` and `.pi/post-fix.json`
in the task worktree. The regression covers parsed field equality, deterministic
repeat writes, old pretty fixtures, raw-byte digests (both attestation modes),
unchanged source files, real main/twin exports and checked part merge.

## Conservative impacted-population follow-up

The selector at `de414945f` returned 299 files (`narrowed`, no forced-full paths).
The known standalone `tests/test_native_a4_serving.py` has zero pytest items and
was explicitly excluded with that reason; its manual GPU gate was not run.
PB selected the remaining 298 files in 20 shards: 5,309 passed, 1,218 skipped,
one failed (the new doc referenced an issue missing from the offline snapshot).
No CUDA device or allocations; 0 uncollected modules in the reported surfaces.
This first aggregate was **red**, not a clean full-suite receipt.

The failing issue-reference file passed all three tests on pristine `f82f879eb`
(PB `f0f540a699fe45439f2987262eeb8c8d259eb1869c75d7102ff47ccb127a2d8a`).
Refreshed the snapshot through `tools/refresh_issues.py`. Retake
`1c752bc25320e3442ac510d5d6dc20317d2da5b205841ecb1e5664aa336e2b1e`
passed six tests with one attributable module skip and no reconciliation gaps.
It includes the three issue-reference tests and the three manifest regressions.

One original shard exited 0 but omitted the second of two equal helper-issued
module skip records (`tests/test_kl_tool_decode_regime.py`). The retake above
records that exact file's collection skip separately; it does not claim its
external-instrument tests executed. Reason verbatim:

> box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument -- nothing set KL_TOOL_DIR and its default is not resolved for this root (set KL_TOOL_DIR; documented default /home/rob/dq-runs)

Filed the shared collection-accounting finding in PrismaBuild issue 1220 rather
than weaken reconciliation. Original counts and all verbatim skip reasons remain
in `.pi/impacted-pb.json`; the explicit standalone exclusion is in
`.pi/impact-pytest-selection.json`. This is a CPU-only selection plus focused
retakes, not an all-green single aggregate or a device qualification.

Hosted bytes-only CI additionally caught a missing optional dependency in the
new refresh test: `ModuleNotFoundError: No module named 'safetensors'` on run
36306852805. That case now explicitly skips when torch or safetensors is absent;
the compact writer and raw-byte reader tests remain dependency-free.
