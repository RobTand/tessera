# Exact cached-unit producer boundary — 2026-09-04

This is producer/API evidence for PrismaQuant #183 and the shared identity
needed by #186. It is not a served measurement, eligibility change, or
promotion. No GPU action was submitted for this work.

## Source and interface

The isolated branch starts at `dcf8ee89faa788056f3051922d1b6ed626d4a1cc`.
Implementation commits are `f82ee23` (projected cache intake), `42edfee`
(shared dense identity/records), and `6343137` (complete-plane and actual
reach/profile refusals). The normative interface is ARCHITECTURE §3.2.

- `experiments/tessera_producer_plan.py SRC --stack-plan PLAN --out OUT`
  serializes the existing producer planners' source slices and canonical roles,
  including the actual full source checkpoint identity.
- `tessera.cached_unit.encoding_input_identity` binds dense or projected
  encoding inputs without inventing expert fields. `unit_input_identity` adds
  the producer's explicit projection. Logical tensor names include `.weight`;
  the returned unit/cache key removes exactly that suffix. Physical source
  tensor names retain their checkpoint spelling.
- `make_unit_record` and `verify_cached_unit` use one exact-wire validator.
  `CachedUnitBundle` requires complete selected expert coverage and the source
  checkpoint seal. Export creates fresh projected identities from source slices
  and actual Hessians, then `pack_fused` wraps the accepted original blobs.
- `AcceptedUnit.wire_bytes` is the plane-region size, matching
  `ExportedUnit.exact_bytes`; `record.blob_bytes` is the complete unit file size.
  Fused wrapper bytes are additional and are accounted by the exporter.

Source archive for guarded PQ tests (tracked `src/`, including package assets):
`/mnt/shared/tessera-runs/pq183-producer-source-6343137.tar`, SHA256
`50b2b29da4b7e0c818cdde442303ebe9e40fd0f4a04d8d8603a2da3450a0d14f`.
It is extracted inside a PrismaBuild worker; no installed Tessera pin changed.
The source seal is deliberately conservative, so a different producer source
can refuse reuse even when the behavior-derived fixture identity is unchanged.

## Red before green

Every action below used deployed PrismaBuild `aa6d3cfa2f77`, tag `x86`,
`cpu=1`, `mem_gb=4`, `/home/rob/venvs/pb-cpu/bin/python`, and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. The mode is serial,
device `torch 2.11.0+cpu reports no CUDA device`, with **zero skips, zero
uncollected modules, and an empty skip-reason map** for each targeted run.
These counts cover no CUDA-gated surface. Expected failing actions were
retried by the deployed pool; those retries are not independent populations.

| Action | Targeted evidence | Observed pre-fix failure |
| --- | --- | --- |
| `2a327f81ce138d78b6ce522c49d5358f7a95634e6412c18becf4ae4ce71cb75b` | Initial 19 cases, 19 failed | `ModuleNotFoundError: No module named 'tessera.cached_unit'`; planner cases: `AttributeError: module 'cached_test_exporter' has no attribute 'project_expert_plan'` |
| `87cfbff069251a76d28591cb3e961ba751092572ff83b8b032e5d36af146232a` | Four additional/export cases against pristine base, 4 failed | The same absent producer API/module failures |
| `d17722f024143bb1cd2a8bfdca775c307c3c8f7383148112150b28e50f540ce4` | Generic identity accessor, 1 failed | `AttributeError: module 'tessera.cached_unit' has no attribute 'encoding_input_identity'` |
| `15c875dcba0ca413912c1b04423b26290dfcb15f2449022f8217c574cc632569` | Generic dense record, 1 failed | `src/tessera/cached_unit.py:149: KeyError: 'projection'` |
| `3de18faec110023612ee7055e4d2d1ef220902cc32e7282557eddc8ea74de24a` | Relabeled reach and sole incomplete terminal, 2 failed | `tests/test_cached_producer.py:291` and `:317`: `Failed: DID NOT RAISE ValueError` |

All 26 cases passed on `6343137` in 106.23 seconds, with the same CPU-only
population and zero skips/uncollected modules. The test that invokes the
actual exporter replaces `encode_linear_planes` with a function that raises;
it verifies all emitted fused members equal the selected cached blobs exactly.
The malformed-wire cases update their blob digests, so they exercise wire
validation, not merely a stale checksum refusal.

Final targeted action:
`53c23fb8debbbf952ef7d3168160f4ebe9ed885d9308a741bb8f584c81d9d1fd`,
worker-observed exit 0. Population:
`/mnt/shared/tessera-runs/pq183-producer-green3-surface.json`.
Snapshot `c773ad627b15b2610be991a8e09c3a530975cbed`, verified effective-source
SHA256 `d6b81f2bc8406e491351627ac110b64cbc5ba60dce650c517403a13c388926af`.
CAS receipt `4950aca0bf5aaf61cf128be64e513fe24b3a599598ba2b1f3d7a9a5388d0f6bd`;
result `4069bb4f006eaf417b4b5f084d41b76ecce798c7e0cd3c8f8685c81a6d6f1214`.

## Broader validation and limits

The exact-base impacted selector ran in its own PrismaBuild checkout after
fetching the source Git bundle. Action
`e679c0e69ed2a7bb2796f4b0a1631e095d814a39bf3445f8e6a47bffc787217b`
returned `full`, with `tests/conftest.py` reverse-reachable from unresolved
dynamic loaders. This caused a serial, CPU-only full run to be submitted,
action `30eb1f5ff2699022a6a6b25f37de24bc58197bf5a27de3c6b9ba10e57e48bcf8`,
surface path `/mnt/shared/tessera-runs/pq183-producer-full-surface.json`.
At this receipt's first recording (2026-09-04 21:16 UTC) it remains in flight.
It measures `42edfee`, not the later hardening commit; no full-suite result is
claimed here. The coordinator's eventual merge result still needs its own
paired device-population receipt.

The PQ packed campaign remains fail-closed. Selected interpolated rates without
an actual measured blob are refused; there is no re-encode fallback. This
work does not supply a source-matched served KL A/B or qualify a new recipe.
The independent dense LFM alias fix `a6306b4` belongs to the serving campaign
worker and is not duplicated in this branch.

One off-task prose correction was made separately in `9f9ec91`: exporter help
no longer incorrectly says BF16 lacks a route. No off-task code fix or new
issue was hidden in this change.
