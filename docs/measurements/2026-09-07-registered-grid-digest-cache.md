# Reuse immutable registered grid digests — 2026-09-07

The reader repeatedly recomputed complete payload-grid SHA-256 digests while
searching `encoder_profile_id`, even though `SERIALISABLE_GRIDS` already holds
those exact digests. A 30.004-second py-spy observation of the original dense
anchor preparation contains direct
`read_unit_artifact → parse_unit_artifact → encoder_profile_id → grid_digest`
stacks. The raw observation and both-host Netdata/process samples are retained
at `/mnt/shared/tessera-measurements/first-model-20260907/joint-prepare-live-profile-03/`.
This is an observed hotspot, not a measured speedup for this change.

`grid_digest` now reuses the registry's digest for the exact registered grid
object when its saved fields remain identical. Registration checks recursively
immutable builtin values once. The cache is bounded by the initial registry;
unknown grids and later registry additions use the original hashing algorithm.
Field replacement, including an `object.__setattr__` bypass of `frozen=True`,
invalidates the shortcut. No digest grammar, profile input, refusal or decoder
operation changes.

The optimized reader package source hash is
`14df443217e2a6a1bc4857532f0b5ead7fa7f5755dbd61e96605659940a8a1bc` under
`tessera.cached_unit.encoder_source_sha256`'s complete source-tree algorithm.
The frozen original producer remains
`57809bff862b880dc397e6d271a80c04d6c87d1af3bba12c076648fc5443355c`.
An external consumer may load the reader under a separate package namespace;
`experiments/qualify_reader_namespace.py` checks that coexistence against the
retained small wire fixtures. That CPU qualification is distinct from the
coordinator's actual 14-wire GPU comparison and before/after measurement.

Pre-fix PB action
`f0c419d9f06716fb1be305928837d6b0788713d2e0d959368915d170384b2702`
ran on dl380g10, Python 3.14.4, Torch 2.11.0+cpu, pytest `-n 4 --dist worksteal`,
with native threads fixed at one. Four registered-grid call-count regressions
failed at `tests/test_grid_digest_cache.py:24` (`assert 2 == 0`); nine mutation
and unknown-grid checks passed, with no skips or missing collection. The
post-change action `cc4a0bc2564cda9fcbb9bb0dddfb6051d5d2b141a3722a52d27849039cc7e530`
on Sparklina passed all 13 new tests but failed collection of the existing
Torch-dependent compatibility file because that worker's pb-cpu environment
has no Torch (71 Torch-dependent modules unavailable). It is not a passing
compatibility run. Follow-up CPU compatibility and namespaced-reader actions
require the installed x86 CPU Torch environment; production GPU performance
and wire evidence are recorded by the coordinator separately.

## Completed CPU validation

The x86 worker stopped announcing while the follow-ups waited. Unclaimed
follow-ups were withdrawn. Two attempts using the original producer image on
Sparklina (`15d9f7d525c9…`, `7fb26c3222a5…`) failed before execution because
that image is absent there; Sparky retries were withdrawn before launch after
the coordinator identified its quiesced supervisor. These are environment
failures, not test passes. The final runs reused the established PQ CPU
container environment through `experiments/reader_cpu_checks.sh`:
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
GPU visibility disabled, existing shared pytest dependencies, native threads one.

PB `d0f09fdd6f4ab67de1d5d59aa8264cf41d14041590f58697f6cfe3589a801b60`
completed on Sparklina with exit 0: **20 passed, zero skipped, zero modules
missing collection**, Python 3.12.3, Torch 2.13.0+cu130 with no CUDA device,
pytest 8.4.2 / xdist 3.8.0, four workers. Command:
`bash experiments/reader_cpu_checks.sh -m pytest -n 4 --dist worksteal --durations=5 -p no:cacheprovider tests/test_grid_digest_cache.py tests/test_ktuple.py -k 'digest or profile or unknown_grid or free_grid'`.
This is CPU correctness coverage, not CUDA surface coverage or timing evidence.

PB `bf5405ec544bf69024a3e7ccfe0f9d4a2a50da92cc96506f7731cd39c31a1d8c`
completed with exit 0 on Sparklina, one CPU: the original producer as `tessera`
and optimized reader as `tessera_reader_<source_sha256>` decoded all **11**
retained legacy fixtures bit-exactly, and both implementations refused profile
and payload corruptions (**44** refusal checks). Imported modules stayed under
their respective declared package roots; primary producer identity remained
unchanged. Result:
`/mnt/shared/tessera-clean-runtime-20260907/reader-grid-cache/namespace-cpu.json`,
SHA256 `f9c0304181b388438e7be01d709b81e619906afa2ac771c52d111065b2a212d7`.

Compile PB `236662d00ac1f627b5965c38fab330cf9b6626e30f1a02c7e38d161331e89ec1`
also completed with exit 0. Terminal/CAS payloads and the namespace result were
independently rehashed; retained bindings are in
`/mnt/shared/tessera-clean-runtime-20260907/reader-grid-cache/verified-receipts.json`.
The impacted-test selector selects 187 files because the alphabet reaches
`conftest`; the coordinator owns that broad integration population. Production
14-wire GPU equality and before/after work-per-joule remain unmeasured here.
