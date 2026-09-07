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
