# Cold encoder identity fixtures and refit diagnostics — 2026-09-07

The selected CPU suite exposed extra records in
`test_the_diagnostic_records_the_optimiser_that_ran`: the first measured encode
recorded four 16-row identity-fixture refits before its four 64-row refits.
Its warm comparison recorded only four. A measurement could therefore mix
internal fixture arithmetic with the requested unit's arithmetic.

The same failure reproduced on pristine `de963ccf` under PB
`c3d56c7678a51c004d5502eb59521fb95f1a13d19fc5c0ae42867fe06f953d3e`:
8 passed, 1 failed, exit 1. This was discovered while validating the native
MoE producer and fixed separately. The original selected-suite failure was
`bb465bdfe6ba94c58facdee816e807ab0fb53b1ef2aca6c7a55a8aff8bf597fa`
(1,297 passed, 68 skipped, one failed, no missing collection).

The opt-in diagnostic sink now excludes refits while the existing encoder
identity build guard is active. The regression explicitly empties the
identity memo, so test order cannot hide the cold-call case. The existing
arithmetic comparisons still verify which optimizer ran in each schedule.
No encoder decisions, format bytes or production defaults changed.

PB `f6b109783ee6a7710ca34884ad7d774cda49ce6c7d9b4b436b0acca0561898bf`
passed all **47 tests**, with no skips or missing collection, in
`tests/test_refit_trailing.py`, `tests/test_scale_refit.py`, and
`tests/test_encoder_identity.py`. It ran on dl380g10 with eight reserved CPUs,
16 GiB memory, pytest `-n 8 --dist worksteal`, and OMP/MKL/OpenBLAS threads
bounded to one. Torch was `2.11.0+cpu`; there was no GPU or performance claim.
Exit status and the actual CAS payload were checked independently; payload
SHA256: `4aa985d3968205db5abd57f8bc38f11fc002ee055c2bce2f94835312bdc820e4`.

The selected 48-module CPU population subsequently passed **1,299 tests** under
PB `05650b2a715464a1db6f2eca2648b84a60379e786b0b9fde6dce0b9cfce133b9`,
exit 0 on dl380g10. There were 68 skips: 52 CUDA/encoder checks and 16 checks
requiring historical artifacts absent on that worker; no modules failed
collection. The run reserved 40 CPUs and 80 GiB, with pytest `-n 40` and one
native thread per worker. Recorded memory peak was 12,304,764,928 bytes, so the
memory reservation was conservative; no OOM or telemetry errors occurred.
The checked surface is
`/mnt/shared/tessera-native376-resource/native-moe-selected-cpu-r2.json`,
SHA256 `93ee9e0d56a9611b53cde3a9d698c2054238952ebb71174adc6c45f2e164e78c`.
The independently rehashed CAS payload is
`57c804c595dada5866384da4ce0061f49cf66bacccee7edb91d1950a60431024`;
it retains the exact skipped tests and reasons. This remains CPU validation.
