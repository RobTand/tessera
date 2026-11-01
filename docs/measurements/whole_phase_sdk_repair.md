# Whole-phase public SDK consumer repair

Scope: metadata-only repair for tessera#1316. Both consumers previously duplicated
v1-only, exactly-one-raw-phase assertions; neither called the removed storage_tiers
export. The shared assertion now defers generic validation and consumption order
to public read_data_manifest and manifest_read_entries, explicitly refuses empty
reads, then asserts one expected nonempty cumulative interval. Numeric returns
ordered reads; repair reports independently retain normalized phase ranges. T4's
member restrictions remain separate and unchanged.

These are source-contract checks using the seeded test-only facade, not an
installed/deployed SDK exception reproduction or GPU numerical qualification.
See tests/fixtures/pb_manifest_sdk/README.md for imported interpreter/source
identities, verbatim parser/layout snapshot b1168d0703f74e9bb8496a263f3f0f270d483959
and byte identity with served PrismaBuild 7bf539a90402fbab88f58bd834e2abf7ee5e1e34.
No fixtures or acceptance tests were changed. The repair-report synthetic probe
substitutes only frozen-original admission digest; it does not prove the frozen
bank is available. No numerical qualification was restarted.

Inherited pre-fix evidence is retained in that README and this Work Package:
original six cases fail only numeric's former range return assertion on both arms;
complete files have 30 failures each, demonstrating schema rejection/return-shape
mismatch. Issue 1310 may overlap, but no per-case evidence establishes equivalence;
use the supplied targeted receipts for intake comparison. Raw-log retention issue
1299 and native/executor qualification remain separate.

## Verification

Command (both complete files combined):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_t4_prefetch_staged_qualification.py tests/test_value_prefetch_numeric.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

An initial implementation run failed with 59 failures due to missing helper import
bindings; fixed imports before retry. Retained operation:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3d0611b3e1ebdb115ae5b44c4b3d260166a6a7ea3e4a1ee0a6f8fdaf114ab359","size":162167}},"operation":"4c35890bbe8740977000acd7a9de42da2e4b5c6892f75bf5f110c01eab601fc0","receipt":"05eb60ef22da8ce7e921c6369f12a169b658863f90f17989ce3de919920c8c94"}]

Post-fix x86: 96 passed, 1 existing admitted-synthetic-bank skip; zero devices
allocated. All metadata controls executed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"713499efdae451d69b6953ef394f3520314c5698dc5ac4e4c3808304f095ab3e","size":1963}},"operation":"d7bab47485d41f46727e91b2a8700b5dae570f33aa96e73486752311c7e5f6ae","receipt":"c1e517fed08103d38ff3aa09c4b0b2cc57d570a6688143c3ed214d6a13ebcfed"}]

Post-fix GPU arm: 95 passed, 2 existing skips (admitted synthetic bank and
CPU-only native-map proof); zero devices allocated. Runtime 4.30 seconds.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"60c2d0c7d0dfc586d0f8aca1a0b45b09c07f8b2f5355ee5a087dda728fafc009","size":2006}},"operation":"7d9b61f29342d7d5c0cf0094d71924bdb62ee5bb748544bae75440afbc2cce67","receipt":"3165b2afef48a62b24b6e1b152ea10bb62a82939bca18e383606db49b97078e7"}]

Additional x86 guards:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py tests/test_routed_gate_staged_store.py tests/test_routed_gate_staged_store_sdk.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

231 passed, 1 installed-public-SDK dependency skip. Zero device allocations.
The skipped integration is not proved by the snapshot facade. Final diff review
confirms no protected acceptance edits or new artifact-opening paths; diff --check
passes. The retained targeted x86 stdout was read again and confirms the former
numeric valid-control mismatch at its entries assertion, not an import failure.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d4ff85caae3574dea1e4a750fa02a47960236790c50e3fa0fe2809b1c88b3b28","size":1958}},"operation":"c6de8f343345fd9f33c4b8c7d0c1ee38f6d6156179238dff2bed095a5d7eaa53","receipt":"5cd4121bd6067ba23db5dc33716afeb7c9cd9dbfc37d8f4e1955b5716e5964a2"}]

## GPU-arm preflight rationale

The inherited targeted pre-fix GPU receipt already supplies the targeted run
before this sweep (2909a1bd17d70489e1df00764d0ecdce097630be6a692ee87ab07c38d415f5b2).
The next run decides post-fix O1/O2 compatibility with the configured GPU-arm
interpreter, not CUDA performance. CPU cannot prove that interpreter compatibility.
The smallest mandated population is the two complete acceptance files, expected
under two GPU-host minutes with zero device allocations; no compilation, staging
or CUDA numerical execution is requested.


## Before the public-SDK fix

The following retained research evidence predates the completed consumer fix.
These outcomes describe the original base, not the current fixture.


Base: Tessera `17b3fdb396410698c9a142615345f046d2f94a5d`. Unlike the
historical attempt, this base already includes a partial public-SDK fix in
`2d0522a32c56633d3d9d4a5397a6a8ac500dfcfb`. Do not attribute current failures
to the removed unsupported attribute.

GPU targeted-run plan (written before submission): decide whether all six
original metadata cases execute under the pinned GPU arm's Python and sealed
site with this same source fixture. x86 cannot prove that interpreter's result.
Smallest selection: the two original parametrized test names (six cases), two
workstealing workers, no native reads, builds, staging or CUDA work. Expected
GPU-arm occupancy: under one minute, with zero device allocations. A complete
file run follows only after this targeted reproduction, to verify the additional
schema and reporting regressions on that arm; expected occupancy under one
additional minute, still no CUDA numerical qualification.

Targeted x86 result: 1 failed, 5 passed, no skips/device allocations. The
numeric valid control fails because it returns normalized ranges, not ordered
entries; the other five original cases pass. Interpreter and snapshot identities
are captured in retained stdout. Targeted commands use:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_t4_prefetch_staged_qualification.py tests/test_value_prefetch_numeric.py -k 'test_public_manifest_validator_enforces_exact_whole_file_members or test_public_pb_readset_and_whole_bank_phase_contract' -q -rP -n 2 --dist worksteal --durations=10 --tb=short --basetemp="$TMPDIR/pytest"
```

Targeted GPU result: also 1 failed, 5 passed, no skips/device allocations,
2.63 seconds. Same failing numeric valid-control return-shape assertion.
This matching pair is available for intake's comparison with issue 1310;
that issue's per-case evidence is still absent, so equivalence is not assumed.
Issue 1299's raw-log retention work is separate.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5c88a37f8ad0a5fd79ae0712d1e5472e6c679b40ba0583ef6ccdae1f50094fc1","size":7402}},"operation":"2909a1bd17d70489e1df00764d0ecdce097630be6a692ee87ab07c38d415f5b2","receipt":"bcc60d6e02112d073e6eca1326d71db210156cf37af9a7089b590e2afb56d650"}]

Complete-file command, run once on each arm after targeted reproduction:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_t4_prefetch_staged_qualification.py tests/test_value_prefetch_numeric.py -q -n 2 --dist worksteal --durations=10 --tb=short --basetemp="$TMPDIR/pytest"
```

x86: 30 failed, 66 passed, 1 skipped in 8.48 seconds.
GPU: 30 failed, 65 passed, 2 skipped in 3.34 seconds. Both allocated zero devices.
Failures are consumer refusals of valid v2/trailing-zero controls or numeric
range-versus-entry assertions, not import/discovery errors. Negative mutation
cases first validate an accepted control; current v2 failures occur there.
The repair-report regression currently reaches and passes the normalized JSON
report assertion, then fails its ordered-read return assertion. It prevents
fixing that return while silently serializing read entries as `phases`.
Existing skips remain: explicit admitted synthetic-bank proof on both arms,
and CPU native-map proof on the GPU-visible arm. These are not metadata skips.
The actual deployed-SDK exception and GPU numerical qualification remain
unconfirmed; neither is established by this pinned source facade.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"605118362442131eca4c4c305365852b0c7df4a345c3edcbcdaf7a166d76e92b","size":50559}},"operation":"71d53ac4f06851dbb0bd9669887bb87b7b9e65e4a12e656254a81694e0175993","receipt":"170fa57f3b1cb2c19c5190856cdd1020cb20d0db58a8d602ab4b1779ee187d56"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b9e5e24f39922cf6e968add9ed25dab00e9208ab7a476573d705c45eb6273579","size":48956}},"operation":"b20838ec4b8e9f962abba563857164848af3dc5f78cccbcf634e36f0a86edbcf","receipt":"4007221f69fb96aa0281122f2a7be8378ef7237e24e9f3d88a8e65310e18b65f"}]

Repository x86 guards: the same pytest options applied to
`tests/test_issue_refs.py tests/test_impacted_tests.py` passed 212 tests in
104.70 seconds, with no skips or device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1efe7b60d487370f2264f8c89aded8187c0c544ae2f27c2c0d75b8673c81590a","size":1761}},"operation":"8345ec398594bd6bec6ec562639247222fd53f4a0210742b015e933dac802ec2","receipt":"5c31f9e91389e30132d83bfc8d0c33ccf5b0265e391b1910642d21710bbb955a"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b40a85bd5be8d4dfc6022cda58c61b595e048c6cccaab3fbf11ba5cdc5e9cb07","size":8476}},"operation":"8c2adb67cddb1f1fa130e29b7166363bd67ff996fbaa4c88e46f44ab70d03dd4","receipt":"19b93b003e4327f485a565ed7ec5d14807d178d03311ea63c87cc8f69fccc6aa"}]


## Return-contract repair, 2026-11-01

Both public validators retain their original manifest and normalized phase-range
return shape. Public ordered reads are validated by the shared assertion and
remain internal to it. Restored acceptance assertions cover v1, v2, trailing
empty phases and v2 consumption order. This metadata repair runs no GPU kernels
and does not qualify the retained synthetic bank or serving behavior.
