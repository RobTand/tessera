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
