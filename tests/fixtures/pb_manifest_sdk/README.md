# Pinned PrismaBuild public manifest SDK test snapshot

Source project: **PrismaBuild**, Robert Tand, <https://github.com/RobTand/prismabuild>.
Source commit: `b1168d0703f74e9bb8496a263f3f0f270d483959`.
License: `LicenseRef-PrismaBuild-Attribution-1.0`; the complete upstream license
and attribution addendum are retained in `LICENSE`.

Recovered from the earlier reviewed Tessera attempt
`7d1dfa8ef3facd7b3d2c15dbb26850fbf0f8ab08`, not reconstructed as a permissive
mock. Research also compared `data_manifest.py` and `manifest_layout.py` against
served PrismaBuild main `7bf539a90402fbab88f58bd834e2abf7ee5e1e34`: their hashes
are identical. No historical checkout is needed when tests execute.

`data_manifest.py` and `manifest_layout.py` are full, byte-for-byte copies of
`src/prismabuild/` files at the snapshot commit. They retain upstream semantics,
including parser bounds, validation/refusals, v1/v2 behavior and read-order
expansion. There is no substitute/generic parser.

`core.py` is only the dependency closure needed by `data_manifest.py`: exact
class/function/constant excerpts listed below, plus the upstream future import,
collections import, `re` and `NoReturn` imports. `ActionContractError` and its
`PrismaBuildError` base are owned by `digest_primitives.py` upstream and
re-exported by upstream core; their exact class definitions are colocated here
to avoid loading the unrelated execution/storage runtime. No class/function
body or constant value was edited. This is test-only, not a replacement SDK.

## Retained source identities

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `data_manifest.py` | 13680 | `16351f0dacd4d69b26f98f85d3f0f18191346179be4a0eb7f44d2aeebee033b3` |
| `manifest_layout.py` | 18951 | `0bb5913c8799b8acbffd320d82dbbf67e670106f3fafc974510c133ccfe190d3` |
| `core.py` | 2699 | `ce418a2aa426bbda65043a84577231ed572de15f7a59ea21899f57274624c067` |
| `LICENSE` | 2985 | `d3a4336c80f5b7022c536ab05b266ecd9a753984f663120b9b49315e4e42597b` |

Verbatim excerpt ranges at the source commit (1-based inclusive):

| Upstream file | Definition | Lines |
| --- | --- | --- |
| `src/prismabuild/digest_primitives.py` | `PrismaBuildError` | 14–15 |
| `src/prismabuild/digest_primitives.py` | `ActionContractError` | 18–19 |
| `src/prismabuild/core.py` | `_SHA256_RE` | 366 |
| `src/prismabuild/core.py` | `_fail` | 441–442 |
| `src/prismabuild/core.py` | `_exact_mapping` | 445–458 |
| `src/prismabuild/core.py` | `_text` | 461–489 |
| `src/prismabuild/core.py` | `_nonnegative_integer` | 498–503 |
| `src/prismabuild/core.py` | `_sha256` | 532–535 |

## Usage and limits

Import `whole_phase_sdk` from `tests/_whole_phase_sdk.py` into a test module
(using the repository's test-helper import convention) and request the fixture.
It always loads this snapshot, even if an installed SDK exists, so SDK
availability cannot silently skip metadata acceptance. It loads explicitly
named files under a unique private package, never directory iteration or
`sys.path` injection. All `sys.modules` changes use pytest's `monkeypatch` and
are restored at fixture teardown.

Only `read_data_manifest` and `manifest_read_entries` are exported on the public
client facade. `prismabuild.storage_tiers` is deliberately empty. No public
phase-range export is supplied. Internal functions in the verbatim snapshot
are not an application import surface.

The fixture prints the actual interpreter, snapshot commit, resolved source
paths and SHA-256 identities (visible in captured setup output or with `-rP`).
It does not access the upstream checkout at runtime. These tests establish a
pinned manifest source contract, not SDK installation, deployed exceptions,
queue/storage services, native loading, staging, or CUDA numerical behavior.

Current layout semantics matter: invalid v1 phase tables return no public read
entries, while some invalid v2 declarations fail during parsing. Cumulative
frontiers, not advisory v1 `bytes`, define ranges. A trailing zero-read phase
adds no nonempty range; a leading zero-frontier phase is refused by this
snapshot's public layout helper. The tests preserve those distinctions.

## Acceptance harness correction

The earlier attempt required ordered read entries from numeric `validate_readset`
but left the original valid v1 assertion expecting phase-range dictionaries.
The revised assertion consistently requires entries. A separate regression
requires `repair_readset`'s JSON `phases` field to remain normalized ranges,
with retained entry declarations unchanged. The synthetic repair probe
substitutes only the original-file frozen digest admission gate; output hashing,
manifest parsing, validation, and reporting remain real. It is not proof about
the unavailable frozen bank.

## Current research run evidence

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
