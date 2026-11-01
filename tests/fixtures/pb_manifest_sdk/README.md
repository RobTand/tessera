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
