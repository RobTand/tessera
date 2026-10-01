# Mapper, package and pricing-fixture controls

This records the test-only repair for #753, #766 and #768. Production mapper
refusals, replay-generation refusals, source-profile bytes, kernels, the wire and
runtime pins are unchanged. One commit owns each issue.

## Changes and causal failures

- #753 uses the existing `module_name_mapper` and the producer's actual
  `_weights_mapper_table`. Raw drop-rule probes remain; the test does not invent
  a new `None` filter or change production replay semantics.
- #766 treats absent installation metadata as a named test prerequisite. A real
  noneditable Git installation is tested separately; a skip is not that proof.
- #768 resets the test's memoizer/live-index setup, not the production refusal.
  Four regression cases contaminate that state before invoking the producer
  test. The missing/ambiguous replay-generation refusal remains unchanged.

Pre-fix receipts (failed actions have no successful CAS receipt):

| Scope | PB action | Result |
|---|---|---|
| Unchanged pinned runner | `010250f9267bbf557b61538ed898b49142cff86900eed47987009e25f50abb42` | 14 failed, 10 passed, 0 skipped/errors/uncollected; 24 JUnit/population outcomes |
| Missing-install setup | `a65023b47bb5b54a9a6ee3091e6798bdf66e4c2d0f4f30dbd7fc36aa27e229a2` | 2 failed, 6 passed, 0 skipped; 8 reconciled outcomes |
| Contaminated pricing setup | `7ea6c7ac44dfedff174ee950ad1e8052adeb5a19581f8c66c36b985fc5fe279c` | 4 failed, 23 passed, 0 skipped; 27 reconciled outcomes |

The pinned runner's failures were thirteen
`AttributeError: 'WeightsMapper' object has no attribute 'get_unstacked_mapper'`
and one `importlib.metadata.PackageNotFoundError: No package metadata was found for tessera-quant`.
The missing-install regression observes that same package exception instead of
the required named skip. The pricing regressions observe
`missing or ambiguous live replay-table generation` before test isolation.

## Pinned-runtime GREEN and actual installed control

Both actions completed once on sparklina with worker exit 0. The immutable image
was `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
The installation manifest reports stock vLLM
`0.28.1rc1.dev397+gfd4a15126.d20260904` and torch `2.13.0+cu130`.

| Scope | PB action | Population | CAS receipt |
|---|---|---|---|
| All three changed test files, two worksteal workers | `aa6686d032518151c272db2f084bb73e9bdb0f9128c1e5162efa7852fee28017` | 51 passed, 1 skipped, 0 failed/errors/uncollected; 52 JUnit/population outcomes | `7568b45765b51f79a9c26bd5eb33c5bcaac9b78d32a2b359cdff67af8fa46e8f` |
| Noneditable Git installation, serial selected control | `2c17febe34ca6a2156ccf19b246e66ca6ef909fd369a4558cc2f00d8fd6eb892` | 1 passed, 0 skipped/failed/errors/uncollected | `8144ef44e79685067ecc07d11755d6bafb827e5d904f92544c622a75420dfb76` |

The first skip's verbatim reason is
`the installed-source control requires a noneditable Git package`.
The second action installs the actual source from
`git+file:///work@cb990ed546f848fbcbe230389a34e4d7f933c420` into its own
system-site-packages venv and runs that control without a skip. Its
`installation.json` records the Git `direct_url.json` provenance, actual Python
prefix and borrowed stock torch/vLLM module paths. It is not a manufactured
metadata directory or a replacement vLLM.

Both populations report one NVIDIA GB10, **zero tests allocating on CUDA**, and
`strict_cuda: false`. These are runtime/API and package-byte controls, not CUDA
surface, kernel, serving-numerical, performance or ship-cell qualification.

Artifact roots:

- `/mnt/shared/tessera-measurements/issues-ts-753-766/validation.UvF2xmyS`
- `/mnt/shared/tessera-measurements/issues-ts-753-766/installed.i892cyE4`

Population SHA-256 values are respectively
`b21801850afff6b7f4612463d6a61980d7a534c96be079ba7b5cc8feb807c1e8`
and `c604b6e6a641b40668f89152d29f22038f4ce68e04f1a1f0224b57b1d6b1a26d`.
The verified source snapshots are
`6ad78874d4154ff5e90cfe75a16379e2a839ff6c` /
`5c389c2d3dc2d4185e69d9cd383f7464807dd268aafcb1ab2723bfd6b06b54d8`
and `cb990ed546f848fbcbe230389a34e4d7f933c420` /
`e65d2684489c0eab82a6822c261d14d1e87c7707fe5bee7181c6680ba4de8af7`.
Each population's entry/end source agrees; the parallel workers agree. These
are distinct raw source identities, not a cross-arm source-equivalence claim.

Result blobs were independently checked: 9,206 bytes /
`94cce1cb2aa2006d01789a26c6e7df7801de3bd32f474ffb39a5d65f92091982`
and 5,281 bytes /
`7bbb071d4a09871b5826666f928770a02b352b546238bbee35181e7599962aff`.
Immutable stdout also matches its recorded length/hash. Local claims
`53e1bba597c4e33c9b10ac9c789cd450e11a83c1b19142efdf37f670d2d3953b`
and `1149f7d77e2299175608d8ba36e012e3434578e022495871a044d93610fed819`
pass all nine executed integrity checks, including payload hashing. Full worker
attestations were not independently audited.

CPU2/GPU1/memory8GiB, native threads1, priority-10, explicit 900s action bounds;
PB owned placement. Immediately before launch, host MemAvailable was
74,599,911,424 and 63,014,789,120 bytes respectively, above the required
25,769,803,776 bytes (16GiB floor plus planned8GiB). No CUDA-free-memory gate or
accounting adjustment. WINDOW_ACTIVE was absent before submission. Client
75/74 ended during bounded waits; later worker outcomes above are authoritative.
Nothing was resubmitted to bypass admission.

## Import-graph selected CPU population

The selector narrowed to 95 files. PB divided them into two admitted actions,
with two worksteal workers/native1/CPU2/memory4GiB per shard; aggregate
CPU4/memory8GiB. Both executed once on dl380g10, exit0, torch2.11.0+cpu,
CUDA hidden, allocation surface0, no reconciliation gaps or missing files.

| Action | Passed / skipped | Collected / ran / outcomes | CAS receipt |
|---|---|---|---|
| `33cd6870dde42f81d0974965b7b9e317f6d620b2d1458e724dbeb3a871021e4b` | 1011 / 30 | 1039 / 1039 / 1041 | `65e4f5a80230ee5d25bfd3d94c6a9f56a86574c800dcf55cadf8b32f8bbe8d47` |
| `7a02df5240dc9555e23e6b8a36e498fa0f01cd7ef02a0478bfd8dbb3cdfb8be0` | 1236 / 62 | 1298 / 1298 / 1298 | `5cf8c9330bedf12987585964521e9e79b1e3bb9ad97d2011b9802be36cff928f` |

The first shard has two collection-module skips: missing KL instrumentation and
`could not import 'vllm': No module named 'vllm'`. Aggregate: 2247 passed,
92 skipped, zero failures. Verbatim skip reasons/node IDs and outcomes remain in
`/home/rob/tmp/claude-campaign-20260926/tmp/p2p3/tessera/753-766-768-impacted-green-ts.json`
(SHA-256 `93f618d4f9b332d5017d52004eb71633c1f238f342ded27c001d65beb40f8da3`)
and the immutable result blobs. Skips certify neither the runtime nor CUDA.
Both CPU claims passed the same nine integrity checks; full worker attestations
were not independently audited.

This document was added after tested code. No whole-tree merge-result,
architecture, runtime-contract or deployment claim is made.
