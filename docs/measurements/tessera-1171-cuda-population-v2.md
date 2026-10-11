# Tessera tessera#1171: full CUDA population on the merged tree of PR 1089 (version 2)

Date: 2026-10-11. Author: issuegraph attempt 3. Status: measured.

This version replaces two answers of version 1
(`tessera-1171-cuda-population.md`): the baseline verdict and the
witness-timeout verdict. The population run itself is unchanged and is
restated here. Version 1 stays in the tree with a withdrawal note.

PR 1089 merged as `36ca1dbd76e3e375d20be092d117e10bed644c64`.
This run is a post-merge check. It does not gate the merge.

## What version 2 corrects

1. The reruns of version 1 used pristine master (`adf741ef`) and a branch
   head (`f3b55c93`). Both trees contain PR 1089. A failure on both trees
   cannot show that PR 1089 did not cause it. Version 2 reruns on the
   pre-merge parent `87d222aef` (`36ca1dbd^1`).
2. The reruns of version 1 sealed a different source from the population.
   Version 2 reruns on the exact population checkout `b2875875a`. Its
   effective source hash equals the population hash.
3. Version 1 timed one `resident_bytes` call on one 5 GB shard. It never ran
   `fetch_witness`. Version 2 runs the real route end to end on resident sizes
   up to 60.8 GB, and it measures the default 30 s timeout.
4. Version 1 did not file the red baseline. Version 2 files it as a child.

## Population (unchanged)

Receipt: `/mnt/shared/tessera-suite-receipts/20261010T021500-ts1171-gpu/receipt.json`.
Pool action: `697d537739bdd711d26e052bf3129064b74293939166ae03d536a24fe9afd094`
on `sparklina`. Exit code 1 (25 red tests).

- Commit measured: checkout `b2875875a40d2b594e99f43b63cfe503d73ec2e7`.
  It contains `36ca1dbd`. PrismaBuild snapshot
  `31525413d1628a1b826123b9902ce934b361a62a`.
- `commits_measured.effective_source.agree`: true. Effective source hash
  `f6ec4e79296ed8eaafb0ad3d1ce6fdcf67f86ec5dcf83ef9ba32579ad72f0bb8`.
- Device: `torch 2.13.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10`.
- Image: `localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc5...`
  (vLLM 0.28.1rc1).
- Strict CUDA: true. `cuda_surface.executed`: 2139.
- Passed 11733. Failed 13. Error 12. Skipped 36. Xfailed 9. Xpassed 0.
- `cuda_surface.box_artifact_skips`: empty. `not_collected`: empty
  (zero uncollected modules).

### Skip reasons, verbatim (36 total)

- 10: `requires the published PB SDK`
- 4: `could not import 'prismabuild': No module named 'prismabuild'`
- 4: `output-device refusal requires two CUDA devices`
- 2: `could not import 'prismabuild.client': No module named 'prismabuild'`
- 2: `e2m1-tcq-lut-release does not cut 4 ways along columns`
- 2: `e2m1-tcq-lut-release does not cut 8 ways along columns`
- 2: `needs two CUDA devices`
- 1: `E2M1 publishes no reader range`
- 1: `explicit admitted synthetic bank proof only`
- 1: `native public-reader callback controls require the published PB SDK`
- 1: `physical container TERM→SIGKILL rehearsal requires the published PB scope`
- 1: `public schema/lease controls require the published PB SDK`
- 1: `real cgroup sampler needs an admitted PB CPU scope`
- 1: `the CPU native-map proof requires GPU visibility disabled and this session shows a visible CUDA device (tessera#939)`
- 1: `the guard's fallback import succeeds where vLLM is installed, so this box cannot tell the two readings apart`
- 1: `the installed-source control requires a noneditable Git package`
- 1: `the runtime is importable here; the device lane counts the real call`

No reason cites an absent box artifact.

## Failures: baseline comparison

PR 1089 changed 16 files (`git diff 87d222aef 36ca1dbd --stat`). They are
`src/tessera/endpoint_observer.py`, `src/tessera/endpoint_witness.py`,
`src/tessera/serving/endpoint_runtime.py`, one line in
`src/tessera/serving/__init__.py`, the endpoint tests, two endpoint tools,
`pyproject.toml`, `CHANGELOG.md` and `docs/ARCHITECTURE.md`.
No failing test file and none of its sources are among them.

Each failure group ran alone, with the container arm of the population
(`tools/suite_container.py`, image `c2e75e03`, `--strict-cuda`,
`TESSERA_SOURCE_VERIFIER` declared), on three trees:

| Tree | Commit | Effective source (first 12) | Files verified |
|---|---|---|---|
| P: pre-PR parent | `87d222aefbac679f726a42c34f824d98f2e2a7ca` | `2dde21efaa9b` | 2101 |
| M: PR 1089 merge | `36ca1dbd76e3e375d20be092d117e10bed644c64` | `3dd54fc5c961` | 2112 |
| B: population checkout | `b2875875a40d2b594e99f43b63cfe503d73ec2e7` | `f6ec4e79296e` | 2117 |

Tree B has the population's full effective source hash
`f6ec4e79296ed8eaafb0ad3d1ce6fdcf67f86ec5dcf83ef9ba32579ad72f0bb8`.

| Group | Target | P | M | B | Verdict |
|---|---|---|---|---|---|
| a | `tests/test_serving_mla_mask_runtime_gpu.py` | 1 error | 1 error | 1 error | baseline |
| b | `tests/test_serving_mla_mask_registration.py` | 4 failed, 11 error | same | same | baseline |
| c | `tests/test_graph_attest_eager_levers.py::test_frozen_2db_flags_reach_different_runtime_selectors` | 1 failed | 1 failed | 1 failed | baseline |
| d | `tests/test_mhc_fusion_cuda.py` | 8 failed | 8 failed | 8 failed | baseline |

The failing test IDs are identical on P, M and B for every group
(25 IDs). They are also identical to the 25 IDs of the population run.
The failure text is identical across P, M and B. All 25 fail on the tree
without PR 1089. They are baselines. PR 1089 caused none of them.

Action keys (P, M, B):

- a: `72ab61238e29cd15763189056780105ee8d68c1dbb865c785540c73ae47ac37d`,
  `e01aec9c13972db5f0e45d0de9f53855f65b9baddcb991bb925eb2de9c162167`,
  `e70543f81e3b24a9fdc4a4af781c9f902ab449285867a6a25f1da10690709e3d`
- b: `59bfbd72f947d7a20b3e07766bd047ba4578c6348b6ec7ab6c502e22fb9b3bfe`,
  `5d0ce0294adfae5f9792a06b89b21b54cb59c40746a68d90769e76252002900f`,
  `dd868f5911e9b17600b43cd482107faf2f354e1ce9d9f51ca36a50985298a13b`
- c: `4b88223676cbb71991307d24467914131f79289746655f53a87b9a5e45a74240`,
  `6899676bc49a3145555343b45646420676d600c1037b6a2ae8ab2370b51b393c`,
  `0be5174bf7c6d6a3307daacaa267e131c269f1bb8d60be0180834b93033b0f9c`
- d: `e68c643c59212663a2cecc877a9b06a036c784f65f7408cac34052c1dba20f81`,
  `f97f22d6a2aeecafc79fcce8994c1c1367048c012d724a75b6d80e3d07fe5b01`,
  `77dfae132f8794330e16e5bf52baf2d3186a2a40c53f168c75c6346ebb1af72f`

Receipts: `/mnt/shared/tessera-suite-receipts/20261011-ts1171-v2/rerun-<group>-<tree>/`.

### Cause of the baseline

The population image has vLLM 0.28.1rc1. The red tests need the vLLM
0.30.1rc1 runtime that `docs/ARCHITECTURE.md` names as the inspected runtime.
Group a and b fail on `ModuleNotFoundError: No module named
'vllm.v1.attention.backends.mla.index_group'`. Group d fails on `ImportError:
cannot import name '_hc_prenorm_gemm_outputs'`.

One more run of the same four targets on tree B used the vLLM 0.30.1rc1.dev336
image `sha256:acc684c06ec79aa303deda50a15c422084690d3252d2449e9702b40de571fe63`
(the store id on `sparklina`; tag `nightly-20260929`). The action is pinned to
`sparklina` because an image id is local to one docker store. Action
`2aadf2b388d32a1914b46c4bdc9e0ef30746e0cf5d5a1729841ef0429b443ac1`:
23 passed, 1 failed, 2 errors, effective source `f6ec4e79296e`.

- 15 registration tests and 8 mHC tests pass. Their 23 reds come from the
  old image.
- Group a (2 errors): the file now collects. The fixture
  `native_library` stops with `runtime gate requires the retained MLA DSO`.
  The test needs a retained build cache that a fresh cache dir lacks. The
  failure is an error, not a skip.
- Group c (1 failed): the test starts a child interpreter. That child
  cannot import `pytest` because the container supplies pytest through a
  dependency site that the child does not inherit.

## Endpoint witness path against its 30 s timeout

### What the population loads

The largest checkpoint that a population test names is the routed A4 TP2
export `merged-4c384e60` of `glm-canonical-census-20260908`:
166 safetensors shards, 75103 tensors, `total_size` 183,830,741,633 bytes
(171.2 GiB). The population reads layer-0 slices of these exports
(`tests/native_dense_prefill_cross_check.py`) and does not make the whole
bundle resident. So no population test exercises the witness at bundle size.
The measurement below does, up to 60.8 GB resident.

### Method

`tools/probe_witness_endpoint.py`, started by
`tools/probe_witness_container.sh` in the vLLM 0.30.1rc1.dev336 image:

- A uvicorn listener serves `EndpointWitnessPlugin.attach_router`. The client
  is `tessera.endpoint_observer.fetch_witness`. A request runs
  `observe_endpoint`, `observe_worker`, `observe_tokenizer`, `check_join` and,
  on POST, `publish_witness` with `prove_files`.
- The model is whole shards of `merged-4c384e60` in index order, read
  through `LoadCapture` (`prepare`, `activate`, `input_fact`) and placed on
  the GB10 device. Each rung adds shards.
- vLLM's own world group is initialised (rank 0 of 1).
- Per rung: a GET with a 3600 s timeout, a GET with the real default
  `timeout_s=30`, a POST with 3600 s, and a POST with 30 s. The probe waits
  until the listener is idle between requests.

Stand-ins: the engine (`collective_rpc` runs `observe_worker` on the one
in-process worker, not in a vLLM rank process), the tokenizer
(`tokenizers` on the bundle's `tokenizer.json`), and the loader (no vLLM
`DefaultModelLoader` hooks). TP is 1.

### Results (GB10, one rank)

| Shards | Resident (GB) | Tensors | GET (s) | Re-hash in GET (s) | GET at 30 s | POST (s) | publish in POST (s) | POST at 30 s |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.89 | 29 | 4.2 | 1.3 | ok | 8.6 | 4.5 | ok |
| 2 | 4.02 | 377 | 4.5 | 1.8 | ok | 10.3 | 5.7 | ok |
| 7 | 8.61 | 2187 | 6.9 | 3.9 | ok | 16.8 | 10.0 | ok |
| 17 | 18.43 | 6654 | 11.6 | 8.4 | ok | 32.1 | 20.6 | timed out |
| 33 | 35.19 | 14213 | 19.5 | 16.0 | ok | 55.1 | 35.2 | timed out |
| 49 | 51.65 | 21512 | 27.6 | 23.4 | ok | 131.6 | 104.0 | timed out |
| 58 | 60.76 | 25654 | 31.9 | 27.7 | **timed out** | 142.1 | 110.1 | timed out |

At 58 shards the call `fetch_witness(url)` with the default timeout raised
`runtime evidence is unavailable: timed out` after 30.04 s. The listener
still finished the request in 31.9 s.

Fit over the seven rungs: GET = 2.71 s + 0.481 s per GB (2.08 GB/s).
The re-hash alone is 0.455 s per GB (2.20 GB/s). GET crosses 30 s at about
56.8 GB resident. The fixed 2.7 s comes from the tokenizer read, the
vocabulary receipt (about 32 MB of JSON) and three `fingerprint` calls.

POST adds `publish_witness`. It re-reads and re-hashes every loaded source
file and every observed range (`prove_files`) in the listener's event loop.
The publish step costs 1.0 to 2.0 s per GB. It grows faster once the files
miss the page cache (49 shards and above). A POST passes 30 s from 18.4 GB resident.

### Projection (extrapolated, not measured)

Linear extrapolation of the GET fit:

- Whole `merged-4c384e60` on one rank (183.8 GB): about 91 s.
- TP2, if each rank holds half (91.9 GB) and ranks hash in parallel:
  about 47 s.

Both exceed 30 s. The TP2 line assumes an even split and parallel ranks. No
TP2 serve ran here. The measured result stands without the projection: a GET
at 60.8 GB resident already misses the default timeout.

### Answer to criterion 5

The endpoint witness path completes inside its default 30 s timeout up to
about 52 GB resident per rank (measured at 51.7 GB: 27.6 s). It does not
complete at 60.8 GB resident (31.9 s, client timed out). The publish path
passes 30 s from 18.4 GB resident. The largest model that a population test
names is `merged-4c384e60` (183.8 GB). At that size the path exceeds the
timeout by projection. The review warning of rev-1009-150707-75c8 is correct
for GET and worse for POST.

### Limits

- The 66-shard rung (64.5 GiB) did not run. PrismaBuild stopped the action with
  `gpu_memory_budget_exceeded` on the declared 72 GiB budget while the probe
  loaded it. The five rungs above were complete and printed before the stop.
  The per-rung JSON file is written at the end, so the table comes from the
  action's stdout.
- Stand-ins as listed under Method. A real vLLM rank process, TP2 and
  `DefaultModelLoader` hooks were not measured.
- Page-cache state changed the POST times. Treat POST numbers above 49 shards
  as a cold-cache reading.
- Nothing in PR 1089 changed. The finding goes to a child issue.

## Action keys

- Population: `697d537739bdd711d26e052bf3129064b74293939166ae03d536a24fe9afd094` (red, 25 baselines).
- Reruns: the 12 keys above (all red as the population, same IDs).
- Image run: `2aadf2b388d32a1914b46c4bdc9e0ef30746e0cf5d5a1729841ef0429b443ac1`.
- Probe CPU dry run, first version (device cpu, one shard, in the container;
  two later dry runs of the final code passed with other keys):
  `aa7ac837a4827b1d187ee2b7c25190ecb985e46a4a0c693478ca197965bdf4b5`.
- Probe GPU pilot (shards 1 and 2):
  `951b742cc8a15b4174710b5a068f40c7968dfd78765292ed4b568d0cc52abd05`.
- Probe GPU ladder (rungs 1 to 5 done, stopped by the memory guard):
  `20d6834028d91135857aed5c5d47fc8eab9358e00f8f148c1bc2efa02ad6f204`.
- Image content reference:
  `content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918`.
