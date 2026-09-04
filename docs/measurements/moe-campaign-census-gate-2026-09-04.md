# Complete MoE campaign census gate — 2026-09-04

Source: `1a5e318` (population gate), `fcc6115` (dependency-free shared symbol
normalization), `83eff6e` (sidecar binding), `62da9c4` (actual shape evidence).
Final source head: `62da9c485eac11b16af58b42e8056068fb563db4`.
Each source change includes its architecture update in the same commit.

This is CPU evidence for a receipt gate, **not** a served MoE attestation,
quality measurement, or wire-byte audit. No cells, model bytes, defaults or
GPU workloads changed here. The gate derives its owner/expert/projection
population from the common plan and validated merged sidecars, not a fixed
campaign count. It requires one served routed record per owner in both driven
phases, exact runtime/eager/resident context, owned launch pairs and activation
contracts, canonical actual regimes and declared N/K geometry. It retains
the runtime-selected backend suffix in its output.

The census hashes its actual config/manifest before loading and verifies the
same sidecar seal after both forwards, before publishing. The checker requires
exact supplied-file hashes; host/container aliases do not weaken that join.
Generic missing manifests remain explicit null, but this merged-artifact
campaign requires both files. Checkpoint immutability through serving remains
an orchestration requirement; sidecar sealing is not a tensor/wire audit.

`--require-attested` replays raw records against the **current** contract. It
never trusts embedded historical agreement. Without that flag genuinely
unattested owners can pass the population/dispatch check; they are not promoted.

```sh
python experiments/ts5_census_check.py \
  --plan PLAN.json --checkpoint MERGED_DIR --census RAW.json \
  --runtime-image REPOSITORY@sha256:DIGEST --out NEW_CHECK.json \
  [--require-attested]
```

## Population and execution

Every workload used deployed PrismaBuild on dl380g10: one CPU, `mem_gb=4`,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. No direct local tests
ran. Red jobs were retried by the deployed pool; superseded populations remain
beside the final files below. All population paths share this prefix:
`/mnt/shared/tessera-runs/ts5/lfm25/`.

The ordinary CPU red/green test populations reported verbatim:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Their skip reasons are `{}` and uncollected lists are `[]`.

## Red before the implementation

| Added test group | PrismaBuild action | Pre-fix failure |
| --- | --- | --- |
| Complete population; owner bijection (10 cases); raw context (13); host/container alias; plan/config/projection population (16); promotion replay; duplicate JSON keys | `2b28d3df9175` | `FileNotFoundError: .../experiments/ts5_census_check.py` — 43 failures, the checker did not exist |
| CLI passed/refused output, input fingerprints and exclusive receipt creation (2 cases) | `651529b193ad` | `can't open file .../experiments/ts5_census_check.py` / `assert 2 == 0` or `assert 2 == 1` — 2 failures with the new checker removed only inside the disposable PB snapshot |
| Missing/null/wrong/extra raw sidecar seals (4 cases) | `792cbd93262f` | `Failed: DID NOT RAISE ValueError` |
| Producer exact-file hashes and absent manifest | `792cbd93262f` | `AttributeError: module 'tools.tessera_route_census' has no attribute 'checkpoint_sidecar_hashes'` |
| CLI whitespace-only config mutation | `792cbd93262f` | `assert 0 == 1` — wrong file bytes were accepted |
| Config/manifest changes during serve (2 cases) | `01b2e8b7a175` | `TypeError: checkpoint_sidecar_hashes() got an unexpected keyword argument 'expected'` |
| Malformed/symbolic/noncanonical/zero-dimension shape, wrong decode regime, wrong N or K (7 cases) | `0f25b66f378a` | `Failed: DID NOT RAISE ValueError` |
| Dependency-free routed agreement import | `791b417868f1` | `ImportError: torch deliberately unavailable in pure census regression` from `moe_route.py:75`, imported by `all_structure_agreement` |

The shape red also included one prefill-equals-decode case already refused by
the prior generic distinction check; that redundant passing case was removed.
The retained seven cases each failed before the fix.

Red population files, respectively:

- `astra-ts5-census-check-red-r2.surface.json`
- `astra-ts5-census-check-cli-red-r1.surface.json`
- `astra-ts5-census-sidecar-red-r1.surface.json` (6 failed)
- `astra-ts5-census-seal-red-r1.surface.json` (2 failed)
- `astra-ts5-census-shape-red-r1.surface.json` (7 failed, 1 redundant case passed)
- `astra-census-no-torch-red-r1.surface.json` (1 failed)

The initial test setup attempt (`e5c60a7fa531`) failed collection because a
decoder constant was imported from the wrong module; it is not counted as
pre-fix gate evidence. A later no-torch run `03b2f92a823a` exposed test-level
imports of the runtime route/telemetry. It failed collection at the actual
`moe_route.py:75` torch import. Those test constants now derive from the pure
scheme table. That red population reported no torch, 0 skips, 59 uncollected
modules, including the then-broken collector test module:
`astra-ts5-census-pure-red-r3.surface.json`.

## Final green

Action `3a407f810de65c358ad600c7296de7c651706b4330fc6f9d46b4b89cc6403775`
ran **126 passed, 0 failed, 0 skipped, 0 uncollected**, serial CPU with torch
2.11.0+cpu and no CUDA. Snapshot:
`17d0ebf85f74cb96d7f935a66db7edf91c715498` of final source `62da9c4`.
Receipt: `ef0cb65a594f299dd24e473e87fef28e4cc70ff78e66c0fd1d89a58edfd490bf`.
Population: `astra-ts5-census-check-green-r3.surface.json`.

Explicit files covered:

- `tests/test_ts5_census_check.py`
- `tests/test_census_no_torch.py`
- `tests/test_census_cell_agreement.py`
- `tests/test_census_engagement.py`
- `tests/test_census_runtime_scope.py`
- `tests/test_census_runtime_wiring.py`
- `tests/test_route_census_module_space.py`
- `tests/test_ts5_serving_gates.py`

Action `154157293f0cb827f4c31bbbd1bd4ae182d920f4577fa1422e7db37c0c292120`
ran the entire collector test file with torch imports explicitly blocked:
**60 passed, 0 failed, 0 skipped, 58 uncollected modules**. The collector
itself collected and ran all 60; the 58 are the repository's other
torch-dependent modules named in the population. The no-torch hook also makes
`find_spec('torch')` return None, matching the dependency-free population's
collection behavior. CLI subprocesses still use the ordinary CPU interpreter.
Snapshot: `554a85bf27823e3650962a107197e2220316319e` of final source `62da9c4`.
Receipt: `8f7318d79139af67bfcbd7e0cd53df507b6faff739329d8293beab38874fb6e0`.
Population: `astra-ts5-census-pure-green-r1.surface.json`; skip reasons `{}`.

```text
tessera surface: NO CUDA -- no torch in this interpreter (ModuleNotFoundError)
tessera surface: 0 test(s) skipped, 58 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

The independently integrated pure-helper repair also passed 32 targeted tests
in action `387c970c844bfbfab49f07d240710c5ecb4ae16cf0f6f6cc37bab333205a0b4d`,
CPU torch/no CUDA, 0 skips/uncollected. Its new subprocess regression blocks
torch import while executing actual routed agreement.

## Selector boundary and integration handoff

Action `6972cbb1019ae6041beefd957f86e2e2d0d8e4fbf2ce9a7bdb30e308c53a4d55`
compared exact fetched base `0f371de9ca543c23a4f8c9c8de2fdab29936dfce` with
the parentless final snapshot, using the recorded direct-tree fallback. It
returned `narrowed` with 49 files because the shared scheme helper has a broad
reverse-import population. Receipt:
`35f792d5e71b8716a92460d313df271533abdcdf8e58977cb2f2146df6380530`.
Exact output:
`/mnt/shared/prismabuild-fleet/cas/blobs/fb/fbae555c0238cd5fddd6f27b133f435af346a8f9b31eb130f298b18abe1dd002`.

That selection is **not claimed complete**: it omits the dynamically loaded
`tests/test_route_census_module_space.py` consumer of the changed census tool.
This file was explicitly covered in the 126-test run above. The selector
defect was filed for a separately dispatched repair; it is not hidden by the
green gate evidence. Remaining selected integration coverage belongs to the
coordinator's merge population run, as agreed; this worker did not duplicate
a broad or whole-tree suite.
