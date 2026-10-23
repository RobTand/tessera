# Cached-routed fixture export repair — 2026-10-22

Repair for tessera#1124, on base `07d0734440ca676376e94de1c51f31528f3db157`.
The ordinary `--cached-routed` invocation accepted its arguments but never assigned
`plan`. It now calls the existing `supplemental_plan(original)` when neither
`--dense-only` nor routed-reference inputs select a different plan.

This retains original routed recipes, selects the eleven existing dense
supplements, and passes other tensors through. The original plan is not mutated.
No published wire, archive, manifest, contract, runtime pin, or qualification
state was changed. GPU qualification remains with RobTand/prismaquant#2459.

## Regression and limits

`tests/test_pq2459_fixture_export.py` covers both `--device cpu` and
`--device cuda` in each of these paths:

- Real subprocess preflight activates the already committed source archive and
  uses its actual plan/recipe/authority parsers and safetensors shape reader on
  small generated source tensors. It must find eleven dense and three routed
  source tensors, qualify zero cells, and publish neither plan nor artifact.
- Backend-dispatch probes assert the complete supplemental plan, unchanged
  original plan, cache/producer/Hessian argument forwarding, CPU module versus
  CUDA progress-wrapper selection, and backend return-status propagation.
  Source activation and backend execution are mocked only in these probes.

These tests do **not** validate cached payload intake or perform an actual
fixture export. The `cuda` option tests preflight/command selection, not CUDA
encoding. No GPU workload, scorer, numerical measurement, or cell qualification
was performed. Existing published fixture geometry and hosted-gate properties
are covered by the unchanged packet, issue-reference, and runtime-pin tests.

## Pre-fix evidence

Command through `run_tests` on x86:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_pq2459_fixture_export.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

All four regression cases failed with:
`UnboundLocalError: cannot access local variable 'plan' where it is not associated with a value`.
The preflight subprocesses failed at `validate_serving_plan(plan)` (exporter line
151); the dispatch cases failed at plan serialization (line 170).
Population: NO CUDA, torch 2.11.0+cpu, zero skips/uncollected modules and zero
device allocations. Result: 4 failed in 14.46s.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1e049245cadcd6928d27d22c8dc89e07f1665fb531cc2f0b42d04d75282af8f2","size":18809}},"operation":"5ec1afbfc888bff218cf358041a82e23b0aed4da42e33f3c13ba008921a18cea","receipt":"1d6e70a0d772a939bd8551634c89a9552389998ffb46ed4282f1fbf3490d1afb"}]

## Post-fix evidence

Both arms ran this command through `run_tests` (default arm, then `arm="gpu"`):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_pq2459_fixture_export.py tests/test_pq2459_fixture_inputs.py tests/test_issue_refs.py tests/test_runtime_image_pin.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

The first post-fix x86 attempt reported 2 failed, 43 passed in 13.50s. Both
preflight commands succeeded, but the new test incorrectly parsed all stdout as
JSON despite the authority loader's existing log message. The test now parses
the final receipt line; no production logging behavior was changed. Population:
NO CUDA, torch 2.11.0+cpu, zero skips/uncollected modules and device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7fde0d7abe2897e80c88ab4b6e9ae29df421dcecd5bf768d9c36f353b951cf3e","size":7904}},"operation":"a8c61036d5d5a156146e09ca0c1363a3bb628f78dc8b2c692a66eef835b23fc0","receipt":"770451a899983368f79e6ba692f557339459c55f746c5da5c009f9c77938c74a"}]

Final results, with retained stdout inspected using `read_test_artifact`:

- x86: **45 passed in 15.65s**. NO CUDA, torch 2.11.0+cpu; zero skips,
  zero uncollected modules, zero tests allocating on the device.
- GPU arm: **45 passed in 4.80s**. CUDA torch 2.13.0+cu130, one NVIDIA GB10;
  zero skips, zero uncollected modules, zero tests allocating on the device.
  Device visibility is not evidence of GPU qualification.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"695d7c2b0c40e71b651fc50b822df450ac3562ffac0e77fca01e9879b6c86ce4","size":1588}},"operation":"9aeb6b0230cb621d2c1fa358bbc3849a9f9a42fcd25988b80ee09d095c64721c","receipt":"e32973a9da468f64bef144ec3205585f052b11af0c79d81167cc90f646e75107"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fe7cb64669d6775694c6881dfb17de1978e99f32b3189a418fa6605f0a0bfd81","size":1353}},"operation":"cb0ea77187da04ba402d608ae2f6782783e35366339190ccbc24c32519f5d87a","receipt":"5e6b5665aa381ce4f831e16f01a1698ea0a69da20422897a7fe317de27116f56"}]

After adding this evidence note and the changelog entry, the issue-reference
selection was rerun on both arms with the same pytest flags, selecting only
`tests/test_issue_refs.py`: x86 **3 passed in 7.68s**; GPU **3 passed in 2.52s**.
Each reported zero skips/uncollected modules and zero device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"435e589a7582e9042df2b55ac7de57442b6457e9543e76a55f278663dcb76705","size":664}},"operation":"ed2459620df098580b2356894787834319f8457fe2b2319e5c4b0f51289db588","receipt":"d35faccb75c590d446b0cc9c220f7c97fcd08c6723ee761142932eb6e1ae43d9"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"aaf2d235f02f9dbb875742bb038f0ddbe23e4b1c777046c55fc60d7e0f30861d","size":577}},"operation":"c92f3fcacaf00d515a14f528adc578a9868bb49ebe8020ceac71b933e9ebf8f4","receipt":"64cf53fcf5bb90f5c78b52dad08f6c82ec5ca0bce1e5ca10b3d0843965b7bddb"}]

The impacted-test selector was run in the worktree with
`python3 tools/impacted_tests.py --ref 07d0734440ca676376e94de1c51f31528f3db157 --json`.
At that point it saw the one tracked exporter change and returned `verdict: none`
(no reverse-reachable tests). The new, untracked regression file and the three
existing relevant gate files were explicitly selected above regardless.
The full merge-result suite is left to the coordinator; the byte-baseline audit
was not run because rendering, planes, and wire layout did not change.

## Hosted pure dependency follow-up

The release-batch failure occurred after the plan repair, at
`tests/test_pq2459_fixture_export.py`'s body-local `import torch` in both
preflight parameters. The hosted `pure` workflow deliberately installs neither
Torch nor the other tensor dependencies. Its import-time collection probe
cannot discover a dependency used only inside a test body.

The preflight test now uses function-local `pytest.importorskip` for Torch,
NumPy, and `safetensors.torch`. These are dependencies of the real sealed-source
preflight and its tiny checkpoint writer, not CUDA-availability conditions.
Both device parameters still execute the real parsers on dependency-equipped
CPU workers. The two pure backend-dispatch tests remain unconditional. The
existing Torch-free body regression now includes this module; no new harness
or source fixture was needed. Application code, packet, archive, wire bytes,
runtime pins, and the zero-qualified-cells state remain unchanged.

### Pre-fix regression

With the module added to the existing body-probe roster, before changing its
imports, the following x86 command failed:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_collection_probe.py::test_a_collectable_module_does_not_reach_torch_inside_a_test_body -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

The guard failed at `assert proc.returncode == 0`: its Torch-blocked child
reported both `test_cached_routed_preflight_uses_sealed_parsers[cpu]` and
`[cuda]` as failures, with **2 failed, 81 passed, 15 skipped**. The outer
regression reported **1 failed in 28.58s**, NO CUDA (torch 2.11.0+cpu), zero
outer skips/uncollected modules and zero device allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3a436761243c258262211f7797b7d85e61993d006ad73888ec67fe63d1179f97","size":3269}},"operation":"0d3d4bd51bbc2192b06d32e56c5d1ec1eb6f983bcf5172f6d2dc1e030956e1c6","receipt":"d81bb1a57603e3d4c2b5065c45c302b0f3e306af4d297f6d45daf3ec495cabb5"}]

### Post-fix x86 checks

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_pq2459_fixture_export.py tests/test_collection_probe.py tests/test_pq2459_fixture_inputs.py tests/test_issue_refs.py tests/test_runtime_image_pin.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Result: **261 passed in 168.14s**. NO CUDA, torch 2.11.0+cpu, zero skips,
zero uncollected modules and zero device allocations. This includes the real
sealed-parser preflights as well as the now-passing Torch-free child probe.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"19c5120a91c0cbee512ca51d1869644174fe6d91a358f863031e4de290f77b18","size":1855}},"operation":"b3ae177d9347853a831f0d9dd9c565d4b026eaf41f178efa02dced17fa4d24ba","receipt":"22b65091860a6a87a00f640f3f7728878afb0631b439297822153fab7c33da3c"}]

A direct Torch-blocked run makes the retained pure coverage explicit:

```sh
PYTHONPATH=src TESSERA_COLLECTION_PROBE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -c 'import sys; sys.modules["torch"] = None; import pytest; raise SystemExit(pytest.main(sys.argv[1:]))' tests/test_pq2459_fixture_export.py -q -rs -p no:xdist -p no:cacheprovider --durations=10 --basetemp="$TMPDIR/pytest"
```

Result: **2 passed, 2 skipped in 1.11s**; all four cases collected. Both
preflight cases skipped with the verbatim reason
`could not import 'torch': import of torch halted; None in sys.modules`.
Both dispatch cases passed. NO CUDA, no Torch in this child interpreter, zero
uncollected modules and zero device allocations. The import block must stay in
this process, so this small diagnostic runs without xdist. This reproduces the
missing-Torch condition, not a full hosted-CI environment or hosted workflow run.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3c28683b1451a334fe88db55f24797d6ba900bb9f361a573ca78cea91042b1f4","size":1331}},"operation":"7ef9a10c530463d32ae9d0eec4502416a125a9777731ddd64769442955cd3a38","receipt":"ec6b1ec10afe30a6f80e37ea519a157736befa8245d1e43e84fe910672c11664"}]

### GPU-arm validation plan (recorded before execution)

Decision: ensure the function-local guards still run both real parser cases,
rather than skipping, with the GPU arm's Python 3.12 and CUDA Torch packages.
The x86 Python 3.14/CPU Torch environment cannot establish that dependency
combination. The smallest check is the four cases in
`tests/test_pq2459_fixture_export.py`, with two workstealing workers, native
threads limited to one, and temporary files under the action's `$TMPDIR`.
Expected GPU-arm occupation: less than one minute; expected device allocations:
zero. No encoding, engine, scorer, numerical qualification or sweep is planned.

### GPU-arm result and scope of verification

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_pq2459_fixture_export.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Result on `arm="gpu"`: **4 passed in 4.53s**. CUDA Torch 2.13.0+cu130,
NVIDIA GB10 visible, zero skips/uncollected modules and zero device allocations.
Both parser cases executed. This is interpreter/dependency and parser coverage,
not GPU numerical coverage or qualification.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ed4a39a6bf95f2d0825612579fa0a9ee2caf71cda4ced997fd32ff8d081ef885","size":701}},"operation":"3a7e84114d02619ad9cecfd7ae222729831cfb7a489417c490fcb844f5a42e6f","receipt":"c6c0083b23817bc73450cc753147e82df16d904c789a8786219d26f3470e3789"}]

Measured source: base `c8d94b943ec42add0db470b2d8bdba52300186a8` plus
this four-file follow-up (two tests, this appended evidence, and CHANGELOG).
The selector command
`python3 tools/impacted_tests.py --ref c8d94b943ec42add0db470b2d8bdba52300186a8 --json`
returned `narrowed`, four changed files, 420 selected test files and no
`forces_full`; both modified tests were selected. Its unresolved-loader and
unplaced-read conservatism also selected many unrelated consumers. That entire
broad selection was not rerun for this test-only dependency correction; the
focused checks above were run, and the full merge-result/hosted suite remains
with the coordinator. `git diff --check` passed. No claim is made that hosted
CI itself has rerun, or that parent workload/scorer or GPU qualification ran.

After the documentation edits, the x86 guards were rechecked with
`PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_runtime_image_pin.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"`:
**40 passed in 12.08s**, NO CUDA, zero skips/uncollected modules and zero device
allocations.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"bd3e6ef57734ac7f897a648802a6ea89b633acc954598218e790d0ec7b97f2f0","size":1580}},"operation":"67c7cd424c0686bb4cf6a594bd928f91de70713c4a9975f76966f6aa1913308b","receipt":"0f5e39b40201fc3d5974964aaad49b300081fa3a1cecf0e8fc357b673b11aae9"}]
