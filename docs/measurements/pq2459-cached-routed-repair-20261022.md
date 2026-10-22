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
