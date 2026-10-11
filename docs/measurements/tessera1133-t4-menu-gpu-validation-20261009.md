# tessera#1133 GPU validation of the T4 performant menu (2026-10-09)

Parent issue: tessera#1103, "Publish the canonical TESSERA_E2M1_K2 performant menu".
This receipt records one validation run through PrismaBuild.
It adds no performance evidence and no serving evidence.
The machine-readable copy holds the verbatim controller populations:
`tessera1133-t4-menu-gpu-validation-20261009.json`.

## Result

- The policy checks pass on x86 and on a GB10 worker.
  The dense and routed menus stay `[896]`.
  The 20 approved cells allow admission.
  A foreign build waits.
- The native execution checks pass on a GB10 worker (compute capability 12.1).
  All 21 selected native tests passed.
  The whole GPU run is 49 passed, 0 failed, 0 skipped and 0 modules not collected.
  20 tests allocated on the device under `--strict-cuda`.
- The native checks pass in container image
  `localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b`.
  That image carries `tokenspeed_triton` 3.8.10. Its Triton lowers block-scaled FP4 MMA.
- The same native selectors error in the legacy pool GPU venv `prismaquant-cu130`.
  An inventory probe lists 72 host venvs. 68 carry stock `triton` 3.6.0 and none carries `tokenspeed_triton`.
  That is an environment gap. The code under test is not the cause, because the same source passes in the image.
  The earlier attempts of this item stopped at that gap.
  The container image removes it without any repository change.

## What each leg covers

| Leg | Selector | What it shows | What it does not show |
|-----|----------|---------------|-----------------------|
| Metadata | `tests/test_rung_performant_policy.py`, 28 tests | The menu rule, the admission rule and the wait rule, on fixture tables built from the approved scope | The execution of any cell |
| CPU decode | `test_code_nibbles_match_subset_semantics`, `test_plane_decode_matches_materialize_stock`, 6 tests | The archived wires parse and decode on the CPU | Anything on the device |
| Native execution | `test_native_fp4_mma_is_emitted_not_emulated`, `test_gemm_matches_arithmetic_reference`, `test_grouped_gemm_matches_the_dense_kernel`, 21 tests | The Triton kernels run on the GB10 FP4 MMA instruction and agree with their references | Performance, serving, and the shapes of the approved cells |

The native tests check the following:

- `test_native_fp4_mma_is_emitted_not_emulated` (1 test).
  A probe kernel compiles for the device and returns the known answer K.
  Its PTX carries `mxf4nvf4` and `mma.sync`.
- `test_gemm_matches_arithmetic_reference` (16 tests: M in 1, 8, 32, 128; groups w13 and w2; ranks 0 and 1).
  The dense A4 span-2 GEMM matches the exact fp64 product of the decoded codes and scales.
  The test bound is a relative error below 2e-3.
- `test_grouped_gemm_matches_the_dense_kernel` (4 tests: groups w13 and w2; ranks 0 and 1).
  The grouped kernel matches the dense kernel row by row.
  The w13 stack has two experts. Its dispatch repeats an expert id, and a second dispatch leaves an expert empty.
  The w2 stack has one expert.
  The test bound is a relative error below 1e-6.

The wires are the archived fixtures at `/mnt/shared/astra-native-a4/data`.
They declare family `TESSERA_NVFP4`, grid `E2M1x2`, body `TCQ` and `q256` 896.
This run does not claim that their shapes equal the shapes of the approved cells.

## Source

Every validation action of this attempt ran master `69aae020048458a168296ef534ad29938299cf83`.
That commit is the merge of PR 1131.
Each action was submitted from a clean worktree, so the snapshot has no local delta.
PrismaBuild seals that commit as the snapshot parent.
The raw snapshot commits differ, because each action carries its own closure stamp.
A shared parent does not prove equal source.
So the table below lists the effective source hash that each controller population reports.

| Population | Snapshot commit | Effective source sha256 | Files verified |
|------------|-----------------|-------------------------|----------------|
| CPU, `4aa13e926f4e` | `d676252b60edf65bc725a4a17361bc8a59bfdb18` | `4d068a285daa1e8817577c363c3595bb7f27ff828903eb3d6dce2ada6658ba69` | 2116 |
| GPU slice, `4311cf8975c0` | `4d1cd05011d37c84a47ea567fc05a0bb45b490b3` | `4d068a285daa1e8817577c363c3595bb7f27ff828903eb3d6dce2ada6658ba69` | 2116 |
| GPU native, `d65fa0ffe74a` | `38298605dbdf7ea0c862458e8a534070c79a0b49` | `4d068a285daa1e8817577c363c3595bb7f27ff828903eb3d6dce2ada6658ba69` | 2116 |
| CPU, superseded, `8241a594d333` | `258220fc655888467fc9966c5f12bd8d1328c957` | `5277b4dfd6aed2da08aa423af65074c12dd8eb458e3325f167edb32678c88d60` | 2117 |

The three verified populations agree on one effective source.
The superseded CPU action declared no source verifier.
It counted the PrismaBuild closure stamp as a source file, so its hash differs.
Its results are the same, and the CPU row above replaces it.

## Actions

All seven actions ran at priority 10, with one attempt each.
The five validation actions of this attempt were published from celestia.
The times below are UTC on 2026-10-09. They run from publication to finish.

| Role | Key | Host | Device | Elapsed | PB status / exit | Result |
|------|-----|------|--------|---------|------------------|--------|
| Metadata and decode, CPU | `4aa13e926f4e` | dl380g10 | none | 27.9 s | executed / 0 | 34 passed in 18.02 s; 21 native selectors collected |
| Metadata and native, GPU | `d65fa0ffe74a` | sparklina | GB10 | 14.3 s | executed / 0 | 49 passed, 18 subtests passed in 9.83 s; 20 allocated |
| Runtime identity probe | `2230cd9647df` | sparklina | GB10 | 2.6 s | executed / 0 | backend `tokenspeed_triton`; PTX tokens `mxf4nvf4`, `mma.sync` |
| Two-test slice, GPU | `4311cf8975c0` | sparky | GB10 | 10.2 s | failed / 4 | 2 passed; `--strict-cuda` refused the run (see below) |
| CPU, superseded | `8241a594d333` | dl380g10 | none | 20.8 s | executed / 0 | 34 passed in 14.39 s |
| Pool venv, earlier attempt | `15685e3628e2` | sparky | GB10 | 5.7 s | failed / 1 | 28 passed, 21 errors; 0 allocated |
| Pool venv inventory, earlier attempt | `3b59b3bca1ed` | sparky | GB10 | 15.6 s | executed / 0 | 68 of 72 host venvs list `triton` 3.6.0; none lists `tokenspeed_triton` |

Full action keys:

- `4aa13e926f4efb725ceecc667c29deec9b0f2f34fd8c2e18293e73dbcbadd463` (CPU, 22:16:10 to 22:16:46)
- `d65fa0ffe74a538ce710a1a88ef6498c10541fe2c26934cca30f52cd9bbe65d9` (GPU native, 22:14:07 to 22:14:26)
- `2230cd9647dfcbf5e43d1dbbd852833f4190e6ec418d5bc516d989510fc00c34` (identity probe, 22:16:15 to 22:16:22)
- `4311cf8975c0c6f48a85dfafc596e77ec70015364f37ab6c63b805b1e5fda9be` (two-test slice, 22:13:22 to 22:13:38)
- `8241a594d333c291370881c83d6adfcb11baac49322df56c01009e10f78af9ca` (CPU, superseded, 22:11:59 to 22:12:23)
- `15685e3628e24603a3aae647679fcf8c8941aa9313633f916d9acc5b1a3715d3` (pool venv, 21:48:42 to 21:48:57)
- `3b59b3bca1ed7c48a5aaf1bc90cfa4445f3e048e623c3e167a9b4ecd09c9e00f` (pool venv inventory, 21:58:16 to 21:58:39)

The CAS receipts of the four executed validation actions of this attempt:

| Action | Receipt sha256 |
|--------|----------------|
| `4aa13e926f4e` | `732162e03fe6870ed7f61f17dddbcd900292af3dd2b0ad5c017f99dd25a8dc03` |
| `d65fa0ffe74a` | `505043a05401d84dbb9f60e42361cb19de881095d9d975326ac796cd57210200` |
| `2230cd9647df` | `7fea33fa9e4973e49268cbfb0477adb49e5cf865e1b141c9ddd89a7da7752c37` |
| `8241a594d333` (superseded) | `53d31efa5f6ceb2285459d05094ce3e7960b7631a73742d728c451d209bad248` |

The failed two-test slice has no CAS receipt.
PrismaBuild published none for it.

## Metadata checks

The CPU action printed this readout from the policy fixtures.
It reads the values that the tests assert.

```
MENU_READOUT {"approved_cell_count": 20, "approved_cell_statuses": ["allow"], "approved_status": "allow", "foreign_build_reasons": ["performance_admission_not_established"], "foreign_build_status": "wait", "menu_TESSERA_E2M1_K2": {"dense": [896], "routed": [896]}, "qualified_menu": {"dense": [896], "routed": [896]}, "rung_768_first_cell_reason": "performance_admission_not_established"}
```

| Acceptance point | Value read | Tests that assert it |
|------------------|------------|----------------------|
| Dense menu stays `[896]` | `performant_rungs("TESSERA_E2M1_K2", "dense")` is `(896,)` | `PerformantPolicy::test_menu_is_structure_specific` |
| Routed menu stays `[896]` | `performant_rungs("TESSERA_E2M1_K2", "routed")` is `(896,)` | `PerformantPolicy::test_menu_is_structure_specific` |
| Published menu | `qualified_menu` is `{dense: [896], routed: [896]}` | `TimingPublication::test_approved_scope_admits_exact_cells_and_waits_elsewhere` |
| Approved cells allow | status `allow` for all 20 cells at rung 896 | the same test; `test_approved_cell_records_bind_M_shape_and_kind` |
| Foreign scope waits | build `other-build`: status `wait`, reason `performance_admission_not_established` | the same test |
| Other rung waits | rung 768: reason `performance_admission_not_established` | the same test |

All 28 policy tests passed in the CPU action and in the GPU action.
The GPU action ran them in a population that has a CUDA device.
The 20 approved cells are rows of a fixture table that the test builds from `E2M1_K2_PERFORMANT_MENU`.
No test in this run executes those cells.

## Native execution coverage

Runtime identity, read by action `2230cd9647df` in the same image on sparklina:

```
IMAGE_INSPECT sha256:38ead3cab558d85c2261c43dccb76207f5b25c639281cf9a24714455f14a9dc4 ["localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b"] created=2026-09-28T08:25:54.65219962-04:00
IDENTITY {"backend": "tokenspeed_triton", "backend_module_version": "3.8.10", "capability": [12, 1], "device": "NVIDIA GB10", "jsonschema": "4.26.0", "numpy": "2.3.5", "ptx_tokens": ["mxf4nvf4", "mma.sync"], "python": "3.12.3", "tokenspeed-triton": "3.8.10.post20260721", "tokenspeed_triton": "3.8.10.post20260721", "torch": "2.13.0+cu130", "torch_cuda": "13.0", "triton": "3.7.1", "vllm": "0.28.1rc1.dev397+gfd4a15126.d20260904"}
```

- The probe calls `tessera.kernel_a4.native_fp4_backend()` and `native_fp4_mma_ptx_tokens()` on the checked source.
  The backend is `tokenspeed_triton`, the first build in the preference order.
  The PTX has `mxf4nvf4` and `mma.sync`. The token `fma.rn.f32` is absent.
- The device is an NVIDIA GB10, compute capability 12.1, driver 595.99.02.
  The driver value comes from the PrismaBuild attestation of action `d65fa0ffe74a`.
- The native action has no print of the backend.
  [INFERENCE] It used `tokenspeed_triton` too, because the image and the import order are the same.
- The scoped test runner is pytest 9.0.3 with pytest-xdist 3.8.0.
  Its dependency site is `/mnt/shared/tessera-suite-envs/ts-readiness-20261002-py312/site-packages`.
  Its content seal is `2bacba69864fad0e86481f94e6d3ea157e9d9ddab24e7f6efb3c72692658fdf6`.
- The CPU action ran with Python 3.14.4, torch 2.11.0+cpu, numpy 2.5.2, pytest 9.1.1,
  pytest-xdist 3.8.0 and jsonschema 4.26.0. vLLM is absent there.
- The two Sparks both report this image to PrismaBuild.
  The two-test slice passed its native probe test on sparky.
  The full native selection ran on sparklina.

The fixture wires and their sha256:

| File | Bytes | sha256 |
|------|-------|--------|
| `a4-config.json` | 140563 | `b449cd6246c60f7892f95794c442acb395cb67813530bca8b734323c9a74812d` |
| `gate_proj_wire.bin` | 4195630 | `2e87c5101b034e1dd4124b53c2f93ae1dcce230f1cb3d153d764c9b481f9cb16` |
| `up_proj_wire.bin` | 4195628 | `dcdcf51adf82a23174678c3266c4754cad876a46d728402392abbb084a1e7282` |
| `down_proj_wire.bin` | 4195583 | `e8f05c06ebd3ee5ee873121635edc852ec58e57e6cf9155f16c68b53fc548fa4` |

The CPU decode tests parse and verify the full containers.
That is the integrity path of the loader.

## CUDA allocation, skips and collection

The controller wrote one population per validation pytest action.
Each file lives in `/mnt/shared/tessera-suite-receipts/ts1133-20261009/`.

| Population | sha256 | Device | `--strict-cuda` | Passed | Failed | Error | Skipped | Not collected | Tests that allocated |
|------------|--------|--------|-----------------|--------|--------|-------|---------|---------------|----------------------|
| `surface.cpu-verified.json` | `705c65ac1d0eec81e3242b372ae3318b1c5940914a26730cfb35770a198f93b0` | torch 2.11.0+cpu, no CUDA device | no | 34 | 0 | 0 | 0 | 0 | 0 |
| `surface.gpu-full.json` | `9e20de95f61c921680f3084ec5697bceb7beb02d09fe1d8f8d5ee9a662f16053` | torch 2.13.0+cu130, 1 CUDA device, device 0 = NVIDIA GB10 | yes | 49 | 0 | 0 | 0 | 0 | 20 |
| `surface.gpu-pre.json` | `0ee8e7decf99524048908c74681b8590be596e693d1163e238c93637852d96d5` | torch 2.13.0+cu130, 1 CUDA device, device 0 = NVIDIA GB10 | yes | 2 | 0 | 0 | 0 | 0 | 0 |
| `surface.cpu.json` (superseded) | `b977ffcd16fc7c751c29fce780428d43daeff6850108eda837046209a8adef90` | torch 2.11.0+cpu, no CUDA device | no | 34 | 0 | 0 | 0 | 0 | 0 |

- Every population has `skip_reasons` equal to `{}`. No test skipped, so there is no skip reason to quote.
- Every population has `not_collected` equal to `[]`.
  `box_artifact_skips` is `{}`, so no test skipped for a missing checkpoint or serve log.
- The CPU action collected the 21 native selectors and did not run them.
  Its log says that the run did not exercise the CUDA-gated surface.
- `cuda_surface.executed` counts the tests during whose call torch's allocator recorded a new allocation.
  It is a floor.
- The population does not name the 20 allocating tests.
  The two worker shares report 10 each.
  [INFERENCE] They are the 16 `test_gemm_matches_arithmetic_reference` tests
  and the 4 `test_grouped_gemm_matches_the_dense_kernel` tests.
  The two-test slice shows that the backend test and the probe test allocate nothing in their call.
  The probe test allocates in its fixture setup. The policy tests use no device code.
- The two-test slice selected the backend report and the probe test only.
  Neither test allocates in its call. So `--strict-cuda` refused the run, as designed:
  `--strict-cuda: no test in this run allocated on the CUDA device. A device this session never used is not coverage of the surface it was submitted to cover -- every other check here is satisfied by a run that collected the suite and skipped all of it (tessera#152).`
  The pytest summary of that slice is `2 passed in 3.79s`.
  PrismaBuild records the exit status 4 and the state `failed`.
  The slice is a preflight, not a validation result.

## The pool venv refusal

Earlier attempts of this item ran the same selection in the legacy GPU venv
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python` on sparky.
Action `15685e3628e2` records the result: `28 passed, 21 errors in 3.25s`, and `0 test(s) allocated on the device`.
The 28 passes are the policy tests. The 21 errors are the native tests.
Each error is a fixture error with this message:

```
tessera.errors.GrammarError: test_kernel_a4: triton cannot compile or run block-scaled FP4 MMA for this device (RuntimeError: PassManager::run failed); the A4 span-2 native kernels refuse rather than emulating the multiply in bf16
```

The log shows an MLIR assertion in the Triton pass `TritonGPUAccelerateMatmul`
(`DenseElementsAttr::get ... isIntOrIndex`).
The module under that pass holds `tt.dot_scaled ... lhs = e2m1 rhs = e2m1`.
The code refuses by name and does not emulate the multiply.
That is the designed behavior of `require_native_fp4_mma`.

Action `3b59b3bca1ed` listed 72 host venvs on sparky.
68 of them list `triton` 3.6.0. Four list no Triton. None lists another Triton or `tokenspeed_triton`.
That probe read host paths only.
It cannot see inside a Docker image, so it could not see the image of this receipt.

The reading is as follows.
The legacy pool venv cannot cover the native A4 surface.
The container GPU arm of `tools/merge_suite.py` (`--gpu-image`) can.
That arm runs `tools/suite_container.py`, and this run used that launcher directly.
A Triton build for the pool venvs that lowers `tt.dot_scaled` for sm_121 is follow-on fleet work.
This validation does not wait for it.

## Commands

The native action, as sealed:

```
/usr/bin/python3 tools/suite_deadline.py --timeout-s 1800 --kill-after-s 5.0 -- /usr/bin/python3 tools/suite_container.py --image localhost/prismaquant/spark-vllm-nccl230@sha256:c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b --deps-site /mnt/shared/tessera-suite-envs/ts-readiness-20261002-py312/site-packages --deps-sha256 2bacba69864fad0e86481f94e6d3ea157e9d9ddab24e7f6efb3c72692658fdf6 --surface-dir /mnt/shared/tessera-suite-receipts/ts1133-20261009 --cache-dir /home/rob/tmp/tessera-suite-cache/ts1133-full-20261009 --data-root /mnt/shared/prismabuild-fleet --artifact-root TESSERA_A4_WIRE_DIR=/mnt/shared/astra-native-a4/data -- /usr/bin/python3 -m pytest tests/test_rung_performant_policy.py tests/test_kernel_a4.py::test_native_fp4_mma_is_emitted_not_emulated tests/test_kernel_a4.py::test_gemm_matches_arithmetic_reference tests/test_kernel_a4.py::test_grouped_gemm_matches_the_dense_kernel -v -p no:cacheprovider --surface-json /mnt/shared/tessera-suite-receipts/ts1133-20261009/surface.gpu-full.json -n 2 --dist worksteal -p xdist.plugin --strict-cuda --durations=10
```

The submission used `pbrun.py --gpu --tag gb10 --cpus 2 --demand mem_gb=16 --priority 10 --timeout-s 1800`
with `--container-image` set to the image above.
It declared `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MAX_JOBS=1`, `PYTHONPATH=src`,
`TESSERA_A4_WIRE_DIR=/mnt/shared/astra-native-a4/data` and `TESSERA_SOURCE_VERIFIER`.
The scope memory peak was 4.47 GiB. The two-test slice peaked at 1.93 GiB.
The 16 GB reservation fits both peaks.

The CPU action ran a shell script with four legs.
The exact script is in the JSON copy.
The pytest legs are:

```
/home/rob/venvs/pb-cpu/bin/python -m pytest tests/test_rung_performant_policy.py tests/test_kernel_a4.py::test_code_nibbles_match_subset_semantics tests/test_kernel_a4.py::test_plane_decode_matches_materialize_stock -n 2 --dist worksteal --durations=10 -ra -p no:cacheprovider --surface-json /mnt/shared/tessera-suite-receipts/ts1133-20261009/surface.cpu-verified.json
/home/rob/venvs/pb-cpu/bin/python -m pytest tests/test_kernel_a4.py::test_native_fp4_mma_is_emitted_not_emulated tests/test_kernel_a4.py::test_gemm_matches_arithmetic_reference tests/test_kernel_a4.py::test_grouped_gemm_matches_the_dense_kernel --collect-only -q -p no:cacheprovider
```

It was submitted with `--tag x86 --cpus 2 --demand mem_gb=8 --priority 10 --timeout-s 1800`.

## Deviations from the plan

- **Priority.** The acceptance text says priority 0. The plan and the standing order fix priority 10 for this item.
  Issue tessera#1133 carries the `campaign` label, and the campaign's GPU work runs at 10 (Rob, 10-09).
  The action key does not include the priority, so priority 0 would run the same actions.
- **Interpreter and launcher.**
  The plan names `python -m pytest`. The native action runs `/usr/bin/python3 -m pytest` inside the image.
  It uses the repository launcher `tools/suite_container.py`, which seals the image and the dependency site.
- **Report options.**
  The launcher grammar refuses `-ra` and `--junitxml`.
  The native action used `-v`. The per-test PASSED lines are in the action log.
  The controller population holds the counts and the skip reasons.
- **No new entry point.** The run changes no test, no tool and no source.
  Two helper scripts ran as action text only: the menu readout and the identity probe.
  Both read values and change no state. The JSON copy holds their text.
- **Superseded CPU action.** Action `8241a594d333` ran first and declared no source verifier.
  Action `4aa13e926f4e` repeated it with the verifier, so that all populations share one source hash.
- **The pool CPU venv.** The `pb-cpu` venv on dl380g10 had `jsonschema` 4.26.0 at run time.
  Two earlier CPU actions failed with `ModuleNotFoundError: No module named 'jsonschema'` (tessera#1042):
  `d5865e694fce997a8ef0e26c9f269640806e24ccffb7a332521be843f6ffa32a` and
  `d370c65485d246fdc8d73c7a4349bba45cdc7bf63fedb678ba95f3904ef72f67`.
  The next action, `62f9a62e6f3f2f1caeeffa85e517fd8a7f1bd4ba21336f21b62a09dae95f4af3`,
  ran `pip install jsonschema` in that shared venv and passed 28 policy tests.
  [INFERENCE] The CPU leg passes today because of that side effect, not because of a reviewed provisioning change.

## Not claimed

- No performance claim. The run makes no timing comparison.
  The elapsed times above are action durations, not kernel measurements.
- No serving claim. The image is a PrismaBuild-local image.
  `docs/ARCHITECTURE.md` pins it for the NoPE graph runner, and the 2026-10-04 container-arm receipts used it.
  It is not `versions.default_serve_image` of the runtime contract.
  This run does not attest what vLLM executes in the contract's serve image.
  tessera#938 tracks that no single sealed image runs the whole strict-CUDA suite.
- No claim about the approved performance population.
  The 20 approved cells were not executed.
  The native shapes come from the archived wires.
- No claim for other Triton builds, other devices or the second Spark beyond the two-test slice.
- The native checks use fixture wires from layer 3 of one model.
  They show that the kernels agree with their references on those inputs.
