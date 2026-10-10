# Tessera#1171: full CUDA population on the merged tree of PR 1089

Date: 2026-10-10. Author: issuegraph attempt 2. Status: measured.

## Tree measured

The GPU run used checkout `b2875875a40d2b594e99f43b63cfe503d73ec2e7`.
That tree contains merge `36ca1dbd76e3e375d20be092d117e10bed644c64`.
The working tree was clean. The run is a post-merge check.
It does not gate the merge.

PrismaBuild sealed the source as snapshot
`31525413d1628a1b826123b9902ce934b361a62a`.
The effective source hash is
`f6ec4e79296ed8eaafb0ad3d1ce6fdcf67f86ec5dcf83ef9ba32579ad72f0bb8`.
The receipt shows `commits_measured.effective_source.agree` is true.
One arm ran, so agreement is direct.

## Population

Receipt: `/mnt/shared/tessera-suite-receipts/20261010T021500-ts1171-gpu/receipt.json`.
Pool action: `697d537739bdd711d26e052bf3129064b74293939166ae03d536a24fe9afd094`.
Host: `sparklina`. Exit code: 1 (25 red tests, see below).

- Device: `torch 2.13.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10`.
- Strict CUDA: true.
- Passed: 11733. Failed: 13. Error: 12. Skipped: 36.
- Xfailed: 9. Xpassed: 0.
- `cuda_surface.executed`: 2139 (above zero).
- `cuda_surface.box_artifact_skips`: empty (no test skipped for a absent box artifact).
- `not_collected`: empty (zero uncollected modules).

## Skip reasons, verbatim (36 total)

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

No reason cites a absent box artifact. The 36 count exceeds the 13 of
`d11dc01` because the tree grew new PB-gated tests since that commit.

## Failures: all 25 are baselines, zero regressions of PR 1089

The full-run log names 13 failed and 12 error tests. Each group ran
again, file-alone, on pristine master and on the measured commit.
Pristine master seals as effective sha `adf741ef` (2123 files).
The measured commit seals as effective sha `f3b55c93` (2118 files).
Every group fails the same on both trees. Each failure is a baseline.

| Group | Target | Master key | Measured key | Master | Measured | Verdict |
|---|---|---|---|---|---|---|
| a | `tests/test_serving_mla_mask_runtime_gpu.py` | `57f5e86f` | `3e0eac2e` | 1 error | 1 error | baseline |
| b | `tests/test_serving_mla_mask_registration.py` | `11adaef3` | `e0923f4c` | 4 failed, 11 error | 4 failed, 11 error | baseline |
| c | `test_frozen_2db_flags_reach_different_runtime_selectors` | `08135741` | `329cb516` | 1 failed | 1 failed | baseline |
| d | `tests/test_mhc_fusion_cuda.py` | `aa5c4bf9` | `1b63985f` | 8 failed | 8 failed | baseline |

Full keys: `57f5e86f1362847db95d162d23c9850e7eab53b786c0bf896a38c8a74e153865`,
`3e0eac2ea91f4931a7eb54b86bcd0a449684d499bc381812008ae8a7a00bbc97`,
`11adaef3d7d3a20e6ec990519609496928c1c5e9ffce4ae87e8e630c70258c14`,
`e0923f4cc141d71296f67f07f8150adfabcedf75c1891b9d5e5580923ff09c0`,
`0813574104607a1f241a59cdd568d4022bbbafb263d69975a2397c139c8d9184`,
`329cb51695acda05b88be5282e5c420860e009e4fde15fcb6d87c3e4e46bc57b`,
`aa5c4bf9da087862a7c9184481c688cd30aa90f524aa3b12ec1a41503457a215`,
`1b63985f476d7aaa8b30b384d422e075bd0919de0867ef47b64c91f8af20ecaa`.

The 25 red test IDs (from action `697d5377` stdout):

- `tests/test_serving_mla_mask_registration.py::test_explicit_mask_skip_flag_registers_stock_enum` (failed)
- `tests/test_serving_mla_mask_registration.py::test_registration_default_off_does_not_replace_stock` (failed)
- `tests/test_serving_mla_mask_registration.py::test_registration_refuses_competing_override` (failed)
- `tests/test_serving_mla_mask_registration.py::test_native_factory_selects_pass_buffers_once_per_device` (failed)
- `tests/test_graph_attest_eager_levers.py::test_frozen_2db_flags_reach_different_runtime_selectors` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[float32-33-33]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[float32-1024-2048]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[float32-2048-2048]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[float32-2049-2049]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[tf32_halfway-33-33]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[tf32_halfway-1024-2048]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[tf32_halfway-2048-2048]` (failed)
- `tests/test_mhc_fusion_cuda.py::test_fused_equals_stock_bitwise[tf32_halfway-2049-2049]` (failed)
- `tests/test_serving_mla_mask_runtime_gpu.py` (collection error, ImportError)
- `tests/test_serving_mla_mask_registration.py::test_decode_mixed_or_capture_never_enters_native[False-False]` (error)
- `tests/test_serving_mla_mask_registration.py::test_decode_mixed_or_capture_never_enters_native[True-True]` (error)
- `tests/test_serving_mla_mask_registration.py::test_unknown_stock_source_stays_stock` (error)
- `tests/test_serving_mla_mask_registration.py::test_stock_calibrated_non_mg_plan_never_enters_native` (error)
- `tests/test_serving_mla_mask_registration.py::test_prefill_scope_comes_from_metadata_and_is_restored[0-0-1-True-True]` (error)
- `tests/test_serving_mla_mask_registration.py::test_prefill_scope_comes_from_metadata_and_is_restored[1-1-1-True-False]` (error)
- `tests/test_serving_mla_mask_registration.py::test_prefill_scope_comes_from_metadata_and_is_restored[1-2048-0-False-False]` (error)
- `tests/test_serving_mla_mask_registration.py::test_prefill_scope_comes_from_metadata_and_is_restored[0-0-0-False-False]` (error)
- `tests/test_serving_mla_mask_registration.py::test_compiled_forward_keeps_stock_scope_without_gpu_queries` (error)
- `tests/test_serving_mla_mask_registration.py::test_dispatch_identifies_selected_schedule[None-mg_mask_skip_pass_buffers]` (error)
- `tests/test_serving_mla_mask_registration.py::test_dispatch_identifies_selected_schedule[decode-stock_mg]` (error)

Rerun logs show the same IDs on both trees. No failure is a regression
of PR 1089. The owning lanes carry these reds on pristine master.

## Endpoint witness path against its timeout

Largest model the population loads: the routed A4 TP2 checkpoint
`merged-4c384e60` from `glm-canonical-census-20260908`.
Largest single shard: `part-00045-model-00104-of-00120.safetensors`
(5,368,754,216 bytes on disk; 5,368,709,120 resident tensor bytes).
The a4, a8 and a16 bundles peak at this same size.
The largest packed-expert shard (`Qwen3.8-Flash-Next`
`model-00062-of-00131`) is 3,510,240,000 bytes.
No population test resident-loads a full bundle.

The probe (`tools/probe_witness_timing.py`) calls
`tessera.serving.endpoint_runtime.resident_bytes`. That is the exact
function `observe_worker` calls per request
(`endpoint_runtime.py:286`). The 30 s bound is the `fetch_witness`
timeout (`endpoint_observer.py:23`, `timeout_s=30`).
The probe places the full shard on CUDA and hashes each resident byte.

- Prior probe, action `673056b0b6dce35fd1928849895561518bbb6618c53e6a961d3675b33b5bb601` (sparklina): 2.9 s, within timeout.
- New probe, action `782c049c81abb3b00a25c046fe678311e6aa2433fd3b1d7df4db6fc474bc2529` (sparky): `PROBE model=merged-4c384e60 shard=part-00045-model-00104-of-00120 resident_bytes=5368709120 file_bytes=5368754216 elapsed_s=2.5 timeout_s=30.0 within=True`, exit 0.

The endpoint witness path completes far inside its timeout on the
largest model the population loads.

## Action keys

- Full GPU population: `697d537739bdd711d26e052bf3129064b74293939166ae03d536a24fe9afd094` (red, 25 baselines).
- Reruns: the 8 keys in the table above (all reproduce, all baseline).
- CPU preflight: `83402970722b71f9c324d0aad8a8d99352593f49754917c49fd56b10a163e075` (pass, dl380g10).
- Prior probe: `673056b0b6dce35fd1928849895561518bbb6618c53e6a961d3675b33b5bb601` (pass, 2.9 s).
- New probe: `782c049c81abb3b00a25c046fe678311e6aa2433fd3b1d7df4db6fc474bc2529` (pass, 2.5 s).
