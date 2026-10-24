# Current-source BF16 dense census — uncompleted (artifact prerequisite absent)

## Pre-scheduling reuse decision

Inspected checkout `6793b979523e84bf046595adf7449f00c6973330` before scheduling any serve. A focused refresh of shared evidence for https://github.com/RobTand/tessera/issues/1380, https://github.com/RobTand/tessera/issues/1367 and the related https://github.com/RobTand/tessera/issues/1492, https://github.com/RobTand/tessera/issues/1458, https://github.com/RobTand/tessera/issues/1459, https://github.com/RobTand/tessera/issues/1482 scope did not establish a matching current raw-BF16 dense served receipt. This is a bounded search result, not proof that none exists elsewhere.

Inspected under `/mnt/shared/tessera-measurements/`:

- `eng-t16-epilogue-production-20261007/corrective-review-2-projection/finite-eager/result.json` and `finite-graph/result.json`: single-rank device-forward projection proofs; the eager receipt includes the intended raw pair but runtime source identity is `92e42d8452a5bee9f9f3abc983d054f03cedfc9d60e1f629f62d0c67e80ed707`, not this checkout. Not complete served-model coverage.
- `eng-dense-lane-test-20261007/gpu-v1/dense_lane_test.json`: synthetic packed wire, expressly no checkpoint/loader qualification.
- `tp2-served-census-sol-20261007/serve-data.json` and `served-window/tp2-eager-served.json`: FP8 routed MoE, not BF16 dense.
- `eng-t16-epilogue-production-20261007/gpu-complete-ba26bf6d-20261007/launcher-fixed/gpu0/surface.json`: historical test population, not a current serve.

Reuse decision: retain these only in their original scopes; do not relabel them. Historical T16D1/T16D2 folded receipts (census SHA256 `561f7e8c136bf9d236c11d5067cfd97924254ec86dd8db3848a380f7dd731c28` and `6b4b48d40145087097bb115371e6a794b0ff72bc9b19141311bc6fd7d5d8d497`) cannot satisfy the raw-BF16 slice. No new broad qualification campaign is authorized here.

## Pre-run bounded plan

Before any serve: use the admitted `gpu` arm for a read-only prerequisite probe (expected <1 GPU minute, no CUDA allocations). CPU inspection cannot establish actual vLLM dispatch. Determine configured artifact availability, image/build identity and serving serialization before launching. If a prerequisite is absent, stop without provisioning, substitution or retries.

Conditional serve budget: approximately 5–10 GPU minutes, one current-recipe BF16-only dense artifact, one rung, resident TP1 eager, short prompt and 16 generated tokens to exercise prefill and decode. Candidate is the existing Qwen3-0.6B BF16 root-7/q256=1792 export documented in `experiments/bf16_export_qwen.sh`; it is not yet selected or admitted. Use conservative existing memory settings (census default utilization 0.3; bound KV/concurrency using existing CLI controls if supported by observed configuration), no compiled mode and no dispatch override. Serialize via the existing configured device mechanism; do not invoke historical install/container/submission wrappers.

Predeclared expected pair: (`tessera::fused_window_dense`, `native_fused_window_dense_bf16`). Require `--require-decoder native_fused_window_dense_bf16` and derive `--expect-modules` from the actual exported `config_groups` target union, retaining that derivation. No model-name-based geometry assumption is acceptable.

Before serve, inspect every prepared role including packed projections: legacy reader layout, window_bits=14, rows divisible by 4, columns divisible by 32 and >=128, valid run pairs/permutation/tile words, native library and shared-memory support. Record actual predicate results and artifact/wire/recipe/rung hashes. Capture effective argv/environment, imported paths/hashes, build/compiler/binary identity, observed immutable image/runtime/device and actual engine eager/TP/residency configuration. A caller-supplied eager label is not observation.

## Source trace and limits

`src/tessera/serving/bf16_route.py` prepares and applies the native module and records its selected pair. `src/tessera/serving/native_window.py` selects the fused dense owner only when all roles and its library are admitted; otherwise preserve (`tessera::window_gemm_dense`, `native_window_gemm_bf16`) and `lane_reason` without calling that success. `src/tessera/routed_fused.py:fused_dense_window_supported` owns geometry/layout/native/shared-memory checks. Actual predicate execution and dispatch are still unobserved.

`src/tessera/window_gemm.py` and `src/tessera/serving/csrc/routed_fused_window.cu` retain FP32 post-dot row scaling before BF16 output. Native split-K can add a separate scale-applying reduction: one custom op does not establish one CUDA launch. No split-K configuration has yet been observed.

This census cannot restore folded claims or attest cell agreement. It supplies no KL, numerical equivalence, general-device qualification, performance, single-CUDA-launch, contract or pin authority. https://github.com/RobTand/tessera/issues/1383 retains publication and https://github.com/RobTand/tessera/issues/1348 broader matrix/performance acceptance. TrustLadder owns issue updates and closure.

## Observed prerequisite outcome — stopped before serve

The single admitted `run_tests(arm="gpu")` probe completed with exit 0. This was a read-only metadata probe, **not a census pass**. It used `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -c ...` to print interpreter/configuration, distribution versions, and the candidate/config existence checks using `tests/box_artifacts.py` roots. The verified retained stdout records:

```text
python /usr/bin/python3
distributions {"torch": "2.13.0+cu130", "vllm": "0.30.1rc1.dev336+gaf5b4857e.d20260929"}
candidate /mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7
candidate_exists False
config_exists False
```

No configured TESSERA artifact-root override or `OUT` appeared in the filtered environment. The environment also reported `NVIDIA_CPU_ONLY=1` and `NVIDIA_VISIBLE_DEVICES=void`; these are environment observations, not a CUDA device query. The probe imported no Torch runtime and allocated no device tensors. The raw filtered environment remains in retained stdout, SHA256 `f9804536bece24324477ef1dbc64610acd8dd381d28ff2e9e60dd36153d20aca` (2850 bytes); `read_test_artifact` verified stdout and receipt.

The precise failed prerequisite is availability of the documented root-7 artifact and its actual config sidecar in the admitted arm. No substitute artifact was selected and no export, native build/load, serve, environment repair or unchanged retry was attempted. This does not prove all possible artifacts absent. Artifact geometry, recipe admission, module count, prepared-role predicates, binary/compiler identity, immutable executing image, CUDA/driver/device, serialization acquisition, effective serve argv, engine eager/TP/residency and split-K remain **unobserved**. Distribution metadata is not proof of imported serving modules. No fallback or fused selection was observed.

O1/O2 remain unmet. There is intentionally no `census.json`, `config.json`, `census.log` or successful `provenance.json` at `experiments/results/bf16_dense_current_master_census/`: no actual census output or sidecar exists to retain, and a fabricated packet would hide the missing prerequisite. This note and the retained operation document an uncompleted disposition only. Completion requires an existing-owner admitted artifact handoff and actual serialized serving availability, not a runtime/kernel change in this slice.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f9804536bece24324477ef1dbc64610acd8dd381d28ff2e9e60dd36153d20aca","size":2850}},"operation":"c632b7327414f7480661251a4bf92dae087c20f9647d46cd0bf801d63e252d4b","receipt":"c01365429c4c1b5d6332308a7846d934c10a9f886594d6399cc32f89e3d2b4d0"}]

## Repository validation

Initial x86 combined acceptance/issue-reference/impacted-test check: 3 failed, 211 passed, no skips, NO CUDA, zero device allocations. Two failures explicitly identify missing `census.json` and `provenance.json`. The third was the offline issue snapshot not containing the newly cited issues; this note now uses the supplied canonical issue URLs rather than unsupported shorthand. No issue status or snapshot was fabricated. These are evidence-delivery failures, not a kernel regression or an impossible acceptance harness.

Command: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_bf16_current_master_census.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"03396293cd39e571db158d3fc2d38fa32582a02c1865236122802b31275ccae7","size":6217}},"operation":"1effa16478e31d8ae3d59d0d3ed22a36c60336e236ba359476c0b151632cb97a","receipt":"1a5f639bc7c67b4352445da0817f2201f3d5460f2bd8a37786a60e058c414460"}]

Final x86 acceptance plus issue-reference rerun (same command above omitting `tests/test_impacted_tests.py`): **2 failed, 3 passed**, no skips, NO CUDA and zero device allocations. Issue-reference guards now pass; both acceptance failures remain the explicitly missing actual evidence. The prior impacted-test guards passed. No GPU acceptance or served execution is claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b4a9ae3050ebe31c761351da941336d0fecebf44dd5a787246798fec36cd6187","size":3016}},"operation":"d160731cc47098aceca1bbfaceb0e681d27831c288c536e8a8f2dfda2f89d135","receipt":"17d740e794307c2f54b52a6821c61271addc05c286a6dc8874ad09eb69ef5845"}]
