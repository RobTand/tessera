# tessera#1176 k2 CPU preflight receipt (2026-10-10)

Scope: CPU dry run of the gates this issue names.
It changes no kernel code, no flag default, and makes no power claim.
It scores no teacher and seals no TR3 receipt.

## What ran

One PrismaBuild action on `dl380g10`, CPU only, priority 0:

- Action key: `ab0eb804df71b02c10bbdcabf1803805a1b648f61395e5580923ff3f3b223c64`.
- Snapshot parent: `4c0fd6aca04f1c3e95b094ae006c2479043c3372` (clean tree, equals `origin/master`).
- Demand: 4 CPUs, 8 GiB. Peak RSS: 791.2 MiB. Elapsed: 9 s.
- Command: gate imports for `comparison_arm_identity`, `comparison_input_intake`
  and `served_generation_client`, then pytest with `-n 4` on:
  `tests/test_served_generation_client.py`,
  `tests/test_comparison_intake.py`,
  `tests/test_serving_mla_mask_registration.py`.

## Result

- Gate imports print `gate imports ok`.
- 30 passed, 1 skipped, 0 failed.
- Population: torch 2.11.0+cpu, no CUDA device, 0 tests allocated on device.
- Skip reason, verbatim: `could not import 'vllm.v1.attention.backends.registry':
  No module named 'vllm'`. This skip is the expected CPU shape, not a defect.

## What is not done

- No manifest is frozen. The intake needs owned teacher arm files, and this
  repository does not define the two teachers for the masked-tile candidate.
- No GPU serve started. The served TR3 scorer (`measure_glm_tr3_vllm.py`) and
  the pact-u4 teacher serve procedure live outside this repository.
- A first attempt with action key
  `7d7af297fb9647a4dab4a720b8ca6d3e8f1e8cf36b1bc951ab9a610d88c18c1e`
  failed in 1 s. The cause was the probe command in the submission, not the
  code: `comparison_arm_identity.py --help` runs single-arm mode and exits 2
  with `unset: ARM_ARTIFACT ARM_AUDIT_ROSTER ARM_AUDIT_ROSTER_SHA256 STAGE`.
  The resubmission without that probe passed.
