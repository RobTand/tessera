# Issue 198 and branch-audit validation (2026-09-05)

The completed parallel populations both publish source commit
`8c21637ae168697b587dcf7db94ed194ce8abd62` and independently verified tracked-source SHA-256
`d45278d3b27b17b1da812193ac9e91623c7a8ebfd3e9b9fe8fe4e5012d05667d`. These are selected validation populations,
not the full release suite. No PrismaBuild was invoked.

| population | selected files | mode | passed | failed | skipped | xfailed | uncollected | elapsed |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CPU, aarch64, torch 2.10.0+cpu, no CUDA device | 126 | xdist 8, worksteal | 1945 | 0 | 505 | 0 | 0 | 370.70 s |
| GPU, aarch64, torch 2.13.0+cu130, NVIDIA GB10 | 129 | xdist 12, worksteal, strict-CUDA | 2485 | 0 | 5 | 1 | 0 | 410.20 s |

The GPU count covers the union of issue 198's 89 selected files and the
cleanup's selected files. The CPU run covers the cleanup selection; issue
198 also had 1,401 passed / 338 skipped on its initial 87-file CPU selection,
then 62 passed / 0 skipped for its final focused additions. The strict-CUDA
controller records 472 tests whose calls allocated on the device. Neither
completed parallel population skipped for missing box artifacts.

Controller receipts (worker shares are not whole populations):
[CPU](data/branch-recovery-2026-09-05/validation-cpu.json),
[GPU](data/branch-recovery-2026-09-05/validation-gpu.json).
The one xfail is the intentional historical checkpoint byte-equality check
already scoped by issue 198; its weight-error comparison is resolved.

## Commands and environment

Known-good image:
`vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
Task-only test dependencies installed in the image and user-local CPU Python:
pytest-xdist 3.8.0, execnet 2.1.2. No PyTorch or serving-runtime upgrade.
Both arms set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and
MAX_JOBS to 1. Eight CPU plus twelve GPU processes use the box's 20 cores;
a one-second /proc/stat sample during the run recorded 100% host CPU busy.
This is an operational observation, not a matched throughput benchmark.

```
python3 tools/impacted_tests.py --ref origin/master...HEAD --json
python3 -m pytest -q -n 8 --dist worksteal <126 selected CPU files> \
  --durations=15 --surface-json /home/rob/tessera-runs/issue-198/audit-parallel-cpu.json
# Inside the pinned Docker image:
python3 -m pytest -q -n 12 --dist worksteal <129 selected union files> \
  --durations=15 --strict-cuda --surface-json /results/audit-parallel-gpu.json
```

Both arms supplied TESSERA_PRISMAQUANT_WORKTREE=/home/rob/prismaquant and
KL_TOOL_DIR=/home/rob/dq-runs. Exact file lists, commands, install logs,
worker shares, timings and full logs are under
`/home/rob/tessera-runs/issue-198/`: `audit-impacted-final.json`,
`validation-union.json`, `validation-additional.json`, and
`audit-parallel-{cpu,gpu}.{log,json}`.

## Failures and interrupted runs: definitive disposition

- The first 89-file CUDA run at 71228ad completed with 1,709 passed,
  59 failed, 5 skipped, 1 xfailed. Two failures reproduced in the three failing
  files on pristine merged master (2 failed / 82 passed / 0 skipped /
  0 uncollected, GPU-visible, 12 tests allocated). Root bypassed chmod; an FP8
  test expected an obsolete error message. Both expectations are corrected.
- The other 57 failures reproduced with the real plan-to-census file order
  (57 failed / 23 passed, no skips or uncollected modules). The isolated census
  file passed all 60 cases with zero skips. A new isolated CLI regression
  first failed with `ModuleNotFoundError: No module named
  'tools.tessera_route_census'`; making Tessera's tools a regular package
  fixes the namespace collision. Its failure and passing receipts are in
  `census-import-prefailure.log` and `census-cli-prefailure.log`, with the
  completed parallel populations covering the fix.
- Serial reruns were deliberately interrupted when the user required full
  machine use: GPU 1,058 passed / 0 skipped at 794.23 s, CPU 451 passed /
  20 skipped at 382.70 s. Both exited interrupted, not green. Their partial
  receipts remain at `validation-final-gpu.*` and `audit-final-cpu.*`; the
  completed parallel populations above supersede them.
- The additional wire-plan file passed 7 cases with zero skips and zero
  uncollected modules in the GPU image, with zero actual device allocations;
  this is a logic check, not a CUDA coverage claim. It is included again in
  the final 129-file union.
- Six touched Python modules compiled with in-memory compile checks. Bash
  syntax and git diff whitespace checks passed. The retired campaign stub
  exits 2 with its receipt pointer before doing any work.

## Skip reasons, verbatim

### CPU

- 86: `the lane is a CUDA kernel`
- 84: `needs a CUDA device`
- 84: `the encoder is a CUDA path`
- 52: `the fused window Viterbi is a CUDA path`
- 37: `encoder is a GPU job`
- 29: `the Viterbi is CUDA`
- 29: `the captured TCQ trellis is a CUDA path`
- 26: `the encoder is a GPU job`
- 24: `the kernel lane is a CUDA path`
- 19: `the fused window Viterbi is a CUDA path and needs triton`
- 14: `the Tessera encoder is a CUDA path`
- 8: `the fused window Viterbi is CUDA`
- 6: `needs CUDA`
- 5: `the kernel lane runs on CUDA`
- 1: `E2M1 publishes no reader range`
- 1: `could not import 'vllm': No module named 'vllm'`

### GPU

- 2: `e2m1-tcq-lut-release does not cut 4 ways along columns`
- 2: `e2m1-tcq-lut-release does not cut 8 ways along columns`
- 1: `E2M1 publishes no reader range`
