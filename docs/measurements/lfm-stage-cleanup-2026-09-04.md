# LFM one-shot stage cleanup — 2026-09-04

Source `fedf615` fixes both the teacher and census/student drivers; targeted
helper tests follow in `05a5571`. No model bytes, image, corpus, execution mode,
quality threshold or serving command changed. `docs/ARCHITECTURE.md` changed
with the cleanup implementation.

The old finalizers removed a container with the fixed stage name even when
the prelaunch gate had just refused that pre-existing name. Ownership now
starts immediately before the subprocess launch, after preflight and log
creation. A prelaunch refusal records the observed state but cannot remove
the foreign container or replace the original refusal with a cleanup assertion.

The shared helper stops telemetry before cleanup, joins for at most two
seconds, and gives all Docker/GPU subprocess waits one 90-second deadline.
The earlier independently bounded waits could total 170 seconds including
the monitor join, exceeding the outer 120-second grace. These are bounds
derived from timeout settings, not measured GPU-run durations. Failed commands,
inspection errors, remaining GPU work and deadline exhaustion produce unsafe
cleanup receipts rather than being interpreted as an empty process list.

## Red before green

All runs used deployed v1 PrismaBuild on dl380g10, one CPU, `mem_gb=4`, with
OMP/MKL/OpenBLAS thread counts set to one. There were no local tests or GPU
workloads. Every quoted test population reported:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Skip reasons were `{}` and uncollected modules `[]` throughout.

The behavioral regression extracts each actual driver's finalizer AST and
executes it with fake container/GPU operations and `launched=False`. On the
original source both cases failed in action `44b71015f950`:

```text
test_prelaunch_name_collision_never_removes_existing_container[ts5_lfm_teacher_bound.py]
test_prelaunch_name_collision_never_removes_existing_container[ts5_lfm_served_bound.py]
tests/test_ts5_bound_collision.py:50: AssertionError
AssertionError: [['docker', 'rm', '-f', 'conflicting-stage']]
```

Population: `/mnt/shared/tessera-runs/ts5/lfm25/astra-bound-cleanup-collision-red-r1.surface.json`.
Both then passed in action
`a5a3a437fe7d90ac15998d341f494a6e6fbb0abc73f4104c32dc32cf34efb5a1`,
2 passed in 1.10 seconds. Receipt:
`68010fdf62dad8ecb24b9c8ee6f1cf71122f372f53c0ffa3874c23772a1249b1`.

All ten helper cases were also run against pristine
`2fd045f8bd2fbc09f7140018b942ecbc9f8c1047` with only the new test file added,
in action
`198d8d71192526e45cf444f92ad2a2abb6f3e04ca766051f2c5bff42a0401f6c`.
Each failed with `FileNotFoundError` for `experiments/ts5_stage_cleanup.py`:

- `test_owned_container_removed_and_release_verified` (completed false/true)
- `test_unlaunched_stage_does_not_remove_preexisting_container`
- `test_existing_gpu_work_prevents_release_even_after_container_removed`
- `test_absent_owned_container_needs_no_removal_but_still_checks_gpu`
- `test_subprocess_failure_becomes_unsafe_receipt` (timeout, nonzero process, OS error)
- `test_inspection_removal_and_gpu_share_one_deadline`
- `test_telemetry_join_error_is_recorded_not_raised`

Final combined action
`7b60047bbe9d3a70eed294638b7c29deace14053292feb08a6dd3c5d377d5484`
ran `tests/test_ts5_bound_collision.py` and `tests/test_ts5_stage_cleanup.py`:
**12 passed, 0 failed, 0 skipped, 0 uncollected modules**, on the CPU
population above. Receipt:
`e45801dde1b0e1019f5519c1353ee29debf0e64fb6bd922293df3f4554044de3`.

Selector action
`f3071b784f7b444c52a56f86338df489edc2826463d3d66f0a596bb99fa86855`
compared the snapshot against fetched exact base `2fd045f8bd2fbc09f7140018b942ecbc9f8c1047`,
using its recorded parentless direct-tree fallback. Verdict: `narrowed`,
with exactly the two test files in the final green action. Selector receipt:
`b89a530cb1894e53996c88ca6f65a81564781049420eea6cdc0b0cd962319a16`.

## Bounded review outcome

The surrounding fixed-1024 teacher/student filenames, raw 4096-position
metadata, exact image/eager context and existing KL comparison were inspected
without executing them; no comparability blocker was found for this campaign.
The adjacent nondefault `TESSERA_KL_TOPK` forwarding defect was handed to the
coordinator and fixed separately, not folded into cleanup. These CPU tests are
not a served-quality result or MoE attestation.
# Explicit retry namespace after the first census refusal

The actual `census-bound-r1` action stopped before model load because the
NFS-root-squashed container could not read an owner-only merged shard.
Its verified cleanup was safe, and automatic retries correctly refused the
existing r1 output directory. A new explicit positive `--attempt` now selects
fresh output/container/local-census names; it does not overwrite or reuse r1.

Four added pure namespace tests failed before this addition under PB
`790688233993` at `tests/test_ts5_stage_attempt.py:14`,
`AssertionError: campaign driver has no explicit attempt namespace`.
Afterward, PB
`86cbe958a258e1e49b80bf601825984ee89194576fb380772d793e8f91f8632d`
returned zero: 16 passed in 1.18 seconds across namespace, actual-driver
collision and shared-deadline tests. Both runs were serial dl380g10 CPU,
torch 2.11.0+cpu/no CUDA, 0 skipped and 0 uncollected modules. Green CAS:
`dd50e17eb3de8c2b57a24900571cde9159f1be8530bc7a7181a9f92aae140e61`.
