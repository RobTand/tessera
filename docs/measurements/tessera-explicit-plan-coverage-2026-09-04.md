# Explicit plans bind final checkpoint populations

Code: `740f79e` (regressions), `aaf1ab1` (shared validator and gates), based on
`894305ee41b648e563bc1fa672948f5f6cf8c4af`.

The serving-part merger already checked source ownership and exact output
tensor coverage, but it derived expected encoded tensors from the part's
module manifest. An omitted expert stack could remain internally consistent
BF16, with its source tensors retained and its target ignored. Neither that
merge nor the sidecar checker proved that every explicitly planned stack had
actually been encoded. This was a P1 gate gap found while reviewing final
campaign coverage and fixed in the same session; no deferred issue was filed.

`serving_parts.validate_explicit_plan` now reads the already-sealed plan before
the merger creates its output. Planned expert-stack names must equal emitted
and declared stack sets. Every stack must carry its complete expert/projection
population, consume its source projections, and declare and emit the requested
grid/rung. Explicit dense tensor choices must have an emitted role at the
requested grid/rung. Implicit dense defaults have no complete plan roster and
are outside this additional check.

The sidecar checker calls the same validator against
`export_identity.options.plan`. `--plan-json` adds an external common-plan
comparison and refuses a missing manifest. Its routed-MoE summary must name
the same stack population. The current campaign therefore derives its
22-stack requirement from its plan rather than pinning a model-size literal.
Existing version-one parts and encoded containers remain compatible; this is
an assembly/serve-gate change and requires no encoder restart.

## Pre-fix evidence

The complete added regression set was cherry-picked onto the unchanged base.
PrismaBuild action prefix `23b9ae88a8de` produced **7 failed, 1 passed,
23 deselected**, on dl380g10, torch `2.11.0+cpu`, no CUDA, zero skipped tests
and zero uncollected modules. The passing case was the complete explicit-plan
matrix control. Failure lines:

```text
test_explicit_plan_cannot_lose_a_whole_stack_to_bf16
E   Failed: DID NOT RAISE ValueError
test_explicit_plan_checks_each_emitted_role[grid-BF16]
E   Failed: DID NOT RAISE ValueError
test_explicit_plan_checks_each_emitted_role[q256-896]
E   Failed: DID NOT RAISE ValueError
test_explicit_plan_checks_declared_group_rungs
E   Failed: DID NOT RAISE ValueError
test_explicit_plan_requires_every_source_expert
E   Failed: DID NOT RAISE ValueError
test_embedded_plan_cannot_claim_an_unwritten_stack
E   AssertionError: assert 0 != 0
test_explicit_plan_requires_manifest
E   TypeError: main() got an unexpected keyword argument 'plan_json'
```

The whole-stack omission case contains 22 planned source stacks, 21 encoded
stacks and one stack retained as BF16, with consistent source ownership,
files, hashes, indices and supplied module records. The previous merger
published it; the fixed merger refuses before its output directory exists.

## Final qualification

Action
`36542676fc1f9982e54f52b4aa48b4c080f24e9a73bf7a65febde496b4100ed3`
ran the impacted-test selector against the exact fetched base and its
parentless snapshot. Verdict: `narrowed`, selecting
`tests/test_serving_parts.py` and `tests/test_ts5_sidecar_check.py`.
Those files and the legacy merge guard were run through PrismaBuild.

Result: **59 passed, 37 skipped, zero modules uncollected**, in 44.90 seconds;
dl380g10, torch `2.11.0+cpu`, no CUDA, observed exit status `0`. The 37 skip
reasons were all, verbatim:

```text
encoder is a GPU job
```

CAS receipt:
`6a6fc1a169f30ee336218a6ec0d7868b9bca67d9bcf404a0270b5812b73c068c`.
The actual CPU expert encode comparison still passed: partitioned and direct
exports produced bit-identical tensors and equal schemes/accounting. This run
does not claim CUDA coverage or served-MoE quality. No PrismaBuild code or
runtime deployment changed.
