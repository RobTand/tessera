# Draft 17 refusal audit

The rule is external and unchanged. This audit covers section 4's INVALID list.

| Rule precondition | Refusal in this tool |
| --- | --- |
| Wire stamp differs | `row.run`, before `verify_cases`, checks every role's `matches_attested_wire`; adapters also bind loaded wire hashes. |
| Image, vLLM, torch, kernel source differs | `common.check_runtime`, `common.check_source`, `serving_build.check_retained_source`. Version and source refusals use `Invalid`. |
| Served identity or audit fails or cannot finish | `row.check_served_identity`, `row.audit_served`, and `row.check_served_after`; `serving_build.compare_binaries` includes the byte audit, SASS/PTX witness and negative controls. |
| C1 sample, unfinished, wrong population, omitted/repeated role or cut, omitted/changed configuration, unmatched pair | `row.planned_units`; the join also derives the geometry and configurations from signed wires through `scope.declared_modules`. |
| C1 wrong source or contract | `row.check_manifest_source`; join checks the nine common bindings and the supplied contract's file hash. |
| X false where claimed | `row.require_x`, `scale_launch.finite_references` and the routed case checks. |
| Incomplete coverage claimed complete | `row.coverage`, `row.verify_cases`, `scale_launch.check`. |
| Nonfinite reference | `row.finite_reference`, before the case's native call. |
| Repeated quantizer bytes differ | `row.Dense.quantized`, `routed.Routed.quantized`. |
| Mutated reference not flagged exactly | `row.run`, before `verify_cases`, checks all `negative_controls`. |

No named refusal is absent. This is a source audit, not a test result. The CPU
job checks the new join and canary conditions. The row-local `diagnostic_cells`
field is not a joined verdict. The join reads only individual unit evidence.

GPU minutes use whole-device NVML kernel-resident samples. This is a sampled
activity-time measurement, not SM throughput or a performance claim. The receipt
keeps sample coverage, errors, power, memory and wall time. A missing measurement
is null with its error, never a zero-minute claim.
