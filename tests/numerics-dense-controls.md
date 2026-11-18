# Dense fault controls

The lead runs the CPU checks and real-artifact controls through Flux without a GPU.
Use `tests/test_numerics_dense_controls.py` for the CPU check.
Keep all test files under `tmp_path`.
Keep caches, bytecode, and replay output in local Flux scratch.
Overlay both Python files from this change onto the captured tool checkout.

The runner requires HEAD to equal the captured tool commit.
Its separate index hashes actual `experiments` files, including untracked files.
The runner checks that tree before and after each group and during assembly.
The tree must equal the captured tool commit's tree.
The runner hashes every original unit file before and after each group.
These hashes include all attempts, partial receipts, JSON-line streams, payloads, and `phase.json` files.
Each replay copies the full unit attempt trees into its separate scratch root.
Only these copies supply tensors to the module compare.
The counted root and its receipts do not change.

Use `--prepare` followed by all four prepare paths.
Use `--capture` followed by the four paired capture paths in the same order.
The pairs cover TP1 gate, TP2 gate, TP1 diagnostics, and TP2 diagnostics.
TP2 includes both ranks in one process.
Each path has the form `<root>/units/<key>/attempts/<id>/{prepare,capture}/`.
The module compare reads every attempt and selects the newest finished pair.
An earlier FAIL remains FAIL.
An open STOPPED attempt or an attempt ID gap refuses comparison.

Supply `--manifest`, `--population`, `--q1`, `--contract`, `--scratch`, and `--out` in each process.
Keep scratch and report paths outside the original module root.
Use canary S with one shared `gate_up_proj` module, TP1, the TP2 pair, and both roles.
Require all canary groups to pass before a counted receipt counts.
Keep both C1 controls with the first counted row.

Eligibility comes from capture receipt units with role == "unit" and admitted_by_v68 is True.
Cases retain the full-row ordinal order across the streamed artifacts.
Each group plans all 13 controls without reading R0 output.
The runner writes control_table with real case ids before the first replay.
The table below uses <case> for the selected case id.

| Control | Replay group | Case choice and plant | Expected check text |
| --- | --- | --- | --- |
| composition-staged | R4 | First eligible sweep case. Change staged at the zero index. | `apply differs from the same staged launches` |
| sweep-inside | R2 | First eligible sweep case. Change output and staged at the first true X index. | `<case>: exact output differs` |
| sweep-outside | R1 | Same sweep case. Change output and staged at the first false X index, if present. | No findings. `NOT_APPLICABLE` if no outside element exists. |
| chunk_pairs-inside | R2 | First eligible chunk_pairs case. Change output and staged at the first true X index. | `<case>: exact output differs` |
| chunk_pairs-outside | R1 | First eligible chunk_pairs case with a false X index. Change output and staged there. | No findings. `NOT_CHECKED`, or `NOT_APPLICABLE` if absent. |
| diagnostic-inside | R3 | First eligible diagnostic case. Change output at the zero index. | `apply differs from the same staged launches` |
| diagnostic-outside | R1 | Same diagnostic case. Change output and staged at the first other index. | No findings. `NOT_CHECKED`, or `NOT_APPLICABLE` if absent. |
| scale_index-inside | R2 | First eligible scale_index case. Change output at output_index. | `<case>: exact output differs` |
| scale_index-outside | R1 | Same scale case. Change output at the first other index. | No findings. `NOT_CHECKED`, or `NOT_APPLICABLE` if absent. |
| composition-outside | R1 record only | First eligible sweep case. No plant and no replay. | `NOT_APPLICABLE` |
| wrong-scale-index | R2 | First other eligible scale case whose wrong read changes the reference. Use that value at output_index. | `<case>: exact output differs` |
| incomplete-C1 | R0, counted row only | Same case as wrong-scale-index. Reduce reads_covered after a positive C1 probe. | `incomplete scale-index coverage before the freeze` in unattested |
| wrong-C1-domain | R0, counted row only | Same case as wrong-scale-index. Increase array.N after a positive C1 probe. | `C1 scale case has the wrong read tuple or array size` in invalid |

Wrong-scale-index lands in R2 on a case distinct from scale_index-inside.
If no such case exists, planning fails. No control leaves the set.
Only active plants must have distinct cases within a replay.
R3 and R4 stay separate because their finding text has no case id.
R2 requires the exact union of four case findings in ordinal order within each target unit.
The runner calls `dense_phases.host_compare` once per replay for all module pairs.
The real `check_pair` reads streamed cases and binds each capture to its prepare digest.
The runner compares each case's findings, invalid, and unattested lists with exact equality.
It also checks the complete stream union against the planned checks.
The receipt retains only the first finding per unit, as the step-5 tool requires.
The runner checks this first finding separately from the complete stream union.
A missing finding, an extra finding or a wrong outcome fails the group.
The combined test replaces one plant with the original value and requires failure.

The canary has exactly five replays. The jobs can run at the same time.

| Group | Replay |
| --- | --- |
| R0 | Unmodified baseline. Defer both C1 controls. |
| R1 | All active outside plants. Require PASS with no findings. |
| R2 | sweep-inside, chunk_pairs-inside, scale_index-inside and wrong-scale-index. Require FAIL with their exact union. |
| R3 | diagnostic-inside alone. |
| R4 | composition-staged alone. |

Use --group R0 through --group R4 in separate jobs, with separate --scratch and --out paths.
Each process writes one group report. A group report cannot permit counting.
Use --assemble R0.json R1.json R2.json R3.json R4.json with the same required path arguments.
Assembly runs no replay.
It requires five valid reports with the same original artifact hashes, source tree, and control table.
It also requires R0 PASS with rc 0 and checks every baseline target through the real join.
The expected baseline join result is CANARY_REFUSED for a canary and PASS for a counted row.

On a canary, R0 records both C1 controls as DEFERRED_TO_COUNTED_ROW, never PASS.
On the first counted row, R0 runs each C1 mapping control separately before assembly checks the baseline join.
Those probes use real memo recompute and operand validation before a native launch.
Their refusals test compare and join mapping only. Their result is MAPPING_ONLY, never DETECTED.
The counted row therefore has seven replays. Complete these controls before the ordinary row join.

If a group fails, use --only <control> for each control in that group with separate scratch paths.
The single form runs only that control. It does not run a baseline first.
Keep the expected lists fixed after a run. Repair the runner, then repeat the compares on the same captures.
Without --group, --only or --assemble, the runner runs the same groups and assembles their reports in one process.
A missed plant or changed original artifact makes the report INVALID.
No real-artifact result or control-time measurement exists yet. Routed controls remain separate work.

The expected check texts are unchanged from `8871a1db`.
The exact-output text comes from `experiments/numerics_rows/row.py:189`.
The composition text comes from `experiments/numerics_rows/dense_phases.py:405`.
The C1 domain text comes from `experiments/numerics_rows/scale_launch.py:66`.
The coverage text comes from `experiments/numerics_rows/row.py:217`.
It remains `incomplete scale-index coverage before the freeze`.

The lead runs BEFORE at `01c00d377141bcc744857ae53ca0a4494eeaaede` with the new test file only.
Every new test first asserts that the runner has `module_inputs`.
The BEFORE failure text is `feature absent: streamed unit fault controls`.
The test file permits an absent runner at collection, then fails at that first assertion.
No fixture calls new runner helpers before the assertion.
The lead runs AFTER with both Python files and requires all selected tests to pass without skips.
The maker runs only `symcheck.py` and `git diff --check` on the host.
No host test, project import, numerical run, or Flux command is part of the maker's checks.
