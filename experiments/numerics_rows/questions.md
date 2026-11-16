# Lead answers, 2026-11-16 09:03 UTC

- Run the BF16 subnormal operand on the value library. Record "not representable" for E4M3 and E2M1.
- Require the frozen rule hash and H1 receipt as explicit row inputs.
- Inject an async device assert in a helper kernel inside the recorded boundary. Keep the cell binary unchanged.
  Label this control as a harness check of error attribution, not of the cell kernel.
- Record C1 time and peak memory after the first module of each arm.
- Draft one population JSON per arm for the lead to review and sign.
- Transfer separation across M configurations only when the pair, rung, cut, loaded binary, and compiled kernel symbol agree.
  Record the compiled symbol per M configuration. Draft 8 defines this rule.
- The lead runs `/home/rob/tmp/ci-speed/start-when-idle.sh` when the key count is zero.
  The CPU job exports `bazel-targets.txt` and does not run `bbrun`.

The freeze record and signed populations remain required inputs.

- `rows.md:210`, draft 8, accepted at 12:51 UTC: `arrays` keeps each array's exact fields.
  Unit fields use `scales` and `bases` lists, the union of `families`, and summed counts.
  `passes_per_family` stays on each array. There are no open spec questions.

## Inputs that remain pending

- H1-G at `d1f8c7c6` records `runtime.sms: 48` in
  `/mnt/shared/flux-jobs/tessera-1717-H1-G-d1f8c7c6-20261116T113444Z/results/h1.json`.
  That observed count is recorded in the population drafts and the prepared population job inputs.
- Five population arms have completed stamp reads. The next CPU job reads all six with bounded
  metadata reads and compares the five completed outputs. The lead must review and sign the results.
- H1-N still needs its GPU run. P3 confirmation, the signed populations and the frozen rule remain required.
