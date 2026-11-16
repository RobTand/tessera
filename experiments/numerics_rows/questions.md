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

## Open at the next handoff

- `rows.md:210`: a unit checks several scale arrays, which can use different bases and digit counts.
  Proposed receipt types: `scale` and `base` are lists, `families` is their union, and `passes_per_family`
  maps each family to its total. `arrays` retains each array's exact base, families, counts and read coverage.
  The other counts are sums. The current implementation uses these types. The spec has not been edited.
- `rows.md:70`: C1 needs the device's observed SM count for `dense_k_split`. The received receipts do not
  record it. Proposed answer: read `runtime.sms` from the next H1 receipt, then put that value in the
  population JSON before review. The draft files keep `sms: null` until that result exists.
- `rows.md:109`: each population path is present, but its wire stamp must be read in an admitted CPU job.
  The prepared population job writes the stamped drafts in its own results directory. Proposed answer:
  review those outputs, copy the reviewed files to the population directory, and set `reviewed: true`.
