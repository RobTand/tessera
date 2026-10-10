# ts-integrator private tools, archived on a branch (D65.4)

These scripts were standalone files on `/mnt/shared/tessera-measurements/`, outside any repository. D65.4 makes
that a defect, so they are committed here as retained history. Nothing here runs in CI. Dated files under
`docs/measurements/` are append-only (AGENTS rule 10).

## population/

- `population_audit.py`: reads a stored 12-shard population and the pool's own queue endings, and judges it
  GREEN or NOT GREEN. It replaced an aggregate that dropped return codes and counted a missing summary as zero
  (tessera#1069). Retained because it still reads old text-output populations.
- `run_population_v2.py`: ran 12 shards through PrismaBuild and judged them with the audit. Caps clients at
  eight (D21) and declares one native thread. **Superseded** by `tools/selected_population.py` (PR #1072).
- `source_audit.py`: compares the effective source of every action in a population with the parent, through
  `tessera._dev.suite_source.measured_source`. It checks sealed inputs only, not runtime behavior. Its owner
  check matches by construction. Run for PR 1028 as PrismaBuild actions `8f9fe3b8e232…` and `eb6a13ae27cc…`.
  It needs a Tessera checkout with the PrismaBuild tools at `/mnt/shared/prismabuild-fleet`.

## pb1496-harness/

- `harness.py`, `harness-final.py`, `harness-v2.py`: the acceptance harness for the impacted-test selector
  (PB1496). `harness-final.py` gave 129 scenarios on master `61d3195565` (106 ok before the `Path.glob` fix,
  126 after). `harness-v2.py` gave 144 of 144. Results stay on `/mnt/shared/tessera-measurements/` in
  `ts-integrator-pb1496-acceptance-20261006/` and `ts-integrator-pb1496-final-20261006/`.
  The harness needs files inside a PrismaBuild snapshotted repository to run (`pbrun`).

## Lost

`/tmp/mutate.py` (the mutation check of PR #1072's five set-level rules) was lost at the reboot. Its result is
`mutations.txt` on shared storage and the table in the PR #1072 body. It was about 25 lines: disable one rule,
run `tests/test_selected_population.py` through `pbrun`, restore the file.
