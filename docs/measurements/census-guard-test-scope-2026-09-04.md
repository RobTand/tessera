# Census guard test scope — 2026-09-04

Integration CI run `33916752261` on `7527f8b` exposed two test failures:
`test_route_grid_and_census_guard.py` selected the first assignment named
`missing` anywhere in the census source. The new batch/decode population
helper now precedes `main`, so the tests inspected its guard rather than
the unchanged family-map guard. No runtime refusal was removed or weakened.

The test helper now parses the source and requires one unambiguous direct
`missing` assignment in `main`. An added regression puts an unrelated
population helper before it. The decoder-short mutation still evaluates
the actual family guard, retaining the original defect check.

PrismaBuild action
`0a2c42301b83bd8e6abfb7d248b8765f7d494784fdeb61339216c99c22c3d34e`
showed the unchanged selection failing all three cases (4 passed). The new
case failed at line 172: `AssertionError: assert 'TESSERA_FAMILIES' in
'    missing = sorted(set(batch) ^ set(decode))'`. The two existing cases
failed at lines 136 and 148 against the same unrelated guard. The deployed
pool retried the red action; its final outcome retains the red output.

The corrected test file passed all 7 cases in 1.15 seconds under action
`bff19b04703c7c4272b31c2b21c036ab00eca1d9ea0a16e8d781f6dbdaa872f8`.
Original worker return code: 0. Receipt:
`136d568b643189c8fa45b47b356ea0454b6933a307adabecc3466da2eb5629eb`.
Both arms used dl380g10, serial CPU, 1 CPU/4 GiB, torch 2.11.0+cpu,
with zero skips and zero uncollected modules. Verbatim population:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Skip reasons were empty. This is test-harness evidence, not a served result
or a whole-tree integration run. No master baseline suite was run.
