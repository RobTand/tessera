# MoE campaign and census gate corrections — 2026-09-04

These checks used PrismaBuild on dl380g10, serial CPU, torch 2.11.0+cpu.
Every test run reported zero skipped tests and zero uncollected modules:

```
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

No GPU, served-quality, or positive MoE contract attestation is claimed here.

## Campaign status and sidecar completeness

Pre-fix action
`72b0f5d469804df2f1e2193792683fefa581ea849d22c4e04af3723e24edde28`
ran the new campaign and sidecar regression files against source `307c0d1`:
five failed, one passed. The failure lines were:

- `test_failed_route_census_cannot_report_successful_campaign`: `assert 0 != 0`,
  while the driver reported `teacher=0 student=0 census=23`.
- `test_exact_wires_cover_canonical_and_lfm_source_names`: `assert 1 == 0`
  for both LFM shard layouts and the canonical single-file layout. The canonical
  numbered-shard case was the passing control.
- `test_maximum_stride_does_not_hide_a_missing_expert_projection`: `assert 0 != 0`.

Fixes `3c9fa6d` and `5bcd97b` make census success mandatory and check every
declared expert projection under its canonical or runtime shard spelling.
The sidecar reads indexed shards, or all safetensors files without an index.

Green action
`1c99f3462ffd7cd5ba26ff713cdba4f8a0ac1ffe5bb06d6c7ac1a8d67f3aada0`
reported 11 passed; receipt
`2d4d7a21fda411817223ab453563ecbff7ac07ba667f7715cdee336219087752`.
The selector used exact fetched base
`307c0d15a9a56c98883058118862c4e383edc22c` and a direct tree comparison because
the PrismaBuild snapshot has no merge base. Its `narrowed` list was the two
new test files and `tests/test_impacted_tests.py` (the master selector fix was
cherry-picked to support this snapshot comparison).

## MoE structure, declaration ownership, and cell rung

The old census checked only dense cells, looked up rungs at the child record
name rather than the declared stack, and read only top-level `q256` although
MoE stores it per group. All three would leave even a future measured MoE cell
unchecked. Fix `625198d` aggregates per-structure agreement in schema 2,
resolves each record through its declared owner, and derives a cell rung only
when every group and role agrees. Missing cells and mixed rungs stay
unattested. Exact observed backend suffixes remain in records; a base-symbol
cell can match that entry point, while a backend-specific cell still requires
its exact backend.

Pre-fix action
`66f225d2eeb66a51b093807f061a2e57cf403c467b3254c7f8ca3266ff7cdb30`
reported four failed, 11 passed. The added regression functions failed at
`declared_rung` and `all_structure_agreement` with `AttributeError` because
neither operation existed. They exercise group/role rung unanimity, mixed
dense/MoE records, owner lookup, launch and backend mismatch refusal, and
absent cells/owners.

Action `ccac6c5754e5c1703555e25ecd169ef99b9ba6fa2a5b1b79af118cc0d97eb1a5`
ran the selector successfully but a mistyped test filename caused pytest
exit 4 with no tests run. This is not green evidence. The selector narrowed
the cumulative changes to:

```
tests/test_census_cell_agreement.py
tests/test_census_engagement.py
tests/test_impacted_tests.py
tests/test_route_census_module_space.py
tests/test_ts5_serving_gates.py
tests/test_ts5_sidecar_check.py
```

Corrected action
`3c4ef8a0651986fc27edc4b3452642d32746f1cc63b6e977e823348565a5b4f5`
ran exactly that list: 50 passed in 1.69 seconds, with the device population
above. Receipt:
`6cf8c07e12af6c2d87fef7463b74a49a0eebbb2cb728fbd580920cc3c668776a`.

Side finding fixed in separate prose commit `e3c11ff`: the census ownership
docstring now states its actual rule, one declared owner per record; it does
not claim the reverse cardinality is checked.
