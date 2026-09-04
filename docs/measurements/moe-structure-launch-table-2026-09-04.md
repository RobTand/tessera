# Structure-aware route launches

`scheme.ROUTE_LAUNCHES` now carries the structure each launch serves. Dense
callers keep their default; the routed FP8 stack publishes its resident-only
modular fused-MoE entry point. Its census expectation and the contract's
launch derivation read that same table. This adds no attested cell, changes
no runtime image, and does not claim a compiled MoE serve was measured.

PrismaBuild pre-fix action
`ea0f66f044a4` ran the six new cases against the old source and reported
6 failed, 42 passed. All new regression functions failed:

```text
test_launch_table_structures_follow_the_dispatch_builders
tests/test_serving_contract.py:548: KeyError: 'structures'
test_moe_launches_are_structure_specific_and_resident_only[batch|decode]
TypeError: route_launches() got an unexpected keyword argument 'structure'
test_moe_census_expectation_is_derived_from_the_shared_launch_table
tests/test_serving_contract.py:595: AssertionError
test_cell_launch_derivation_uses_the_cells_structure[batch|decode]
ValueError: synthetic.executes is [('vllm.fused_moe.modular_kernel', 'torch_materialize_stock')] but the TESSERA_FP8 route makes [('torch._scaled_mm', 'torch_window')]
```

The synthetic cells exist only inside the tests. The table mutation test
proves the expert census derives its expectation rather than merely holding
an identical second spelling.

Post-fix action
`3b33d7a876b4e6472a26440af60018105bf1576e0ef25b72174659994ccba036`
ran `test_serving_contract.py` and `test_serving_moe_route.py`: 57 passed in
175.85 seconds. Receipt:
`9f29bf92cdb23ceab226c10c8fba65154c1cb3c0821a3c27330f02d75589b338`.
Both runs used dl380g10, serial CPU:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Selector action
`ca57897f8c04cf7b26bd53f88edd2abfb95f84df5de41ee53c0b3f64e1d864f8`
returned `narrowed` with 46 files against fetched base
`894305ee41b648e563bc1fa672948f5f6cf8c4af`. The remaining selected files are
submitted separately in action
`87a9114a88776dfe85c67e4a1ba114e3214e9b7c3a16989006c512f1e7c6fd05`;
that population must be read before treating the selected validation as
complete. The previously green sidecar test is excluded from the repeat.
