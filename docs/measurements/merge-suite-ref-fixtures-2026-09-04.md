# Merge-suite reference fixtures in parentless pool snapshots

The coordinator found a test-infrastructure defect while preparing the final
MoE integration suite. `test_the_receipt_states_which_tree_it_is_about`
required the checkout running pytest to contain `master` or `origin/master`.
PrismaBuild's parentless snapshots contain neither. The production receipt
correctly reported an unknown comparison, but the test treated that honest
answer as a failure. Earlier targeted runs fetched a base ref to satisfy the
test; the final suite should not need that incidental setup.

The test now constructs its own repositories covering local master, remote
master, a distinct branch head and no master ref. It checks the exact head,
resolved reference and three-state comparison. No production behavior,
runtime default, wire bytes or population gate changed.

## Pre-fix failure

PrismaBuild action
`faf2030f1841ab5989453ba6a0890b651f36474c07bc6d04a772cf5c9c7f6eb6`
ran the original test in the parentless snapshot without fetching references:

```text
tests/test_merge_suite.py:305: AssertionError
E       AssertionError: assert 'none resolved' != 'none resolved'
1 failed in 1.17s
```

Receipt: `c346b8a837767afce23f14717567ca64dd94c91157ae2a0e71b9e9fb3bf6a63b`.
The action wrapper required pytest exit 1 and returned zero on that expected
failure; its successful action status is not a green test result.

## Targeted green

PrismaBuild action
`4ad5183f1ded1e86f6f5da9ca4e7358f07683ac9de74e2ac47c36b136adf1551`
ran the complete touched file, `tests/test_merge_suite.py`, without fetching
references: **41 passed in 2.91 seconds**, zero skips and zero uncollected
modules. Receipt:
`edb9923a102ee2a93cf0b78d807327efd3e21c08b98806c86b9209d2ee7ef434`.

Both runs were serial on dl380g10, one CPU and 4 GiB reserved. Their population
block was:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

These are targeted test-infrastructure checks, not the final full CPU/CUDA
merge receipt or a served model measurement.
