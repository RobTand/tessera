# MoE role boundary validation — 2026-09-04

The routed-MoE sidecar validator previously checked each group's role count
and total geometry, but not canonical role names or equal gate/up row counts.
For example, `w13` roles `[gate_proj, 32], [up_proj, 96]` could agree with
their individual blobs and total 128 rows, yet the runtime splits the decoded
tile at row 64. Likewise, consistently swapping sidecar roles and containers
could swap the gate/up meaning without a reader refusal. This is a P0 format
validation defect; it was fixed directly rather than filed for another worker.

`scheme.MOE_GROUP_PROJECTIONS` now owns the canonical projection order shared
by writer and reader. Validation refuses other names/order and unequal `w13`
role rows. Valid canonical artifacts have unchanged bytes and runtime layouts.
The architecture description was updated with the guard in commit `8ef072d`.

## Pre-fix failure

PrismaBuild action
`375c56453d948843439560d9aceb518c73afa6c7760594e3736fbae3ffab2aba`
ran `tests/test_serving_moe_scheme.py` against the unfixed source based on
`307c0d15a9a56c98883058118862c4e383edc22c`, with only the new regressions added.

All six new cases failed with `Failed: DID NOT RAISE ValueError`:

- `test_group_roles_must_name_the_runtime_projections_in_row_order` at line
  118: swapped gate/up, duplicate gate, source shard aliases instead of
  canonical roles, and a gate role in `w2`.
- `test_gate_and_up_boundaries_must_match_the_runtime_equal_halves` at line
  128: the 32/96 and 63/65 row splits, both retaining the correct total.

Result: **6 failed, 10 passed**, serial on dl380g10, torch `2.11.0+cpu`:
`NO CUDA -- torch 2.11.0+cpu reports no CUDA device`; **0 skipped, 0 modules
not collected**. This run did not exercise the CUDA-gated surface. The
deployed worker retried the same deterministic failing tests three times.

## Green targeted runs

Action `554f3ce8ab744e6c42fac5f247186d9ab6265c81daaefa45f36deb287c9a78ed`
ran these files serially with a 1-CPU, 4-GiB reservation:

- `tests/test_serving_moe_scheme.py`
- `tests/test_export_moe_write.py`
- `tests/test_export_moe_layouts.py`
- `tests/test_serving_moe_route.py`

Result: **63 passed, 7 skipped, 0 modules not collected** on dl380g10,
torch `2.11.0+cpu`; `NO CUDA -- torch 2.11.0+cpu reports no CUDA device`.
The verbatim skip reason was **7 — `the encoder is a GPU job`**. This run did
not exercise the CUDA-gated surface. CAS receipt:
`9a7f2743fe80b8135ae064e8ed43802e460e60698c6111db99c71192e5c80a2c`.

Selector action
`c2db7e15b6751e761ce9b41d0c7db9b11d98cc8a4006722097173d0cba6bbc91`
fetched the exact branch Git bundle and ran `tools/impacted_tests.py --ref
307c0d15a9a56c98883058118862c4e383edc22c..HEAD --json` inside the PrismaBuild
snapshot. Its verdict was **narrowed, 45 test files**. The direct two-tree
comparison was explicit because the snapshot is parentless; this branch's
older selector still counts PrismaBuild's generated closure marker among
changed files. Its import graph selected the test files, not that marker.
Selector result blob:
`d46d89ee510777d0c08d049d09836de4881db1d84e164f4e9435c5d1234e531a`.

Action `c35c5ea31f95d0fa38316bc771c9a674f8fb05c861b76f4d557833e4927a4c2f`
ran the remaining 42 selected files plus `tests/test_audit_byte_baseline.py`
with `-n 2`, a 2-CPU, 4-GiB reservation, and one BLAS thread per worker.
Already-tested selected files were not repeated. Result: **604 passed,
293 skipped, 0 modules not collected**, dl380g10, torch `2.11.0+cpu`:
`NO CUDA -- torch 2.11.0+cpu reports no CUDA device`. This run did not exercise
the CUDA-gated surface. CAS receipt:
`00aee00b8883f8ad97f6a32e25c6c6d4c3ca764b9aeb3b601e983e2d8630c196`.
Population: `/mnt/shared/tessera-runs/ts5-moe-role-invariant-8ef072d-surface.json`.

Its skip reasons, verbatim:

```text
81  the encoder is a CUDA path
79  needs a CUDA device
29  the Viterbi is CUDA
24  the kernel lane is a CUDA path
23  the lane is a CUDA kernel
14  the Tessera encoder is a CUDA path
10  the encoder is a GPU job
 6  needs CUDA
 6  /home/rob/tessera-runs/compile-dispatch is not on this box
 5  the reach checkpoint is not on this box
 5  the kernel lane runs on CUDA
 3  Qwen3-0.6B is not on this box
 2  no stock twin
 1  could not import 'vllm': No module named 'vllm'
 1  the two surviving compile caches from 2026-09-02 are not on this box
 1  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box
 1  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box
 1  E2M1 publishes no reader range
 1  the shipped checkpoint is not here
```

No GPU jobs or served quality measurements were performed for this fix. These
CPU tests establish schema refusal and existing CPU behavior, not MoE quality
promotion. The coordinator still owns the merged-tree GPU/integration gates.

Separate off-task prose correction: scheme and test documentation now describe
one container per projection and distinguish the built expert route from a
packaged served-quality qualification. No decision or runtime predicate moved
in that prose-only commit.
