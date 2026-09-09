# Multi-stack research selected MoE declaration control, 2026-09-09

Issue #442, bullet 1 preparation. CPU only. This record covers the declaration
and dispatch half; it establishes nothing about load, generation, TP2
collectives or any measured serving result.

## What it checks

#432 validated the JSON to config path at TP1 against one fixture stack. One
stack cannot separate "checks the first routed target" from "checks every routed
target", and a real LFM2.5-8B-A1B checkpoint declares 22. Two tests close that
gap in `tests/test_serving_moe_dispatch.py`:

- `test_every_declared_stack_selects_the_packed_owner` declares all 22 stacks,
  `model.layers.2` through `model.layers.23`, at `TESSERA_FP8`/`E4M3`/q1024 with
  32 experts, and asserts every one dispatches to `build_tessera_moe_method`
  with the same `ResearchSelectedMoeConfig` object, 22 calls, one execution
  identity across all of them rather than one per layer.
- `test_one_off_route_stack_among_many_refuses_before_loading` puts a single
  off-route stack at position 0, 11 and 21 among the 22 and requires
  `require_targets` to refuse from any position.

The off-route stack is `TESSERA_NVFP4`/`E2M1x2`/`TCQ`/`LUT` at q256 896. q256
1024 was tried first and `validate_tessera_moe_scheme` refused it earlier, at
the rung range this build's decoder reads for `TESSERA_E2M1_K2`, which is
`[896, 896]`. That refusal is correct but it is not the one under test, so the
fixture moves to a rung inside the range and the refusal under test is the one
that fires.

## Result

Tree: `claude/issue-442-moe-serving-qualification`.
Target: dl380g10, tag `x86`, `/home/rob/venvs/pq-cpu312/bin/python`,
torch 2.10.0+cpu, no CUDA device. Priority -10, two shards, two workers each.

| shard | files | result | receipt |
|---|---|---|---|
| 0 | `tests/test_serving_moe_dispatch.py` | 48 passed in 3.76s | `a0189b64e8056e9b5a6a1d3e83ba207865e7d4cfe052978fc48c840900631d6b` |
| 1 | `tests/test_moe_execution.py` | 14 passed in 3.52s | `9e64a1bb83954c51681a74b751871bcce40567e963cc9501bb521461a8795402` |

Receipts live under `/mnt/shared/prismabuild-fleet/pb-queue/done/`. Zero tests
skipped and zero modules uncollected in both shards. The run did not touch the
CUDA gated surface, so its pass count is not coverage of that surface.

## Limits

This is a stubbed vLLM surface, the same isolation the surrounding file already
uses. It does not load a checkpoint, does not construct a real
`prepare_tessera_packed_moe_experts` geometry, does not generate text, and does
not exercise TP2. `expected_tensor_parallel_size` 2 appears here only as a
declared value that reaches the builder, never as a multi-rank serve.
