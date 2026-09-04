# Whole-layer serving export and checked assembly

Code: `0f71fd1` (tests), `59b8d44` (partition/assembly), `8b18eed`
(shared direct-export/merge accounting). Qualification tree: `8b18eed`, with
the LFM construction receipt from `7e32950` included as `10401cb`.

The new `--partition INDEX/COUNT` uses the existing serving exporter and its
full plan. Layer `L` belongs to `L % COUNT`; non-body tensors belong to index
zero. Both workers validate the same complete plan, and only execution is
partitioned. No expert stack or fused dense module is split. The LFM routed
layer interval 2 through 23 assigns eleven stacks to each of two workers.

A part writes only its owned tensors and has `tessera_part_config.json`
instead of a loadable `config.json`. The existing `merge_tessera_parts.py`
entry point recognizes serving parts and proves the complete partition roster,
source-tensor ownership, source/config/tokenizer hashes, encoder source and
behavior fixture hashes, plan/options and dispatch image identity, and exact
output-index/header/hash agreement before creating an output directory.
It copies the tensors unchanged into uniquely named shards and writes the
final config last. The assembled safetensors envelope may have different
header overhead from a one-file export; the wire containers are unchanged,
and `checkpoint_bytes` records the actual envelope separately.

The image field records the digest pinned by the dispatch command, not an
independent observation of the container. PrismaBuild's campaign receipt is
the runtime evidence. This CPU qualification is not a served MoE quality or
GPU-performance result.

## Pre-fix evidence

The complete added test file was cherry-picked onto the unfixed `307c0d1`
tree, without the implementation. PrismaBuild action, identified by its unique
queue prefix `ed4f857e1001`, failed collection with:

```text
E   ImportError: cannot import name 'serving_parts' from 'tessera'
```

Before the exporter hook existed, the helper tests already passed and the
exporter integration test failed independently in action `23823302494c`:

```text
python -m pytest: error: unrecognized arguments: --partition 0/2 --partition-runtime-image test/image@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
FAILED tests/test_serving_parts.py::test_exporter_writes_only_owned_tensors_and_withholds_loadable_config
1 failed, 16 passed in 1.16s
```

Both were dl380g10 CPU runs: torch `2.11.0+cpu`, no CUDA, zero tests skipped,
zero modules uncollected. Collection failure is a failure, not coverage.

## Positive evidence

`test_partitioned_expert_wires_equal_one_process_export` performs actual CPU
encoding of two complete synthetic expert stacks. It assembles the two parts
and compares every output tensor bit for bit against one direct export, along
with exact module records, config, routed-MoE manifest, wire bytes, container
bytes, parameter counts, per-family accounting and passthrough bytes.
Action `e18ea0fe3ffe9332e7ef70b5ed909a99f03e7614be6f586709d5fe78a0d13045`
passed all eighteen partition tests in 46.75 seconds: CPU torch `2.11.0+cpu`,
no CUDA, zero skipped tests and zero uncollected modules.

The impacted-test selector used the exact fetched base
`1cc6e1a0acda76918441884387f07b7cae872241` against its parentless snapshot.
Action `3b7079bcd4908639ff4ba205064182e2d28fa3324138d8cd5941f2221d1b31a9`
reported `narrowed` and a direct tree comparison. Its eleven selected files
were run with the MoE layout/write tests, legacy merge guard and byte-baseline
audit added. Action
`be5d7b420f9834fd0f8e715c3d9f8af76b7364e75fd36460de77b4ad64585f94`
was green at **303 passed, 57 skipped, zero modules uncollected**, in 115.94
seconds, CPU torch `2.11.0+cpu`, no CUDA. The submitting observer timed out
at its 120-second wait boundary; the finished worker record supplies exit
status `0`, and its CAS receipt is
`e0fde8160d237b59dd2b7cd90df7f394efb8ecf4da0ace6406a9d48754ed113a`.
Skip reasons, verbatim:

```text
37  encoder is a GPU job
 7  the encoder is a GPU job
 6  /home/rob/tessera-runs/compile-dispatch is not on this box
 1  the PrismaQuant tree with tessera_formats is not on this box
 1  the PrismaQuant tree or the allocation outputs are not on this box
 1  Qwen3-0.6B is not on this box
 1  the two surviving compile caches from 2026-09-02 are not on this box
 1  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box
 1  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box
 1  E2M1 publishes no reader range
```

After the final shared-accounting change, action
`2ef623ab54428167e3281ebedf65e7d44370399bb278509b503fe05da2e910ce`
ran the partition tests, export construction gate, ignore completeness and
serving export gate. Result: **42 passed, 2 skipped, zero modules
uncollected**, in 59.69 seconds, CPU torch `2.11.0+cpu`, no CUDA, observed exit
status `0`. CAS receipt:
`17c62a1e10d457d954222bed11adc17729109f5981eda0b939ae3384c37470c4`.
The two skip reasons, verbatim:

```text
1  the encoder is a GPU job
1  E2M1 publishes no reader range
```

All execution used the deployed PrismaBuild v1 worker generation
`aa6d3cfa2f77-1788542034-2b84265567ac`, with four declared CPU cores and
eight GiB memory. No PrismaBuild source or deployment change was needed.
