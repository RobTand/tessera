# Merged shard read permissions — 2026-09-04

Producer fix: `fa29270`. The completed assembly adds owner/group/other read
bits to each destination shard after transfer, without adding write or execute
bits. Default copy assembly leaves source-part permissions private and keeps
both source and destination payload bytes unchanged. Encoder outputs, model
bytes, runtime image, profile and quality thresholds do not change.
`docs/ARCHITECTURE.md` was updated in the same source commit.

## Actual failure

The full-model census failed while vLLM inspected safetensors metadata, before
the forward. The preserved log
`/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2/census-bound-r1/action.log:45`
records:

```text
PermissionError: [Errno 13] Permission denied: '/mnt/shared/tessera-runs/ts5/lfm25/astra-campaign-r2/full-model/part-00001-model.safetensors'
```

The coordinator observed mode `0600` on both assembled shards. `copy2` had
preserved the private encoder file mode, so the root-squashed serving identity
could not read the published artifact. The coordinator owns the separately
scoped repair of that existing artifact and its full-byte before/after seal;
this commit changes future assembly, not the live files.

## Red and green

All tests ran through deployed v1 PrismaBuild on dl380g10, one CPU and
`mem_gb=4`, with OMP/MKL/OpenBLAS thread counts set to one. No local compute or
GPU workload was run. Every population reported verbatim:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

Skip reasons were `{}` and uncollected modules `[]`.

Action `ae98dcf5bdad` ran the three new
`test_merged_shards_add_read_bits_without_changing_private_parts` cases on the
unfixed source. Real input modes were `0600`, `0400`, and `0640`. All three
failed at `tests/test_serving_parts.py:107`; pytest rendered the modes in
decimal:

```text
[384]: AssertionError: assert 384 == (384 | 292)
[256]: AssertionError: assert 256 == (256 | 292)
[416]: AssertionError: assert 416 == (416 | 292)
```

The same three cases passed after the fix in
`3b4a3b90568ac1e707517544207156de3b1968d06684b54b45962f81e0e7cb5c`,
3 passed and 24 deselected in 1.16 seconds; receipt
`a82a161247cb054b1cc21429575841e79eaff77e0ecaa044328d117b694d09c6`.
Each case checks the exact new destination mode, unchanged destination/source
bytes, and unchanged private source mode.

Full targeted-file results:

- `tests/test_serving_parts.py`: 27 passed in 112.43 seconds, action
  `aa2302e6bbbe43f41922c92de287894a31b36d7602d5793c1aeddda5f2620c7a`,
  receipt `a86efa0661e363077da0b4831f6abdcced3765cdf0f4815bc9459d7d73c660b7`.
- `tests/test_ts5_census_check.py`: 60 passed in 1.89 seconds, action
  `0e5e569377a95c470e27b77480cd7189c9bdd70f1deb1131474bd0e4374dbdd9`,
  receipt `40fae247e7fdb0013afd34f3629ac84855963dd6e9e51ce2587881b11be1c48a`.

Selector action
`b67d71a6d1a9e6a0e522168c45b4c8033fa01826cfc1271c3a90838a965b1da5`
compared the immutable snapshot against fetched exact base
`a86a67044233364a61157a64fc312f2424831388` using its parentless direct-tree
fallback. It returned `narrowed` with exactly those two files; receipt
`c500363992f4daccf1128a1ce78b15b6452f1a517dad89674c3ae0fe3e0cea9e`.
Population files are under `/mnt/shared/tessera-runs/ts5/lfm25/` with the
`astra-merged-read-bits-` prefix. This is CPU mode/byte-preservation evidence,
not a successful served census or a MoE promotion.
