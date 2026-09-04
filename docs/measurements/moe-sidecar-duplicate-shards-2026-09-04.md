# MoE sidecar duplicate tensor ownership

The sidecar preflight previously combined shard headers by assigning each
tensor name into a dictionary. A second shard with the same name replaced the
first entry, so its later exactly-one-wire check could report `NO PROBLEMS`
despite duplicate physical owners. The reader now refuses at the duplicate,
naming the tensor and both shards. This covers indexed and unindexed exports.

PrismaBuild pre-fix action
`b4fc2b2dc53acf5f7f3b9d384d0610ff4ba70f1f61c1799b3a9e99101ff105ac`
ran the new regression against the unchanged reader. Both indexed and
unindexed cases failed at `tests/test_ts5_sidecar_check.py:75`:

```text
E       Failed: DID NOT RAISE ValueError
2 failed, 5 passed in 0.98s
```

Post-fix action
`2bdb2ceacbd104b3b09448c3bc5f7752bad58d0998a2c5d7babca673e2e9b850`
ran the same file: 7 passed in 0.97 seconds. CAS receipt:
`0cb0d3f9d186561b36526e578678cdadd5116dbdb1e83d9aa9240309287f12de`.
Both populations were serial CPU runs on dl380g10:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

The green population is recorded in
`/mnt/shared/tessera-runs/ts5/lfm25/astra-sidecar-duplicate-green-r1.surface.json`.
PrismaBuild selector action
`56fda01b8118ed7e40c75df79cdab367f6a60e03312bf93fc6b2399cfa2ddbd7`
compared its snapshot directly against fetched base
`894305ee41b648e563bc1fa672948f5f6cf8c4af` and returned `narrowed` with only
`tests/test_ts5_sidecar_check.py`. An earlier selector submission used an
unfetchable abbreviated Git ref and ran no selector or tests; it is not
validation evidence.
No byte layout or served-route attestation changed.
