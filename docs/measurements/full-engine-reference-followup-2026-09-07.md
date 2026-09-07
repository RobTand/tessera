# Original-wire full-engine follow-up — 2026-09-07

This follows PR #400 and does not replace its actual refused R3 resource
receipt or admit a fixed-resource/timing price.

## Allocation diagnostic

The actual R3 receipt refuses an inapplicable source-BF16 FlashInfer owner
rule before deriving checkpoint ownership. A separate CPU replay removed
only that rule and its recorded error from an in-memory copy, first asserting
that the exact required library was absent. The original capture and receipt
remain unchanged. This hypothetical input is diagnostic evidence only.

PB `556738ea0c0ef65459bd4bbe9b0990e7e1cf4d90dd6c68075d9b02ed78063d88`
ran on Lina with one CPU, 24 GiB aggregate memory, no GPU and native threads
bounded to one. Exit 0, cleanup/release, CAS receipt and payload hashes were
independently checked; receipt
`ca41323136ad8a34fab2df4a6ffbcb5a02366be5a2441512dd3f931a02c5f956`.

The diagnostic derived 165 checkpoints and 74,062 allocation generations.
It remains incomplete: 121 allocations lack owners, 4260 repeated checkpoint
observations report unmatched host buffers, and 4537 external CUDA records
remain unattributed. These failures survive omission of the inapplicable rule.
No resulting category total is an admitted full-engine fixed price.

Artifacts are under
`/mnt/shared/tessera-native376-resource/full-resource-diagnostic-replay-r1`.
`input-transform.json` records the exact transformation and source capture
SHA. The result, 85,829,691-byte diagnostic ledger and cProfile are retained.
All result-declared artifacts were rehashed.
`verified-diagnostic-summary.json` has SHA-256
`95ca69bb1d45b4757b88f9b442e482b992e3d379bcf1fd67787e0c1070776b92`.
The diagnostic does not establish an engine speedup or resource admission.

## Independent timing capture

After the replay and other admitted Lina tests completed and released their
resources, a new direct vLLM timing capture started at 18:07:03 UTC on Lina.
It uses the same immutable source `3340b253`, exact original-wire configuration
and calibration as R3, without the intrusive resource collector. Before launch
Lina had no GPU process/container and 120,908,692 KiB MemAvailable.
The owned container and both process/host telemetry are recorded under
`/mnt/shared/tessera-native376-resource/full-engine-reference-timing-r1`.

Startup loaded the 4.78 GiB checkpoint in 0.58 seconds; a bounded stack sample
then observed the older CPU `wire.unpack_body` path inside native MoE
`process_weights_after_loading`. This is one startup sample, not a saturation
or performance claim. The pinned package deliberately remains the same as R3;
it does not include the separate newer reader work. The result is outstanding.

The timing capture finished at 18:11:13 UTC, retaining both arms and their
profiler traces. Each arm produced the same two output tokens as warmup.
The partition arm contains 512-token prefill and one-token decode, with all 38 units and 39
adjacent gaps per step, including the stock asynchronous output-copy join.
Its original analyzer refused `missing GPU operations or unique CPU observation
ranges`, so the launcher/container exited1; the owned container was removed.
All 4967 stock vLLM core files and the sealed native cache entries stayed
unchanged. The raw failure is preserved, not rewritten as a successful launch.

The actual trace contains 79 CPU `user_annotation` ranges and 80
`gpu_user_annotation` projections sharing their names, alongside 1066 GPU
operations. The analyzer selected named ranges without checking category,
so it miscounted the GPU projections as duplicate CPU observations. This is
a parser defect, not missing raw GPU activity. The exact partition-arm inputs
are capture SHA-256
`c24df65ab485a5726550ca3b386a891dc0d7e3fc1082451a342da3701a2cbe13`
and profiler SHA-256
`d946eb14637086489710c825403d2406f432c16457f7cf35ff2081107520b466`.

The timing audit, covering 18 input/output/provenance files, original refusal,
container identity/exit and Netdata sample quality, is
`full-engine-reference-timing-r1/timings/verified-timing-audit.json` under
`/mnt/shared/tessera-native376-resource`, SHA-256
`36f969cfd947251dc012bbfe23963d9be3bee6b3b61ca1de75e3cce999a02302`.
All ten Netdata files were acquired/rehashed; CPU/RAM dimensions have 252 valid
samples each, framebuffer and memory clocks have zero, and GPU core clocks/
power have 239 valid samples and 13 nulls each, on both hosts. No missing
observations were interpolated.
