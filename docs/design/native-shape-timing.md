# Native shape-time receipt, first slice

Tessera #688's producer schema is `tessera.shape_time_panel.v1`. The first
implemented slice is exactly one E4M3 FP8 dense operator, with a uniform
rung, explicit TP1 rank0 native layer geometry, eager execution and resident
weights. Routed, TP2, multiple-row and alternate-family panels refuse.
This is an operator-sum proposal, not placement, serving p95, quality,
runtime-cell qualification or a format-menu change.

`tessera.serving.timing_panel.validate_panel` is passive and stdlib-only.
The caller supplies an independently frozen expected runtime. The receipt
binds the actual runtime and independent producer records, raw runtime
contract, canonical wire, native preparation, raw CUDA-event samples,
observed per-call routes, gzip profiler trace, native ELF binary and full
raw telemetry by absolute path, byte length and SHA-256. Its single scope
is rebuilt through the existing census planner and dispatch registry.

The canonical fused frame reader is shared with `tessera.fused`; existing
framing bytes and public names remain unchanged. The passive replay uses
`container.parse` to verify manifests, plane digests, geometry, rung,
body, plane and span. Hardware grid/profile reconstruction remains the
ordinary native loader's responsibility; its preparation evidence is
bound to the same wire and actual runtime. The validator does not expand
weights or simulate a device.

A positive row requires exactly one backed cell matching image, package
commit/source, versions, platform, regime, residency and rung. It must
declare Tessera's plugin, satisfy required flags and name the actual
`(symbol, decoder)` pair. The shared lane wire predicate and the existing
census launch-agreement rule must both pass with positive coverage.
Unattested/unsupported agreement cannot become a timing price. Each timed
call must have a fresh served record with its actual shape and activation
contract; missing, error or changed routes refuse.

At least three finite positive CUDA-event samples are required. The median
uses `statistics.median`, and quartiles use
`statistics.quantiles(method="inclusive")`; cached summaries are compared
to the raw samples. Profiler evidence requires actual CUDA kernel events.
Both Sparks' raw power/CPU/load/swap/available-memory contexts and fast
power samples retain the timed interval. Energy remains `hold` for
cross-host clock alignment (PB #1440); no work/J claim is accepted.

Every new Python source changes the package source identity. Current
b40/v42 panels are not rewritten, and a new runtime/census must be frozen
and accepted before a native GPU pilot or any measured panel is admitted.
CPU stand-ins validate this contract without claiming GPU execution.

`tools/tessera_shape_time_panel.py` is the supported repository application.
`check-request REQUEST.json` performs a CPU preflight without preparation or
measurement. `measure --request REQUEST.json --output NEW_DIRECTORY` calls
`lane.build_tessera_method`, `create_weights`, loads the exact owned wire into
the plugin parameter, and calls `process_weights_after_loading` and `apply`.
It clears the latest route before every call so retained records cannot stand
in for a launch. Output geometry and an initial finite-output check must pass;
prepared-weight fingerprints and both source identities are checked again
before publishing the final receipt. These controls do not establish numerical
quality or end-to-end serving correctness.

The request schema is `tessera.dense_shape_time_request.v1`, with exactly
`schema`, `expected_runtime`, `scope`, `prefix`, `scheme`, `wire`, `sampling`
and `netdata_hosts`. `scope` uses the existing native census request fields;
`scheme` is the existing dense wire declaration and `wire` is a bound file.
`sampling` names integer `samples >= 3`, integer `warmup_iterations >= 1`,
integer `seed >= 0` and finite positive `steady_s`. `netdata_hosts` explicitly
names both `sparky` and `sparklina`. No topology or runtime is inferred.
Preflight requires the exact packaged contract digest and a possible positively
backed cell/wire join; the actual native pair must join again after preparation.

The actual imported Tessera package supplies its commit through a clean,
tracked Git checkout or VCS installation metadata owning that imported file.
The launcher image declaration, package source digest, raw contract digest,
Torch/vLLM versions, device platform and serving flags must equal the frozen
expected context. The repository tool has a separate Git commit and source
closure. Neither identity substitutes for the other. `native_packed_bytes`
records the existing prepared object's named tensor byte sum, including views;
it is not a deduplicated whole-process residency measurement.

CUDA-event sampling reuses `tools.a4_measure.time_call`; profiling uses
`torch.profiler`; fast power and raw Netdata reuse the existing instruments.
The requested steady interval supplies useful native calls for box observation,
not additional timing prices. Raw responses must contain result samples.
Every artifact is file- and directory-synced, and the final panel is validated
before its durable publication. Only then may repository tooling report one
committed PB publish unit. The package validator has no PB dependency.

This application has CPU controls only until a separately accepted native
GPU pilot qualifies the complete path. Its source changes invalidate the old
runtime-source join. No existing b40/v42 or other cell is relabeled, no census
is published, and no PrismaQuant timing consumer or pin changes in this slice.
