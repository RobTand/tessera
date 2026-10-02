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
