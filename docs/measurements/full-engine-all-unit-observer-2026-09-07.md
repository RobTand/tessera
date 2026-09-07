# Complete native-boundary observation and reference binding — 2026-09-07

This records implementation validation for Tessera #399 / PR #400. It is not a
full-model fixed memory or timing price. The earlier source-BF16 R4 allocation
capture and its unresolved ownership are retained separately in
`full-engine-resource-r4-2026-09-07.md`.

The all-unit observer resolves the canonical census to dense native `apply`
methods and routed-expert `apply` methods, excluding the router. A shared method
object dispatches by actual layer identity; missing, ambiguous and monolithic
methods refuse. Resource boundaries observe actual input/output backings and
reference-model parameters/buffers, while same-object aliases registered outside
canonical native modules remain fixed. Storage-category conflicts still refuse.

A separate timing worker records adjacent intervals on the same CUDA-event
sequence, explicitly joins main-stream and stock asynchronous copy completion,
and checks unique profiler launch ownership and observed streams. It never
subtracts independently measured operator medians or summed kernel durations.
The binary32 representation bound only checks event-interval recomposition.
Collection health, observer overhead, full resource ownership and final runtime
admission remain unqualified; both fixed-resource and timing prices remain null.
The paired profiled control/partition runs use one identical warmup, cold prefix
state, the same 512-token input, and exactly two identical generated tokens.

The reference verifier binds all original-wire members and a closed checkpoint
file roster to the exact source identity and canonical census. This checks the
export statement and bytes, not the exporter/PB provenance admission. The final
reference export remains a dependent artifact owned by the original-wire export
work; a source-BF16 observer qualifier cannot replace it.

Validation through PrismaBuild:

- Action `8e7226c5e39c1ea53aeb8f4d38743e3fb5e615347e2f5599285844cc7e7044c5`:
  compile checks for all seven changed/new Python implementation modules, then
  the reference, boundary, timing, resource and worker tests. **100 passed,
  0 skipped, 0 modules missing** on dl380g10, Torch 2.11.0+cpu; no CUDA surface
  was exercised. Six xdist workers, worksteal scheduling, six reserved CPUs,
  8 GiB aggregate memory, native threads bounded to one.
- Terminal return code 0 and complete resource cleanup verified. CAS payload
  `57ceb9913a18af11f8203b56458b860e81db184e619048341d00a3d10386521c`
  was read and independently rehashed. Terminal record:
  `/mnt/shared/prismabuild-fleet/pb-queue/done/8e7226c5e39c1ea53aeb8f4d38743e3fb5e615347e2f5599285844cc7e7044c5.json`.
- Source-BF16 timing qualifier action
  `997eaf62bbe316a4c8d1c2d880f206e4f12fea6c802876f455e6744bf09256b4`
  was still queued when this implementation record was written. No GPU timing
  partition or observer overhead result is claimed here. Actual captures and
  subsequent findings must be recorded as additional dated evidence.

The compile command was `python -m compileall -q` over
`full_engine_reference.py`, `full_engine_timing_boundaries.py`,
`full_engine_timing_worker.py`, `full_engine_timings.py`,
`capture_full_engine_resources.py`, `full_engine_resources.py` and
`full_engine_worker.py` under `experiments/`. The test command was
`python -m pytest -q -n 6 --dist worksteal` over the corresponding five test
files (the timing worker and launcher use the worker/boundary tests).

## Actual stock zero-token cleanup regression — later 2026-09-07

The queued source qualifier `997eaf62bbe316a4c8d1c2d880f206e4f12fea6c802876f455e6744bf09256b4`
ran after fleet recovery. At 14:52:32 UTC its first observed scheduler call had
zero scheduled tokens and a finished warmup request ID. Stock model runner
`v1/worker/gpu/model_runner.py:1575` updates/frees request state and returns
without a model forward for that call. The timing worker incorrectly consumed
it as the expected 512-token prefill and raised. The live engine exception is
the pre-fix regression; no timing partition was produced.

The observer now retains an explicit housekeeping CPU range for zero-token
calls without consuming a prefill/decode step. A GPU operation in such a range
still refuses the partition because it lies outside the two measured steps.
It is not silently assigned to a fixed bucket or removed from the GPU trace.
PB `e708b3ab81fc1161aa62ccaf4cd68978d84319c4c7976ef1825729bc89bf719d`
passed **27 tests**, 0 skips/0 missing, including cleanup-state and unexpected
cleanup GPU-work cases, on dl380g10/Torch 2.11.0+cpu with four workers and bounded
native threads. Exit 0, complete cleanup and independently rehashed CAS payload
`3ab2352539e9e05f2b6b6068d6390071c050e09ad7224c38ceec782306b50bad` verified.

The failed GPU container outlived its crashed engine worker. After retaining
logs and exact container/PB ownership, it was stopped; the launcher recorded
exit 137, removed that container and captured all ten Netdata series across
both hosts with no missing series. PB recorded failure and complete resource
cleanup, with no successful CAS payload. Artifacts, the original traceback,
pre-stop inspect, stop reason, and telemetry are retained under
`/mnt/shared/tessera-native376-resource/full-engine-timing-r1/`. Netdata index
SHA-256: `e5f1fe898cc39ecdb188a3c6ce005f5bedb86bc8356ce7634d7c10cfd2d3edf2`.
This is an observer integration failure, not a runtime performance result.
