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

## Actual default-stream context qualification — later 2026-09-07

Corrected source run `34cbdd270e205cda1abd178d1f00a3bcaa386f628ccde791beb72ac76602b6da`
passed the zero-token cleanup path, then refused CUPTI's stream-ID query with
code 20 for the stock CUDA default stream. It exited 1, PB recorded complete
cleanup, and all ten Netdata series were retained with zero missing series
under `full-engine-timing-r2`. No timing run or successful CAS payload exists.

A tiny stock-image GPU qualifier isolated this without another model load:
`753b2dbffc0cd6fb930ac7cd97d25de3cf80f96bdac161021afe2f9fa702f67c`.
CUDA stream handle zero plus a null context returned code 20. Passing the actual
`cuCtxGetCurrent` context returned stream ID 7, matching its profiler kernel.
An explicitly created stream returned ID 13, also matching the profiler. The
observer now queries the active context and asks CUPTI to verify membership;
missing contexts and failed queries still refuse. No stream ID is guessed.

The exact fixed observer function then ran in PB
`deea806b620e6da2e1fa9923816282c9e15fb38e3a30bab1f58ea6b8958ec1f5`,
returning IDs 7 and 13 for the default and created streams and matching both
recorded GPU operations. Exit 0, complete cleanup and CAS payload
`72383ad8140bb92e3a46d22ab56add4c5084e56945dfc01f5db7c2d63245a6f9`
were independently verified. Result SHA-256
`c6f6165ba64b1e2cb18a6123b0f01849a248b1ddad93c8ad8bd3c509978c292b`,
profile SHA-256
`2214eb16ca5ea2c3869dd98d6f6827c12b88e0c0f5608fe8f9fa13c3d7b82719`;
files are under
`/mnt/shared/tessera-native376-resource/timing-stream-qualification-r2/`.
Both tiny checks used the selected immutable stock image on GB10, two reserved
CPUs, 4 GiB aggregate memory and a 2 GiB GPU subset. They qualify only this
stream lookup, not timing overhead, profiler completeness or full-engine cost.
An earlier attempt `02eb556c1f83` failed before CUDA work because its root user
could not write the NFS output; the checks above ran as UID/GID 1000.
