# Capture process ownership

Lead decision, 2026-11-17. More than one compute process does not prove that a
foreign process used the GPU. Capture accepts worker descendants. A foreign or
unresolved compute PID refuses telemetry success.

Source F is `272507f18de39fc90f03c1588373c031a9e43717`.
IMG-N is `acc684c06ec7`. The source reads below did not execute the capture.

| Path | Source read | Process consequence |
|---|---|---|
| Torch extensions | F `src/tessera/routed_fused.py:618,634`; IMG-N `torch/utils/cpp_extension.py:2859,2893` | Torch runs ninja. Compiler children can exist. Ancestry decides ownership if one appears in NVML. |
| Profiler and CUPTI | `dense_phases.py` uses `torch.profiler.profile`; IMG-N `torch/profiler/profiler.py:294` enters the native profiler | Profiling starts in the worker. This source read does not prove that all native helpers stay in process. |
| Library and kernel rosters | `dependency_code.py:78`, `build.py:103`, `serving_build.py:100` | cuobjdump reads library files in subprocesses. c++filt also runs as a child. These calls do not request a CUDA launch. |
| Tessera and vLLM imports | F extension loader above; IMG-N `vllm/__init__.py:14`, `vllm/distributed/parallel_state.py:1872` | Imports reach native and optional dependencies. A CPU trace cannot prove that no CUDA child can start. |
| TP2 cuts | `dense_fused_oracle.py:130-143`; the group loop in `dense_phases.device_capture` | One worker creates a world-size-one gloo group. The layer holds each cut's TP size and rank. Cuts run sequentially. |

IMG-N has torch `2.13.0+cu130`, Triton `3.7.1`, and vLLM
`0.30.1rc1.dev336+gaf5b4857e.d20260929`. The inventory binds their library files.
No claim that all capture paths create only one process is needed.

The host launcher reads `podman inspect --format {{.State.Pid}}` after container
start. The payload uses `exec`, so that PID becomes the Python worker. The host
records the PID and `/proc/PID/stat` start ticks before the worker imports torch.
The worker waits for this record. No environment input is added.

The container gets read-only mounts of that record and the host `/proc`.
The sampler reads only `stat` for ancestry. It checks the worker's start ticks
to detect PID reuse. Missing, changed or cyclic ancestry is unresolved.
The source read on sparklina found no `hidepid` option on `/proc`.
The receipt records total, owned, foreign and unresolved process counts.
Raw NVML PIDs remain a single diagnostic list. No container-PID equality test is used.

The worker captures its start environment before importing torch. Later launch
checks compare captured inputs and math settings. They do not repeat the fresh
import refusal. The early CUDA allocation therefore does not invalidate the worker.

GB10's unavailable device-memory reading remains null. The preflight records the
lead's exact waiver. The host's vLLM check is authoritative. The worker records
its limited PID namespace. Compare the raw utilization before and after context
creation with phase power and activity before using GPU-busy time for bookings.
