# Changelog

## 2026-10-05 — issue953: bounded preflight headroom wait (Window4)

`LocalArm.preflight` now waits up to 900 seconds for the unchanged 114 GiB
`MemAvailable` predicate to become genuinely true before each arm, in graph
and eager modes alike, instead of refusing on the first synchronous sample.
The wait polls on the existing `run_rank` cadence, checks the live guard on
every poll and again immediately before declaring ready, and is capped by the
existing window envelope (cleanup reserve and any tightened peer absolute
window), so a shortened deadline still refuses before any model work. Expiry
raises the original unchanged refusal text; the 16 GiB physical floor and
guard cancellation still refuse inside the wait; the model never launches
below 114 GiB. One terminal wait report per call — threshold, 900-second
bound, initial/last GiB, wall/monotonic times, every exact sample, and reason
`ready`/`headroom_timeout`/`lifecycle_cancelled`/`deadline`/`error` — is appended
to `rdv/headroom-preflight-rank<rank>.jsonl` and persists through refusal,
cancellation, deadline or unavailable readings (unknown values stay unknown).
The model-start boundary rechecks the same 114 GiB bar after source checks
and the peer barrier, recording a per-arm synchronous sample before container
work; an earlier ready wait never authorizes a below-threshold launch.

Deterministic regressions (`tests/test_graph_attest_headroom_preflight.py`)
drive the real common preflight path on a shared fake clock: below-to-ready
for both ranks in both modes, 900-second expiry with the original refusal and
no model command, cleanup-reserved, tightened-peer and already-expired
deadlines, guard peer cancellation, the 16 GiB floor inside the wait, and
guard latency past the bound without a negative sleep. The below-to-ready
regression fails on the pre-change behavior, which refused instantly with no
wait report. `docs/ARCHITECTURE.md` is re-stamped and
`experiments/graph_attest_702/RUNPLAN-artifact.md` documents the wait in its
resource contract.

No threshold, floor, cache/KV/shape, artifact/runtime/client semantic, cap or
admission predicate changes. The unchanged 104/102 GiB demand caps mint no
114 GiB admission, the wait produces no performance or fit evidence, and no
approval of the historical frozen producer's source transfers to the new
producer head.
