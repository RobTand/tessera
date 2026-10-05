# Changelog

## 2026-10-05 — issue959: D30 Window4 107 GiB admission and dual-rank memory abort

Window4 per-host `MemAvailable` admission moves from the historical 114 GiB
predicate to 107 GiB on both hosts — the retained conservative hybrid model/KV
estimate 98 plus the explicitly unmeasured graph/host allowance 6 plus reserve
3 — with the 104 GiB host / 102 GiB GPU-subset caps unchanged. The 98 is
derived, not measured: the larger of the 94.4/96.1 GiB hybrid estimates plus
the 1.625 GiB KV uplift from 384 MiB to 2 GiB per rank, rounded conservatively
up, and those hybrids come from the graph arm's MemAvailable ready drop and the
eager L512/c4 transient-KV assumption, not from measured 384 MiB peaks. Engine
RSS and context sizes were never recorded and stay unknown; cgroup charge,
MemAvailable and GPU-used quantities overlap, so no naive sum of them is a
measured decomposition. The 900-second bounded headroom wait and the
model-start recheck are unchanged against the 107 GiB bar. The old 114 GiB
predicate and its sampled 16 GiB floor are historical only: the b5 run recorded
under them stays immutable. Historical derivation is documented in
`/home/rob/fleet/inventory/kernels-window4-headroom-equation-packet-20261005.json`
(SHA-256 `1a28bff770790a09b0df4573d46acf31e44aa12c6118cf31b9a32a279ffdb4f8`).
A new 1 Hz whole-box guard fails both exact-owned ranks
when either host samples strictly below 2 GiB. The local comparison precedes
shared queue/journal work; exact-owned TERM starts before peer publication and
the immutable first FAILED marker is published before any termination grace.
Termination/publication errors remain secondary to the original trigger.
SIGKILL follows 10 seconds if needed, shortened only by the finite deadline;
cleanup-only inspect/signal operations retain a fixed five-second rescue budget
after expiry without renewing model work. The rendezvous itself is not
terminated, and both ranks record a failed-cleaned acknowledgement so an early
failed-rank exit cannot cause a native withdrawal that shortens the peer's
10-second grace. A sampled guard is not continuous immunity.
The failed-cleaned wait checks whether the peer still owns its claim, exits on
peer death and reserves five seconds for broker stop; missing custody remains
explicit and is never reported as physical handoff.
`window_driver.py --prepare` emits `rdv/memory-policy.json` once and binds its
SHA into the submitted inputs; each rank emits
`rdv/memory-samples-rank<N>.jsonl` and `rdv/memory-summary-rank<N>.json` —
baseline, minimum, raw samples and counters kept so a future window can measure
the 6 GiB allowance, not a measured RSS decomposition — and the Envelope
outcome carries the termination list with container termination evidence.
Protected sysctl, ARC, cache and service settings are untouched. No
performance, fit or pin claim is minted, and old
producer approvals do not transfer: pool review must check both the derivation
and the guard against the exact new producer head before one fresh native
gang. The failing-before behavior regression and the admitted CPU physical
guard smoke are required by the issue and live with its source and test
slices; this entry records the contract they enforce.

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
