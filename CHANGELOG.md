# Changelog

## 2026-10-05 — issue953: bounded preflight headroom wait (tests/docs)

Tests and documentation for the CEO-authorized bounded headroom wait in
`LocalArm.preflight` (source: sibling commit `6a25d230c`, Window4). The wait
gives the unchanged 114 GiB `MemAvailable` predicate up to 900 seconds to
become genuinely true before each arm (graph and eager), polling on the
existing `run_rank` cadence with the live guard checked every poll, capped by
the existing window envelope; expiry raises the original refusal text, the
16 GiB floor still refuses inside the wait, and one terminal wait report
(threshold, bound, initial/last GiB, times, exact samples, reason
`ready`/`headroom_timeout`/`lifecycle_cancelled`/`deadline`) is appended to
`rdv/headroom-preflight-rank<rank>.jsonl` per call.

- New deterministic regressions: `tests/test_graph_attest_headroom_preflight.py`
  (fake clock/meminfo reader over the real common preflight path; below-to-ready
  for both ranks and both modes, 900-second expiry with the original refusal and
  no model command, cleanup-reserve/tightened and already-expired envelope
  deadlines, guard peer cancellation, 16 GiB floor breach, guard latency past
  the bound without negative sleep). The below-to-ready regression fails before
  the change, which refused below threshold instantly with no wait report.
- `docs/ARCHITECTURE.md` re-stamped; `experiments/graph_attest_702/RUNPLAN-artifact.md`
  resource contract extended.
- No threshold, floor, cache/KV/shape, artifact/runtime/client semantic, cap or
  admission predicate changes; the unchanged 104/102 GiB caps mint no 114 GiB
  admission, and no performance, fit or source-approval claim is made or
  transferred. Tests were not executed by this change; integrated runs are the
  parent's.
