# Changelog

## 2026-10-05 — issue964: the serving source pin stamps instead of refusing (D32)

The native shape-time pipeline's run-identity gates no longer refuse a serve
whose running Tessera tree differs from the frozen expected context. The
worker's before-device code identity and frozen-software comparison, the
installed CPU preflight's recorded commit and observed software context, and
the panel/observation validators' observed-vs-frozen runtime now go through
the new stdlib `tessera.dev_mode.seal_check` (the twin of
`prismaquant.dev_mode`, PQ #1147): dev mode is ON unless
`PRISMAQUANT_DEV_MODE` is exactly `0`, a mismatch prints one `[DEV-MODE]`
line naming both values and the run continues with the stored data, and
certified `0` raises each site's verbatim refusal. The GLM MTP draft shard
narrowing (`mtp_draft_shards.install`) treats its inspected loader source
digest the same way: default dev mode stamps one `[DEV-MODE]` line per
module and still installs the narrowing (the 164 GiB whole-checkpoint
fallback becomes the certified-mode decline); an unreadable source and a
rebound loader signature still fail closed in both modes. A dev-mode
installation proof verifies intact installed bytes at the commit the running
tree claims instead of the pinned one, so the receipt stays honest without
re-imposing the stamped pin. Still refusing in both modes: the installed
`runtime_contract.json` bytes against their own pinned digest, preflight
module byte bindings, every evidence/bound-bytes check, foreign runtime
origins and import-environment isolation, mid-run stability re-observations,
and the fail-closed VCS commit resolution that names what ran. Regressions in
`tests/test_native_timing_panel.py` and `tests/test_native_shape_time_application.py`
demand stamp-and-continue on a source-pin difference (failing on the
pre-change refusal) and the verbatim certified refusal; `tests/test_dev_mode_seal_check.py`
pins the helper. `docs/ARCHITECTURE.md` is re-stamped in the same commit.

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
