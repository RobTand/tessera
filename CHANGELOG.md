# Changelog

## 2026-10-08 — shared geometry in the native footprint fixture

The native E2M1 footprint fixture uses its existing `window_geometry.TILE_ROWS` import.
The manifest byte comparison stays unchanged. No kernel module alias is added.

## 2026-10-08 — native T4 accounting callers

The offline planner and manifest refresh use explicit WINDOW role metadata.
The planner requires window bits and the shared row tile, with no TCQ fields or Torch import.
The refresh tool verifies span-one LUT WINDOW wires and preserves source bytes through hard links.
Both callers retain the shared native accountant and its shape and byte checks.

## 2026-10-08 — native projection byte audit integration

The projection audit selects the served recipe before it prices native tensors.
All three weight families use the same 256-column fixture.
T4 WINDOW roles use the shared row geometry and retain no TCQ trellis tables.
The audit records source columns, so direct buffer checks derive bytes from the actual tensor shape.
The source, wire, native resident and direct buffer checks stay separate.

## 2026-10-07 — independent eager snapshot for T4 graph evidence

The graph helper copies the eager result before warmup, capture and replay.
Persistent output buffers cannot change this reference snapshot.
The CUDA regression uses a stateful buffer to expose the old false equality.
The 192 prior mode-0 graph claims remain withdrawn.
New graph receipts compare independent eager snapshots.
GPU-exclusive timing rows are diagnostics, not quiet-host or measurement-class qualification.


## 2026-10-07 — real vLLM hash test setup

The hash tests use a fresh interpreter, so another test's fake package cannot supply the runtime.
The real interpreter imports the platform IR kernels before it computes a config hash.
The test keeps all configured provider priorities and checks both mode and dispatch identities.

## 2026-10-07 — native fused T-4 served lane

Contract v63 records the native T4 serving cutover.
Contract v64 withdraws the current FP4 activation attestations.
Versions v61 and v62 stay reserved for PR1046 and PR1028.
Source approval does not qualify native arithmetic, D41 or serving cells.

Dense and routed E2M1x2 exports use the named WINDOW L14 recipe over LUT16.
The pure widths 1..8 cover q128..q1024, including the full-width cap.
Serving exports, cached-unit records and control prices use the same recipe.
Research defaults and explicit TCQ encodes stay unchanged.
CPU tests replay the actual serialized bytes and check plane and full-file charges.
The byte audit includes all pure classes and both structures.
The shared rate guard checks expert strides and gate/up strides before encode.
The down projection can use a separate rate when all experts agree.
Dense and routed owners share the native fused E2M1 library.
The dense route quantizes activations once and writes each role into the final output.
Resident tensor owners expose the actual prepared tables, scale planes and descriptors.
The current activation container stays empty until a new serving cell has its own evidence.
The qualification harness checks actual serving paths, numeric references and CUDA graph equality.
It retains physical byte counts and raw timing samples for later D41 review.
The real-unit screen uses bounded GLM tensor ranges and explicit research TCQ.
The native receipt collector uses the route's resident tensor interface.
The input manifest declares one bounded phase for the selected GLM units.
The step-4 preflight builds the required native FP4 extension and records its path and digest.
Its dispatch qualifier uses the current WINDOW entry points.
Historical TCQ fixtures remain outside the published contract.
A feasible common-budget screen completes its comparison without an exact byte match.
The receipt keeps the exact-match flag and the actual byte slack separate.
The measurement encoder drains retained Viterbi plans after each unit.
This bounds residency across byte-plan probes without a new encoder default.
The T8 planner uses a high-rate reference only to estimate fixed serialized overhead.
Actual encoded bytes still decide the selected rate and every exact-match claim.
The D41 measurements and the real-unit screen remain pending.
This change does not assert measured route eligibility or quality.

## 2026-10-07 — projection audit geometry import

Read TILE_ROWS from tessera.window_geometry in the projection byte audit.
Keep the byte matrix, pricing rules, and contract v60 unchanged.

## 2026-10-07 — isolated source-boundary test paths

Keep outside-link fixtures inside each test's private scratch directory.
The scanned repository remains a separate subdirectory and the filesystem boundary guard stays unchanged.

## 2026-10-07 — Torch-free construction preflight

Read configuration shapes and producer declarations without a Torch import.
Use metadata-only fixtures for input width and reachability checks.
The real CPU Linear construction path still requires Torch and vLLM.

## 2026-10-07 — explicit GLM projection exports

Use one dense owner rule for KDA, MLA inputs, and DSA key and head weights.
Read KDA layer identity from the source construction config.
Keep standalone MLA queries outside the KDA group.
Read output partitions from the construction receipt.
Apply the stock NoPE row padding only to the declared key input member.
Keep original BF16 tensors, vision biases, and the qkv source prefix.
Select router and vision projections only through explicit plan entries.
Refuse stock twins that the stock constructor cannot load.
Price direct consumer buffers through the runtime adapter's shared byte rule.
Extend the existing byte audit with NoPE padding and direct buffer cases for all three commissioned weight families.
The audit records the wire price, native resident price, and direct buffer price separately.
It checks each direct price against the decoded buffer and records both exact sums.

Install selective routes for explicitly declared projection units.
Keep every unselected BF16 module on its original stock method.
Read the KDA replication indices from the real layer.
Retain the DSA FP32 head cache and the MLA BF16 split matrix.
Share their activation and decoded weight contracts with PrismaQuant.
Do not change a serving pin, kernel, or default.
Remove obsolete empty Tessera configuration assertions from stock passthrough tests.
Keep the geometry-demotion copy check and the exact BF16 control.

Record input width variants without a false reachability refusal.
Keep true quant-config and output-partition disagreements as refusals.

Use a Torch rendezvous URL for the real two-host TP2 artifact proof.
Record source and image identities without an identity refusal.

Publish the selected current-image construction receipt as the active GLM entry.
Retain the previous GLM receipt as unchanged history.
Declare the actual node count and node rank for native TP2 members.

## 2026-10-07 — projection construction and load checks

Add a small-artifact projection smoke tool with a portable CPU dry run.
The device path loads real wires through stock vLLM projection classes.
It compares eager and CUDA graph outputs with decoded FP64 references.
The dtype and operation-count bounds include the dense cast, bias, and rank reduction.
Byte views implement the stock and graph checks. No fixed numeric tolerance applies.
The checks include stock BF16 controls, FP32 indexer gates, absorbed MLA BMM,
router output dtype, vision biases, and replicated KDA roles.
Device results remain separate from CPU input proof.
The construction census now retains every instance, input width, parameter shape,
and replicated shard identifier. It disables unused prefix caching explicitly.
The same census entry point provides a portable preflight that reads real config
shapes and producer imports without model construction. Its output is not a census.
The constructor check uses real vLLM Linear classes on the supported CPU platform.
Stock, T-8, and T-16 construction views use fresh processes in one admitted action.
The selected views retain actual quantization method calls and explicit output partitions.
The existing action guard accepts an explicit job policy without a new dispatcher.
It records the measured CPU peak and the unmeasured GPU overhead separately.
The new policy adds a three-GiB margin and aborts below two GiB.
It sends SIGTERM, then SIGKILL after ten seconds. PrismaBuild owns scope cleanup.
The old measurement policy and its callers remain unchanged.

## 2026-10-07 — T-16 routed cells: served-census pin restored

Contract v59 restores the four TESSERA_BF16_K1 routed cells to R1024 with run table [4].
Independent review refused widening on sweep rows, prototype speed, and quality screens without served receipts.
It refused bit 8 on the rate-8 down anomaly.
No rung joins these cells without a served receipt.
The sweep results stay as research evidence.
The served re-census stays open.
This source change claims no graphics processor qualification.

## 2026-10-07 — issue 1018: producer policy correction

Both table producer paths preserve applicable reader correctness findings as holds.
Partial timing cells wait before class reconstruction and retain explicit failure states.
Class inheritance filters held and refused donors before it selects timing anchors.
The producer restores measured R896 scope under the D41 half-bit decision.
R896 uses its recorded per-cell cost and receives no pure-rung speed credit.
The correction leaves the active index, consumer policy, and campaign blocker clock unchanged.
Issue 1018 stays open. This source change claims no graphics processor or serving qualification.

## 2026-10-06 — issue 688: #685 baseline-band comparison consumer

Add the #688 acceptance consumer for the third acceptance line: the four
routed rows must reproduce the #685 after-run medians within their IQR.
`tessera.serving.panel_baseline` (schema
`tessera.shape_time_baseline_comparison.v1`, reviewer CLI
`tools/tessera_panel_baseline.py`) reconstructs the preserved bench's
nearest-sample quartile rule verbatim and proves it against every recorded
cell of a preserved table before comparing anything: the recorded band is
the band the historical bench wrote, never a rebuilt or interpolated one.
Rows compare only on a matched (structure, module, family, grid, rate) key
with agreeing rank-local geometry, and agreement is never presumed: a row
with no geometry of its own is a nonpassing `geometry_missing` verdict in
the JSON receipt (the CLI still exits 0), never a borrow of the reference
shape. A median outside the recorded band is likewise a `gap` verdict
carrying both numbers (CLI 0), not a reproduction and not a refusal.
Malformed geometry, duplicate comparison rows, a timing cell
that is not a JSON object, and a cell without a numeric `samples_ms`
array are named `ValueError` refusals (CLI exit 2), not silent passes or
tracebacks; the reviewer CLI binds, pins and proves one owned read per
file. Dense panel rows project from a validated
`tessera.shape_time_panel.v1` receipt;
routed rows refuse projection by name until the panel schema grows one.
Historical runtime-identity drift stamps and continues under the landed
development-mode split; byte integrity, execution scope, grammar and
comparison-key checks stay refusals. The verdict is about the comparison
only: it is not a measurement, an admission, a price or a pin, and the
full #688 native acceptance stays open.

## 2026-10-06 — issue 1005: actual T4 geometry reader coverage

Add a default-off measurement reader for actual mixed-rate span-two TCQ and
dense twelve-bit WINDOW E2M1 pairs. Preparation reuses existing packed BODY,
forest and WINDOW plane owners. Native FP4 compute decodes inside the mainloop;
the stock-byte renderer is diagnostic only. CPU quality explicitly selects
the served structure and records each recipe and exact encoder bytes.
The wrapper uses the existing development-mode stamp for image identity drift,
while image absence and representation/byte bounds still refuse. Wider research
WINDOW widths are not this reader's scope. Original forwarding and wording-only
adapter assertions are replaced with actual code/scale-byte controls, including
mixed boundaries, zero-width points, out-of-scope windows and canonical row-cut
history. Incoming TCQ register indexing advances with the local pair and stays
bounded before shifting. Grouped preparation shares a TCQ label lookup only
when the actual current table shape, dtype and bytes agree across experts,
independently of generator names or profile identities. No stored bytes,
serving defaults, recipes, pins, canonical table or admission gate change.
Remove the obsolete wrapper-text identity assertion and refresh offline issue
references; actual image-identity mode and byte-boundary controls remain.

## 2026-10-06 — issue 1002: bounded seeded control and piece-major phases

Replace the five-arm 90-minute diagnostic with two explicit native gang
phases: two fresh OFF blocks first, then matched OFF and piece-major alone.
Each phase is priority -10, at most 1800 seconds including cleanup, with no
campaign release wait. All eleven seeded L2048 outputs remain mandatory.
Serving nondeterminism is recorded without suppressing later measurement;
actual OFF timing selfvariation is preregistered, not an invented quality
tolerance. Piece-major collects the existing profile and power evidence.
Original memory/isolation guards and the strict 33-output ship gate remain.
Both diagnostic blocks use the existing eager route counters. Native members
declare complete staged inputs; the existing reader lease holds full-file
descriptors through model lifetime with no bulk origin fallback, then releases
only after exact owned physical cleanup.
Default-off, quality, adoption, ship and pin decisions are unchanged.
Fresh same-entry CPU preflight and exact-head review precede changed GPU work.
Split staged-digest, filename and preregistration provenance into existing D32
stamps; keep actual ranges, byte integrity, timing/protocol, caps and cleanup
as refusals. Add real-file development/certified boundary regressions and
remove permanent observer-AST and command-echo-only tests.

## 2026-10-06 — issue 995: exact-field native span-two reads

Native A4 SELECT and POINT reads load a second byte only when the meaningful
field crosses the first byte. Zero-width POINT fields do not load. Big-endian
extraction, code lookups, GEMM and scale/epilogue algebra are unchanged, as are
wire bytes, prepared allocations, shape admission and serving gates. The
boundary regression compares decoded states, packed codes and scale bytes with
exact field and stock wire decoding; it compares no GEMM output, and its native
part needs CUDA and skips without it. Detecting an unused out-of-bounds read
requires compute-sanitizer with the caching allocator disabled, not numerical
parity alone. D41 native
rows require fixed-build remeasurement before allocation; old timings remain
historical. No serving default or runtime pin moves.

## 2026-10-05 — issue968: explicit MNBT8192 ship-window selector

Add `WINDOW_MODE=ship-eager-4096-8192` and the exact eager4096/eager8192
plan to the existing D30 managed-rank harness. Both arms retain the same
A8S/socket/TP2/resident/c1 bindings, rank1 client, profilers, ownership,
review gates and D30 guards/caps. Neither the graph-control default nor the
Window4 eager2048/eager4096 pair is extended. CPU selection, refusal and
subprocess-lifecycle tests qualify source behavior only, not 8192 memory fit,
quality, performance, shipping or pin admission. Parent and D5 exact-head
review precede any admitted 8192 model leg.

## 2026-10-05 — D32 (PR961): managed-window run-identity seals stamp and continue

Rob's standing D32 direction (sealing off until further notice) reaches the
graph-attest Window4 producers. Dev mode is ON unless `PRISMAQUANT_DEV_MODE`
is exactly `0`; the run-identity comparisons in
`experiments/graph_attest_702` go through the existing
`tessera.dev_mode.seal_check` and, on a mismatch, print one `[DEV-MODE]` line
and continue with the stored data instead of refusing: `require_producer`
keeps its signature and returns the stored expected producer digest without
Git or digest computes in dev, `producer_sha` reports the stored
`PRODUCER_SHA256` (or `NOT_COMPUTED`) instead of hashing, and the
`inputs`/`prepare`/`submit`/`preflight`/`run_rank` provenance, source-hash
and frozen-control drift comparisons stamp rather than refuse. The
restamped memory-policy value comparison against the running policy is a
seal: it stamps in dev and refuses certified. The exact-HEAD checkout
comparison is now a seal too (certified keeps the refusal); still refusing in
both modes: the exact executing-code parent/D5 review, the `inputs.json`
`GRAPH_WINDOW_INPUT_SHA256` and memory-policy SHA own-byte integrity, PB
claims/nonces/action keys/owned
scope, the OOM floor and disk admission, the manifest resource/safety fields
and every format gate. Dev mode adds no release gate: no recompute, archive,
re-seal, re-pin or identity-proof packet is required or performed for
identity drift, and no format pin or serving default moves. Regressions in
`tests/test_graph_attest_producer_identity.py` and
`tests/test_graph_attest_headroom_preflight.py` demand the dev
stamp-and-continue behavior (failing on the pre-change refusals) and the
verbatim certified refusals; every pre-existing OOM/headroom/lifecycle/
own-byte/foreign-safety assertion is retained. `docs/ARCHITECTURE.md` and
`experiments/graph_attest_702/RUNPLAN-artifact.md` are re-stamped in the
same commit.
Corrective review restores the live TP2 rank-agreement check (image, source
digest and config digest) as a plain refusal in both modes. These compare the
two executing gang halves, not a run against recorded identity. Real-protocol
CPU regressions cover each mismatch in default dev, explicit dev and certified
modes for graph and eager windows, requiring neither server to launch.

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
## 2026-10-05 — issue964: the serving source pin stamps instead of refusing (D32)

The native shape-time pipeline's run-identity gates no longer refuse a serve
whose running Tessera tree differs from the frozen expected context. The
worker's before-device code identity and frozen-software comparison, the
installed CPU preflight's recorded commit and observed software context, and
the panel/observation validators' observed-vs-frozen runtime now go through
the new stdlib `tessera.dev_mode.seal_check` (the twin of
`prismaquant.dev_mode`, prismaquant#1147): dev mode is ON unless
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
re-imposing the stamped pin. Still refusing in both modes: which CASE
executed (execution mode, residency, TP geometry, requested serve flags --
comparability, not run identity), the installed `runtime_contract.json`
bytes against their own pinned digest, preflight module byte bindings,
every evidence/bound-bytes check, foreign runtime origins and
import-environment isolation, mid-run stability re-observations, and the
fail-closed VCS commit resolution that names what ran. Regressions in
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
