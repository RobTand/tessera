# Run plan: the managed artifact graph control (tessera#702, prismaquant#1586)

**Not a model-run result or a launch authorization.** The repair is based on merged
PR930 (`f1c07473a74bb5999cce51134c769ed7efb45d6c`). It first extracts ownership and
deadlines into `managed_window.py`, then cuts the existing TP2 recipe over to two
published PrismaBuild actions. The serving exemption does not exempt the batch
48-choice equality validation. Neither rank starts the other host, by SSH or any
other unadmitted launcher. PrismaBuild alone owns admission, placement and scope
release; this recipe does not provide atomic two-host reservation.

## Frozen control, not the future headline artifact

| Input | Required value |
|---|---|
| Artifact | `/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported` (A8SE752MN control) |
| `config.json` SHA-256 | `3f5c2c7381aae1c02d486c645ec6015cd1a60eb41faa5686541a15f523d79898` |
| Image | `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a` |
| Source | One clean, frozen, reviewed/landed shared checkout; explicit issues-owned `SOURCE_COMMIT` and `SOURCE_SHA256`, never a receipt-time restamp |
| Fabric | **SOCKET**, explicitly supplied as `FABRIC=socket`, `NCCL_IB_DISABLE=1`; CEO clarification `dec-1005-005132-3a81` supersedes the earlier RoCE relay |
| Topology | TP2, mp executor, nnodes 2; rank0/API on sparklina `10.100.96.2`, local rank1/headless on sparky `10.100.96.1` |
| Concurrency / length | c4 / 8,448 |
| MTP | One speculative token, draft TP2, triton MoE |
| Graph flags | `{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}`; existing plugin captures sizes 2/4/6/8 |
| KV | 2,147,483,648 bytes **per rank**, `fp8_ds_mla` |
| Other flags | resident, triton MoE, `TESSERA_FUSED_E4M3_MMA=e4m3`, autotune off, chunked prefill 2048, prefix caching off, breakable graphs off, NoPE off |

Order is exactly `aE1 eager / aGR graph / aE2 eager`. Each arm runs two passes of
the entire unchanged 48-choice equality set, then all four existing single-request
long screens. No fourth arm, reduced case set, smaller cache or automatic retry is
approved. Long traffic above `index_topk` remains a screen, not long-context
equality, served BF16 KL or a speed result. Inconsistent eager passes are
non-qualifying; preserve them without guessing a fabric/NCCL cause.

The A8 control receipt is not transferable to the new all-allowable T8 headline
artifact, whose true bytes and config digest must be supplied separately. Full
artifact equality, compiled runtime cells, D13 serving pin, scientific quality,
served profiling/power and ship-card qualification remain separate gates.

## Reviewed v2 integration

PR942 is merged as `39e3d950226f1c9fec38d4baaaccb59fc267e12b`; the reviewed eight
files are retained, including v2 schema/parser/tests and explicit socket plan.
This branch was rebased only onto that landed source. Shared wrapper/doc/test
conflicts retain local-rank containment plus the producer's v2 fabric coverage.
The v2 builder consumes both measured NCCL banners; v1 is inspectable, not a card
qualification. Runtime and producer identities are separately bound below.

## Published actions and finite ownership

`drive_tp2.sh` now delegates to `window_driver.py`. `--dry-run` only inspects
inputs and renders the two local serve commands. It launches nothing. A direct
`arm_tp2.sh ARM` no longer starts containers; the one-arm wrapper is command
inspection only. `--prepare` creates one fresh invocation root, input digest and
the supported native-gang manifest of full member contracts. `--submit` checks
exact producer parent/D5 approval, unchanged inputs/manifest, the current census
queue condition and fresh D1, then submits through published
`pbgang.py --manifest` and waits through published `pbwait.py --json
--wait-s6000` on both exact member keys, writing `native-gang.json` and
`member-completion.json`; there is no private group writer or second dispatcher.

Each action executes `rank_window.py --rank 0|1` on its nominated host. Local
Docker uses PB's ordinary shim, inherited CPU affinity and memory parent. The
local rank/controller and rank0's equality subprocesses are admitted children,
not recursive submitters. A fresh invocation UUID, full action key, broker nonce,
scope, container owner, claimed host/start and input SHA bind every rendezvous
and stage marker. Only a matching **live CLAIMED** peer can authorize a launch;
DONE, FAILED, touched stale files and a new attempt of the same key cannot.

The deadline is the **earliest rank claim time + 5,400 seconds**. Rendezvous,
image/source/memory preflight, all three arms, probes and owned cleanup consume
that single envelope. It never restarts for a new arm or the later-admitted rank.
Peer admission has its own 3600-second cap, claim-relative, inside this same
envelope. It is not a separate 3600+5400 allocation or a per-arm budget reset.
Work stops before the final 180-second cleanup reserve. Subprocess timeouts kill
the exact owned process group, including its CPU descendants. Startup also has
an 1,800-second bound, clipped to what remains. A failure or timeout publishes
partial evidence and stops all later arms; there is no automatic fourth arm.

Each launch records its full CID/cidfile and launch intent before proceeding.
Cleanup checks CID, scope/owner labels, attempt/run labels and Docker cgroup
parent before removing **that exact local container**. Ambiguous launch failures
use the unique name only for discovery, then require the same ownership checks.
Log, removal and bounded-copy failures remain failures, not discarded stderr.
Local output and cache directories survive failure; no model, unrelated cache,
foreign container or tagged image is deleted. The install copy replaces only
`/ext/tessera` inside this invocation's unique owned cache directory; compiled
extension/Triton caches are reused across the three arms.

Before the next arm, each rank proves its owned containers and GPU descendants
empty. After both controllers exit, the worker's exact-attempt broker export must
prove stopped/empty/tickets-settled/released inside the common deadline. The
collector reuses PB's `export_verdict_proves_empty`; payload exit zero or a missing
old CID is not enough. `handoff.json` releases logical ownership only after both
terminal records, all local cleanup records and both broker proofs agree. A
failed/copy-incomplete handoff retains ownership/evidence, even if ordinary PB
reservations have already been safely released by the worker. A waiter expiry is
not terminal or cleanup proof: retain exact keys and renew the published
completion client, never drain the other owner or silently start a new window.

## Resource contract (caps, not measured fit)

| Per action | rank0 / sparklina | rank1 / sparky |
|---|---:|---:|
| Aggregate CPU ceiling | 8 | 6 |
| Shared host DRAM cap, GiB | 104 | 104 |
| GPU subset of that shared DRAM, GiB | 102 | 102 |
| Physical GPU | 1 exclusive | 1 exclusive |
| Native OMP/MKL/OpenBLAS/NumExpr/MAX_JOBS threads | 1 | 1 |
| Declared local scratch/output allowance, GiB | 8 | 8 |

Combined peaks are 14 CPUs, 208 GiB host DRAM, a 204 GiB GPU **subset** (not extra
DRAM), and two exclusive physical devices. CPU demand covers target/draft engine
processes, NCCL progress/helpers and the local controller; rank0 additionally
has the API and one-at-a-time equality client. These are conservative declared
ceilings, not measured concurrent-use claims. The 104 GiB cap stays an admission
ceiling over the derived 98 GiB model/KV estimate plus the 6 GiB explicitly
unmeasured graph/host allowance; the GPU subset
leaves 2 GiB inside the host cap for host-only work. This does **not** establish
that graph pools or cold JIT fit; refusal/OOM must retain its exact attempt and
memory evidence. Do not copy EXL3's 100 GiB budget or reduce demand to force admission.

**Memory admission (issue959 D30 amendment, 2026-10-05).** Both hosts require
**107 GiB MemAvailable** before each arm: the retained conservative hybrid
model/KV estimate **98 GiB**, plus the explicitly unmeasured graph/host
allowance **6 GiB**, plus reserve **3 GiB**. The historical **114 GiB**
predicate and its sampled **16 GiB floor** are no longer live; they belong only
to the immutable b5 run recorded under them. Historical derivation is documented
in `/home/rob/fleet/inventory/kernels-window4-headroom-equation-packet-20261005.json`
(SHA-256 `1a28bff770790a09b0df4573d46acf31e44aa12c6118cf31b9a32a279ffdb4f8`).
The 98 is derived, not measured: the larger of the two hybrid
estimates 94.4/96.1 GiB plus the 1.625 GiB KV uplift from 384 MiB to 2 GiB per
rank, rounded conservatively up to 98. Those hybrids come from the graph arm's
MemAvailable ready drop and the eager L512/c4 transient-KV assumption — they
are **not** measured 384 MiB peaks. Engine RSS and context sizes were never
recorded and stay unknown; cgroup charge, MemAvailable and GPU-used quantities
overlap, so no naive sum of them is a measured decomposition. The 104/102 GiB
caps above are unchanged. No cache reduction is approved; four resident
requests and all capture sizes/cases remain required.

**Bounded headroom wait (issue953, CEO-authorized 2026-10-05).** When the
synchronous sample sits below 107 GiB, `LocalArm.preflight` waits up to **900
seconds** for the same 107 GiB predicate to become genuinely true, in graph and
eager modes alike, before each arm. The wait polls on the existing `run_rank`
cadence, checks the live guard on every poll and immediately before declaring
ready, and is capped by the existing window envelope (cleanup reserve and any
tightened absolute window), so a shortened deadline still refuses. Expiry
raises the 107 GiB preflight refusal; guard cancellation refuses inside
the wait; the model never launches below 107 GiB.
One terminal wait report per call — threshold, 900-second bound, initial/last
GiB, wall/monotonic times, every exact sample, and reason
`ready`/`headroom_timeout`/`lifecycle_cancelled`/`deadline`/`error` — is appended
to `rdv/headroom-preflight-rank<rank>.jsonl` and persists through refusal,
cancellation, deadline or unavailable readings (unknown values stay unknown).
Before container work, the model-start boundary rechecks the same 107 GiB bar
after source checks and the peer barrier and records its per-arm synchronous
sample at `rdv/<arm>-launch-headroom-rank<rank>.json`. A prior ready wait never
authorizes a below-threshold launch. The unchanged 104/102 GiB
demand caps and every other admission predicate are untouched: this is a bounded
predicate-satisfaction wait, not a new admission path, and it produces no fit,
speed or pin evidence and no transfer of the historical frozen producer's
source approval. Deterministic regressions: `tests/test_graph_attest_headroom_preflight.py`.

**Dual-rank memory guard (issue959 D30 amendment).** While the window runs,
each rank samples whole-box `MemAvailable` at 1 Hz. A sample strictly below
**2 GiB** on either host fails **both** exact-owned ranks. The local comparison
precedes shared queue/journal work; exact-owned TERM starts locally before peer
publication. The first exact-attempt FAILED marker is immutable and precedes
any termination grace. Secondary signal/publication/journal failures are recorded
alongside, never substituted for the trigger. SIGKILL follows ten seconds if
needed; expiry can shorten the grace, but cleanup-only inspect/signal controls
still get a fixed five-second budget to dispatch KILL, not to renew model work.
Both live ranks acknowledge failed-cleaned before returning; a dead peer ends
that wait immediately, and the wait retains five seconds for broker stop.
Absent physical custody remains explicit. A sampled guard is not continuous:
an inter-sample drop is not intercepted and clean samples are not immunity.
`window_driver.py --prepare`
emits `rdv/memory-policy.json` once and binds its SHA into the submitted
inputs; each rank writes `rdv/memory-samples-rank<N>.jsonl` and
`rdv/memory-summary-rank<N>.json` (raw samples with baseline/minimum/counter
bookkeeping plus the summary). The Envelope outcome carries the termination
list, and container termination evidence is retained. The samples artifact
exists so a future window can measure the 6 GiB allowance against observed
data; it is not a measured RSS decomposition of the 98 GiB estimate. Protected
sysctl, ARC, cache and service settings are untouched by the guard.

Runtime disk caps cover the unique local cache (6 GiB), local work including
cache/logs (6.3 GiB) and shared arm records (0.2 GiB total). The declared 8 GiB
local allowance also covers checkout/PB-log overhead; shared output/CAS allowance
is 1 GiB. The artifact is read in place, not copied. Staging a new image/artifact
is not included: declare its actual output and take a separate fresh D1 if needed.
Preparation and submission each run D1 on both local scratch mounts (`--need-gb 8`)
and shared output (`--need-gb 1`). Historical free-space figures do not qualify a run.

## Superseding execution order and separate source identities

CEO decision `dec-1005-012825-0f3f` supersedes the old no-launch hold and the
census/EXL3 ready-first ordering. After exact-head parent/D5 approval, start this
control only when the current campaign769 halves are not READY/CLAIMED; if they
are, wait for their terminal physical owned cleanup. Never queue a second paired
window before PB1519 rollout. EXL3 is not an additional prerequisite. Optional
`--handoffs` now names only `census769`, binding actual owner identities and
terminal broker proofs. No foreign drain or implicit cleanup inference is allowed.

The read-only `watch_window_queue.py` timer reads the published `pbstatus --json`
and the two explicit full campaign keys, records complete/partial status and
actual read time, and never submits. The requested 01:30 UTC read was actually
entered at 01:31:37.306 UTC: its complete later view had no named census half
READY/CLAIMED and only two CPU jobs. It is **not** evidence of the queue at 01:30.
That miss is preserved in the record, not restamped. Submission checks the queue
again in code and retains its view; a partial view refuses a new pair.

The frozen **runtime** may be merged PR942
`39e3d950226f1c9fec38d4baaaccb59fc267e12b`. Its source/config/hooks/equality-script
digests bind the measured Tessera and eventual receipt. The **producer** is this
separately reviewed PR. In both prepare and submit the checkout comparison
against the full `PRODUCER_COMMIT` is itself a D32 seal: default dev mode
stamps one `[DEV-MODE]` line and continues, certified keeps the verbatim
refusal, and the exact executing-code parent/D5 review — submit compares
every executing-code root's HEAD with the reviewed head — refuses in dev
and certified modes alike. The remaining producer identity comparisons
(`experiments/graph_attest_702` clean, every producer file equal to its
`git show PRODUCER_COMMIT:path` object, and the combined digest equal to
`PRODUCER_SHA256`) are D32 run-identity seals: under default dev mode
(`PRISMAQUANT_DEV_MODE` not exactly `0`) each mismatch prints one
`[DEV-MODE]` line and the run continues with the stored data —
`require_producer` returns the stored expected digest without Git or digest
computes, and `producer_sha` reports the stored `PRODUCER_SHA256` or
`NOT_COMPUTED` — while certified `PRISMAQUANT_DEV_MODE=0` keeps the legacy
refusals verbatim, including the exact missing-object boundary for a
parentless snapshot. A typed label plus a self-supplied disk digest is still
not authority, and dev mode adds no release gate: no recompute, archive,
re-seal, re-pin or identity-proof packet is performed or required for
identity drift. Parent and D5 review JSON must still name the producer
commit in both modes. The rank's own-byte seals never stamp: the submitted
`inputs.json` against its action-environment `GRAPH_WINDOW_INPUT_SHA256`,
`memory-policy.json` against its recorded SHA, and the policy content
itself refuse in both modes; only the restamped policy-value comparison
against the running policy stamps in dev and refuses certified, as do the
prepared source/control drift comparisons in `prepare`/`submit`/`preflight`/
`run_rank`. PB claims, nonces, action keys, owned scope, the OOM floor, disk
admission, the manifest resource/safety fields and every format gate are
unchanged. Runtime choice and fabric remain explicit. The old dual-launch
#942 wrapper is never
run twice or used to launch an unadmitted remote rank.

Every model row is an exclusive **measurement**, host class gb10, priority **10**
with a Goal reason, fixed legitimate rank-host tags, and 3600-second peer admission
cap inside the unchanged 5400-second common window. Requested PB timeout is 5400;
effective timeout/worker ceiling must be taken separately from actual terminal
`detail.execution_timeout_s`, `execution_timeout_requested_s`,
`execution_timeout_ceiling_s` and `execution_timeout_clamped`. Unknown before
execution is recorded as null, never asserted equal to the request. Once both
live attempts meet, each rank publishes an exact `both_halves_claimed` event;
this is admission evidence, not model readiness or qualification.

`rank_window.py --role-preflight` is an admitted CPU-only check on each actual
rank host: frozen runtime/producer/artifact identity, real local image resolution
and generated shell syntax, with zero containers/model/CUDA work. It does not
waive the later 107 GiB preflight. `--prepare-role-preflight`
prepares these checks with 1 GiB output admission; model submission still repeats
fresh D1 for its full 8 GiB local/1 GiB shared allowance.

Preparation/CPU checks use the published client from celestia. GPU measurement
**submission** uses the published PB client on a GB10 origin per D26, because
celestia cannot seal live accelerator evidence. SSH may transport that PB client
command only; all containers and equality clients remain admitted local children.
The producer checkout must be staged read-only on shared storage for that origin.

`drive_tp2.sh PLAN --prepare ROOT --census-key KEY0 --census-key KEY1` requires
explicit `TS, SOURCE_COMMIT, SOURCE_SHA256, PRODUCER_COMMIT, PRODUCER_SHA256,
ARTIFACT, FABRIC=socket, RECEIPTS=ROOT/arms`. Add `--prepare-role-preflight` for
CPU role checks. Submit each CPU row through published PB using `--rank 0|1
--run ROOT/inputs.json --role-preflight`, demand cpu=1/mem_gb=2, no GPU, timeout
120 and one native thread. After actual role receipts and exact-head reviews,
`drive_tp2.sh PLAN --submit ROOT --reviews JSON` submits the supported measurement
manifest through published `pbgang.py --manifest` and keeps the published
`pbwait.py --json --wait-s6000` client alive on both exact member keys. The review JSON's
`parent` and `D5` objects each carry `verdict: APPROVE` and the producer
`head_sha`, with real review provenance. Never resume/rewrite a submitted root.

After terminal physical handoff, run the **frozen runtime's** existing v2
`receipt.py` through CPU PB, using `ROOT/arms`, `ROOT/receipt-manifest.json`,
`--eager aE1,aE2 --graph aGR --commit SOURCE_COMMIT`; the entire invocation's
failed markers are checked before receipt-manifest emission. Keep the not-measured
served BF16 KL and graph-vs-eager speed exclusions. Inspect every full 48-choice
pass, eager repeatability, both ranks' replay/classes and exact v2 serve tuple;
receipt building alone qualifies no compiled cell, pin, scientific release or
future T8 artifact.

Inspect eager repeatability, require 48/48 for both complete passes in every arm,
both ranks' captured-size/class replay evidence and all source/image/fabric
fields, then call the unchanged v2 verifier against the exact serve tuple.
Retain hashes, both rank keys/nonces/CIDs, terminal records, logs and CAS receipts.
No receipt builder success alone establishes a scientific release or compiled cell.

## Verification scope

Selected independent CPU tests use published `pbtest.py` fanout, xdist worksteal
and durations, after fresh D1. The before controls test the real old dry-run:
remote rank launch and missing whole-window declaration fail. After controls
render the corrected real recipe. The real protocol is also exercised with two
CPU controller subprocesses and real child-process failures/timeouts/termination.
Only the local device adapter and private claim records are simulated; the
production local cleanup control flow additionally uses bounded CPU CLI exits to
exercise wrong-owner, log, removal and copy failures. These are not live Docker
shim/broker qualification, CUDA/NCCL, two-Spark rendezvous, GPU descendants,
physical memory-floor, full-artifact equality, residency, speed or power evidence.
No local suite, full suite, real model window or source/pin/quality promotion is
part of this repair.

The approximately 55-minute expected / 80-minute planning-worst / 90-minute hard
window remains **derived, not measured at TP2**: three 15–22 minute arms, first
cold JIT +5–10 minutes, graph capture +1–3 minutes, plus rendezvous/cleanup. A fit
claim requires the later real finite window, not these CPU controls.

## Latest ship-window mandate (pending exact artifact/client inputs)

CEO staged-candidate decision `dec-1005-023421-75cf` supersedes the diagnostic
90-minute reservation: after the real PR943 identity fix and exact-head review,
window3 is one exclusive SOCKET pair, approximately two hours. The public
runtime candidate is `2dbac1910c88254d9c6391f02a34c4b07e516803`, packaged contract
v56 raw SHA-256 `47f180efaf97faa5c411df5d48f9da7dff4b9c9fc0c3ddbf9f815bcd4d0aed78`,
from the qualified/reviewed draft PQ2262 candidate. That runtime/package identity
is separate from this containment producer. Main-pin landing/promotion waits for
the packet; private608bb/1770 evidence is not transferable.

Campaign must freeze the all-T8-rates artifact (A8S only as an explicit fallback)
and the actual October5 EXL3 client/protocol before serving. One source, image,
artifact and fabric are held throughout. The graph receipt preserves the full
48-choice two-pass population, capture replay and c4 scope. The speed/profile
matrix is L512/2048/8192, c1, TP2, MNBT2048, using exactly that EXL3 client. Its
different concurrency is recorded as a separate scope, never borrowed from c4.
Fresh in-process profiles and both-Spark power belong in the same packet.

Requested ship envelope is approximately7200 seconds; peer admission and live
measurement bounds must be named separately and effective PB ceiling read from
actual PB endings. The tested diagnostic source above still enforces5400 and
the A8 nomination: do not mislabel or launch it as the expanded ship window.
After its identity review closes, the exact supplied ship tuple requires its own
finite-plan cutover/review, preserving owned cleanup and one-pair-at-a-time.
No ship artifact/client identity, effective7200 ceiling, model result, pin
promotion or scientific qualification is fabricated by this plan. The latest CEO
order below supersedes the former ship-before-Window4 sequencing.

## Separate issue946 eager Window4 mode

PR943 is merged at `f1c457dc16f9909e00b5d582a5da14b3a19dff10`. Its local-rank
ownership, producer-object/disk proof, rendezvous, 5400-second common deadline,
180-second cleanup reserve, broker proof and failure retention are reused. The
named `WINDOW_MODE=window4-eager-2048-4096` selects only `plan-eager-window4.txt`: two
arms, `eager2048` then `eager4096`, each eager/socket/TP2/c1. The original default
`graph-control` plan remains c4/MNBT2048 and its two complete 48-choice passes plus
long screens are unchanged. Window4 emits no graph-equivalence or ship receipt.

The requested panel is L512/2048/8192 with output128, c1, ten timed trials plus
one excluded warmup per cell. Model length8448, MTP1/draft TP2, KV2GiB per rank,
image, resident/E4M3, NoPE/SP and release flags remain equal between these arms;
only the declared chunk setting changes. MNBT8192 is not a third arm.

Runtime is frozen independently at public `2dbac1910c88254d9c6391f02a34c4b07e516803`,
v56 raw contract `47f180efaf97faa5c411df5d48f9da7dff4b9c9fc0c3ddbf9f815bcd4d0aed78`.
`PQ_PIN_COMMIT` is exactly the CPU/install-qualified, parent/CEO/D5-approved
corrective PQ2264 head `e36e60b77b3d2ab0c5265272515958a0cb67d32b` for controlled use.
This separate producer head still requires actual parent/D5 approval before
submit. A runtime approval never approves new producer bytes.

`ARTIFACT_MANIFEST` is a shared copy of campaign's complete128-file A8S inventory,
canonical content digest `45407d43e09381b73197498d7f37c848c167c03a83d415c68e63c615d99eb840`.
Current file roster, byte lengths and all non-shard metadata hashes are checked;
shipping-body hashes reuse the supplied authenticated full-file audit, not a
new175GB body rehash. This does not attest the later all-T8-rate v1 artifact.

`eager_benchmark.py` binds the exact October5 `stock-client-source` identity,
timing/profile programs and existing prompt/profile manifest hashes supplied by
campaign. No instrument source, prompt, formula or request population changes.
Rank1/sparky runs the original host client, as EXL3 did. Endpoint8142, target
model ID, eager/socket/runtime labels and the fresh output/profile namespaces
are explicit differences. The original profile helper's SSH only inspects rank0
trace bytes; it never starts a rank. Both ranks' inactive torch profilers use
fresh arm namespaces; the same L512/max1, L8192/max1 and L512/max8 profile cells
run after timing. No profiled response is pooled into throughput.

The conservative104GiB host/102GiB GPU-subset caps per rank are admission
ceilings, not the old98GiB/c4 fit claim. CPU ceilings remain8/6 including the
single client on rank1; threads are bounded at1. Both hosts still need107GiB
MemAvailable at model preflight.
The real4096 model must fit this unchanged contract or retain its refusal.
Five-second per-rank MemAvailable and actual cgroup current/claim-lifetime peak
host charges accompany requests. CUDA coverage of cgroup charges is unproven;
these must never be relabelled as4096-only GPU allocation peaks.

Existing `box_power_window.py` collects both-Spark Netdata over the exact timing
episode, requesting one-second cadence and retaining actual returned grouping,
coverage, power, CPU, available memory and swap I/O. Energy scope includes three
excluded warmups and excludes profiles; any work/J calculation must use that
same population and observed coverage. Missing data is not a power/utilization
inference. Model fit or speed is never inferred from warmup or the190ms operator
screen.

The newest CEO order supersedes caller-manual drain: PACT priority0 quanta run
while recipe qualification/review proceeds. At real Window4 readiness, publish
the two priority10 exclusive measurement halves. PrismaBuild's host election
fences new lower-priority work and lets already-running quanta finish normally.
The temporary D28 supply cap was lifted at the witnessed f577 rollout; the
two rows here are the one pair, not an assertion that the old cap still applies.
No stopped-publisher permission, zero-live price marker, caller queue withdrawal,
payload kill or new scheduling layer is used. The complete already-taken queue
snapshot authenticates sealed managed-pair ownership: another paired window or
invalid ownership refuses, while legitimate lower-priority price rows do not.
One supported `pbgang.py --manifest` native gang owns the fixed-rank topology
pair, waited through `pbwait.py --json --wait-s6000` on both exact member keys.
Only actual both-claimed and terminal/owned cleanup observations are reported.

Use the existing driver with `WINDOW_MODE`, `PQ_PIN_COMMIT`, `ARTIFACT_MANIFEST`,
and the usual frozen source/producer/artifact/receipt environment.
`--prepare ROOT --prepare-role-preflight` prepares only CPU role inputs, then
published PB runs `rank_window.py --role-preflight` on each real rank host. It
reads that admitted role's real cgroup sampler and runs CPU-only temporary
profiler-config parsers in the exact pinned image, with GPU visibility disabled
and owned containers proven absent afterward. No model or CUDA workload starts.
A separate fresh model root reserves full D1
allowances; `--submit ROOT --reviews JSON` requires both producer approvals and
`runtime.parent`/`runtime.D5` approvals naming the exact pin head. All failed
attempts, logs, CAS receipts and requested/effective PB bounds are retained.
Both ranks execute the unchanged instrument's `verify-profile` within their own
admitted scopes, then rendezvous on the verified stage before cleanup or the
next arm. Profile order comes from frozen `declared_cells`, not a second roster.

CPU failing-before control: action `f585e74f487a11271d07a2ca7d88a2960b1b6cb37ad5142199874725a1cc927e`
ran5 tests:4 new named-mode controls failed,1 legacy refusal passed;0 skips,
torch2.11.0+cpu/noCUDA. Passing controls and actual model evidence are separate
records, not asserted by this source/documentation change.
