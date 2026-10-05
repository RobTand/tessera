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
a supported `pbcampaign.py` manifest. `--submit` checks exact producer parent/D5
approval, unchanged inputs/manifest, the current census queue condition and fresh
D1, then keeps the published campaign completion client attached.

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
ceilings, not measured concurrent-use claims. The 104 GiB cap budgets the derived
98 GiB model/KV peak plus 6 GiB unmeasured graph/host allowance; the GPU subset
leaves 2 GiB inside the host cap for host-only work. This does **not** establish
that graph pools or cold JIT fit; refusal/OOM must retain its exact attempt and
memory evidence. Do not copy EXL3's 100 GiB budget or reduce demand to force admission.

Both hosts still require **114 GiB MemAvailable** before each arm. The derivation
is the historical 94.4/96.1 GiB release peaks at 384 MiB KV, plus 1.625 GiB for
2 GiB KV, rounded conservatively to 98 GiB plus the unchanged **16 GiB physical
floor**. The floor is sampled every five seconds, not a proof against transient
undershoot. No cache reduction is approved; four resident requests and all
capture sizes/cases remain required.

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
separately reviewed PR: `PRODUCER_COMMIT` and `PRODUCER_SHA256` seal the rank,
recipe, driver, timer, wrappers and plan bytes separately. PB snapshots the
producer checkout, not the runtime checkout. Neither identity is restamped at
receipt time. Parent and D5 review JSON binds the exact producer head; runtime
choice and fabric remain explicit. The old dual-launch #942 wrapper is never
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
waive the later 114 GiB preflight or 16 GiB floor. `--prepare-role-preflight`
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
manifest and keeps its published completion client alive. The review JSON's
`parent` and `D5` objects each carry `verdict: APPROVE` and the producer
`head_sha`, with real review provenance. Never resume/rewrite a submitted root.

After terminal physical handoff, run the **frozen runtime's** existing v2
`receipt.py` through CPU PB, using `ROOT/arms`, `ROOT/receipt-manifest.json`,
`--eager aE1,aE2 --graph aGR --commit SOURCE_COMMIT`. Keep the not-measured
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
