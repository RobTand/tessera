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

## Receipt-source dependency and shared-file integration

The CEO engineer owns PR942, `opus/702-receipt-fabric-v2`, pushed head
`f7403d1dc49dd87c81d021287afa884e81cc0120` when inspected. It adds
`graph_equals_eager.v2`, with fabric as a ninth scope field. At TP2 the builder
reads **both ranks' actual NCCL banners**, refuses requested/observed mismatch,
and v1 receipts remain inspectable but cannot verify cards. This repair does not
copy or edit that schema/parser implementation. Its measured `fabric_observed`
record retains the builder's `rank0:Using network Socket;rank1:Using network Socket`
grammar, derived from each local log, not from socket defaults.

PR942 and this repair both change `arm_tp2.sh`, `RUNPLAN-artifact.md` and portions
of `test_graph_attest_tp2.py`. Integrate only reviewed/landed source: retain the
managed local-rank entrypoint, explicit SOCKET nomination, v2 parser/tests and the
observed-banner record. The plan reader also accepts PR942's explicit per-arm
`FABRIC=socket`, but refuses an arm that differs from the frozen common tuple.
Do not freeze a live writer's worktree or silently use the pre-v2 source for a
card. Parent and D5 review of the final frozen source precede any real model start.

## Published actions and finite ownership

`drive_tp2.sh` now delegates to `window_driver.py`. `--dry-run` only inspects
inputs and renders the two local serve commands. It launches nothing. A direct
`arm_tp2.sh ARM` no longer starts containers; the one-arm wrapper is command
inspection only. `--prepare` creates one fresh invocation root, input digest and
a supported `pbcampaign.py` manifest. `--submit` checks exact-source parent/D5
approval, unchanged manifest/inputs, both preceding handoffs and fresh D1, then
runs the published campaign **with its completion client**, not detached.

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

## Order and commands (not executed as a model window)

The newer CEO order is ready-first: whichever corrected/reviewed census769 or
EXL3 baseline is ready goes first, then the other, **then 702**. Campaign must
supply both hosts' terminal physical handoffs for **both** preceding windows.
`--handoffs` reads a JSON object with exactly `census769` and `EXL3`; each value is
the two owner-supplied identities (`action_key`, `nonce`, `scope_id`, `host`,
`claimed_unix`, `window_end_unix`). The recipe looks up their real PB terminal
records and broker cleanup, not caller-written “empty” assertions. It never
edits another owner's record or drains a foreign scope.

Run preparation from celestia against a clean shared checkout containing the
reviewed/landed managed repair and v2 dependency. Issues supplies the source
commit/digest; do not infer a tuple from the socket setting or choose a future
headline artifact. Supply exact-source review records before submission:
`parent` and `D5` objects each with `verdict: APPROVE`, `head_sha` equal to the
frozen source and the retained review URL/evidence. Use a fresh root every attempt.

```bash
PB=/mnt/shared/prismabuild-fleet/repo/tools
CLIENT=/home/rob/tmp/pb-submit-celestia-20261003/bin/python
: "${TS:?clean frozen shared source}" "${SOURCE_COMMIT:?issues-owned commit}"
: "${SOURCE_SHA256:?issues-owned src digest}" "${RUN:?fresh shared invocation root}"
: "${HANDOFFS:?both preceding terminal physical handoffs}" "${REVIEWS:?parent and D5 reviews}"
export TS SOURCE_COMMIT SOURCE_SHA256
export ARTIFACT=/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported
export FABRIC=socket RECEIPTS="$RUN/arms"

# Command inspection is also a check: submit it through PB, CPU-only.
"$CLIENT" "$PB/pbrun.py" --cwd "$TS" --tag x86 --cpus 1 --demand mem_gb=2 \
  --env TS="$TS" --env ARTIFACT="$ARTIFACT" --env RECEIPTS="$RECEIPTS" --env FABRIC=socket \
  --timeout-s 120 -- /bin/bash experiments/graph_attest_702/drive_tp2.sh \
  experiments/graph_attest_702/plan-artifact.txt --dry-run

# Preparation is read-only inspection/submission bookkeeping; no container starts.
bash "$TS/experiments/graph_attest_702/drive_tp2.sh" \
  "$TS/experiments/graph_attest_702/plan-artifact.txt" --prepare "$RUN" --handoffs "$HANDOFFS"

# Only after parent/D5 review and the recorded window handoffs. Never run this during repair.
bash "$TS/experiments/graph_attest_702/drive_tp2.sh" \
  "$TS/experiments/graph_attest_702/plan-artifact.txt" --submit "$RUN" --reviews "$REVIEWS"

# After complete terminal physical handoff: existing receipt builder, CPU-only PB.
"$CLIENT" "$PB/pbrun.py" --cwd "$TS" --tag x86 --cpus 2 --demand mem_gb=2 --timeout-s 120 \
  -- /home/rob/venvs/tessera-train-8bff20d0/bin/python \
  experiments/graph_attest_702/receipt.py "$RECEIPTS" "$RUN/receipt-manifest.json" "$RUN/receipt.json" \
  --eager aE1,aE2 --graph aGR --commit "$SOURCE_COMMIT" \
  --not-measured "served KL against BF16 under graphs" --not-measured "graph-vs-eager speed"
```

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
