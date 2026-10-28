# G2 encoder decision handoff — BLOCKED

Decision package: [Tessera issue 1398](https://github.com/RobTand/tessera/issues/1398).
Recorded 2026-10-23 UTC by the implementing agent, **not** by an approving owner.
Scope: issue 652's G2 best-form Viterbi encoder, tile 64,4,2; retained generated-input
before-profile evidence. This record neither restarts measurement nor completes the
unrecovered original O1–O4. D1–D5 identify this handoff only.

## Authority and deliberate stop

[RobTand, 2026-10-04 02:36:39Z](https://github.com/RobTand/tessera/issues/652#issuecomment-5975838660)
assigns **v6 Kernels Lead** campaign custody (parent Main) and **v6 Review and
Integration Lead** sole repository integration. It explicitly says: “This is custody,
not closure or new GPU/adoption authorization.” Its broader served headline is not
an encoder-scope approval.

[RobTand, 2026-10-23 14:49:39Z](https://github.com/RobTand/tessera/issues/652#issuecomment-6367674007)
and [14:49:44Z](https://github.com/RobTand/tessera/issues/1387#issuecomment-6367675440)
record a deliberate stop: the last two reviews retain the same three failing
acceptance tests and all four unmet objectives, the completion packet is absent,
required DRAM/resident-bandwidth and selected-candidate evidence is missing, and
spend exceeds the soft budget. Preserve that stop. These are attributed reports
of review outcomes, not independently recovered review receipts.

## One focused retrieval/confirmation pass

Read the retained G2 note at
[tessera-window-best-form-2026-09-09.md, lines 334–440](tessera-window-best-form-2026-09-09.md)
and the whole packet below. On 2026-10-23 UTC, read-only GitHub API GETs retrieved
issue 652 and 1387 comments, issue 1398 comments, and bodies/comments for issues
1351, 1353, 1392, 1391, 1386 and 1390. The responses fit the requested 100-comment
page; no next-page response was needed. Issue 1398 returned `[]`. The G2-specific
[1387 reconciliation referral](https://github.com/RobTand/tessera/issues/1387#issuecomment-6368125800)
requires independent artifact/admission authentication and the owners' decision;
it does not supply either. Later comments on that aggregate issue concern other
work packages and are not this encoder's contract.

The available `gh` executable returned `Permission denied` (exit 126); read-only
HTTPS retrieval via Python succeeded instead. A bounded search of this checkout
and the served TrustLadder source did not recover the originating brief or reviews.
TrustLadder source exposes read-only ledger SQL (`cli.py:344–350,610`,
`ledger.py:2678–2692`), Temporal listing/query (`cli.py:368–390`) and GET transcript/
provenance routes (`dashboard.py:132–159`). Those are source interfaces, not a
confirmed accessible live workflow or owner communication channel. Listing the
live TrustLadder state directory was denied. No live ledger address, run IDs for
the final reviews, or usable owner-confirmation interface was established.
No workflow mutation, submission, publication or write-based owner contact occurred.
**No new owner confirmation was obtained.** Do not describe these GETs as one.

### Missing original acceptance contract (D1)

Responsible workflow: TrustLadder's owning workflow for
`tessera-65c0767797af-wp1`, through v6 Kernels Lead and the sole integrator.
Required return through that existing workflow:

- Original research brief containing verbatim O1–O4, with originating run/workflow
  identity, content digest and source revision.
- Original protected acceptance files and designation/manifest: exact repository
  paths, bytes/digests, source revision, three failing node IDs and exact commands.
  Their filenames and assertions are **unknown**, not inferred from local tests.
- Final two review receipts before the October 23 stop: exact review run IDs,
  source/acceptance-file identities, objective verdicts, stdout/stderr and retained
  test receipts for all three failures. Their receipt IDs and full outcomes remain
  **unresolved** beyond RobTand's report above.

No unnumbered issue-652 bullet has been mapped to an O label. No stageprev,
MLA, serving or other campaign's tests/objectives substitute for these records.
No original objective completion or original acceptance pass is asserted.

## Bounded retained-evidence reconciliation (D2)

Whole packet, read successfully in this pass:
`/mnt/shared/tessera-measurements/issue-652-geometry-sol-20261006/issue652-before-profile-packet-final.json`.
Observed size **128268 bytes**; SHA-256
`490b4fd62e20be1cbf832c7720ce6f4ab65119ab6dbc2743ec40c15f41f9567a`,
matching the advertised whole-file digest. Issue 1386's historical unreadability
report is **not a current observation**. No shared packet was edited.

All six `controls_reused` receipt files were readable. For each, this pass
recomputed SHA-256 of the canonical JSON body excluding `receipt_sha256`
(sorted keys, compact separators), matched its stored and packet-referenced digest,
and matched declared action key, result digest and manifest digest to the packet.
PrismaBuild `src/prismabuild/core.py:3575–3622` in served revision
`43234d0fbd42d6ed76da38658cb9f447ed19a606` confirms the body-digest distinction.
This is **metadata/digest reconciliation only**, not execution of its full validator.

| Control | Matched canonical receipt-body SHA-256 |
| --- | --- |
| Unprofiled q1088 rows2048 | `40914fb09e48449ade9c8fc701d3e13486140e59bfeb266b83ff6ae9c262b9d3` |
| Unprofiled q1088 rows4096 | `3e186fe8eb903fefd0a85f1463a26fa4386f5ff8ce8f7451669db43a6b387596` |
| Unprofiled q1152 rows2048 | `4e34eed1ae4819643a3900af6a5c00559cf2c2043a5288366cc61e273818d5de` |
| Unprofiled q1152 rows4096 | `5520408d810f5a14a8202e23a578e42b854ad1b4b54cf0f8c8508310e229960c` |
| Torch profile q1088 rows4096 | `8d7ae2b8018b10550d973340003880c9e0f3b3a4750e03e8ce18e18a3ff34cdf` |
| Python sampling q1088 rows4096 | `5a60a3901646f78c28b0d749ca0dec617d8df264c4dfbcadd59f9a5d64cf817f` |

Exact action keys and CAS paths remain in the digest-bound packet's
`controls_reused`; this table does not replace them. Independent result/payload,
log, terminal-return, worker/native/runtime attestation and admission-chain
verification was **not completed**. Readability of all other referenced artifacts
was not exhaustively checked; none is newly declared inaccessible or authenticated
by omission. The inaccessible original workflow artifacts are enumerated above.
No receipt is owner-declared applicable or superseded in the recovered G2 records;
**all six applicability decisions remain pending** v6 Kernels Lead disposition
and sole-integrator review. Age, a digest match and packet terminal claims do not
supply that decision.

The packet's `identity_stamp` says original snapshots differ and were retained
without a historical identity gate. Its `exact_missing_full_issue_acceptance`
retains these limits: generated weighted targets, not captured G2 calibration or
complete encoder-unit inputs; sampled q1088/4096 Nsight captures only (eight
filtered launches, nineteen replay passes, default cache flush/base-clock controls);
no direct DRAM counters (L2 proxies only), no production-resident bandwidth ceiling
or causal end-to-end limiter; no selected production candidate/improvement,
adversarial tie population, packed-wire byte comparison or full-unit matched
seconds/joules. Generated-workload torch/Python profiles do not isolate full-unit
host overhead. No serving qualification or issue closure follows.

## Missing owner disposition (D3)

**BLOCKED:** no attributable G2 retained-adoption versus fresh-qualification
disposition was recovered from either designated owner. There is no successful
qualification action ID, scoped adoption artifact or fresh encoder exact-tuple
approval to name. The closed [issue-690 gate](2026-10-09-690-gpu-gate.md)
illustrates separate obligations; it supplies no G2 permission or runtime pins.

Before reconsideration, v6 Kernels Lead must return a dated, authored, durable
G2-specific disposition, with URI/content digest, identifying each adopted,
superseded or still-pending receipt and choosing retained adoption or fresh
qualification. The sole integrator must retain an independently attributable
review of that exact scope. Any positive decision must bind:

- source commit **and content identity**, incumbent and selected candidate;
- workload/recipe and captured input hashes, shapes and coverage;
- actual hardware/device identity and placement;
- runtime/toolchain/image content identity, flags, graph/cache/clock policy;
- native build/source/toolchain/artifact evidence and the successful qualification
  action plus scoped adoption artifact where required.

These values are **not established as an approved tuple here**. Source/runtime
changes require explicit owner applicability/rebase disposition; do not silently
carry the old snapshot, substitute a fleet default or import issue-690 image pins.

## Finite conditional handoff, not a launch (D4)

| Dependency / responsible owner | Evidence required before reconsideration |
| --- | --- |
| Original WP workflow custodian, v6 Kernels Lead, sole integrator | Original brief, protected file identities and final two reviews described above; resolve the actual failing contract first. |
| v6 Kernels Lead and sole integrator | G2-specific qualification/adoption/exact-tuple decision and independent review described above; retain the October 23 stop until explicitly resolved. |
| Existing issue-1351/1353 measurement execution and Netdata configuration owners | Expose the owning project/configuration path and an admitted host-required execution route with reachable configured Netdata from the actual measurement namespace; supply fresh placement/telemetry receipts for each required GB10. |
| Related issue-1392 owners, via those same execution owners | Confirm the common admission/paired-evidence dependency and scope boundary; its MNBT2048/4096 L8192 TP2 authorization/evidence is not G2 encoder approval. |
| Existing issue-1391 / 1386 / 1390 owners, coordinated by sole integrator | Confirm boundaries: different A-256s custody/runtime scope / packet readability / demonstrated post-approval loader-schema defects respectively. No shared artifact edit until confirmation. |

The [1353 owner-published record, RobTand, 2026-10-22 23:25:27Z](https://github.com/RobTand/tessera/issues/1353#issuecomment-6353553300)
places the unavailable execution/network capability outside served repositories;
its owning configuration project/path is not exposed. This is the **concrete
configuration-owner dependency**, not proof Netdata is currently down. Published
issue scopes were retrieved, but fresh confirmation from those owners was not.
The related [1392 HOLD report](https://github.com/RobTand/tessera/issues/1392#issuecomment-6366265912)
is not a viable encoder execution path. No new executor, collector, host-specific
arm, environment workaround or duplicate issue is justified.

Conditional sequence for the existing owners only:

1. Return the missing workflow/authority records and scope confirmations once;
   independently authenticate the selected retained artifact chain. If unavailable,
   remain BLOCKED here. Do not repeat unchanged audits.
2. If fresh work is required and explicitly authorized, execution owners first
   establish cheap real-invocation admission/placement/telemetry preflight under
   existing configuration. Expected evidence: actual source/runtime/device,
   admission, fresh terminal result, time-aligned reachable telemetry and retained
   native/input identities. Generic test-arm availability is insufficient.
3. Kernels Lead names the **single smallest qualification action** from the
   recovered original contract, exact case, command, action identity and approved
   budget; the integrator reviews before execution. That action cannot honestly
   be named now because the protected contract and tuple approval are missing.
   No candidate sweep or production optimization is proposed. Retained adoption,
   if chosen, must instead name the successful retained action and adoption record.
4. Current execution budget is **zero GPU minutes / zero launches**. Any later
   action needs an owner-recorded time/spend cap, timeout and stop criteria before
   launch. Stop on refusal, identity mismatch, missing telemetry/native evidence,
   acceptance failure or cap exhaustion; no automatic retry or sweep. Return the
   first action's terminal evidence to both owners before any expanded campaign.
5. Sole integrator accepts or rejects applicability, then hands only the approved
   missing scientific legs and recovered checks back to the existing campaign.
   TrustLadder retains publication/workflow transitions; intake retains duplicate
   decisions. This document executes none of that sequence.

## Preservation and verification (D5)

Only this new handoff is changed. Production source/defaults, wires, serving/runtime
pins, source-digest/admission gates, historical evidence, shared owner artifacts
and protected acceptance checks are untouched. No GPU measurement, submission,
publication, physical-action escalation or scientific acceptance claim occurred.
Local documentation guards verify repository hygiene only, not authority,
original O1–O4, receipt applicability or the original measurements. Manual review
of the attributed evidence and missing-record inventory remains the acceptance
method for this decision package.

Verification command (x86 only; no GPU check requested):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Observed: **212 passed in 143.61s**, no skips, NO CUDA, zero device allocations.
Retained stdout was read back with `read_test_artifact`. No original acceptance
checks were available to run; no replacement acceptance test was created.

```text
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2bd625f5d3dc36f5ae23f366c5b800eefeff86ccbae7c9dbf7bd75c39988e156","size":1803}},"operation":"2e09e495b46caf0dbe8e85d520d6df994d7748b35509cc3dd544a9c4cec8f223","receipt":"9825765cfd2b4d3154ce2766fc08f45f8b77ad47eea65c5a246a6e1deac5711f"}]
```
