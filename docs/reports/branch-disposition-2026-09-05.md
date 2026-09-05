# Branch and leftover-work disposition (2026-09-05)

Issue #198 is closed and PR #200 is merged at `39fba4f`. This audit covers all
74 originally observed branch references (local plus origin), 45 worktrees,
and all nonignored changes in the 10 dirty worktrees. The per-reference tips,
commit lists, status paths, decisions and reasons are in
[the machine-readable audit](branch-disposition-2026-09-05.json).

Every item is integrated, salvaged as bounded historical evidence, superseded,
or retired. Nothing here is an unassigned future task. The user authorized
removing every branch except master after this disposition.

## Work worth retaining

- Recovered completed trailing-refit and raised-reach served receipts that
  never reached master; byte-checked the former against its original commit,
  retained exact KL JSONs and hashes, and recorded the limits in
  [the recovery receipt](../measurements/branch-recovery-2026-09-05.md).
  Existing defaults and evidence grades are unchanged.
- Corrected the LUT encoder docstring from its already committed oracle
  receipt. This is prose, not a new numerical result.
- Fixed the census CLI import collision exposed by validation with the real
  PrismaQuant checkout: its regular `tools` package hid Tessera's namespace
  package. Tessera's repository tools now form a regular package too, and an
  isolated CLI regression first failed with `ModuleNotFoundError`.
- Corrected two pre-existing validation assumptions: chmod is writable by
  root with CAP_DAC_OVERRIDE, and FP8 now refuses an unsupported TCQ body
  before checking sidecar/blob agreement. The tests exercise those actual
  refusals; runtime admission rules did not change.
- Retired the completed fixed-namespace trailing-refit launcher. A prior
  duplicate execution raced into the same export directory; its proposed
  PID-only mkdir lock was not suitable to restore. The refusing stub points
  to the completed receipts, and the old driver remains in git history.

## Non-ancestral histories that needed content review

Patch-equivalent branches are already integrated. The remaining histories
were reviewed against their closed issue decisions and the current owner of
the functionality, rather than classified from their commit dates.

| branch family | disposition and evidence |
|---|---|
| `claude/ts-75-served` | **recovered evidence; retire precursor**. Original byte/gate JSONs match surviving receipts exactly. Preserve completed prefill KL and tail limitations in branch-recovery-2026-09-05.md. Reject PID-only mkdir lock; retire completed fixed-namespace run_all launcher. |
| `codex/ts-114-receipt` | **superseded**. The linked-worktree failure row is historical, not a current master receipt. Issue 114 closed on the corrected 66f6480 pair and 005a53f ledger; retain original failing row in the bundle rather than overwrite current population provenance. |
| `codex/ts-5-moe-partition` | **integrated**. Whole-layer partition and checked assembly are in src/tessera/serving_parts.py and the current export/merge drivers; issue 5 closed by PR 127 at 196c142. |
| `codex/ts-5-plan-coverage` | **integrated**. All tests introduced at 740f79e remain in test_serving_parts.py and test_ts5_sidecar_check.py; current versions add shard permission and duplicate-name coverage. |
| `codex/ts5-moe-role-invariant` | **integrated**. Current scheme.py describes per-projection containers; moe_route.py diagnoses expert rows. Current GLM/LFM harness scope supersedes the old prose, and measured MoE cells supersede dense-only assumptions. |
| `codex/ts5-sidecar-duplicate-shards` | **integrated**. census.py now says closed-world table, independent of schema revision; duplicate-shard gate and regression are on master. |
| `issue39-baseline-reach` | **superseded**. Issue 39 explicitly closed via _value_cases and scope-mutation regression at 74f573f / 780817a; this session-kill snapshot is an earlier implementation. |
| `muse/ts-101-encver` | **rejected alternative**. Issue 101 explicitly rejected the hand-maintained ENCODER_VERSION design. Current encoder_identity.py derives a behavior-fixture identity (34ff395 and successors); do not reintroduce a second version counter. |
| `muse/ts-75-trailing` | **superseded**. Wedged-worker harness precursor superseded by merged equal-pass trailing objective (9add21d / f735d94), current refit_trailing_pair.py and recovered completed served receipts. |
| `muse/ts-86-ignore` | **integrated; retire stale receipt**. Unified written-ignore and construction census supersede old report. Issue 86 explicitly retired b0da95e and the wrong vision qkv spelling; current qkv_proj mapping and completeness gates own this policy. |
| `muse/ts-87-landfloor` | **superseded; retain historical evidence**. Raised-only upward landing integrated at 9d9767b with _bump_below_floor shared by land_at_least. Precursor census/test counts remain historical in bundle; later unraised-row decision and recovered served A/B bound current claims. |
| `perf/ldlq-tcq-graph` | **superseded**. 30cf0d6 has separate TcqGraph/TcqTables/_viterbi_core and obsolete experimental APIs. Master 2903784 replaced it with one _TCQPlan.run for reference/capture, bounded per-thread residency, shared capture lock, and test_tcq_graph.py identity coverage. Retire both old benchmark drivers. |
| `ts-12-mechanism` | **integrated**. All three experiment files at 057011b are byte-identical to master: dense4_plane_census.py, dense4_plane_mechanism.py, dense4_reach_sweep.py. Issue 12 closed with later served b8 parity at cc3eb27 / f0a2d0c. |
| `ts-5-preRebase-b0da95e` | **rejected precursor**. Issue 86 explicitly superseded this old expert-ignore branch: wrong qkv spelling and incomplete allow-list. Current unified written passthrough and construction census retain the useful rank guard. |
| `wf/ts-112` | **superseded**. Current merge_suite.py and tessera._dev suite source/deadline machinery supersede the checkout-shape guard and receipt recorder. Old 2060/0/13 and 1548/3/509 rows are historical, not a current validation population; keep them in the bundle. |
| `wf/ts-113` | **rejected inference; superseded experiment**. Its prefill-identity witness was used to argue compiled lane separation despite rebuild noise. Current original receipt retracts that inference; fresh matched r6 population and independent audit in tessera-compiled-decode-kl-r6-2026-09-04.md supersede it. Do not restore the old architecture claim. |
| `wf/ts-83-scope` | **superseded**. Historical trace timings with unattributed copy/gather kernels do not establish a controlled served latency comparison. Current window-GEMV and r6 receipts state campaign scope; retire the older cross-campaign latency interpretation. |
| `wf/ts-87` | **integrated mechanism; recovered evidence**. Current raised-only landing owns the code. Preserve the completed historical served A/B, including its unresolved chunk-level sign, in branch-recovery-2026-09-05.md; retire predecessor implementations, hardcoded drivers, and interim test/count logs. |

## Dirty worktrees

Tracked and staged binary patches and status lists were saved before any
branch removal. Untracked files remain at their original paths in detached
worktrees; these recovery copies are retired, not live branches or queued
work. No user files need to be discarded to leave a single branch.

| worktree | definitive disposition |
|---|---|
| `/home/rob/tmp/fleet/ts-ladder-wire` | **superseded**. Single-unit scratch SSE probe is replaced by committed experiments/encoder_evidence_drift.py and the measured minor-7 issue 198 receipt. |
| `/home/rob/tmp/musefix/ts-78-79-guards` | **retired**. Hardcoded run-tests.sh references old six-slot fleet locks, CPU lists and venv paths; it adds no product capability. |
| `/home/rob/tmp/musefix/ts-84-reach` | **rejected alternative**. Sigma-ceiling refusal changes accepted encoder inputs. Issues 84/89 resolved this through requested/realized/delivered/saturation reporting; window_table_reach and the arity correction are on master. Do not replace reporting with the abandoned refusal policy. |
| `/home/rob/tmp/musefix/ts-86-ignore` | **superseded**. Untracked impacted_tests.py is an older five-function selector; current shared resolver handles dynamic imports, namespace/package dependencies and collection policy. |
| `/home/rob/tmp/musefix/ts-89-dyadic` | **retired duplicate measurement**. The tracked refit2 JSON differs only in secs (16 timing values); payload hashes and quality fields agree. The untracked one-unit decompose screen is superseded by committed residue/band/full-H analyses and does not qualify a new sigma default. |
| `/home/rob/tmp/musefix/ts-99-glmattn` | **retired**. Deletion of pbrun_result.txt removes a worker receipt, not a product fix. Construction-census admission and name-mapper attestation are integrated under issues 99/108. |
| `/home/rob/tmp/ts111-master` | **integrated; retire obsolete assumptions**. Lane/census regression functions are in current tests. Remaining dense-only roster and not-built-MoE expectations are obsolete now that the contract has measured MoE cells. |
| `/home/rob/tmp/wf108-master` | **integrated**. All 11 modified name-mapper functions are AST-identical to master; mapper_probes.py function is also identical. Issue 108 merged at 50792cc. |
| `/home/rob/tmp/wf87-101` | **superseded**. Staged raised-floor implementation and tests are predecessors of 9d9767b and the shared _bump_below_floor helper. The unraised boundary was separately decided in issue 115. fixid_probe.py only prints identities already exposed by the API. |
| `/mnt/shared/ts50-lut-landing` | **salvaged prose; retire remaining scratch**. Corrected current LUT docstring from its committed receipt. Extra ceiling-refusal test exercises a guard already present; scratch entry/identity checks and report printouts add no current gate. Do not transplant uncommitted historical pass-count or two-point power claims as new measurements. |

## Cleanup and recovery

The verified pre-cleanup bundle is
`/home/rob/tessera-runs/issue-198/branches-before-cleanup.bundle`, SHA-256
`11ee51fd2dec83f656d8754ede76cd4795755e41c24824f5eec74b413401bbc8`.
Tracked/index patches are under the adjacent `worktree-patches/` directory;
original ref and worktree inventories and content comparisons are retained
there too. A final bundle captures the cleanup PR before reference removal.

Cleanup detaches obsolete worktrees at their existing commits, preserving
all files, then removes their local and origin branch references. Master is
the sole surviving local branch and sole origin branch. The post-cleanup
inventory is recorded outside the checkout so documenting the operation
does not require creating another branch after it.
