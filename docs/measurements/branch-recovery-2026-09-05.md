# Historical measurements recovered during branch cleanup (2026-09-05)

These are recovered 2026-09-04 receipts, not new GPU runs. They were left on
unmerged branch histories while master retained an unfinished report. Each
source file and its SHA-256 is recorded in
[data/branch-recovery-2026-09-05/manifest.json](data/branch-recovery-2026-09-05/manifest.json).
The audit checked the original three trailing-refit JSONs against commit
`4b1bbe67ff08b5ad6eb80bfc9e335af81950d475`: every byte agrees. It also checked
that each served pair has identical metric identity and checked alignment.

## Trailing refit: retain the completed evidence, keep the method opt-in

Recovered from `claude/ts-75-served` at
`a5d90a0f1eeaf3ac5050bf0d33c319b600babca9`. This closes the historical
`PENDING` sections of
[tessera-refit-trailing-served-2026-09-04.md](tessera-refit-trailing-served-2026-09-04.md).
The original campaign is finished; its checkout is no longer required by that
report's in-flight retention instruction.

The historical control and trailing full-H Jacobi arm have equal wire lengths
(220,301,312 bytes each), identical packed codes on all 196 units, and changed
scale planes on all 196. The evidence is the saved byte comparison, not an
assertion that the current encoder reproduces those artifacts. Both KL
receipts are prefill, eager, Qwen3-0.6B, top-1024 support, 4,088 positions on
the same corpus and BF16 teacher. The reported means are lower bounds.

| historical readout | control a4h1 | trailing bjac |
|---|---:|---:|
| mean KL lower bound | 0.520080596 | 0.513790709 |
| mean KL upper bound | 4.05967844 | 3.99206644 |
| p99 KL lower bound | 3.3649649 | 3.46221565 |
| maximum KL lower bound | 7.6786529 | 8.98586777 |

The mean improves while the tail worsens. The saved historical gate reports
`promoted: true` for B-Jac under that campaign's screen and served criteria;
this audit preserves that output as data and makes no current production
promotion. One model/corpus and an old encoder do not qualify a new default.
The alternative GS and every-pass arms failed the historical cross-model
screen. No further run is queued: retain this bounded evidence and keep the
existing opt-in status.

The old branch also proposed a PID-only mkdir lock after duplicate jobs raced
into the same export namespace. That implementation has a publish gap and no
host identity, and is rejected. The completed fixed-namespace `run_all`
launcher is retired rather than revived as a new campaign against cached
historical artifacts. Its original implementation remains in git history.

## Raised reach landing: retain the unresolved served comparison

Recovered from `wf/ts-87` at `264ac6abfafdb621ba1fe309ed5bc2db57c2d367`, whose
code change was integrated separately as `9d9767b`. Its A and B encoders were
`cc3eb27` and `c2b8d83`. This is the raised-row round-up experiment, distinct
from the later unraised-row boundary decision in
[tessera-unraised-reach-boundary-2026-09-04.md](tessera-unraised-reach-boundary-2026-09-04.md).

The saved receipts describe Qwen3-0.6B E4M3 q256=1024, the stock FP8 twin,
eager prefill, top-1024 support, 4,088 positions. They report:

| historical arm | mean KL lower bound |
|---|---:|
| A: nearest | 0.15116254289226014 |
| B: raised rows upward | 0.15324651042926637 |
| A2: same bytes as A, served last | 0.15116254289226014 |

The original branch's paired analysis reports chunk-level t=1.70 with seven
degrees of freedom across eight chunks: it does not resolve the sign at a
two-sided 5% threshold. That analysis is historical and was not rerun in this
audit. The summary fields of A and A2 agree exactly in the recovered JSONs.
Do not promote a weight-space improvement into served-quality evidence, or
roll back a reach invariant on this unresolved sample. The integrated
raised-only reach rule and the later decision to keep nearest rounding for
unraised rows remain the dispositions; no new experiment is left pending.

## What was not recovered as current implementation

Old hardcoded drivers, transient test counts, overwritten timing-only reruns,
and precursor encoder implementations remain retired in the recovery bundle.
The current shared encoders, tests, census, and source-identity machinery
already supersede them. The per-branch and dirty-worktree decisions are in
[the cleanup audit](../reports/branch-disposition-2026-09-05.md).
