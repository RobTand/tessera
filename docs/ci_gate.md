# Hosted CPU merge gate

Every PR push selects tests with `tools/impacted_tests.py`, Tessera's reverse
import graph. Selector errors select the full population. Changes to gate or
workflow files also select the full population. The final check is `pr-gate`.
The workflow handles `opened`, `synchronize`, and `reopened` events. Label
changes take effect on the next push or reopen.

The final `gate` check runs the full CPU population on same-repository Mergify
batch heads. Their branches start `mergify/merge-queue/`, and their author must
have login `mergify[bot]` and numeric ID 37929162. Same-repository PRs labelled
`gate-canary` use that full path too. The checkout is the event's head SHA,
which is the integrated commit for a batch draft.

## CPU populations

Both lanes start with the selected test files. The bytes lane has no torch.
`tests/conftest.py` reports files it cannot collect because their third-party
dependencies are absent. It also proves byte-layer imports, the empty-denylist
refusal, the shrink-only boundary allowlist, and the built wheel and sdist.
The torch lane installs CPU torch 2.11.0 and Triton 3.8.0 from pinned
requirements. Triton is needed to import native owners for CPU metadata tests;
it does not make these runs CUDA qualification. Both set `CUDA_VISIBLE_DEVICES`
to the empty string. They use four xdist workers, `--dist worksteal`, and one
native thread per worker, within inherited CPU affinity.

Every shard uploads JUnit, a verdict, and `--surface-json` output. Reports
live in the ignored `.pytest_cache` directory so source-identity tests still
measure a clean checkout. The population
reports device presence, CUDA allocations, uncollected modules, and verbatim
skip reasons. CUDA-gated tests skip on these runners. A CPU pass does not prove
the CUDA population. The coordinator's PrismaBuild GPU run with `--strict-cuda`
remains the CUDA qualification, as described in `AGENTS.md`.

## Scheduling and verdicts

The partition and retry code is adapted from PrismaBuild's hosted gate.
`GATE_SHARDS` is the single per-lane shard setting, four. Full runs
therefore use eight test jobs across `ubuntu-24.04` and `ubuntu-24.04-arm`.
The partition sorts files by recorded duration, then path, and assigns each
to the least-loaded shard. Unknown files use the known p90, or one second for
an empty inventory. Small PR selections use one shard per lane.

The aggregate checks that every selected file belongs to exactly one shard in
each lane. Verdicts must match the plan's base SHA, head SHA, population, mode,
lane, and assignment. Exit 0 passes, 1 is a final test or budget failure, and
75 means no verdict. A separate retry job reruns only missing verdicts, once.
A later pass never erases an earlier failure. A full gate refuses an empty
CPU population; an empty PR selection records that no tests were selected.

## Per-test budgets and quarantine

`tools/ci/gate_budget.json` starts in measurement mode. The 20-second default
and exception list cannot grow against the event's base commit. Enforcement
uses PB's 1.5 tolerance. It starts only after hosted measurements establish
which exceptions are needed. An unavailable base commit gives no verdict.
Timings include setup and teardown through xunit1 JUnit.

`tools/ci/gate_quarantine.json` records exclusions and their owning issues.
The three projection artifact mutation cases are excluded under open issue
[the projection fixture issue](https://github.com/RobTand/tessera/issues/1566). Their module-scoped
safetensors fixture contaminated later consumers in the first hosted canary;
the consumer tests remain active. Hosted failures must be investigated and assigned
an open issue before exclusion. No exclusion claims CUDA coverage.

## Branch protection transition

`ci.yml` provides a separate `branch rule` job using the unchanged branch-rule
script and legacy exemptions. Authenticated Mergify batch drafts bypass the
PR naming step in the workflow. `pure` retains that step during rollout and
runs its bytes tests with xdist. Its boundary and distribution proofs remain.
After this gate lands, the lead must set classic required checks to only
`branch rule`. The queue then judges `pure`, `pr-gate`, and `gate` according
to its policy. This PR does not change repository protection or queue rules.

## Qualification

The fixed-master inventory, hosted canaries, measured durations, budget
exceptions, and quarantine owners are recorded with their source SHAs and
PrismaBuild action keys during rollout. Measurement mode is not a completed
budget qualification. No hosted timing claim is made before those runs finish.

Boundary-proof failures produce final failure records before pytest starts.
Passing records require both timing evidence and a controller population with
no CUDA execution. The quarantine planner checks that every excluded node has
an owning issue that remains open. Reruns swap the architectures by shard index
so each file can be qualified on both architectures.

`tools/ci/gate_evidence.py --manifest RUNS.json --output REPORT.json` reads
retained JUnit repeats and their CPU populations. `--durations-output` writes
the per-lane median file times. `--budget-output` builds a bootstrap exception
list from hosted observations, using twice each measured maximum rounded up.
Every source JUnit digest is retained. The manifest records lane, repeat,
JUnit path, surface path, and the PB action key or hosted run and artifact.

Each xdist worker publishes a collection receipt. Every assigned file must
appear in that receipt or in the population's explicit missing-dependency
list. Collected helper modules with no test items remain accounted for. An
all-skipped or all-deselected assignment can pass with pytest exit 5 only when
those collection and population checks complete. It never claims a CUDA pass.

The first four-job trial used two shards per lane. Its ARM torch job alone
took 585 seconds after a 67-second plan, already beyond the ten-minute gate
target before aggregation. The next trial uses four shards per lane. This
is a sizing trial, not a completed p90 qualification.
