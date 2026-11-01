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
to the empty string. They use four xdist workers, `--dist loadgroup`, and one native thread per
worker, within inherited CPU affinity. The collector groups consumers of module-scoped fixtures by file, so each
shared fixture builds once. Cases without module fixtures can share all workers.
Files enter the scheduler longest first; `--no-loadscope-reorder` preserves
that duration order instead of sorting files by their test counts.

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
Hosted scheduling estimates use the larger architecture median per-file sum
from both architecture assignments of retained file and grouped scheduling
trials, excluding quarantined nodes. The optimized matched-reach fixture uses
its hosted cached-fixture measurements. The earlier PrismaBuild estimate remains
recorded as provenance.
Unobserved files retain their PB estimates.
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

`tools/ci/gate_budget.json` enforces a 20-second default, with exceptions
measured in retained hosted canary reports. Exceptions use twice the largest
observed case duration, rounded up. The default and exception list cannot
grow against the event's base commit. Enforcement uses PB's 1.5 tolerance.
An unavailable base commit gives no verdict.
Bootstrap limits include cold-worker observations from both architecture
assignments of the fixed partition. Timings include setup and teardown through
xunit1 JUnit. Scheduling group
suffixes are removed from test names before budget and evidence lookup;
parameter values retain their original identity.

`tools/ci/gate_quarantine.json` records exclusions and their owning issues.
The three projection artifact mutation cases are excluded under open issue
[the projection fixture issue](https://github.com/RobTand/tessera/issues/1566). Their module-scoped
safetensors fixture contaminated later consumers in the first hosted canary;
the consumer tests remain active. One shape accounting node depends on a host-local
PrismaQuant checkout, owned by [the portable accounting test issue](https://github.com/RobTand/tessera/issues/1567).
Six A4 CPU wire nodes depend on unpublished archived artifacts, owned by
[the portable A4 fixtures issue](https://github.com/RobTand/tessera/issues/1568).
One Stageprev receiver-name proof also requires the published PB source tree,
owned by [the observer contract issue](https://github.com/RobTand/tessera/issues/1562).
All matched-reach tests remain active. A worker group derived from fixture scopes builds
their shared grid fixture once. The fixture also reuses standard-normal inverse-CDF
samples across sigma searches, then applies the original scalar multiplication.
Bit-for-bit tests compare it with the encoder's source at every tested width.
Its patch is scoped to fixture construction; production code is unchanged.
The five consumers previously excluded under [the fixture cost issue](https://github.com/RobTand/tessera/issues/1569)
are restored. Grouping alone passed ARM but still timed out on hosted x64;
standard-sample reuse removes that repeated inverse-CDF work.
The remaining tests in those modules stay active. The T4 prepare-inputs tests
use the pinned public PrismaBuild SDK instead of being quarantined. Hosted failures must be investigated and assigned
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
PrismaBuild action keys during rollout. Initial hosted measurements enable
budget enforcement; three passing full canaries are still required before
qualification is complete.

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
target before aggregation. Four shards per lane reduced the cost. A later full trial passed in 581
seconds with file-based worker scheduling; grouped consumers let independent
cases share workers instead of serializing each file. Final-head canaries
measure this scheduling change.

`--canary-manifest CANARIES.json` also accepts retained GitHub metadata as a
list of objects with `run` and `jobs` fields. `run` is the REST workflow-run
response, and `jobs` is its jobs array. The report measures from run creation
through completion of the final `gate` job, including scheduler delays. A
rerun entry must also supply `requested_at`, the UTC timestamp retained when
its REST rerun request was sent. GitHub keeps the original `created_at` on
reruns, so it cannot measure later attempts. Without that timestamp, a rerun
has no wall-time verdict. The report
reports p50 and nearest-rank p90 for completed passing runs and retains failed
runs separately. Per-shard times separate dependency installation, the test
step, and other job work. Missing timing evidence stays null. A timing report
with fewer than three passing full canaries does not complete qualification.
