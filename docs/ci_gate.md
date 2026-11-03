# Merge gate

The merge gate runs on Bazel through BuildBuddy (#1589); [docs/bazel.md](bazel.md) describes the build.
`.github/workflows/gate.yml` keeps the required check names: `pr-gate` on ordinary pull requests, needed to queue, and `gate` on Mergify's batch drafts, needed to merge. It has three working jobs.

- **Bazel.** It starts `tools/ci/bazel_gate.py` on a Remote Bazel runner through `tools/ci/remote_gate.py`. Base and head are Mergify's queue-aware refs for a batch draft (`scopes-git-refs`). For an ordinary pull request they are GitHub's merge commit and its first parent, current master.
  - bazel-diff v70.0.2 selects the impacted targets.
  - A change to Bazel's own inputs (`MODULE.bazel`, its lock, `.bazelrc`, the root `BUILD.bazel`, `tools/bazel/**`), to `gate.yml` or to the gate scripts tests everything.
  - A Python file no target holds fails the gate: the BUILD files are stale. Any other file no target reads selects nothing.
  - Detection that can't establish what to test fails the gate; it never passes empty.
  - The selected tests run on amd64 remote executors, and the runner lane on the runner's own checkout.
  - The impacted targets' packages are uploaded to Mergify as scopes (`scopes-upload`).
  - In the same run, the generated BUILD files, the lock export and the static population parity (`tools/ci/bazel_parity.py`) must be current. The graph-impact check (`tools/ci/graph_impact_tessera.py`) prints its summary line and fails a deepening change.
- **Boundary.** On a hosted runner with only the no-torch pins, it keeps the proofs the no-torch lane made beside its tests: the byte layer imports without torch, an empty denylist refuses, the wheel and sdist ship what they must, and the standalone-boundary allowlist only shrinks (`tools/ci/gate_boundary.py`). `tools/ci/gate_policy.py` adds the base-relative policies below: budgets never grow, and quarantine grows only in a change of its own, with open owners.
- **Post-merge.** On every push to master, two disjoint shards per architecture cover `//...` on amd64 and arm64, excluding manual/GPU targets and the amd64 runner lane on arm64. It records `POST_MERGE_RECORD`, which says whether the merged change's own selection would have caught each failure. An architecture that executes no test is an error.

**Full runs.** Fallback and post-merge selection return a plan. Each shard is a separate Remote Bazel run capped at 55 minutes. Only gate fallback permits at most two concurrent shard runs per repository, `--jobs=16` each, using self-hosted action pools without cloud fallback. Selected gates, post-merge records and development runs stay serial. Mergify's 30-minute `checks_timeout` and merge conditions are unchanged. TL3 keeps BuildBuddy priority when a fallback batch lands while its queue is backed up; no scheduler is added. Every shard uses a committed `tools/bazel/full_targets/*.targets` file through `--target_pattern_file`, never inline full-population labels. The generator partitions every eligible rule, including non-test build targets, by `sha256(label).digest()[0] % 2`. A live query must match the committed lists and manifest. The client requires exactly two disjoint, complete shard results per architecture at the tested HEAD and keeps every invocation ID in the merged record. Missing results, build errors and hard TIMEOUTs fail. Fulls disable both remote result acceptance and local test-result caching.

**History.** Before the first Bazel query in every runner, the client hydrates all origin branches, tags and blobs with the approved bbrun full-history preflight. It verifies the tested HEAD is unchanged, that no reachable objects are missing, and that no shallow or promisor metadata remains. Fetch disables automatic GC and maintenance. Cleanup accepts an already-removed promisor marker; other failures retain their stage and type without Git diagnostics or exception messages. Interrupts propagate.

**Budgets.** Per-test budget overruns appear in `budget_warnings` and summaries without changing a verdict. Assertions and hard TIMEOUTs fail on the first attempt. There are no budget-only or timeout reruns. The budget-growth check remains enforced.

**Secrets.** No child the gate starts on the runner, other than Bazel, gets `BUILDBUDDY_API_KEY`, `MERGIFY_TOKEN`, `GITHUB_TOKEN` or any `*_API_KEY` or `*_TOKEN`. `tools/ci/check_no_secret_leak.py` must pass before any artifact upload.

CUDA-gated tests are not covered: every Bazel test runs with `CUDA_VISIBLE_DEVICES` empty, and the 30 files whose every node needs CUDA are tagged `gpu` and left out. Their run stays on PrismaBuild with `--strict-cuda`.

The sections below describe the hosted gate this replaced. Its populations, budgets and quarantine rules carry over into the Bazel runner (`tools/bazel/python_runner.py`); its scheduling does not.

## CPU populations

Both lanes start with the selected test files. The bytes lane has no torch.
`tests/conftest.py` reports files it cannot collect because their third-party
dependencies are absent. It also proves byte-layer imports, the empty-denylist
refusal, the shrink-only boundary allowlist, and the built wheel and sdist.
The torch lane must report torch and zero uncollected modules. Both the runner
and final aggregator enforce this, and setup asserts a CPU torch installation.
The final summary quotes skip and uncollected counts, device details, CUDA
allocation counts and verbatim skip reasons for each lane.
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
The master-merged trial refreshes observed file sums from hosted JUnit;
unobserved files retain their prior measured estimates.
`tools/ci/gate_refresh.py --manifest MANIFEST.json` replays this bootstrap
refresh from retained artifacts. Growth is allowed only while the gate policy
is absent at the PR base; subsequent budget changes must only tighten.
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

`tools/ci/gate_budget.json` enforces an 80-second default, derived by rounding up
twice the largest measured cold planning consumer, 39.884 seconds. That shared
initialization can be charged to different consumers on different workers;
the previous 20-second default failed otherwise passing qualification attempts.
The retained observations name each consumer, architecture, run, and attempt.
Exceptions for cases above the default use twice the largest
observed case duration, rounded up. The default and exception list cannot
grow against the event's base commit. Enforcement uses PB's 1.5 tolerance.
An unavailable base commit gives no verdict. These are hang limits, not a
fine-grained cost ratchet. The tolerance is a runner constant. The default pytest timeout is 300 seconds;
when a measured slow-node budget times 1.5 exceeds that cap, the collector
uses that exact allowance for the node. The global timeout cannot stop a
measured exception before its allowed budget. Final JUnit budget checks
still enforce the unchanged JSON limits and tolerance.
Bootstrap limits include cold-worker observations from both architecture
assignments of the fixed partition. Timings include setup and teardown through
xunit1 JUnit. Scheduling group
suffixes are removed from test names before budget and evidence lookup;
parameter values retain their original identity.

`tools/ci/gate_quarantine.json` records exclusions and their owning issues.
The policy compares lane, target and owner against the PR base. Entries can
only shrink, including narrowing a whole file to an exact node with the same
owner. Growth or changing an owner requires a PR touching only the quarantine
file and optional Markdown or reStructuredText docs. Rename checks include both
the source and destination paths. When the base has no policy,
the initial reviewed policy bootstraps with the gate. An unreadable base gives
no verdict. The plan checks owners using its job token with `issues: read`;
closed issues and PR numbers fail policy. GitHub unavailability gives exit 75.
Collection validates every exclusion against exact collected node IDs or whole
files and removes items by equality, never pytest's prefix-based `--deselect`.
The torch lane validates entries from modules unavailable to the bytes lane.

`tools/ci/gate_flaky.json` maps exact pytest node IDs to an open deflake issue
and a reason. The planner reuses the quarantine owner client and job token;
closed or missing issues and PR numbers fail policy. It includes marked files
in every selection, so collection refuses missing nodes even on unrelated
changes. `gate_collect.py` runs a marked failing item at most twice, with
normal fixture teardown between attempts. Unmarked failures are final.
Every retry emits `GATE_FLAKY_RETRY` with its node, issue and reason in pytest
output and the shard's job summary. The report crosses xdist to its controller,
which writes the summary once. JUnit records the final outcome and retry issue,
and includes both attempts' time in the node's budget.
Remove a mark in the deflake fix before closing its issue; a remaining mark
with a closed owner fails the next plan.
Never rerun a gate to obtain a pass. This single item retry requires an
explicit mark with an open owner; it does not retry a failed gate or shard.

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
Land in this order: merge [the gate PR](https://github.com/RobTand/tessera/pull/1573) through the current queue with `pure` still
required; the lead lands the queue configuration in a fresh non-draft replacement
for [the draft queue PR](https://github.com/RobTand/tessera/pull/1563); prove one real Mergify batch ran the full `gate`; only then the lead
switches classic protection to `branch rule` alone. Check a fresh ordinary PR
reports `pure`, `branch rule` and `pr-gate` first. Sweep open PRs lacking a
`branch rule` check before switching protection. The queue judges its configured
head and batch checks. This PR does not change repository protection or queue rules.

## Qualification

The fixed-master inventory, hosted canaries, measured durations, budget
exceptions, and quarantine owners are recorded with their source SHAs and
PrismaBuild action keys during rollout. Initial hosted measurements enable
budget enforcement; three passing full canaries are still required before
qualification is complete.

Boundary-proof failures produce final failure records before pytest starts.
Passing records require both timing evidence and a controller population with
no CUDA execution. The benchmark import probe added by current master is
quarantined at one exact torch node under its open P2 owner because the probe
and benchmark import are outside the gate repair footprint.
The quarantine planner checks that every excluded node has
an owning issue that remains open. Reruns swap the architectures by shard index
so each file can be qualified on both architectures.

`tools/ci/gate_evidence.py --manifest RUNS.json --output REPORT.json` reads
retained JUnit repeats and their CPU populations. `--durations-output` writes
the per-lane median file times. `--budget-default-seconds` sets the measured
bootstrap default; `--budget-output` builds a bootstrap exception
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
through completion of the final `gate` verdict step, including scheduler delays.
It retains the job-level completion timestamp as well. On 2026-11-01, GitHub
reported job completion about one hour later than the completed verdict and
cleanup steps. The report uses the verdict step and each shard's `Complete job`
step when present, falling back to job timestamps for older metadata. A
step timestamp before the run origin or job start belongs to a different clock.
In that case the report uses job completion as a conservative upper bound and
names that timing basis. It never subtracts a guessed clock offset. A
rerun entry must also supply `requested_at`, the UTC timestamp retained when
its REST rerun request was sent. GitHub keeps the original `created_at` on
reruns, so it cannot measure later attempts. Without that timestamp, a rerun
has no wall-time verdict. The report
reports p50 and nearest-rank p90 for completed passing runs. With three
qualifying samples, p90 is the maximum. It and retains failed
runs separately. Per-shard times separate dependency installation, the test
step, and other job work. Missing timing evidence stays null. A timing report
with fewer than three passing full canaries does not complete qualification.

The quarantine schema accepts only `lanes` and `issues`, with exactly the
`bytes` and `torch` lists and one owner per listed target. The planner parses
that file once; growth checks, open-owner checks and deselection use those
same records. Extra keys, including the former test-only `entries` shape,
are policy failures. The comparison uses the event's base tip. A stale PR
that still carries an entry removed on master must merge master before it
can pass. A Mergify batch containing quarantine growth and another PR's
code fails the quarantine-only rule; land the growth separately first.
