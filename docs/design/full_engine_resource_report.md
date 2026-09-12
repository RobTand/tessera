# Full-engine resource report: the frozen producer schema

Status: producer contract for #399, 2026-09-12; step-boundary derivation added
the same day. **Admission stays closed until a
consumer recomputes this report and agrees with it.** This document freezes the
schema; it does not claim a measurement. No GPU, served, latency, quality or
capacity measurement was run for this document.

The consumer half is PrismaQuant `docs/design/runtime_fixed_resource_admission.md`
(merged 2026-09-08, issue #420). That document specifies what the consumer must
independently check. This one specifies what the producer must emit so the
consumer *can*. Where the two disagree, the consumer document wins on field
semantics and this one is stale.

## Why the producer may not certify itself

`analyze_engine_resource_ledger` today returns `fixed_resources: None`,
`timings: None` and `admission: "not_implemented"`, with six named
`qualification_gaps`. The fix is to **derive** those quantities, not to relax
those gates.

The safety property that makes derivation admissible is that the producer's
arithmetic is reproducible from the raw observations by a party that never runs
producer code. So `derived` is a *claim*, never an input: the consumer
recomputes every number in it from `partition` and `observations`, and any
disagreement refuses. `qualified: true` is not a field in this schema, and a
proof hash is not qualification.

The observation itself cannot move to the consumer. The recorder must attach
before CUDA initialization inside the engine's own process —
`analyze_engine_resource_ledger` raises unless
`history_started_before_cuda_initialization is True` — and PrismaQuant may not
import the serving runtime (its AGENTS.md principle 5). Capture is Tessera's by
constraint. Admission is PrismaQuant's by design.

## Schema identity

`tessera.full_engine_resource_report.v1` — a new closed schema, distinct from
`tessera.full_engine_resource_capture.v1` (raw capture) and
`tessera.full_engine_raw_resource_ledger.v1` (the replay this report is built
on). A field this document does not name is a refusal, not an extension.

Refusing inputs, in the reader before any arithmetic: unknown fields, missing
fields, duplicate JSON keys, nonfinite values, negative sizes, booleans where an
integer is declared, duplicate IDs, and unknown enum values. Every artifact
reference is `{path, sha256}` and is rehashed on read.

## The seven envelope members

Member names are the consumer document's, verbatim, so the two readers cannot
drift apart on spelling.

| Member | What the producer emits |
| --- | --- |
| `identity` | The full-engine run ID and that run's original runtime digest; the v2 context, cost and relation digests; the original source model and the concrete exported checkpoint, bound separately. Both identities are retained, never collapsed. |
| `reference` | The complete canonical census, the fixed auxiliary assignment, one selected row per serving unit, the whole-member `RuntimeBinding`, and the original wire and exporter artifact identities. This must partition the consumer's independently supplied expanded model roster. |
| `workload` | Raw calibration file identity and selected token row; actual prompt IDs, generated decode IDs and sampling; cold/warm cache protocol; batch, scheduled token counts and request boundaries — enough that the consumer recomputes the workload digest rather than reading it. |
| `execution` | Selected *and* actual graph mode, residency, topology, worker/process/device roster, cache capacity, streams and concurrency. A changed execution coordinate invalidates the report; it does not adjust it. |
| `observations` | Every raw artifact, each naming its observation run and interval: the startup-to-finish capture, allocation API/activity arguments, Torch snapshots and history, owner views, KV observations, timing captures and profiles, and the observer-qualification artifacts. |
| `partition` | Exact allocation-generation, serialized-extent and timing-interval membership, derived from `observations`. Stable semantic owner IDs map to concrete observed generations. No ownership is ever inferred from a pointer alone. |
| `derived` | Recomputed fixed `RuntimeResources`, candidate charges for the reference rows, domain totals, and explicit scope. Declared numbers must equal the consumer's recomputation. |

## Per-domain qualification replaces the global gap list

The current `qualification_gaps` is a flat list of six strings, which can only
say "something is missing". This schema replaces it with `domains`: one record
per domain, each carrying its own state, and `derived` carries a number **only**
for domains in state `closed`.

```
domains[<name>] = {
  "state": "closed" | "open" | "refused",
  "evidence": [<observation ids that close it>],   # empty unless closed
  "reason": "<why it is open or refused>",         # null when closed
}
```

The six domains are exactly the six current gaps, renamed to what they must
establish rather than what is absent:

| Domain | Closed when |
| --- | --- |
| `worker_startup` | The recorder is proven attached before CUDA initialization in the engine's own worker process, and the startup interval is covered end to end with no unobserved prefix. |
| `history_join` | Every Torch allocator event has its reciprocal CUPTI memory operation and every CUPTI record its Torch event, with delayed frees preserved; `unattributed_external_records` is empty. |
| `external_closure` | Every external, static, context and host backing has its own disjoint observed charge. `external_native_peak_bytes` stops being null here, and nowhere else. |
| `provenance_admission` | The consumer's relation check binds this run's manifests to the native runs it is related to, with each original manifest retained. |
| `cache_capacity` | KV and recurrent views are deduplicated by physical backing generation, pool sizes and resolved limits are recomputed from raw worker records, and the selected capacity policy is bound. |
| `timing_partition` | Ordered native apply intervals and directly measured adjacent gaps recompose the whole measured step within documented event rounding, with every launch bound to its CPU scope, correlation, stream and device. |

### What is checked today, and what only states its absence

Two of the six have an implemented closure check: `history_join` reads
`unattributed_external_records`, and `external_closure` reads
`external_native_peak_bytes`. Both also go `refused` when the ledger carries
unresolved `issues`.

The other four — `worker_startup`, `provenance_admission`, `cache_capacity`,
`timing_partition` — have **no implemented check**, and `qualify_domains` states
that as their reason rather than closing them. It takes the ledger and nothing
else, so no caller can close a domain by supplying an artifact nobody reads. A
domain that closes because an argument was truthy is `qualified: true` spelled
differently, and this schema does not have that field.

`worker_startup` is the subtle one. The replay refuses outright a capture whose
recorder attached after CUDA initialization, so reaching a parsed ledger does
prove that half. It does **not** prove the recorder ran inside the engine's own
worker process, which is the other half of what this domain must establish and
is the work #399 still owes.

A domain is `refused`, not `open`, when the evidence exists and contradicts the
model — overlapping candidate execution, a physical extent claimed twice, a
foreign device in the trace. `open` means unobserved; `refused` means observed
and wrong. The consumer treats both as blocking, and the distinction is for the
person reading the report, not for the gate.

**Unknown quantities stay null.** No whole-engine residual, no independent-median
subtraction, and no tolerance may become a fixed charge. A domain left `open`
subtracts its term from what `derived` may express; it never gets filled in.

## What `derived` may contain, and the composition

Per the consumer's scalar adapter:

`fixed_resident + sum(candidate_resident) + fixed_activation + max(candidate_activation) + fixed_scratch + max(candidate_scratch) + fixed_KV`

Each term is emitted only when every domain it depends on is `closed`:

| Term | Depends on |
| --- | --- |
| `fixed_resident`, `candidate_resident` | `worker_startup`, `history_join`, `external_closure` |
| `fixed_activation`, `candidate_activation` | `worker_startup`, `history_join` |
| `fixed_scratch`, `candidate_scratch` | `history_join`, `external_closure` |
| `fixed_KV` | `cache_capacity` |
| every timing price | `timing_partition` |

Any term whose dependencies are not all closed is `null`, and `derived.scope`
names which terms are expressible. A partially expressible `derived` is a valid
report; a `derived` that fills a gap is not.

**An unclassified allocation nulls every term, whatever the domains say.** A row
with no single supported owner category, or with no supported lifetime class,
has neither an owner nor a lifetime — so it could belong to any term, and no
term is complete while one exists. `derived.scope.unclassified_allocation_count`
carries the count and `partition.unclassified_allocations` names every row.

### A classified allocation that no term charges

The classifier produces nine `(owner, lifetime)` cells and the composition
charges **seven**. Two fall through: a KV backing whose lifetime is transient
rather than resident, and a candidate allocation with no unit in its scope
stack, because every candidate term is keyed by unit. Such a row is classified
and billed to nothing, which makes the composition **silently short**.

That is the one error direction this schema must never take. An overcount
wastes headroom; an undercount hands a serving gate a budget smaller than the
engine needs, and on a unified-memory box that is an OOM that kills the job. So
an uncharged row is named in `partition.uncharged_allocations`, counted in
`derived.scope.uncharged_allocation_count`, and **nulls every term** exactly as
an unclassified row does. Charging it somewhere would be inventing a rule the
composition does not have, and the composition is the consumer's to reproduce
rather than the producer's to extend.

### The declared step boundary

An allocation that is freed with no unit interval containing either end is a
real transient — on a live engine attention, norms, routing, sampling and every
startup transient land there. Charging it as fixed scratch assumes once per
step; treating it as startup assumes never again. Both are fills, so with no
boundary to read, such a row stays **unclassified** and, by the rule above,
nulls every term.

The boundary was already observed and never declared. The worker snapshots
`execute:N:begin` on entry to a step and `sample:N:end` once the sampler
returns, so the extent of an engine step is in `checkpoints` with a resolved
`trace_index` on each end. What was missing was a structured statement that
those two labels bound one step. The replay could not recover it by reading the
label text: a boundary inferred from a naming convention is a boundary nobody
declared, and one stray checkpoint named `execute:3:begin` would become a step.

So `unit_intervals` gains a sibling. `step_intervals` carries
`{step_id, begin_checkpoint, end_checkpoint}` per step, written by
`declare_step_interval`, which refuses a label the capture never snapshotted, a
reversed interval, a duplicate id and an overlap with the step before it.
**Declaring a step costs no additional synchronize-and-snapshot**; both
checkpoints already existed, and `max_checkpoints` already budgeted three per
execute call.

The interval spans the sampler on purpose. `execute_model` and `sample_tokens`
are separate worker calls, and a step ending at `execute:N:end` would leave the
sampler's allocations outside every step — the direction that silently drops
per-step bytes. There is no fallback to that label: a step whose sampler never
ran stays undeclared, which fails the coverage claim closed rather than
publishing half a step.

Engine ordering is vLLM's and is never asserted here. Two guards stand in for
that claim: `step_intervals` that overlap refuse, and a unit interval that
overlaps a step boundary without being contained in it refuses, exactly as a
crossing unit interval already does.

### What a declared boundary licenses, and what licenses the boundary

With complete coverage, a row outside every unit interval is classified by
**liveness**, not by its allocation index alone:

| Where the lifetime sits | Class | Charged to |
| --- | --- | --- |
| Contained in one declared step | `scratch` | `fixed_scratch` / `candidate_scratch` |
| Overlaps a step without being contained | `activation` | `fixed_activation` / `candidate_activation` |
| Overlaps no declared step | `non_step` | nothing — see below |

A buffer allocated before a step and freed inside it is live while that step
runs and has to be charged, which an allocation-index test would miss. A buffer
live across a step boundary is carried, the same reading already given to a row
that outlives its unit; `fixed_activation` and `fixed_scratch` are separate
additive terms, so calling it carried neither double-counts it nor drops it.

**Coverage is what licenses the inference, and it is declared and checked, not
assumed.** The capture carries `step_coverage` with `declared` — which must
equal the number of intervals carried, or the replay refuses — and `executed`,
the worker's own count of engine steps it ran while armed. The state is
`complete` only when the two agree and at least one step exists; otherwise it is
`partial`, and a capture that declared no steps at all is `unobserved`. Partial
and unobserved behave identically and identically to before: every row outside
every unit interval is unclassified and nulls every term. An allocation from an
engine step nobody declared is live during a step that is not in the list, so
"live during no declared step" would stop being a proof and become a guess.

### A row live during no declared step

The seven terms compose **one engine step**. A row proven live during none of
them is outside that scope, so no term charges it — and unlike an uncharged row,
this is a proof from a declared boundary rather than a cell the composition
forgot. It does not null the terms.

That exemption is only safe while it is visible, because the excluded population
is exactly the startup peak: model-load staging, the profile run, graph capture
scratch. An engine still has to fit those bytes, and a gate handed only the
per-step budget would admit a body that OOMs before it serves a token. So:

* `partition.non_step_allocations` names every such row.
* `derived.scope.non_step_allocation_count` counts them, and stays readable even
  when every term is null, so the hazard is visible before there is a price on it.
* `derived.non_step_transient_peak_bytes` prices them, by the same simultaneous
  sweep as every other transient maximum. It is **not** one of the seven terms
  and does not share their gate: it needs `history_join` and `external_closure`,
  exactly as a scratch term does, and `cache_capacity` has no bearing on it.
* `derived.non_step_transient_peak_scope` says what the number is. While
  `worker_startup` is open the prefix before the recorder attached is
  unobserved, and unobserved off-step bytes can only be missing ones — so the
  peak is published as a **floor** and labelled one, rather than going null. A
  floor is disclosed understatement, and it is only ever readable in the regime
  where `scalar_budget_bytes` is null and nothing is admitted anyway.
* `derived.placement_obligation` states the rule in the artifact:
  **`max(scalar_budget_bytes, non_step_transient_peak_bytes)`**. The budget alone
  is the smaller of two numbers the box has to satisfy.

**The consumer must apply the same filter.** `observations.step_intervals` and
`observations.step_coverage` are carried so it can: a filter that removes rows
from every term and that only the producer can compute is a producer rule nobody
else can check, which is the shape this schema exists to refuse. Until
PrismaQuant #420 carries the matching rule, its independent recomputation will
disagree with `derived` and refuse — which is the safety property working, and it
means the report admits nothing until the consumer catches up. This is a
producer **and** consumer change.

### The second blocker on the same rows: ownership

Removing the lifetime blocker does not make a real capture derive a scratch
term, because the same rows have a second, independent gap and it is not this
one. `observed_categories` is populated in `_checkpoint_owners`, which walks the
allocations that are **live at a checkpoint**. A row allocated and freed strictly
between two checkpoints is never in that set, carries no category, and is
unclassified for want of an owner whatever its lifetime says.

That is not a hypothetical. The banked fixture's own intra-unit allocation
(`0:4608:1`, 512 bytes, `inside_unit`) carries `observed_categories: []` and is
unclassified today for a reason that has nothing to do with step boundaries. On
a live engine every genuine scratch row — unit-local and step-local alike — is in
that position, so `fixed_scratch` and `candidate_scratch` stay null even with
complete step coverage.

What a declared boundary does unblock is the population that **spans** a
checkpoint while sitting outside every unit: a buffer live at `execute:N:begin`,
at a unit boundary, or at `sample:N:end` has an observed owner and, until now,
no lifetime. Those rows stop being unclassified — and because an unclassified row
nulls every term, removing them is what lets any term carry a number at all.

Note where those rows land. A buffer spanning a checkpoint necessarily spans a
step or unit boundary too, so it classifies as `activation`, and both activation
terms depend on `worker_startup`. So this population stops **blocking** the
scratch terms without itself becoming expressible at v1. That is the honest
shape of the change: it removes a blocker, it does not close a domain.

Closing the rest needs an owner for a row that no checkpoint sees. Torch's
history already carries per-allocation `frames`, and
`full_engine_native_owners.checkpoint_site_owners` already matches a validated
rule against them — but it walks `live` at a checkpoint and assigns `shared`,
which is not an owner class. Extending allocation-site attribution to every row,
with a rule that can name `fixed` or `candidate`, is the owed input. It is a
separate issue with its own evidence requirements and it is **not** solved here.

The composition can exceed the measured instantaneous peak, because independent
maxima need not coincide. That is disclosed conservatism. It is not permission to
charge one physical extent twice, and the reverse move — deriving a term by
subtracting a peak from a peak — is prohibited outright.

Transient maxima are computed by sweeping allocation and free events and taking
the maximum **simultaneous** sum over the declared interval. A sum of
per-allocation maxima is not a peak.

The scalar device budget is **TP1, one device, resident, eager, explicit
GPU-allocation scope**. It is not a physical-memory model for two ranks or for
GB10 host/UMA backings, where GPU-addressable bytes, CPU RSS and pinned pages can
name the same physical storage. A report outside that scope refuses rather than
projecting; neither rank sums nor rank maxima may be written into v2 scalar
fields.

## Invariance is scoped to one assignment

An observation of one reference assignment does not establish that shared
workspace, cache policy, launch paths or persistent buffers stay unchanged under
every alternative the allocator might pick. The first bounded report covers **one
complete assignment, one row per unit**. A multi-option table needs an explicit
source-bound ownership/capacity invariance rule and qualified substitution
controls, which this schema does not yet define. CPU fixtures cannot establish
that invariance for GLM.

## Refusal tests the producer owes

Each must fail before the implementation and pass after, and each tampering test
updates the outer hashes so the semantic reader — not only the checksum reader —
is what catches the defect:

omitted unit, owner or extent; duplicate alias; candidate/fixed overlap; reused
pointer generation; live-free mismatch; missing parent segment; unknown API;
stale source, calibration, workload or runtime; altered cache capacity; foreign
rank or device; missing timing tail; overlapping streams; unsupported boundary;
nonfinite or boolean numeric fields; changed `derived` totals; and
assignment-dependent shared state.

A positive synthetic receipt proves the parser and the recomputation contract
only. The fixture at `tests/fixtures/full_engine_resource_ledger.json` says so on
its face — `"synthetic CPU-only parser fixture, not a GPU measurement"` — and a
report built from it carries that scope. A positive *real* report additionally
needs the qualified original measurements on the named hardware.

### The four unclosable domains, and what v1 does not observe

This is the largest known gap in v1 and it is stated here rather than left for a
consumer to discover. `worker_startup`, `provenance_admission`, `cache_capacity`
and `timing_partition` carry a state, but v1 emits **no observation a consumer
can read them out of** — no worker-startup record, no runtime provenance
relation, no KV or pool observation, no timing capture, no stream or launch
record. A consumer that independently recomputes therefore holds all four open
whatever the report claims, and `fixed_resident`, `candidate_resident`,
`fixed_activation`, `candidate_activation` and `fixed_KV` can never become
numbers. **The scalar composition cannot complete at v1 even on a perfect
capture**; only `fixed_scratch` and `candidate_scratch` are reachable.

Three of the consumer design's negative tests are unreachable for the same
reason — altered cache capacity, missing timing tail, overlapping streams — and
`derived.terms.fixed_kv` is a declared term with nothing to recompute it from.

`step_intervals` and `step_coverage` are carried the same way and for the same
reason: a capture that declared no step says `unobserved` rather than carrying
nothing, because a missing key cannot be told from a forgotten one.

So `observations` names each owed member explicitly and sets it to null:
`worker_startup_records`, `runtime_provenance_relation`, `kv_observations`,
`timing_captures`, `owner_views`, `observer_qualification`. **Named and null,
never absent** — a consumer must be able to tell "this capture did not observe
it" from "the producer forgot to carry it", and a missing key says neither.
Closing any of those four domains means first emitting its member here.

**The test seam is gone, not merely recorded.** `derive_partition` once took a
caller-supplied `domains` mapping, because four domains can never close from a
real ledger and the composition would otherwise be untestable; the partition
then carried `domains_source` to say which it was. Recording it was not enough.
The object that escaped was the **partition**: it is public, it returned every
term filled, and the only refusal lived in an assembler that path never reached,
so the guard was reachable only by a monkeypatch. A seam that produces a
shippable artifact is not a seam. The parameter is removed; the arithmetic is
exercised through `_compose_terms`, which returns values and never an artifact.
`domains_source` stays, always `derived`, because it is the assertion that
crosses the repository boundary — a hand-written artifact is now the only thing
that could claim otherwise, and the consumer refuses that spelling.

**A composition below the observed simultaneous peak is a contradiction, and it
raises.** When no allocation is unclassified and none is uncharged, every
charged byte is inside some term, so the composed budget has to cover the
simultaneous peak over the rows the terms charge. With nothing excluded as
off-step that is the whole capture, so the ledger's own
`torch_observed_live_peak_bytes` — the same quantity, requested allocation bytes
excluding allocator rounding, as the observation's own scope field says. Below
it is not disclosed conservatism; it is an undercount, and an undercount hands a
serving gate a budget smaller than the engine needs, which on unified memory is
an OOM rather than a spill. It raises rather than nulling a term, because a
composition that contradicts its own observations is a defect in the producer,
not a property of the capture. Each of this module's three silent undercounts —
a cell no term charged, a candidate row carrying no unit, and a per-unit maximum
taken over units that can be live at once — would have been caught by this one
comparison. When nothing was excluded as off-step, the ledger's peak and this
module's own sweep over the same rows are required to be **equal**, and a
disagreement raises naming both: it means the two modules replayed different
rows, which is a defect in one of them rather than a property of the capture.

**An allocation is charged to the outermost unit on its scope stack.** Three
things have to name one interval. The replay reads `unit_invocation` from the
outermost containing interval and decides `lifetime_scope` against that same
interval; charging the innermost split a row's lifetime basis from its charge.
It also broke the composition: unit intervals may **nest** — the replay refuses
only *crossing* — so two sibling inner units hold rows that are simultaneously
live, and `max(candidate_scratch)` returned one of the two. Outermost intervals
cannot overlap each other, so a maximum over them is a maximum over genuine
alternatives, which is what the composition assumes.

**A reader refuses before any arithmetic runs.** `derive_partition` is a public
entry point taking a dict, so it checks each row's numbers first: a positive
integer size that is not a boolean, a non-negative history index, and a free
index that is not before its own allocation. The analyzer cannot emit any of
these. The last one matters most: the sweep would settle that free first, drive
the running sum negative, and return a peak that **hides** live bytes rather
than inflating them.

Two smaller rules follow from the same principle. Every id in a domain's
`evidence` must name a member `observations` actually carries, and assembly
refuses otherwise: evidence that points at nothing cannot be checked. And
`derived` does **not** restate `partition`'s `domains` — two copies of one claim
invite drift, and a consumer would have to compare them to learn which is
authoritative.

## Assembling the envelope

`assemble_full_engine_resource_report(ledger, *, reference, workload,
execution, artifacts=())` builds the seven members.
`identity`, `observations`, `partition` and `derived` are read or derived from
the ledger. The other three are coordinates the raw ledger does not carry, and
each is **refused by name** when absent rather than defaulted — a report that
invents its own reference row or workload digest is exactly the failure the
consumer's independent recomputation exists to catch.

Absent is not the only way a declared member can be empty. Each is checked
against this schema's field set (`reference`: `canonical_census`,
`runtime_binding`, `selected_rows`; `workload`: `calibration`, `prompt_ids`,
`sampling`; `execution`: `graph_mode`, `residency`, `topology`), and a member
every one of whose fields is null is refused too — a dict of nulls is truthy, so
"is it empty" was never the question.

The **execution coordinate is refused, never projected over**. `partition.scope`
stamps one composite topology, `tp1_single_device_resident_eager`. A caller
declaring `tp2`, `cudagraph` or `offloaded` used to get a report whose scope
contradicted its own declaration inside one envelope, with nothing comparing the
two. The supported coordinate is `{graph_mode: eager, residency: resident,
topology: tp1}` and anything else refuses, because this schema says a report
outside its scope refuses rather than projecting, and a stamped constant is
exactly the projection it forbids.

A capture that declares itself synthetic keeps saying so: `fixture_provenance`
is carried from the capture through the ledger into `identity`, so no artifact
derived from a synthetic fixture can read as a measurement.

## Delivery boundary

This document freezes the schema. The step-boundary derivation above is
implemented; it closes no domain and admits nothing. #399 and PrismaQuant #420
both remain open until a report with closed domains is recomputed and accepted by
a consumer that never ran producer code. Three things are owed before a real
capture derives a number: allocation-site ownership for a row no checkpoint sees,
the four unclosable domains' observation members, and the consumer's matching
off-step filter.
