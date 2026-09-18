# Full-engine resource report: the frozen producer schema

Status: producer contract for #399, 2026-09-12; step-boundary derivation added
the same day; off-step rows classified without an owner class, 2026-09-17
(tessera#478). **Admission stays closed until a
consumer recomputes this report and agrees with it.** This document freezes the
schema; it does not claim a measurement. No GPU, served, latency, quality or
capacity measurement was run for this document.

The consumer half is PrismaQuant `docs/design/runtime_fixed_resource_admission.md`
(merged 2026-09-08, issue #420). That document specifies what the consumer must
independently check. This one specifies what the producer must emit so the
consumer *can*. Where the two disagree, the consumer document wins on field
semantics and this one is stale.

## Why the producer may not certify itself

`analyze_engine_resource_ledger` returned `fixed_resources: None`,
`timings: None` and `admission: "not_implemented"` with six named
`qualification_gaps` until tessera#399 (2026-09-18); it now returns
`admission: None` beside a `pricing_scope` naming the partition report, and
`derived.admission`, `derived.fixed_resources` and `derived.timing_terms` are
**derived** there from the ledger's own classified event lifetimes and domain
evidence. The fix was to derive those quantities, never to relax the gates.

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

`tessera.full_engine_resource_report.v2` — a closed schema, distinct from
`tessera.full_engine_resource_capture.v1` (raw capture) and the replay this
report is built on, which is `tessera.full_engine_raw_resource_ledger.v1`
without the ownership derivation and `…v2` with it. The v2 raw ledger
(2026-09-18, tessera#548) is the v1 rows plus exactly the two things #548
names: a non-null `owner_views` observation, and the native boundary tensors
carried one allocation row per `(unit_id, invocation, kind)`, with a collision
on that key refusing the ledger rather than publishing an ambiguous boundary
row (`full_engine_ownership.boundary_rows`). The version is set from the
derivation having run (`full_engine_resources._derive_ownership`), never
declared beside it, so "the schema says v2" and "the observation is present"
are one fact. v2 (2026-09-18, tessera#399) adds three `derived` members — `admission`,
`fixed_resources`, `timing_terms` — and one `partition` member,
`observer_allocations`; every v1 member keeps its name and meaning, and the
seven-member envelope and the `observations` key set are unchanged. A consumer
pinned to v1 refuses a v2 report by its schema string — PrismaQuant's
`read_full_engine_resource_report` raises on an unsupported `schema` before it
reads any field — and that is the intended boundary: the PrismaQuant consumer
learns v2 in its own change, never by this producer's say-so. A field this document does not name is a refusal, not an
extension.

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

### What is checked today

All six domains have an implemented closure check in `qualify_domains`, which
takes the ledger and nothing else, so no caller can close a domain by supplying
an artifact nobody reads; a domain that closed because an argument was truthy
would be `qualified: true` spelled differently, and this schema does not have
that field. `history_join` reads `unattributed_external_records`,
`external_closure` reads `external_native_peak_bytes`, `worker_startup`
recomputes its equalities from `worker_startup_records` (routed-owner receipts)
or from `owner_views.dense_startup_check` (dense artifacts), `cache_capacity`
recomputes them from `kv_observations`, `provenance_admission` recomputes the
`runtime_provenance_relation`, and `timing_partition` reads `timing_captures`.
All six go `refused` when the ledger carries unresolved `issues`.

`provenance_admission` closes on `runtime_provenance_relation`
(`tessera.full_engine_runtime_provenance_relation.v1`): named equality checks,
each carrying the values it compared and `agree` — the ledger's identity digests
against the plan's, the configuration digest the launch recorded against the
one the worker observed, the core manifest digest and `core_files_unchanged`
against the manifest's file count, the image id the launch declared against the
one the installer inspected, the plugin installer evidence digest,
`package_files_unchanged_from_installer`, `module_identity_errors == []`, and
the collector and workspace library digests. The domain recomputes `complete`
from the checks; a relation whose `complete` says true over a disagreeing check
refuses.

`timing_partition` closes on exactly one `timing_captures` record of schema
`tessera.full_engine_timing_observation.v1` that names the same served object
as the ledger — equal `configuration_sha256`, `model_sha256`,
`assignment_sha256`, `canonical_units_sha256` and `runtime_manifest_sha256`
(`TIMING_BOUND_IDENTITY`) — whose `partition.established` is true, and every
one of whose qualification checks (`single_worker`, `identical_tokens`,
`arms_complete`, `step_shape`, `stream_coverage`, `gpu_operation_counts_agree`,
`device_exclusivity`) passed. The workload digest is deliberately not bound:
the timing pass declares its own workload (identical-token control and
partition arms on the calibration prompt), and its digest travels with the
terms. The record is the timing pass's own; the domain recomputes nothing
about timings itself and `derived.timing_terms`
(`tessera.full_engine_timing_terms.v1`: `workload_sha256`, `timing_samples`,
`phases`) restates the record's per-phase, per-sample terms.

`worker_startup` is the subtle one, and it needs two independent sides before it
closes. The replay refuses outright a capture whose recorder attached after CUDA
initialization, so reaching a parsed ledger does prove that half; the other half
is that the sample was taken in the engine's own worker process after
`process_weights_after_loading` and after `lock_workspace()`, and that the
ledger's `fixed`-owned, never-freed rows sum to exactly the routed-owner
receipt's own `resources.resident_bytes`. Neither side is trusted about the
other: the sample is the engine's, the figure is an independently produced
artifact's, and the equality is what ties them together.

A dense artifact has no routed-owner receipt; its second side is the artifact's
own `tessera_serving_manifest.json`, whose digest the plan binds. The worker
samples `torch.cuda.memory_allocated()` at arm beside each roster unit's
`resident_bytes_resident_mode`, and the replay (`dense_startup_check`) requires
the ledger's candidate-owned resident rows per unit to **equal** that figure and
the allocator sample to bound the ledger's live bytes at `ready_for_workload`.
A disagreeing unit lists its resident rows by census owner or allocation site
under `resident_rows`, the total is `manifest_unpriced_resident_bytes`, and the
domain refuses. The equality is exact by design: on the 2026-09-18 mixed3
capture 110 of 112 units agree and two do not, because the exporter's formula
prices per-row scales for FP8 (`export_tessera_serving.py:2546`) but neither the
BF16 `row_scale` buffer (`:2539`) nor the NVFP4 4-byte
`trellis_input_global_scale` (`:2520`), and the runtime publishes no table
naming each family's resident tensors. That is an export-side finding, recorded
rather than absorbed: a `manifest <= ledger` reformulation would be satisfied by
any leaked plane.

`cache_capacity` has the same two-sided shape and one extra condition. The
physical backings are the runtime's own deduplicated storages, re-added by the
consumer, and the ledger's `kv`-owned, never-freed rows must sum to that same
extent. Unlike the startup sample this record may only come from a **read-only**
pass: the intrusive resource pass sets `runtime_admission` false and says in its
own pass evidence that its synchronized snapshots are timing- and
admission-ineligible, so the record that closes the domain comes from a stock
engine's own worker RPC. The intrusive pass's record is still carried, as the
capacity witness the two passes are compared with.

A domain is `refused`, not `open`, when the evidence exists and contradicts the
model — overlapping candidate execution, a physical extent claimed twice, a
foreign device in the trace. `open` means unobserved; `refused` means observed
and wrong. The consumer treats both as blocking, and the distinction is for the
person reading the report, not for the gate.

**Unknown quantities stay null.** No whole-engine residual, no independent-median
subtraction, and no tolerance may become a fixed charge. A domain left `open`
subtracts its term from what `derived` may express; it never gets filled in.

## What `derived` may contain, and the composition

`derived` carries three members beside the terms, each derived from the
partition and never set by a caller: `admission`
(`derive_admission(partition)`), a verdict `admitted` | `refused` with its
reason, admitted only when no domain is open or refused and every term is
expressible; `fixed_resources` (`derive_fixed_resources(partition)`), the state
`expressible` | `inexpressible`, the seven terms, the scalar budget, the
non-step peak, `unavailable_terms`, the invariance scope and the pricing scope;
and `timing_terms` (`derive_timing_terms(ledger, partition)`), the timing
observation's per-phase, per-sample terms restated, null unless
`timing_partition` is closed. No residual, no median subtraction and no
tolerance enters any of the three.

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

**An owner class is required exactly where a term reads one.** The classifier
asks the lifetime question first, because its answer decides whether the
ownership question is one the composition asks at all. A row proven live during
no declared engine step is charged by no term, so no term reads its owner class:
it is off-step on its lifetime alone, and its `owner_class` is `null` when no
census saw it. Every other row needs a class, because a term that charges it
needs the invariance the class asserts. Nulling the whole partition for a field
no term reads refuses on an absence of information rather than on a
contradiction, and that population dominates a real capture: 19,828 of the a5
capture's 21,104 unclassified rows, 79.5 GB of its 80.2 GB, are rows no
checkpoint census saw and that its own complete step coverage proves live during
no step (tessera#478).

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

**The coverage gate has no headroom, and that is the first thing a real capture
will answer.** `executed` is `FullEngineResourceWorker._resource_calls`, which
counts *every* `execute_model` call the worker makes while the recorder is
armed, including calls past `max_execute_calls` that take no checkpoint and
therefore declare no step. The capture driver sets `max_execute_calls: 2`
against `max_tokens: 2` — one prefill, one decode. If the pinned vLLM makes any
further armed call, a finish-request step or a scheduler pass carrying no
tokens, then `executed` is 3, `declared` is 2, coverage is `partial`, and every
row outside every unit interval is unclassified again. That is the gate working:
it fails closed, exactly as it should, on a capture whose step list is not known
to be complete. But it means a real capture may derive nothing for a reason
nobody planned, and the fix is driver-side headroom in `max_execute_calls` —
declaring every armed step the engine actually runs — not a looser classifier.
Nothing here asserts what the pinned engine does; the count is observed, and
this paragraph says what to look at when a real capture comes back `partial`.

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
term, because the rows a term charges have a second, independent gap and it is
not this one. `observed_categories` is populated in `_checkpoint_owners`, which
walks the allocations that are **live at a checkpoint**. A row allocated and
freed strictly between two checkpoints is never in that set and carries no
category.

That is not a hypothetical. The banked fixture's own intra-unit allocation
(`0:4608:1`, 512 bytes, `inside_unit`) carries `observed_categories: []` and is
unclassified today for a reason that has nothing to do with step boundaries. On
a live engine every genuine scratch row — unit-local and step-local alike — is in
that position, so `fixed_scratch` and `candidate_scratch` stay null even with
complete step coverage.

The same gap does **not** block the off-step population, and since tessera#478 it
no longer nulls the terms through it. An off-step row is charged by no term, so
the owner class it lacks is a field nothing in the composition reads; it is
named in `partition.non_step_allocations` with `owner_class: null` and priced in
`derived.non_step_transient_peak_bytes`, exactly as an owned off-step row is.
What that ordering changes on the a5 capture is 19,828 rows and 79.5 GB of the
80.2 GB unclassified total; what stays unclassified is the 1,276 rows some term
would charge — 899 unowned in-step transients, and the 370 `shared` boundary
tensors and runtime roots below.

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

Closing the rest needs an owner for a row that no checkpoint sees, and a
supported class for the rows a census does see and labels `shared`. Torch's
history already carries per-allocation `frames`, and
`full_engine_native_owners.checkpoint_site_owners` already matches a validated
rule against them — but it walks `live` at a checkpoint and assigns `shared`,
which is not an owner class. Extending allocation-site attribution to every row,
with a rule that can name `fixed` or `candidate`, is the owed input. It is a
separate issue with its own evidence requirements and it is **not** solved here.

Neither is the `shared` label itself. The native apply's boundary tensors and
the persistent runtime roots are observed, and what they lack is not a name but
an invariance: whether their bytes move with the selected assignment is a
property of a **set** of assignments, and one capture observes one. Labelling
them `fixed` or `candidate` from a single capture would write a claim no
observation supports, so they stay unclassified and keep nulling the terms they
would be charged to. The boundary ledger that resolves them is v2 work on both
sides of the schema.

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

### Every owed observation has a producer; what still holds terms null

`worker_startup_records`, `kv_observations`, `runtime_provenance_relation`,
`timing_captures` and `owner_views` all have a producer and a closure check
(tessera#399, 2026-09-18). `owner_views` is
`tessera.full_engine_ownership_observation.v1`: one view per Torch allocation —
`class` (`candidate`, `fixed`, `kv`, `observer` or null), `unit`, the `rule`
that decided it and the allocation `site` — with the rules declared in
`full_engine_ownership.RULES` and the evidence they were applied to (the plugin
package path and file inventory, the vLLM root and file inventory, the observer
roots and libraries, the plugin JIT prefix, the JIT cache prefixes), beside the
external CUDA record classes (`plugin_jit_static`, `observer_static`,
`image_static`, `image_jit_static`, `unit_window_external`,
`library_external`), the boundary geometry witness, the transient gap witness
and the dense startup check. An `observer` view is charged to nothing and its
rows leave the whole-capture peak; a null view is unclassified and named, never
guessed. `observer_qualification` is still carried as null.

What still nulls every resource term on a real capture is named, not filled.
The `pending_548` rows — shared allocations made after `before_model_load`
(boundary tensors, runner roots, the BLAS workspace, the flashinfer workspace),
whose owner the tessera#548 two-assignment measurement decides — stay
unclassified with that reason (358 on the 2026-09-18 mixed3 capture, 360 on the
eugr one). The TCQ replay tables are built once per distinct trellis and are a
per-family presence cost the terms have no home for; they are attributed to a
unit only when its family has exactly one unit, and otherwise stay unattributed
with that reason. And the dense manifest figure disagrees with the ledger on two
layer-0 units for the export-side reason above, so `worker_startup` refuses
there. `timing_partition` closes on the same artifact, so `derived.timing_terms`
carries the timing observation's terms while the resource terms stay null — a
partially expressible `derived`, which is a valid report.

The two once-unreachable negative tests are reachable now: a timing record
that names a different served object, or whose `established` is false,
refuses `timing_partition`; an altered pool, an overlapping backing or an
intrusive-pass record refuses `cache_capacity`.

`step_intervals` and `step_coverage` are carried the same way and for the same
reason: a capture that declared no step says `unobserved` rather than carrying
nothing, because a missing key cannot be told from a forgotten one.

So `observations` names each owed member explicitly: each carries its record
when a pass observed it, and an explicit null or empty list (with the refusal's
reason beside it) when it did not. **Named, never absent** — a consumer must be
able to tell "this capture did not observe it" from "the producer forgot to
carry it", and a missing key says neither.

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
the two unclosable domains' observation members, and the consumer's matching
off-step filter.

### How the observations are actually produced

One rank's capture is three passes on the same box and the same configured run;
each pass is its own process, and nothing about their pointers is ever compared.

1. **The intrusive resource pass** runs the ledger and the startup sample:
   `python -m experiments.capture_full_engine_resources --config ... --model ...
   --census ... --collector ... --core-manifest ... --runtime-evidence ...
   --output <resource-capture> --calibration ... --calibration-sha256 ...
   --all-units --rank R --world-size W --receipt <routed-owner-receipt>
   --receipt-sha256 ...`. It writes `worker-<pid>/capture.json` (the ledger),
   `worker-startup.json` beside the plan (taken at arm, after
   `lock_workspace()`), and `worker-<pid>/kv-observation.json` -- its own KV
   view, whose pass evidence says `read_only: false` and which therefore serves
   only as the capacity witness.
2. **The read-only KV pass** boots a stock engine with a stock worker and one
   worker RPC: the same command with `--observation-mode kv` and no `--collector`
   or `--receipt`, writing `<kv-capture>/kv-observation.json` with
   `read_only: true`. Its plan identity must be byte-identical to the resource
   pass's, which is why the two invocations share every identity-bearing input.
3. **The timing pass** boots a stock engine with the timing worker:
   `--observation-mode timings --timing-samples N`, one warmup, then per sample
   a control arm and a partition arm on identical tokens, each written to
   `worker-<pid>/sample-<n>-{control,partition}/` with its capture, partition
   and profile. `experiments.full_engine_timing_observation` builds
   `timing-observation.json` from those arms, the launcher's host
   compute-process census (`host-vitals.log`) and the worker's NVML census at
   arm and finish; it exits 3 when the partition is not established, and the
   step-4 driver refuses the phase on that.
4. **The report** binds them: `python -m experiments.report_full_engine_resources
   --capture-dir <resource-capture> --launch-dir <resource-launch>
   --core-manifest <runtime-inventory> --kv-observation
   <kv-capture>/kv-observation.json --timing-observation
   <timing-capture>/timing-observation.json --output <report>`. The ownership
   evidence is read from the launch directory's own records (the worker's
   `runtime-observation.json`, `per-job-runtime.json`, `launch-summary.json`,
   `jit-preflight.json`, the core manifest and the worker startup sample) and
   listed as an artifact of its own; `--without-ownership` is the explicit
   opt-out that leaves `owner_views` null. The join refuses a second pass from a
   different run, a different rank or a differently configured pool, and records
   in its own artifact that the passes were different processes whose pointer
   identities were never compared.

## Observing a served Tessera artifact

The source-BF16 launcher derives its roster from a canonical census and refuses
any checkpoint that carries a `quantization_config`. A served artifact carries
its own roster: `tessera_serving_manifest.json` names every quantized module by
its vLLM module path and every HF tensor the module fuses. `experiments/
capture_full_engine_resources.py --artifact --all-units` reads that manifest
through `experiments/full_engine_artifact.py`, and the observer plan carries an
`artifact_checkpoint` member in place of a census-derived
`reference_checkpoint`:

* **Roster.** One row per manifest module, `g:` for a fused module with several
  HF roles and `l:` for a single-role module, sorted by unit id like the census
  roster. `identity.canonical_units_sha256` digests it.
* **Assignment.** Each unit maps to the family the manifest serves it in
  (`tessera.artifact_observer_assignment.v1`); the report's
  `reference.selected_rows` carries `{unit, format}` with that family as the
  format.
* **Identity.** `identity.model_sha256` digests the bytes the engine loads —
  `config.json`, the manifest and every weight file — never the directory name.
* **Candidate rule.** Artifact parameters are not named `.weight`
  (`wire_bytes`, `trellis_input_global_scale`, ...), so the source-BF16 suffix
  rule would charge every quantized weight as *fixed*. Artifact mode requires
  the native-apply boundary (`--all-units`) and classifies exactly the
  parameters and buffers each boundary module owns as candidate, with a tensor
  also registered outside every owner staying fixed
  (`reference_candidate_tensor_ids`, the same rule the original-wire reference
  proof uses). That rule runs wherever the native-apply boundaries resolve; it
  is not gated on which checkpoint member the plan carries. The suffix rule
  (`parameter_category`) is the fallback for a run with no resolved boundary,
  and it now refuses when a candidate weight is not stored in a source float
  dtype, rather than silently charging that unit's format-sized scales to
  `fixed_resident`.
* **Calibration.** The fixture is `int64[n, 512]` `calibration_ids` for any
  `n >= 1`; the observer reads row 0 and records `rows` in the workload so the
  digest names the shape as well as the bytes.
* **Runtime.** The plugin is installed from a frozen source tree into the
  serving lane's pinned image through `experiments/full_engine_plugin_install.py`,
  which records the same evidence the stock installer does (image identity,
  vLLM core manifest unchanged before and after, plugin files and entry point)
  and reports `upstream_commit` as the installed vLLM reports itself; the plan
  compares it with the selected configuration.

`experiments/report_full_engine_resources.py --capture-dir <observer output>`
replays the ledger and assembles the report, reading `reference`, `workload`
and `execution` from the plan and the selected configuration.

### Host owners that were never CUDA allocations

A checkpoint's owner census walks the runner's persistent state and yields CPU
tensors too. A pinned host tensor is a CUDA allocation the argument domain
records, so a pinned owner with no host allocation to join is a `history_join`
gap. A **pageable** host tensor was never a CUDA allocation: no join was owed,
and reporting it as unmatched charged the scope with a gap it does not have.
Owner rows now carry `pinned`, and `_checkpoint_owners` scopes a row that says
`pinned: false` out of `gpu_allocations_only` into
`pageable_host_observations` — observed and counted, never charged and never an
issue. A row that says pinned, or an older row that does not say, stays a join
gap. The consumer's checkpoint field set is frozen, so the report carries these
beside the checkpoints as one artifact
(`tessera.pageable_host_observations.v1`) rather than inside the rows.

### A checkpoint's storages are the capture's ownership, not one census

Ownership is a property of an allocation. A storage one checkpoint's census
bound to a named owner is that owner's backing for its whole lifetime, and the
consumer recomputes every checkpoint that way: the storages live at its index
that carry any observed owner. A checkpoint that listed only what its own
census yielded at that moment disagreed with that on every unit boundary of a
real capture — a native boundary tensor observed as `native:<unit>:input` is
still live, and still owned, at the next unit's checkpoint — so the ledger
restates `storages`, `owner_count` and `unique_owned_storage_bytes` over the
whole capture's ownership after the replay (`_project_checkpoints`), and keeps
the census the checkpoint itself took as `census_owner_count` and
`census_storage_count`. `owner_count` is therefore the number of owners bound
to a live device storage at the checkpoint; pinned-host and pageable-host
observations are carried separately and are not in it. The census counts and
the pageable-host observations travel in one artifact,
`tessera.checkpoint_census.v1`, because the consumer's checkpoint field set is
frozen.

The per-step batch descriptor (`GPUModelRunner.execute_model_state`) is not a
persistent runtime root: the runner rebuilds it every step, so naming its
tensors by path bound one owner id to a new backing each step, which the
consumer refuses as a duplicate alias. Its allocations stay unowned.

## The engine step after the last token

The pinned engine core (`vllm/v1/engine/core.py`, `step()`) keeps stepping while
the scheduler still holds a request, and the request that just produced its
last token is still held for one more step: `execute_model` is called with a
`SchedulerOutput` that schedules no token, the runner returns its empty output
without a forward, and `sample_tokens` is never called. Measured on the first
served-artifact captures: a `max_tokens=2` request executes three armed calls,
the third with `total_num_scheduled_tokens == 0` and no `sample:3:end`
checkpoint (capture a4 of receipts `399-qwen3-0.6b-20260913`).

Two consequences are wired in. The plan declares `generated_tokens + 1`
execute calls (`declared_steps`) and records every armed call, declared or
not, with its scheduler output, so a capture that executes beyond its budget
names the step it missed. And a declared step that scheduled no token is
closed at its own `execute:N:end`: that is the step's whole extent, not a
sampler fallback -- a step that scheduled tokens and was never sampled still
stays undeclared and holds the coverage claim open. Measured on capture a5 of
the same receipts: `step_coverage` declared 3, executed 3, state `complete`,
with `step:3` spanning `execute:3:begin` to `execute:3:end`, and PrismaQuant's
`step coverage is 'partial'` refusal gone while every other refusal kind stayed.
