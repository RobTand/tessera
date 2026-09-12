# Full-engine resource report: the frozen producer schema

Status: producer contract for #399, 2026-09-12. **Admission stays closed until a
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

## Delivery boundary

This document freezes the schema. It does not implement the derivation, close any
domain, or admit anything. #399 and PrismaQuant #420 both remain open until a
report with closed domains is recomputed and accepted by a consumer that never
ran producer code.
