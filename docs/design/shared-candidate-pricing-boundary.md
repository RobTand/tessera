# Shared-candidate pricing boundary

Decision recorded 2026-09-29 for tessera#567: retain conservative per-unit
pricing until a downstream consumer supports a distinct shared-candidate term.
This is the coordinator's option B. It is not a completed shared-term feature.

## Existing behavior

`src/tessera/decode.py:_replay_tables` memoizes tensors by `(forest, code,
device)`. Calls hitting the same retained entry share tensors. Cache eviction,
separate processes, and separate devices can produce distinct allocations, so
a repeated key is not evidence of one allocation across the whole run.

The producer's seven-term partition has no shared-candidate term.
`experiments/full_engine_resource_partition.py:uncharged_allocations` names a
candidate allocation with no unit as uncharged. `derive_partition` then makes
all composition terms unavailable; `compose_scalar_budget` returns `None`.
The arithmetic-only `_compose_terms` helper is not a shippable report and must
not be used to bypass that admission path.

Keep per-unit pricing, even when it overcounts shared tables. Do not relabel
candidate-dependent bytes as fixed, divide them by the number of units, or
populate a unitless candidate row and interpret its missing charge as zero.
The current refusal is safer than an undercount a consumer would use for fit.

## Boundary for a later change

A shared term needs both a producer and its downstream consumer. Its membership
must bind the candidate/trellis identity, device/rank scope, and actual allocation
lifetime. The consumer must reproduce the same composition and account for
shared and per-unit storage without charging one physical extent twice.

Closing #624 fixes routed resident-layout accounting; it does not add this
shared-term domain or its consumer. The later measured routed compose-table
allocation is a separate retained resource, not proof that the replay cache's
storage is globally shared.

This CPU slice changes no partition schema, domain, ownership class, formula,
wire, runtime gate, or numerical behavior. No new GPU footprint or served
measurement is claimed. #567 stays open for the cross-repository term/consumer
and its measured acceptance. The boundary-only decision above is historical;
the following opt-in Tessera composition implements the producer-side CPU term
without changing an external reader or claiming measured serving acceptance.

## Opt-in Tessera composition

2026-09-30, #567: `export_serving --shared-candidate-pricing` writes an additive
`tessera.shared_candidate_resident_pricing.v1` block under
`shared_candidate_resident_pricing`. It requires a whole artifact (not a serving
part) and an explicit replay specification for every NVFP4 module. Existing
`modules.<module>.resident_bytes_resident_mode`, family summaries and totals
remain conservative per-unit prices, so old receipts are never reinterpreted.
The new block owns its own `resident_bytes`:

```
sum(private_resident_bytes[module]) + sum(group components' bytes)
```

Each group has a trellis identity SHA-256 (grid digest, rate, forest blocks,
convolutional-code memory and generators), sorted unique module members, and
three ordered components: `subsets`, `table_next`, `table_sub`. Shapes, dtype
(`int64`), contiguous layout and exact bytes are measured from the same CPU
builders as `replay_table_bytes`. Private bytes subtract each module's explicit
table specifications from its unchanged legacy resident price. No table is
assigned to a first module; identical groups compose once, distinct groups add.
This is a price, **not proof that a cache key owns one allocation**.

`decode.replay_resident_observation` observes weak references to existing
replay-table generations; it never fills the cache or retains tensors. A group
with zero or more than one complete live generation refuses. Observation names
the actual process ID, rank, device and whole-storage component addresses.
Startup observation v2 carries this proof and the manifest composition;
startup check v2 joins each component to exactly one allocation generation
that was live before `ready_for_workload` and never requested/completed a free
in the capture. It requires exact bytes, the owning plugin allocation site
(`encode._subset_table` for subsets, `decode._replay_tables` for transitions),
and no private owner, unit scope or conflicting category. Missing, overlapping,
aliased, stale or ambiguous storage, membership, dtype/layout, process/rank or
device refuses by name. An evicted table that leaves an extra live component
in the ledger does not disappear: an unknown candidate row remains uncharged
and prevents composition.

The partition independently recomputes this join from the ledger, including
private unit sums and the live-at-ready bound; stored closure flags are not
admission. Before indexing ownership views it requires the original list to
name each allocation exactly once, with no omitted or extra views. Readiness
comes from exactly one ledger `ready_for_workload` checkpoint; its integer
trace index must agree with the startup check and is the allocation/live-byte
cut used by requalification. The union of shared group members must equal the
complete NVFP4 module roster, independently of agreement between the manifest
and observed copies. Only qualified rows receive the `shared_candidate` owner class.
`tessera.full_engine_resource_partition.v2` adds the term
`shared_candidate_resident` keyed by group ID, gated by the same
worker-startup/history/external domains as per-unit residency. Its scalar
composition adds all shared groups once to the existing seven terms. Unknown
unitless candidates still null every term. Legacy partitions remain v1.
`tessera.full_engine_resource_report.v4` carries the independently captured
process ID so an in-process witness cannot be transferred across ranks or
process lifetimes. Report v2/v3 remain unchanged for legacy observations.

Synthetic CPU fixtures and actual CPU-built table observations establish the
metadata and refusal contract only. They do not establish a CUDA footprint,
served quality, image acceptance, or compatibility with a pinned external
consumer. No pin, kernel, numerical path, encoded byte or serving gate changes.
