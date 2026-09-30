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
and its measured acceptance.
