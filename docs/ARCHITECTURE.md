# Tessera plan-to-serve architecture

Allocation, export and serve for Tessera checkpoints: who proposes rungs,
who prices bytes, and what has to be served before an allocation ships.
Numbers below are citations, not claims -- each points at the measurement or
the code that owns it.

## 1. Scope

This doc covers the path from a PrismaQuant rung assignment to a served
Tessera checkpoint: `experiments/plan_from_layer_config.py` (assignment to
plan), `experiments/export_tessera_serving.py` (plan to checkpoint),
`tools/tessera_route_census.py` (checkpoint to route), and `tessera.control`
plus `experiments/uniform_control.py` (the gate that judges the result).
The wire itself is `docs/schema/prismaquant.tessera.v1.md`; the menu the
allocator sees is `docs/tessera-one-format.md` §5.

## 2. The pipeline

An allocator proposes one rung per Linear. The converter translates that
assignment into the exporter's `--plan-json`, refusing what one checkpoint
cannot serve (non-Tessera quantised choices, fused groups split across two
families) and stamping coverage and accounting into `<plan>.provenance.json`.
The exporter encodes what the plan names and the manifest states what is on
disk; the census checks every module serves on its declared family.

## 3. Bytes: priced == served

The sidecar's charged bits and the export manifest's `wire_bytes * 8` agree
per unit, checked by `experiments/check_wire_against_plan.py`. A plan that
leaves a body Linear unnamed does not get a passthrough: the exporter falls
back to its `--grid`/`--q256` default, so the converter names every unpriced
Linear `"BF16"` explicitly.

## 4. Allocation and the uniform gate

A candidate on Tessera's rate axis claims that *choosing* rungs beats
spending the same bytes at one rung. The sections below are that claim's
checks, in pipeline order.

### 4.1 The allocator proposes; nothing here re-prices quality

The converter carries the DP's rungs through member by member, including
per-member (mink) rates inside fused groups. No single group rate is derived
(`min` / average / `max` would be taste, not arithmetic). The converter
prices bytes only.

### 4.2 Unservable assignments are refused at plan time

A fused module whose members took two families has no single route to decode
it, so the converter refuses rather than writing rungs the exporter is about
to discard as BF16 (`--allow-fused-disagreement` writes the plan that will
serve and records the demotion).

### 4.3 Every plan carries its uniform control, unserved

The sidecar prices the one-rung plan that weighs what the candidate weighs
(`tessera.control.uniform_control`, issue #3). It records rather than
refuses, and it says plainly that neither arm was served: a built control is
not a passed gate.

### 4.4 The export writes only wires the pinned runtime decodes

`check_recipe` gates the default and every plan override against the
packaged `runtime_contract.json` before the first encode (issue #41).
Overridden refusals land verbatim in the manifest.

### 4.5 The census attests the route, not the quality

`tools/tessera_route_census.py` records, per residency mode, that every
module serves on its declared family. A clean census with exact bytes is
necessary and, by tessera#1, not sufficient.

### 4.6 The stock twin isolates the wire from the kernel

`--stock-twin` writes the same wires materialised for vanilla vLLM, so a
served comparison is one encode under two servings rather than two encodes.

### 4.7 The verdict is served KL against the byte-matched control

`experiments/uniform_control.py verify` asserts the match on the bytes that
shipped and, given both KLs, states whether the candidate beat its control.
`tessera.control.control_block` carries that verdict beside the bpp.

### 4.8 Dominated rungs are screened by bytes, proved by decode

The rate axis is not monotone in bits on small units
(`tessera.control.rate_menu`, issue #43), so the menu a selector is offered
is the screened frontier, and the pruning is recorded rather than silent.

### 4.9 Rung monotonicity in the L1 currency is measured false

The additive-Fisher L1 surrogate (`0.5 · h_trace · output_mse`) does not
rank Tessera rungs the way served KL does: on the seven units the
allocator priced it scored the allocation 0.889 (re-measured; 0.856 as
interpolated) where serving the same seven units reads 1.93x against it,
and the six moves above R1006 it scored a 1.30x net win serve as a 1.19x
loss (`docs/measurements/tessera-allocated-served-2026-09-02.md` §6, §7).
Any cost path that ranks rungs in that currency on the assumption that
quality rises monotonically with the rung must refuse or warn rather than
quietly rank.

### 4.10 REQUIRED: the continuous Tessera menu ships only validated-surrogate-selected

Tessera#1 is not a marginal mis-ranking: at 4.0 bpp the surrogate-selected
allocation serves 2.00x worse KL than the byte-matched uniform arm (2.33x
at 3.0, 2.88x at 5.0), and 95% of the whole-body gap in log terms sits on
the seven units the surrogate itself priced
(`docs/measurements/tessera-allocated-served-2026-09-02.md` §5, §7). The
bytes were exact to the unit, the census was clean, and the surrogate
scored the losing moves a win. A default that is measured to invert the
answer is not a default.

So the menu's recipe **requires** `SELECTION_MODE=validated-surrogate`:
a plan at more than one (grid, rung) embodies a rung selection, and it
ships only after the served byte-matched uniform-control gate passes
(§4.7). Surrogates generate, real KL selects; this menu is the case where
the generate step and the select step disagree badly enough to matter,
because its candidates differ by **rung** rather than by **format**.

`COST_MODE=aura` is not an accepted substitute until someone measures AURA
on a rung sweep: its KL-adjoint objective may or may not fix the
mispricing of the gain above R1006, and that must be measured, not
assumed. Until then `validated-surrogate` is the honest requirement.

In this tree the requirement is enforced where the allocation enters it:
`experiments/plan_from_layer_config.py` stamps every sidecar with the
`selection` block (`tessera.control.selection_requirement`, derived from
the plan's own distinct rungs, never from a roster) and warns on
stdout when a mixed-rung plan has no served verdict. A uniform plan
embodies no rung selection and has nothing for the gate to check.
