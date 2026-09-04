# The contract names the launch, and the census it was attested from agrees

**Issue #111.** 2026-09-04. Branch `wf/ts-111`, base master `766033c`.
`lane_eligibility` schema **v4**, which lands as changelog entry **13**; the
branch ships `contract_version` **14** (entry 14 is the separate fix that the
window GEMV is loaded by *both* window routes, not only the FP8 one).

## What was wrong

A `lane_eligibility` cell said which A-side contract ran and which rungs a
receipt covered. The **launch** -- which GEMM, off which decoder -- appeared
only inside the cell's `id`. The E4M3 family published
`tessera_e4m3_k1_dense_sm121_{decode,batch}_scaled_mm_w8a8`, so the contract's
machine-readable answer to "what does this runtime execute in the decode
regime on an E4M3 wire" was the materialised FP8 pair, in every case, and no
cell named the window-GEMV lane at all.

That was accidentally true while the lane was unreachable (#104: every
allocated checkpoint we held carried a column rate outside
`kernel_window_gemv.SUPPORTED_RATES`, so every unit refused the lane at load).
It stopped being true on
`/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024` -- q256
1024, root rate 4 exactly -- whose served census records:

```
decode  : symbol tessera_window_gemv::gemv, decoder window_gemv, M1  -> 112 of 112
prefill : symbol torch._scaled_mm,          decoder window_gemv, M64 -> 112 of 112
```

(`/home/rob/tessera-runs/ts104/census-R1024-readable.json`, GB10, vLLM 0.28,
`TESSERA_SERVE_MODE=streamed`, eager, `problems: []`.)

Principle 14 from the other direction: not an unattested claim, but an
attested one gone stale under a checkpoint the runtime can finally serve.

Note what was **not** wrong. `activation_contract` is correct in both regimes
and does not move: `fp8_route.apply()` runs its per-token FP8 quantiser on
every path and hands the GEMV the dequantised values, so the executed A side
is `fp8_per_token_dynamic` whichever launch runs
(`docs/design/window-gemv-a-side.md` §5, decision (a)).

## The shape chosen, and why not the other one

#111 offered two shapes and picked neither.

**(1) A decode cell conditioned on the lane predicate at
`native_extensions[tessera_window_gemv].lane.requires`.** Rejected. A cell's
`predicates` are a closed `{fact, op, value}` grammar over structural facts
(`rate_q256`, `k`, `in_features`, ...) with the ops
`equals | in | multiple_of | at_least | at_most`. The lane predicate is
"`rate_set(root_from_q256(q)) ⊆ {1, 2, 4}` and `window_bits = 14`", which that
grammar cannot express: a reader would have to run
`grammar.bresenham_rate_schedule`'s rate arithmetic to resolve a cell -- the
very thing shape (2) exists to avoid, and something a producer reading the
table with no Tessera import cannot do.

**(2) Two cells with an explicit rung condition.** Adopted, and completed.
`rungs_q256` stays an enumerated list and a reader resolves no predicate. But
the rung alone cannot separate the two launches, because the one attested
E4M3 rung (1024) **is** lane-readable and what runs there is decided by the
**residency**: `fp8_route`/`bf16_route` both set `layer.tessera_gemv = None`
in `resident`, so the lane exists in `streamed` alone. The residency was
already a machine-readable cell field, spelled as an OR
(`TESSERA_SERVE_MODE=resident|streamed`) -- which read as formatting and was
really a claim that both residencies execute the same thing. So the E4M3
family splits into four cells, one per `(regime, residency)`, and the
condition needs no new vocabulary in any consumer.

## What landed

**`executes` on every cell** -- a list of `{symbol, decoder}` -- **derived,
not asserted.** `contract.validate_serving_contract` builds the expected set
from `scheme.ROUTE_LAUNCHES`, a new torch-free table of the launches each
route makes and the conditions it makes them under, narrowed by three axes
the cell already carries: the regime, the residency its `TESSERA_SERVE_MODE`
flag names, and the lanes each rung **reaches** under the predicate
`native_extensions[].lane.requires` publishes. Any disagreement is a refusal
at contract load.

`ROUTE_LAUNCHES` is not a second spelling of anything: `fp8_gemv.census_expected`
and `bf16_route.census_expected` are now derived from it, and it is the home
of `WINDOW_GEMV_SYMBOL` (previously a literal in both route modules). The
census tool's admissive pair sets are therefore the same table the contract
validates against.

The shipped table:

| cell | rungs | residency | executes |
|---|---|---|---|
| `tessera_e2m1_k2_dense_sm121_decode` | 896 | both | `torch._scaled_mm` / `native_span2` |
| `tessera_e2m1_k2_dense_sm121_batch` | 896 | both | `torch._scaled_mm` / `native_span2` |
| `tessera_e4m3_k1_dense_sm121_decode_resident` | 1024 | resident | `torch._scaled_mm` / `torch_window` |
| **`tessera_e4m3_k1_dense_sm121_decode_streamed`** | **1024** | **streamed** | **`tessera_window_gemv::gemv` / `window_gemv`** |
| `tessera_e4m3_k1_dense_sm121_batch_resident` | 1024 | resident | `torch._scaled_mm` / `torch_window` |
| `tessera_e4m3_k1_dense_sm121_batch_streamed` | 1024 | streamed | **both** `tessera_window_gemv::gemv` and `torch._scaled_mm`, / `window_gemv` |
| `tessera_bf16_k1_dense_sm121_decode` | 1792 | both | `torch.mm` / `torch_window` |
| `tessera_bf16_k1_dense_sm121_batch` | 1792 | both | `torch.mm` / `torch_window` |

Every one of those eight values is what the derivation returns; none is typed
into the JSON and believed. Note the `batch_streamed` row: the prefill decoder
is `window_gemv`, not `torch_window` -- with the lane prepared the tile comes
off the lane's own kernel decode -- so the old batch cell's `id` was stale on
the decoder half too, at the same rung, for the same reason.

### The same defect, one regime over

The first cut of this table gave `batch_streamed` the materialised launch
alone and conditioned a dead decode launch on a rate set. Both came from
reading one word two ways.

**The kernel's `decode` is `M <= GEMV_MAX_M`** -- eight token counts, which is
what `fp8_gemv.decode_is_gemv` decides. **This contract's `decode` is the
one-row forward**, and its `batch` is *every* M > 1: `CENSUS_PHASE_REGIMES`
says so in its own words ("a regime is a *problem shape* and the batch cell
covers every M > 1 forward, not only a first prefill"), and a census record is
stamped in that vocabulary. Run the dispatch over every M it distinguishes
(`range(1, GEMV_MAX_M + 2)`, both values of `rate_one`) and bucket by the
contract's regime:

| regime | what the dispatch launches, lane prepared |
|---|---|
| `decode` (M = 1) | `tessera_window_gemv::gemv`, always -- the rate-1 refusal starts at the 4-row tile, so no rate reaches this regime |
| `batch` (M > 1) | the `gemv` **and** the kernel-decoded tile under the stock GEMM: M = 2 is a GEMV, M > 8 is not |

So the batch cell publishes two launches, the decode cell one, and no cell
anywhere carries a rate condition. The version that shipped a single batch
launch was true of the 64-row prefill the census drives and false of the
runtime -- **#111's own failure, in the regime the issue did not name**, and
found only because the correction had to answer "which M does this cover".

`tests/test_serving_contract.py::test_the_launch_tables_regimes_are_the_routes_own_dispatch`
is the tie that closes it: it derives both `regimes` fields from the routes'
own `decode_is_gemv` and asserts exact equality per regime, so a launch the
dispatch cannot make fails as loudly as one it can. Put the old table back and
it reports `batch: dispatch=[gemv, _scaled_mm] table=[_scaled_mm] -> MISMATCH`.
The routes' `census_expected` moves with it: `batch` gains the GEMV pair (a
census driven at 4 rows would have been falsely refused) and `decode` loses
the kernel-decoded-tile pair (impossible at one row).

**Two more rules, both closing a way for the table to lie.**

* Two cells of one `(platform, family, structure, regime)` must cover
  **disjoint** residencies. Without it a consumer resolving "what runs here"
  among equal-status matches gets whichever cell it read first, which is a
  real hazard the split would otherwise introduce.
* A cell `id` is its **scope** (`family_structure_platform_regime[_mode...]`)
  and may never name a launch, because an id that names a launch is a second,
  unparsed spelling of `executes` -- the one that went stale.

**The drift test #111 asked for.** The chain is cell -> `lane.requires` ->
`kernel_window_gemv`. The second link was already tied
(`test_the_published_lane_predicate_is_the_kernels_own_constants`); the first
is new and is broken at both ends in `tests/test_lane_reachability.py`:

| mutation | refusal |
|---|---|
| `lane.requires.column_rates` loses rate 4 (what the kernel losing its 4-bit lane does) | `cells[3].executes is [('tessera_window_gemv::gemv','window_gemv')] but the TESSERA_FP8 route makes [('torch._scaled_mm','torch_window')] in the 'decode' regime at residency ['streamed'] on rung(s) [1024]` |
| `lane.requires.window_bits` becomes `[12]` | same refusal |
| the GEMV cell claims residency `resident` | same refusal, `residency ['resident']` |
| q256 1006 attested and added to the GEMV cell | `... the route makes [gemv, ('torch._scaled_mm','torch_window')] ...` |
| a second cell covering `decode`/`resident` | `... both cover (...) at residency 'resident'` |

## The receipt: the census this was attested from, replayed against it

Deriving `executes` proves the **document** agrees with the **code**. Only a
serve proves the code agrees with the machine, so
`census.cell_launch_agreement` joins every served route record to the cell
covering its `(platform, family, structure, regime, residency, rung)`, and
`tools/tessera_route_census.py` writes the block into the receipt and refuses
a disagreement.

Replayed offline over all 224 records of the R1024 census by
`experiments/ts111_replay_cell_agreement.py` -- no GPU, no re-serve, and the
receipt is on disk (`/home/rob/tessera-runs/ts111/replay-R1024.txt`):

Abridged from that file (the script prints the whole
`cell_launch_agreement` block as JSON):

```
receipt      : /home/rob/tessera-runs/ts104/census-R1024-readable.json
checkpoint   : /mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024
mode / eager : streamed / compiled = False
device       : NVIDIA GB10 (sm_121)
modules      : {'decode': 112, 'prefill': 112}

=== the shipped table ===
 decode : 112 modules, 112 covered, 0 unattested -> tessera_e4m3_k1_dense_sm121_decode_streamed: 112
 prefill: 112 modules, 112 covered, 0 unattested -> tessera_e4m3_k1_dense_sm121_batch_streamed : 112
 agrees : true      problems: []

=== the pre-#111 claim (decode = the materialised pair), same records ===
 agrees : False     problems: 112
 first  : decode: model.layers.0.mlp.down_proj executed
          ('tessera_window_gemv::gemv', 'window_gemv'), which cell
          'tessera_e4m3_k1_dense_sm121_decode_streamed' does not publish
          (it executes [('torch._scaled_mm', 'torch_window')]).
```

That second block is the issue as a check rather than an argument: put the
old claim back and the very records it was filed on refuse it, 112 times.
`tests/test_census_cell_agreement.py` pins both directions on two of those
records copied verbatim, so CI carries the same evidence without the file.

The check is **eager-only** and says so in the receipt: a compiled record
stamps both launches as one `a+b` pair because one graph serves every M, and
no cell publishes that form. A compiled census writes
`{"agrees": null, "skipped": ...}` rather than reading a traced record as a
disagreement.

## Scope: what this does and does not cover

**BF16 is covered by the mechanism and gains no cell.** `TESSERA_BF16_K1`'s
attested rung 1792 is root 7, outside `SUPPORTED_RATES`, so its own streamed
GEMV lane is unreachable there -- and the derivation returns `torch.mm` /
`torch_window` for it without being told to. That is the point: the BF16 gap
#111's scope note describes is closed by the same rule, and the day a
reachable BF16 rung is attested, the derivation produces its GEMV cell (and
refuses a hand-written one that disagrees).
`test_the_bf16_familys_own_attested_rung_cannot_reach_the_lane` still pins
the premise.

**Nothing about the wire, the encoder, quality or performance moved.** No
bytes, no rung, no route, no reader range, no activation contract. No GPU work
was run for this change: the only measurement it rests on is #104's census,
which already existed, replayed.

**PrismaQuant fails closed until it is widened, and this is not silent.** Its
`lane_eligibility` parser pins `tessera.lane-eligibility.v3` exactly
(`prismaquant/lane_eligibility.py`, `LANE_ELIGIBILITY_SCHEMAS`) and refuses
unknown cell keys (`_require_keys(..., optional={"requires_plugin"})`), and
`prismaquant/tessera_runtime_contract.py` pins the same string plus a
transcribed `TESSERA_DEV_PIN_ANSWER`. A v4 table therefore raises
`LaneEligibilityError` there rather than mis-reading. It also has to key its
candidate resolution on the serve flag (or union same-rank cells) before it
reads a table where one `(family, regime, rung)` has two cells. That work
belongs to PrismaQuant and is not attempted here; the dev pin is in any case
already stale against contract v12 (it transcribes one `native_extensions`
row and Tessera publishes two since #104).
