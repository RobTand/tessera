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

### 4.4a "The pinned runtime" is a digest, and a harness refuses without it

The pin is one string, `runtime_contract.json`'s
`versions.attested_on.image`, and it is a digest reference
(`vllm/vllm-openai@sha256:...`), not a tag: a tag is a name upstream can
repoint, so two boxes can hold two builds under it while every receipt
records the same four words (issue #100). `tessera.serving.runtime_image`
is the only reader; every wrapper in `experiments/` that starts a container
gates on it *before* taking the serve lock and refuses -- exit 2 plus a JSON
record naming the `docker pull` that fixes it -- rather than warning. The
check is membership in docker's `RepoDigests`, never `.Id`, which is the
manifest digest under the containerd snapshotter and the config digest under
overlay2: the same image reads two ids on the two GB10s. Both KL wrappers
stamp the resolved digest into the build sidecar's `identity`; the local id
rides in `provenance`, so a cross-box pair does not fingerprint itself apart.
Images outside the pinned repository (Mia's GLM image) are resolved and
stamped, not refused.

### 4.4b The export writes only where the runtime routes it

A wire is only worth writing on a Linear the runtime hands to this plugin, and
on GLM most attention is not one: `LinearBase.__init__` takes
`UnquantizedLinearMethod()` in the `quant_config is None` branch **without**
calling `get_quant_method`, so a projection built with `quant_config=None` is
invisible to every quantization plugin — it cannot refuse, warn, or see the
prefix (issue #99). `runtime_contract.json`'s `construction` block (contract
v11) publishes, per architecture, which Linear patterns the runtime offers a
quant config; the rows are generated from the census receipts under
`docs/measurements/construction/`, which `tools/tessera_construction_census.py`
observes by building the model the way the loader does with a probe quant
config. The export refuses a planned module that is not `offered`, names the
prefixes, and offers two escapes: `--passthrough-unrouted` (source precision,
the safe direction) and `--allow-unrouted` (write it anyway, stamped into the
manifest's `serving_gate`).

### 4.4c A LANE inside a route is gated too, and by the rung

`check_recipe` asks whether the *route* publishes a decode for these bytes.
That is not the same question as whether a named *lane inside* the route can
read them, and the second one had no producer side at all until issue #104.
The window GEMV (`kernel_window_gemv`) repacks each column's code stream at
that column's own rate and has a 16-row lane only where R bits per code is a
whole number of bytes, so it reads `SUPPORTED_RATES = (1, 2, 4)` and nothing
else. A rung is a *root* rate that `grammar.bresenham_rate_schedule` realises
by mixing the two rates bracketing it -- so q256 1006 (root 3.93) is columns
at rate 3 and columns at rate 4, and **every** unit of such a checkpoint
refuses the lane at load, module by module, through a substitution the route
reports as a served module. All six allocated checkpoints under
`/mnt/shared/tessera-runs/allocated` carried a rate outside the set, so no
artifact we held could exercise the lane at all.

The predicate is therefore published (`runtime_contract.json` v12,
`native_extensions[].lane.requires`, which *is*
`kernel_window_gemv.SUPPORTED_RATES` and `WINDOW_BITS_SUPPORTED` --
`tests/test_lane_reachability.py` ties the two ends) and read on both sides:

- **Plan time.** `experiments/export_tessera_serving.py --require-lane LANE`
  calls `scheme.refuse_unreachable_lane` at argument time, beside
  `check_recipe`, for the default rung and every plan override. It needs no
  shape -- reachability is a function of the rung alone (`grammar.rate_set`)
  -- so it refuses before a unit is encoded and names the offending rates.
  The flag is stamped into the manifest as `requires_lanes`.
- **Serve time.** `tools/tessera_route_census.py --require-lane` (or the
  artifact's own `requires_lanes`) makes an arm in which the lane took zero
  modules in every phase a REFUSAL, and writes a `lane_engagement` block
  (`tessera.serving.census`) whose `all_required_engaged` is `true`, `false`
  or `null` -- the third meaning nobody said what to require. A load-time
  lane refusal is a value on the layer (`telemetry.note_lane_refusal`), so
  the receipt says *why* the lane took nothing instead of leaving it on
  stderr.
- **After the fact.** `tools/tessera_lane_preflight.py` answers the same
  question from the bytes of a checkpoint somebody else built, over every
  unit, and exits non-zero.

The rate axis was **not** narrowed to fit the kernel: it is 2-D and
continuous by design, and pinning the format to three values so one lane can
be exercised would pay a permanent quality cost for a measurement
convenience. The kernel's constraint is the kernel's; the checkpoint comes to
it (`docs/measurements/tessera-gemv-lane-reachable-2026-09-03.md`). Scope:
`TESSERA_BF16_K1`'s attested rung is q256 1792 -- root 7 exactly -- so that
family's streamed GEMV lane is unreachable at its own attested rung for the
same reason, and no BF16 receipt covers it.

The lane belongs to **both** window routes, and the published table says so
since contract v14: `bf16_route.prepare_bf16_gemv` repacks through
`kernel_window_gemv.prepare_value_unit` exactly as `fp8_gemv` does and
branches its `apply` on the same `layer.tessera_gemv`, so
`native_extensions[tessera_window_gemv].routes` lists `TESSERA_FP8` and
`TESSERA_BF16`. It listed one until then, which said a BF16 serve is
unaffected by whether the `.so` mapped -- the exact claim a consumer keys a
serve fingerprint on.

### 4.5 The census attests the route, not the quality -- and engagement, not agreement

`tools/tessera_route_census.py` records, per residency mode, that every
module serves on its declared family. A clean census with exact bytes is
necessary and, by tessera#1, not sufficient. It is also not sufficient
*within* a route: the per-module check is a check on agreement, and the
streamed FP8 route's decode regime legitimately admits both the GEMV pair
and the materialised one, so a serve in which the lane prepared for nothing
passes it module by module (§4.4c). Hence `lane_engagement`: an arm that
requested a route and got zero units on it is a failed census, in a field a
gate reads. The verdict is over the census, not over each phase: a lane owns
a *regime*, not a serve -- the window GEMV decodes M <= 8
(`kernel_window_gemv.GEMV_MAX_M`), so the census's prefill forward takes the
torch decode by design, and only zero modules in *every* phase is the void
the field exists to catch. The per-phase counts stay in the block, and
`all_required_engaged` is three-valued so "nobody said what to require" never
reads as "everything required was engaged".

### 4.5b What the contract says a serve EXECUTES, and the join that checks it

A `lane_eligibility` cell says: on this platform, for this payload family, at
these rungs, in this regime, at this residency, the plugin executes **these
launches** on a route with this status. The launch half is
`executes` -- a list of `{symbol, decoder}` -- and it arrived with
`lane_eligibility` schema **v4** (contract v13, issue #111). Before it, a
cell published the A-side contract and the rungs, and the launch appeared
only inside the cell's `id`: the E4M3 family said `..._decode_scaled_mm_w8a8`
for both regimes and no cell named the window-GEMV lane at all. That was
accidentally true while the lane was unreachable (§4.4c) and false the moment
a rate-constrained artifact was served -- the R1024 census records
`tessera_window_gemv::gemv` on 112 of 112 modules in the decode regime
(`docs/measurements/tessera-lane-eligibility-executes-2026-09-04.md`).

Three things follow, and each is a rule rather than a value:

- **The value is derived, never asserted.** `contract.validate_serving_contract`
  builds each cell's `executes` from `scheme.ROUTE_LAUNCHES` -- the torch-free
  table the routes' own `fp8_gemv.census_expected` / `bf16_route.census_expected`
  are built from, and the home of `WINDOW_GEMV_SYMBOL` -- narrowed by the
  regime, by the residency the cell's `TESSERA_SERVE_MODE` flag names, and by
  the lanes each rung reaches under `native_extensions[].lane.requires`. So a
  cell naming the GEMV cannot outlive `kernel_window_gemv.SUPPORTED_RATES`:
  drop rate 4 from the published predicate and the document stops validating
  (`tests/test_lane_reachability.py`).
- **The residency is a condition, not a label.** Both window routes set
  `layer.tessera_gemv = None` in `resident`, so the lane exists in `streamed`
  alone and one rung's decode regime has two answers. The E4M3 family
  therefore carries four cells, and two cells of one `(platform, family,
  structure, regime)` must cover **disjoint** residencies -- otherwise a
  consumer resolving "what runs here" gets whichever cell it read first. A
  cell `id` is now its scope and never a launch, because an id that names a
  launch is a second, unparsed spelling of `executes`.
- **The census closes the loop.** Deriving `executes` proves the document
  agrees with the code; only a serve proves the code agrees with the machine.
  `census.cell_launch_agreement` joins every served route record to the cell
  covering its `(platform, family, structure, regime, residency, rung)` and
  refuses a disagreement, and `tools/tessera_route_census.py` writes the block
  into the receipt. It is eager-only and says so: a compiled record stamps
  both launches as one `a+b` pair because one graph serves every M.

`TESSERA_BF16_K1` gains `executes` and **no** GEMV cell -- its attested rung
1792 is root 7, outside `SUPPORTED_RATES`, so the derivation returns the torch
window decode under `torch.mm` without being told to. The day a reachable BF16
rung is attested the same derivation produces its GEMV cell.

Schema v4 is **not** additive: a v3 reader must not read a v4 cell, both
because `executes` is a key it does not know and because the E4M3 decode
answer it would have read off one cell is now two. PrismaQuant's
`lane_eligibility` parser pins `tessera.lane-eligibility.v3` exactly and
refuses unknown cell keys, so it fails closed (loudly, not silently) until it
is widened.

### 4.5a A served KL names which FORWARD it scored

`kl_tool.py dump` has two regimes and they are two metrics. `--regime
prefill` is the default and is every served number this repo quoted before
2026-09-03: it scores prompt positions, so each one comes out of one
many-row forward over the chunk. A lane that serves only small M --
`fp8_gemv` routes `M > GEMV_MAX_M` to the materialised path -- therefore
never executes on a scored forward, and a two-arm A/B over such a kernel
returns a *bit-identical* null in both arms, which reads exactly like a
strong positive result (tessera#83's served leg, tessera#102).

`--regime decode` feeds each chunk's prefix incrementally with the serve's
prefix cache on, so every scored position is an M = 1 forward; the claim is
checked against the serve's own `usage.prompt_tokens_details.cached_tokens`
(hence `--enable-prompt-tokens-details`) and refused when it does not hold.
The teacher must be re-dumped in the same regime, and `compare` refuses a
cross-regime pair outright -- there is no override, because the two regimes
run different kernels over different position sets.

`TESSERA_ROUTE_TRACE=<absolute path>` (off by default, eager only) makes the
serve write a launch histogram keyed by route **and problem shape**, which is
what lets a receipt show that its scored forwards were the shapes it claims:
the fallback arm reports `torch._scaled_mm` in both regimes, so the shape is
the discriminator and a symbol-only record cannot tell two lane states from
one wearing two names. `experiments/decode_regime_kl.sh` takes both regimes
off one serve with a trace snapshot per stage;
`docs/measurements/tessera-decode-regime-kl-2026-09-03.md` is the receipt. It
measures the gap: over byte-identical bytes with the GEMV lane on in one arm
and refused in the other, the prefill regime reads `KL >= 0.000000` at 100.00%
top-1 agreement while the decode regime reads `KL >= 0.012111` at 91.02%, and
the trace shows why -- 28 672 `tessera_window_gemv::gemv` launches on the
decode dump's scored forwards, zero on the prefill dump's.

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

### 4.11 REQUIRED: a per-plane promotion is won by units, not by the geomean

The LUT refit objective was promoted on a 1.38% six-unit geomean that won
on 2 of 6 units, while the served KL quoted for the pick measured the
other arm (tessera#65,
`docs/measurements/tessera-ldlq-lut-plane-served-2026-09-02.md`). So a
per-plane promotion now clears five legs in
`tessera.control.assert_plane_promotion`: the GLM six-expert gate exactly
as the 2026-09-02 receipt wrote it, a geomean that beats the incumbent, a
strict majority of the receipt's own units, a served KL on the promoted
arm that beats **the incumbent's own served KL at matched bytes**, and the
`landing` the per-unit ratios were taken at (below). The
geomean is derived from the per-unit ratios, so it cannot arrive without
them, and a served number for a different arm is not evidence. `served_bar`
takes no default for the same reason the other three legs are ratios: it is
the arm being replaced, so it moves whenever a promotion lands. (The
receipt's 0.640 is the *stock* wire and was the incumbent only for "levers
vs no levers"; as a default it would have passed a candidate serving 0.60
over an `h^1.0` incumbent at 0.5310.)

No default moves by this, and `tests/test_plane_promotion.py` is what makes
that checkable rather than asserted: it runs the receipt's own six-unit
record through the gate, watches `hessian` refuse at 2 of 6, and pins
`DEFAULT_REFIT_OBJECTIVE["lut16"]` to the `h^1.0` that refusal leaves
standing. Flipping that default without a promotion this gate accepts turns
the suite red.

#### The fifth leg: a screen taken off the wire does not promote

On the LUT plane a per-block scale lands on one of sixteen E4M3 entries, and
`tessera.encode.lut_landing` can remove that landing to read issue #50's
ceiling. **The arms reorder when it does.** Six dense Qwen3-0.6B units,
E2M1x2 `q256=896`, LDLQ 1.0/32, held-out `out` geomeans (tessera#85): on the
wire Gauss-Seidel 0.9627 beats Jacobi 0.9864 beats `h^1.0` 1.0000; with the
landing removed Jacobi 0.7057 beats Gauss-Seidel 0.7274 beats `h^1.0`
0.7843. So every on-wire arm score on this plane is a **joint** measurement
of the refit and the table fit, and the receipts reported it as a property
of the refit.

Two consequences, and only one of them is a refusal.

* **Refused.** `assert_plane_promotion` takes `landing`, defaulting to
  `tessera.encode.LUT_LANDING_WIRE` -- the state every encode runs in -- and
  refuses anything else by name. The landing-disabled column holds the most
  attractive numbers ever measured on this plane -- Jacobi at 0.7057 against
  the on-wire default -- and a six-unit record at that level with a unit
  majority clears all four of the older legs; the gate had no way to ask what
  its ratios were ratios *of*. (#85 publishes geomeans, not per-unit
  `landing=none` ratios, so `tests/test_landing_ordering.py` demonstrates that
  with a synthetic record at that level, geomean 0.708, and says so.) The
  claim is
  caller-asserted exactly as `served_arm` is, and knowable for the same
  reason: non-wire ratios exist only inside a `lut_landing` context, whose
  sink already reports `serialisable=False`.
* **Recorded, not refused.** `tessera.control.landing_ordering` puts the two
  orderings side by side and derives `same_best`, `same_order` and the
  inverted arm pairs as values (`tessera.landing_ordering.v1`), with no
  tolerance -- a "disagree by more than x%" would be a threshold from
  intuition. A disagreement does **not** block a promotion: what ships is the
  landed wire, so the on-wire ordering is the correct measurement of the
  shipped object rather than a confound in it. "Gauss-Seidel plus this
  landing beats Jacobi plus this landing" is true and is the sentence a
  default selection needs; what #85 corrects is the attribution, and an
  attribution error is fixed by reporting the pair. Refusing on it would also
  pin one measurement -- one wire, one `(sigma, block)`, six weight-space Qwen
  units, no serve -- as a standing rule about the plane, which is the
  roster-not-rule failure AGENTS.md rule 3 names. The disagreement is a
  **re-run trigger** for the day a better landing lands (issue #50).

The pair is not free and is not readable off `refit_diagnostics`. That
instrument's `continuous` leg is a within-call quantity by its own contract --
for a 1-D metric it records the separable parabola, equal to the weighted
error only up to a constant -- and the arms being ranked are 1-D (`h^1.0`)
against full-H (Jacobi, Gauss-Seidel). The diagnostics give the *size* of the
landing leg within one arm; the ordering across arms costs one extra
`lut_landing("none")` encode per arm (`experiments/lut_landing_ceiling.py`,
no serve and no KL). `tests/test_landing_ordering.py` pins both halves.
