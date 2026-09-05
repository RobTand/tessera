# Tessera plan-to-serve architecture

Allocation, export and serve for Tessera checkpoints: who proposes rungs,
who prices bytes, and what has to be served before an allocation ships.
Numbers below are citations, not claims -- each points at the measurement or
the code that owns it.

**Provenance:** current as of the unreleased `v0.1.0` candidate (2026-09-05):
base `3317036` (wire minor 7), encoder-evidence scope correction #198; CI at `df1bc20`,
packaging metadata at `cd3190a`; contract v19, lane-eligibility schema v8. Re-stamp this
line with any change to the wire, the recipe table, the serving lane, the
plugin contract or a gate (AGENTS.md principle 10).

## 1. Scope

This doc covers the path from a PrismaQuant rung assignment to a served
Tessera checkpoint: `experiments/plan_from_layer_config.py` (assignment to
plan), `experiments/export_tessera_serving.py` (plan to checkpoint),
`tools/tessera_route_census.py` (checkpoint to route), and `tessera.control`
plus `experiments/uniform_control.py` (the gate that judges the result).
The wire itself is `docs/schema/prismaquant.tessera.v1.md`; the menu the
allocator sees is `docs/tessera-one-format.md` §5.

### 1.1 Integration suite placement

`tools/merge_suite.py` dispatches its device populations only through
PrismaBuild. Live GPU submissions require an explicit `--gpu-tag` and pass
`--exclusive`: the deployed scheduler derives the complete GPU reservation
from that worker's advertised capacity, rather than treating one logical slot
as physical exclusion. The GPU arm remains serial under `--strict-cuda`,
which has three legs since tessera#152: it refuses a device-less session
before anything runs, refuses at the end a run in which no test allocated on
the device (torch's own allocator counter, published as
`cuda_surface.executed`), and refuses a run that skipped because this box
holds no checkpoint or serve log a gate needs -- `tests/box_artifacts.py`
resolves those roots and names each one's environment variable, so a GPU box
without them cannot claim the surface;
the x86 arm spends its declared `--cpus N` as pytest `-n N` (serial at one).
Each pytest process explicitly receives `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MAX_JOBS=1`, overriding pool
defaults and preventing each xdist worker's native math or extension compiler
from multiplying its one-CPU share. The per-process limits are recorded in
each arm's receipt; these environment settings are not an OS-level CPU quota.
The sealed inner command also uses `tessera._dev.suite_deadline`, launched through
`tools/suite_deadline.py` with the arm's named Python. `--timeout-s` must be
positive and finite; expiry signals its owned process group with TERM, then
KILL after a five-second grace. The leader remains unreaped during the grace,
so its PID cannot be reused and a resistant child is killed even if the leader
exits on TERM. The supervisor explicitly owns/restores SIGCHLD disposition;
inherited auto-reaping cannot erase child failure status or that PID anchor.
Every wait after KILL is bounded by that same grace: a leader that cannot be
reaped (a D-state process on a wedged GPU) is reported on stderr and the
attempt exits 137, so the deadline that bounds the run is itself bounded.
Normal command status passes through; expiry remains non-green
even if a TERM handler exits zero. Supervisor TERM/INT kills and reaps its
owned group before returning a nonzero status. This is a per-attempt deadline,
not a bound on queueing/retries, detached sessions/containers, or descendants
left after a command completes before its deadline. Host `timeout` binaries
are not trusted as interchangeable: the dl380g10 uutils 0.8.0 probe returned
137 while leaving a same-group child in state S; both Sparks had GNU 9.4.
The deployed pbrun parses but does not apply
its own `--timeout-s`; its worker may report outer status 1 for any nonzero
inner result, so a receipt must not claim it observed numeric 124/137 merely
from that outer status. The command and deadline/grace are retained per arm.

Each population retains the actual Git snapshot commit and separately records
`tessera.suite_source.v1`: SHA-256 over every tracked source path, executable
mode, and actual file/symlink bytes, checked against the snapshot blobs. Only
the exact generated closure member verified against that action's sealed CAS
request is omitted. The action-prefix directory is a bounded lookup hint,
not proof: the full action key, snapshot commit, container owner, closure
hash/size, logical path and generated filename fingerprint must all agree.
Other closure-looking tracked files remain source. Original-head and dirty
stamps are never substituted for the actual source hash. Post-materialization
dirty state, ambiguous/missing requests or failed verification yield `unknown`.
That hash is of a **span**, not of an instant: `tests/conftest.py` captures the
identity above its first import of the code under test and the publication is
bound to it, so a checkout fast-forwarded cleanly mid-run publishes `unknown`
rather than attesting a tree nothing tested. Under `-n`, each worker reports
its own entry-bound identity through xdist's `workeroutput` and the
controller's population is `unknown` unless every executing process agrees with
it -- the controller runs no tests, so its own hash describes its filesystem
until they do. The `tessera.suite_source.v1` receipt string is unchanged: the
span and worker fields are additive, and `verified` became harder to earn, never
easier.
The merge receipt keeps raw commit agreement and effective-source agreement
separate; legacy populations cannot establish the latter. Population pass
counts alone still do not establish a same-source merge check.
The merge **verdict** now reads those fields rather than reporting them beside
an unrelated conclusion: a green exit requires, per arm, a recognised surface
schema, the `population` role, readable counts with something passed, and --
for an arm submitted under `--strict-cuda` -- a device, an armed gate, a
`tessera.test_surface.v3` population, a positive `cuda_surface.executed` and no
`box_artifact_skips`; and, across arms, `commits_measured.effective_source.agree
== true`. Different verified source is refused, unknown provenance is
incomplete, and a pre-v3 population cannot answer the execution question, so it
is not green. Each arm's own result stays readable in the receipt's
`arm_results`, which is deliberately not the merge verdict and never sets the
exit status.
`tools/impacted_tests.py` reuses this verified exclusion: a closure-shaped
tracked file is not ignored by name, and unverifiable metadata forces a full
selection. Verified PB metadata still permits narrowed selection.
Both normal and parentless diffs use Git's NUL-delimited path protocol, so
display quoting cannot conceal metadata under tab/newline-containing paths.
A path named in `OPAQUE` -- `docs/schema/`, `pyproject.toml` -- forces the full
suite whatever its extension: the wire spec is a Markdown file, and filtering
inert suffixes first meant the rule that names it could never see it.
An import of `pkg.mod` is an edge to `pkg.mod` **and** to every package
`__init__` above it, because importing a submodule executes them; a file loaded
by explicit path gets an edge to that file only, because loading by path does
not. A relative import in a package's own `__init__.py` climbs from that
package, not from its parent, so `src/tessera/__init__.py`'s `from .manifest
import SCHEMA_ID` is the `tessera.manifest -> tessera` edge it looks like. Each
file is also a node under every name an import root on `sys.path` gives it:
`tests/` holds no `__init__.py`, so `tests/box_artifacts.py` answers to
`box_artifacts`, which is how every test in the tree spells it. An ambiguous
alias edges to every candidate.
Explicit Python file loaders also contribute dependency edges: the non-executing
`tessera._dev.source_dependencies` resolver follows finite `Path` expressions,
lexical bindings, loader aliases and repository globs, using the file path rather
than the loader's arbitrary module label. A resolved target inside the tree is
an exact edge whatever its suffix -- a non-Python file is a node under its own
repository-relative path -- and a resolved target outside the tree is neither an
edge nor an unknown. An unresolved recognized loader conservatively seeds its
importing module and downstream tests for every non-inert change; an unresolved
loader reaching a conftest forces the full population. An unresolved *read* is
an unknown module only for a module that can parse or execute Python source
(`_SOURCE_BUILTINS`/`_SOURCE_ATTRIBUTES`/`_SOURCE_QUALIFIED`, matched by
resolved symbol so `re.compile` and `model.eval()` are not it): bytes are a
Python dependency once something runs them, and treating every unnameable read
as "any module in the tree" is what held the verdict at `full` for every change
(#148). A conftest that execs the `test_*.py` files below it is probing its
collection targets, and that edge is excluded from the walk that forces full --
as a dependency it closes a cycle that makes one uncertain test file uncertain
for the whole population. The selector reports those unresolved importers in its
receipt.
One-directory conftest globs resolve to ordinary edges; recursive, escaping or
otherwise unresolved path expressions retain the conservative fallback.
Parameter, return and annotated-assignment expressions retain potential loader
dependencies even when annotation evaluation is deferred. Function annotations
use the defining scope (including a method's class scope), not value-parameter
locals; generic type-parameter names remain unknown in a separate annotation
scope instead of borrowing an outer file path.
Explicit `Path.read_text`/`read_bytes`/`open` and builtin/`io.open` source reads
also create edges, including aliased readers, and for a resolved path this is
independent of what the reader does with the bytes. Runtime-selected or shadowed
paths remain conservative unknown edges and propagate to downstream tests when
the reader can execute Python; a parameterized filename in a source-executing
module is not silently treated as no dependency. What this misses is a source
read the resolver never sees -- `subprocess.run([sys.executable, path])` above
all -- which was never an edge here.
A conftest **reached** -- changed, or importing anything changed -- reaches its
entire test population, because pytest imports it for every test at or below its
directory and a fixture consumer names the fixture rather than the module the
conftest built it from; collection-probe edges are excluded from that walk, so
one changed test file does not select the population through the conftest that
execs it. A delegated runner-fix task records
its targeted regression evidence while the coordinator owns the final full
dual-population integration run.

## 2. The pipeline

An allocator proposes one rung per Linear. The converter translates that
assignment into the exporter's `--plan-json`, refusing what one checkpoint
cannot serve (non-Tessera quantised choices, fused groups split across two
families) and stamping coverage and accounting into `<plan>.provenance.json`.
The exporter encodes what the plan names and the manifest states what is on
disk; the census checks every module serves on its declared family.
Construction and route censuses share one runtime-mapper adapter: the current
`get_rename_mapper` name-only view takes precedence over the earlier
`get_unstacked_mapper`; a directly exposed mapper is replayed as-is. An
existing wrapper method that fails is not silently ignored.

The shared producer fusion rule names LFM dense `feed_forward.w1/w3` as the
constructed `feed_forward.w13`, for both quantized targets and explicit BF16
passthroughs. Routed `feed_forward.experts.N.w1/w3` remain projection leaves
owned by the MoE stack; no dense alias applies to them. This naming comes from
the pinned LFM construction receipt, not a fallback in the serving plugin.
`export_tessera_serving.fused_module` is the one statement of that roster --
q/k/v, every non-routed gate/up including `mlp.shared_experts`, and `w13` --
and the converter's `fused_key` delegates to it rather than restating two of
its rows (tessera#211), so the plan-time fused check and the export-time one
read one rule.

One A-side value binds the same roster. A W4A4 module's
`trellis_input_global_scale` is the **minimum** of its members' donated
`input_global_scale`s (`tessera.fused.shared_input_global_scale`, the join's
one home): the route hands the value unmodified to vLLM's native quantiser,
which stores each group-16 block scale as `e4m3(block_amax/6 x scale)` clamped
at 448, so the value is capacity/amax -- inverse in the activation range --
and the min member scale is the largest calibrated amax, the range the fused
GEMM's one input tensor actually spans. A max-join picks the smallest
calibrated range and silently clips every wider member's peak activations;
the exporter took the max until RobTand/prismaquant#196 flagged the
divergence against PrismaQuant's `unify_fused_sibling_input_global_scales`
(min-scale = max-amax, the same rule at calibration time). The stock twin
carries the joined value on every member for the same reason: vLLM reduces
whatever the members carry into one scale per fused module -- warning, not
refusing, when they differ -- and the twin exists to execute the A side this
export serves. The join accepts only members that agree to within one bf16
ULP (`fused.FUSED_INPUT_SCALE_ULP` = `torch.finfo(torch.bfloat16).eps` =
2^-7): the route casts every A tensor to bf16 before the quantiser sees it,
so a calibrated amax is an observation of a bf16 tensor and scales from ONE
calibration land within one step of that lattice. A wider spread is two
calibrations -- mixed draws, mixed policies, or a group never calibrated
jointly -- and is refused where the bytes are decided rather than joined
into a distribution nobody measured; the fix is a joint recalibration (one
amax over the members' shared input, which is what both repos' calibrators
already emit), not a wider bound.

### 2.1 Whole-layer export parts have one checked assembly

`export_tessera_serving.py --partition INDEX/COUNT` gives a complete decoder
layer to `layer % COUNT`; non-body tensors belong to index zero. Every worker
validates the same full plan before selecting its work, so a fused module and
an expert stack cannot be divided between workers. Each worker reads and writes
only its owned source tensors. A part has `tessera_part_config.json`, never a
loadable `config.json`; it is not a checkpoint until assembly.

`merge_tessera_parts.py` recognizes these serving parts separately from the
older shard-split wire exports. Before creating its output it requires every
partition exactly once, exact source-tensor ownership and coverage, identical
source/config/tokenizer hashes, encoder source and behavior fixture hashes, full plan/options and
dispatch-pinned runtime digest, and an index matching the hashed output files.
It copies the containers unchanged under unique shard names, unions the schemes
and ignores, derives totals from the combined module records, and writes the
final `config.json` last. Each assembled weight shard gains owner/group/other
read bits so a root-squashed serving container can read it; no write or execute
bits are added. Default copy assembly leaves the private source-part modes and
all payload bytes unchanged. The runtime digest here names what the dispatch was
asked to run; PrismaBuild's campaign receipt supplies the execution evidence.

Source coverage alone does not prove that a plan was fulfilled: an omitted
expert stack can remain internally consistent BF16. Before publication the
merger therefore reads the sealed `identity.options.plan` and requires every
explicit quantized target to have its emitted roles at the requested grid and
rung. Planned expert-stack names must equal both manifest and config stack
sets, including each stack's complete expert/projection population and source
projection coverage. Implicit dense defaults have no complete plan roster and
are outside this additional check. `ts5_sidecar_check.py` repeats the same
validator before serving, using the merged `export_identity.options.plan` or
an explicit `--plan-json` that must agree with it; its routed-MoE summary must
name the same population too. Existing version-one parts remain readable and
their containers do not change.

The direct (non-partitioned) export runs the same validator against its own
emitted roles and declared schemes before it writes `config.json`, and every
path that used to demote an explicit quantized target to BF16 passthrough in
silence now refuses before the first encode, naming the tensor and the field:
a shape the grid cannot cut, a `--layers` smoke bound that excludes a planned
tensor, a fused group whose explicitly planned members cannot share one
scheme, and `--passthrough-unrouted` reaching a module the plan names.
Implicit `--grid`/`--q256` defaults keep their deliberate passthrough
fallbacks, and an explicit `PASSTHROUGH`/`BF16` entry is still a passthrough
(tessera#211).

## 3. Bytes: priced == served

Every artifact the exporter writes has exactly one legal length: the encoder
declares one terminal per unit, at the depth it used. Since schema minor 7
(2026-09-05, tessera#144) the wire *can* carry a shorter rung on that encode:
the D5 plane order puts COMPLETION behind every plane a decode needs and
ahead of RELEASE only, and COMPLETION is cut by depth level, so a shallower
completion rung is a byte prefix (`docs/schema/prismaquant.tessera.v1.md`
§1h; `tests/test_audit_container_accounting.py` lays one on the exporter's
bytes and reads it back), and the reader reads every plane at the terminal's
count -- a shorter S6b refinement, completion depth or release plane means
what schema §3c item 3 says, every other plane is whole or refused by name
(`unit_artifact._refuse_partial_planes`). Whether the exporter writes a ladder is a separate
decision, and on today's recipe table (every default rung at `completion=0`)
there is no rung to shorten: no release note may claim truncatable artifacts,
and nothing has measured that truncation is worth bytes anywhere
(`docs/reports/tessera-terminal-ladder-2026-09-04.md`). Every artifact this
tree writes is minor 7; minors 0–6 read unchanged (`tests/test_ladder_wire.py`
holds the reader to eleven pre-change artifacts under `tests/data/legacy/`),
and the encoder identity moved with the terminal records.

The sidecar's charged bits and the export manifest's `wire_bytes * 8` agree
per unit, checked by `experiments/check_wire_against_plan.py`. A plan that
leaves a body Linear unnamed does not get a passthrough: the exporter falls
back to its `--grid`/`--q256` default, so the converter names every unpriced
Linear `"BF16"` explicitly.

### 3.1 Which encoder cut the bytes is on the artifact

Three identities travel with a Tessera checkpoint and they answer different
questions. `encoder_profile_id` binds the **arguments** a reader must
reproduce — code, grid, span, body, plane, window width, reach spellings — and
is input-only by decision: it "contains nothing an encode alone can produce".
`CONTAINER_VERSION` versions the **container** around the bytes. Neither can
see an **encoder** change: same arguments, different bytes out. That gap
merged two differently-encoded halves once already (issue #78), and closing it
is issue #101.

The third identity is `tessera.encoder_identity.encoder_fixture_id`, and it is
**derived from behaviour, not declared**: a fixed, tiny fixture set is encoded
at fixed arguments and the result is hashed, so the value moves exactly when
the encoder's output moves and never when a comment or a refactor does. It is
a sibling of the profile id and never an input to it. It rides in the manifest
from schema minor 6 (minor 7 carries the same field) and in
`tessera_config.json`; `merge_tessera_parts.py`
compares the stamped value across parts. `encoder_identity.resumable` states
the rule for whether a cached unit may be reused. The explicit expert-cache
intake below calls it and pays the memoized CPU fixture once; merge guards
continue to compare recorded identities without computing it. The untagged spelling —
the encoder the field was born against — writes no field and no minor, so
every artifact already on disk is byte-identical across the bump. The wire is
`docs/schema/prismaquant.tessera.v1.md` §1g, which also states what the fixture
set is blind to.

Issue #87 is the first post-identity encoder move: a CHANNEL row raised to the
body's reach now lands that lower bound upward instead of rounding below it by
one fp16 ulp. The behaviour-derived identity moves as a consequence, so a
default artifact written by this encoder carries `encoder_fixture_id` and uses
the already-defined minor-6 envelope (minor 7 since tessera#144, which
carries the same field); minor 6 is the identity-bearing
container, not a new reach-floor field or a wider reach schema. An untagged
artifact names the pre-#87 encoder and `encoder_identity.resumable` refuses it
under the new encoder. Tests that assert an older record minor explicitly ask
`build_unit_artifact` for that born-against spelling, while a byte-for-byte
rebuild inherits the parsed artifact's identity.

Issue #116 adds the exact half-ulp boundary that #87 deliberately left to
issue #115: an unraised E4M3 row whose RMS scale is inside the body's reach in
float arithmetic while its nearest fp16 word is one word below that bound.
Because the identity was already live, adding that witness may not re-label
unchanged arm-A artifacts. Its encoded arm-A contribution therefore has a
measured historical digest and is neutral while it matches; if the encoded
contribution differs, its self-delimiting bytes join the identity hash. The
historical digest is never bumped. This is content identity rather than a
monotonic version: rolling the encoder back to the exact baseline makes the
witness neutral again, and the full identity rolls back only when the other
fixture outputs do too. New shipping structures still add ordinary fixtures
and re-base the identity; baseline-neutral witnesses are only for newly found
blind spots inside a structure the live identity already claimed to cover.

### 3.1a A served receipt names the encoder that wrote its artifact

Runtime evidence remains evidence about the bytes actually served. It does
not establish that a later encoder writes those bytes, even with the same
recipe and source. Schema v8 makes this distinction readable in
`evidence.artifact`: null means no encoder-reproduction comparison is
recorded; a record names the historical artifact and full encoder commit,
then the comparison commit, single unit, payload relation, weight-SSE
relation and measurement receipt. The validator checks the closed record
and returns it through `cell_evidence`. The comparison is permanently scoped
to that commit and unit, never implicitly to the latest tag.

The four dense E4M3 cells name
`gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook`, built at `8070ec6`.
The measurement and reproduction command are in
`docs/measurements/encoder-evidence-scope-2026-09-05.md`. Other cells carry
null because this comparison did not measure their artifacts. Weight SSE
is a screen: it never changes the evidence grade, supplies a served KL, or
proves that historical KL is a bound on a fresh encode. The existing KL
entries retain their historical artifact scope. This is a pre-release
contract correction; no released-schema migration is provided.

### 3.2 Exact campaign unit intake (explicit, not a serving qualification)

`experiments/tessera_producer_plan.py` reads source headers and an explicit
stack plan, then calls the exporter's existing expert planners. Its JSON names
the physical source tensor, logical tensor, source slice, expert, canonical
role and both unit/group dimensions. PrismaQuant can invoke this producer tool
without importing the serving runtime or restating its expert grammar. Source
tensor names retain their actual suffix; logical tensors include `.weight`,
and allocation/cache keys remove exactly that suffix via `ActivationSource`.

`tessera.cached_unit` seals original dtype/shape/weight bytes, the actual
per-unit Hessian plus capture identity and full activation settings, resolved
recipe, encoder behavior/source identities, and the whole blob digest.
Its `encoding_input_identity` is shared by dense and projected campaign
callers; `unit_input_identity` adds the producer's explicit expert projection.
Both use the same unit-record construction and wire verifier, while the
export boundary still requires the projected identity and exact field equality.
`export_tessera_serving.py --cached-expert-units MANIFEST` requires exact
coverage of the planned experts and the full source checkpoint seal. It
checks those receipts against the actual source slices and capture, validates
wire geometry/rates/profile/reach/encoder identity and complete plane extents,
and wraps accepted blobs in
`pack_fused` unchanged. Their original unit-id spelling is preserved. A
missing selected rung, including an interpolated rate with no measured blob,
refuses; this intake has no encode fallback. The ordinary dense encode path
and all defaults remain unchanged. The cache mode requires a fresh output
directory and records each accepted blob's SHA in the export manifest.

These are producer evidence and tests only. They do not promote a recipe,
open an eligibility cell, or replace the source-matched served measurements
required for the PrismaQuant campaign bridge.

### 3.3 The native decode is held to the reference at load

The NVFP4 route was the one route whose decoder reached generation
unchallenged: the FP8 and BF16 routes have always decoded once at load and
refused on inequality with `tessera.decode.materialize_*`, while the span-2
route prepared the module and returned (tessera#130).
`ops.prepare_tessera_module` now decodes the module once through the same
native op the forward runs, and holds both uint8 planes to what
`tessera.stock.materialize_stock` writes for the same roles on the moved LUT
tables and the shared global (`_torch_fallback_tile`, the same reference the
resident fallback substitutes) -- `torch.equal`, no tolerance, because the
reference is bit-exact. A difference refuses the module at load naming the
vLLM prefix, the role, how many bytes of how many differ, and the first
differing tile (row, role row, group-16 column block), so a refusal says
where the decoder went wrong rather than only that it did
(`_require_reference_agreement`, `src/tessera/serving/ops.py`). It runs in
both residencies and takes no operator knob, because the other two routes
take none. It is vacuous on the resident fallback, where the substitute IS
the reference and there is nothing independent to hold it to. What it costs
per module at load is not measured.

### 3.4 Declared weight transforms are refused at the materialisation boundary

Segment-2a diagonals and the branch rotation are transforms the encoder
prices and the wire carries; `reconstruct_unit` undoes them *after* the
codes-times-scale product. Every materialiser that stops at that product used
to drop them silently -- a rotated E4M3 window wire passed the whole FP8
preparation gate (both decoders dropped the same transform and agreed), and
the fitted-diagonal NVFP4 fixture's stock dequant was 3.29 max abs off the
reconstruction (tessera#233). `decode.require_untransformed` is now the one
home for the refusal, called by `materialize_fp8`, `materialize_bf16` and
`stock.materialize_stock`, so every consumer that decodes through them --
the dense FP8/BF16 routes' load-time reference decodes, the routed-MoE
expert decode, the NVFP4 route's stock reference and torch fallback, the
stock twin exporter -- refuses a transformed unit at load, naming the field
(`unit.diagonals` / `unit.rotation`). The kernel lanes' packers keep their
own refusals of the same fields (`lane_planes`, `kernel_window`,
`kernel_window_gemv`). Untransformed wires -- every shipping export default
-- are byte-for-byte unaffected; `tests/test_transform_refusals.py` drives
the refusal through each consumer on real wire bytes.

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

The bound is the *structure's* (issue #135). A dense module is held to the
format row's `reader_rate_range_q256`, which is the route's own decoder. A
routed-MoE stack is served by `moe_route` into a fused-MoE kernel the
contract attests as its own structure, so `scheme.refuse_unserveable_wire`
takes `structure` and, for `routed_moe`, first refuses a route
`MOE_BUILDERS` has no builder for (`refuse_a_family_with_no_expert_route`,
the one home for that rule at plan, gate and load) and then reads the
union of `rungs_q256` over the `lane_eligibility` cells of that structure
(`scheme.attested_cells`) rather than the row's range. The exporter reads
the source's shapes before the plan so it knows which plan entries are
stacks, refuses a stack at a rung only the dense route reads by the cells'
ids, stamps `structure` on every `serving_gate` override, and writes
`attested_by` -- the routed_moe cell ids whose rungs hold the stack's rung
-- on the stack's manifest record. Until then a routed stack at any E4M3
rung in `[256, 2048]` passed the gate and reached the encode while the
routed cells attest exactly q256 1024.

### 4.4a "The pinned runtime" is a digest, and a harness refuses without it

The pin is one string, `runtime_contract.json`'s
`versions.default_serve_image` (schema v6, #131; `versions.attested_on.image`
before it), and it is a digest reference
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
An explicit digest reference on any repository must be present in that
image's `RepoDigests`, and its requested digest is the stamp even when the
local image has other aliases. Missing or mismatched explicit images refuse
before a census or serve (#126). Floating tags outside the default pinned
repository remain resolved and stamped without being compared to that
unrelated pin; they cannot supply an exact-runtime census context. Scoped
lane images do not change the existing dense default. The record also names
the reference it resolved to, and `experiments/tessera_plugin_run.sh` exports
that reference and the record itself into the container it starts, under the
two variable names the module owns -- so a tag on the command line becomes a
digest inside the run rather than travelling as a name that identifies nothing.

`experiments/serve_lock.sh` is the one lock protocol for every serve and every
GPU-only probe.  Acquisition publishes one symlink at the host-local
`serve.lock` pathname; its target binds PID, `/proc` start ticks and a random
nonce, so publication has no directory-to-owner gap and PID reuse is not
ownership.  All new-protocol observe/reap/publish transitions run under a
host-local `flock` guard, so two dead-owner reapers cannot unlink each other's
replacement token.  Release first matches that exact target.  A dead token is
reaped only after the PID/start pair no longer names its publisher and Docker
reports no running container.  During the rolling transition, the same
pathname also excludes legacy directory-lock clients; old directories retain
their stricter hour-old, dead-owner, no-container recovery rule. Publication
treats the pathname as the exact destination (`ln -T`), so an existing legacy
directory cannot turn acquisition into successful creation of a link inside it.
Legacy owner liveness is `/proc/<pid>` existence: denial of permission to signal
a process is never treated as evidence that it has died.

`serve_and_dump_kl.sh` reaps its named container on every exit, including an
unexpected shell failure. Successful removal is remembered so normal exit
preserves the collected log. The wrapper releases its serve lock only after
removal succeeds; failed cleanup retains ownership and refuses certification.

`experiments/serve_metrics.sh` owns the speculative-decoding refusal for every
wrapper that dumps logprobs (`serve_and_dump_kl.sh`, `decode_regime_kl.sh`,
`tessera_plugin_served.sh`): spec-decode makes `/v1/completions` return the
draft model's numbers, so a serve publishing `vllm:spec_decode` may not be
measured. The check fetches the **complete** `/metrics` response to a file and
greps the file — never `curl | grep -q`, which cannot detect the condition it
owns, because `grep -q` exits at the first match, curl then fails its write,
and under `set -o pipefail` the `if` reads false and the wrapper dumps anyway
(tessera#247, the same early-consumer-exit already fixed in the startup-log
gate). A transport failure, a non-200 and an empty body each refuse: a serve
that cannot be asked is not a serve without spec-decode. The response the gate
read is kept beside the serve log as the evidence for that verdict.

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

A row is a *normalised* pattern over many modules, so the census does not let
one member speak for the rest: it keeps the first answer and records the exact
prefixes that differed under `disagreements`. Those rows now travel into the
block, their patterns are struck from `offered`, and `classify_construction`
answers `disagreement` for every member of such a pattern (issue #204). A
4-layer cut of a 92-layer model observes four members of each pattern, so a
pattern that answered both ways clears none of them — reading the first
member's `True` as a clearance is how a wire lands where the runtime builds
BF16. No committed receipt carries a disagreement, so the key is absent from
every current row.

The census answers "which patterns" in the *runtime's* namespace and a
producer names modules in the *checkpoint's*, so the two are joined by
`contract.vllm_module_name` -- the one place this repo computes what vLLM
would do rather than reading what it did. The algorithm is code
(`WeightsMapper._map_name_with_shard`), not a table, so it cannot be derived
from the receipt: the receipt publishes the rename table and vLLM keeps the
loop. Principle 14 is therefore met by attestation rather than derivation, in
three parts (issue #108):

* **The replay is faithful and its scope is named.**
  `contract._MAPPER_FIELDS_REPLAYED` is `orig_to_new_{substr,prefix,suffix}`,
  applied in vLLM's order with vLLM's semantics -- a substring rule replaces
  *one* occurrence, and the prefix and suffix loops fall *through*, each rule
  seeing what the last one rewrote. All three differed before #108, and none
  can fire on the two committed censuses, which is why it went unseen.
* **Every other field refuses, by name.** `_require_replayable_mapper` raises
  on any non-empty field outside that set -- `orig_to_new_renaming` (a list of
  transformers objects, not replayable from JSON), `orig_to_new_regex`, a
  populated `orig_to_new_stacked` (which `get_unstacked_mapper` empties, so a
  populated one says the census fell back to the raw mapper and the receipt's
  own field name is wrong), and whatever vLLM adds next. The refusal fires at
  export, where the bytes are decided.
* **The census admits every field, so the refusal has something to see.**
  `tools/tessera_construction_census.py::_weights_mapper_table` reads
  `dataclasses.fields` off the runtime's own mapper instead of a hardcoded
  four-name roster, which used to drop `orig_to_new_regex` and
  `orig_to_new_renaming` on the way into the receipt -- so a producer reading
  that receipt would have mapped the name as though the rule were not there.
  The receipt shape is unchanged for a mapper using only the replayed fields,
  so the two committed censuses stand as taken.

The attestation itself is `test_vllm_module_name_agrees_with_the_real_weights_mapper`
in `tests/test_serving_name_mapping.py`: inside the pinned image it runs the
committed receipt tables *and* synthetic tables built for the three
divergences through both the real `WeightsMapper` and `vllm_module_name`, name
by name, and fails on any disagreement. Receipt:
`docs/measurements/construction/vllm-module-name-attestation-2026-09-04.md`.

### 4.4c A LANE inside a route is gated too, and by the rung

`check_recipe` asks whether the *route* publishes a decode for these bytes.
That is not the same question as whether a named *lane inside* the route can
read them, and the second one had no producer side at all until issue #104.
The window GEMV (`kernel_window_gemv`) repacks each column's code stream at
that column's own rate and has a 16-row lane only where 16 codes of R bits are
one 64-, 32- or 16-bit chunk (`chunk_width_supported` in
`csrc/window_gemv.cu`), so it reads `SUPPORTED_RATES = (1, 2, 4)` and nothing
else -- R = 3 would need 6-byte lanes. A rung is a *root* rate that
`grammar.bresenham_rate_schedule` realises
by mixing the two rates bracketing it -- so q256 1006 (root 3.93) is columns
at rate 3 and columns at rate 4, and **every** unit of such a checkpoint
refuses the lane at load, module by module, through a substitution the route
reports as a served module. All six allocated checkpoints held on the
build box at the time carried a rate outside the set, so no artifact we held
could exercise the lane at all; those checkpoints are box-local and are not
in this repository.

**The set has one home, and it is the kernel** (issue #145).
`csrc/window_gemv.cu` declares `TESSERA_GEMV_RATES(X) X(1) X(2) X(4)` and
`TESSERA_GEMV_WINDOW_BITS 14`; every rate dispatch in the file is *generated*
by expanding that list, and every window width -- template argument,
`TORCH_CHECK` and its message -- is that macro. `tessera.kernel_roster` (parse
only, torch-free, so a producer without torch can read it) parses the
declaration off the same path `kernel_window_gemv._ext` hands to
`cpp_extension.load`, and `SUPPORTED_RATES`, `WINDOW_BITS_SUPPORTED`,
`serving.ext.WINDOW_GEMV_LANE.requires` and hence the contract's published copy
are all that one parse. A rate therefore exists in the kernel and in the
eligibility gate, or in neither. Before #145 they were three literals tied to
each other by tests and to the kernel by nothing -- and already reading
differently: `switch (it.rate)` spelled rates 4 and 2 as case labels and
reached rate 1 through a `default`, so the source parse the issue proposed
would have answered `(2, 4)`. `tests/test_kernel_roster.py` pins that no
second spelling comes back: no rate switch hand-writes a case, no window width
is a literal, the parse fails closed on a source it cannot read
(`KernelSourceError`, never an empty roster), and the packaged contract's
`requires` is checked against the parse directly rather than through the two
Python spellings between them.

The predicate is therefore published (`runtime_contract.json` v12,
`native_extensions[].lane.requires`) and, since contract v20, it is the
lane's **whole** predicate (#264): beside the rates, window, body and plane
it states the decoration classes the loader refuses by name -- RELEASE
overrides, diagonals, rotation, a TP shard's start state
(`layout.slice_unit` stamps `INITIAL_STATE`; the kernel supplies
`state_{-1} = 0` itself) and the grid's arity. Four published conditions
against nine refused was #104's failure mode one class over: a TP row
shard read `READABLE`, exit 0, at preflight while every module fell back
at load. One loader clause stays unpublished on purpose --
`prepare_from_parsed`'s scalar-256-native grid check is an entry-point
fact of the E4M3 table build, and the same extension reads BF16 window
wire through `prepare_value_unit` (a scalar grid of 65536 codes), so
publishing it would call wire unreadable that the lane serves. The
decision has ONE home, `scheme.decide_lane_requirements`: the plan-time
gate, the byte-time report, the loader and `bf16_route`'s gate all decide
a unit through it over the published block, so the predicate and the
loader agree by construction, and a requirement the contract grows is
*refused* by any gate that has not learned it -- on the plan side too,
which used to skip unknown fields. The block is read on both sides:

- **Plan time.** `experiments/export_tessera_serving.py --require-lane LANE`
  calls `scheme.refuse_unreachable_lane` at argument time, beside
  `check_recipe`, for the default rung and every plan override. It needs no
  shape -- reachability is a function of the rung alone (`grammar.rate_set`)
  -- so it refuses before a unit is encoded and names the offending rates.
  The decoration classes are decided from the plan's own statement: the
  keyword defaults state the exporter's pinned plan (whole, undecorated
  units on a scalar grid), and a caller planning a rotated, diagonal- or
  release-carrying wire, or one to be sliced into shards, says so and is
  refused by name. The flag is stamped into the manifest as
  `requires_lanes`.
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
  unit, and exits non-zero. It decides each unit through
  `scheme.lane_wire_report` — the byte-side twin of the plan-time gate, over
  the facts `scheme.wire_facts_of_parsed` keeps off the parse — so the same
  four published requirements are read on both sides. It checked only the
  rates until issue #206, which made the offline checker weaker than the gate
  it exists to double-check: a unit at rate 4 with an L=12 window on a LUT
  plane was published `READABLE`, `reachable: true`, exit 0. A requirement the
  contract grows and that reader has not learned raises rather than passing,
  and a directory holding no `.wire_bytes` is a refusal rather than
  `READABLE 0/0`. Its report is `tessera.lane_preflight/2`, which carries the
  requirements each verdict was decided against and the refusals by name.

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

### 4.4d The expert stack is a STRUCTURE, not a module

A Tessera scheme carries `structure`, and it selects the vLLM method rather
than decorating it. `dense` is one `tessera.fused` container per `LinearBase`.
`routed_moe` is a `RoutedExperts` stack: one container per expert per
PROJECTION -- the granularity the checkpoint's tensors, the runtime's shard
ids (`w1`/`w3`/`w2`) and `tessera.moe_layout`'s cells all already have -- and
the sidecar declares the expert count plus the two GROUPS the fused-MoE kernel
reads (`w13` = gate then up in one matrix, `w2` = down), each with its own
geometry, roles and rungs. Per group it declares a `wire_stride`, not a
`wire_bytes`: the manifest writes `global_scale` as an exact varint ratio, so
a blob's length follows its data and differs expert by expert at one shape and
rung, and what a rectangular parameter can promise is the row width. The
blob's true length rides beside it and
`moe_layout.unpack_moe_wires` refuses a stride that is not the maximum its
lengths imply -- the one check that catches a sidecar disagreeing with the
bytes.

`scheme.MOE_GROUP_PROJECTIONS` derives canonical role names from the runtime's
shard order and is shared by the exporter and reader. The scheme refuses any
other role names or order, even when the blobs agree with the sidecar; `w13`
also requires equal gate/up row counts, because the runtime splits its tile
at `N`. Matching total `[2N, K]` geometry alone does not prove that boundary.

`tessera.serving.moe_route` decodes those containers into exactly the
parameters vLLM's own per-channel FP8 MoE path reads (`w13_weight [E, 2N, K]`
and `w2_weight [E, K, N]` in `float8_e4m3fn`, `w13_weight_scale [E, 2N, 1]`
and `w2_weight_scale [E, K, 1]` in fp32) and then IS
`CompressedTensorsW8A8Fp8MoEMethod` at `strategy: channel`: the same
`select_fp8_moe_backend`, `convert_to_fp8_moe_kernel_format`,
`make_fp8_moe_quant_config` (`per_out_ch_quant` and `per_act_token_quant` both
true) and `make_fp8_moe_kernel`. It writes no kernel. That the runtime's
fused-MoE kernels *accept* a per-channel weight scale on sm_121 is read off
the runtime's own `is_supported_config` predicate, never asserted
(`experiments/results/moe_decode_target_probe.json`: MARLIN, HUMMING, TRITON
and BATCHED_TRITON accept `(kFp8StaticChannelSym, kFp8DynamicTokenSym)`).

The wire parameter carries its OWN `weight_loader`, because
`RoutedExperts.weight_loader` dispatches on the substrings `weight`/`scale`
and would return `False` for `w13_wire` -- writing nothing, silently
(`docs/measurements/tessera-moe-wire-loader-2026-09-03.md`). What
`load_weights` calls is `param.weight_loader`, so the parameter is the seam.

Which families have an expert route is `scheme.MOE_BUILDERS`, and the
absences are measured: the NVFP4 expert arm resolves on this build only under
a `swiglu_limit` clamp that changes the arithmetic the experts execute
(`docs/measurements/nvfp4-moe-oracle-2026-09-02.md`), and a BF16 expert stack
is the passthrough `quantization_config.ignore` already gives. The route
refuses, by name: expert parallelism and tensor parallelism inside an expert
(the stride invariant needs every expert's blob, and no expert slicer has been
run), a residency other than `resident`, a non-gated MoE, and any
expert/hidden/intermediate size that disagrees with the sidecar.

**The exporter writes it.** A `--plan-json` entry keyed `<moe>.experts` -- the
STACK, not one of its leaves, because vLLM builds one method for the stack --
gives every expert of it one rung; `export_tessera_serving.py` then writes one
container per expert per projection under `<moe>.experts.{e}.{proj}.wire`,
derives each group's `wire_stride` as the maximum over that group's blobs, and
declares the `routed_moe` scheme through
`scheme.validate_tessera_moe_scheme` -- the reader is the gate, so the writer
is held to it before the config is written rather than at load. Everything the
route would refuse at load has a plan-time twin (`plan_expert_stack`): a family
with no expert route, an expert population that is not exactly the source
config's declared `0..E-1` -- an interior gap, a missing tail expert, or a
truncated contiguous prefix alike, because `E` comes from config.json
(`_stack_config_geometry`, the same contract the packed path has always proved
against; tessera#213), a missing projection, geometry that differs across
experts or from the config's `hidden_size`/`moe_intermediate_size`, and rows
or columns the route cannot cut. A dense Linear failing the last of those is
passed through when it was implicitly planned; a stack cannot be, because vLLM
builds one method for the whole of it. The construction
gate covers the stack too, through the census's `offered_non_linear` row --
a `RoutedExperts` stack is not a `LinearBase`, so it is recorded there and a
classifier reading only `offered` called it `absent`. An unplanned stack is
untouched: source precision, named in `ignore` at the FusedMoE prefix.
`experiments/moe_write_readback_check.py` writes a stack on the CPU and reads
it back through the plugin's own functions -- scheme validation, per-container
parse, `unpack_moe_wires`, `prepare_tessera_moe_experts` -- and it is where the
stride stops being an argument: at one shape and one rung the blobs of a group
differ in length, because the manifest's `global_scale` is an exact varint
ratio whose width follows its value. On the GPU,
`experiments/moe_route_load_probe.sh` takes the same bytes further: its
`positive_exported` arm loads what the exporter wrote through
`RoutedExperts.load_weights` and executes it through vLLM's own fused-MoE
kernel, matching the arm the probe encoded itself digit for digit while the
bytes on disk differ.

The unpacked source grammar has two attested spellings for that same route.
GLM owns the stack at `mlp.experts` and calls its source leaves
`gate_proj`/`up_proj`/`down_proj`; LFM2.5 owns it at
`feed_forward.experts` and calls them `w1`/`w3`/`w2`. The exporter normalizes
both to the scheme's canonical roles but keeps the source spelling in each
emitted wire name, so the model's own `FusedMoEFactory(ckpt_names=...)` mapping
supplies the shard id to the wire parameter's loader. Two source spellings for
one canonical role are refused rather than resolved by checkpoint order.

The LFM construction row is derived from
`docs/measurements/construction/lfm25-8b-a1b-eugr-0281rc1.json`, taken on the
exact EUGR 0.28.1rc1 image recorded in that receipt. It offers
`model.layers.*.feed_forward.experts` as a non-Linear `RoutedExperts` stack;
the `short_conv.conv1d` projection is never offered and remains source
precision. This records construction eligibility only: the dense runtime
attestation remains on its own pinned image, and this row does not promote a
routed-MoE quality cell.

The PACKED 3-D source layout is accepted only under an explicit plan
convention. `out_first_chunked` is `gate_up [E, 2N, K]` with gate then up and
`down [E, K, N]`; `in_first_interleaved` is `gate_up [E, K, 2N]` with gate/up
alternating and `down [E, N, K]`. The exporter checks those exact shapes
against `config.json`, slices canonical per-expert gate/up/down matrices, and
stamps the convention as `source_layout` on the routed-MoE scheme and each
manifest role. It does not infer either fact from dimensions: when `hidden_size
== 2 * moe_intermediate_size` gate/up is square, and no dimension states
chunked versus interleaved. A missing or unknown convention is refused before
encoding. Old schemes default to `unpacked_per_expert`, the only source layout
their writer supported.

Contract v16 adds exactly two `routed_moe` cells: E4M3/q1024, resident, eager,
sm_121, on the exact EUGR image, for decode and batch. The full LFM2.5 receipt
has all 22 planned stacks / 2,112 projection containers and observes M1 decode
and M64 prefill on the modular TRITON route. Its source-bound usable BF16
teacher comparison covers 4,096 prefill positions: top-1024 KL lower bound
0.0831613565, top-1 agreement 85.1074%, with the recorded tail/upper-bound
limitations. This is not full-vocabulary KL or decode-quality evidence, and
no new numeric quality threshold is implied. Exact identities and results are
in `docs/measurements/tessera-lfm-campaign-2026-09-04.md` §§7–8.

**What is NOT claimed.** The eight dense cells and their default image remain
unchanged. Compiled/streamed MoE, other MoE rungs/images, TP>1 and expert
parallelism remain unattested. The historical three-stack GLM cut's unusable
reference still cannot support a quality verdict. Construction capability is
not itself attestation; the full LFM artifact, census and quality receipts
supply the narrower served claim.

The full-model LFM teacher campaign uses
`experiments/ts5_lfm_teacher_bound.py`: the encoder's sealed source identity
must match hashes of the BF16 source both before and after its read-only
serve. Its receipt binds those checks to the exact image, corpus, tokenizer,
eager prefill mode, dump and build sidecars, and reference-usability result.
The earlier revision-labelled teacher remains historical evidence; checking
its dump hash now does not retroactively establish loaded-weight identity.
This is provenance for the quality measurement, not a new quality threshold.
The separate census/student stages in `ts5_lfm_served_bound.py` likewise
compare every merged shard and sidecar with the checked assembly seal before
and after execution, mount the exact model directory read-only, and preserve
the raw census or matched teacher/student comparison alongside their hashes.
The default artifact and seal remain `full-model` and
`merge-action-r1/artifact-seal.json` under the campaign directory. Explicit
`--model` and `--seal` overrides must be supplied together; the seal's
`checkpoint` must exactly name the selected model before its bytes are read
or its container is launched. The receipt records both selected paths and the
seal hash, and the existing pre/post checkpoint-identity equality still binds
every shard and sidecar. Selecting a new pair never edits the original pair.
Each stage owns one exclusive GPU reservation through verified cleanup.
The teacher and plugin-student wrappers pass `TESSERA_KL_TOPK` to both the
server's logprob limit and the dump request's explicit `--top-k`; a nondefault
support request must not silently fall back to the dump tool's default.
The teacher and served-stage drivers share `experiments/ts5_stage_cleanup.py`.
Container ownership begins only immediately before their launch call; a
prelaunch name collision is observed but never removed by the refusing action.
Cleanup stops telemetry first, joins it for at most two seconds, and shares
one 90-second deadline across all Docker/GPU subprocess waits, within the
120-second outer cleanup grace. Failed inspections or an exhausted deadline
produce an unsafe cleanup receipt; they never count as an empty process list.
An explicit positive `--attempt` selects fresh output, container and local
census paths after a failed stage. Automatic retries retain the same attempt
and refuse its existing directory; no previous receipt is overwritten.

### 4.5 The census attests the route, not the quality -- and engagement, not agreement

`experiments/ts5_moe_served.sh` requires success from all three arms: the
teacher dump, the student comparison, and the route census. Successful KL
arms cannot turn a failed census into a successful campaign action.
Before serving, `experiments/ts5_sidecar_check.py` reads the indexed shards
(or all safetensors files for an unindexed checkpoint), requires exactly one
wire per declared expert and role under its canonical or runtime shard name,
and refuses a repeated tensor name across shards before aggregating their
headers. It recomputes each group's maximum wire length from those headers. A missing
projection cannot pass merely because another wire has the declared stride.

The route census emits `tessera.serving.route_census/2` with its measured
`runtime={image,execution_mode}`. Its `tessera.cell-launch-agreement.by-structure/1`
aggregate contains `tessera.cell-launch-agreement/3` per-structure blocks under
that context. Each record resolves its rung through its declared owner; a MoE
stack has one rung only when every group and role agrees. Dense cells cannot
cover routed experts, and absent MoE cells remain explicitly unattested. A
routed record retains its exact observed backend suffix; matching also accepts
the route-owned base symbol when a cell publishes that entry point. An
explicitly backend-specific cell still requires that exact backend.
Compiled dense per-cell agreement remains unsupported because its trace combines
launches across shapes. A compiled routed-MoE record can agree only when the
record and the runtime-scoped cell each name one launch. Unsupported records
are counted as unattested and retained verbatim in the receipt.

An eager record attests the regime its forward RAN, not the one its phase is
named after. The phase label is what the census asked for; `M` is what the
machine did, and resident FP8 publishes one launch pair for both regimes, so an
eight-row forward filed under the decode phase used to be counted as covered,
agreeing decode evidence (issue #207). `scheme.eager_regime_problem` owns that
rule — `regime_of_m` for the map, `parse_eager_shape` for the spelling — and
both readers call it: `census.cell_launch_agreement` refuses a covered eager
record whose shape is absent, unparseable, or of the other regime, and the
census's own `phase_shape_problems` now reads every record against its own
phase instead of asking only whether the two phases differed somewhere. A
refused record is counted unattested and its problem makes the block disagree.

Lane eligibility schema v5 additionally requires each cell's `runtime` scope:
an exact `image` manifest reference and a nonempty, distinct `execution_modes`
list (`eager`, `compiled`). Image and execution mode participate in overlap
and lookup alongside platform, family, structure, token-count regime and
residency. A missing context or mismatched image/mode is unattested; the global
serve-image pin (`versions.attested_on.image` at v5, `versions.default_serve_image`
since v6) is never an implicit cell fallback. Its existing
dense pin remains unchanged. The eight dense cells preserve both measured
execution modes on that pin; the migration receipt records the historical
headers and their contemporaneous global image binding separately, because
those older census files did not each record a digest. Existing cell IDs stay
stable, with an optional hash of canonical runtime scope to distinguish
disjoint variants; IDs must be unique, and explicit fields decide eligibility.

Lane eligibility schema v6 (contract v17, #131 and #133) makes a cell name
its own evidence instead of borrowing a global. `versions.attested_on` is
gone: it did double duty -- the toolchain the contract was written against
and the runtime the cells were measured on -- and was false for the two
`routed_moe` cells (measured on the EUGR image under vLLM
`0.28.1rc1.dev397+gfd4a15126.d20260904`, not `vllm/vllm-openai` under
0.28.0). Each cell's `runtime` now carries `vllm` and `torch` as its own
receipt records them, beside the v5 `image` and `execution_modes`
(`contract.RUNTIME_SCOPE_KEYS` / `RUNTIME_VERSION_KEYS`;
`cell_runtime_versions` requires the closed object, `cell_runtime_scope`
stays the census's lenient join reader), and the validator refuses two
toolchains under one digest. `versions` is closed to `{tessera,
plugin_entry_point, default_serve_image}` and every field is checked;
`default_serve_image` is the pin of §4.4a and must be an image some cell
attests. Every cell also carries a required, closed `evidence` object --
`{grade, artifact, kl: [{kind, top_k, regime, execution_modes, receipt}], smoke:
{status, receipt, attribution, control}}` (`contract.EVIDENCE_KL_KINDS`,
`EVIDENCE_SMOKE_STATUSES`, `EVIDENCE_GRADES`, `EVIDENCE_RECEIPT_ROOT`,
and schema v7's `EVIDENCE_SMOKE_ATTRIBUTIONS`, `EVIDENCE_CONTROL_REFERENCES`,
`EVIDENCE_CONTROL_OUTCOMES`) -- so a gate can read what
grade of evidence a cell rests on, never prose. The premise this corrects:
every served KL in this repository, dense and MoE alike, is a `kl_tool`
top-1024 teacher/student-intersection lower bound, so no cell grades
`kl_full_vocab`; what separates the cells is the regime the bound was
scored in, the execution modes, the smoke on record, and the population. A
`kl` entry must be in the cell's own regime (a prefill bound written into a
decode cell is refused -- the confusion #133 is about), and `grade` is
derived from the entries and checked, like `executes`: `route_only` when
nothing attests quality in the cell's regime, else `kl_lower_bound`, else
`kl_full_vocab`. On the shipped table every batch cell is `kl_lower_bound`,
every decode cell is `route_only` except `tessera_e4m3_k1_dense_sm121_decode_streamed` (the only route a decode-regime KL was scored against:
`tessera-decode-regime-kl-2026-09-03.md` eager, `tessera-compiled-decode-kl-r6-2026-09-04.md` compiled), the BF16 cells record a greedy smoke, the
`routed_moe` cells record a repetitive one. `qualification` is not
overloaded with the grade.

A smoke also names the **control** it was compared against (schema v7,
contract v18, #195), because `status` alone was deciding admission and could
not carry the decision. Both `routed_moe` cells record `repetitive`; the BF16
**source**, given the same prompt on the same pinned image, eager, returns the
identical completion character for character
(`docs/measurements/moe-evidence-debt-2026-09-04.md` §7), so the repetition is
the model and the prompt. That fact lived in prose while PrismaQuant's pin
refused the whole routed-MoE lane on `status` -- right under the rule it was
given, wrong about the runtime. So `smoke.control` is `null` or the closed
`{reference, outcome, receipt}` of a reference run
(`reference: bf16_source`; `outcome: identical_completion |
different_completion`, which says the completions match or differ and never
that the reference was healthy), and `smoke.attribution` is **derived** from it
and checked the way `grade` is derived from `kl`: no control `unattributed`,
`identical_completion` `shared_with_reference`, `different_completion`
`not_shared_with_reference`. A consumer refuses on `status == "repetitive"`
**and** `attribution != "shared_with_reference"`, never on `status` alone. The
two MoE cells carry the control; the eight dense cells are `unattributed`, six
having run no smoke and the two BF16 cells no reference arm. No status word,
grade, rung, route or byte moved.

Rob decided #133 on 2026-09-04: both `routed_moe` cells keep
`device_qualified`, with the evidence debt recorded rather than closed. Read
the two cells as qualified on **one artifact, one residency mode, one
execution mode, one regime** -- an LFM2.5-8B-A1B E4M3/q1024 checkpoint served
resident and eager on the pinned EUGR image -- whose only quality number is
the batch cell's top-1024 lower bound of 0.0832 (upper bound 1.179); the
decode cell grades `route_only` and rests on the census. That is not thinner
than the dense cells on the `grade` axis, where each MoE cell sits exactly
where its regime's dense cells sit; it is thinner on three axes the table
shows less plainly: the MoE cells attest `eager` only where the dense cells
attest `eager` and `compiled`, they require `TESSERA_SERVE_MODE=resident`
because streamed routed MoE has no route, and the population is one model
with no field to say so. The measurements that would thicken it, ranked with
their costs and with the field each one changes, are in
`docs/measurements/moe-evidence-debt-2026-09-04.md`; the cheapest that moves a
`grade` is a decode-regime top-1024 bound on the artifact that already
exists. That one was blocked and is not any more (tessera#192): on a hybrid
conv/SSM serve the decode sweep could not reach M = 1 at all, because vLLM
runs its prefix cache in `mamba_cache_mode=align` there and only a prefix the
serve has already *answered* is resumable, so a sweep that visits each length
once resumed one stride behind and forwarded `stride+1` rows. `kl_tool
--decode-prime` lifts it by issuing each scored prefix once unscored first;
the teacher half is dumped and fingerprinted on the cells' own digest
(`docs/measurements/hybrid-decode-prime-2026-09-05.md`). The student dump and
the compare are still owed, so no `grade` moves yet. Receipt existence is
`tests/test_cell_evidence.py`'s (the wheel ships no docs); the LAWS tables
are that file and `tests/test_serving_contract.py`.

A cell's `predicates` list narrows the cell to units for which every
`{fact, op, value}` row holds, and the validator refuses anything outside the
closed grammar (`contract.CELL_PREDICATE_FACTS`: `payload_family`, `k`,
`n_sub`, `rate_q256`, `role_split`, `in_features`, `out_features`;
`CELL_PREDICATE_OPS`: `equals`, `in`, `multiple_of`, `at_least`, `at_most`;
values typed per op; one row per `(fact, op)`). Until #134 the field was
required and never read, so `["anything"]` validated. Every published cell
carries `[]`: each is unconditional over its scope. A consumer that cannot
resolve a predicate refuses the cell, never skips the rule, and that refusal
is now a gate rather than a sentence: `contract.refuse_unevaluated_predicates`
raises on any cell whose `predicates` is non-empty, and both consumers of a
cell call it -- `scheme.attested_cells` (so the export gate and
`refuse_unserveable_wire` with it) and `census.cell_launch_agreement`. Neither
selects on a predicate, so the first narrowed cell would otherwise have been
read as unconditional. Publishing one is a consumer change, not a document
change.

The census requires `--runtime-image` as an exact digest reference, checked
before loading vLLM, and since #132 that flag is a CROSS-CHECK rather than the
source of the scope: the launcher resolves the image through docker's
`RepoDigests` and exports the resolved reference and its record into the
container, the census compares its argument against that record, and a run
where the two disagree -- or where the launcher declared no image at all --
refuses before the first model load. There is no opt-out, because a receipt
stamped `operator_asserted` is the same defect wearing a field name. Nothing
inside a container can ask the daemon what it is running, so this is the
LAUNCHER'S DECLARATION and is named one throughout: a host process can export
the same pair by hand, so calling it an attestation would claim more than the
mechanism delivers, and a misnamed claim about another runtime is worse than
an absent one (principle 6). The receipt therefore records the mechanism in
`runtime_image_declaration` (`source`, the two variables, and the record
verbatim) and its absence marks a receipt written before this gate. Making it
unforgeable needs something the launcher cannot write from outside -- a digest
file mounted read-only, checked against the pin -- and until that exists the
honest name is the whole of the guarantee. Its
existing `--compiled` flag determines both the recorded execution mode and
`LLM(enforce_eager=...)`. The plugin wrapper injects its resolved image after
caller-supplied Docker environment arguments; census callers pass that
container value instead of reconstructing an image from the global pin.
Historical tag callers must supply an exact digest for new censuses. Offline
replay reads only the receipt's explicit runtime context: missing context
remains unattested, and a mode contradicting the receipt's `compiled` field is
refused.

`tools/tessera_route_census.py` records, per residency mode, that every
module serves on its declared family. The join is made in MODULE space: the
route records come off `named_modules()`, the declared targets come off
`config_groups` in the checkpoint's namespace, and the model class's own
`hf_to_vllm_mapper` is replayed over the targets before the two are matched --
the same translation `TesseraConfig.apply_vllm_mapper` makes at load, and
without it a mapped architecture joins nothing. A clean census with exact
bytes is necessary and, by tessera#1, not sufficient. It is also not sufficient
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

**A routed expert stack joins by containment, and is graded by its structure.**
vLLM builds one quant method for the declared stack prefix and attaches it to
the `RoutedExperts` child it constructs underneath, so the route record lands
at `<layer>.mlp.experts.routed_experts` while the checkpoint declares
`<layer>.mlp.experts`. An exact-name join reads that as two faults at once --
a served module nothing declared, and a declared module nothing served: eight
problems over three stacks on the first served MoE census -- three records and
one roll-up line in each of the two phases -- every one of them that single
cause. So a record whose `kind` is `moe` joins to the one declared
target that CONTAINS it, and to none if two do -- ambiguity is reported, never
resolved by picking the longer prefix -- while a dense record still joins only
to itself (`join_records_to_declared`,
`tests/test_route_census_module_space.py`).

The structure then decides what that record is graded against. A stack serves
under `TESSERA_FP8` -- same family, same wire, same activation contract -- and
a different dispatch: one materialised launch through vLLM's own modular
fused-MoE kernel, at every M, with no GEMV lane and nothing for a compiled
forward to combine. Resolving the expectation from the FAMILY alone hands the
stack the dense route's pair set and refuses a serve that did exactly what the
route intends, so it comes from the route that owns the dispatch
(`moe_route.census_expected`, the same ownership rule as
`fp8_gemv.census_expected`). Both derive from `scheme.ROUTE_LAUNCHES`, whose
`structures` axis keeps dense and routed launches distinct. Existing launch
lookups default to dense; routed FP8 admits only its resident modular-kernel
launch, in both regimes. The contract's launch derivation passes each cell's
structure to the same lookup. Its symbol is compared without the backend suffix
the record carries (`...modular_kernel:TRITON`): `select_fp8_moe_backend` is
vLLM's predicate over the kernels on the box, so which backend ran is kept in
the receipt's histogram and is never pinned by an expectation of ours.
The suffix comparison lives in dependency-free
`scheme.moe_census_symbol_base`; the runtime route re-exports that helper.
Receipt agreement therefore needs neither torch nor a vLLM import, including
when comparing routed-MoE records in the pure CI population.

`census.STRUCTURE_BY_RECORD_KIND` maps a record's `kind` to the
`lane_eligibility` structure whose cells could cover it (`moe` ->
`routed_moe`). Contract v15 counted those records but left all unattested.
Contract v16 covers only the measured LFM scope in §4.4, using the owner-to-rung
join and explicit runtime context; the historical GLM receipt does not borrow
the EUGR image attestation. The validator derives the positive
`lane_eligibility.structures` set from those receipt-bearing cells, while
`scheme.STRUCTURES` is only the upper bound on what dispatch can execute. Thus
adding a future dispatch structure cannot attest it by omission from a
hand-maintained "unserved" denylist: without a served cell, naming it in the
structure axis is refused. The axis is a non-empty, duplicate-free string list
and equals the cells' first-occurrence projection exactly; set-equivalent
duplicate or reordered spellings are not a second form of the same contract.

`experiments/ts5_census_check.py` gates the full routed-MoE campaign receipt.
Its common plan, merged config and serving manifest must describe exactly one
nonempty expert-stack population, including every expert projection and rung;
the roster and geometry come from the scheme helpers, not a campaign count.
Both driven phases must contain exactly one served record per planned owner,
under the explicit serving-image digest, eager execution and resident mode.
The checker replays construction mapping and ownership, validates launch pairs
and activation contracts even when no cell exists, and requires distinct,
nonempty eager shapes at every owner using the census's shared shape check.
Shapes use telemetry's canonical `M<n>:N<n>:K<n>` spelling; `scheme.regime_of_m`
must map each observed M to the phase it claims. The campaign also compares
N/K with that owner's validated expert tile geometry, so arbitrary strings or
two batch-shaped observations cannot claim decode/prefill coverage.
It preserves actual backend suffixes. Host/container checkpoint paths may
differ: the raw census records `checkpoint_sidecars`, SHA256s of the exact
`config.json` and `tessera_serving_manifest.json` bytes read inside its process,
and the campaign checker requires equality with both supplied file hashes.
The census verifies those sidecar hashes again after both forwards and refuses
to publish a served receipt if either changed.
The generic census permits a missing manifest and records it as null; this
merged-artifact campaign does not. The assembled artifact is held unchanged
through serving; these sidecar hashes do not replace the checked assembly's
tensor/wire validation. Exporter image provenance is not a serving-image
eligibility rule.
The initial check permits genuinely unattested owners. `--require-attested`
replays the same raw records against the **current** packaged contract and
requires every planned owner covered in both phases, ignoring stale embedded
agreement. The result fingerprints its inputs and current contract. This is a
population/dispatch receipt gate, not a wire audit or the separate served-KL
quality gate, and publishes no cells itself.

### 4.5b What the contract says a serve EXECUTES, and the join that checks it

A `lane_eligibility` cell says: on this platform, for this payload family and
structure, at these rungs, in this regime and residency, under this exact image
and execution mode, the plugin executes **these launches** on a route with
this status. The launch half is
`executes` -- a list of `{symbol, decoder}` -- and it arrived with
`lane_eligibility` schema **v4** (contract v13, issue #111). Before it, a
cell published the A-side contract and the rungs, and the launch appeared
only inside the cell's `id`: the E4M3 family said `..._decode_scaled_mm_w8a8`
for both regimes and no cell named the window-GEMV lane at all. That was
accidentally true while the lane was unreachable (§4.4c) and false the moment
a rate-constrained artifact was served -- the R1024 census records
`tessera_window_gemv::gemv` on 112 of 112 modules in the decode regime
(`docs/measurements/tessera-lane-eligibility-executes-2026-09-04.md`).

The following are rules rather than measured values:

- **The value is derived, never asserted.** `contract.validate_serving_contract`
  builds each cell's `executes` from `scheme.ROUTE_LAUNCHES` -- the torch-free
  table the routes' own `fp8_gemv.census_expected`, `bf16_route.census_expected`,
  and `moe_route.census_expected` are built from, and the home of
  `WINDOW_GEMV_SYMBOL` -- narrowed by the cell's structure, regime, by the
  residency the cell's `TESSERA_SERVE_MODE` flag names, and by
  the lanes each rung reaches under `native_extensions[].lane.requires`. So a
  cell naming the GEMV cannot outlive `SUPPORTED_RATES` -- which is the
  kernel's own declaration (§4.4c): drop rate 4 from `TESSERA_GEMV_RATES` and
  the document stops validating (`tests/test_lane_reachability.py`).
- **The regime is *this* contract's, and two vocabularies say "decode".** Here
  `decode` is the one-row forward and `batch` is every M > 1
  (`contract.CENSUS_PHASE_REGIMES`, which is also what stamps a census
  record); the kernel's `decode` is `M <= GEMV_MAX_M` and spans eight token
  counts. Reading the second into a cell is how the batch cell first published
  the prefill launch alone -- true of the 64-row shape the census drives, false
  of the 2-to-8-row forwards the same regime covers, where the lane serves its
  own `gemv` exactly as it does at one row. So the E4M3 batch/streamed cell
  executes **both** launches and the decode/streamed cell executes the GEMV
  alone, and neither is conditioned on a rate.
  `tests/test_serving_contract.py::test_the_launch_tables_regimes_are_the_routes_own_dispatch`
  derives both regime sets from the routes' own `decode_is_gemv`, over every M
  the dispatch distinguishes rather than the two anyone drove.
- **The residency is a condition, not a label.** Both window routes set
  `layer.tessera_gemv = None` in `resident`, so the lane exists in `streamed`
  alone and one rung's decode regime has two answers. The E4M3 family
  therefore carries four cells, and two cells of one `(platform, family,
  structure, regime, runtime image, execution mode)` must cover **disjoint**
  residencies -- otherwise a
  consumer resolving "what runs here" gets whichever cell it read first. A
  cell `id` is now its scope and never a launch, because an id that names a
  launch is a second, unparsed spelling of `executes`.
- **The census closes the loop.** Deriving `executes` proves the document
  agrees with the code; only a serve proves the code agrees with the machine.
  `census.cell_launch_agreement` joins every served route record to the cell
  covering its `(platform, family, structure, regime, residency, rung)` under
  the measured image and execution mode. Missing or mismatched runtime context
  is unattested; a covered launch disagreement refuses. The route census writes
  that context with the block. Compiled dense agreement remains unsupported
  because its trace combines launches as `a+b`; a compiled routed single-launch
  observation may be compared with its explicitly scoped cell.
  `experiments/ts111_replay_cell_agreement.py` replays a receipt offline under
  its recorded runtime context. The historical R1024 replay (a box-local run log, not
  a tracked receipt) recorded 112 of 112 in both
  phases and 112 refusals under the pre-#111 negative control; a source receipt
  missing explicit runtime context now remains unattested under v5.

`TESSERA_BF16_K1` gains `executes` and **no** GEMV cell -- its attested rung
1792 is root 7, outside `SUPPORTED_RATES`, so the derivation returns the torch
window decode under `torch.mm` without being told to. The day a reachable BF16
rung is attested the same derivation produces its GEMV cell.

Schema v4 is **not** additive: a v3 reader must not read a v4 cell, both
because `executes` is a key it does not know and because the E4M3 decode
answer it would have read off one cell is now two. As read on 2026-09-04,
PrismaQuant's parser pinned `tessera.lane-eligibility.v3` exactly
(`prismaquant/tessera_runtime_contract.py:120` at its `1eb88c4e`) and refuses
unknown cell keys, so it fails closed (loudly, not silently) against v4 and
the v5 this tree publishes until that repository widens it
(RobTand/prismaquant#189 carries the v5 reader). Schema v6 (contract v17,
#131/#133) is not additive either -- `runtime.vllm`/`runtime.torch` and
`evidence` are new required cell keys and `versions` is renamed -- so the
RobTand/prismaquant#189 reader must widen once more; the v3 pin fails closed against v6 exactly
as against v4 and v5, and a reader that took `versions.attested_on` through
`.get` (`tessera_runtime_contract.py:1168` at `1eb88c4e`) now reads `None`
there and must move to `default_serve_image` and the per-cell toolchain. Schema v7 (contract v18, #195) is not additive for
the same reason and one more: `evidence.smoke` gains the required
`attribution` and `control` keys, and a v6 reader goes on deciding off
`smoke.status` a decision that `status` alone no longer carries -- which is
the failure the version exists to announce. RobTand/prismaquant#192, which
carries the pin, moves its predicate to `attribution`.
Schema v8 (contract v19, #198) requires `evidence.artifact` (§3.1a).
Pre-release readers must accept that scope explicitly; older closed readers
refuse the new schema rather than silently discarding it.

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
On an attention-only model that is enough. On a hybrid conv/SSM model it is
not, and the refusal fires on every position: the recurrent state is resumable
only from where a request *ended*, so a sweep that visits each prefix length
once resumes one stride behind itself. `--decode-prime` (off by default, so
the receipts below reproduce request for request) issues each scored prefix
once unscored -- warm-up shaped, no `logprobs`, so it cannot be mistaken for a
scored forward -- before scoring it, and the scored request then forwards one
row. Priming the *shorter* prefix instead does not work, and a serve started
with `--mamba-cache-mode all` is not an alternative: vLLM 0.28.1 accepts the
flag and logs `falling back to 'align' mode`. Both measured in
`docs/measurements/hybrid-decode-prime-2026-09-05.md`.
The teacher must be re-dumped in the same regime, and `compare` refuses a
cross-regime pair outright -- there is no override, because the two regimes
run different kernels over different position sets.

The BF16-reference gate derives its target population from that corpus's
contract before it reads quality. With `prepends_bos: true`, the injected BOS
conditions the first corpus token and every token in every chunk is scored;
without it, each chunk's unconditioned first token is omitted. The derived
count must equal the contract's `scored_positions` and the dump's position
count. It is never inferred from the dump shape: a malformed pair cannot pick
the interpretation that lets itself through.

The same contract owns the decode-to-prefill row mapping.
`experiments/decode_regime_subset.py` publishes the prefill regime restricted
to the decode regime's positions, which is the only reading in which the two
regimes differ by the executed forward alone; a chunk's prefill row stride is
therefore read as the contract's `scored_positions / n_chunks` -- `seqlen`
with a prepended BOS, `seqlen - 1` without -- and never spelled `seqlen - 1`
(tessera#249). It was spelled: on a BOS corpus every chunk after the first
folded its positions into the preceding chunk, in bounds, under the
matched-position claim. `--seqlen` is now a cross-check on the contract rather
than the mapping's source, and the three payloads must agree on contract
digest, tokenizer bytes and chunk geometry -- with both prefill arrays sized
as the contract says and the prefill student the same artifact as the decode
student -- before any of them is indexed.

`TESSERA_ROUTE_TRACE=<absolute path>` (off by default, eager only -- under
compile it declines and counts nothing, which is enforced since #113 rather
than described) makes the
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
decode dump's scored forwards, zero on the prefill dump's. Both arms of that
receipt served `--enforce-eager`; the wrapper takes `TESSERA_LANE_EAGER=0` for
a compiled serve, where the trace declines and the attestation is
`compile_identity`'s per-arm AOT key plus the mutual KL itself. When
`TESSERA_KL_PROFILE_DIR` is set, the wrapper also enables vLLM's torch
profiler, starts it immediately before the decode dump, stops it before the
prefill dump, and writes `window_gemv_trace_summary.py`'s kernel summary. That
is the compiled arm's runtime launch evidence; it is not inferred from the
compile-time route trace. Its launch gate counts the manifest's source-role
**units**, because every scored decode position launches the GEMV once for each
role. Its fallback refusal gate separately counts module containers, because
preparation refuses once per container. Thus the #113 population's 256 scored
positions over 112 modules / 196 units requires exactly 50,176 GEMV launches
in a lane arm and 112 preparation refusals in a fallback arm; collapsing those
two manifest axes is a failed gate, not a tolerance. Profile export and parsing
are also distinct phases on GB10: the wrapper stops the decode-only profiler,
takes the matched prefill dump, reaps the serve, then parses the exactly-one
`rank0` model-worker trace under a 900-second timeout and a 64-GiB
`MemAvailable` headroom gate matching the action's declared allocation (not a
hard parser memory cap), and records peak parser RSS. It hashes the
complete profiler-file roster and stream-scans every excluded trace, refusing
if one contains `window_gemv`; an API-process trace is excluded by evidence,
not by filename alone. Parsing before reap drove the shared UMA pool to 3.6 GiB
available in #113 r4 even though the parser itself peaked at 20.1 GiB RSS. The
extension inventory hashes the module directory named by
`ext.WINDOW_GEMV_MODULE_NAME`, not unrelated private temporary directories under
the same container TMPDIR. A postprocessing continuation preserves the original
campaign identity and records each stage's actual source; its gate allows only
controller/test/documentation changes and refuses any serving, plugin, encoder
or wire difference. Its late completion power sample is labelled recovery-time.
Recovery refuses when either stage-seal marker exists but fails verification;
only a stage whose two seal markers are absent can be postprocessed and sealed.
The first arm taken for #113 read: compiled, the
same two arms had mutual `KL >= 0.012585` at
88.67% in the decode regime and `0.000000` at 100.00% in the prefill one --
and that decode number is **below** the same-artifact rebuild delta measured
beside it (`KL >= 0.019423`, arm A re-served, one lane state, two builds), so
**only the eager pair is currently a lane-attributable KL difference**; a
compiled serve on this stack stamps `inductor_deterministic: false`
(`docs/measurements/tessera-compiled-decode-kl-2026-09-04.md` §7).

The fresh #113 r6 population now completes the compiled measurement with two
lane builds, two fallback builds and one same-dispatch compiled BF16 teacher
on Sparklina (`tessera-compiled-decode-kl-r6-2026-09-04.md`). Actual profiler
evidence verifies 50,176 GEMV launches per lane arm and zero per fallback arm,
with 112 fallback preparation refusals. Cross-arm decode KL lower bounds are
0.017367–0.017437 (A as reference), versus 0.006155 for the A rebuild and zero
for the B rebuild. However, cross-arm prefill is also nonzero (0.019502), so
this is not an isolated M=1 GEMV quality effect. The teacher lower-bound ordering
flips between A1 (0.437342), A2 (0.442105) and B1/B2 (0.438400); these are
top-1024 bounds with substantial unobserved mass, not full-vocabulary KL or a
quality winner. #113 closes as a completed measurement, without changing a
promotion gate. R6 reaps the serve before rank0 parsing; the two sampled memory
intervals stayed above 21 GiB available, and peak parser RSS was 4.34 GiB.

**The eager decode-regime gap is now fully accounted for, and it was half a defect
and half accumulation order** (#110) -- the issue offered those as alternatives
and the answer was both. One real term was found, fixed,
and re-served: the streamed lane handed the
kernel `(a_q.float() * a_scale).to(bfloat16)`, folding the per-token fp32
activation scale into a bf16 operand, where `torch._scaled_mm` multiplies the
fp8 codes and applies both scales in its fp32 epilogue. An E4M3 code is exact
in bf16; the code times an fp32 scale is not, so the fold cost 1.6e-03 relative
rms per Linear output -- one bf16 rounding -- against a MEASURED fp32 reduction
error of 1.7e-07, some 10 000x below it. The
lane now applies `a_scale` to the fp32 output, which is the rule `bf16_route`
already held for the weight side. Every other candidate is closed by an
artefact: the arms are one inode, prefill (where both take the materialised
path) reads exactly 0.000000 over 4088 positions, and the serve logs differ only
in the 112 intended refusals -- same attention backend, FlashInfer autotune
`Saved 0 configs` in both, the same lone JIT compile in both -- so the `M <= 8`
branch is the only place a difference can live, and inside it the fold is the
only term above 1.6e-07. Priced as the deterministic rounding it is, on the
served position set and in the served decode regime, the fold reads
`KL >= 0.007160` at 95.70% top-1 against the measured 0.012111 at 91.02%: the
same size, within 1.69x. An earlier Gaussian screen read 1/38 of that, and a
one-variable matched pair says why: same regime, same arm, same 256 positions,
same relative rms, the route's per-token E4M3 activation quantiser in the loop
or out of it, reads `KL >= 0.007788` against `KL >= 0.000113` -- **68.9x** from
that variable alone. The screen had priced a W8A8 term on a trajectory carrying
no W8A8 activation rounding. The residual 1.69x is the emulation's weight
operating point: degrading the weights toward the served arms' distance from
BF16 (`--weight-bits 4`, same regime, arm and chunks) reads `KL >= 0.012073`
against 0.012111 served, a factor of 1.48 in the right direction. The emulation
is still CPU, HF rather than vLLM, with an fp32 residual stream, so the served
re-run after the fix is what decides.

**That re-run has landed** (pool action `6c90ba1b`, 2026-09-04; same two arms
through one inode, same corpus, and `vllm/vllm-openai@sha256:61fc8a896b0a...`,
the digest #102 served on). The arms now read `KL >= 0.005947` at 96.88% top-1
where they read `KL >= 0.012111` at 91.02%: **2.04x on the bound, 2.88x on the
top-1 flips.** Three controls hold it up -- the untouched arm B reproduces #102
at `KL >= 0.000000` / 100.00%, and so does the prefill regime the fix does not
touch, over 4088 positions -- and against the BF16 teacher the fixed arm A moves
from `0.436065` onto arm B's `0.432477` (reading `0.432401`), so the fallback
was the accurate arm and the lane has joined it. **The residual `0.005947` is
now attributed, and it is not a difference between the arms** (pool action
`6921120b`): serving arm A a SECOND time, changing nothing, and comparing the
two decode dumps reads `KL >= 0.005985` at 95.31% top-1 -- the same size as, and
slightly larger than, the arm-vs-arm figure. What makes that an attribution
rather than a coincidence of magnitudes is that the decode regime reproduces
exactly on the *other* arm: arm B is served in the same regime -- same decode
attention kernel, same scheduler, same KV-block allocation, same 256 scored
M = 1 forwards -- differs from arm A in the GEMM alone, and a serve a day after
#102's reads `0.000000` at 100.00%. So the decode pipeline minus this kernel is
exact across serves, and the spread has one place left to live. Prefill, where
the GEMV never executes on a scored forward, is a second and weaker null in the
same direction (`0.000000` over 4088 positions, across arms and across serves),
and a cold inductor cache changed nothing. It was also predicted from fp64 first: at `M = 1` the lane
matches `torch._scaled_mm` on 99.90-100.00% of bf16 output words and matches
*itself on a rerun* on the same 99.90-100.00%.

**So the lane's reproducibility floor is a number a receipt can cite:
`KL >= ~0.006` at `~95%` top-1 in the decode regime**, on 256 M=1 positions of
one 0.6B checkpoint. A served decode difference at or below it is noise; the
`0.012111` #110 filed was above it, and that excess was the fold. Read the
lane's "bit-exact" receipts as claims about the **decoded tile** only, never
about the GEMM over it -- the tile is bit-exact 196/196 and the GEMM is
reproducible only to that floor.
`docs/measurements/tessera-gemv-a-side-2026-09-04.md` section 6d is the receipt.

### 4.6 The stock twin isolates the wire from the kernel

`--stock-twin` writes the same wires materialised for vanilla vLLM, so a
served comparison is one encode under two servings rather than two encodes.

### 4.7 The verdict is served KL against the byte-matched control

`experiments/uniform_control.py verify` asserts the match on the bytes that
shipped and, given both KLs, states whether the candidate beat its control.
`tessera.control.control_block` carries that verdict beside the bpp.

**The verdict is published only over evidence the gate validated**
(tessera#225). `ByteMatch` is where the four numbers a match reads have their
domains — whole positive bit totals for both arms, an exact positive
parameter count, and a tolerance that is a fraction of the candidate's own
bits in `[0, 1)` — and each is refused by field name before anything is
divided; `assert_byte_matched(0, 800, 1)` used to return an accepted match
reporting a perfect `relative_slack` of 0 for an arm of no bytes. And a
*measured* verdict now requires `control.match.byte_matched` to have held,
because its own sentence is "against the byte-matched uniform": an unmatched
pair — which `uniform_control(..., assert_match=False)` builds deliberately,
to report the E2M1x2 coset hole rather than paper over it — stays
representable as the unserved block, an explicitly unqualified diagnostic,
and cannot become a victory. Both KLs are validated as finite and
non-negative for the same reason the totals are: the verdict divides them.

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

**Before any leg, the evidence has to be evidence** (tessera#224). The gate
validated each per-unit ratio as positive finite and then merely
`float()`-converted the other four numbers, so an ordered comparison stood in
for a domain check and mathematically invalid evidence promoted: a served KL
of `-inf`, a `glm_ratio` of `-inf`, a `served_bar` of `+inf`, an infinite GLM
cross-check against an infinite bar. Each number is now checked against the
domain its own definition gives it and refused by field name — a ratio of two
errors is finite and strictly positive (zero is a division artifact and the
geomean reads it in logs), a KL divergence is finite and non-negative, and the
two are spelled apart rather than sharing one word that fits neither. The
`glm_bar` override **tightens the pinned `GLM_GATE` and never relaxes it**:
comparing only against the caller's own bar let `glm_ratio=1.5` promote under
`glm_bar=2.0`, the coordinator's cross-check answering to the arm it checks.
The domains live on `PlanePromotion` as well as in the assertion, because the
class is public and `promotion_block`'s "only a promotion this gate accepted
reaches here" has to be true by construction; the `geomean` and `wins` it
publishes must be the pair the promotion's own unit ratios make.

No default moves by this, and `tests/test_plane_promotion.py` is what makes
that checkable rather than asserted: it runs the receipt's own six-unit
record through the gate, watches `hessian` refuse at 2 of 6, and pins
`DEFAULT_REFIT_OBJECTIVE["lut16"]` to the `h^1.0` that refusal leaves
standing. Flipping that default without a promotion this gate accepts turns
the suite red.

#### Before the legs: the screen's own recorded proofs are read, not printed

A promotion gate reads a screen document, and that document records the
evidence that can invalidate the experiment that produced it:
`drift_control_identical` (the same arm run first and last in one process
reconstructed the same weights), `landing`/`serialisable`/
`sink_vs_wire_bit_identical` (the sink the arms were scored off IS the wire
that ships), and, on a trailing arm, `matched_pair`'s `codes_identical`,
`bytes_equal`, `inner_objectives_equal` and `inner_refits_identical` (the two
arms differ in the last scale plane and in nothing else).
`experiments/refit_trailing_pair.py` wrote and printed all of it;
`experiments/refit_trailing_pair_gate.py` read none of it, so a screen whose
control DIFFERS, or whose trailing arm changed its packed codes, still reached
`assert_plane_promotion` on its ratios alone and could print PROMOTED
(tessera#250). `experiments/refit_trailing_screen.py` is now the one home for
that reading, and both the producer and the gate call it: a failed control or
a mislabelled landing refuses the whole document, in **both** populations,
before a ratio is computed; a failed matched-pair leg refuses the arm that
claims the pair. A proof that is absent refuses exactly as a proof that is
false does.

**Which arms owe the matched-pair proof is derived from the receipt, not
listed.** A trailing pair is an arm whose recorded refit schedule carries the
control's inner objectives with the trailing one swapped -- `1,1,1,2` against
`1,1,1,1` -- and whose `refit_diagnostics` records no coupled landing, because
#50's coupled landing re-assigns blocks and is *expected* to move the codes
the next trellis pass sees. Naming `B-Jac`/`B-GS` would pass on the day the
roster is wrong and would let a receipt exempt an arm by deleting its proof.
`plane_moved=false` is recorded and deliberately not required: an arm whose
lever reached nothing is an ineffective arm, which is a result and not a
broken comparison.

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
  **re-run trigger** for the day a better landing lands (issue #50). That
  landing has since landed and did **not** cash the ceiling: the coupled
  landing wins the Qwen screen (0.8037x) and **fails** the GLM six-expert
  cross-check at 1.0160x, so it stays opt-in and the default is unmoved
  (`docs/measurements/tessera-coupled-landing-glm-2026-09-04.md`).

The pair is not free and is not readable off `refit_diagnostics`. That
instrument's `continuous` leg is a within-call quantity by its own contract --
for a 1-D metric it records the separable parabola, equal to the weighted
error only up to a constant -- and the arms being ranked are 1-D (`h^1.0`)
against full-H (Jacobi, Gauss-Seidel). The diagnostics give the *size* of the
landing leg within one arm; the ordering across arms costs one extra
`lut_landing("none")` encode per arm (`experiments/lut_landing_ceiling.py`,
no serve and no KL). `tests/test_landing_ordering.py` pins both halves.

### 4.12 The LDLQ block is derived from the unit, and the default is a floor nobody has to defend

`ldlq_block` sets LDLQ's error-feedback granularity, and until tessera#60 it
was one round number (`export.DEFAULT_LDLQ_BLOCK = 32`) applied to every unit
of every model. It is measurably the wrong shape of knob:
`compensate.block_penalty(H_reg, block)` prices what a block costs against
full feedback in closed form, and at `b=32` that is **9.8%** of full feedback
on dense Qwen attention against **0.14%** on GLM experts -- a factor of 70
between two populations one constant has to serve (and 14.7% on `q`/`k`/`v`
specifically against `o_proj`'s 1.4%, a factor of ten inside one model;
`compensate.block_penalty` and `tessera-dense4-gap-2026-09-03.md`).

`ActivationSource.ldlq_block` therefore takes either the width it always took
or a **budget**, `{"max_penalty": ratio}`, and `ActivationSource.block_for`
derives each unit's block from that unit's own Hessian through
`compensate.choose_ldl_block` at `floor=1` (the `encode_unit(ldl=...)` path's
floor, per tessera#95; the 16 that `choose_ldl_block` used to inherit is
`compensate.compensated_targets`' constraint, not this path's). At budget 1.02
one export gives `q`/`k` b4-b8, `down_proj` **b64** -- coarser than today's
default -- and GLM experts **b256 or coarser**, eight times cheaper than the
constant they run today. That is the thing a global flip cannot do.

**The default does not move, and the reason is a serve, not a preference.**
The weight-space case for a smaller constant was strong -- 6 of 6 units, a
geomean crossing parity -- and it mostly did not carry. Bracket `b32 -> b8 ->
b32`, one session, one teacher, byte-identical files, drift control
`0.000e+00` on both metrics (tessera#60,
`docs/reports/ldlq-block-served-ab-2026-09-04.md`): weight space predicted
0.9373x, served all-position realised **0.9806x** -- a carry fraction of
**0.31** -- and on confident positions **1.0003x**, no carry at all. A
weight-space geomean on this axis is over-optimistic by about 3x.

The budget is opt-in for the same reason: it is priced, not promoted. Nothing
in the default path calls it, `DEFAULT_LDLQ_BLOCK` is untouched, and the wire
is unchanged -- proved by two exports at a served arm's verbatim arguments
hashed on raw safetensors byte ranges (`HEAD~1` vs `HEAD`, and `HEAD` vs the
bracket's arm at `82cdf513`), both byte-identical, both `--layers 1`.
`tests/test_ldlq_block_budget.py` splits 4 regression guards (which pass on
the pre-change tree, as guards must) from 16 that fail there.

**What the same bracket settles about the dense 4-bit route** (tessera#12,
`docs/measurements/tessera-dense4-gap-2026-09-03.md`): at equal residency --
three files of 870,290,032 bytes, one corpus, one teacher -- the 4.0-bpp
Tessera wire serves at 0.5099719526 against PrismaQuant NVFP4 GPTQ+JSO's
0.5105764372, **0.9988x, parity**. The 1.254x deficit the route was measured
at is gone; a 0.12% margin on one 4,088-position corpus is not a lead and this
doc does not read it as one. Attribution matters more than the endpoint: the
2026-09-02 LDLQ + `h^1.0` refit work closed most of it, the merges between
that artifact and `82cdf513` closed a further 2.1% with no recipe change, and
the block is worth 1.94% against its own session's control. Quoting the
published incumbent instead of re-running it would have credited the block
with twice its size.

## 5. Packaging and release

The distribution is `tessera-quant`; the import name is `tessera`
(`pyproject.toml`). What the wheel carries, how the plugin registers once
installed, and what the two CI jobs prove are gates in their own right, so
they are recorded here with the rest.

### 5.1 The plugin is delivered by an entry point

`pyproject.toml` declares one entry point in the `vllm.general_plugins`
group, `tessera = tessera.serving:register`. vLLM loads every plugin in that
group at start-up; `register` imports vLLM lazily and registers
`quant_method="tessera"` (`src/tessera/serving/__init__.py`). Nothing the
operator passes selects the plugin: the checkpoint's `quantization_config`
names the method, and the only operator knob is `TESSERA_SERVE_MODE`
(`resident` or `streamed`, `src/tessera/serving/lane.py`). The entry point
has to resolve without vLLM present, because the producer imports the same
package on a box that has none; `tests/test_packaging.py` holds it to that.

### 5.2 What the wheel ships besides Python

Three non-Python files are opened at run time, and each is declared in
`[tool.setuptools.package-data]` because an editable install reads the tree
and would never notice one missing:

| File | Opened by | Why it is in the wheel |
|---|---|---|
| `tessera/serving/runtime_contract.json` | `contract.contract_path()` through `importlib.resources`, by the plugin at load and by the producer preflight | the attested-cell table (§3, §4.4d); repo-root arithmetic is refused so a wheel, an editable install and a checkout read the same bytes |
| `tessera/serving/csrc/tessera_nvfp4.cu` | the NVFP4 route's JIT build (`ext.py`) | the span-2 decoder |
| `tessera/serving/csrc/window_gemv.cu` | `tessera.kernel_window_gemv._ext`, which `serving/fp8_gemv.py` and `bf16_route.py` load through; the loader resolves the path from `ext.NATIVE_EXTENSIONS[].source` via `ext.native_source_path`, so the published `source` and the compiled file are one inode | the window-body GEMV. Until #134 a second, byte-identical copy at `tessera/csrc/window_gemv.cu` was the one compiled while this one was the one published; `tests/test_serving_fp8_gemv.py::test_the_published_source_is_the_file_the_loader_compiles` now captures the JIT call and asserts `samefile` |

`tests/test_packaging.py` refuses either half of that table on its own: a
glob that matches no file, and a runtime data file no glob covers.
`tools/check_wheel.py` asserts the same on a *built* wheel, installed with no
dependencies into an empty directory and imported with the source tree off
the path, and prints the wheel's own file list; CI runs it on every push and
the publish job runs it on the bytes it is about to upload.

**What the wheel deliberately does not ship.** `src/tessera/_dev/` is the
repository's own tooling -- the merge-suite deadline helper
(`_dev/suite_deadline.py`), the PrismaBuild source-identity reader
(`_dev/suite_source.py`), and the import-graph analyser behind
`tools/impacted_tests.py` (`_dev/source_dependencies.py`). It lives under
`src/` because `tools/` imports it by module name, and until #151 it
therefore installed into every consumer's `site-packages`. One line of
packaging config drops it -- `[tool.setuptools.packages.find] exclude =
["tessera._dev*"]` -- so the modules are named by a pattern the build reads,
not by a roster kept beside it. `exclude` is intent, and a rename, a
namespace sweep or a stale `build/` tree reinstates a package silently, so
the proof is on the artifact: `tools/check_wheel.py` reads that same config
and refuses any wheel or sdist member whose dotted package matches, and
`tests/test_packaging.py` refuses a pattern matching no package (which would
excuse the artifact by excluding nothing) and any shipped module that names
`tessera._dev` (which would install an ImportError no checkout can see).
`suite_source`'s receipt schema string, `tessera.suite_source.v1`, is a wire
identifier already stamped on receipts under `/mnt/shared` and read back by
`tools/merge_suite.py`; it did not move with the module.

The **sdist is the source that rebuilds that wheel, and nothing else**. It
is deliberately not a runnable test suite: a testable sdist is a claim this
tree cannot back, because the suite reads `tools/`, `docs/` and
`experiments/` and pins per-box absolute paths (#153), so it runs from a git
checkout and only from one. `MANIFEST.in` states that decision and `prune
tests` carries it out -- without it setuptools' directory sweep shipped 149
test modules and left out the `conftest.py` that collects them (#151).
Neither half of that is left to a sweep again: `tools/check_wheel.py`,
given the sdist beside the wheel, derives the expected contents *from the
wheel's own namelist* -- every source the wheel ships, under `src/`, plus
the build inputs -- and refuses anything else, in either direction, and
`tests/test_packaging.py` refuses a `MANIFEST.in` directive naming a path
that has moved (which would silently stop excluding anything). Measured on
this tree at `4f2f95a`: the sdist rebuilds a wheel with an identical
72-entry namelist (66 payload paths plus six `dist-info` entries), and holds
76 paths by the count `check_wheel` prints (the archive's own root entry and
the regenerated `egg-info` excluded), none of them a test module.

Three extras: `serve` installs a stock `vllm>=0.28` so the entry point has a
host, `kernels` installs Triton, and `native` installs the `ninja` both of
them need to build the packaged `.cu` sources -- named once there and
referenced by the other two (`tessera-quant[native]`), so a consumer cannot
install a runtime that reaches the JIT build without its builder. A PyPI
vLLM is a **working install, not an attested one**: every cell in the
contract is pinned to an image digest (§3), and a serve on any other runtime
gains no claim from it.

### 5.3 The JIT build and what degrades without a compiler

The native extensions are built by torch at first use from the packaged
`.cu` sources and need an `nvcc` and a `ninja` on the box
(`src/tessera/serving/ext.py`, "TOOLCHAIN"). The `ninja` is declared -- the
`native` extra, which `serve` and `kernels` both reference. The `nvcc` is
not installable from PyPI and is therefore a documented requirement, stated
in the README beside the install commands and here; `ext.py` records where
each is looked for. When a build is unavailable the outcome is per extension
and per residency, and it is a value the route record stamps, never a
boolean:

- `substituted` -- a *named* substitute decoder ran and the serve is a
  different numeric object than the native one. The resident NVFP4 route
  decodes once at load and may substitute `tessera.stock.materialize_stock`;
  the window GEMV substitutes the torch window decode in both residencies.
- `refused` -- no serve exists. The streamed NVFP4 route decodes inside a
  traced forward whose data-dependent shapes the substitute cannot run, so it
  refuses instead of serving something else (`ops.prepare_tessera_module`).
  A module also refuses at load, in either residency, when the native decode
  disagrees with the reference (§3.3): a build that exists and is wrong is
  not a serve.

The decoder that actually ran is the `decoder` field on every route record
(`telemetry.py`), which is how a fingerprint tells a native serve from a
fallback one. A census (`tools/tessera_route_census.py`) that reads
`substituted` on a module is reporting a serve the contract does not attest.

### 5.4 The two hosted jobs, and what each does not prove

`.github/workflows/ci.yml` has two jobs.

**`pure`** runs on every push to master and every pull request, in an
interpreter with pytest and nothing else: the bytes-only tests (whatever
`tests/conftest.py` can collect without torch, reported with the modules it
could not), an import of the byte layer that asserts torch never entered the
process, the empty-denylist refusal (build item 11), and the wheel and sdist check
above. It proves the parser's dependency boundary and what both
distributions contain.
It does not run a CUDA kernel, a decoder against a served artifact, or the
merged suite: those are the two-population suite of §1.1, dispatched through
PrismaBuild and read in `docs/status/suite-populations.md`, and a serving
claim also needs a served receipt (§3).

**`publish`** runs only on a `v*` tag and only after `pure` is green. It
refuses a tag that does not name the version in `pyproject.toml`, builds the
sdist and wheel, runs `tools/check_wheel.py` on both, and uploads with
`pypa/gh-action-pypi-publish` under an OIDC token (`id-token: write`); there
is no API token in the repository.

The trigger is a bare `v*` tag match, and a tag is not a review gate -- so
before the job builds anything it refuses a commit that is not reachable from
`origin/master` (`.github/scripts/require_tag_on_master.sh`, under a
`fetch-depth: 0` checkout, because a shallow clone answers reachability from
whatever history it happens to hold). Every other outcome is a refusal too: a
shallow checkout, or a branch the runner could not read.

The job also runs in the GitHub `environment` named `pypi`, which is what
gates the OIDC token: the ancestry script decides *which commit* may publish,
and the environment is where a person decides *whether to*, before the
credentials exist. Half of that gate is not in this repository and cannot be.
The environment is created in repository settings and protects nothing until a
required reviewer is added to it; its deployment rule has to admit tags, since
"protected branches only" refuses a tag-triggered run; and PyPI's Trusted
Publisher must name the same environment, because the environment is one of
the claims the token exchange matches. So this section claims only the
workflow key -- `tests/test_ci_workflow.py` asserts it is job-level, which is
the half a test in this tree can read. Whether the protection exists is #17's
question and Rob's alone, and until it is answered this key buys the ordering,
not the review.

Every `uses:` in the file names a commit SHA, with the version it was in a
trailing comment. A tag or a branch (`@v4`, `@release/v1`) is a ref another
account can move, so what a job runs is decided after review, by someone
else; a SHA is the code that was reviewed. `tests/test_ci_workflow.py` holds
that rule over every workflow rather than over a list of actions.

### 5.5 The version has one declaration

`pyproject.toml`'s `[project] version` is the only place the version is
written. `tessera.__version__` reads it rather than restating it: out of a
checkout from that file, out of an installed wheel from
`importlib.metadata` -- the checkout first, because installed metadata on
`sys.path` can describe a different tree than the one being imported
(`src/tessera/__init__.py`). `tessera.serving.__version__` re-exports the
same object, so the string vLLM's compile-cache key folds in
(`serving/compile_identity.py`) and the census publishes cannot disagree
with the distribution, and a version neither reader can produce is refused
rather than guessed.

Three copies remain that no code here can derive: the `Documentation` URL's
tag, the release tag README.md pins its own links to (it says so, because
relative links do not resolve from a PyPI page), and the contract's
`versions.tessera` / `versions.plugin_entry_point`, which are bytes a
producer's receipts bind to. None is left to review.
`tests/test_packaging.py` fails when either disagrees with the declaration;
`tools/check_wheel.py` reads the declaration too -- it restates neither the
version nor the entry point -- and refuses a built wheel whose `Version`
metadata, whose entry-point value, or whose installed `__version__` is not
the declared one; and the publish job's tag check reads the same table
(§5.4).
