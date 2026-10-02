# Native shape-time receipt, first slice

Tessera #688's producer schema is `tessera.shape_time_panel.v1`. The first
implemented slice is exactly one E4M3 FP8 dense operator, with a uniform
rung, explicit TP1 rank0 native layer geometry, eager execution and resident
weights. Routed, TP2, multiple-row and alternate-family panels refuse.
This is an operator-sum proposal, not placement, serving p95, quality,
runtime-cell qualification or a format-menu change.

`tessera.serving.timing_panel.validate_panel` is passive and stdlib-only.
The caller supplies an independently frozen expected runtime. The receipt
binds the actual runtime and independent producer records, raw runtime
contract, canonical wire, native preparation, raw CUDA-event samples,
observed per-call routes, gzip profiler trace, native ELF binary and full
raw telemetry by absolute path, byte length and SHA-256. Its single scope
is rebuilt through the existing census planner and dispatch registry. The local
public API validates against its own packaged runtime. External panels use
`validate_external_panel` with the private immutable result of an actual
installed-runtime CPU preflight, never a boolean or a copied extension roster.
That phase runs the installed reader's unchanged strict validator on its own
hash-bound raw bytes, after RECORD, root, commit and source/software checks.
The existing planner's private validated-document core then retains the
producer registry hash without claiming that producer is the runtime.

The canonical fused frame reader is shared with `tessera.fused`; existing
framing bytes and public names remain unchanged. The passive replay uses
`container.parse` to verify manifests, plane digests, geometry, rung,
body, plane and span. Hardware grid/profile reconstruction remains the
ordinary native loader's responsibility; its preparation evidence is
bound to the same wire and actual runtime. The validator does not expand
weights or simulate a device.

A positive row requires exactly one backed cell matching image, package
versions, platform, regime, residency and rung. A declared cell code pair
must be complete and match; a legacy cell with neither code field uses the
independently frozen expected runtime commit/package digest/raw contract,
checked against actual installation and RECORD at entry and exit. No expected
identity is derived from the observed receipt. It must
declare Tessera's plugin, satisfy required flags and name the actual
`(symbol, decoder)` pair. The shared lane wire predicate and the existing
census launch-agreement rule must both pass with positive coverage.
Unattested/unsupported agreement cannot become a timing price. Each timed
call must have a fresh served record with its actual shape and activation
contract; missing, error or changed routes refuse.

At least three finite positive CUDA-event samples are required. The median
uses `statistics.median`, and quartiles use
`statistics.quantiles(method="inclusive")`; cached summaries are compared
to the raw samples. Profiler evidence requires actual CUDA kernel events.
Both Sparks' raw power/CPU/load/swap/available-memory contexts and fast
power samples retain the timed interval. Energy remains `hold` for
cross-host clock alignment ([PrismaBuild clock alignment](https://github.com/RobTand/prismabuild/issues/1440)); no work/J claim is accepted.

Deploying these new Python sources as a runtime changes its package identity.
The repository producer instead loads its own passive schema in a separate
process while the native worker imports an independently pinned installed
runtime. An unchanged b40/v45 runtime retains its original cells; current
b40/v42 panels are not rewritten or relabeled. A GPU pilot still requires
separate acceptance of the exact observed runtime and requested cell.
CPU stand-ins validate this contract without claiming GPU execution.

`tools/tessera_shape_time_panel.py` is the supported repository application.
`check-request REQUEST.json` checks the request's owned bytes, scope and
possible cell/wire join; installed contract validation is explicitly pending.
`preflight-request --request REQUEST.json --request-sha256 SHA256 --output NEW_DIRECTORY`
runs the actual CUDA-disabled installed contract phase and publishes an
unmeasured plan with bound result/returncode evidence. `measure --request REQUEST.json --request-sha256 SHA256 --output NEW_DIRECTORY` calls
`lane.build_tessera_method`, `create_weights`, loads the exact owned wire into
the plugin parameter, and calls `process_weights_after_loading` and `apply`.
It clears the latest route before every call so retained records cannot stand
in for a launch. Output geometry and an initial finite-output check must pass;
prepared-weight fingerprints and both source identities are checked again
before publishing the final receipt. These controls do not establish numerical
quality or end-to-end serving correctness.

The request schema is `tessera.dense_shape_time_request.v1`, with exactly
`schema`, `expected_runtime`, `scope`, `prefix`, `scheme`, `wire`, `sampling`,
`netdata_hosts`, `contract`, `runtime_python`, `worker_timeout_s` and
`record_verifier` and `producer_identity`. The latter binds an independently
host-attested clean Git checkout plus the existing source-tree and tool-closure
identities. `seal-producer --output FILE` runs on the host before submission;
the container recomputes the exact source hashes and records
`commit_source=sealed_checkout`, without claiming an in-container Git observation.
`record_verifier` binds the published PB stdlib installation
checker; it is repository tooling, not a package dependency. The
contract and runtime interpreter are absolute bound files; the contract is
the pinned runtime's raw bytes. The timeout is an explicit positive integer. `scope` uses the existing native census request fields;
`scheme` is the existing dense wire declaration and `wire` is a bound file.
`sampling` names integer `samples >= 3`, integer `warmup_iterations >= 1`,
integer `seed >= 0` and finite positive `steady_s`. `netdata_hosts` explicitly
names both `sparky` and `sparklina`. No topology or runtime is inferred.
Preflight requires the exact independently bound runtime contract digest and a possible positively
backed cell/wire join; the actual native pair must join again after preparation.

`expected_runtime.package_root` explicitly binds the actual installed package
path. A bound `runtime_origins` artifact records its required module files
and the reused installation checker's complete VCS/RECORD proof. Source
digests are refreshed at each boundary rather than trusting a cached value.
The actual imported Tessera package supplies its commit through a clean,
tracked Git checkout or VCS installation metadata owning that imported file.
The launcher image declaration, package source digest, raw contract digest,
Torch/vLLM versions, device platform and serving flags must equal the frozen
expected context. The repository tool has a separate Git commit and source
closure. Neither identity substitutes for the other. `native_packed_bytes`
records the existing prepared object's named tensor byte sum, including views;
it is not a deduplicated whole-process residency measurement.

The producer reuses `step4_capture_launch.run_phase` for the installed CPU
contract preflight and the subsequent native subprocess in the existing admitted action, with inherited CPU affinity and
thread bounds. Its argv removes `PYTHONPATH` and uses the explicit interpreter
with `-I -B`. The native worker adds repository tools, never repository `src`,
and refuses a foreign package root, Python path or cached runtime owner. All
required module origins are recorded; package source and raw contract must
match at entry before CUDA setup and again at exit. Its versioned job/result
files bind worker source, request and evidence. The existing
`bench_native_operator.native_runtime_context` owns TP1 setup/teardown.

CUDA-event sampling reuses `bench_native_operator.time_apply`; profiling uses
`torch.profiler`; fast power and raw Netdata reuse the existing instruments.
The requested steady interval supplies useful native calls for box observation,
not additional timing prices. Raw responses must contain result samples.
Every artifact is file- and directory-synced, and the final panel is validated
before its durable publication. Only then may repository tooling report one
committed PB publish unit. The package validator has no PB dependency.

The first accepted native GPU pilot and independent fresh installed CPU replay
qualify the bounded execution path at q896/M512/N=K256/TP1 under b40/raw0869/f8.
They establish no full-menu, serving, numerical/quality, compiled or cell qualification. The producer commit and schema origin are independent from the worker
runtime's source; no unchanged pinned runtime requires requalification solely
for a new producer checkout. No existing b40/v42 or other cell is relabeled, no census
is published, and no PrismaQuant timing consumer or pin changes in this slice.

The installed runtime uses its owning noneditable Git VCS metadata and full
RECORD proof even when the image has no Git binary. Producer Git metadata
never substitutes for that runtime commit.

The first application slice requires the prepared owner's actual fused lane
before timing. Its warm call must map exactly one ELF matching that native
owner's published extension. The existing process-map observer binds those
already-loaded bytes before and after sampling; it never calls a library
loader merely to obtain evidence. Triton preparation is explicitly unsupported
by this first ELF receipt slice, without changing that serving lane's admission.

The measurement command requires the externally sealed request SHA-256.
The producer reads one owned byte buffer, checks that digest and parses
that same buffer; a separate launcher hash followed by a second read is
not a sealed request boundary.


The original request and versioned worker job each have an independently checked
SHA-256 over the same owned bytes parsed by the child. The actual CPU phase's
argv, returncode, source origins, validator file and raw contract are bound into
the external validation result. Entry/exit software identities must agree and
CUDA must remain uninitialized. A native phase still independently validates
its installed reader and observes actual hardware before TP1/CUDA setup.
External CLI `check` requires `--request`, `--request-sha256` and a fresh
`--preflight-output`; it reruns the installed CPU validation before replaying
measurement evidence. A serialized success record alone is insufficient.

Passing `--expected-panel-sha256` and `--observation-out` additionally makes
`check` publish the versioned handoff `tessera.shape_time_observation.v1`. It
hashes one owned buffer of the panel and checks it against the externally
supplied digest, reruns the same actual installed CPU preflight and the same
`validate_external_panel`, and only then writes the observation durably
(`publish_json`, file then directory sync). The document binds the panel,
request, expected runtime, raw contract, every evidence and preflight
reference, the sealed original measurement producer identity, a distinct
replay-validator identity, and the actual CPU-preflight invocation it reran.
It carries the admitted scope, the observed lane and cell, the raw CUDA-event
samples and warmup count, the fixed claims, and the producer's own sampling
semantics: `sample_unit=single_apply`, one 2-D M-by-K operator apply that a
consumer may key at `batch_size=1` for the panel's M prompt rows. That
projection is not evidence of end-to-end batch-1 serving. The observation is
data transfer from the existing validator, not a second validator, runtime
contract or pin, and it is emitted only on the external-preflight path.

The authoritative replay of the recorded pilot keeps its original producer
closure: `--producer-root` points `check` at the original producer source tree,
so the sealed producer identity and the executed worker are that producer's,
while the observation records the replay tool's own identity separately. If
the original replay does not succeed against that closure, `check` refuses
rather than resealing history.
