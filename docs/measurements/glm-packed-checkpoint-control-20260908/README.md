# GLM packed checkpoint reconstruction control — prepared 2026-09-08

This extends the existing native07 harness for the checkpoint bridge in #430.
The tested implementation is `6c0e77e66e`, based on exact
`f3b6953f0c5821ab2108b6db9942f9a2e0179e91`. No `src/tessera` files, encoder,
packaged cells or pins changed. The original bridge producer's separate
record-parser correction remains a dependency of native source freeze.

## Control behavior and scope

`experiments/glm_packed_moe_control.py` accepts explicit
`configuration_source: checkpoint_json` on its packed arm. It reads the
checkpoint config bytes once, verifies the requested SHA256, and compares the
JSON quantization declaration with the bound original fixture's scheme,
source target, ignore set and explicit research settings. The ordinary plugin
registers through stock vLLM, its registry must return exactly `TesseraConfig`,
and that class reconstructs the config through `from_config`. The stock model
constructor must retain the ordinary config and parsed research object.
The existing Python subclass remains confined to the original Python mode.

A new `checkpoint-reconstruction.json` records the exact config binding,
registered class/key, research object and canonical quantization digest.
Existing native create/load/finalization, independent stock parity, selected
tile/scale, package origin and complete stock-core checks remain in place.
The packed owner must additionally be in `loading` after creation and `ready`
after finalization. TP2's existing trained-gate, shared-expert and one-final-
reduction controls remain conditional on TP2; the initial TP1 request does not
claim to execute them. No generation control was present in this bounded
harness, and this change does not introduce or claim full text generation.

This establishes a route-reconstruction control, not a full checkpoint export
or whole-engine automatic sidecar-discovery test: the harness parses the JSON
quantization block and passes its registered ordinary config into the existing
stock constructor. The original four-layer configuration constructs on meta;
one real GLM MoE owner is loaded on CUDA. The original q512 role wires repeat
one trained expert's projection over 288 expert identities. This fixture is
not 288 independently trained experts or six full models.

## CPU regression and validation

PrismaBuild regression `fb8100df8dc3` failed all ten new cases against the
unchanged harness. The pre-fix line was:

```
ImportError: cannot import name 'checkpoint_quant_config' from 'experiments.glm_packed_moe_control'
```

The positive tests reconstruct through the ordinary class under clearly
stubbed CPU vLLM registration and verify the actual shared dispatch receives
the parsed research object for TP1 and TP2. Negative cases reject a changed
file hash, missing/null research block, wrong TP/backend/chunk, source target
or wire stride before registration. These CPU stubs are not native evidence.

The local impacted selector returned `narrowed`: 57 test files. The existing
selected-owner lifecycle file was added, giving 58 files. PB distributed them
into 20 actions, each with two pytest workers, worksteal distribution, one
native thread per worker, 2 CPUs, 5 GiB and a 300-second action deadline.
The published client and exact arguments are in `cpu-tests-command.json`.
All actions completed on dl380g10: **1509 passed, 71 skipped, zero failures or
errors, zero uncollected modules**. Device: **torch 2.10.0+cpu reports no CUDA
device**; no test allocated on CUDA. This is not CUDA-surface coverage.

Verbatim skip reasons and counts are retained in `cpu-audit.json`: 51 encoder
GPU cases, one Triton-only import, one CUDA extension positive arm and 18
unavailable historical box-artifact cases. Those artifacts name `TESSERA_RUNS_DIR`,
`KL_TOOL_DIR`, `TESSERA_PRISMAQUANT_DIR` or `TESSERA_PRISMAQUANT_WORKTREE` in their
recorded reasons. No required CLI regression test was skipped.

Compile action `23ae2b2b3356` exited zero for the modified harness/test modules.
The audit checks the 20 green actions, compile action and expected red
regression against actual terminal return codes, sanitized completed cleanup,
canonical receipt/producer digests, actual CAS payload bytes/size and
`pb_verify_claim` results. All success claims have `checks_passed: true`;
that API's `attestation_verified: null` is retained without claiming the
additional full-manifest attestation check. Every source bundle was hashed and
fetched; green snapshots differ from the tested commit only by their generated
PB closure. The regression snapshot differs from base only by the new tests
and its closure. All 20 controller populations report the same verified source
identity, with no source mutation during testing.

## Draft native invocation — no launch yet

`prepared-invocation-draft.json` binds the draft request and checkpoint config.
The checkpoint config SHA256 is
`2e860568c5a5a664291c554d1185e250c1f792273126146306931274169e8fca`.
The shared copies are under
`/mnt/shared/tessera-measurements/glm-packed-checkpoint-control-20260908/`.

The proposed initial control runs on sparklina in the existing exact stock
vLLM image, with 4 CPUs, native threads bounded to one, 48 GiB, a 600-second
launcher deadline and no network. It reuses native07's original wire/stock
producer manifests and saved inputs. All 864 projection wires load, but only
saved decode and empty cases execute; `measure: false` omits timing loops and
performance profiles. The memory limit reuses the prior TP1 control limit and
is not a measured full-model capacity claim. Root must inspect current host
load and select the allowed affinity immediately before any launch.

Source archive/commit remain explicitly null in the draft request, and no
package or harness archive has been frozen. After the parser correction is
reviewed, root can integrate this control, freeze exact source/harness bytes,
and review a completed manifest before authorizing the GPU launch. No TP2
launch is prepared or authorized by this initial invocation. Any future TP2
scope needs both-box approval and retains native07's shared-input and stock
runner predicates.

The manifest lists exact acceptance predicates: successful owned-container
exit and cleanup, unchanged stock core and controls, authenticated installed
source/package origins, ordinary JSON-selected config through construction,
wire-only loading followed by the ready packed owner, no persistent full FP8
expert pool or retained stock kernel/config, and exact native output/selected
tiles/scales against the independent stock fixture. Retain both-Spark Netdata
for the actual launch interval and the harness's synchronized owner/load memory
observations. No timing, work-per-joule, fit, generation, quality or production
qualification claim follows from this proposed reconstruction control.

## Completed native control — 2026-09-08, supersedes the preparation status above

The coordinator reviewed the strict-carrier correction and froze the native
runtime archive from exact `07ad344c3275bb2fa7ce2432f93d89945d66f4c2`.
The control harness remained separately frozen at
`6c0e77e66e3d70d7dd81f854e4bfba4cf9fe024d`; it was not folded into the producer
source package used for pricing/export. Both archive member sets, file modes
and bytes were independently compared with all 1,179 Git blobs in their
respective commits. Native07's six installer-support files retained their
original hashes. The final input and invocation are
`request-tp1-01.json` and `prepared-invocation-launch-01.json`.

Root corrected the provisional CPU choice: sparklina cores 0–3 are efficiency
cores. Final execution used performance cores **15–18**, each reporting
capacity 1017. A concurrent external validation container initially held cores
15–16. The launch waited until its exact PB action `c40e2deac94e` terminated
and completed resource cleanup; its failed validation is separate from this
control. The final fresh host check found no containers or GPU compute apps
and 0–1% per-core activity across the selected mask. The recorded `taskset`
mask propagated to Docker's exact `15,16,17,18` cpuset. The original capture
continued independently on sparky; no native workload was launched there.

Sealed launch plan SHA256:
`a4594a552c099a011a296b23d4ecdc1ed8f363bfbb0a815584c9006ad95af3a1`.
Request SHA256:
`c0649bced44c2d73c28c3600ab72166be1b06bd640affb567de74dedcb265464`.
The exact SSH command was recorded in `launch-command-01.json` before start.
The existing direct vLLM launcher used the reviewed stock ARM image, 48 GiB,
no network and a 600-second deadline. No timing loop or follow-up GPU action
was run.

Actual remote, launcher and container exit codes are all **0**. No timeout or
OOM occurred; the owned container was removed. The final native receipt is
`native-tp1-01/receipt.json`, SHA256
`5b3224b9ceb596d34227abec670ff28c9e9d5abdb0036d2d5077abf339217719`.
It establishes:

- The bound checkpoint JSON reconstructed the ordinary registered
  `tessera.serving.config.TesseraConfig`, with Triton decoding, TP1 and chunk
  bound 8. The actual stock GLM factory retained this ordinary config.
- All 864 projections loaded. Stock loader return values name the destination
  parameters, so the roster contains 576 `w13_wire` and 288 `w2_wire` entries;
  complete per-expert/per-role coverage is enforced by native finalization.
- Wire-only expert parameters transitioned through the guarded loading/ready
  lifecycle to the packed owner. After preparation only the shared routing
  bias remained as a parameter; the packed owner reported 1,853,603,840 tensor
  bytes. No persistent full FP8 expert pool or stock kernel/config was retained.
- Decode and empty-input output are finite and exactly equal to independent
  stock FP8, with exact selected tiles/scales. All six output-file input
  tensors are byte-identical to their original saved-input counterparts.
- All 4,967 stock vLLM core files stayed unchanged. All 70 installed production
  source files match the reviewed runtime archive, and all 47 loaded Tessera
  modules' file/spec origins resolve to those exact bytes.

The archive intentionally contains five `tessera._dev` source files that the
archived `pyproject.toml` explicitly excludes from the installed package through
`exclude = ["tessera._dev*"]`. The audit derives this packaging exclusion and
records those five exact files. The installed wheel's independently recomputed
source seal is
`7d1e3b011eaba8d3a779656e4e3cfda9d9c3db41b7cf7d35aedb3d8b094f99df`.
That is an installed serving-package identity, not a replacement for the raw
producer checkout's source seal; no priced-wire receipt is relabelled.

`native-tp1-01-audit.json` binds the complete native artifact roster and the
checks above. The coordinator independently verified the source, inputs,
checkpoint, owner, output checks and launcher roster in
`root-native-tp1-artifact-source-audit.json`. Both-Spark Netdata CPU, RAM, swap-I/O
and board-power series cover the actual launch interval, with 64 samples per
chart and no missing context. `netdata.json` is supplemental post-launch
telemetry, so its hash is bound separately from the launcher's 16 final
artifacts. Synchronized in-process create/load/prepare allocator observations
remain in the native receipt; no timing or performance delta is claimed.

The result is the bounded JSON reconstruction gate described above. It does
not establish automatic full-engine checkpoint discovery, complete checkpoint
load, text generation, independently trained expert diversity, trained routing
or shared-expert execution in this TP1 control, TP2 collectives/fit, whole-model
quality, prefill performance or production qualification. Issue #430 remains
open for those full-engine gates. The harness PR merges current master
`396b944cb320c2458e22e870fef435e9d8ac6510`, retaining the reviewed test split and
strict-carrier fixes; the measured native runtime archive remains exact 07ad.
