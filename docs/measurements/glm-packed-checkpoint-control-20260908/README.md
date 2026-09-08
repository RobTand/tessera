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
