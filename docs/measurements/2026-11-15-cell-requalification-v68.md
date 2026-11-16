# Eight stub route cells, contract v68

Issue: tessera#1683. Measured code: `5c2ef6deec3e354236f83616935faa7c14028737`.
Publication base: `937eaeccf1c76103142db48d60466ff97a620413`.
The complete `src/tessera` trees at those two commits are identical,
Git tree `ea9f9073412a56868305cca1d72cc3591a08f736`.
Issue #1383 requires affected requalification when executable behavior changes.
This mint changes qualification rows, not dispatch or kernel behavior.
The receipts and scope file retain the measured source identity. They do not claim a new GPU measurement of publication HEAD.

## Measured scope

| Receipt | Image suffix | Stub | Modules per phase | Dense decoder | Routed decoder | Census rungs |
|---|---|---|---|---|---|---|
| `glm53_requal_v68_e2m1_census.json` | IMG-G `f8dbe1a0` | Original source layers 0-3 | 3 | `native_fused_window_dense_e2m1`, 2 modules | `native_routed_fused_window_e2m1`, 1 stack | 896 |
| `glm53_requal_v68_bf16_census.json` | IMG-N `5be13705` | Original B, 8 layers | 9 | `native_fused_window_dense_bf16`, 8 modules | `native_routed_window_classes_bf16`, layer 7 | Dense 832/880/960/1024/1088, routed 1024 |

Both receipts record `served`, `problems: []`, TP1, eager, resident,
SM121, decode M1 and prefill M64. Before the mint every module was unattested.
R2 job `ƒ2isM6n5MtK` exited 0. R3 run 3 also exited 0.
Their result verdicts, stderr and census records were read before this mint.
The receipts and exact checkpoint configs are committed under `experiments/results/`.
`glm53_requal_v68_scope.json` records their SHA-256 values, config digests,
source digest and measured commit. The replay checks the exact receipt bytes,
config binding, module counts, layer counts, rungs, runtime and launch pairs.

IMG-G is `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
It has vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904` and torch `2.13.0+cu130`.
IMG-N is `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`.
It has vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929` and torch `2.13.0+cu130`.
Runtime kernel-build names retain the legacy toolchain convention.
They are not hashes of compiled shared libraries.
The package source digest is `826984285ef1deeb78e786861c3c2306d9477b07a5fa20d7644fb87b92a8e3c4`,
under `tessera.package_source.v1`. This digest was calculated from the measured source clone.
The census versions block supplies the commit, not that calculated digest.

IMG-G uses the research GLM NoPE CUSTOM backend. IMG-N uses stock attention,
with the research NoPE plugin off. Both use `fp8_ds_mla` KV, triton MoE,
FlashInfer autotune off and trust-remote-code. Full details remain in each receipt.

The E2M1 artifact has 867 units, 3 dense roles and 864 routed roles.
The BF16 artifact reuses 876 exact B cache units, 12 dense roles and 864 routed roles.
Every other B module is source-precision passthrough.
The BF16-only census excludes the background E4M3 qualification mismatch in #1699.
It does not qualify or repair those E4M3 cells.

## Mint and limits

Option S adds four `_SCOPED_EXECUTION_RECEIPTS` rows.
Each binds platform, family, structure, both regimes, resident mode, exact image,
eager execution, census rungs and one observed pair.
The v66 precedent is `06f53994`; its source receipt predates the mint.
The selector does not open the receipt or verify its commit or kernel build.
The replay supplies byte and config checks; the receipts and scope file retain measured source provenance.

Mint four E2M1 cells on IMG-G and four BF16 cells on IMG-N,
dense and routed, decode and batch. BF16 dense rungs: 5 measured (832, 880, 960, 1024, 1088); 769-1279 admitted by the run-table rule (v52).
Historical folded and TCQ records retain their old identities and cannot qualify current pairs.
Historical E4M3 replay tests retain their original scopes. Current BF16 replay has its own test.
The current experimental execution registry is unchanged.
The scope does not add an E2M1 cell on IMG-N or a BF16 cell on IMG-G.

All eight cells are `device_qualified`, `route_only`, resident and eager.
Smoke stays `not_recorded`. Decode cells retain the required open graph-gap record;
batch cells carry no KL record. No full-model quality, timing, compiled execution,
TP2, streamed residency, pin or default change is claimed.

The initial mint left the E2M1 activation quantizer entry absent.
R4 job `ƒ2j483KUNKZ` then ran at mint commit `3d5799c7156fc7d780d25b2e142f98474a8d5228`.
Its tool exit status was 0 and its verdict was `attested`.
Receipt: `/mnt/shared/flux-jobs/tessera-option-c-r4-v68-3d5799c7-20261115T230414Z/flux/activation-quantizers-3d5799c7.json`.
File SHA-256: `fb2904a0397db6c6a5e1213ea8982a01f6fae1a715800043f1cf34be4f0bc987`.
The exact bytes are committed as `experiments/results/glm53_requal_v68_activation_quantizers.json`.
The follow-up puts the unchanged generated entry in the schema's sm_121 image list.
It records IMG-G, NVIDIA GB10, native `torch.ops._C.scaled_fp4_quant`, and 11 vectors.
No probe value, scale, code, generator digest or runtime field was edited.
The v39-pattern test now requires this exact receipt for the current E2M1 cells.
Publication still requires the prepared CPU and remote validation jobs.

## Validation state

CPU jobs ƒ2j76ARLESj and ƒ2jBAYELkxP passed BUILD --check, but pytest failed
5 and 6 cases, respectively. A dependency-scoped CPU run is pending this fix.
Self-hosted bbrun follows under the lead's control. No GPU measurement is added.
