# Every priced GLM rung on the GLM serving image: eight TP1 eager censuses (2026-09-26)

**Result.** Eight route censuses of eight-layer GLM-5.3-Flash stubs, served
on the GLM serving image, recorded **all 21** declared Tessera modules of each
stub as served, in **both** the decode and the batch regime, with
`verdict: served` and `problems: []`. Together with the v38 receipt
(`tessera-glm-x-census-2026-09-26.md`) they cover these rungs:

| family | structure | rungs (`q256`) | launch |
|---|---|---|---|
| `TESSERA_E4M3_K1` | dense | 832, 864, 896, 928, 960, 1024, 1088 | `tessera::window_gemm_dense` / `native_window_gemm` |
| `TESSERA_BF16_K1` | dense | 832, 864, 880, 896, 928, 960, 1024, 1088 | `tessera::window_gemm_dense` / `native_window_gemm_folded` |
| `TESSERA_E2M1_K2` | dense | 896 | `tessera.kernel_a4.a4_span2_gemm` / `native_span2_gemm` |
| `TESSERA_E4M3_K1` | routed_moe | 832, 864, 896, 928, 944, 960, 1024, 1088 | `tessera.native_window_moe.NativeWindowMoE.__call__` / `native_window_moe_compact` |
| `TESSERA_BF16_K1` | routed_moe | 1024 | `tessera.native_window_moe.NativeWindowMoE.__call__` / `native_window_moe_compact_folded` |
| `TESSERA_E2M1_K2` | routed_moe | 896 | `tessera.kernel_a4.a4_span2_grouped_gemm` / `native_span2_grouped` |

Contract **v39** mints the four E2M1 cells
`tessera_e2m1_k2_{dense,routed_moe}_sm121_{decode,batch}_resident` and widens
the six E4M3/BF16 cells v38 minted on this image to the rungs above. Both A4
launch pairs leave `scheme.EXPERIMENTAL_LAUNCHES`. Issue tessera#604 (second
half).

The same change **withdraws** four E2M1 cells whose launches this build cannot
make: the two routed cells on image `a5424378` that named
`(vllm.fused_moe.modular_kernel, torch_materialize_stock)`, and the two dense
cells on the serve-image pin that named `(torch._scaled_mm, native_span2)`.
Both launch rows leave `scheme.ROUTE_LAUNCHES`. The receipt for the first E2M1
routed stub served on this build is stub D below: the grouped native route,
not the materialising one.

**Nothing wider is attested.** Every census ran eager, TP 1, resident. Every
cell is grade `route_only` (no KL arm was run) and smoke is `not_recorded`. No
timing or throughput is claimed. Shared-expert blocks are dense Linears in the
contract's grammar, so they join the dense cells; "shared" is not a structure.

---

## 1. What was served

| | |
|---|---|
| image | `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`, vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, torch `2.13.0+cu130`, python 3.12.3 |
| device | `NVIDIA GB10`, compute capability `[12, 1]`, platform token `sm_121` (sparky) |
| tessera | `50e67a9d0`, the first commit of this change (the routed withdrawal). The contract edits after it move no route code. |
| serve | the v38 settings: `TESSERA_SERVE_MODE=resident`, `TESSERA_RESEARCH_GLM53_NOPE=1`, `--attention-backend CUSTOM`, `--kv-cache-dtype fp8_ds_mla`, `--moe-backend triton`, `--kernel-config '{"enable_flashinfer_autotune": false}'`, `--trust-remote-code`, eager, `--gpu-memory-utilization 0.45`, `--kv-cache-memory-bytes 4294967296`, `--max-model-len 4096`, `--expect-modules 21` |

### The stubs

Each stub is `GLM-5.3-Flash-BF16` truncated to its first eight layers: layers
0-2 are dense MLPs and layers 3-7 are MoE layers with 288 routed experts and one
shared-expert block. Every tensor kept is byte-identical to the source
checkpoint's. The config keeps the first eight entries of each per-layer list
and sets `num_nextn_predict_layers` to 0, so no MTP layer is built. Attention
Linears, routers, embeddings and the head stay at source precision.

**No wire was encoded for these censuses.** Every Tessera unit is a wire the
union encode campaign (`glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01`)
had already written, taken with its own receipt: routed units from that
campaign's `cost.pkl` `tessera_expert_wires`, dense and shared units from its
per-unit checkpoint journals. The exporter's cached-unit intake re-verified
each unit against the stub's source tensor, its committed calibration Hessian
and the original encoder package (source sha256 `a4c92094…`), and hard links
carried the wire bytes. Every wire's recipe equals `served_recipe` at its rung
and structure: `WINDOW` / `CHANNEL` / span 1 / `window_bits 14` for E4M3 and
BF16 (BF16 `channel_sigma 1.0`), and `TCQ` / `LUT16` / span 2 for E2M1.

| stub | dense MLP (gate_up, down) L0-L2 | shared (gate_up, down) L3-L7 | routed L3-L7 |
|---|---|---|---|
| A | (B832, E832), (B960, E960), (B1024, E1024) | (E1088, B1088), (E832, B832), (E960, B960), (E1024, B1024), (B1088, **B864**) | E944, E864, E832, E960, B1024 |
| B | (E832, B832), (E960, B960), (E1088, B1088) | (B832, E832), (B960, E960), (B1024, E1024), (B1088, E1088), (E1024, **B880**) | E928, E896, E1024, E1088, B1024 |
| D | all E2M1 896 | all E2M1 896 | all E2M1 896 |
| S1-S5 | as A | as A, except L7 down: **B896**, **B928**, **E864**, **E896**, **E928** | all E832 |

`E` is `TESSERA_E4M3_K1`, `B` is `TESSERA_BF16_K1`. Each bold rung exists in
the union campaign only as layer 7's shared-expert `down_proj`, so that one
module is its whole receipt. The same holds for each routed E4M3 rung other
than 832: one routed stack carries it.

### The receipts

Each receipt is committed byte for byte as
`experiments/results/glm53_u1_stub_<stub>_tp1_eager_census.json` beside its
`glm53_u1_stub_<stub>_config.json`. The receipt's `checkpoint_sidecars` records
the config's digest.

| stub | census wall clock (UTC) | receipt sha256 | config sha256 | decoder coverage per phase |
|---|---|---|---|---|
| A | 23:35:16-23:38:55 | `51210678…` | `1992ba21…` | window_gemm 7, window_gemm_folded 9, moe_compact 4, moe_compact_folded 1 |
| B | 23:39:08-23:42:00 | `1de904a5…` | `499bbad8…` | window_gemm 8, window_gemm_folded 8, moe_compact 4, moe_compact_folded 1 |
| S1 | 23:42:00-23:44:31 | `ac37c0ad…` | `fa7bf0d0…` | window_gemm 7, window_gemm_folded 9, moe_compact 5 |
| S2 | 23:44:31-23:47:13 | `f21480e4…` | `5d9cffb8…` | window_gemm 7, window_gemm_folded 9, moe_compact 5 |
| S3 | 23:47:13-23:49:44 | `5b574f67…` | `8f6a1847…` | window_gemm 8, window_gemm_folded 8, moe_compact 5 |
| S4 | 23:49:44-23:52:13 | `c80b789a…` | `f069816d…` | window_gemm 8, window_gemm_folded 8, moe_compact 5 |
| S5 | 23:52:13-23:54:44 | `bba21efa…` | `a3a6afa7…` | window_gemm 8, window_gemm_folded 8, moe_compact 5 |
| D | 23:54:44-23:58:04 | `11c6c2a3…` | `425ddb41…` | span2_gemm 16, span2_grouped 5 |

All eight: `verdict: served`, `problems: []`, 21 modules per phase. In stub D
every record carries activation contract `e2m1_group16_ue4m3_static`.

## 2. The activation quantizer table on this image

An E2M1 cell requires its image to carry an `activation_quantizers` entry.
`experiments/attest_activation_quantizer.py emit --platform sm_121` ran inside
this image on sparky's GB10. Its 11 vectors are byte-identical to the table
generated on `a5424378`, which `verify` against that entry confirmed in the
same run. The entry is appended to `activation_quantizers.platforms.sm_121`;
the `a5424378` entry stays, since it is a measurement of that image.

## 3. What is not attested

These rungs are priced by the joint cost table or listed by the contract but
have no cell on this image, because the union and extension encode caches hold
no complete wire set for them and no new encode was in scope:

- routed `TESSERA_E4M3_K1` 880 and 912: every layer's stack is incomplete in the
  caches;
- routed `TESSERA_BF16_K1` other than 1024;
- dense and shared `TESSERA_E4M3_K1` 880, 912 and 944;
- `TESSERA_BF16_K1` 1792, in any structure;
- `TESSERA_E2M1_K2` 128 through 768, in any structure.

The withdrawal also removes these attestations, with nothing to replace them:
E2M1 dense on the serve-image pin, E2M1 dense at streamed residency, E2M1 in
compiled mode, and routed E2M1 at 128 through 768.

## 4. Reproduce

The stub source, the bundle manifests and the eight exports are under
`/mnt/shared/tessera-runs/moe/u1-stubs-20260926/`. The exports ran as
PrismaBuild actions `5e7b6ccd…` (A), `60472fd0…` (B), `448533dd…` (D),
`489c4c50…` (S1), `975369d7…` (S2), `4e3644fa…` (S3), `66fb559c…` (S4) and
`28b9f39d…` (S5), each `experiments/export_tessera_serving.py --device cpu
--cached-units … --cached-hessian-identity committed` against the union
campaign's Hessian references and producer package. `tests/test_glm_u1_census_cells.py`
replays each receipt against the packaged table.
