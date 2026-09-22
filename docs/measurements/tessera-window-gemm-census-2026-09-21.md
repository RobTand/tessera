# The fused window GEMM, served: earning the dense cells back (2026-09-21)

**Result.** The dense native window GEMM — `tessera::window_gemm_dense` /
`native_window_gemm`, the single launch `fp8_route.apply` and `bf16_route.apply`
have made since `1b767a207` — is **served** on the platform's own pinned image.
Four route censuses on a GB10 record **all 112 declared Tessera modules** on
that pair, in **both** the decode and the batch regime, in **both** residency
modes, for **both** dense families, with `verdict: served` and `problems: []`.

This is the receipt contract v31 asked for when it withdrew the eight dense
window-GEMV cells: *"A served census of the native window GEMM is what earns
cells back."* Contract **v34** mints four of them —
`tessera_e4m3_k1_dense_sm121_{decode,batch}` at `q256 = 1024` and
`tessera_bf16_k1_dense_sm121_{decode,batch}` at `q256 = 1792` — and the pair
leaves `scheme.EXPERIMENTAL_LAUNCHES`.

**Nothing wider is attested.** Eager execution only; grade `route_only` on all
four cells, because no KL arm was run; one rung per family; `sm_121` only; TP 1;
dense structure only. The A4 pairs and the compact window MoE adapter stay
experimental. Issue tessera#545.

---

## 1. What was served

| | |
|---|---|
| image | `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` — the `sm_121` platform entry's own `serve_image`; vLLM 0.28.0, torch 2.13.0+cu130, python 3.12.3 |
| device | `NVIDIA GB10`, compute capability `[12, 1]`, platform token `sm_121` (sparky) |
| tessera | `affae68ed` (master at the time of measurement), plugin entry point `tessera` present in the container |
| E4M3 artifact | `/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024` — Qwen3-0.6B, grid `E4M3`, `q256 = 1024`, body `WINDOW`, plane `CHANNEL`, 112 modules, family `TESSERA_FP8` |
| BF16 artifact | `/mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7-plugin` — Qwen3-0.6B, grid `BF16`, `q256 = 1792` (R = 7), body `WINDOW`, plane `CHANNEL`, 112 modules, family `TESSERA_BF16`; the same bytes the contract v5 receipt served (hardlinked `model.safetensors`) |
| driver | `tools/tessera_route_census.py` inside `experiments/tessera_plugin_run.sh`, under `experiments/serve_lock.sh` |
| forwards | a 64-token prefill and a one-row decode, the two regimes `lane_eligibility` declares |

Both artifacts predate the current master and were exported by earlier
encoders. They are read here as **bytes at a rung**, which is what a cell
attests: each one's `config_groups` scheme declares the family, grid, body,
plane and `q256` the contract's `attested_wire` stamps for that rung.

## 2. The four arms

Every arm ran eager, `--gpu-memory-utilization 0.3`, `--expect-modules 112`,
`--require-decoder native_window_gemm`, `--no-manifest-lanes`, with
`--runtime-image` cross-checked against the reference the launcher resolved from
docker's `RepoDigests` (issue #132).

| arm | family / rung | `TESSERA_SERVE_MODE` | decode phase | batch phase | verdict |
|---|---|---|---|---|---|
| `e4m3_k1_dense_q1024_resident_eager` | `TESSERA_E4M3_K1` q256 1024 | `resident` | 112 × `window_gemm_dense`/`native_window_gemm` | 112 × same | `served`, `problems: []` |
| `e4m3_k1_dense_q1024_streamed_eager` | `TESSERA_E4M3_K1` q256 1024 | `streamed` | 112 × same | 112 × same | `served`, `problems: []` |
| `bf16_k1_dense_q1792_resident_eager` | `TESSERA_BF16_K1` q256 1792 | `resident` | 112 × same | 112 × same | `served`, `problems: []` |
| `bf16_k1_dense_q1792_streamed_eager` | `TESSERA_BF16_K1` q256 1792 | `streamed` | 112 × same | 112 × same | `served`, `problems: []` |

Each record carries `state: served`, the family's own activation contract
(`fp8_per_token_dynamic` for `TESSERA_FP8`, `bf16_unquantized` for
`TESSERA_BF16`), and a `policy` equal to `<route>:<mode>`. `other_route_modules`
is 0 in every phase of every arm, and `tessera_modules_by_family` names one
family per arm.

### Receipts

Written to `/mnt/shared/tessera-runs/receipts/545-window-gemm-20260921/`, with
`SHA256SUMS` beside them.

| receipt | sha256 |
|---|---|
| `e4m3_k1_dense_q1024_resident_eager.json` | `5157e9a26b0dd28ad3ed665c09c6ec199c01b0650691fcea582e6ae6f761380b` |
| `e4m3_k1_dense_q1024_streamed_eager.json` | `0725eefa6fbdf6a7981d4520c5e70c7aaec4af24aa95c019f54f95c60929b69b` |
| `bf16_k1_dense_q1792_resident_eager.json` | `a4a2cf47d2acea917a402250db581d99f148791ef1835d4882eaed7ddd3fcb55` |
| `bf16_k1_dense_q1792_streamed_eager.json` | `8e6f6bcea309df3c203544f488d3924b6b7ed23c8edff9be8578810a502320dc` |

Those are box paths. A cell's `evidence.receipt` must be a repository path, so
this document — not the JSON — is what a cell could cite; these four cells cite
nothing, because `route_only` carries no receipt path.

## 3. The void-census guard, and why it is not a lane here

Issue #104's lesson is that per-module agreement passes vacuously when a regime
admits two launches: four censuses once recorded 112 of 112 modules refusing the
window-GEMV lane at load, each with `problems: []`, and the two arms of the
experiment were one lane state wearing two names. `--require-lane` exists for
that, and it **could not be used here**.

`census.lane_engagement` resolves a required lane through
`contract.lane_decoder`, which reads `native_extensions`. The one entry there is
`tessera_window_gemv` (decoder `window_gemv`) — the lane `1b767a207` retired
from the dispatch. The native window GEMM carries `lane: None` in
`scheme.ROUTE_LAUNCHES` because it is a **launch, not an extension lane**, so no
`--require-lane` value can name it. Worse, the E4M3 artifact's manifest still
stamps `requires_lanes: ["tessera_window_gemv"]` from its 2026-09-04 export,
which the census believes by default; left alone it would have refused all four
arms for a lane that no longer exists. `--no-manifest-lanes` set that stale
stamp aside, and the receipts therefore read
`lane_engagement.all_required_engaged: null` — the honest value for "no lane was
required", not a guard that was skipped.

The refusal was taken from the launch side instead. `--require-decoder
native_window_gemm` makes `required_decoder_coverage` refuse any phase in which
that decoder took zero modules, and each receipt's `decoder_coverage.phases`
reads `{"native_window_gemm": 112}` for both `decode` and `prefill`.

Independently of the flag, #104's ambiguity does not exist on this route.
`census_expected(..., include_experimental=True)` for dense `TESSERA_FP8` and
`TESSERA_BF16` is a **singleton**: the route makes one launch and raises rather
than falling back. There is no fallback for a void census to hide in.

## 4. What the receipts could not attest, and why the cells do not claim it

**Compiled execution.** `tools/tessera_route_census.py` states that compiled
dense launch agreement is unsupported: a compiled trace combines launches as
`a+b` for a graph serving every M, so a compiled dense record is counted
`unattested` and retains its exact record. Running a compiled arm would
therefore produce a receipt that cannot join a dense cell. The four cells scope
`execution_modes: ["eager"]` rather than claim a mode the instrument cannot
join. The contract v5 era published `["eager", "compiled"]` on dense BF16 cells;
that scope is not re-asserted here.

**Quality.** No KL arm was run. `derive_evidence_grade` reads `route_only` off
zero `kl` entries, which is the accurate statement: the dispatch is attested and
nothing here bounds KL on these rungs under this build. A dense KL against the
folded stock twin, in the shape `tessera-bf16-route-served-2026-09-02.md` used,
would raise the grade; it is owed, not claimed.

**Smoke.** `not_recorded`, `attribution: unattributed`, no control.

**Rungs, platforms, world size.** One rung per family (the rung each artifact
serves). `sm_121` only — `gfx1201` and `gfx1151` keep null `serve_image` and no
dense cells, and the gfx1201 BF16 cells v31 withdrew are **not** restored, since
no ROCm census of this launch exists. TP 1: a single-rank receipt reads
identically at any world size, so no world size is named.

**Agreement at measurement time.** All four receipts record
`cell_launch_agreement` as `unattested` 112/112, correctly: no cell covered the
pair when they ran. Re-running one arm against the v34 contract would turn that
into `covered_by_cell`; that closure is not part of this receipt set and is not
claimed.

## 5. What moved in the contract

* Four cells added to `lane_eligibility.cells`, all `device_qualified` /
  `backed_with_serve_flag` / `TESSERA_SERVE_MODE=resident|streamed`.
* `(WINDOW_GEMM_SYMBOL, _DECODER_NATIVE_WINDOW_GEMM)` removed from
  `scheme.EXPERIMENTAL_LAUNCHES`. This is the same change: `contract.
  _validate_cell_executes` derives `executes` from `route_launches` with
  `include_experimental=False`, so a cell naming an experimental pair is
  refused, and a pair removed without its cells would put an unattested launch
  in front of every contract reader.
* `contract_version` 33 → 34. `lane_eligibility.schema` stays `v10`; the
  activation-quantizer schema stays v2; no format row, `attested_wire` stamp,
  platform entry, routed cell, TP/EP bound or served byte moves.

Additive for a lane reader — four cells appear where a reader previously
resolved `unattested`. Not additive for a reader that derives a cell's
`executes` from `scheme.route_launches` itself.
