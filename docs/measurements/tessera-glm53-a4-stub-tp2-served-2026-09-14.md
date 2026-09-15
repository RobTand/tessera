# Two-rank GLM-5.3-Flash stub serve: routed E2M1x2 experts at q896 on two GB10s (2026-09-14)

**Status: closed for what it claims.** This is the receipt behind the two `routed_moe` cells that
contract v28 adds for `TESSERA_E2M1_K2`:

- `tessera_e2m1_k2_routed_moe_sm121_decode_resident`
- `tessera_e2m1_k2_routed_moe_sm121_batch_resident`

It records one eager serve of the GLM-5.3-Flash 4-layer stub, tensor-parallel over two DGX
Spark (GB10) boxes, resident, with Tessera's route telemetry on. It is a route census on
both ranks. It is not a KL, not a graph-mode serve, and not a full-model serve.

The timestamps in the logs are UTC and read 2026-09-15T02:28Z. The local date on the serving
boxes was 2026-09-14, which is the date in this file's name.

The findings:

1. **Both ranks served every Tessera module through the routes the cells publish.** The routed
   E2M1x2 experts executed `vllm.fused_moe.modular_kernel` over `torch_materialize_stock` under
   `e2m1_group16_ue4m3_static`. They ran at one row (the decode regime) and at 2 to 2048 rows
   (the batch regime). The runtime chose the `FLASHINFER_CUTLASS` NVFP4 MoE backend on both
   ranks.
2. **Every family was cut on at least one shard axis at a world of two.** Each per-rank shape
   is half the checkpoint's full dimension on the axis the vLLM layer class cuts. The dense
   E2M1_K2 modules were cut on both axes, E4M3_K1 on rows, and BF16_K1 on columns.
3. **The cells do not claim a smoke.** The serve's greedy check matched a BF16 recording on 29
   of 32 tokens, but a 4-layer stub's text degenerates at every precision, and the check is not
   in the record form `experiments/moe_greedy_smoke.py` defines. The cells publish
   `not_recorded`.

## Setup

| Item | Value |
|---|---|
| Tessera tree | `44d20d670700d719a50052e6050372fd5d9b9c2f` (master: #507 plus #505), mounted read-only into both containers |
| Checkpoint | GLM-5.3-Flash 4-layer stub, `glm53-4layer-a4-e2m1x2-q896-l2`: 4 layers, hidden size 4096, one dense layer (intermediate size 12288), 288 routed experts per MoE layer (moe intermediate size 2048, 8 per token) |
| Tessera modules | `layers.0.mlp.gate_up_proj` E4M3 q1024; `layers.0.mlp.down_proj` BF16 q1792; `layers.1.mlp.shared_experts.gate_up_proj` and `.down_proj` E2M1x2 q896; `layers.1.mlp.experts` E2M1x2 q896, routed. Layers 2 and 3 carry passthrough experts that stock vLLM serves. |
| Image | `localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb`: the pinned EUGR spark-vllm image with system NCCL 2.31.2 replaced by 2.30.7, identical on both boxes |
| vLLM | `0.28.1rc1.dev397+gfd4a15126.d20260904` (rank 0 log, engine init line) |
| torch | `2.13.0+cu130`, as `experiments/results/nvfp4_moe_route_load_probe.json` records for the same image digest; the rank logs do not print it |
| Ranks | rank 0 on sparky, rank 1 on sparklina; `world_size=2` in both rank logs |
| Collective fabric | NCCL `NET/IB` over RoCE on the direct Spark-to-Spark ports (`Using network IB` in both rank logs) |
| Serve flags | `--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp --enforce-eager --attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend flashinfer_cutlass --max-model-len 4096 --gpu-memory-utilization 0.25 --kv-cache-memory-bytes 4294967296` |
| Environment | `TESSERA_SERVE_MODE=resident`, `TESSERA_ROUTE_TRACE` set, `TESSERA_RESEARCH_GLM53_NOPE=1` |
| Memory watchdog | one per box, 16 GiB `MemAvailable` floor and PSI full at or above 20 |

`--attention-backend CUSTOM` and `TESSERA_RESEARCH_GLM53_NOPE=1` select the GLM-5.3 NoPE
attention backend, which is a property of this model and not of the routed MoE route. That
backend refuses graph mode by name (`glm53_nope.py`, tessera#508), which is why the cells name
`execution_modes: ["eager"]` only. A second run of the same command without `--enforce-eager`
failed on both ranks with `Tessera GLM53 NoPE is eager-only` before any weight loaded.

## What was served

Each rank wrote its own `tessera.route_trace/1` file. Both are committed byte for byte:

- `experiments/results/glm53_a4_stub_tp2_route_trace_rank0.json` (sparky, worker pid 502, 9
  flushes), sha256 `98b06898c63298dfc9c75cd05c6ad32f8cc003c2fecf78b793d1fe629db5ae53`
- `experiments/results/glm53_a4_stub_tp2_route_trace_rank1.json` (sparklina, worker pid 427, 8
  flushes), sha256 `92c116c073b238b060a0474e85b072bb241e1e90fed1e8e1e7fcd5dd31da4374`

The two files differ only in their header. Both hold the same 40 entries with the same counts:
five routes, each observed at M in {1, 2, 4, 5, 6, 178, 356, 2048}. Each route launched 94 times
at M1 and once at each other M, so 101 launches per route per rank. Every entry has `modules: 1`.

| Kind | Policy | Activation contract | Decoder | Symbol | Per-rank shape | Module and cut axis |
|---|---|---|---|---|---|---|
| moe | `TESSERA_NVFP4:resident` | `e2m1_group16_ue4m3_static` | `torch_materialize_stock` | `vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS` | N2048:K4096 | `layers.1.mlp.experts` |
| dense | `TESSERA_NVFP4:resident` | `e2m1_group16_ue4m3_static` | `native_span2` | `torch._scaled_mm` | N2048:K4096 | `layers.1.mlp.shared_experts.gate_up_proj`, row |
| dense | `TESSERA_NVFP4:resident` | `e2m1_group16_ue4m3_static` | `native_span2` | `torch._scaled_mm` | N4096:K1024 | `layers.1.mlp.shared_experts.down_proj`, column |
| dense | `TESSERA_FP8:resident` | `fp8_per_token_dynamic` | `torch_window` | `torch._scaled_mm` | N12288:K4096 | `layers.0.mlp.gate_up_proj`, row (full 24576 rows) |
| dense | `TESSERA_BF16:resident` | `bf16_unquantized` | `torch_window` | `torch.mm` | N4096:K6144 | `layers.0.mlp.down_proj`, column (full 12288 columns) |

The module column comes from the checkpoint's `quantization_config`, which declares exactly
these five Tessera modules, one per family, rung and shape. The axis column follows from the
layer class vLLM builds for each: a merged gate/up projection is column-parallel and gives a
rank its own rows; a down projection is row-parallel and gives a rank its own columns.

The routed symbol carries the backend that the runtime picked. A cell publishes the runtime entry
point, so `tests/test_route_census_regimes.py` removes the suffix through
`scheme.moe_census_symbol_base`, as the route census does. That test joins every entry in
both files to an sm_121 cell by family, structure, regime (from the entry's own M), residency,
activation contract and launch pair.

## Load, memory and power

These numbers come from the rank logs, a 1-second `MemAvailable` sampler on each box, and
Netdata. They are observations of one run, not profiled claims.

| Record | Rank 0 (sparky) | Rank 1 (sparklina) |
|---|---|---|
| Container | `35df681fe6df` | `6ff7c44f5322` |
| `Model loading took` | 17.88 GiB, 49.14 s | 17.88 GiB, 54.45 s |
| KV cache | 4.0 GiB reserved by `kv_cache_memory_bytes`; 733,184 tokens | same |
| `MemAvailable` baseline to minimum | 116,511 to 80,895 MiB (34.8 GiB drawn) | 119,545 to 85,477 MiB (33.3 GiB drawn) |
| PSI memory full avg10, maximum | 0 | 0 |
| GPU power, window mean and peak (140 W envelope) | 12.2 W, 69.7 W | 11.8 W, 67.2 W |

The serve reported ready 140 s after launch. It ran from 02:28:30Z to 02:31:42Z, and the watchdog
did not fire.

## Greedy check

After the serve was ready, one client sent three raw-completion prompts at temperature 0 for 32
tokens each. The summary is committed as `experiments/results/glm53_a4_stub_tp2_smoke.json`.

- For `The capital of France is`, 29 of 32 greedy tokens equal a BF16 recording of the same stub
  served at a world of two, with the first difference at token index 5.
- The other two prompts returned 32 tokens each with finish reason `length` and no error.
- The text of all three, and of the BF16 recording, is degenerate: a 4-layer stub is not
  coherent at any precision.

This check is agreement with BF16 on one prompt. It is not the per-(prompt, form) record that
`evidence.smoke.record` requires, so neither cell publishes a smoke word.

## What this receipt does not show

- **No KL.** No arm of this serve was scored against a reference. Both cells are `route_only`.
- **No graph mode.** The attention backend refuses it, so no compiled launch was observed.
- **One rung per family, and one module per dense family.** The E4M3_K1 dense module was cut on
  rows only and the BF16_K1 module on columns only.
- **The dense cells' image is a different image.** The dense sm_121 cells name the packaged
  vanilla vLLM pin. This serve ran the dense routes on another image, so it adds nothing to
  those cells; it shows only that the same routes executed at a world of two.
- **No KV-head replication.** The stub's attention has 64 heads and no `QKVParallelLinear`, so
  `num_kv_head_replicas` was never above 1.
- **Not the full model.** No 45-layer serve has run at any world size.
- **No world-size attestation in contract v28.** The traces are a two-rank census of all three
  families, but v28 leaves `tensor_parallel.units[].max_world_size` at 1 for every family.

## Sources

- Rank logs, outside the repository: `/home/rob/tmp/glm-a4-stub-serve/logs/m44e1.rank0.log`
  (sha256 `ee1ce836d1b4f7fcda6b2d870dc7acfa792edb0364cb16385e766f3b20f65c8d`; engine version at
  line 23, `world_size=2 rank=0` at line 25, `Using network IB` at line 53, the NVFP4 MoE
  backend at line 1654, model loading at line 1684) and `m44e1.rank1.log` (sha256
  `6255cf1f2457f94947101a7e63f3fe2517f7e3ce4634bdc55ce6ecc1c4b2c940`; the backend at line 1520,
  model loading at line 1535).
- Serve script: `/home/rob/tmp/glm-a4-stub-serve/run.sh`, run with
  `TS=/home/rob/tmp/ts-glm-tp2-44d20d670 MOE_BACKEND=flashinfer_cutlass TESSERA_SERVE_MODE=resident EAGER=1 GPU_UTIL=0.25 KV_BYTES=4294967296 WATCHDOG_GIB=16`.
- Campaign record: `/home/rob/dq-runs/glm-campaign-takeover-20260913/claude-takeover/a4/RUNTIME.md`,
  the section on the A4 stub serve on tree `44d20d670`.
