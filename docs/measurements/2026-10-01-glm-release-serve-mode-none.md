# GLM-5.3 release serve: compilation mode NONE (2026-10-01)

The GLM-5.3 T-8 release serve passes
`--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'`
with `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (tessera#774). Against the same serve
without `"mode":"NONE"`, served prefill is 6.4% faster at L8192 c1, and the
per-rank chunk profile attributes the gain to the elementwise kernels. This
page records the measurement, the numerics evidence and what the flag does not
cover.

## Why the flag is needed

With `enforce_eager=False` and breakable CUDA graphs off, vLLM keeps
`mode: VLLM_COMPILE` on GLM-5.3. The model is not torch-compiled on this image,
but the mode still resolves `custom_ops` to `['none']` and the `rms_norm` and
`fused_add_rms_norm` IR ops to `native`. Every custom op in the served prefill
then runs its eager fallback; the KDA output norm alone costs about 72 ms per
2048-token chunk. Mode NONE resolves `custom_ops ['all']` and
`rms_norm ['vllm_c', 'native']`, the resolution an eager serve uses. This is
graph cause 1 in
[the nightly graph-equivalence receipt](2026-09-30-glm-nightly-cells-and-graph-equivalence.md).

## Setup

| Item | Value |
|---|---|
| Artifact | A8S, `/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported` |
| Tessera | `bb088715`, `TESSERA_FUSED_E4M3_MMA=e4m3` |
| Image | `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705...` (vLLM `0.30.1rc1.dev336`) |
| Topology | TP 2 over RoCE, sparklina rank 0, sparky rank 1 |
| Serve | `max_num_seqs 4`, `max_model_len 8448`, `max_num_batched_tokens 2048`, MTP off, breakable graphs off |
| Client | host client, frozen prompt set v1, 5 trials per cell, medians |
| Default arm | A8SESH, window `u4-A8SESH-20260930T2132Z`: `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| Mode NONE arm | A8SESHMN, window `u4-R1-20261001T0058Z`: adds `"mode":"NONE"`; nothing else changes |

## Served speed

| Cell | TTFT default | TTFT NONE | Prefill tok/s default | Prefill tok/s NONE | Decode tok/s default | Decode tok/s NONE |
|---|---|---|---|---|---|---|
| L512 c1 | 651.8 ms | 632.9 ms (-2.9%) | 785.4 | 809.0 (+3.0%) | 14.42 | 14.55 |
| L512 c4 | 1717.2 ms | 1820.6 ms (+6.0%) | 942.8 | 1119.3 (+18.7%) | 38.07 | 37.27 |
| L2048 c1 | 1336.9 ms | 1256.2 ms (-6.0%) | 1531.9 | 1630.3 (+6.4%) | 14.40 | 14.52 |
| L2048 c4 | 4660.6 ms | 4346.8 ms (-6.7%) | 1498.2 | 1602.2 (+6.9%) | 30.51 | 30.95 |
| L8192 c1 | 5256.3 ms | 4941.6 ms (-6.0%) | 1558.5 | 1657.8 (+6.4%) | 14.40 | 14.50 |
| L8192 c4 | 14470.6 ms | 13643.4 ms (-5.7%) | 1539.4 | 1630.2 (+5.9%) | 17.87 | 18.40 |

At L512 c4 the median TTFT rose while the median prefill rate rose 18.7%. The
two medians disagree, and the cause is not established; treat that cell as
unresolved.

Energy at L8192 c1: prefill power per box 74.5 W and 74.7 W of the 140 W
envelope, and 0.0956 J and 0.0901 J per input token (-5.8%). Netdata's
`nvidia_smi.gpu_power_draw` for the cell, mean and maximum per box: sparky
62.3 W / 84 W and 67.1 W / 84 W; sparklina 62.9 W / 78 W and 58.2 W / 76 W.

## Per-rank chunk profile

`torch.profiler` in the latency serve, one prefill step per row: a
2048-token chunk at L8192 (mean of four) and the whole 512-token prompt at
L512 (one). Rank 0's components; the table lists every component that moved
by at least 1 ms at either length.

| Component | L8192 default | L8192 NONE | L512 default | L512 NONE |
|---|---|---|---|---|
| Unfused elementwise | 97.3 ms | 29.0 ms | 15.9 ms | 5.2 ms |
| NCCL all-reduce | 91.7 ms | 76.9 ms | 38.2 ms | 41.0 ms |
| Norm and activation | 11.7 ms | 8.2 ms | 1.9 ms | 1.3 ms |
| BF16 projection GEMM | 174.9 ms | 171.9 ms | 55.3 ms | 55.5 ms |
| Tessera routed MoE | 554.0 ms | 556.4 ms | 423.6 ms | 417.3 ms |
| KDA recurrence and conv | 50.7 ms | 52.3 ms | 11.1 ms | 11.1 ms |
| Chunk wall, rank 0 | 1310.5 ms | 1223.4 ms | 629.1 ms | 612.9 ms |
| Chunk wall, rank 1 | 1310.3 ms | 1223.3 ms | 628.1 ms | 611.3 ms |

The mHC kernels are unchanged (125.1 ms and 124.3 ms at L8192): they are
TileLang and DeepGEMM ops, not custom ops.

## Numerics

- **The TR3 panel is unchanged.** Full-vocabulary KL against the teacher over
  the 25 windows (51,175 positions) is 0.027885896312391557 in both windows,
  and every domain mean is identical. The TR3 serve is eager in both arms, so
  this shows that the scorer and its serve did not move. It does not score the
  graph serve.
- **What carries the claim to the graph serve.** Mode NONE gives the graph
  serve the eager scorer's operator resolution. On u1 stub B, sbG4 measured
  that resolution as the whole of cause 1, and with `mode: NONE` (sbG3) the
  prefill token of every choice equals eager's. Under FULL_DECODE_ONLY the
  prefill runs outside the captured graphs.
- **The graph serve's own prefill logits were not scored.** A compiled TR3
  under this configuration is the direct receipt
  (PrismaQuant's `measure_glm_tr3_vllm.py --execution-mode compiled`).

## What the flag does not cover

- **Decode under graphs.** FULL_DECODE_ONLY captures at
  `max_seq_len = max_model_len`, which freezes the GLM indexer's long-context
  branch. On this image a graph serve is eager-equivalent only at
  `max_model_len <= 2048`; the release serve runs 8448. This is cause 2 in
  [the nightly graph-equivalence receipt](2026-09-30-glm-nightly-cells-and-graph-equivalence.md)
  (tessera#702), and mode NONE does not change it.
- **The MTP drafter under graphs** (tessera#695).

## Data

- Speed, energy and TR3: `/mnt/shared/tessera-measurements/glm-pact-u4-20260927/results/A8SESH-nightly-20260930/run`
  and `.../A8SESHMN-nightly-20260930/run`.
- Profiles: `components-A8SESH-L{512,8192}.json` in
  `/mnt/shared/tessera-measurements/t8r-speed-20260929/l512-20260930/analysis/`,
  and `components-A8SESHMN-L{512,8192}.json` in
  `/mnt/shared/tessera-measurements/mhc-elementwise-20260930/r1/`.
