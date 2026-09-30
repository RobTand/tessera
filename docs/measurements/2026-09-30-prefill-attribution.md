# GLM-5.3-Flash T8R prefill: where the L8192 remainder goes

**Status:** measured attribution; no serving code changes. The 09-29
attribution left about 847 ms per 2048-token prefill chunk that no Linear
kernel explained. About 90% of it now has a name:

- 36% is the routed kernel re-decoding experts under a real routing;
- 46% is attention, mHC and glue work the Linear bench never ran;
- 9% is the TP2 all-reduce.

The last 10% is unmeasured. A decode-to-scratch plus grouped-GEMM routed path
cannot beat the fused window kernel below about 4,200 to 5,500 tokens per
step. The serve steps at 2,048 tokens, so that path was not built.

## Setup

- **Model:** the GLM-5.3-Flash Tessera-8 release,
  `pact-e4m3-accuracy-20260928/release-t8/exported`.
  - 45 layers: 34 KDA and 11 sparse MLA.
  - 42 routed MoE layers: 22 at R1024, 17 at R1088 and 3 at R832.
- **Image:** `spark-vllm-nccl230@sha256:f8dbe1a0`, vLLM
  `0.28.1rc1.dev397`.
- **Served frame.** The served TP2 cells are relaunch 4, which ran Tessera
  `83460680`. Its routed kernel source (`routed_fused_window.cu`,
  `65e05fdd`) is byte-identical to `b40c93cb`, the base of the 09-29
  bench. The remainder is served time minus that bench's balanced-routing
  Linear sum:

  | Cell | Remainder |
  |---|---|
  | L512 | 99.7 ms |
  | L2048 | 821 ms |
  | L8192 | 3,390 ms (847 ms per chunk) |

- **TP1 trace.** A single-box TP1 serve of the first 12 layers ran on
  sparky, on Tessera master `b5689dc049`.
  - Layers 0 to 11 are 9 KDA and 3 MLA layers, and 3 dense and 9 MoE layers.
    The loader was restricted to those layers.
  - `torch.profiler` recorded one L8192 request (four M = 2048 steps) and one
    L512 request after warm-up.
  - Inside every step the GPU was busy 99.0 to 99.9% of the time (idle gaps
    of 0.9 to 6.5 ms per 681 to 701 ms step). At TP1 no host or allocator
    gap is left to find.
- **Recorded routing.** The same serve recorded every routed call's top-k ids
  for one L8192 request (36 calls) and six L512 requests (54 calls).
  `bench_t8r.py --routing` replays them through the real expert stacks at
  TP2 rank-0 shapes on one GB10. The replays ran on the served kernel
  (`b40c93cb` source) and on master. Consecutive chunks of one layer are
  concatenated for M = 4096 and 8192.
- **Receipts:** `/mnt/shared/tessera-measurements/t8r-speed-20260929/`
  - `skew-b40c-20260930T012957Z`
  - `skew-20260930T012953Z`
  - `bigm-20260930T013246Z`
  - `scratch-20260930T013246Z`
  - The recorded ids are in `prefill-routing-20260930/`.

## Routing skew

A balanced routing gives every expert about 57 of the 16,384 routes at
M = 2048, so each expert fills one 64-route superblock. The recorded
routing is skewed:

- Up to 1,245 routes land on one expert.
- It needs 1.29 to 1.44 times the balanced superblock count (mean 1.36).
- The fused kernel decodes each expert tile once per superblock, so the
  extra superblocks are extra decodes.

At M = 512 each expert gets about 14 routes, and the recorded routing needs
0.99 times the expert count in superblocks. Every touched expert is decoded
exactly once per forward already.

Routed kernel time per step, TP2 rank 0, summed over the 22/17/3 layer mix:

| Kernel | M | Balanced | Recorded | Ratio |
|---|---|---|---|---|
| served (`65e05fdd`) | 512 | 849.1 ms | 911.8 ms | 1.074 |
| served (`65e05fdd`) | 2048 | 944.7 ms | 1,248.2 ms | 1.321 |
| master (`5892fd19`) | 512 | 642.9 ms | 664.7 ms | 1.034 |
| master (`5892fd19`) | 2048 | 709.6 ms | 912.2 ms | 1.286 |

- **Per stack at M = 2048:** each stack's recorded/balanced ratio is 1.27 to
  1.34, and its own layer's routing lands in the same range.
- **R1024 at M = 512:** it costs 1.15 (served) or 1.12 (master) times
  balanced, at 0.99 times the superblocks. That excess is load imbalance
  or a tail this data does not explain.
- **Effect on levers (a) and (b):** under recorded routing they are worth
  more than the balanced bench said. Per step they save 247 ms at M = 512
  and 336 ms at M = 2048, against 206 ms and 235 ms balanced.

## The 847 ms per chunk, named

Relaunch-4 frame, TP2 rank 0, per M = 2048 chunk.

| Component | ms | Share | Source |
|---|---|---|---|
| Routing skew in the routed kernel | 300.6 | 35.5% | measured: recorded-routing replay on the served kernel, scaled onto the 09-29 routed sum |
| KDA chunked recurrence and layout copies | 162.8 | 19.2% | TP1 trace, x0.5, 34 layers |
| mHC pre/post (runs on every rank) | 122.8 | 14.5% | TP1 trace, x1.0, 45 layers |
| MLA sparse attention, absorbed bmm, cat/masked_fill | 67.1 | 7.9% | TP1 trace, x0.5, 11 layers |
| Glue: residual/norm, SwiGLU, router top-k/sort, chunk-end lm_head | 33.1 | 3.9% | TP1 trace |
| All-reduce, 91 x 16 MiB | 77.8 (p50 73.9, p90 102.7) | 9.2% | prior link measurement |
| **Named** | **764.3** | **90.2%** | |
| Residual | 83.2 | 9.8% | unmeasured |

How the rows were built:

- **TP1 trace rows.** These are "single-box TP1, scaled: estimate, not a
  serve". Each component's per-layer time in the 12-layer trace is scaled
  to its layer count in the 45-layer model. Work split by attention head is
  halved for TP2; work every rank repeats on the full hidden state is kept
  whole.
- **The x1.0 factors were read from the image's model code.** mHC runs on
  the full `[M, 4, 4096]` stream on every rank. The model has a
  sequence-parallel path that shards it, but `use_sequence_parallel_moe`
  requires expert parallelism and a data-parallel size above 1, and the
  TP2 serve has neither. The MLA indexer projections are replicated
  (`disable_tp=True`).
- **All-reduce count.** Each layer has one all-reduce in the attention
  `o_proj` (RowParallel, KDA and MLA). It has a second after the MoE or the
  dense MLP `down_proj`. The shared experts pass `reduce_results=False`,
  and the MoE runner's late path reduces the combined output once. The
  vocab-parallel embedding adds one.
- **All-reduce time.** A 16 MiB all-reduce took 0.812 ms (p50), 0.855 ms
  (mean) and 1.129 ms (p90) on the two-box RoCE link on 2026-08-23, with
  NCCL 2.28.9. The serving image carries NCCL 2.30.
- **Residual.** It holds TP2 host and synchronization gaps, which are
  unmeasured (none exist at TP1), and the error in the x0.5 factors. It also
  holds about 5 ms of double count: the 09-29 sum priced MLA `kv_b` as a
  GEMM, and the served path runs it as the absorbed bmm counted above.

At L512 the same replay puts 62.6 ms of the 99.7 ms remainder on routing
skew.

The stock vLLM work in the table (KDA, mHC, MLA, glue and the all-reduce)
comes to about 460 ms per chunk, and none of it is Tessera's. The EXL3
reference serve runs the same architecture at the same settings (eager,
TP2, 2,048-token steps), so it must do equivalent work. Its image
(`eecb36e1`) was not inspected here. The routed kernel is where the two
serves differ by construction.

## A decode-to-scratch routed path

The fused window kernel already decodes into tensor-core tiles in shared
memory. The only question left was whether a two-phase path could win:
decode every touched expert to a global scratch, then run a grouped GEMM.

- **Scratch width.** The decoded weights are E4M3 values: the E4M3-family
  table is `native[codes[state]]` widened exactly to f16
  (`routed_fused.py`, `compose_table16`). One byte per weight is therefore
  an exact scratch.
- **Lower bound.** Any such path must at least:
  - read the wire once;
  - write the decoded weights once;
  - read the decoded weights back once;
  - do the GEMM's arithmetic.

  `scratch_bound.py` measures the rates that bound these on one GB10.
  Device copy runs at 239.4 GB/s (read plus write). A dense BF16 GEMM at the
  routed shapes runs at 79 to 100 TFLOPS. The bound is
  `T_lower = max((wire + 2 x 3.62 GB) / copy rate, dense GEMM time)`. It
  assumes memory and arithmetic overlap perfectly and charges nothing else,
  so it bounds any implementation.

Fused kernel time (master, recorded routing) against `T_lower`, per layer and
rank:

| M | R1024 fused | R1024 lower | R1088 fused | R1088 lower | R832 fused | R832 lower |
|---|---|---|---|---|---|---|
| 512 | 15.7 ms | 37.9 ms | 16.1 ms | 38.4 ms | 15.2 ms | 36.5 ms |
| 2048 | 21.0 ms | 37.9 ms | 22.7 ms | 38.4 ms | 21.8 ms | 36.5 ms |
| 4096 | 30.6 ms | 37.9 ms | 36.7 ms | 38.4 ms | 35.5 ms | 36.5 ms |
| 8192 | 51.9 ms | 37.9 ms | 64.9 ms | 38.4 ms | 63.5 ms | 36.5 ms |

The crossover, interpolated linearly in M, is about 5,500 tokens per step for
R1024, 4,340 for R1088 and 4,230 for R832. Below it no scratch path can
win. The serve steps at 2,048 tokens, where the bound is 1.7 to 1.8 times
the fused kernel, and at M = 512 it is 2.4 times.

## Levers these numbers point at

All three figures are estimates, not serves.

- **A 128-route superblock.** One decoded tile would feed two 64-row
  accumulator sets. Each output row keeps its own K order, so the output
  stays bitwise equal and the route stays the same symbol and decoder, with
  no contract-table change. On the recorded ids it cuts work items at
  M = 2048 from 390 to 300.5 (0.77x). At M = 512 it cuts nothing (0.99x).
- **One M = 8192 step for L8192.** On recorded routing the routed kernel
  takes 2,436 ms for one M = 8192 step, against 3,649 ms for four
  M = 2048 steps (0.67x). That is a serve-setting change
  (`max_num_batched_tokens`). The EXL3 reference cell was measured at 2,048,
  so a matched comparison would re-measure both.
- **mHC sharded across TP ranks.** This would save about 61 ms per chunk.
  It is stock vLLM model code, and the sequence-parallel path exists but is
  gated on expert and data parallelism.

## Reproduce

```bash
# TP2 rank-0 shapes, one GB10, recorded routing (bench_t8r.sh mounts --routing read-only)
bench_t8r.sh . OUT --groups experts.R1024.L10,experts.R1088.L11,experts.R832.L42 \
  --ms 512,2048 --routing /mnt/shared/tessera-measurements/t8r-speed-20260929/prefill-routing-20260930
BENCH_SRC=<b40c93cb src> bench_t8r.sh ...   # the served kernel
BENCH_PY=scratch_bound.py bench_t8r.sh . OUT --ms 512,2048,4096,8192 --routing <same dir>
```
