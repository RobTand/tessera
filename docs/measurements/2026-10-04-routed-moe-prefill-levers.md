# Routed MoE at served prefill: where the time goes, and what can be removed

**Status:** kernel timings and Nsight Compute on one GB10, plus a re-read of the
A8SE752VB served profile. No serving code changes, and no serve was run. Nothing
here has been measured end to end; every "ms per chunk" below is a kernel-bench
delta scaled to the served chunk, so it is a ceiling or an estimate, not a gain.

Goal 2 needs about 338 ms removed from each 2048-token chunk per rank (1157 ms
today, GLM-5.3-Flash Tessera-8 `TESSERA_E4M3_K1`, L8192, c1, TP2, MNBT 2048).
Routed MoE is the largest bucket. This note breaks it down and prices the levers.

## Where the time goes (served profile, per chunk per rank)

Source: the A8SE752VB traces (`glm-pact-u4-20260927/results/A8SE752VB-val787-20261001/run`,
rank 0 on sparklina, rank 1 on sparky), kernel events inside the four GPU-side
`execute_context_1(2048)` windows, divided by four. The table sums kernel time.
Against the GPU-complete windows (`kernels-2500-critical-path-reconciliation-20261004`,
analysis v2), the windows are 1151.2 ms per chunk on rank 0 (4604.74 ms / 4) and
1151.0 ms on rank 1 (4603.99 ms / 4). The GPU-busy interval unions inside them (the
record's `busy_inside_gpu_windows`) total 4590.573 ms on rank 0 and 4589.892 ms on
rank 1 — 1147.6 and 1147.5 ms per chunk — so the device is idle inside the windows
for 14.17 and 14.10 ms per request (about 3.5 ms per chunk, 0.31%). In-window kernel
intervals barely overlap: the record's `overlap_factor_leaf_over_busy` is 1.00065,
and its scope is in-window — rank 0's leaf-sum line over its busy-union line inside
these windows (the same two fields give 1.00071 on rank 1); it is not a whole-trace
figure. The table below is a kernel-time sum inside those windows: it approximates
the in-window total to within the idle and the overlap just stated, and nothing
beyond them.

| Part | Kernel | rank 0 ms | rank 1 ms |
|---|---|---|---|
| Expert gate/up, decode + MMA + SwiGLU | `routed_fused_kernel<true,0,…,4,false,128>` x42 | 362.7 | 358.2 |
| Expert down, decode + MMA | `routed_fused_kernel<true,2,…,4,false,128>` x42 | 198.2 | 199.6 |
| Unpermute + top-k sum (fixed order) | `token_sum_kernel` x42 | 25.3 | 27.2 |
| Routed + shared add | `CUDAFunctor_add` x42 | 8.4 | 8.3 |
| Activation FP8 quant (routed share, 84 of 98 launches) | `dynamic_per_token_scaled_fp8_quant` | ~11.3 | ~11.6 |
| Router GEMM | `cutlass…s16816gemm_bf16_128x128` x42 | 2.9 | 2.9 |
| Top-k | `single_group_topk` x42 | 1.4 | 1.3 |
| Sort, fill, scatter (permute bookkeeping) | radix sort, fill, scatter | ~3.1 | ~3.1 |
| **Routed MoE, total** | | **~613** | **~612** |

- There is no separate token permute or gather kernel: the fused kernel reads
  activation rows through the route-sorted index, and `token_sum` is the unpermute.
- There is no separate weight-dequant kernel either. Decode happens inside the fused
  kernel (one shared-memory table lookup per weight), so "decode" and "GEMM" can
  only be separated by ablation (below).
- Cross-rank traffic inside the MoE: none. The experts are tensor-parallel, and the
  sequence-parallel ReduceScatter and AllGather around each layer are already the
  lever table's NCCL row (81 to 91 ms, all layers), not counted here.
- Every served routed layer is the R1024 one-run rung (rate 4, q256 1024, 4.02 bits
  per weight). The trace's template arguments `<…, 4, false, 128>` say so, and so do
  the artifact's 42 layers.

## Rooflines

- **Weight stream:** the 42 routed layers hold 153.0 GB. At 2048 tokens x top-8 every
  expert is touched, so each rank streams 76.5 GB per chunk: **313 to 330 ms** at the
  232 to 244 GB/s the repo has measured (`2026-09-30-fp8-prefill-roofline.md`).
- **E4M3 MMA:** 17.3 TFLOP per rank per chunk of routed work at 246.5 TFLOP/s = ~70 ms.
- **Today:** the two fused launches take ~560 ms, 1.7 to 1.8 times the weight-stream
  floor. Effective weight-read rate: served, 51.0 GB / 0.360 s = ~142 GB/s for gate/up
  and 25.5 GB / 0.199 s = ~128 GB/s for down; on the bench at M = 2048 (balanced),
  1.215 GB / 7.90 ms = ~154 GB/s and 0.607 GB / 4.43 ms = ~137 GB/s. Bench power is
  ~60 W of the 140 W envelope.

So the most any routed-kernel change can remove at MNBT 2048 is about 230 to 250 ms
per chunk.

## The fused kernel, decomposed (one GB10, layer 10)

`kern_action.sh` on sparklina, PB `ecfd04bc` (diagnostics) and `fbde999d` (prefetch),
TP2 rank-0 shapes. The weights are layer 10 of `pact-e4m3-accuracy-20260928/release-t8/exported`,
the bench's default artifact. Every `bench_t8r.json` records it. An earlier revision of
this note named the served artifact `glm53-a8-bf16menu-20260930`, because
`kern_action.sh` exported that path, but `bench_t8r.sh` never read the export. The
export is now gone, in its own commit. The bench layer stands in for the served layer
for these reasons:
- In both artifacts the layer-10 experts are R1024, one run, `wire_bytes_rank` 1.82 GB.
- Their 864 expert tensors have the same names, dtypes, shapes and byte sizes. The
  3,643,435,584 bytes is the combined expert-tensor population of one artifact,
  not a per-tensor size.
- Nine sampled tensors (experts 0, 143 and 287; gate, up and down) are byte-identical:
  9 of 864 hashed, not all. Expert 0's gate hash `c721ddc3…` matches the served wire
  hash that the #826 diagnosis recorded, and every served digest matches the pin in
  `routed_gate_826_inputs.json`.
- Receipt: `layer10_expert_tensor_hash_receipt.json` with its script
  `hash_layer10_sample.py`, both under the measurement root. The script re-derives
  every digest from the two artifacts (CPU-only PrismaBuild action `65757292`,
  CAS receipt `1fc629ea`).

release-t8 as a whole mixes rungs (R1024, R1088 and R832). Only its layer-10 stack ran
here. Routing is balanced, plus recorded prefill routing (`routing/` under the root):
two M = 2048 captures (layer 3 chunk 0, `ids-414-000000`; layer 10 chunk 0,
`ids-414-000007`), and layer 3's chunks concatenated for M = 4096 (chunks 0 to 1) and
M = 8192 (chunks 0 to 3). Recorded ids from another layer replayed through layer 10's
weights are a routing-skew proxy, not that layer's own traffic. Each arm runs forward then
reverse; ratios are the two passes summed against master's two. Source: master
`9cb2f04a`. Diagnostic arms are master with one edit, and their output is wrong by
design, with one exception noted below. Each bounds what removing that one cost
could save.

Root: `/mnt/shared/tessera-measurements/opus-moe-prefill-20261004T223553Z`
(`ab_table_round1.json`, `ab_table_round2.json`, `<arm>-time[b]/bench_t8r.json`).

Master at M = 2048, per layer: gate/up 7.90 ms (balanced) to 8.33 ms (recorded),
down 4.43 to 4.67 ms.

| Arm (time vs master) | gate/up, balanced | gate/up, recorded | down, balanced | down, recorded |
|---|---|---|---|---|
| no table lookups (`noLUT`) | 0.968 | 0.982 / 0.983 | 0.974 | 0.969 / 0.970 |
| no per-chunk producer barrier (`noProdBar`) | 0.988 | 0.978 / 0.974 | 0.998 | 1.011 / 1.012 |
| no MMAs (`noMMA`) | 1.013 | 1.004 / 0.998 | 1.025 | 1.029 / 1.032 |
| no epilogue stores (`noStore`) | 0.976 | 0.968 / 0.964 | 0.881 | 0.902 / 0.907 |
| no per-item table reloads (`noTable`) | 0.996 | 0.987 / 0.978 | 0.997 | 0.999 / 1.002 |
| no word copies (`noWords`) | 1.310 | 1.287 / 1.273 | 1.206 | 1.228 / 1.226 |
| claim 2 items per atomic (correct) | 1.502 | 1.556 / 1.512 | 1.109 | 1.107 / 1.103 |
| claim 4 items per atomic (correct) | 2.178 | 2.223 / 2.167 | 1.118 | 1.132 / 1.140 |
| A prefetch distance 0 (correct) | 1.101 | 1.491 / 1.484 | 1.071 | 1.074 / 1.074 |
| A prefetch distance 2 (correct) | 1.010 | 1.142 / 1.139 | 1.036 | 1.026 / 1.027 |
| A prefetch distance 8 (correct) | 1.006 | 0.998 / 0.999 | 1.004 | 0.996 / 0.995 |

- Every correct arm's output was bitwise equal to master's in every cell and pass.
- `noTable` (load the tables only for a block's first item) was meant as a
  wrong-output diagnostic, but its output equals master's in all 16 round-1 cells.
  Either this layer's per-expert tables are identical, so the reload is redundant, or
  the edit changed nothing for these inputs. That is unresolved: the tables were not
  compared. If the tables are identical, a bitwise no-reload variant is worth only its
  0 to 2%.
- `noWords` is not a valid ceiling. Without the copies, the decode reads stale shared
  memory, and the lookup pattern (so the bank conflicts) changes with it. It is
  reported, not used.
- No single cost is large. The lookups, the producer barrier, the MMAs, the table
  reloads and the activation loads (#739's `noA`: gate/up 0.986, down 1.00 on this
  rung) are each at most ~3%. Only the down launch's output stores reach ~10%.
- Claiming several consecutive items per block keeps an expert's table resident but
  spreads one expert's items over time. Measured (NCU): L2 hit rate 86% -> 71%, issued
  warps per scheduler 0.32 -> 0.16. Inference, not measured: with one item per claim,
  the ~48 blocks work on about 3 experts at once and share those experts' activation
  rows in L2. Rejected; it is not on the branch.
- The activation prefetch distance of 4 is at the optimum. Distance 0 costs up to 49%
  under recorded routing.

### Nsight Compute, master, M = 2048 (PB `c9ee92c6`, import `f9cbe0f7`)

| | gate/up | down |
|---|---|---|
| issued warps / scheduler | 0.32 | 0.30 |
| eligible warps / scheduler | 0.43 | 0.40 |
| L2 throughput (hit rate) | 72-74% (86%) | 66-67% (84%) |
| L1 hit rate | 1.8% | 9% |
| shared pipe wavefronts (of which bank conflicts) | 58% (47%) | 53% (47%) |
| tensor pipe | 18.7% | 16.6% |
| stall cycles per issue: barrier / wait / MIO / short scoreboard / long scoreboard | 5.2 / 2.4 / 1.4 / 1.2 / 0.4 | 5.6 / 2.3 / 1.3 / 1.2 / 1.1 |

One 512-thread block per SM (121 registers, 57.6 KB of shared memory for gate/up) and
eight producer warps that move in lockstep per chunk. Measured: low issue and
eligibility, barrier-dominated stalls, and no single ablation worth more than ~3% (the
down stores aside). L2 is the busiest unit, with ~4.8 GB of L2 reads per gate/up
launch (150M sectors) against a 1.2 GB wire. Inference, not shown by these
measurements: the time goes to the serial per-chunk chain inside the one block (wait
for the words, producer barrier, decode, store, consumers' barrier), and each
ablation removes one link while the rest still serialise. Inference too: a restructure
that raises L2 traffic would lose, as the claim-group arm did.

## Batch size changes the picture

Layer 10's weights (R1024), with layer 3's recorded routing: `ids-414-000000` at
M = 2048, chunks 0 to 1 at 4096, chunks 0 to 3 at 8192. Per layer, forward and reverse
mean:

| M per step | gate/up + down | per 2048 tokens |
|---|---|---|
| 2048 | 12.64 ms | 12.64 ms |
| 4096 | 16.21 ms | 8.11 ms |
| 8192 | 28.55 ms | 7.14 ms |

The wire is read once per step, so a larger step amortises it. Across 42 layers that
is about 190 ms (MNBT 4096) or 231 ms (MNBT 8192) per 2048-token chunk, at bench
scale. `token_sum`, the quant and the router scale with tokens and do not shrink.
This is a serving-configuration change, not a kernel change: the rest of the model
(attention chunking, memory) moves too, the 09-30 roofline note records that rank 1
does not fit at MNBT 8192 today, and none of it is measured end to end.

At M = 8192 the activation loads do matter. #739's `noA` arm takes 20% off gate/up
there, so #739 becomes relevant if the step grows.

## Lever table (ms per 2048-token chunk per rank)

Served scale: gate/up ~360 ms, down ~199 ms. Bench deltas are scaled by the served
launch time. The single-removal ceilings are not additive. Each was measured with
everything else in place, so they overlap one another, and all of them overlap the
restructure row's 0 to ~230 ms. Do not sum the rows.

| Lever | Removable | Output | Confidence | Effort |
|---|---|---|---|---|
| MNBT 4096 / 8192 (step size) | ~190 / ~231 (routed fused only) | not established bitwise for the model (MoE rows are per-token, attention chunking moves) | medium: bench, one layer's routing | config + memory fit; Rob decides |
| Producer-pipeline restructure (two producer groups on alternate chunks, or two blocks per SM in a smaller footprint), toward the weight-stream floor | 0 to ~230; unmeasured | bitwise by construction (same B tiles, same K order) | low: no ablation shows a single cause | high: multi-day rewrite of the chunk loop |
| Down-epilogue store coalescing (stage rows in shared memory, 16-byte stores) | ceiling 19-24 (down) + 9-13 (gate/up); realistic ~5-10 | bitwise (same values) | medium on the ceiling, low on the fraction | medium |
| Table-lookup bank conflicts | ceiling ~11-18 | bitwise if the table layout changes only | low | medium-high |
| Producer barrier per chunk | ceiling ~4-9 | bitwise | medium | part of the restructure |
| #739 activation ring (R1024, M 2048) | ~0, measured flat to 4.7% slower | bitwise | high (739's A/B `2e099289`) | done |
| Activation prefetch distance | 0 (4 is the optimum) | bitwise | high | none |
| Claim groups / table residency | negative | bitwise | high | rejected |
| Fold `token_sum` into the down epilogue | up to ~26 | not bitwise (summation order) | medium | high; needs a semantics decision |
| Routed + shared add | ~8 | bitwise if folded into `token_sum` | owned by #799 | #799 |
| Fuse activation quant upstream | ~5 of ~11 | bitwise in principle | low | lives in stock vLLM code; no fork |
| Router, top-k, sort | ~7 total | - | - | not worth a package |

**No bitwise-safe lever worth 20 ms or more per chunk was found that a measurement
supports, so no kernel prototype ships with this note.** The only kernel path toward
the ~320 ms floor is a structural rewrite of the producer pipeline, and its gain is
unmeasured.

## Correction to the planning table

Historical, now withdrawn. As of 2026-10-04 (`kernels-family-qualification`,
`prefill_2500_ranked_levers_20261004`, generated 20:58Z), the kernels lever table
credited the routed row with a #739 ceiling of about 84 ms per chunk (15% of the
fused kernels). That credit is withdrawn in the same record (row note
`withdrawn_739_ceiling`: "15 percent was a scope-level planning ceiling, not the
one-run served artifact opportunity"). The 15% was measured on the R1088 and R832
two-run rungs and at M = 8192. On the R1024 one-run rung, which is every served layer, #739's
own A/B at M = 2048 has the no-load ceiling at 1.4% (gate/up) and 0% (down), and the
ring itself at 1.00 to 1.05 times master. It is worth about 0 ms on the served chunk
today.

## Reproduce

```bash
# arms: <root>/src-<arm>/src, libraries built by build_ext.sh into <root>/ext-<arm>
KERN_ROUTING=<root>/routing KERN_STEPS=time KERN_MS=512,2048,4096,8192 \
  bash experiments/t8r_speed/kern_action.sh <root> master noLUT ...   # through pbrun --measurement, one GB10
```
