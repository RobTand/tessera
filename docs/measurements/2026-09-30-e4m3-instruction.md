# The E4M3 tensor-core instruction in the fused window kernel

**Status:** numerics measured on real wires; kernel speed measured (5–7.5%
faster at M = 2048 and 8192, equal power); not yet served. The library
is experimental (contract v46): its two launch pairs stand in
`scheme.EXPERIMENTAL_LAUNCHES`, no census cell names them, and the default
instruction stays f16 until a served census earns them cells.

The T-8 routed kernel widened each E4M3 byte to f16 and ran
`mma.sync.m16n8k16.f32.f16.f16.f32`. GB10 runs
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` at 246.5 TFLOPS, twice
the f16 rate (`2026-09-30-fp8-prefill-roofline.md`). This note records what
the E4M3 instruction changes in the kernel and how far its output moves from
the f16 instruction's on real wires.

In short:

- The E4M3 instruction is a third library of the one kernel source,
  `tessera_routed_fused_mma_e4m3`. `TESSERA_FUSED_E4M3_MMA=e4m3` selects it
  per process. It stamps its own decoders,
  `native_routed_fused_window_e4m3mma` and
  `native_fused_window_dense_e4m3mma`.
- Both instructions form the same exact products and accumulate them in
  fp32, so the two libraries differ only in summation order. On seven
  routed rungs and two dense modules of real GLM-5.3-Flash wires, at least
  99.86% of output elements are bitwise equal between the two, and no
  element differs by more than one bf16 unit in the last place of its row
  maximum.
- Tables and operand tiles take half the shared memory: 46,544 B fixed for
  the gate/up launch and 29,968 B for down and dense, against 91,600 B and
  58,640 B. The gate/up launch now reaches every rate from 1 to 8.

## What changed in the kernel

- **Decode table.** The composed table holds the E4M3 bytes,
  `native[codes[state]]`, as `uint8 [E, 2^14]`: 16 KB per table instead of
  32 KB.
- **B tile.** The decoded weights go to shared memory as bytes, k-major, with
  an XOR swizzle on their 16-byte units. `ldmatrix.trans` plus a byte
  permute builds the k32 fragment.
- **A tile.** The per-token E4M3 activation is staged unconverted, 32 bytes
  per row, in fragment order.
- **Epilogue.** The accumulator is fp32. The epilogue reads the same
  elements in a different register order and applies the same fp32
  epilogue: `acc * a_scale * w_scale`, the SwiGLU, and one bf16 rounding.
- **The f16 and value libraries** are unchanged. Their SASS is identical to
  contract v45's up to symbol names.

## Numerics

Both E4M3 libraries form the same products: an E4M3 byte widens to f16
exactly, and the product of two E4M3 values is exact in fp32. The E4M3
instruction accumulates at full fp32 width, rounds toward zero and keeps
subnormals, as the f16 instruction does on sm_121 (measured by
`experiments/t8r_speed/fp8_roofline.py`). The two libraries are therefore
two fp32 summation orders of one function of the wire, 32 products per
instruction instead of 16. No split-K promotion is needed.

### Routed oracle

`experiments/routed_pair_oracle.py --mode oracle --families e4m3` runs the
served adapter over real wires from the canonical census cache (layer 3, 16
experts, M = 1, 64 and 512) and holds every stage and the end-to-end output
to a dtype-derived bound. With `TESSERA_FUSED_E4M3_MMA=e4m3` it also builds
an f16-instruction twin over the same bundles and compares the two at
K = hidden (gate/up) and K = intermediate (down).

- Every rung passes every stage and the end-to-end bound, and the route
  record names `native_routed_fused_window_e4m3mma`.
- R912 has no fused-lane case: the oracle expects the compact adapter at
  that rung, and the cache holds only part of its wire set.

| Rung | M | Forward bitwise equal | Max diff (bf16 ulps of row max) | Pair diff / bound | f16 twin error / bound |
|---|---|---|---|---|---|
| R832 | 1 / 64 / 512 | 0.999756 / 0.999992 / 0.999956 | 0.0625 / 0.125 / 0.5 | 9.0e-5 / 1.8e-4 / 6.9e-4 | 1.4e-3 / 7.3e-4 / 1.7e-3 |
| R896 | 1 / 64 / 512 | 1 / 0.999996 / 0.999801 | 0 / 0.0625 / 0.5 | 0 / 8.8e-5 / 7.5e-4 | 1.7e-4 / 1.4e-3 / 1.4e-3 |
| R928 | 1 / 64 / 512 | 1 / 0.999989 / 0.998584 | 0 / 0.125 / 1 | 0 / 1.7e-4 / 1.4e-3 | 0 / 1.4e-3 / 3.1e-3 |
| R944 | 1 / 64 / 512 | 1 / 0.999229 / 0.999558 | 0 / 0.5 / 0.5 | 0 / 7.2e-4 / 7.1e-4 | 0 / 1.4e-3 / 1.8e-3 |
| R960 | 1 / 64 / 512 | 0.999756 / 0.999969 / 0.999979 | 0.125 / 0.5 / 0.5 | 1.8e-4 / 7.0e-4 / 7.4e-4 | 7.0e-4 / 7.2e-4 / 1.5e-3 |
| R1024 | 1 / 64 / 512 | 1 / 0.999985 / 0.999766 | 0 / 0.25 / 0.5 | 0 / 3.6e-4 / 7.3e-4 | 1.7e-4 / 7.2e-4 / 1.4e-3 |
| R1088 | 1 / 64 / 512 | 1 / 0.999989 / 0.999825 | 0 / 0.125 / 1 | 0 / 1.7e-4 / 1.4e-3 | 3.5e-4 / 7.2e-4 / 1.4e-3 |

At each stage separately, gate/up (K = hidden) and down (K = intermediate)
stay within 1 bf16 ulp of the row maximum and at least 99.975% bitwise
equal. The pair's difference is no larger than the f16 library's own
distance from the fp64 reference, which is what two summation orders of one
exact sum should show.

### Dense oracle

`experiments/dense_fused_oracle.py --mode oracle` loads the FP8 dense modules
of GLM-5.3-Flash stub B (q256 1024, rate 4) through the serve's own builder
at TP1 and both ranks of TP2, in both residencies, at M = 1, 3, 64, 512 and
2,048. With the E4M3 instruction selected it adds an f16-instruction twin
over the same bytes.

- All 51 cases pass with no bound violations. Of these, 34 run the E4M3
  instruction; the other 17 are the BF16 folded family, which the change
  does not touch.
- The E4M3 instruction against its f16 twin: at least 99.998% bitwise equal,
  at most 1 bf16 ulp of the row maximum.
- The largest error against the fp64 reference is 0.85 of the bound for the
  fused lane and for the Triton window GEMM alike.
- Real dense wires cover rate 4 only. Every rate from 1 to 8 is covered on
  synthetic wires by `tests/test_dense_fused_window.py` and
  `tests/test_routed_fused_window.py`.

## What this does not measure

- **End-to-end KL.** The offline decoded evaluator (G3) swaps decoded
  weights into a transformers forward and applies the activation
  quantizer as a hook. It never runs a Tessera kernel, so it scores both
  libraries identically and cannot gate this change. A served route census
  is the first measurement that runs the new library end to end.
- **Served speed.** The A/B below times one routed module per call on one
  GPU. The route census of the 8-layer GLM stub (PrismaBuild `7e514909`)
  served the new library on every required lane in both phases, and was
  refused only because no cell names its pairs yet; no full-model serve
  has timed it.

## Speed

PrismaBuild action `41c77d65` on sparklina timed the routed MoE module
alone, with `experiments/t8r_speed/bench_t8r.py`, in four arms run one
after another:

- **A:** master `1381c3b716`.
- **B:** this branch at `de3d0f3108`, with `routed_fused.E4M3_MMA_DEFAULT`
  flipped to `"e4m3"` (the only edit).
- **B′:** B again, as a drift control.
- **A′:** A again, as a drift control.

Each arm ran the same three rungs of the release T-8 artifact, 30 timed
iterations per cell. At M = 512, 2048 and 8192 the cells replay the routing
recorded from real prefills (`prefill-routing-20260930`; 55, 37 and 10
replays), and M = 1 and 8 use balanced routing. B and B′ produce
bitwise-identical output on all 312 cells.

Median wall time per call, A → B, with the geometric mean of the B/A ratio
over the replays:

| M | R832 (layer 42) | R1024 (layer 10) | R1088 (layer 11) |
|---|---|---|---|
| 1 | 0.61 → 0.59 ms (0.964) | 0.52 → 0.51 ms (0.983) | 0.64 → 0.61 ms (0.956) |
| 8 | 3.46 → 3.33 ms (0.963) | 3.05 → 2.94 ms (0.964) | 3.70 → 3.56 ms (0.963) |
| 512 | 15.25 → 14.58 ms (0.957) | 16.27 → 15.68 ms (0.965) | 16.05 → 15.42 ms (0.961) |
| 2048 | 21.91 → 20.62 ms (0.947) | 20.90 → 19.89 ms (0.951) | 22.58 → 21.47 ms (0.953) |
| 8192 | 64.76 → 61.06 ms (0.941) | 52.24 → 48.26 ms (0.925) | 65.41 → 61.34 ms (0.936) |

How far to trust each row, from the B′/B and A′/A spreads (geometric mean
per rung and M):

- **M = 2048 and 8192:** B′/B is within 0.4% of 1 on every rung, against a
  5–7.5% A→B gain. This is the claim: at prefill sizes the E4M3
  instruction's library is 5–7.5% faster per call.
- **M = 512:** B′/B and A′/A are within 0.4% of 1 on R832 and R1088, so
  their 3.9% gains hold. On R1024, the first rung each arm ran, both repeats
  ran faster than their first arm (B′/B 0.957, A′/A 0.954): a warm-up drift
  that each arm's first run carries. The drift-symmetric ratio
  `sqrt(B·B′) / sqrt(A·A′)`, which cancels it, is 0.966 there, so R1024's
  M = 512 gain is 3.4%.
- **M = 1 and 8:** one cell each. B′/B and A′/A are within 0.7% except
  R1024 at M = 1 (0.967 and 0.970). The 2–4% decode gain is suggestive, not
  established.

The drift-symmetric ratio on every other rung and M is within 0.6% of B/A.

Power was the same in both arms. The in-process sampler read 76–82 W on
every timed cell of either arm. Netdata on sparklina agrees at box level:
2-minute averages of 76–81 W while cells ran (60–63 W only at group
loads), 75 W over 04:28–05:14Z. That is 55–58% of the GB10's ~140 W
envelope, so the kernel is not power-bound, and work per joule improves by
the same ratio as time.

Scope: these are kernel timings of one routed MoE module on one GPU, not a
served number. Routed MoE was 35.5% of the M = 2048 prefill chunk at TP2
(`2026-09-30-prefill-attribution.md`), so a 5–7.5% kernel gain is worth
about 2–3% of prefill wall time before any other change. Receipts:
`/mnt/shared/tessera-measurements/t8r-speed-20260929/mma8-ab-20260930T035428Z`
(`master-routed`, `mma8-routed`, `mma8-routedb`, `master-routedb`, each with
`bench_t8r.json`).

## Receipts

- Routed oracles: `/mnt/shared/tessera-measurements/t8r-speed-20260929/mma8-oracle-20260930T035954Z`
  (R832, R960, R1024, R1088; source `23fbe2f02f`) and
  `/mnt/shared/tessera-measurements/t8r-speed-20260929/mma8-oracle-20260930T040551Z`
  (R896, R912, R928, R944).
- Dense oracle: `/mnt/shared/tessera-measurements/t8r-speed-20260929/mma8-oracle-20260930T040712Z/dense`
  (source `92ed3a5d5e`).
- Image: `spark-vllm-nccl230@sha256:f8dbe1a0`, sparky.
