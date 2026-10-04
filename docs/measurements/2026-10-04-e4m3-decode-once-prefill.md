# Decode-once E4M3 for T-8 dense prefill (tessera#931): first operator receipt

**Status:** operator timings and a correctness receipt on one GB10, with
seeded Gaussian weights encoded at q256 1024. The lane is default-off and is
not wired into any route yet. No serve was run.

T-8 dense modules (`TESSERA_E4M3_K1`) decode the wire inside the GEMM, and at
prefill M that decode sets the time
([2026-10-04-bf16-projection-gemm.md](2026-10-04-bf16-projection-gemm.md)).
`tessera.serving.e4m3_prefill` decodes a prepared module's weights once to
plain E4M3 bytes with their fp32 row scale (`decode_e4m3`). It then serves
large M with `torch._scaled_mm`, row-wise (`prefill_apply`).

In short:

- **At the served M = 2048 the decoded lane is 2.7–3.7× faster than today's
  T-8 lane** (both before the quantiser they share) **and 1.3–2.0× faster
  than BF16** (quantiser charged).
- **Over the four main GLM-5.3 projection classes, that is 65 ms per
  2048-token chunk per rank against BF16 if they were T-8.** Against today's
  T-8 lane it is 171 ms.
- **The decode is exact.** It reproduces the stock materialisation's E4M3
  values and row scale bitwise at q256 832, 1024 and 1088.
- **From M = 256 up, the output was bitwise equal to the served fused T-8
  lane** on every shape measured. Below that M the fused lane splits K and the
  two differ, inside the derived bound.
- **The lane should take M ≥ 256.** That is the smallest measured M at which
  it beats today's T-8 lane on all four shapes. At M = 64 it loses on the KDA
  input projection (0.95×) and on MLA `o_proj` (0.61×).
- **Memory:** decoding every BF16 projection would add 3.04 GB per rank
  (2.83 GiB), and each TP2 rank holds its own copy, so 6.09 GB across both.
  With the T-8 packed copy (about 1 byte per weight at q1024) a rank holds
  about 6.1 GB of projection weights, the same as today's BF16 projections
  (6.08 GB).

## The contract, and why the decode is exact

`window_gemm` states the E4M3 family's contract. The weight is the E4M3 byte
`native[codes_of_state[state]]`. The activation is vLLM's native per-token
FP8 quantisation with scale `a_scale[m]`. The output is
`y = bf16((acc * a_scale[m]) * w_scale[n])`, where `acc` is the fp32 sum of
exact E4M3 products. `DecodedE4M3` holds the same bytes and the same fp32 row
scale (`PreparedDenseNativeModule.row_scale()`), so `_scaled_mm` computes the
same function. Only the fp32 summation order and the association of the two
epilogue multiplies may differ.

The decode is the served decoder, not a second one. Each role's frozen Triton
`PreparedWindowGemm` is called with its row scale replaced by ones, on an FP8
identity whose per-token scale is one. Each output is then one exact product
`1 * w` plus exact zeros, times one, twice. The result is the E4M3 value
itself, bf16 holds it exactly, and the cast back to E4M3 is exact. A zero may
come back unsigned, which is numerically the same weight. The test admits
that and nothing else.

## Correctness receipt

PB `58e2764f9fd7` ran on sparky, image `spark-vllm-nccl230@sha256:5be13705`,
branch head `46f802a672`. It ran `tests/test_e4m3_prefill.py` through
`experiments/routed_fused_tests.sh`: **8 passed**.

- `test_decode_once_reproduces_the_stock_bytes_and_row_scale[832|1024|1088]`
  uses a three-role module (256 + 128 + 32 rows, 512 columns) and a decode
  chunk (192) that does not divide the columns. It checks that the decoded
  values equal `stock.materialize_stock`'s `weight`, that the bytes are equal
  up to signed zero, and that the scale is bitwise the stock `weight_scale`
  and the module's `row_scale()`.
- `test_the_scaled_gemm_is_the_e4m3_contract_within_the_derived_bound[1|7|64|512|2048]`
  checks the decoded lane against the fp64 definition, within
  `fused_bound.dense_bound` charged with a K split equal to K. That bound
  holds for any summation order, so the test does not depend on which FP8
  kernel cuBLAS picks. Worst ratio to the bound: 0.81, 0.84, 0.84, 0.85,
  0.85. Against the served module, the difference was within twice the bound,
  and the outputs were in fact bitwise equal (fraction 1.000000) at every M.
- Pre-fix: on master the test module fails at
  `from tessera.serving.e4m3_prefill import ...` with `ModuleNotFoundError`.
  The module is new, so that line was not separately executed.

The bench repeats the check on the GLM shapes up to M = 2048. Worst decoded
ratio to the bound: 0.24 on `kda_in` and `o_proj`, 0.68 on `q_b`, 0.13 on
MLA `o_proj`. The bitwise-equal fraction against the served fused lane was
1.0 at M ≥ 256 and 0.9997–0.9999 at M ≤ 64.

## Operator timings

The same action ran `experiments/t8r_speed/bench_e4m3_prefill.py` through
`bench_t8r.sh` and wrote
`/mnt/shared/tessera-measurements/e4m3-prefill-20261004/run1-20261004T233316Z/bench/e4m3_prefill.json`.
Each cell is the median CUDA-graph replay of 30 iterations. Every module took
the fused lane. `quant` is vLLM's per-token quantiser alone; both T-8 lanes
need it.

| Module (rows x cols) | M | BF16 | T-8 fused | decoded | quant | decoded vs T-8 | decoded+quant vs BF16 |
|---|---|---|---|---|---|---|---|
| kda_in (12448 x 4096) | 64 | 0.515 | 0.262 | 0.275 | 0.007 | 0.95× | 1.83× |
| | 256 | 0.822 | 0.633 | 0.297 | 0.007 | 2.13× | 2.71× |
| | 512 | 0.810 | 1.206 | 0.343 | 0.009 | 3.52× | 2.30× |
| | 2048 | 2.551 | 4.372 | 1.195 | 0.067 | 3.66× | 2.02× |
| o_proj (4096 x 4096) | 64 | 0.211 | 0.073 | 0.021 | 0.007 | 3.40× | 7.5× |
| | 256 | 0.226 | 0.192 | 0.067 | 0.007 | 2.88× | 3.05× |
| | 2048 | 0.790 | 1.355 | 0.367 | 0.054 | 3.69× | 1.88× |
| q_b (8192 x 1536) | 64 | 0.081 | 0.054 | 0.042 | 0.007 | 1.29× | 1.65× |
| | 256 | 0.152 | 0.154 | 0.042 | 0.007 | 3.69× | 3.11× |
| | 2048 | 0.539 | 1.053 | 0.392 | 0.015 | 2.69× | 1.32× |
| MLA o_proj (4096 x 8192) | 64 | 0.360 | 0.132 | 0.216 | 0.007 | 0.61× | 1.61× |
| | 256 | 0.437 | 0.369 | 0.311 | 0.009 | 1.19× | 1.37× |
| | 2048 | 1.705 | 2.787 | 0.815 | 0.218 | 3.42× | 1.65× |

The JSON holds every M from 16 to 8192.

**Per 2048-token chunk per rank** (calls per chunk from the A8SE752VB trace),
counting the decoded GEMM plus the quantiser against today's BF16 GEMM:

| Class | calls | saved |
|---|---|---|
| KDA input projection | 34 | 43.9 ms |
| KDA o_proj | 34 | 12.5 ms |
| MLA o_proj | 11 | 7.4 ms |
| MLA q_b | 11 | 1.5 ms |
| **total** | | **65.2 ms** |

This charges a separate quantiser per call. A quantiser fused into the
preceding norm, or shared between GEMMs on one input, would add to the
saving. The remaining small projections (q_a+kv_a, the indexer) were not
measured here.

**Large-M anomaly (outside the served regime).** When the activation reaches
32 Mi elements (M·K ≥ 2^25: `kda_in` at M = 8192, MLA `o_proj` at M ≥ 4096),
both FP8 lanes slow by 4–6× per doubling of M. For example, decoded `kda_in`
takes 2.48 ms at M = 4096 and 16.2 ms at M = 8192. BF16 does not slow. The
cause was not measured. Re-reading an activation that no longer stays in L2,
once per output-tile column, would account for it. The serve's MNBT 2048 keeps
every call at M ≤ 2048. If MNBT is raised, a fixed M chunk inside
`prefill_apply` is the first thing to measure.

**Load-time decode cost:** 0.247 s for `kda_in` (12448 x 4096), 0.013 s for
`o_proj`, 0.053 s for MLA `o_proj` and 0.004 s for `q_b`. At 34 KDA and 11 MLA
layers that is about 10 s per rank at load.

## Memory on GB10 at TP2

These figures are per rank, computed from the TP2 rank shapes. They cover
KDA: in 12576x4096, o 4096x4096 and the two 4096x128 low-rank projections,
times 34 layers. They cover MLA: q_a+kv_a 2048x4096, q_b 8192x1536,
indexer-q 4096x1536, indexer-k 288x4096 and o 4096x8192, times 11 layers.
The decoded copy costs one byte per weight plus 4 bytes per row, as measured:
`kda_in` is 51,036,800 B = 12448·4096 + 4·12448.

| | per rank |
|---|---|
| Projection weights | 3.039 G |
| Today, BF16 | 6.08 GB |
| T-8 packed at q1024 (~1 B/weight) | ~3.04 GB |
| Decoded E4M3 copy | 3.04 GB (2.83 GiB) |
| T-8 packed + decoded | ~6.08 GB |

So a T-8-projection artifact with this lane holds about the same projection
bytes per rank as today's BF16 artifact. The two TP2 ranks each hold their own
copy, so the decoded copies total 6.09 GB across the pair.

## What is not done

The lane is not wired into any route. The served integration still needs:

- dispatch in `PreparedDenseNativeModule.apply` at M ≥ 256, behind a
  default-off flag;
- the decoded copy held and counted by the module's residency accounting
  (`named_tensors`/`packed_bytes`);
- a launch symbol and decoder for the census;
- a runtime-contract cell for the lane;
- an artifact whose projections are T-8. That is campaign's accuracy
  decision; today's A8SE752VB export keeps the projections in BF16.

## Reproduce

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py --cwd . --tag gb10 --exclusive \
  --cpus 4 --demand mem_gb=32 --detach \
  --container-image content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918 \
  --env ORACLE_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a \
  --env TEST_RUNNER_SP=/home/rob/venvs/pb-cpu/lib/python3.12/site-packages \
  -- bash experiments/t8r_speed/e4m3_prefill_action.sh OUT
```
