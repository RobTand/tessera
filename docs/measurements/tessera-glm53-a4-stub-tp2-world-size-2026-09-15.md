# World size 2 for the three Tessera families: two-rank serves, rank traces and a single-rank KL (2026-09-15)

**Status: closed for what it claims.** This is the receipt that contract v29 names in
`tensor_parallel.world_size_receipts[glm53_a4_stub_tp2_sm121]`. It is the evidence that raises
`tensor_parallel.units[].max_world_size` from 1 to 2 for `TESSERA_E2M1_K2`, `TESSERA_E4M3_K1`
and `TESSERA_BF16_K1`. The grade stays `route_only`.

`max_world_size` is an attestation: the largest tensor-parallel world a served receipt covers.
The contract's own bar for raising it is a two-rank serve, a route trace from each rank, and a
KL against a single-rank arm. This receipt carries all three for a world of two on sm_121
(GB10). It says nothing about quality: the KL here compares two ranks with one rank on the same
checkpoint, not the checkpoint with its BF16 reference.

The findings:

1. **Every family executed on both ranks, in two separate two-rank serves.** Both serves ran
   the same checkpoint on the same image and tree. Each rank's route trace records all three
   families' routes, so all three units move, and no unit moves on the strength of another
   unit's trace.
2. **The A4 stub's TP2-vs-TP1 divergence exceeds a BF16 control's.** On the shared token ids,
   the median |d logprob| is 4.26 times the BF16 stub's and the p99 is 1.11 times it
   (tessera#514). A second single-rank serve of the A4 stub agrees with the first bit for bit,
   so the excess is not serve-to-serve noise at one rank.
3. **The grade does not move.** Nothing here scores the checkpoint against a reference, so the
   receipt attests that the routes execute at a world of two and publishes the divergence it
   measured. Whether that divergence is acceptable is not a claim this contract makes.

## The two-rank serves

Both serves used the same setup:

| Item | Value |
|---|---|
| Image | `localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb` |
| Tessera tree | `44d20d670700d719a50052e6050372fd5d9b9c2f` |
| Checkpoint | GLM-5.3-Flash 4-layer stub, `glm53-4layer-a4-e2m1x2-q896-l2`: `layers.0.mlp.gate_up_proj` E4M3 q1024, `layers.0.mlp.down_proj` BF16 q1792, `layers.1.mlp.shared_experts` E2M1x2 q896, `layers.1.mlp.experts` E2M1x2 q896 routed |
| Ranks | rank 0 on sparky, rank 1 on sparklina, one GB10 each |
| Collective fabric | NCCL over RoCE on the direct Spark-to-Spark ports |
| Environment | `TESSERA_SERVE_MODE=resident`, `TESSERA_ROUTE_TRACE` set, `TESSERA_RESEARCH_GLM53_NOPE=1` |
| Memory watchdog | one per box, 16 GiB `MemAvailable` floor and PSI full at or above 20 |

The rendezvous flags (`--master-addr`, `--master-port`) are omitted from the tables in this
file and from the contract because they name one fleet's addresses.

### Serve 1: the route census (2026-09-15T02:28Z)

This is the serve that `docs/measurements/tessera-glm53-a4-stub-tp2-served-2026-09-14.md`
records in full, and the one behind contract v28's routed E2M1_K2 cells.

- Flags: `--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp --enforce-eager --attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend flashinfer_cutlass --max-model-len 4096 --gpu-memory-utilization 0.25 --kv-cache-memory-bytes 4294967296`
- Route traces:
  - `experiments/results/glm53_a4_stub_tp2_route_trace_rank0.json`, sha256
    `98b06898c63298dfc9c75cd05c6ad32f8cc003c2fecf78b793d1fe629db5ae53`
  - `experiments/results/glm53_a4_stub_tp2_route_trace_rank1.json`, sha256
    `92c116c073b238b060a0474e85b072bb241e1e90fed1e8e1e7fcd5dd31da4374`

### Serve 2: the KL arm (2026-09-15T04:29Z)

This serve produced the TP2 logprob dump the KL below reads.

- Flags: `--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp --gpu-memory-utilization 0.25 --enforce-eager --attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend flashinfer_cutlass --kernel-config '{"enable_flashinfer_autotune":false}' --max-model-len 4096 --kv-cache-memory-bytes 4294967296 --trust-remote-code --max-logprobs 1024 --served-model-name kl-target`
- Route traces, committed byte for byte from the serve's output directory:
  - `experiments/results/glm53_a4_stub_tp2_kl_arm_route_trace_rank0.json` (worker pid 501, 13
    flushes), sha256 `c3ae7b61eebc4def08bc5449a713bb0e06b0f656ffdf2f78a07edb49eb4e64ac`
  - `experiments/results/glm53_a4_stub_tp2_kl_arm_route_trace_rank1.json` (worker pid 426, 13
    flushes), sha256 `7b739bf6e9dc6c5582830aebe02bb0848172a837ee9c4d21dfa0b9cae9172e24`

The KL arm's two traces hold the same entries with the same counts. The prefill forwards ran
at 512 rows, eight launches per route:

| Kind | Policy | Activation contract | Per-rank shapes at M512 |
|---|---|---|---|
| moe | `TESSERA_NVFP4:resident` | `e2m1_group16_ue4m3_static` | N2048:K4096 |
| dense | `TESSERA_NVFP4:resident` | `e2m1_group16_ue4m3_static` | N2048:K4096 (rows), N4096:K1024 (columns) |
| dense | `TESSERA_FP8:resident` | `fp8_per_token_dynamic` | N12288:K4096 (rows) |
| dense | `TESSERA_BF16:resident` | `bf16_unquantized` | N4096:K6144 (columns) |

## Which units move, and why

A unit moves when its route appears on every rank of every serve this receipt names. The
contract maps a route to its family through `contract.PAYLOAD_FAMILY_BY_ROUTE`, and
`tests/test_serving_contract.py` derives `executed_units` from the four trace files rather than
reading it from this page.

| Unit | Route in the traces | Axes cut at a world of two | Moves |
|---|---|---|---|
| `TESSERA_E2M1_K2` | `TESSERA_NVFP4`, dense and routed MoE | rows and columns | 1 to 2 |
| `TESSERA_E4M3_K1` | `TESSERA_FP8`, dense | rows only | 1 to 2 |
| `TESSERA_BF16_K1` | `TESSERA_BF16`, dense | columns only | 1 to 2 |

All three `tensor_parallel` units are `kind: tessera_wire_family`, so there is no other unit to
leave at 1. The table's axis column is a limit on what the traces show, not on the attestation:
`max_world_size` is per unit, and `loader_axes` beside it says what the loader does with each
axis. The E4M3_K1 column cut and the BF16_K1 row cut are loader facts that no served trace
covers.

## The single-rank KL (tessera#514)

Three arms, each a pair of `kl_tool` top-1024 prefill dumps over the n8 x s512 corpus contract,
compared with three metrics. The committed table is
`experiments/results/glm53_a4_stub_tp_single_rank_kl_514.json`. It carries each dump's npz
sha256 and each comparison's sha256, and it is what the contract's
`single_rank_kl` block copies.

- **A4, TP2 vs TP1.** The A4 stub at a world of two (serve 2) against the same stub on one GB10.
- **BF16, TP2 vs TP1.** The BF16 GLM-5.3-Flash 4-layer stub, which carries no Tessera module, at
  a world of two against one GB10, with the same image, flags and environment. This is the
  control: the divergence that vLLM's own tensor-parallel numerics produce on this model.
- **A4 floor, TP1 vs TP1.** A second single-rank serve of the A4 stub against the first.

| Metric | A4 TP2 vs TP1 | BF16 TP2 vs TP1 | A4 TP1 floor |
|---|---|---|---|
| \|d logprob\| on shared ids, p50 | 0.1132 | 0.0266 | 0 |
| \|d logprob\| on shared ids, p99 | 0.667 | 0.602 | 0 |
| \|d logprob\| on shared ids, max | 3.04 | 2.87 | 0 |
| Renormalized shared-support KL, mean | 0.0257 | 0.0117 | 0 |
| Renormalized shared-support KL, p99 | 0.177 | 0.169 | 0 |
| Renormalized shared-support KL, max | 0.269 | 0.424 | 0 |
| Top-1 id agreement | 81.02% | 92.44% | 100% |
| Top-8 id set exact | 33.86% | 76.08% | 100% |
| Top-1024 KL lower bound, mean | 0.00614 | 0.00274 | 0 |
| Top-1024 coverage, mean | 0.263 | 0.272 | 0.263 |

The A4 excess over the BF16 control is 4.26 times at the median |d logprob| and 1.11 times at
the p99. The contract's validator derives both ratios from the two arms' published metrics and
refuses a typed value that disagrees.

**Read the |d logprob| and renormalized columns.** The stub is near-flat: the top 1024 ids hold
about 0.26 of the mass at a position. The lumped top-K KL lower bound therefore describes a
quarter of the distribution, and it is shown for completeness only. The renormalized KL
discards the mass outside the shared ids, so it is a secondary reading.

The A4 excess is spread across positions rather than concentrated at the start of each chunk:
the median |d logprob| at position 0 is 0.087 and over positions 16 onward it is 0.111. The
BF16 control's are 0.026 and 0.025.

## What this receipt does not show

- **No quality grade.** Every arm compares the checkpoint with itself at another world size.
  The grade stays `route_only`, and the contract's validator refuses any other grade on a
  world-size receipt.
- **No cause for the A4 excess.** It is measured, not explained. tessera#514 tracks the
  per-layer TP1-vs-TP2 comparison that would place it.
- **Nothing about KV-head replication above 1.** The stub declares 64 key/value heads, so a world
  of two needs no replication, and no attention module is a Tessera module. v29 publishes the
  loader's replication rule in `tensor_parallel.kv_head_replication` with
  `exercised_by_receipt: false`.
- **One world size.** A world of four or more is not covered.
- **One platform, one image, eager only.** The GLM-5.3 NoPE attention backend refuses graph mode
  (tessera#508).
- **Not the full model.** No 45-layer serve has run at any world size.

## Sources

- The dumps, comparisons and serve records are on the shared evidence volume under
  `dq-runs/glm-first-artifact-claude-20260914/tp2-equivalence-20260915/`, in `kl-tp1-tp2/`,
  `kl-bf16/` and `kl-a4-tp1-rerun/`. The npz sha256 values in the committed table identify them.
- Serve driver and comparison script: `tp1-kl.sh` and `tp-equivalence.py` in the same directory.
