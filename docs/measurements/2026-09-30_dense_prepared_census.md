# Dense prepared-bundle census witness

Issue: [#638](https://github.com/RobTand/tessera/issues/638).

## Scope

This component witness exercises FP8 and BF16 prepared native bundles at M=1
and M=3, resident and streamed, eager and Torch-compiled. It checks the emitted
route against the census helper and rejects the retired materialized/GEMV pair.
The production helpers are unchanged: their correction landed previously.

This is not a whole-model vLLM serve, a full route census, a KL measurement,
a ship-cell promotion, or a performance comparison. The existing route fixtures
substitute vLLM configuration construction; FP8 uses their reference activation
quantizer, not a stock-vLLM quantizer qualification. Compiled outputs have a
shape check, not an additional numerical qualification.

## Population and causal evidence

| Population | PB action | Result |
|---|---|---|
| Initial historical helpers | `d7da2dd4b7db4349bd70c149440e01dad03ffbe3e33ed9ef80a201dd3cd9bc57` | 8 passed, 8 failed, 0 skipped; CUDA executed 16. Four BF16 compiled failures are causal; four FP8 compiled failures were fixture module-registration errors, not producer defects. |
| Corrected fixture, historical FP8 helper | `af300947e2745458932ca966b5db340c93e8e99978e4ba6bf01a7b60592359ab` | 4 eager controls passed, 4 compiled cases failed, 0 skipped/errors/uncollected; CUDA executed 8. |
| Current helpers, both families | `05fae5755168318db3fb6275942cf1afbe4ca5bf553ab8f7d21890c95bfadaa0` | 16 passed, 0 failed/skipped/errors/uncollected; CUDA executed 16. |

The pre-fix assertion is `AssertionError: census accepts an unobserved retired
pair`. Only the historical compiled-expectation branches were restored for RED,
not a pristine whole-tree baseline. FP8 compiled forwards emitted
`tessera::fused_window_dense / native_fused_window_dense_e4m3mma`, with
`M*:N256:K1024`, despite the historical helper admitting its retired pair.
GREEN records the same FP8 pair and BF16's
`tessera::window_gemm_dense / native_window_gemm_folded`. All 16 case records,
JUnit outcomes and the controller population agree; no skip reasons or
uncollected modules are present.

## Execution and receipts

Both decisive runs ended on sparklina in one worker attempt. The image is
`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5`.
Device population: `torch 2.13.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10`.
Two worksteal workers share the admitted CPU mask; native/build threads are one
per worker. PB reserves CPU2, mem8GiB, GPU1 and GPU memory8GiB, priority-10,
execution timeout1200s and client wait600s. Admission owns placement; no host
pin, accounting override, repeated completed measurement or scheduler workaround
was used. Client wait expiry before admission was reconciled against the later
worker ending, not treated as a test result.

The coordinator's 18:50Z ruling makes host `/proc/meminfo` MemAvailable
authoritative on GB10. The wrapper requires 16GiB plus the planned8GiB footprint
immediately before launch, and cases retain the16GiB host floor. Budget-scoped
CUDA free is not used. Prelaunch MemAvailable was61,290,397,696bytes for the
corrected RED and43,201,454,080bytes for GREEN. An earlier retry refused all
cases under the superseded CUDA-free predicate; it supplies no producer RED or
GPU qualification. WINDOW_ACTIVE was absent before each client.

GREEN receipt: `936ccfc1613a9d4bc40ec7b7208220e1773215cdfa0ccbe1057f88ca5805b42b`.
Result:20230bytes, SHA-256
`ccddb489a9ea33a273457a7915ab3274d33e527dcb82430c4d09b1732a0ad900`.
Claim:`c1acc00798cb5449d1e89cbcb6899492ed79ddc2ba0e13c2e6d79ce88381822b`.
All nine local-claim integrity checks pass, including payload hash; full worker
attestation was not independently audited. Failed RED has no successful CAS
receipt.

Artifacts:

- Corrected RED: `/mnt/shared/tessera-measurements/issues-ts-638/validation.be5g19ms`.
- GREEN: `/mnt/shared/tessera-measurements/issues-ts-638/validation.kVE0OEYP`.
- GREEN immutable stdout:22802bytes, SHA-256
  `43ab945103846e524b23b4d5487122045aabbd09c5735c0baacf420aedc3dba8`.
- Test SHA-256:
  `1901cbf6b66af72ec1dfb8329ed358b770c6a650eded6e7faa358eca69d678de`.
- Validation-only wrapper archived at
  `/home/rob/tmp/claude-campaign-20260926/tmp/p2p3/tessera/638-prepared-wrapper-ts.sh`,
  SHA-256:`699e38e5c350db47b039b0a7dc022a79fd7c0a95032fe2cc66112c68e0f4d6bf`.
  It was included in the tested snapshot, not installed as a production tool.

GREEN source population: snapshot
`51225e2f20667f164be7a0d6c96a45711a7b5f98`, verified SHA-256
`ba4448bdeea55a3ac8a1a5363c3952c5d4aa63fd8b18d87914d5b92e39d34a3c`.
The source stayed stable during measurement and both workers agreed. This
record was added afterward; no whole-tree equivalence or deployment claim is
made for the subsequent documentation commit.
