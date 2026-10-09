# Issue 1007: retained stock arithmetic review

## Decision

Arithmetic qualification remains false.
D41 promotion remains on hold.
The kernels parent and an independent reviewer must give explicit qualification verdicts.
This report supplies an evidence audit, not either required verdict.
No source change, default change, pin change, or D41 row promotion follows.

The reviewed repository head is `87d222aefbac679f726a42c34f824d98f2e2a7ca`.
PR #1051 supplied the source through merge `c7a57bba3a14e638aea308e32667d359b58b1f5b`.
The retained source record is `kernels-t4-fp4-mma-attestation-sol-20261007`.

## Scope

The stock domain has these limits:

- Physical devices: `sparklina:cuda:0` and `sparky:cuda:0`, both NVIDIA GB10.
- Activation global: 896.
- Weight global: `2^e`, with `-73 <= e <= 116` and the existing overflow guards.
- Reference shapes: N64, K256, and M1/2/3/16/33/65.
- Outputs: float32 dense GEMM and each actual grouped GEMM segment.
- Initial native accumulator: zero.
- Relative tolerance: zero.

Fused activations, router multiplication, split/token reductions, and TP2 arithmetic remain outside this scope.
The library's internal split-K tree belongs to the stock reference contract.
It does not qualify a fused split reduction or a token reduction.

## Derivation review

The derivation owner is [the serving contract, sections 17.7 and 17.8](../tessera-serving-and-moe-contract.md#177-targeted-native-implementation-attestation).
The implementation owner is `src/tessera/fp4_arithmetic.py:257-337`.

The retained native model has exact scaled products, 36-bit alignment, and one FP32 normalization per 64-column atom.
Its atom allowance is `eta = 65*2^-35 + 2^-23`.
The 65 terms cover 64 products and the incoming accumulator.
The four-atom allowance is `GN = (1+eta)^4-1` for K256.
The algebra retains the full geometric expression.

The retained scalar probes support one nearest-even activation division and one nearest-even ratio division.
Thus `rho_x = rho_r = u32 = 2^-24`.
Normal power-of-two weight formation is exact within the stated lattice.
Float32 output multiplication adds its actual `u32` term.
There is no independent BF16 input term or BF16 output term in this comparison.

The stock FP32 reference uses `GR = gamma(2*K,2^-23)`.
The count covers K products and at most K-1 nonzero additions, including its partial-sum tree.
The retained profiles name FP32 GEMV, small-N GEMM, and SIMT SGEMM kernels.
The targeted low-bit cases reject a TF32 or BF16 product model.
These finite probes support the stated implementation model; they do not prove every hardware input.

The FP64 magnitude model uses `GM = gamma(K,2^-52)`.
Converted FP32 products fit exactly within binary64 precision.
The four-term FP64 tensor instruction has at most four FMA result roundings per output.
Its normative precision comes from the [CUDA 13.0 PTX contract](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
The scalar premise comes from the [FMA contract](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-fma).
The retained profiles match only the supported scalar and tensor families.

For `M_upper >= max_(m,n) sum_k |Xh[m,k]|*|W[n,k]|`, the complete allowance is:

```text
FN = (1+rho_r)*(1+GN)*(1+u32)-1
B = [(FN+rho_x)/(1-rho_x)+GR]*M_upper
gamma(n,e) = n*e/(1-n*e)
```

The allowance depends on operand magnitudes, not the expected output magnitude.
It therefore covers cancellation, small outputs, and exact zero without an empirical absolute floor.
The normal lattice and prefix guards remain necessary premises.
The checker converts FP32 outputs exactly to FP64 and inflates its discrepancy outward.
No algebra discrepancy appeared in the retained coefficient audit.
The required reviewers must still judge the finite native and library characterization.

## Exact audit of the retained operands

PrismaBuild action `6e71c1a2b347323d490291d838c62ad92d9b66c2317392c90d2e361241092f3c` ran a new CPU evidence audit.
It used `dl380g10`, one CPU, four GiB, `/tmp`, and one native thread.
Its environment had Torch `2.11.0+cpu`, no visible CUDA device, and zero skips.
It ran no pytest suite and replayed no native GPU probe.
Its exit status was zero, and its resource scope was empty and released.

The audit used exact integer products and sums for every retained dense and grouped reference segment.
It reconstructed grouped weight tensors from retained encoded units.
It confirmed every retained magnitude upper bound against the exact operand contraction.
It independently reconstructed the full rational coefficient and compared it with every saved coefficient.
It passed the saved GPU outputs through the current stock comparison API with `rtol=0`.
It checked 40 dense segments and 60 grouped segments per device.

All 40 retained dense wrong outputs refused on each device.
The prior corrective audit had no grouped wrong-output controls.
The new audit added 60 CPU wrong-output controls per device through the same comparison API.
These controls changed a retained output beyond its derived bound; they did not change the retained GPU output.

Each device retained 760 exact-zero reference contractions.
The smallest nonzero expected output and strongest nonzero cancellation witness were the same entry:

- File: `original-<device>/dense-q895-dense-m65.pt`, index `[17,41]`.
- Exact dot: `-535/34359738368`.
- Sum of absolute products: `90945693277/34359738368`.
- Cancellation ratio: `535/90945693277`.
- Native output: zero.
- Stock output: `-4.0978193283081055e-08`.
- Absolute discrepancy: `11/268435456`.
- Derived allowance: `0.0001797865606896541`.

The audit checked the saved scalar references and stock probes without device execution.
It verified both native input and output digests against the saved reports.
The Sparky native report predates its physical-device field.
The audit used its action receipt and producer hostname to establish physical custody.

## Observed errors and allowances

These numbers describe separate physical captures.
The maximum error and maximum allowance occur in different entries.
They do not form a fitted error ratio.

| Physical device | Path | Maximum absolute error | Allowance at that error |
|---|---|---:|---:|
| sparklina:cuda:0 | dense | 1.7881393432617188e-07 | 0.00017944520050324373 |
| sparklina:cuda:0 | grouped | 1.1920928955078125e-07 | 0.0001410590620717044 |
| sparky:cuda:0 | dense | 1.7881393432617188e-07 | 0.00017944520050324373 |
| sparky:cuda:0 | grouped | 1.1920928955078125e-07 | 0.0001410590620717044 |

The dense maximum error occurs in `dense-q895-routed_moe-m65.pt`.
The grouped maximum error occurs in `grouped-q895-routed_moe-up-m65.pt`, segment one, shape `(2,64,256)`.
The maximum allowance on each device is `0.00018299438220731342` in `dense-q641-dense-m33.pt`.
The discrepancy at that maximum allowance is `1.1920928955078125e-07`.
The checked discrepancy upper bound at the maximum error is `1.788139343261719e-07`.
The report keeps the exact error and its outward checker value separate.

## Retained evidence and receipts

The packet root is `/mnt/shared/tessera-measurements/kernels-t4-fp4-mma-attestation-sol-20261007/`.
Each `original-<device>` directory holds encoded input units and `.pt` captures of operands, outputs, bounds, and controls.
Each `corrective-guarded-<device>` directory holds `audit.json` and `attestation.json`.

| Physical device | Native atom action | Stock action | Original dense/grouped action |
|---|---|---|---|
| sparklina:cuda:0 | `1f41a0ee2fe0f77a5c19ee8961e4b6d6da089fc6eb11ef7d04cc7e4214d6d5ae` | `654d26d4b40d7b654a1e20228dde899b2481c2e7e663282f61ac2af202ca01da` | `3e8b0474bdb0bdca4b92a67cca734c9e9f4fbc3db84b3f9e1de86b5843792f86` |
| sparky:cuda:0 | `416d56c1b36e1ab1840ef5502105745d592c5950519b5ebf87d7acebf7528d5d` | `bc466b44bdd5f3a740d414f4e96322f40b6b38b4fc6a33695f8f2518d83e7307` | `c50dff334df5b95ed8521e381ab84b7f155e7a91b861dbd09b67681934cd878f` |

The scalar actions are `4379a42a8826b31bc25b0e2b02fca9d28c94cb0d41ec1aeb68b1d3bbc6625f6d` for Sparklina and `fa36d16ef9d26976bb8aca68e06a432c3ba963334c7b7d7deb1ba784471828f7` for Sparky.
Their root is `/mnt/shared/tessera-measurements/kernels-fp4-output-boundary-run-flash-20261008/boundary-<device>/`.
Each device has 119 multiplication cases and 119 BF16 conversion cases.
The earlier guarded corrective action is `e1cbf87f4fb0a8486aba4d0a6461e24cb124b4b6d44a90381ae1ed6b5fad5c88`.
The live receipt query returned nine complete, terminal, green records with no unknown action.

Each action receipt is `/mnt/shared/prismabuild-fleet/cas/actions/v3/<first-two-key-characters>/<action-key>.json`.
The new audit receipt digest is `faff08d8d4855078748d4caaa5806452903b4affd0a342f03a0fe72cce9bad63`.
Its payload digest is `1e86185084cf4a434b77a39b9d76937902db2edb2799b44de71aef9db2b5fa6e`.
Its checkout snapshot digest is `b258be4e11ff20310bbba39e0b83424bc7e4757be65e39ff066bdf27b59eac9f`.
That snapshot retains the exact audit script.

The full new report is `/mnt/shared/tessera-measurements/issuegraph-1007-review-20261009/retained-review.json`.
Its SHA256 is `45a3d36de0a0790162dd5f5821876ba44e9cdf9b4267e0c890247647666a98cd`.
Its rows retain each capture path, byte digest, exact magnitude, discrepancy, allowance, and wrong-output refusal.
The report's `source_head` field is null because the worker did not export that environment field.
The receipt binds its actual checkout snapshot; the inspected repository head is stated above.

Two temporary audit versions failed before this successful audit:

- `7e14d4a56df14b6c15e541af75943a1d1c9c118ac56a0fd6b52e676bd84b1425` compared signed-zero bits instead of numerical values.
- `927224b9b3600f88544b7918c9105ec822fea7dc8f7207b12e6a42a0bf3f7293` required a field absent from the historical Sparky report.

Both failures remain failed with exit one and no successful receipt.
They are audit-harness failures, not failed GPU arithmetic results or pre-fix repository regressions.
No repository test or gate changed.

## Required qualification verdicts

| Required role | Qualification verdict observed | Current approval |
|---|---|---|
| kernels parent | Absent | false |
| independent reviewer | Absent | false |

The retained parent review was a targeted source `REQUEST_CHANGES` at head `a361165b05c6de3e949a6866e930ed17e19912b7`.
Its link is https://github.com/RobTand/tessera/pull/1051#issuecomment-6050600649 .
The latest independent `APPROVE` covered static source composition only at head `2370da771dff67899c3f274f23e09a88a1b7e557`.
Its link is https://github.com/RobTand/tessera/pull/1051#issuecomment-6055964283 .
Neither supplies a qualification verdict for this device packet.

The new audit exercised `require_t4_device_qualification` against both actual reports.
Both calls refused: `T4 refused on NVIDIA GB10: arithmetic qualification awaits the required reviews`.
The work agent cannot supply two independent approvals or assume the kernels parent's authority.
The issue therefore remains incomplete until the required reviewers judge the retained packet and this audit.
No new GPU campaign is necessary for the checks in this report.
