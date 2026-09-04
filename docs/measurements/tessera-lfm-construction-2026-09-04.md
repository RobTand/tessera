# LFM2.5 construction eligibility

The `Lfm2MoeForCausalLM` construction row is generated from
`construction/lfm25-8b-a1b-eugr-0281rc1.json`. It records the exact EUGR image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`
and vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`.

PrismaBuild action
`c180f042a96e37db4ed94b9a6841045cc7cb610c6034a0567321180c53e6ef27`
produced the original census on Sparky. The same action's model-loader
receipt records all 22 routed stacks and the model-level delegation of
`w1`, `w3`, and `w2` wires into the owning parameters. Construction used the
meta device and did not run a full-model forward.

The contract regression was shown failing before the row was added:

```text
PrismaBuild b4c808ea106441c2c01bd4302a5245fa034cfa19c1ac3ec6a9b91107b47cf537
tests/test_serving_construction.py:135: AssertionError
E AssertionError: the pinned LFM census receipt is not in the contract
FAILED tests/test_serving_construction.py::test_lfm_routes_the_routed_expert_stack
```

PrismaBuild action
`395c6a131fa5be81a852aff824cd68a35a4dc292a3ead7a774ed6c0883bc2668`
ran `tools/tessera_update_construction_block.py`; its generated diff is the
construction row. Post-change action
`1218a042f8e917d1f83de0a8da6ec639d295fdaee551a66ec157b3aa0dfe628a`
confirmed the generated block matches the receipt and ran
`test_serving_construction.py`, `test_serving_contract.py`, and
`test_serving_export_gate.py`: 81 passed, 1 skipped. Population: dl380g10,
torch `2.11.0+cpu`, no CUDA device, zero uncollected modules. The one skip's
verbatim reason was `E2M1 publishes no reader range`. This is CPU contract
coverage, not CUDA-gated coverage. Population receipt:
`/mnt/shared/tessera-runs/ts5/lfm25/astra-contract-green-r1.surface.json`.

A current-tree construction and source-layout preflight also passed as
PrismaBuild action
`bd4823b3284bfccf89d76fc20162683eb8f2c1f1cba2dcc4d3e2364dc3687025`
on Sparky under a full two-token GPU reservation and a 12 GiB demand. It
constructed the same architecture using the exact EUGR image, then derived
and validated 22 stacks with 32 experts each (2112 source projection
tensors) through the exporter's own `plan_expert_stack`. Its construction
receipt is
`/mnt/shared/tessera-runs/ts5/lfm25/astra-preflight-r1/construction.json`.

These receipts establish construction and source-layout eligibility. They
do not attest encoded full-model quality, promote a `routed_moe` cell, or
change the dense serving image pin.
