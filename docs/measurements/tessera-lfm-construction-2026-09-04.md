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

## Additional tests selected from the contract diff

PrismaBuild `58ced01a7c04ed8bb104aa1a9946dfb919275201d117ca3f1d23e6ceb1dedb8d`
ran the impact selector in its own snapshot against the exact fetched trees
`307c0d15a9a56c98883058118862c4e383edc22c..7e32950a87bb3968759a99039d276458d2cc4229`.
It returned `narrowed`, ten test files and no full-suite trigger. The three
files above had already passed; action
`812ec1224c629876e5ffc816f84349c964e05b47bd230e97eac7c33d6867746e`
ran the other seven: 132 passed, 12 skipped, zero uncollected modules, serial
on dl380g10 with torch `2.11.0+cpu` and no CUDA device. Its population is
`/mnt/shared/tessera-runs/ts5/lfm25/astra-contract-impact-r1.surface.json`.
The skip reasons, verbatim, were:

```text
6 /home/rob/tessera-runs/compile-dispatch is not on this box
1 the PrismaQuant tree with tessera_formats is not on this box
1 the PrismaQuant tree or the allocation outputs are not on this box
1 Qwen3-0.6B is not on this box
1 the two surviving compile caches from 2026-09-02 are not on this box
1 /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box
1 /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box
```

These environment-dependent checks were not exercised by this CPU run; the
passing count does not cover them or the CUDA-gated surface.
