# The Tessera routed-MoE route loads and executes on the pinned runtime

**Date** 2026-09-04 · **Box** sparky (GB10, sm_121) · **Image**
`vllm/vllm-openai@sha256:61fc8a89...` (the pin in
`src/tessera/serving/runtime_contract.json`, vLLM 0.28.0, torch 2.13.0+cu130)
· **Probe** `experiments/moe_route_load_probe.py` · **Result**
`experiments/results/moe_route_load_probe.json`

## 1. What was measured, and what was not

The **load-and-execute contract** of `tessera.serving.moe_route`: vLLM's real
`RoutedExperts` is constructed with a Tessera checkpoint's
`quantization_config`, its own `load_weights` is handed per-expert wire tensors
under the names a checkpoint would carry, the route's
`process_weights_after_loading` decodes them, and the runtime's own fused-MoE
modular kernel multiplies them.

This is **not** a served census and **not** a KL. No model is loaded, no engine
is started, the weights are random, and no artifact exists. A `routed_moe`
cell in `runtime_contract.json` needs a served artifact and **is not earned
here**. Nor is it a performance claim: nothing was profiled and no before/after
delta is offered, so principle 15's two-instrument rule is not invoked. The box
was shared with two other agents' GPU work throughout (31.6 W at start against
a ~140 W envelope; 5.5 W idle earlier the same night), which is exactly why no
timing is reported.

Shape: 4 experts, hidden 512, intermediate 256, top-2, E4M3 grid / WINDOW body
/ CHANNEL plane at q256 1024, 17 tokens, `TESSERA_SERVE_MODE=resident`.

## 2. What the runtime did

| Fact | Value |
|---|---|
| Method vLLM built | `TesseraMoEMethod` |
| Backend the runtime's own oracle selected | `TRITON` (`TritonExperts`) |
| Loader calls into the wire parameters | 12 = 4 experts x 3 projections |
| Parameters filled by `load_weights` | `w13_wire`, `w2_wire` |
| Parameters after `process_weights_after_loading` | `w13_weight [4,512,512] float8_e4m3fn`, `w2_weight [4,512,256]`, `w13_weight_scale [4,512,1]`, `w2_weight_scale [4,512,1]` — the wires are gone |
| Decoded tile vs `tessera.stock.materialize_stock` | **byte-identical**, every expert, every projection |
| Route record the layer reports | `kind=moe`, `policy=TESSERA_FP8:resident`, `symbol=vllm.fused_moe.modular_kernel:TRITON`, `contract=fp8_per_token_dynamic`, `shape=M17:N512:K512`, `state=served` |

The backend was **selected by the runtime**, not pinned by the route:
`select_fp8_moe_backend` chose `TRITON` out of the thirteen it considered, for
`(kFp8StaticChannelSym, kFp8DynamicTokenSym)` — the keys
`experiments/results/moe_decode_target_probe.json` had already recorded as
supported on this hardware.

## 3. The output, against three references

The kernel's output is compared against three torch computations of the same
MoE, and the middle one is the control that makes the other two readable.

| Reference | rel L2 | max abs |
|---|---|---|
| **W8A8 emulated** — activations quantised per token to E4M3 before each GEMM, exactly the contract the route declares | **0.0143** | 0.0022 |
| Dequantised weights, fp32 activations | 0.0449 | 0.0060 |
| The BF16 source weights | 0.1314 | 0.0148 |

Read them in order. The third is the **wire's** error: a 4-bit-rate trellis
encoding of random Gaussian weights with no Hessian, which is not a quality
claim about anything (real weights, a real Hessian and a real corpus are what
the rate-frontier work measures). The second minus the first is the **A-side
contract**: per-token E4M3 on `x` and again on the intermediate, which is what
`fp8_per_token_dynamic` means and is the same contract the dense FP8 route was
measured under. The first is what is left: 1.4%, the difference between the
Triton kernel's fused silu-and-mul plus its own intermediate quantisation and
a torch emulation of the same steps. That residue is **not** a checked
identity, and it is not claimed as one — the identity in this receipt is the
byte-for-byte tile in §2, which is exact.

## 4. The refusals, each shown firing

A route that decodes correctly but accepts bytes it should refuse is a wrong
tensor waiting for a different checkpoint. Each leg mutates one thing and
records both the exception and whether it is the *expected* gate — a leg that
raises something else is reported as a harness fault, not as a refusal.

| Leg | Refused by | Matched |
|---|---|---|
| One payload byte flipped | `SchemaError: plane-region bytes do not match the declared payload digest` (Tessera's reader, before any decode) | yes |
| `wire_stride` understated by one byte | the route's own loader: `a 83082-byte wire does not fit the group's declared wire_stride=83081` | yes |
| `wire_stride` overstated | `moe_layout.unpack_moe_wires`: `stride 87178 is not what its lengths imply (83082)` | yes |
| Expert count wrong in the sidecar | `create_weights`: `this rank holds 4 of 4 experts and the sidecar declares 5` | yes |
| One group's rung wrong in the sidecar | `parse_tessera_expert_blob`: the wire's `q256` is 1024, the sidecar declares 768 | yes |

The third is the one worth naming: `moe_layout.unpack_moe_wires` had no caller
until this route, and the "stride is not the maximum its lengths imply" check
is the only thing that catches a sidecar disagreeing with the bytes it
describes. It fired on real bytes.

## 5. Scope, and what is still owed

- **One shape, one rung, one box.** 4 experts at 512x256, q256 1024, sm_121.
  Nothing here says a 288-expert GLM layer loads; the memory arithmetic alone
  (wires and tile both resident until `process_weights_after_loading`) is
  untested at that size and is called out in the issue report.
- **No exporter writes these bytes.** The wires in this probe were built by
  `tessera.export.encode_linear_planes` and `tessera.fused.pack_fused` — the
  same calls an exporter would make — but
  `experiments/export_tessera_serving.py` still refuses routed experts at plan
  time. Until that lands, no Tessera MoE checkpoint can exist.
- **No served census, no KL, no `routed_moe` cell.** This is what the loader
  *does*; what has been *served* is a different published fact, and the
  `loader_axes` precedent is that the two are published separately or not at
  all.
- **Expert parallelism, TP inside an expert, and `streamed` are refused**, not
  measured.
