# The v23 platform refusal, run on a gfx1201 device (2026-09-12)

**Scope.** gfx1201 (Radeon RX 9070 XT, RDNA4) under WSL2, ROCm torch
2.11.0+rocm7.2.4, HIP 7.2.53211. This proves the **code path**: that the
platform token is derived correctly on a real HIP build and that the contract's
`unbacked` entry refuses before anything touches a kernel. It says nothing
about gfx1151 (Strix Halo, RDNA3.5), which this box can compile for and cannot
run, and nothing about performance — no WSL2 number is a performance claim.

**What was run.** `tessera.serving.native_ops` against the packaged v23
contract, on branch `rdna35/contract-v23-platform-axis` at
`6f27569366b773b4986d0a2734797fae0976c161`.

```
torch 2.11.0+rocm7.2.4.git5fbd98f3 hip 7.2.53211
device.type cuda                       <- ROCm's torch says "cuda" for an AMD device
backend hip  platform_token gfx1201    <- read off gcnArchName, not get_device_capability()
contract_version 23 tessera.lane-eligibility.v10
executes(gfx1201) {'TESSERA_E2M1_K2': None, 'TESSERA_E4M3_K1': None,
                   'TESSERA_BF16_K1': 'bf16_unquantized'}
has_fp8 False has_fp4 False has_cutlass False
require_native_fp8_quant REFUSED: ... publishes TESSERA_E4M3_K1 as unbacked on
  platform 'gfx1201' (backend 'hip') ... This is an attested absence, not a
  missing build artifact.
require_native_fp4_quant REFUSED: ... publishes TESSERA_E2M1_K2 as unbacked ...
```

**The two findings that are not obvious from the source.**

1. `device.type` is `"cuda"` on an AMD device, so the device type cannot tell
   the backends apart; `torch.version.hip` can, and that is what `_backend()`
   reads. Worth stating because the contract key would otherwise be derived
   from the wrong fact.
2. This ROCm build registers **none** of the three operators — `has_fp8`,
   `has_fp4` and `has_cutlass` are all false. Under the old single sentinel the
   diagnosis on this box would have been "vLLM's compiled CUDA operators are
   not registered", which is a build complaint. The v23 refusal is the true
   one: the pinned runtime has no native route for these bytes on this device,
   and that is a fact about the route, not about what compiled.

**What this is not.** No cell. A cell is a receipt with residency, rungs, an
image digest, a toolchain and evidence, taken on a serve; this is a unit-level
run of the refusal path with no serve behind it, which is why v23 publishes a
gfx1201 platform entry and no gfx1201 cell. The cells arrive at v24.
