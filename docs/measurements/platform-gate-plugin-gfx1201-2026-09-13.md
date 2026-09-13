# The plugin's platform gate, run on a gfx1201 (#457)

**Scope, first, because it bounds everything below.** This is a **gfx1201**
(RDNA4) device under **WSL2**, and it proves a **code path**. It is not
evidence about gfx1151 (Strix Halo) numerics — that silicon cannot be executed
here, only compiled for — and **no number on this page is a performance
claim**. Nothing here claims an AMD *serve*: no artifact was loaded and no
forward was run.

## What was run

| | |
|---|---|
| box | `wsl-gpu`, Ubuntu 26.04.1 on WSL2 |
| device | AMD Radeon RX 9070 XT, `gcnArchName` = `gfx1201` |
| torch | `2.11.0+rocm7.2.4.git5fbd98f3`, `torch.version.hip` = `7.2.53211` |
| vLLM | `0.30.0.dev0` (the gfx1201 ROCm build) |
| tessera | `rdna35/457-platform-gating` |
| receipt | `agents/receipts/t457_plugin_gate_gfx1201.txt` on the box; copied to `/home/rob/tmp/agents/t457-platform-gating/receipts/` |

`TesseraConfig.from_config(...)` was built for one dense config group at a
time and `get_quant_method(layer, prefix)` called on a real `LinearBase`
subclass, inside the ROCm vLLM. That is the exact call vLLM's own dispatch
makes.

## The collision, live

```
torch.cuda.get_device_capability(0)  ->  [12, 0]
torch.cuda.get_device_properties(0).gcnArchName -> "gfx1201"
backend.probed_platform_token()      ->  "gfx1201"
```

`(12, 0)` is the tuple NVIDIA's **sm_120** answers with. This is why a platform
key is read from `gcnArchName` and never from a capability, and why
`tools/tessera_route_census.py` no longer mints its join key from one.

## What the contract says here

`contract_version` 23, `lane_eligibility.platforms.gfx1201.executes`:

```json
{"TESSERA_E2M1_K2": null, "TESSERA_E4M3_K1": null,
 "TESSERA_BF16_K1": "bf16_unquantized"}
```

## What the plugin did

| family | outcome |
|---|---|
| `TESSERA_E4M3_K1` | **refused**, `NativeKernelUnavailableError` |
| `TESSERA_E2M1_K2` | **refused**, `NativeKernelUnavailableError` |
| `TESSERA_BF16_K1` | **built**, `TesseraBf16LinearMethod` |

Both refusals, verbatim (the family differs, nothing else):

> `tessera target 'model.layers.0.q_proj': the pinned runtime contract
> publishes TESSERA_E4M3_K1 as unbacked on platform 'gfx1201' (backend
> 'hip'): its lane_eligibility platform entry executes null for this family,
> so there is no native route for these bytes on this device. This is an
> attested absence, not a missing build artifact.`

The message carries the contract's word (`unbacked`), the platform token and
the payload family, and it is raised from `get_quant_method` — before a weight
is created, before `process_weights_after_loading`, and before the route
module or any HIP kernel is reached.

## What the record stamps

```json
{"kind": "dense", "policy": "TESSERA_BF16:resident", "symbol": "torch.mm",
 "tile_m": 0, "shape": "M1:N256:K256", "contract": "bf16_unquantized",
 "state": "served", "reason": null, "decoder": "torch_window",
 "platform": "gfx1201"}
```

`ROUTE_FIELDS` ends in `platform`. Before #457 this record was byte-identical
to the one an `sm_121` serve of the same artifact writes, and a census could
join either to the `sm_121` cell.

## Two defects this run found that no stub could

1. `native_ops._platform_token()` answers `None` for "no device names a
   platform"; passing that through to `require_platform_backs(platform=…)` was
   read there as "the caller did not say", so the gate probed the device and
   found the gfx1201 the caller had just declined to claim. Invisible on CPU
   and NVIDIA, where the two readings agree.
2. Seven `test_serving_dispatch` cases build FP8 and NVFP4 methods to prove a
   checkpoint reaches its route. On this device every one refused — correctly,
   and with nothing to say about dispatch. They now pin the platform they
   describe.

## Not established here

- No Tessera artifact was loaded and no forward was run on this device. The
  serving half of `TESSERA_BF16_K1` on AMD is unmeasured.
- gfx1151 is untouched: it is a compile target on this box, never an
  execution one.
