# Serving a routed-MoE Tessera checkpoint: the cut, the sidecar, and what the census can and cannot say

*2026-09-04. Issue #5 item 5. Branch `wf/ts-5-serve`, off `wf/ts-5` at 77adfcc.*

The write half of the routed-MoE seam landed on `wf/ts-5` (884e841 through
77adfcc, not this branch's work). What it left open, in its own words, was
**"no served census and no KL"**. This is the attempt at that, and the first
thing to say about it is what it is measured on.

## The model is a cut, and every number below inherits that

The only routed-MoE model this box can serve is GLM-5.3-Flash-4layer. Its three
expert stacks are 21.74 G routed parameters and about **3.75 GPU-hours of
encode** — a serving receipt that costs four hours of a shared GPU before the
first load is a receipt nobody takes tonight, and the previous agent scoped it
out for exactly that reason.

So the expert *dimension* is narrowed instead of the ambition.
`experiments/moe_expert_cut.py` keeps experts `0..15` of each stack verbatim,
narrows both router tensors to match (`mlp.gate.weight` and
`mlp.gate.e_score_correction_bias` — a router left at 128 rows over 16 experts
routes tokens to weights that are not there), rebuilds the shard index from what
was actually written, edits `n_routed_experts`, and copies the tokenizer and
processor files byte for byte.

| | source | cut |
|---|---|---|
| routed experts / stack | 128 | 16 |
| `top_k` | 8 | 8 (unchanged) |
| tensors | — | 602 |
| size | — | 7.07 GiB |
| encode | ~3.75 GPU-h | 1012 s |

Same model class, same tokenizer, same real weights, same wire, same loader
path, same tile arithmetic. **Different routing.** Every KL below is
student-against-teacher *on the cut*, which measures the error the Tessera
expert route introduces on these experts. It is **not** a quality claim about
GLM-5.3-Flash: dropping 112 of 128 experts while leaving `top_k` at 8 changes
which experts a token reaches, so the cut's own BF16 teacher is the only honest
denominator, and that is what it is compared against.

## The export

`experiments/moe_export_glm_4layer.sh` on the cut, `--grid E4M3 --q256 1024
--layers 4 --passthrough-unrouted`, through the pool (action `b127654a9fb9`,
sparky, **1012 s**, 60.7-75.8 W against a ~140 W envelope; `gpu_utilization` read
96 % throughout and is non-diagnostic on GB10). Output
`/mnt/shared/tessera-runs/ts5/glm53-4layer-e16-tessera`, 4.9 GB, 598 tensors,
11 shards.

`experiments/ts5_sidecar_check.py` reads it back from safetensors headers and
JSON alone, before any GPU is spent on it:

```
quant_method='tessera' config_groups=11 ignore=159 tensors=598
routed_moe groups=3 other groups=8
  ...layers.{1,2,3}.mlp.experts: experts=16 grid=E4M3 body=WINDOW plane=CHANNEL
    w13: n=32 stride=4215633 max=4215633 spread=5 OK
    w2:  n=16 stride=4219649 max=4219649 spread=1 OK
wires=144  manifest requires_lanes=[]   NO PROBLEMS
```

Three things worth naming.

**The stride is derived three times and agrees three times.** The exporter takes
`max(lengths)` over the blobs it wrote; `moe_layout.unpack_moe_wires` re-derives
it from the loaded lengths and refuses a stride the bytes contradict; this check
recomputes it from the shard headers, sharing no code with either. A checker
that imports the writer it is checking checks nothing.

**The blob length really does follow the data, at GLM shape.** Within one stack,
at one shape and one rung: `gate_proj` 4215632/4215633, `up_proj`
4215628/4215629, `down_proj` 4219648/4219649. One byte, because
`ScalePlane.encode` writes the exact-`Fraction` `global_scale` as two varints
whose width follows their value. That one byte is why the group needs a stride
plus per-blob lengths at all, and it is now measured on 144 real GLM containers
rather than argued from the encoder.

**Eleven config groups, and the eight non-MoE ones are the control.** The
construction census (`docs/measurements/construction/glm53-flash-4layer.json`)
says exactly four Linear patterns are `offered_to_quant_config` on this model —
layer 0's `mlp.down_proj` and `mlp.gate_up_proj`, and layers 1-3's
`shared_experts.*` — which is eight modules, and eight dense groups is what the
exporter wrote. The other twenty patterns are never offered (vLLM builds them
with `quant_config=None`), so `--passthrough-unrouted` leaves them BF16 and the
export writes no wire for them. That is also why this export sidesteps issue
\#99's dense-GLM blocker: the module that fails there, `self_attn.f_b_proj`, is
one of the twenty.

## The census had to be fixed before it could say anything

`tools/tessera_route_census.py` joined its route records to the checkpoint's
`config_groups` targets **by name, in two different namespaces**. The records
come off `named_modules()`; the targets are written in the checkpoint's own
namespace. On GLM-5.3-Flash the model class's `hf_to_vllm_mapper` rewrites
`model.language_model.` to `language_model.model.`, so *nothing joined* — every
declared module would have been reported as reporting no route, on a healthy
serve, and every route as undeclared.

The fix replays the model's own unstacked mapper over the targets before
matching — the same translation `TesseraConfig.apply_vllm_mapper` makes at load
— and records both the mapping and the fact it was applied in the receipt. A
target the mapper drops maps to `None` and is reported, not silently identified
with itself. `tests/test_route_census_module_space.py` covers the five cases
(no mapper, a mapped path, a dropped target, a bare class name, a `re:` pattern);
27 tests pass on the pool (action `82eb43a4c855`).

This is a **defect the dense census never exposed** and would have made the first
MoE census unreadable.

## What the census can still not say, and why that is not a bug to fix tonight

Two things will make the MoE half of the census report problems even on a
working serve, and they are separable:

1. **`MoERunner` wraps `RoutedExperts` as `.routed_experts`.** The route record
   is stamped at `...mlp.experts.routed_experts`; the mapped declared target is
   `...mlp.experts`. Off by one module.
2. **`ROUTE_LAUNCHES` publishes no MoE entry**, so `_expected` cannot own the
   `(vllm.fused_moe.modular_kernel, torch_materialize_stock)` pair the expert
   route launches under.

Both are edits to the census's tables, not to the serving path, and neither
should be made blind: the right time to write a `routed_moe` launch pair is when
a served census has printed what the launch actually is. The eight dense groups
are the control that says the mapper fix works and these two are MoE-specific.

## Status of the serve

*Written while the serve was still running; this section is superseded by
whatever the arms below actually reported.*

`experiments/ts5_moe_served.sh` takes three serves sequentially, because the box
has one serve lock and the two arms of a KL must not be resident at once:

1. the BF16 cut through the stock GLM image → teacher logprobs;
2. the Tessera cut through **Tessera's own plugin** → student logprobs + KL;
3. the Tessera cut a second time, on purpose, for the route census — the route
   record lives on the layer objects inside the worker and an OpenAI-protocol
   serve cannot hand them out.

The corpus is the GLM-tokenized default `corpus_n8_s512.json` (n=8 x 512, 4088
scored positions), and the cut and the export both carry the source's
`tokenizer.json`/`tokenizer_config.json` byte for byte
(`19e77364…`/`98b12715…`), which is what `kl_tool`'s tokenizer-identity gate
compares — so it passes without `--allow-tokenizer-mismatch`.

`--gpu-memory-utilization 0.15` (18.24 GiB of 121.63) is not a round number: the
teacher's own startup accounting says 8.34 GiB of weights plus 4.42 GiB of peak
activation, so 12.76 GiB is the floor and 0.15 clears it by 5.5 GiB of KV. The
pool action declares `mem_gb=20` to match. The earlier draft's 0.35 would have
reserved a third of a shared box for a 7 GiB model.

## The load hop is closed: vLLM read the routed-MoE checkpoint back

This is the result the branch's two docs named as unknown, and it is now
measured. From `served/census.log`, on the pinned Mia GLM image
(`prismaquant/glm53-mia-sm121:487ecf187`, digest `sha256:75ea13ed…`, vLLM
`0.1.dev20051+g487ecf187`):

```
[tsrun] ... plugin ['tessera', 'lora_filesystem_resolver', 'lora_hf_hub_resolver']
INFO ... Initializing a V1 LLM engine ... quantization=tessera, quantization_config=None
Loading safetensors checkpoint shards: 100% Completed | 11/11
INFO ... [model_runner.py:374] Model loading took 5.81 GiB and 29.296122 seconds
WARNING ... [fused_moe.py:1161] Using default MoE config ... E=16,N=2048,device_name=NVIDIA_GB10,dtype=fp8_w8a8
```

Four things in those five lines, none of which had a receipt before:

- **The checkpoint chose the plugin.** `quantization=tessera` with
  `quantization_config=None` on the command line: nothing enabled it, the
  sidecar did.
- **Every shard loaded.** 11/11, no `KeyError`. The model-level hop —
  `Glm5NextModel.load_weights` handing 144 `.wire` names through
  `expert_params_mapping` to the expert loader — is the hop both
  `tessera-moe-export-seam` and `tessera-moe-route-load` left open, and it
  carries weight.
- **5.81 GiB of weights**, against 4.9 GB of Tessera bytes on disk plus the
  BF16 remainder.
- **The expert method was built and configured**: vLLM looked for a fused-MoE
  tuning config at `E=16,N=2048,…,dtype=fp8_w8a8` — E=16 is the cut's expert
  count reaching the runtime's own MoE machinery, and `fp8_w8a8` is the
  `TESSERA_FP8` family's A-side.

**What it did not do is serve a token.** The load was followed by
`Available KV cache memory: -2.02 GiB` and vLLM refused to build a cache. That
is not a Tessera failure: the census drives `LLM(...)` under vLLM's default
chunked-prefill budget of 8192 batched tokens, so its profiling peak is several
times a `--max-num-seqs 8` serve's, and the 18.24 GiB the *serve* needed is not
enough for the *census*. One number was doing two jobs; the driver now has
`TESSERA_CENSUS_MEM_UTIL` (0.35) separate from `TESSERA_GPU_MEM_UTIL` (0.15).
