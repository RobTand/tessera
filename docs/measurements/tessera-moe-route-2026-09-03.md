# The MoE route: what the plan-time guard refuses, what the wire layout is, and what the encode costs (2026-09-03)

Issue #5, the routed-MoE half. This receipt covers the four legs that can be
settled without a served expert forward, and says exactly what the fifth needs.

**Scope, stated first.** There is still **no expert route**. `scheme.ROUTES`
holds three dense families and `scheme.STRUCTURES` is `("dense",)`, so a
checkpoint declaring `structure: "routed_moe"` is refused by name at
`get_quant_method`; the packaged `runtime_contract.json` (v10) says the same as
a value rather than as prose — its `expert_parallel` block is
`{"axis": "vllm_expert_parallel_size", "semantics": "closed_world", "units": []}`,
an empty unit list, which is the machine-readable form of "this build publishes
no expert cell". Nothing below changes that. What is settled is the plan, the
layout, the cost and the decode target.

---

## 0. Where the issue's own text is out of date

Three of #5's premises moved before this receipt, and a reader who takes the
issue body at face value will redo landed work:

| #5 says | what is true on `42615e4` |
|---|---|
| `quantizable()` filters `startswith("model.layers.")`, so GLM-5.3-Flash's `model.language_model.layers.*` "drops everything" | `BODY_LAYER = ^model\.(?:[^.]+\.)*layers\.(\d+)\.` matches the sub-model prefix; `body_layer()` reads the index off that match, not off `name.split(".")[2]` |
| the 864 per-expert 2-D weights "fall through the 2-D path as individual dense modules" | `quantizable` returns a fourth bucket, `routed_shapes`, and `main` refuses a plan that names one **before** `args.out.mkdir` |
| the `[E, 2, nbytes]` blob-length problem needs padding plus companions | `src/tessera/moe_layout.py` is exactly that, with the round trip tested |
| "~4.9-5.7 s per unit ... ~10x below the fused-Viterbi rate elsewhere, so the fused path may not be being taken" | the seconds are right and the inference from them is wrong. The fused path **is** taken — 4 of 4 calls, 0 to the reference, `_step` at 96.03% of self-CUDA in the contended arm and 96.14% in the exclusive one, from both result files (§3). The rate cliff the suspicion rests on was fixed besides: `WINDOW_FUSED_MAX_RATE` was 7 because of a 690-byte-per-thread register spill in the class-minimum scan at R = 8, and with the scan spelled as a runtime loop it is **11** (`encode.py:826-857`), so rate 4 is nowhere near it |

Landed in `8ddd0a2` / `04119df` / `91cae45` (plan-time half) and `c9561e3`
(the wire layout). This branch adds the measurements and the decode-target
attestation, not a second guard.

---

## 1. The plan-time guard, on the real names

`GLM-5.3-Flash-4layer` carries **2592** routed-expert leaves — 3 MoE layers
(`first_k_dense_replace = 1` of 4) x 288 experts x 3 projections — as unpacked
2-D tensors under `model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`.

A plan naming one is refused, and the refusal is the whole of the run:

```
$ PYTHONPATH=src python3 experiments/export_tessera_serving.py \
    /mnt/shared/models/GLM-5.3-Flash-4layer /home/rob/tmp/ts5-scratch/should-not-exist \
    --grid E4M3 --q256 1024 --device cpu --plan-json plan_expert7.json
the plan names 1 ROUTED expert tensor(s), e.g.
model.language_model.layers.1.mlp.experts.7.gate_proj.weight [2048, 4096]. A routed
expert is not a dense Linear: vLLM builds one FusedMoE module per layer, not one Linear
per expert, so a checkpoint declaring
model.language_model.layers.1.mlp.experts.7.gate_proj in config_groups names a module
vLLM never creates and the plugin refuses it at load. There is no routed-MoE export yet -- it needs a `moe`
block this exporter does not write and an expert route the plugin does not have
(issue #5). Remove them from the plan to pass them through as BF16.
$ ls /home/rob/tmp/ts5-scratch/
plan_expert7.json
```

**The output directory was never created.** That is the evidence that the
refusal precedes the encode: `args.out.mkdir(parents=True, exist_ok=True)` is
the first thing `main` does after the plan checks, and it did not run. The
issue's worst case — hours of encode, then a load-time refusal — is not
reachable through this path.

Tests, all four failure modes, `tests/test_export_moe_layouts.py`:

```
$ pbrun.py --cwd /home/rob/tmp/ts5 --env PYTHONPATH=src -- \
      python -m pytest tests/test_moe_wire_layout.py \
      tests/test_export_moe_layouts.py tests/test_serving_moe_dispatch.py -q
36 passed in 7.64s
```

`test_a_routed_expert_leaf_is_never_fused_as_a_dense_pair` pins the exact case
#5 names (`...mlp.experts.7.gate_proj.weight`), and
`test_planning_a_routed_expert_is_refused_before_any_encode` pins that no
`*.safetensors` is written.

`experiments/moe_plan_baseline.py` is the harness that keeps this honest across
changes: it runs the exporter end to end on the CPU over five expert layouts
(GLM/Mixtral/LFM2 unpacked, transformers-5 packed with and without biases) and
digests what each wrote. Its `plan/<case>/<kind>` rows read either the refusal
or `ENCODE REACHED <tensor>` — the mis-plan in one line. This branch adds the
second REAL row: it now classifies **both** source layouts that exist on this
box, `Qwen3.8-Flash-Next` (packed) beside `GLM-5.3-Flash-4layer` (unpacked), as
digests rather than 2592-name lists, and records each packed stack's
orientation beside them.

One thing to know before diffing that baseline across this branch: a
`plan/<case>/<kind>` row is the exporter's refusal **verbatim**
(`_plan_row` returns the raised message), so the routed-expert rewording later
in this branch moves those rows too. The 25-of-62 diff recorded with the
classification change was taken before that rewording and is a record of the
classification change alone; a run today moves both sets, for two unrelated
reasons.

---

## 2. The wire layout

`src/tessera/moe_layout.py`, landed in `c9561e3` and unchanged here.

The problem it solves: the wire blob's length is **not** a function of
`(shape, grid, q256)`. The manifest writes `global_scale` as an exact varint
ratio whose width follows its value, which follows the data — 4215545 against
4215544 bytes on #5's pair, while `exact_bytes` is flat. So `[E, 2, nbytes]`
with one stride is bytes that fit all rows but one.

The layout: `w13_wire uint8 [E, 2, S13]` (gate at `[:, 0, :]`, up at
`[:, 1, :]`) with `w13_wire_len long [E, 2]`; `w2_wire uint8 [E, S2]` with
`w2_wire_len long [E]`. The strides are the **maxima over the blobs being
packed**, derived per pack, so a stride is never a constant carrying slack.
Padding is the dtype's zero and is never read back. `unpack_moe_wires` returns
each `[e, p, :length]` prefix exactly, because `fused.parse_fused` refuses
trailing bytes: a padded blob handed back whole is a refusal there, not a
shorter read.

Four refusals, each by name: a length past its row's declared stride, a length
tensor whose shape disagrees with the expert count, one that disagrees with the
projection count, and a stride that is not what the lengths imply.

`tests/test_moe_wire_layout.py` round-trips the 4215545-vs-4215544 case: two
rows whose lengths genuinely differ, packed at one stride, unpacked
byte-for-byte, each slice parsed by `parse_fused` with nothing trailing.

---

## 3. The encode rate, and which machine ran

`experiments/moe_encode_rate_profile.py`, result
`experiments/results/moe_encode_rate_profile_contended.json`, on
`c49e7a9` (this branch), torch 2.11.0+cu130, NVIDIA GB10, through the pool.
Weights are layer 1's routed experts read out of `GLM-5.3-Flash-4layer`'s own
shards; the recipe resolves as `TESSERA_FP8`'s — `wire_recipe(E4M3, 1024)` ->
body `WINDOW`, span 1, plane `CHANNEL`, `window_bits = 14`. Weights-only arm:
no Hessian, so no LDLQ and no activation-aware refit. A Hessian-fed export
calls the encoder differently (#94) and does not inherit this number.

**The fused path is taken. This is a count, not a reading of a log.**
Per unit, every unit, identically:

```
{'calls_by_rate': {'4': 4}, 'reached_viterbi_window_fused': 4,
 'took_the_reference': 0, 'viterbi_window_calls': 4}
```

Four calls into `encode.viterbi_window`, four of them reached
`window_viterbi.viterbi_window_fused`, **zero** took the pure-torch reference,
all at rate 4. `fused_available()` is `true` and `WINDOW_FUSED_MAX_RATE` is
`11`, so rate 4 is nowhere near the gate. The profiler says it a second way,
in its own vocabulary:

```
"top_self_cuda": [{"name": "_step", "self_cuda_us": 8618429.7, "share": 0.9603,
                   "calls": 1056768},
                  {"name": "aten::copy_", "self_cuda_us": 74494.9, "share": 0.0083, ...}]
```

The Triton `_step` is **96.03%** of self-CUDA time (8.618 s of 8.974 s), and
the next entry is two orders of magnitude down. #5's suspicion — "~10x below
the fused-Viterbi rate elsewhere, so the fused path may not be being taken" —
is answered: the rate **is** the fused path's rate.

**Per-unit seconds, and the contention that qualifies them.** Six units,
`verify=True`:

| tensor | shape | seconds | wire bytes | bpp |
|---|---|---|---|---|
| `experts.0.gate_proj` | `[2048, 4096]` | 10.04 | 4214784 | 4.0195 |
| `experts.0.up_proj` | `[2048, 4096]` | 9.67 | 4214784 | 4.0195 |
| `experts.0.down_proj` | `[4096, 2048]` | 10.19 | 4218880 | 4.0234 |
| `experts.1.gate_proj` | `[2048, 4096]` | 9.85 | 4214784 | 4.0195 |
| `experts.1.up_proj` | `[2048, 4096]` | 9.99 | 4214784 | 4.0195 |
| `experts.1.down_proj` | `[4096, 2048]` | 9.92 | 4218880 | 4.0234 |

Mean **9.94 s/unit** — and that is **twice** the standing figure, so before it
is used to schedule anything: the box was not mine. The in-process sample
`idle_power_w` came back at **67.94 W** before the first encode, and the
box-level series says why. Netdata `nvidia_smi.gpu_power_draw` on sparky, the
instrument no in-process profiler can replace:

```
03:34:10Z 79   03:35:30Z 79   03:37:30Z 82   03:39:20Z 88   03:40:40Z 71
03:34:30Z 78   03:36:30Z 76   03:38:10Z 82   03:40:00Z 85   03:41:30Z 70
   <- seven and a half minutes BEFORE this run started ->    | run starts 03:41:46Z
03:41:50Z 71   03:42:20Z 72   03:42:50Z 71   03:43:10Z 68   03:43:30Z 57
```

63-88 W for the whole quarter-hour, including the seven minutes before my
process existed. The contract doc's §9.4 measured this encoder against **6-17 W
idle**. So a second GPU job held sparky's other slot throughout, and **9.94
s/unit is a contended upper bound, not a clean rate**. Read against the
envelope rather than utilisation, the box sat at ~0.5 of ~140 W the entire
time — which is also §9.4's finding about the encoder alone, so power alone
cannot separate the two jobs *within this window*; only holding the box can,
and the next arm is exactly that.

**The exclusive arm, and what the pair proves.** The same harness at
`--experts 1` under `pbrun --exclusive --gpu-capacity 2`
(`experiments/results/moe_encode_rate_profile_exclusive.json`, tree `721c81e`):

| tensor | shape | seconds |
|---|---|---|
| `experts.0.gate_proj` | `[2048, 4096]` | 5.07 |
| `experts.0.up_proj` | `[2048, 4096]` | 5.09 |
| `experts.0.down_proj` | `[4096, 2048]` | 5.46 |

**5.21 s/unit** over the whole expert — **5.08 s** for the two `[2048, 4096]`
projections alone, which is the shape the contract doc's E2M1x2 row was
measured on, so that is the figure to compare against it (14.1x, not 14.5x).
Both shapes carry 8,388,608 parameters, so a campaign runs at the
three-projection mean either way. And this time the box is provably mine: in-process
`idle_power_w` = **13.59 W**, inside §9.4's 6-17 W idle band; the box-level
series agrees and is unambiguous:

```
04:15:30Z 14   04:16:40Z 14   04:17:50Z 15   04:18:40Z 14   <- idle, three and a half minutes
04:19:10Z 69   04:19:20Z 64   04:19:30Z 70                  <- the encode, and only the encode
04:19:40Z 15   04:20:00Z  6   04:20:10Z 25                  <- idle again
```

Against the contended window's 63-88 W *before the run began*, that is the
whole explanation. **The two arms are a matched pair and the difference is
occupancy, not the tree:** the profiler counted **1,056,768** `_step`
invocations in *both* runs — identical work, 96.03% vs 96.14% of self-CUDA —
at 9.94 s and 5.21 s of wall. The contended arm was **1.91x** slower for doing
exactly the same thing.

So there is no regression from #94 or the #50/#75 refit landing, and #5's own
estimate was right:

| | s/unit | one MoE layer (864 units) | the 4-layer model (2592) |
|---|---|---|---|
| #5's estimate | 4.9-5.7 | — | — |
| §9.4, 2026-09-02, older tree | 5.0 | ~72 min | ~3.6 h |
| **here, exclusive, current tree** | **5.21** | **75 min** | **~3.75 h** |
| here, contended, current tree | 9.94 | ~143 min | ~7.2 h (what sharing costs) |

**Schedule against 5.21 s/unit and hold the box.** The encode leg is hours, not
minutes; sharing sparky nearly doubles it, and the in-process profiler cannot
see that happening — only the power series can.

**A pool defect found on the way, reported not worked around.**
`pbrun --exclusive` computes its demand as `demand["gpu"] = args.gpu_capacity`
with `--gpu-capacity` defaulting to **4** (`tools/pbrun.py:196,234`), while the
live workers declare `gpu: 2` (sparky) and `gpu: 1` (sparklina). A box-local
checkout is tag-pinned to its own box, so `--exclusive` alone asks one box for
twice the slots it has and can never be scheduled:

```
pbrun: no live worker can run this action.
  required tags: ['sparky']
  demand:        {'gpu': 4, 'mem_gb': 16}
```

It then **exits 0**, so a caller that checks the status code reads a refusal as
a success. `--exclusive --gpu-capacity 2` schedules correctly, which is the
documented flag rather than a bypass.

---

## 4. The decode target, attested

Before a route is written, the question that decides its **quality** is whether
the runtime's fused-MoE kernels will take a **per-channel** weight scale.
Tessera's dense `TESSERA_FP8` route decodes the window body over the CHANNEL
plane to a per-channel FP8 pair — E4M3 bytes plus one fp32 scale per output row
— and the plane is not decoration: deleting it costs 0.77-0.84x. An expert
route that had to fold 2048 row scales into one per expert would pay that.

`experiments/moe_decode_target_probe.py` / `.sh`, result
`experiments/results/moe_decode_target_probe.json`, in the pinned build
`prismaquant/glm53-mia-sm121:487ecf187` (`vllm 0.1.dev20051+g487ecf187`, torch
2.13.0+cu130), **on the GPU** — `"cuda_available": true`, `"device": "NVIDIA
GB10"`.

The question is put to the runtime's own predicate.
`select_fp8_moe_backend` decides by calling each candidate kernel's
`is_supported_config(config, weight_key, activation_key, activation_format)`,
so the probe calls exactly that, over every backend the build lists, for three
weight/activation pairs: the one Tessera decodes to, and two controls.

**Why the GPU run is the one that counts.** Without a device the same table
answers `"kernel does not support current device None"` for every pair — the
controls included — so a CPU answer is an answer about the harness. On sm_121
the controls separate, which is what makes the row about per-channel readable:

| pair | supported (backend x kernel x activation-format) |
|---|---|
| **`per_channel_w_per_token_a`** — what Tessera's FP8 route decodes to | **4** of 38 |
| `per_tensor_w_per_tensor_a` — the stock non-block arm (control) | 3 of 38 |
| `block128_w_block128_a` — the stock block arm (control) | 6 of 38 |

The four that take a per-channel weight scale with per-token dynamic
activations, quoting the runtime:

```
MARLIN          MarlinExperts          True    (standard)
HUMMING         HummingIndexedExperts  True    (standard)
TRITON          TritonExperts          True    (standard)
BATCHED_TRITON  BatchedTritonExperts   True    (batched_experts)
```

**Per-channel is served by more kernels than the granularity the stock method
asks for**, not fewer: three for per-tensor against four for per-channel, and
`HummingIndexedExperts` refuses per-tensor by scheme while accepting
per-channel. The refusals are informative too and are the runtime's own words:
`FLASHINFER_CUTLASS` and `DEEPGEMM` refuse per-channel **by scheme**, while
every `*_CUTLASS`, `FLASHINFER_TRTLLM`, `XPU`, `CPU` and `HPC` entry refuses
**by device** on GB10 — the same shape as the NVFP4 MoE finding that needs
`--moe-backend flashinfer_b12x` here.

The granularity is also carried by the config object, one level up, and the
runtime reports it rather than being told:

```
per_out_channel_w_per_token_a -> fp8_w8a8 True  True  [8, 4096, 1] [8, 4096, 1]
per_tensor_w_per_tensor_a     -> fp8_w8a8 False False [8, 2]       [8]
```

(`config_name(torch.bfloat16)`, `per_out_ch_quant`, `is_per_act_token`,
`w1_scale.shape`, `w2_scale.shape`.) `fp8_w8a8_moe_quant_config` accepts
per-out-channel scales, keeps `is_per_tensor` false, and sizes `w1_scale` as
`[E, 2I, 1]` and `w2_scale` as `[E, H, 1]` — one scale per output row per
expert, which is exactly the shape the CHANNEL plane decodes to.

**So the route may keep its scale plane.** That is the finding: the 0.77-0.84x
that folding the plane away would cost is not a price this route has to pay,
and the constraint that would have decided against a Tessera expert route
before it was written does not exist on this build.

**What it does not say — and this half is read off the runtime's source, not
recalled.** A kernel that *would* take a per-channel scale is never handed one
unless some method asks for it. The `stock_keys` leg records which `QuantKey`
constants `Fp8MoEMethod`'s own functions name, in the pinned image
(`experiments/results/moe_decode_stock_keys.json`, run without `--gpus` because
`inspect.getsource` needs no device):

```
__init__                     -> ["kFp8Dynamic128Sym", "kFp8DynamicTensorSym",
                                 "kFp8Static128BlockSym", "kFp8StaticTensorSym"]
get_fused_moe_quant_config   -> []
```

**`kFp8StaticChannelSym` is absent.** The stock method asks for per-tensor or
128-block and nothing else, so a Tessera expert route cannot inherit the stock
selection: it must call `select_fp8_moe_backend` itself with the channel key.
That is §6 item 3, and it is now a quotation rather than a belief.

And none of this is a route: there is still no `ROUTES` entry, no `apply`, no
`routed_moe` in `scheme.STRUCTURES`.

---

## 5. Both source layouts

| leg | unpacked per-expert 2-D (GLM-5.3-Flash) | transformers-5 packed 3-D (Qwen3.8-Flash-Next) |
|---|---|---|
| classified into its own bucket | yes, `routed_shapes` | yes, `expert_shapes` |
| recognised without a `.weight` suffix | n/a (leaves carry one) | yes — `probe` matches the patterns in the `.weight` spelling, and a bare name that fails the rank test is refused rather than filed (`04119df`) |
| refused when a plan names one | yes, with the tensor and its shape | yes, with the tensor, its shape and its orientation |
| passed through as BF16 and named in `ignore` | yes, at the `<moe>.experts` prefix vLLM builds | yes, same prefix |
| a real source on this box | `/mnt/shared/models/GLM-5.3-Flash-4layer`, 2592 leaves | `/mnt/shared/models/Qwen3.8-Flash-Next`, 98 stacks |
| orientation decidable | n/a | yes, `out_first` both: `gate_up_proj [512, 1280, 2560]` against hidden 2560 / moe\_intermediate 640, `down_proj [512, 2560, 640]` |
| **encoded** | **no** | **no** |

Both layouts are handled at PLAN time and neither is encoded. The asymmetry
worth recording is the orientation: a packed GLM-5.3-Flash stack would be
`[E, 4096, 4096]` (hidden 4096, 2 x moe\_intermediate 4096) and **no** dim
comparison can orient it, which is why `packed_expert_orientation` refuses
rather than guessing. Qwen3.8-Flash-Next is orientable, so the packed leg has a
real source to be built against when it is built.

---

## 6. What step 5 still needs

Step 5 is the served census and KL on the 4-layer model. In order, each item
blocking the next:

1. **The exporter's `moe` block.** Encode one unit per (expert, projection);
   frame each as a one-member `tessera.fused` container; write it under a
   suffix the expert mapping routes (`...experts.E.<proj>.wire`, measured in
   `docs/measurements/tessera-moe-wire-loader-2026-09-03.md`); declare the
   module in `config_groups` with `structure: "routed_moe"`, the per-role rungs,
   the expert count and the two strides — the strides must be in the sidecar
   because `create_weights` sizes the parameter before any tensor is loaded.
2. **`create_weights` for the expert route**, registering each wire parameter
   **with its own `weight_loader`**. Not a preference: the stock dispatch tests
   for the substrings `weight` and `scale` in a name rewritten to
   `...experts.w13_wire`, matches neither, and returns `False` without raising
   (measured, same receipt). The length companion is filled from the loaded
   tensor's own extent, which only a loader seeing both can do.
3. **`process_weights_after_loading`**, decoding every expert's wire to the
   stacked tile §4 attests, and `get_fused_moe_quant_config` returning the
   granularity §4 attests is accepted.
4. **`apply`.** `FusedMoEMethodBase.apply` is not abstract but raises, so a
   route must write it; `Fp8MoEMethod.apply` is three lines over a modular
   kernel built in `_setup_kernel`, which is the shape to follow.
5. **`routed_moe` in `scheme.STRUCTURES`**, plus a `lane_eligibility` cell and
   an `expert_parallel` unit in `runtime_contract.json` — and by this repo's own
   rule a cell appears only when a container receipt covers it, so the cell
   comes **after** the census, not before.
6. **The census and the KL.** `tools/tessera_route_census.py` over a served
   4-layer arm, then KL against the BF16 teacher at matched bytes.

**Cost of the encode leg, so it can be scheduled rather than guessed:** one MoE
layer is 288 x 3 = 864 units and the whole 4-layer model is 2592, at **5.21
s/unit** on an exclusive box (§3) — **75 min per layer, ~3.75 h for the model**.
Sharing the box costs 1.91x of that, so this leg wants `--exclusive`.

---

## Provenance

- Tree: `claude/ts-5-moe-route`, base `master` `42615e4`.
- Harnesses added here: `experiments/moe_encode_rate_profile.py` (+ its result),
  `experiments/moe_decode_target_probe.py` / `.sh` (+ its result).
- Harness extended here: `experiments/moe_plan_baseline.py`, second real row
  plus packed orientation.
- Serving build: `prismaquant/glm53-mia-sm121:487ecf187` =
  `vllm 0.1.dev20051+g487ecf187`, torch 2.13.0+cu130.
- All GPU work through the PrismaBuild pool. Two pool facts belong with the
  numbers rather than in a thread: the profile ran on a **shared** box and the
  Netdata series in §3 is the proof, and `pbrun --exclusive` alone cannot be
  scheduled on either GB10 because its demand defaults to 4 GPU slots against
  declared capacities of 2 and 1 — reported in §3, not worked around.
