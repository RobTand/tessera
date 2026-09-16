# GLM-5.3-Flash A4 routed NVFP4 intake: what one rank holds (2026-09-15)

**Status: intake only.** This is the receipt for the NVFP4 arm of
`experiments/bench_routed_load.py` (tessera#527). It records what one TP2 rank's routed
NVFP4 expert tiles occupy when the whole A4 body has loaded, measured by driving the route's
own loader over the merged A4 checkpoint with no vLLM engine.

It is **not** a serve, not a fit verdict for a full model, and not a KL. What it does and
does not cover is stated in full under [Scope](#scope), and the numbers it does not measure
are labelled arithmetic wherever they appear.

## The artifact

| Item | Value |
|---|---|
| Checkpoint | `/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-artifact-exports/a4/merged-4c384e60` |
| Model | GLM-5.3-Flash, `Glm5NextForConditionalGeneration` |
| Format | uniform `TESSERA_E2M1_K2_R896`, 36,423 units, wire bpp 4.000488, contract v29 |
| Routed stacks | 42, layers 3 to 44 |
| Per stack | 288 experts, hidden 4096, moe intermediate 2048, q256 896 |
| Per stack wire | 3,624,321,024 B; `wire_stride` w13 4,195,630, w2 4,195,583 |
| Checkpoint on disk | 171.21 GiB, 166 shard files, 75,103 tensors |

Every one of the 42 routed stacks carries the same geometry, verified across the part
manifests: identical expert count, rows, rung, wire bytes and wire strides. Layer 45 is the
MTP draft layer; its 864 expert tensors are stock `.weight` in BF16 and are named in the
config's `ignore`, so they are not this route's intake.

## What the producer predicts

The sidecar states `resident_bytes_resident_mode` = 4,076,870,400 B per routed stack. Summed
over 42 stacks that is 171,228,556,800 B = **159.47 GiB at TP1**, so **79.73 GiB per rank at
TP2**.

That figure is the stock modelopt tile arithmetic and nothing else. Per stack, per rank, at
TP2 with local intermediate 1024:

| Tensor | Shape | Bytes |
|---|---|---|
| `w13_weight` | [288, 2048, 2048] u8 | 1,207,959,552 |
| `w2_weight` | [288, 4096, 512] u8 | 603,979,776 |
| `w13_weight_scale` | [288, 2048, 256] ue4m3 | 150,994,944 |
| `w2_weight_scale` | [288, 4096, 64] ue4m3 | 75,497,472 |
| `w13/w2_weight_scale_2`, `w13/w2_input_scale` | per expert, fp32 | 6,912 |
| **Total** | | **2,038,438,656** |

This is a producer-side prediction, not a measurement of what the loader does.

## Method

The bench branches on the family the sidecar declares. The NVFP4 arm drives the route's own
loader:

1. `build_tessera_nvfp4_moe_method(scheme, target, "resident", layer)` on a layer stub whose
   `moe_config` carries the rank, the world and GLM's `swiglu_limit`;
2. `create_weights` **inside a `torch.device` context**, which allocates the stock modelopt
   parameter set on the device. Without the context, `create_weights`' bare `torch.zeros`
   lands the tiles on the host, every unit still reports success, and the footprint reads as
   nearly nothing; the arm asserts the tiles are on CUDA and refuses otherwise;
3. per expert, per shard, in the runtime's own shard vocabulary (`w1`, `w3`, `w2`),
   `_load_wire` parses the full container, cuts it to this rank through
   `sharding.shard_parsed_roles` and decodes it into the expert's slot, with the A-side
   `input_global_scale` loaded beside each wire.

Every layer's tiles are kept alive, exactly as vLLM keeps them until
`process_weights_after_loading`, so the final resident figure is what one rank holds with the
whole body loaded.

**Streaming.** A 42-layer body is ~142 GiB of wire, so the arm reads one layer at a time
behind a bounded read-ahead and drops each layer's wires as its units are placed.

**Guard.** Before each expert the arm reads host `MemAvailable` and memory PSI, and stops
with a written record naming the layer and expert it reached if availability falls below
16 GiB or PSI full avg10 reaches 20. On GB10 the GPU allocates out of host memory, so this
is the number that decides whether a load finishes or takes the box down: in tessera#501 an
A8 load reached 112 of 166 shards and hung both Sparks.

**The vLLM seam.** vLLM's `oracle.nvfp4` is stubbed exactly as the route's own tests stub it
(`tests/test_serving_nvfp4_moe_route.py`). Everything the route owns runs for real; the
backend oracle, the kernel and the finalize-time swizzle are the runtime's and are stubbed.
Vendoring the runtime is forbidden (AGENTS.md principle 5), which is why the
load-and-execute contract is measured on the pinned image by
`experiments/nvfp4_moe_route_load_probe.py` instead.

## Scope

What this measures:

- expert intake for the routed NVFP4 stacks of one rank: `create_weights` and `_load_wire`,
  across all 42 layers.

What it does **not** measure, and what therefore cannot be concluded from it:

- `process_weights_after_loading`, the runtime's kernel-format swizzle. vLLM finalizes layers
  after the whole model loads, one at a time, so this is a bounded per-layer transient rather
  than a doubling — but it is unmeasured here;
- any engine: no KV cache, no activation peak, no CUDA graph pools, no NCCL/collective
  buffers, no CUDA context beyond what the bench itself creates;
- non-expert weights: attention, embeddings, the visual tower, the MTP layer and the dense
  and shared-expert Tessera modules;
- anything about correctness of the served result. This is a footprint, not a KL.

## Pilot: two layers

PB `ee87a05d9930c6301e212fad9e2e13cab6839750c0123f01056771edb9042665`, sparky, rc 0, 39.4 s.
Layers 3 and 4, 1,728 units, config `tp2r0`.

| Quantity | Value |
|---|---|
| `resident_final` | 4,076,877,312 B = 3.797 GiB |
| per layer per rank | 2,038,438,656 B = 1.8984 GiB |
| producer prediction, halved | 2,038,435,200 B |
| difference | 3,456 B, exactly half the 6,912 B of per-expert scalar rows TP does not cut |
| `reserved` | 4,490,002,432 B |
| `inactive_split` | 2,054,144 B (2.05 MiB) |
| segments | 25 |
| `alloc_retries` | 0 |
| transient peak above resident | 344 MiB |
| median seconds per unit | 0.0184 |

The allocator reaches a per-layer steady state: `inactive_split` rises to about 14 MiB while
an expert decodes and returns to about 2 MiB when its planes are dropped, segments plateau at
21 after the first layer and 25 after the second, and no allocation is ever retried. That is
the contrast with tessera#501's packed path, which ran to 1.89 GiB of `inactive_split` across
133 to 151 segments before its fix.

## Whole body: 42 layers, one rank

PB `8e395b43ca512bd11a0b21433ffefdb448fae1aec8d399c00c5428a6e2127d36`, sparky, rc 0, 799.4 s,
admitted `--measurement --tag gb10 --priority -10` with the GPU exclusive. All 42 routed
stacks, layers 3 to 44, 36,288 load units, config `tp2r0`, on the merged checkpoint. The
guard did not fire: `stopped_early` is null.

| Quantity | Value | GiB |
|---|---|---|
| `resident_final` | 85,614,423,552 B | **79.735** |
| per layer per rank | 2,038,438,656 B | 1.8984 |
| `reserved` | 86,027,272,192 B | 80.119 |
| `max_allocated` | 85,959,218,688 B | 80.056 |
| `reserved` − `allocated` | 412,317,696 B | 393.2 MiB |
| `inactive_split`, final | 1,275,904 B | 1.22 MiB |
| `inactive_split`, peak | 14,659,584 B | 13.98 MiB |
| segments | 185 | |
| `alloc_retries` | 0 | |
| transient peak above resident | 344,795,136 B | 328.8 MiB |
| wire read | 152,250,428,734 B | 141.79 |
| host `MemAvailable` at end | 25,991,102,464 B | 24.21 |

**The measurement matches the producer's arithmetic.** Predicted 85,614,278,400 B against
85,614,423,552 B measured: 145,152 B apart, which is exactly 42 x 3,456, the per-layer half
of the 6,912 B of per-expert scalar rows that tensor parallelism does not cut. The sidecar's
`resident_bytes_resident_mode` is therefore a faithful predictor of this route's resident
state, now measured rather than asserted.

**Fragmentation does not accumulate.** This is the term tessera#501 was about, and across 42
layers it stays flat:

| Layer | resident GiB | reserved GiB | `reserved`−`allocated` MiB | `inactive_split` MiB | segments |
|---|---|---|---|---|---|
| 3 | 1.898 | 1.900 | 2.0 | 1.98 | 5 |
| 15 | 24.680 | 25.064 | 393.8 | 1.75 | 69 |
| 27 | 47.461 | 47.846 | 393.5 | 1.53 | 117 |
| 39 | 70.242 | 70.627 | 393.3 | 1.31 | 165 |
| 44 | 79.735 | 80.119 | 393.2 | 1.22 | 185 |

`reserved` − `allocated` pins at about 393 MiB from layer 9 onward and never grows;
`inactive_split` *falls* as layers are added, because each expert's planes are released inside
its own load callback; segments grow linearly at 4 per layer; no allocation is ever retried.
For contrast, tessera#501's packed path reached 1.89 GiB of `inactive_split` over 133 to 151
segments on two layers.

**The decode wrote real bytes.** For layers 3 and 44 alike, every tile is on `cuda:0`, the
`w13`/`w2` packed tiles are 99.85% non-zero, the per-expert gate and up globals are equal
(the joined multiplier, 0.000244140625 = 2^-12), and every A-side `input_global_scale` is
finite and positive. A stubbed or misrouted decode leaves zeros.

### Host series

The run window is 2026-09-16T00:10:38Z to 00:23:58Z.

| Source | Reading |
|---|---|
| GPU power (pqteld, 1,600 samples) | mean 18.35 W = **13.1% of the 140 W envelope**; peak 18.9 W = 13.5% |
| GPU utilization | mean 21.2%, peak 25% — recorded, not diagnostic on GB10 |
| `uvm_residual`, peak | 90,274,349,056 B = 84.08 GiB |
| Unified available, min | 24,633,139,200 B = 22.94 GiB |
| Unified used, peak | 105,960,951,808 B = 98.68 GiB of 121.6 GiB |
| Memory PSI full avg10, max | 19.09 |
| cgroup `memory_peak_bytes` | 80,294,375,424 B = 74.78 GiB |
| Process I/O `read_bytes` | 154,613,571,584 B = 144.0 GiB |
| CPU busy (Netdata) | mean 7.9%, peak 21.8% |
| Netdata `system.ram`, sparky | free 45.2 GB down to 2.97 GB; used 14.3 GB up to 93.1 GB |

Sparky's baseline before the run was 2.7% CPU, 113.1 GiB available and 5 W. Sparklina was
idle throughout and is the unloaded control.

**The intake is not GPU-bound.** 18.35 W is 13.1% of the envelope while 36,288 containers are
parsed, cut and decoded, at a median 19.2 ms per unit and 757 s of the 796 s wall clock inside
the load calls. Useful work per joule is about 9.9 MiB of wire per joule (141.79 GiB over
roughly 14,610 J). That is the same shape tessera#501 diagnosed for the packed path, and it
says an intake speed-up is available; this PR does not attempt one, and no before/after speed
claim is made here.

**Memory PSI reached 19.09**, just under the guard's limit of 20, with a minimum of 22.94 GiB
unified available. The pressure comes from page cache for 144 GiB of wire reads, not from the
tiles: `MemAvailable` ended at 24.21 GiB and the box returned to 114.8 GiB available
immediately after the run.

## What the rest of the body adds (arithmetic, not measured here)

Summed from the merged checkpoint's own tensor headers:

| Category | GiB | Tensors |
|---|---|---|
| Tessera wire (routed and dense) | 141.79 | 36,288 |
| MTP layer 45, passthrough BF16 | 13.84 | 889 |
| Attention, passthrough | 11.28 | 664 |
| Embeddings and `lm_head` | 2.36 | 2 |
| Visual tower | 1.05 | 347 |
| Non-expert MLP | 0.80 | 264 |
| Other | 0.07 | 361 |

The Tessera wire renders to 160.26 GiB resident at TP1, of which the routed stacks are
159.47 GiB and the dense and shared-expert modules 0.79 GiB.

Per rank at TP2, weights alone, as bounds rather than a measurement: **94.5 GiB** if every
term is cut, **108.8 GiB** if the tiles are cut and the whole passthrough is replicated. The
true value sits between them and depends on how vLLM shards each passthrough term, which this
bench does not exercise. Two terms move it most: the MTP layer is resident only when
speculative decoding is enabled, and attention is head-sharded at TP2.

## Against the two figures this was meant to replace

**The ~71.3 GiB/rank "measured-extrapolated" figure is refuted.** The routed expert tiles
*alone* measure 79.735 GiB/rank — 8.4 GiB above that whole-rank estimate, before attention,
embeddings, the visual tower, the MTP layer or anything else is counted. Whatever it
extrapolated from, it cannot be a per-rank total.

**The 104.5 GiB/rank routed projection could not be checked.** `a4/footprint-budget.md` is not
present under the artifact export tree, so its derivation is a missing input and is not
reconstructed here. Taken at face value it is *consistent* with this measurement — it sits
inside the 94.5 to 108.8 GiB weights-alone band above — so this run constrains it rather than
confirming or refuting it.

## Does it clear the 16 GiB floor at TP2?

The pool is 121.6 GiB and the floor is 16 GiB, so 105.6 GiB is usable for everything.

| Quantity | GiB | Against 105.6 |
|---|---|---|
| Routed intake, **measured** | 79.735 | clears, 25.9 spare |
| Weights alone, lower bound (every term cut) | 94.5 | clears by 11.1 before KV |
| Weights alone, upper bound (passthrough replicated) | 108.8 | **fails by 3.2 before KV** |
| Weights alone, realistic band | 88 to 95 | clears, 10.6 to 17.6 left |

The routed tiles clear the floor with real margin, and that is what this receipt measures. **A
whole-rank fit is not established here.** It turns on the two terms this bench does not
exercise — whether the MTP layer is resident (13.84 GiB, only under speculative decoding) and
whether attention is head-sharded — and the upper bound genuinely fails. Those want their own
measurement before a full serve, not another projection.

## Environment

| Item | Value |
|---|---|
| Tessera | this PR's head, run from the checkout's `src` |
| Interpreter | `/home/rob/venvs/pq-cu130-tessera-4c384e60/bin/python`, torch 2.11.0+cu130 |
| Box | sparky, NVIDIA GB10, 130,594,091,008 B unified (121.6 GiB) |
| Admission | PrismaBuild, `--measurement --tag gb10 --priority -10`, GPU exclusive |
| CI test | PB `70e6f87967d7ac2533ea7421fb1fdce4235535570f51c300199a6de235bb2f26`, dl380g10, 5 passed in 17.05 s, 0 skipped, 0 modules not collected |

Both Netdata agents are standalone localhost collectors, so the host series covers sparky and
sparklina only; dl380g10 has no series on either agent and its load is reported from PB's own
`resource_profile` instead.
