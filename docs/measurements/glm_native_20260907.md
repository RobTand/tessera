# GLM 288-expert/top-8 native wire control — 2026-09-07

For #412, the actual GLM-5.3-Flash factory constructs the expected routed
owner on immutable official vLLM, selects the resident Tessera method, and
refuses streamed construction at the existing resident-only gate. At q256
1024 and 768, three real source matrices repeated across 288 slots load through
all 864 stock projection-loader calls. Every decoded FP8 byte and channel
scale matches independent producer materialization, and the whole routed
operator matches the independent stock-FP8 leg exactly in three cases.

This is a **repeated-expert functional control**, not full-model generation,
model-quality evidence, a check of every trained expert, an expert-permutation
proof, or a performance result. It changes no runtime cell, release pin,
default, producer eligibility gate, or TP/EP claim. In particular, execution
of research q768 bytes does not admit that rung through the production exporter.

## Inputs and runtime

- Official image:
  `vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`.
  The observed vLLM is `0.28.1rc1.dev451+g1970f3ed4`; the full 4,967-file
  stock-core inventory is checked before installation, after installation,
  and after each run. No core file is modified.
- Tessera is installed without dependencies from the archived source at
  `a29dbec6cf2f720f1cf91eda26cbbe934a737fcf`. Actual loaded module origins and
  the complete installed source roster are verified independently. The
  canonical encoder-source hash is
  `f47631fada735e3fc98fea378d3fda80a6091b97702bc5a0bec07bc33b94f75d`.
- Source: `/mnt/shared/models/GLM-5.3-Flash-BF16`. Its configuration is
  SHA256 `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`.
  The bounded model retains source layers 0–3, original dimensions and vision
  metadata, with MTP disabled. Construction is on meta; its weights are not
  loaded. The runtime class is
  `vllm.models.glm5next.nvidia.model.Glm5NextForConditionalGeneration`.
- The source tensor names are
  `model.language_model.layers.3.mlp.experts.0.{gate_proj,up_proj,down_proj}.weight`.
  All three live in `model-00057-of-00120.safetensors`. Each 16-MiB BF16
  payload is bound separately to its source-index entry, shard-header hash,
  shape, dtype and byte offsets. The final audit independently rereads the
  original payloads and verifies the copied bytes. This does not hash the
  whole 642-GB model or claim coverage of the other 287 experts.
- [Base request](glm_native_20260907/request.json) SHA256
  `96915ace44fe757381c87b996fb926551a3dd5dd29d21d10a2d087e45dcf44ea`;
  [control request](glm_native_20260907/control-request.json) SHA256
  `5c9ec4e848b80cc61c76822ba8034abb2b8fee6dde0bec779bbed769e461bfb9`.

The constructor observes hidden size 4096, intermediate size 2048, 288
experts, top 8, sigmoid scoring, renormalization, one expert group, routed
scale 2.5, and SwiGLU limit 10. Source targets are mapped by stock
`initialize_model`, not by a handwritten rewrite in the control. The owner is
`language_model.model.layers.3.mlp.experts.routed_experts`, with parameter
shapes `[288,4096,4096]` and `[288,4096,2048]`.

## Functional results

| Observation | q256 1024 | q256 768 |
|---|---:|---:|
| Backend actually selected | TRITON FP8 | TRITON FP8 |
| Loaded projection calls | 864 | 864 |
| All projection tiles and scales exact | yes | yes |
| M=1, eight slots: max absolute delta vs independent stock FP8 | 0 | 0 |
| M=36, all 288 slots: max absolute delta | 0 | 0 |
| M=4, input ×64 clamp stress: max absolute delta | 0 | 0 |
| Padded wire parameter bytes before materialization | 3,643,425,792 | 2,737,446,336 |
| Padded wire bits per quantizable expert parameter | 4.021576 | 3.021565 |
| Resident routed-owner parameter bytes | 7,257,195,648 | 7,257,195,648 |
| PyTorch peak allocated bytes, entire harness | 20,981,247,488 | 20,075,277,824 |

The denominator for wire bpp is only the routed projections:
`288 × 3 × 2048 × 4096` parameters. It includes the padded wire arrays,
not immutable model weights. Both rates decode to the same resident FP8
footprint; the lower wire rate does not lower resident FP8 parameter memory.
Peaks include load temporaries, the producer-reference tensors and a second
full stock-FP8 reference stack. They are not a production serving-memory budget.

The independent stock leg constructs its FP8 weights and scales from the
producer's `materialize_stock` outputs; it does not borrow the consumer's
parameters. Both legs use the stock oracle's FP8 kernel, dynamic per-token
activation quantization, and clamp 10. Controlled top-8 IDs exercise the owner
after routing; **the trained router and shared-expert branch are not exercised**.
The source gate/up arithmetic has 7,739 clipped coordinates in the stress case.

For transparency, the source arithmetic screen uses FP32 arithmetic over the
three BF16 source matrices, with the source clamp and routed scale. It does
not emulate a full BF16 serving stack. Relative L2 for M=1 / M=36 / clamp
stress is `0.123836 / 0.127773 / 0.261827` at q1024 and
`0.235839 / 0.233414 / 0.417650` at q768. These synthetic hidden states are
not the common quality probe or calibration draw, and these errors are not KL.

## Execution and verification

Evidence root: `/mnt/shared/tessera-glm-native-20260907`.
[Final audit](glm_native_20260907/final-audit.json), SHA256
`4ca8feb72bd3c3363378fc1e135b1a7cd57c4dbc26a348db79ac7d8bcbc5377b`,
binds source payloads, controls, all completed attempts and their files.

Six independent producer actions ran through PrismaBuild, two CPUs and
16 GiB total memory each, with a 12-GiB GPU subset and native threads set to
one. The original campaign omitted `anywhere`, so the client's default pinned
all six to Sparky; they were already completed when this was inspected. No
completed encoding was repeated to fill another host. Future portable
campaigns should declare `anywhere: true`. The
[PB audit](glm_native_20260907/pb-evidence-audit.json) verifies all six terminal
exit codes, CAS payload hashes and scope cleanup, plus the CPU test action.

The targeted construction suite ran through PB on `dl380g10`, four xdist
workers, worksteal, native threads one: **29 passed**, zero skips, zero
uncollected modules, torch `2.11.0+cpu`, no CUDA coverage. Action:
`569cf7fe88cd9c2507415084282f76ffd37320204cd655cfdc51a752efa00df2`.

vLLM controls ran directly under Rob's explicit 2026-09-07 vLLM exemption,
in the exact official image, on GB10/Sparky, TP=1/EP=1, eager, native threads
one. Each live consumer had a 48-GiB container memory limit. Both consumers
overlapped one another and the root cost run; CPU masks also overlapped that
run. [Overlap record](glm_native_20260907/external-overlap.json) preserves that
fact. Wall times are not isolated measurements. Stock vLLM warned that no
GB10 tuning configuration existed for E=288/N=2048. No throughput, power,
work-per-joule or performant-serving conclusion follows from this control.

All attempts have exact container inspection and cleanup records. Retained
setup failures are: first construction could not write the root-squashed
evidence mount (fixed by matching its owner); first consumer used the older
vLLM model namespace (fixed by deriving the class from the actual model);
the next consumers encountered the meta model's registered owner (fixed by
retiring that exact owner before its CUDA replacement). None modified vLLM.
Final q1024 and q768 runs exited 0, were not OOM-killed, retained unchanged
control hashes during execution, and removed their exact owned containers.

Reproduction uses `experiments/run_glm_native_construction.py`, first with
`--stage construction`, then `resident` / `streamed` and the construction
receipt. Producer campaign rows use `--stage encode --role ROLE --q256 Q`
and the control request, through PB. The consumer uses `--stage control`,
the same request and construction receipt, and reads the completed producer
artifacts named in the request's directory. Exact commands and image IDs are
in each attempt's `launch.json`; the final successful receipts are
[q1024](glm_native_20260907/control-1024-03.json) and
[q768](glm_native_20260907/control-768-02.json).

## Smallest next shared extension — proposed, not implemented

1. Extend the existing `moe_route` prepared owner to retain validated packed
   windows, using `prepare_tessera_fp8_module` / `PreparedWindow` and existing
   role/stride validation. Keep required packed bytes resident on the GPU;
   source-file reads and parsing remain load-time operations.
2. Establish a deterministic GPU mapping from routed global expert IDs to a
   compact selected stack. First prove the runtime's existing `expert_map`
   argument with compact independent stock FP8 tensors against the full
   resident control. This needs distinct trained experts and nontrivial ID
   permutations: today's repeated expert cannot catch a permutation defect.
3. Decode only those selected experts into functional per-invocation FP8
   tensors and call the same stock FP8 fused-MoE kernel. Reuse the existing
   packed-window decoder; do not build another loader or persistent cache.
   `serving/ops.py` documents why a shared aliased mutable decode pool broke
   compiled serving. Preserve clamp, per-token QDQ, weights and reduction.
   Backend support for compact stacks must be measured, not inferred from
   the presence of the `expert_map` argument.
4. Bound prefill by an explicit workspace/token-chunk contract. From the
   measured geometry, FP8 codes for one expert occupy 24 MiB; a single
   decode token's eight selected experts occupy 192 MiB, while selecting all
   288 requires 6.75 GiB. These are shape-derived code footprints, not measured
   allocator peaks or latency savings. Keep arbitrary batches, repeated IDs,
   all-slot coverage and clamp stress in the gate, then profile before/after
   with both-host telemetry and the full common quality draw.

The current expert builder still rejects `streamed` at `moe_route.py:258`.
Only an implemented and measured path can change that gate. A full single-
Spark capacity plan must also include the source's immutable weights, packed
body bytes, scales/tables, KV and transient tiles; the q768 functional pass
does not establish that the complete model fits. TP2 and EP remain separate
gates. The larger single/dual-Spark GLM goal remains open.
