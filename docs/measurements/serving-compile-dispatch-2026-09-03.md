# Eager and compiled are not one program: what the gap actually is (2026-09-03)

`docs/measurements/serving-compile-divergence-2026-09-02.md` established that the
eager-vs-compiled KL is **deterministic** per *(weights, build)* -- 0.026861 on
the FP8 route, 0.244481 on the NVFP4 route, reproduced to six decimals by
independent serves in separate containers -- and attributed its *size* to
"compiling an NVFP4 forward on this model". It did not say what the compiler
changed, and issue #16 asked. This receipt answers that, and the answer is not
the compiler.

**In one line.** vLLM 0.28 runs *different implementations of the same math*
depending on whether it is going to compile: eager gets the CUDA kernels, the
compiled forward gets the torch decompositions for inductor to fuse. Section 2
measures what that is worth: the two RMSNorm implementations disagree on a
third of their output elements by exactly one bf16 ULP, and the NVFP4 quantizer
downstream turns that into a disagreement **18% the size of the whole
quantization error** at the GEMM. Section 3 asks the served ladder whether
pinning the dispatch back closes the KL gap.

One thing the op-level measurement *corrected*: SiluAndMul, which the
`custom_ops` half of the switch moves, is **bit-identical** between its CUDA
kernel and its torch expression on these activations. The gap on this model is
carried by `ir_op_priority` alone.

---

## 1. The switch, in the runtime's own code and its own log

Two defaults flip together on "is inductor going to run":

| what | where | eager | compiled | |
|---|---|---|---|---|
| `custom_ops` base mode | `vllm/config/vllm.py:1392-1399` | `["all"]` | `["none"]` |
| `ir_op_priority` | `vllm/platforms/cuda.py:690-700` | `["vllm_c", "native"]` | `["native"]` |
| ~~`enable_flashinfer_autotune`~~ | `vllm/config/vllm.py:241` (O0) vs `:264,:287,:310` (O1-O3) | `True` | `True` | **not a switch here** |

The third row is struck through because it is wrong as originally written,
and the correction matters: both attested logs resolve
`enable_flashinfer_autotune=True` --
`/home/rob/tessera-runs/compile-dispatch/serve_qwen_dispatch_eager.log` and
`…_compiled.log`, each with `'mode': <CompilationMode.NONE: 0>` and
`<CompilationMode.VLLM_COMPILE: 3>` respectively. The O0-vs-O1 default cited
is a real default in vLLM's config, but it is not what these serves resolved,
so autotune is not one of the things that differs between the two arms here.
The consequence for the ladder is that `compiled-both-noauto` and
`compiled-eagerbackend` pin autotune off *in addition* to their named
variable: both are two-variable arms, and neither is a clean isolation of the
thing its name suggests. Section 3's conclusion does not rest on either --
`compiled-both`, which is the arm that lands on 0.000000, does not touch
autotune at all.

The first two are keyed on the same condition -- `backend == "inductor" and mode
!= NONE` -- and both are documented as deliberate. `CustomOp.default_on`
(`vllm/model_executor/custom_op.py`): *"When PyTorch Inductor is used, 'none' is
the default value, otherwise 'all'."* `CudaPlatform.get_default_ir_op_priority`:
*"Native used by default when compiling, use vllm_c kernels where available when
no codegen."*

What each one moves:

* **`custom_ops`** decides `CustomOp.dispatch_forward`: `forward_cuda` when
  enabled, `forward_native` when not. `SiluAndMul.forward_cuda` calls
  `torch.ops._C.silu_and_mul`; `forward_native` is `F.silu(x[..., :d]) * x[..., d:]`.
  These are two implementations, not one: whether they agree bitwise on real
  activations is a question for section 2, and this receipt does not assert an
  answer from the kernel source, which the pinned image does not ship.
* **`ir_op_priority`** decides the norm. In 0.28 `RMSNorm.forward_cuda` and
  `forward_native` *both* call `ir.ops.rms_norm`; the priority list is what
  picks the implementation, at `vllm/ir/op.py:327` (`IrOp.dispatch`). `native`
  is the fp32 decomposition in `vllm/ir/ops/layernorm.py`; `vllm_c` is
  `torch.ops._C.rms_norm` (`vllm/kernels/vllm_c.py:23`).

The third switch is not a dispatch at all: `--enforce-eager` selects
optimization level 0, which holds FlashInfer autotune off, while every compiled
arm autotunes at warmup (`vllm/model_executor/warmup/kernel_warmup.py:165-172`)
and so may select a different GEMM mainloop. It is a *candidate* second author
of the gap, and the served ladder in section 3 carries an arm that holds it
still so it can be separated rather than assumed away.

vLLM prints the two dispatch values in its startup config line, so this is read off
the two serves that produced the measured pair rather than asserted:

```
/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log:12        (eager)
  enforce_eager=True   'custom_ops': ['all']
  ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'],
                                    fused_add_rms_norm=['vllm_c', 'native'])

/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log:12  (compiled)
  enforce_eager=False  'custom_ops': ['none']
  ir_op_priority=IrOpPriorityConfig(rms_norm=['native'],
                                    fused_add_rms_norm=['native'])
```

**What it is not.** Issue #16 and the earlier receipt both reach for fusion
("fusion changing accumulation order"). The compiled arm's own config line says
`'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False,
'fuse_attn_quant': False, ...}` -- every fusion pass off. It is off because
`enable_norm_fusion` and `enable_act_fusion` (`vllm/config/vllm.py:123-146`)
test for an *enabled custom op* or an nvfp4-declared model, and this artifact is
neither: `custom_ops` is `none`, and `config.json` declares
`"format": "mixed-precision"`, which `ModelConfig.is_nvfp4_quantized`
(`vllm/config/model.py`) does not recognise as NVFP4. So on these artifacts
fusion is not part of the gap at all -- and, conversely, an artifact that
declared `nvfp4-pack-quantized` would take a *further* difference (the
SiluMul+NVFP4 quant fusion) that ours do not.

**Scope.** Everything above is about vanilla vLLM's own layers. The stock FP8
route additionally quantizes its activations through `QuantFP8`
(`vllm/model_executor/layers/quantization/input_quant_fp8.py:30`), which is
itself a `CustomOp` and therefore flips native/cuda with `custom_ops` -- so the
FP8-route number inherits one more switched implementation than the norm and
activation path alone. Tessera's own route calls its quantizers directly and
does not inherit that one; that is a difference in what the two routes are
exposed to, not evidence about either one's size.

**Where Tessera is in this.** Nowhere, on the producing side. Tessera's serving
ops are `torch.library.custom_op` registrations (`src/tessera/serving/ops.py:61,92`,
`fp8_gemv.py:300`) -- a PyTorch mechanism, unrelated to vLLM's `CustomOp` class
and untouched by the `custom_ops` base mode. The switched implementations are
vLLM's own layers, which sit in the forward pass whatever route the Linears
take. That is consistent with the gap appearing on the stock twin (0.247301)
and on both Tessera routes alike, and it means #16 is not a Tessera defect --
though it is still Tessera's problem, because it is our receipts that compared
across the two regimes.

## 2. The op level: what the switch is worth, measured

`experiments/compile_dispatch_divergence.py`, GB10, torch 2.13.0+cu130, vLLM
0.28.0 inside `vllm/vllm-openai:latest`. Real activations: Qwen3-0.6B BF16 at
layer 5, on the first 512-token chunk of `corpus_qwen_n8_s512.json` (the corpus
the served KL is scored on, Qwen tokenizer). Each producer op is run *both*
ways -- `impls["vllm_c"]` against `impls["native"]` for the norms, the
`torch.ops._C` kernel against the `forward_native` expression for SiluAndMul --
and both outputs are then pushed through the quantizers the two routes execute.
Full record: `experiments/results/compile_dispatch_divergence.json`.

**The producers.** bf16 out, 512x1024 (norms) and 512x3072 (activation):

| op | elements differing | max abs diff | rel L2 |
|---|---|---|---|
| `rms_norm` | 26.47% | 0.0625 | 0.002820 |
| `fused_add_rms_norm` | 33.80% | 0.125 | 0.003314 |
| `silu_and_mul` | **0.00%** | **0** | **0** |

Two findings, and the second is the one that corrects the story.

*The norms differ, at exactly one ULP.* bf16 carries 8 mantissa bits, so its
relative step is 2^-8 = 0.0039; the measured rel L2 is 0.0028 and 0.0033 and
`max_abs_diff` is one binade step at the magnitudes involved. This is the fp32
decomposition's double rounding (`x.to(weight.dtype) * weight` then
`.to(orig_dtype)`, `vllm/ir/ops/layernorm.py`) against the kernel's, on a third
of all elements. Not a bug in either -- a different program.

*SiluAndMul does not differ at all.* Zero elements, zero rel L2. So the
`custom_ops` half of the switch, which is what moves SiluAndMul, contributes
nothing numerically here, and the earlier draft of this receipt was wrong to
put it beside the norms. The gap that `--enforce-eager` opens on this model is
carried by `ir_op_priority`, not by `custom_ops`.

**The amplification.** The same two arms, quantized the way each route
quantizes its A side (`scaled_fp4_quant` with the checkpoint's own
`input_global_scale`; `dynamic_per_token_scaled_fp8_quant`):

| op | route | codes/bytes flipped | block scales flipped | rel L2 between arms | rel L2 of the quantizer's own error | arms / quant error |
|---|---|---|---|---|---|---|
| `rms_norm` | NVFP4 | 0.605% | 1.62% | 0.02429 | 0.09350 | 0.26 |
| `rms_norm` | FP8 | 2.85% | (135 token scales) | 0.01280 | 0.02290 | 0.56 |
| `fused_add_rms_norm` | NVFP4 | 0.800% | 2.27% | 0.02754 | 0.09337 | 0.29 |
| `fused_add_rms_norm` | FP8 | 3.73% | (192 token scales) | 0.01504 | 0.02331 | 0.65 |
| `silu_and_mul` | either | 0 | 0 | 0 | -- | 0 |

A one-ULP disagreement upstream comes out **8.6x and 8.3x larger** after the
NVFP4 quantizer (0.00282 -> 0.02429, 0.00331 -> 0.02754) and 4.5x larger after
FP8 -- so NVFP4 amplifies it about **1.9x more than FP8 does**. That ordering is
the same way round as the measured KLs (NVFP4 0.244481, FP8 0.026861) and is
the reason the prediction was worth testing; it is not by itself a derivation of
the ratio, and this receipt does not claim one.

**And at the GEMM.** Both arms' A sides against the checkpoint's own NVFP4
`q_proj` tile, in float32, versus the BF16 product:

    rel L2 arms vs each other   0.017711
    rel L2 eager    vs bf16     0.096628
    rel L2 compiled vs bf16     0.096581
    arms gap / quantization err 0.1833

Neither arm is better -- they are equally far from BF16, to the fourth decimal
-- but they are **18% of a whole quantization error apart from each other**.
That is what an eager-vs-compiled comparison of a W4A4 artifact is measuring
when it thinks it is measuring the artifact.

*Value checks* (this harness computes its own dequantization, and a wrong
nibble order would look like a result): `dequant_sanity_max_rel` is 0.049-0.084,
the quantizer's own `rel_l2` error is 0.091-0.094 on NVFP4 and 0.023-0.025 on
FP8, and the BF16 reference product lands at 0.0966. All four are the sizes FP4
and FP8 rounding should produce; none is O(1).

**Pre-registered prediction for section 3, written before the ladder reported.**
If the norms carry the gap and SiluAndMul carries none, then in the served
ladder `compiled-ir` (which pins only `ir_op_priority`) should recover most of
the eager-vs-compiled gap, and `compiled-ops` (which pins only `custom_ops`)
should recover little of it. If `compiled-ops` recovers a lot, the op-level
picture is incomplete and section 3 says so.

## 3. The served ladder: run, and it decides the question

The ladder ran to completion at 03:00--03:03 UTC on 2026-09-04, on the box the
earlier attempts could not get a serve to start on. Seven arms, each gated by
`TESSERA_KL_REQUIRE_IN_LOG` against its own resolved startup line, so every
arm's pinning is attested by the serve rather than asserted by the script; an
arm whose log did not match produced no dump, and all seven produced dumps.
All five compiled arms resolved `'mode': <CompilationMode.VLLM_COMPILE: 3>`
in their own logs -- they compiled, they did not silently fall back to eager.

Teacher is `eager` throughout. KL is the lower bound over the
teacher--student intersection, top-1024 support, 4088 scored positions,
corpus `076d33ef…`, tokenizer `76f13c8e…`.

| student arm | what it pins | KL (lower, all) | top-1 agree |
|---|---|---|---|
| `compiled` | nothing (the default) | 0.247301 | 70.43% |
| `compiled-ops` | `custom_ops: ["all"]` only | 0.248917 | 70.06% |
| `compiled-ir` | `ir_op_priority: ["vllm_c"]` only | 0.243933 | 69.45% |
| **`compiled-both`** | **both** | **0.000000** | **100.00%** |
| `compiled-both-noauto` | both, autotune off | 0.000000 | 100.00% |
| `compiled-eagerbackend` | `backend: "eager"` | 0.000000 | 100.00% |

**The divergence is entirely the two dispatch defaults, and pinning both
closes it exactly.** Not "closes most of it" -- `kl_lower_max` is 0.0 and
top-1 agreement is 100.00% over every scored position, which is bit-identical
logprobs from a compiled serve and an eager one.

**Neither switch does anything alone.** `compiled-ops` (0.2489) and
`compiled-ir` (0.2439) both sit at the unpinned arm's 0.2473; the three are
one number to within the spread. So the two are gates in series -- `custom_ops`
decides whether the custom kernel is *enabled*, `ir_op_priority` decides
whether the compiled IR *dispatches* to it, and one divergent RMSNorm is
enough to move 30% of the top-1 predictions. Either gate left open leaves the
whole gap in place.

**This falsifies section 2's own prediction, and section 2 is the screen.**
The op-level measurement found `silu_and_mul` bit-identical and concluded that
"the `custom_ops` half contributes nothing -- `ir_op_priority` carries it
alone". The served ladder says `ir_op_priority` alone carries *none* of it.
The op-level result is not wrong about the ops it measured; it was wrong about
which knob controls them, and only the served metric could say so. Recorded
here as a screen that inverted, in the direction the house rule predicts.

**The decomposition is a matched set; the unpinned baseline is not.** The
three pinned arms and `compiled-both` differ from each other *only* in which
pin is present -- all four carry the same `pass_config` with the three fusion
passes off. The `compiled` arm does not carry it, so it differs in two ways;
it is quoted for orientation and it is not what the "both are needed" claim
rests on. That the unpinned arm lands on the same number as the two
single-pin arms is itself evidence the fusion passes are inert here, which
section 2 established independently (`is_nvfp4_quantized()` is False on an
artifact declaring `"format": "mixed-precision"`).

### What this does not settle

- **Reproducibility is untouched.** `historical-compiled vs compiled` is
  0.000000 / 100.00%, but the cache root was present for both, so that is the
  *replay* case already known to be bit-identical. The rebuild-from-empty case
  -- 120 of 196 `.best_config` records differing under one AOT key, a rebuild
  at 0.017117 -- is not addressed by anything here. That is #16's second
  defect and the pinned-cache ask is #30.
- **`historical-eager vs eager` is 0.000000**, which is the reproducibility
  control for the eager lane and is consistent with #60's finding that the
  eager lane's cross-session drift is exactly 0.0.
- **Nothing here says the pinned configuration should be the default.** It
  says the switch is real, is fully explained, and is fully closable. Whether
  Tessera's serving lane should pin it -- and what that costs in throughput,
  which this receipt does not measure -- is a decision.

## 4. What this changes

**The two defects of #16 are separate.** The non-reproducible *build* is
inductor autotuning against a loaded device at compile time -- 120 of 196
`.best_config` records differing under one AOT key with byte-identical
`cache_key_factors.json`, a replay bit-identical and a rebuild 0.017117
(`serving-compile-divergence-2026-09-02.md` section 3). The eager-vs-compiled
*divergence* is a deterministic, config-driven implementation switch that
happens before any of that. They share only a consequence: both make a
cross-regime arm incomparable to another. Issue #60's measurement that the
*eager* lane's cross-session drift is exactly 0.0 is consistent with this split
-- the instability is on the compiled side, and it is not the same thing as the
switch.

**#16 is not a Tessera defect, and the five listed compiled-path breakages are
already handled in this tree.** Verified by reading, each with its site: the
`data_ptr` fingerprint is skipped under `torch.compiler.is_compiling()`
(`src/tessera/serving/ops.py:196`, same in `window.py`); the direct pybind
decode is now `@torch.library.custom_op("tessera::nvfp4_decode_span2_out", ...)`
(`ops.py`); the mutating op on an aliased pool is gone (`ops.py:85-90`,
`fp8_route.py:30`); `int()` on the token dimension is on the
non-compiling branch only (`fp8_route.py:361`); the lazy extension load is
lock-guarded (`ext.py:437`); and residency is folded into the compile-cache key
by `compile_identity.py`. Tessera's serving ops are
`torch.library.custom_op` registrations, which the `custom_ops` base mode does
not touch.

**What changed in the tree.** `build_identity` now parses `custom_ops` and
`ir_op_priority` out of a serve log, folds them into the build fingerprint, and
offers `require_same_dispatch()`; two arms that ran different implementations
cannot certify each other, and a record that does not say which it ran certifies
nothing (schema `tessera.serve_build_identity/1` -> `/2`; no wire, no schema
minor, no `encoder_profile_id`). `serve_and_dump_kl.sh` can pin the dispatch
(`TESSERA_KL_VLLM_EXTRA`) and refuses to dump a serve whose own log does not
match what the arm asked for (`TESSERA_KL_REQUIRE_IN_LOG`).

**What is NOT fixed, and why it is bigger than this issue.** Nothing here makes
an eager arm and a compiled arm comparable. They can be *made* comparable by
pinning -- `--kernel-config {"ir_op_priority":{"rms_norm":["vllm_c"],
"fused_add_rms_norm":["vllm_c"]}}` plus explicit `pass_config` -- but that is a
non-default serving configuration, so a pinned arm is no longer a measurement of
what vLLM does by default. The honest options are (a) never compare across the
two regimes, which the new refusal enforces, or (b) re-take the eager baselines
under compilation. Choosing between them moves what our receipts mean, so it is
Rob's call, not a code change.

## 5. Provenance

Corpus `/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json`, contract sha
`cfbddc2c49078256564dffd32dc5033515ce11f30057c33f0fe457ed5aded59d`, tokenizer
`/home/rob/models/Qwen3-0.6B`, 8x512, 4088 scored positions. Op-level probe on
GB10, torch 2.13.0+cu130, vLLM 0.28.0 in `vllm/vllm-openai:latest`
(`sha256:61fc8a896b0a`). The two attested serve logs are
`/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log:12` (eager) and
`serve_qwen_stock_tessera-k2-graph.log:12` (compiled). No KL number in this
receipt was produced by me; the ones quoted are from
`serving-compile-divergence-2026-09-02.md` and carry that receipt's provenance.
