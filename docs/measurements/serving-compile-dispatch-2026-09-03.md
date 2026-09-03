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
compiled forward gets the torch decompositions for inductor to fuse. The
*prediction* under test is that the two implementations differ by a few ULPs of
bf16 and that a W4A4 route amplifies that -- an FP4 quantizer re-draws it across
codes ~40% apart -- into a fifth of a nat. Section 2 measures the first half and
section 3 the second; until they are filled this line is a hypothesis, not a
result.

---

## 1. The switch, in the runtime's own code and its own log

Two defaults flip together on "is inductor going to run":

| what | where | eager | compiled |
|---|---|---|---|
| `custom_ops` base mode | `vllm/config/vllm.py:1392-1399` | `["all"]` | `["none"]` |
| `ir_op_priority` | `vllm/platforms/cuda.py:690-700` | `["vllm_c", "native"]` | `["native"]` |
| `enable_flashinfer_autotune` | `vllm/config/vllm.py:241` (O0) vs `:264,:287,:310` (O1-O3) | `False` | `True` |

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

## 2. PENDING -- op-level measurement

## 3. PENDING -- the served ladder

## 4. PENDING -- what this changes
