# Tessera bytes through the stock NVFP4 kernel

**What was measured.** One materialised Tessera unit fed to
`vllm._custom_ops.cutlass_scaled_fp4_mm` — the ordinary CUTLASS NVFP4 GEMM, no
Tessera-aware code anywhere in the path.

**Result.** Against the same quantised activations multiplied by Tessera's own
decoded weights, the kernel's output differs by **rel 2.0e-5** — bf16
accumulation noise. The weight path is exact: the weights the kernel sees are
the weights the Tessera decoder produces.

| comparison | rel error | what it is |
|---|---|---|
| kernel vs (same quantised acts) @ Tessera W | **0.000020** | weight path only — the claim |
| kernel vs BF16 acts @ Tessera W | 0.094695 | + W4A4 activation quantisation |
| Tessera decode vs BF16 W | 0.136998 | weight quantisation at 4.0 bpp |

Setup: Qwen3.8-27B `layers.0.mlp.gate_proj` 512×2048, q256=768 with 25% release
(4.0 bpp band), M=256 activations, `scaled_fp4_quant` + `swizzle_blockscale` on
the stock paths, `alpha = global_scale_w / global_scale_a`.

**Why this is the format's central claim.** Tessera's serving argument is that
an artifact is an *ordinary* NVFP4 tensor — that a runtime which has never heard
of Tessera loads and serves it on native tensor cores. Until this ran, that was
an argument. The 2.0e-5 is what turns it into a measurement, and it depends on
the §6b→E4M3 conversion being an exact relabelling under a power-of-two global
scale (see `wire.nvfp4_scale_bytes`); a rounding re-derivation there would have
shown up here as a small but non-zero weight-path error.

## Scope — read this before citing it

Principle 14: a claim about a runtime is attested against the **pinned** runtime
or it is not a claim. This measurement ran on **vLLM 0.19.2rc1.dev86+g9a6a66f3b**
(torch 2.11.0+cu130) on GB10 / sm_121. Current vLLM is 0.26 or later, so this
attests the kernel contract on a runtime that is many releases behind, and it
must be re-run against the pinned serving image before any artifact ships or any
published claim rests on it.

What it does *not* establish, in addition:

- It is a single GEMM, not a served model. No end-to-end load, no eager/graph
  parity, no KL or PPL. Under principle 3 that makes it a screen.
- It does not establish the **route**: whether sm_121 dispatches this to a
  native Blackwell schedule or an older-architecture CUTLASS fallback is a
  separate question, and one this project has been burned by before (73.7% of a
  92 GB body on an `arch::Sm80` fallback, 2026-08-17). `route_status` must come
  from the runtime's own contract table, not from this file.

Reproduce: `tmp/gemm.py`.
