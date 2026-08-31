# Tessera bytes through the stock NVFP4 kernel

**What was measured.** One materialised Tessera unit fed to
`vllm._custom_ops.cutlass_scaled_fp4_mm` — the ordinary CUTLASS NVFP4 GEMM,
with no Tessera-aware code anywhere in the path.

**Runtime:** vLLM **0.28.0**, torch 2.13.0+cu130, **NVIDIA GB10 sm_121**, in the
stock `vllm/vllm-openai:latest` container.

**Result.** Against the same quantised activations multiplied by Tessera's own
decoded weights, the kernel's output differs by **rel 1.0e-5** — bf16
accumulation noise. The weight path is exact: the weights the kernel sees are
the weights the Tessera decoder produces.

| comparison | rel error | what it is |
|---|---|---|
| kernel vs (same quantised acts) @ Tessera W | **0.000010** | weight path only — the claim |
| Tessera decode vs BF16 W | 0.139770 | weight quantisation at 4.0 bpp |
| kernel vs BF16 W | 0.167777 | + W4A4 activation quantisation |

Setup: Qwen3.8-27B `layers.0.mlp.gate_proj` 512×2048, q256=768 at **12.5%
release** — the operating point that is exactly 4.0000 bpp — M=256 activations,
stock `scaled_fp4_quant` and `swizzle_blockscale`,
`alpha = global_scale_w / global_scale_a`.

**Why this is the format's central claim.** Tessera's serving argument is that
an artifact is an *ordinary* NVFP4 tensor — that a runtime which has never heard
of Tessera loads and serves it on native tensor cores. Until this ran, that was
an argument. The 1.0e-5 is what turns it into a measurement, and it depends on
the §6b→E4M3 conversion being an exact relabelling under a power-of-two global
scale (`wire.nvfp4_scale_bytes`); a rounding re-derivation there would have
surfaced here as a small but non-zero weight-path error.

## Stored 4.0 bpp, resident 4.5 bpp — say both, always

| | Tessera @ 4.0 | plain NVFP4 |
|---|---|---|
| stored (artifact on disk) | **4.0000 bpp** | 4.5000 bpp |
| resident (VRAM after load) | 4.5000 bpp | 4.5000 bpp |

Decoding is a **materialisation**, not a kernel: the trellis replay runs once at
load and emits ordinary NVFP4 nibbles. That is exactly what buys the stock GEMM,
and it is also what bounds the claim. Tessera compresses the *artifact*; it does
not reduce serving memory. §7 already names both quantities and this table is
that distinction made concrete — reporting the 4.0 without the 4.5 would be the
kind of accounting this project retracts.

For the EXL3 comparison this is the right axis anyway: that comparison is
artifact bytes at matched quality. But "4-bit" here means 4-bit *on disk*, and
any claim that implies a VRAM saving is false as the format currently stands.

Note also that not every configuration is worth building. At full completion,
body + completion is 3 bits/position from **every** root, so bpp = 3.5 + 4·ε_B
and release is the only dial above 3.5. A 25%-release artifact is 4.5 bpp —
level with plain NVFP4 on disk and behind it after the diagonals — i.e. strictly
pointless. The useful band is ε_B < 25%.

## Scope — read before citing

- **A single GEMM, not a served model.** No end-to-end load, no eager/graph
  parity, no KL or PPL. Under principle 3 that makes it a screen, not a result.
- **It does not establish the route.** Whether sm_121 dispatches this to a
  native Blackwell schedule or an older-architecture CUTLASS fallback is a
  separate question this project has been burned by (73.7% of a 92 GB body on an
  `arch::Sm80` fallback, 2026-08-17). `route_status` must come from the
  runtime's own contract table, not from this file.
- Earlier revision of this document reported 0.19.2 and mislabelled a 4.5 bpp
  configuration as 4.0. Both are corrected above.

Reproduce: `tmp/gemm28.py` (in the `vllm/vllm-openai:latest` container).
