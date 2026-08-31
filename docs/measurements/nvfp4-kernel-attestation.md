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

## Load-time decode: measured, and where it goes

The GEMM is free because the bytes are ordinary NVFP4 by the time the kernel
sees them.  The cost that is *not* free is the replay that produces them, which
runs once per unit before serving starts.  Measured on GB10, one real Linear
(`Qwen3.8-27B layers.0.mlp.gate_proj`, 17408x5120, 89.1M params, R=3 schedule,
12.5% release):

| stage | before | after | note |
|---|---|---|---|
| trellis replay | 561.4 ms | 16.4 ms | scan -> windowed function, fused |
| NVFP4 pack | 12.7 ms | 6.7 ms | narrower dtypes |
| **total decode** | **574.2 ms** | **23.1 ms** | **24.9x** |
| extrapolated, 355B body | 38.1 min | 1.5 min | one-time, per load |

**Why it was slow, and why that was a bug rather than a price.**
`ConvCode.step` is `((bit << memory) | state) >> 1` -- a pure shift register.
The state at row *r* is therefore exactly the previous `memory` select bits, so
the replay is a *windowed function of the stored stream* with O(1) depth, not a
sequential scan.  Walking it row by row spent ~6 kernel launches per trellis
step for 17408 steps.  `decode.replay_body` now checks the shift-register
property against the tables it builds from `code.step` (rather than assuming
it) and takes the parallel path when it holds; a code that fails the check
still decodes down the sequential path.

Four profiler-directed fixes, in the order the profiler named them:

1. **The scan.** Row-at-a-time -> windowed. 561.4 -> 176.9 ms.
2. **The ladder.** ORing in one lagged bit at a time costs `memory` passes over
   a full-size tensor (55.9 ms of `bitwise_or_`, 40% of the replay).  Building
   the window by *doubling* -- a 2k-bit window from two k-bit ones -- is
   log2(memory) passes. 176.9 -> 99.9 ms.
3. **The tables.** Subset and transition tables were rebuilt per call: 264 small
   copies for 128 entries.  `lru_cache` makes that O(distinct trellises)=3.
4. **The dtypes, and the fusion.** Every plane and every intermediate was int64
   -- eight bytes per three bits of payload -- across ~15 unfused passes.  BODY
   and the decoder's whole output are now uint8, and `_decode_core` (replay +
   completion lookup) is one `torch.compile`d kernel.  99.9 -> 16.4 ms.

The isolated replay reaches **1.8 ms** (48.8 G param/s); the 16.4 ms measured
in `decode_codes_mixed` is the per-rate-group column gather and scatter around
it, which a uniform-rate schedule would avoid.

**Two dtype landmines this exposed**, both silent in a way that matters: a
`uint8` index tensor is a *boolean mask* in torch, not an integer index, so
COMPLETION is widened at the reader (`unit_artifact`) while BODY stays narrow;
and `decode_codes` and `decode_codes_mixed` had disagreed about the dtype of a
nibble, which only the release scatter caught.

**Correctness:** `replay_body` parallel == sequential for memory in
{3,4,5,6,8} x R in {1,2,3}; fused == eager bit-for-bit over the same grid
(`test_fused_replay_equals_the_eager_path_bit_for_bit`); the full suite (195
tests) is green with fusion on *and* under `TESSERA_FUSED_REPLAY=0`.

## What this does and does not buy

- **Disk:** no sidecar is needed.  At 1.5 min for a 355B body, decoding at load
  is cheaper than the disk the materialised copy would cost.
- **VRAM:** unchanged at 4.5 bpp.  The decoder still *produces* a standard
  NVFP4 tensor, and that is what occupies memory.  Tessera compresses the
  artifact, not the working set.
- **Only an in-kernel decoder changes the second line**, and that is the thing
  this design deliberately avoids.  See `release-vs-tuple-trellis.md`: the
  RELEASE plane is the part of the current grammar that is *not*
  kernel-friendly, because its positions are data-dependent (ordered by
  decoded magnitude on the pre-release decode), so an in-kernel decoder would
  need an explicit index that costs more bits than release saves.
