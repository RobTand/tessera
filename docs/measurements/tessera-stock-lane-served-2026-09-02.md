# The stock lane, served: Tessera as an NVFP4 and an FP8 encoder on vanilla vLLM

**Date:** 2026-09-02. **Box:** sparky (GB10, sm_121). **Runtime:** `vllm/vllm-openai:latest`
= v0.28.0, image `sha256:61fc8a896b0a…`, no plugin, no fork, no serve flag.
**Tessera:** `406e951` (stock materialiser, exporter, capture lock).
**Model:** Qwen3-0.6B (`/home/rob/models/Qwen3-0.6B`), 196 body Linears, 440.4 M
quantizable parameters; `lm_head` and `embed_tokens` BF16 on every arm.
**Metric:** exact `kl_tool.py` KL-vs-BF16 on the WikiText test corpus
(`corpus_qwen_n8_s512.json`, sha `076d33ef…`, 8 × 512, 4088 scored positions,
top-1024 support, teacher-student intersection, **lower bound**), teacher
re-dumped on the same image. Artifacts under `/home/rob/tessera-runs/stock/`,
dumps under `/mnt/shared/tessera-kl/qwen_stock_*.json.npz`, per-arm compares in
`kl_<arm>.json`, the per-unit error censuses in `rel_mse_by_unit.json` and
`h_weighted_error.json`.

## What was measured

The wire's rate lives on the kernel lane. This measures the *other* thing a
Tessera encoding is: a checkpoint in a format the stock runtime already
serves. `tessera.stock.materialize_stock` writes an E2M1/E2M1x2 unit over a
LUT plane as the compressed-tensors NVFP4 triple and an E4M3 unit over the
CHANNEL plane as the per-channel FP8 pair; `stock_dequant` restates the decode
in vLLM's own arithmetic and `tests/test_stock.py` holds `torch.equal` against
the bytes-only reader on every default wire. Fused groups (q/k/v, gate/up) are
one vLLM Linear carrying one `weight_global_scale`; `share_global` moved 52 of
140 members onto their group's power of two by an exact binade shift, none
refused, and the serve logged no global-scale warning.

Four arms, one writer, one image, one corpus:

| arm | what it is | activations | resident bpp | wire bpp | checkpoint |
|---|---|---|---|---|---|
| `nvfp4-prod` | PrismaQuant production NVFP4 (GPTQ + JSO + static input scales), `fc45-0p6b-nvfp4/exported` | W4A4 | 4.500 | — | 830 MiB |
| `tessera-k2` | Tessera E2M1x2 q896 (coset trellis, LUT16), materialised to NVFP4; `input_global_scale` per Linear copied from `nvfp4-prod` | W4A4 | 4.500 | **4.002** | 830 MiB |
| `tessera-e8` | Tessera E4M3 q1024 (window body L=14, CHANNEL plane), materialised to per-channel FP8 | W8A8 | 8.025 | **4.071** | 1015 MiB |
| `fp8-rtn` | per-channel FP8 round-to-nearest, `amax/448` per row (bit-identical to PrismaQuant's `quantize_dequantize_fp8_dynamic`) | W8A8 | 8.025 | — | 1015 MiB |

The resident rate is the stock format's. The wire column is what the same unit
costs on the kernel lane and is **not** what these checkpoints hold; the
manifest states both per unit. The E4M3 wire carries a 16 KiB window table
per unit and an fp16 row plane, 0.07 bpp on these 1–3 M-parameter Linears.

## Routes (from the serve logs, principle 9)

| arm | kernel vLLM selected | native? |
|---|---|---|
| `nvfp4-prod` | `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM | yes, FP4 tensor cores |
| `tessera-k2` | `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM | yes, same kernel |
| `tessera-e8` | `CutlassFP8ScaledMMLinearKernel` for `CompressedTensorsW8A8Fp8` | yes |
| `fp8-rtn` | `CutlassFP8ScaledMMLinearKernel` for `CompressedTensorsW8A8Fp8` | yes |

Eager serves for the dumps (`--enforce-eager`, the harness's contract); the
graph-mode generation smoke (`experiments/serve_smoke_graph.sh`, no
`--enforce-eager`, one greedy completion) is recorded in §"Graph mode" below.

## Served KL (the gold metric)

| arm | KL ≥ (all) | KL ≥ (confident, n=1709) | top-1 agreement |
|---|---|---|---|
| `fp8-rtn` W8A8, 8.0 resident | **0.0205** | 0.0142 | 91.2% |
| `tessera-e8` W8A8, 8.0 resident (4.07 wire) | 0.4699 | 0.3894 | 63.2% |
| `tessera-e8-reach` W8A8, 8.0 resident (4.07 wire), reach-aware row start (2026-09-02, later) | 0.1512 | 0.1005 | 78.1% |
| `nvfp4-prod` W4A4, 4.5 resident | 0.5106 | 0.4358 | 62.6% |
| `tessera-k2` W4A4, 4.5 resident (4.00 wire) | 0.6404 | 0.5486 | 58.8% |

Read plainly:

* **As an NVFP4 encoder at equal resident bytes, Tessera loses to the
  production encoder by 1.254× KL** (0.640 vs 0.511, W4A4 on the same kernel
  with the same input scales). The advisor's expectation held: a
  coset-constrained code set under the same scale plane does not beat GPTQ +
  JSO at the same residency, and the scale plane alone did not carry it.
* **As an FP8 encoder, Tessera-8 at a 4-bit wire is 23× worse than FP8 RTN at
  the same 8-bit residency** (0.470 vs 0.020; 0.151 vs 0.020 with the reach-aware
  row start, `tessera-dense-reach-fix-2026-09-02.md`). Nobody should serve the
  materialised form of Tessera-8 for quality; its value is the 4.07-bpp wire on
  the kernel lane, and this arm only shows the FP8 route serves the bytes.
* Tessera-8 W8A8 beats production NVFP4 W4A4 by 8% (0.470 vs 0.511) — at
  1.78× the resident bytes, so that is not a comparison anyone should quote.
* Every arm is far above the earlier BF16-decoded readings on this model
  (Tessera K2 0.192, NVFP4-RTN 0.174, W16A16 on the Mia image): the A4 path on
  a 0.6 B model costs more than the weights do. Tessera-K2's weights served
  W4A4 read 0.640 against 0.192 decoded.

## Why the weight-space ranking inverted

Plain weight error ranks the 4-bit arms the way the frontier measurements do;
activation-weighted error ranks them the way the serve does. Diagonal-Hessian
weighting (`h_j = E[x_j²]` per input column from a BF16 forward on 4096 tokens
of text disjoint from the corpus, `experiments/stock_h_weighted_error.py`):

| arm | plain rel. MSE (geomean) | H-weighted rel. err (total) | H-weighted (geomean) | worst unit |
|---|---|---|---|---|
| `nvfp4-prod` | 8.86e-3 | 9.70e-3 | 9.13e-3 | 1.30e-2 |
| `tessera-k2` | 9.12e-3 | 1.034e-2 | 9.96e-3 | 1.30e-2 |
| `tessera-e8` | **5.50e-3** | **1.081e-2** | 7.60e-3 | **1.68e-1** (`layers.2.mlp.down_proj`) |
| `fp8-rtn` | 6.96e-4 | 6.86e-4 | 6.86e-4 | 7.5e-4 |

Tessera-8 has 1.6× less weight error than production NVFP4 and *more*
activation-weighted error in total, and the total is carried by a handful of
units. (The proxy is a diagonal: it puts Tessera-K2 at 1.066× production
NVFP4 where the serve reads 1.254×, because GPTQ optimises the full Hessian
and a diagonal weighting undercounts exactly that compensation; the two
numbers are consistent, the proxy is just partial.) `layers.2.mlp.down_proj` has an input channel whose second moment is
2.2 million times the column median (its top four columns hold 96% of the
Hessian mass); the CHANNEL plane — one fp16 scale per **output row** — gives
that column nothing, so 7.0e-3 of plain error becomes 1.68e-1 of weighted
error, 24× worse. Per kind, Tessera-8 beats production NVFP4 on q/o/up/gate/v
(0.61–0.82×) and loses on `k_proj` (1.35×) and `down_proj` (1.16×), the two
kinds whose inputs carry the outlier channels. NVFP4 and the LUT16 plane are
immune by construction: a per-16 block scale isolates the outlier column and
GPTQ spends the Hessian on it.

This is the frontier's blind spot, not a contradiction of it. The 0.940× EXL3
number was measured on GLM-5.3-Flash routed experts, whose inputs are Gaussian
(rotation measured dead there, 1.003×); dense attention and MLP inputs are
not. The CHANNEL plane's recipe is exposed to input-column outliers wherever
they exist, and the render leg refuses `col_weights` today rather than
consuming them, so nothing in the pipeline sees the exposure before a serve.

## Graph mode

`serve_smoke_graph.sh` (no `--enforce-eager`, `--gpu-memory-utilization 0.6`,
greedy 24 tokens on "The capital of France is"):

| arm | loaded and captured | completion |
|---|---|---|
| `nvfp4-prod` | yes (`FlashInferCutlassNvFp4LinearKernel`) | " the capital of France. The capital of France is the capital of France…" |
| `tessera-k2` | yes (`FlashInferCutlassNvFp4LinearKernel`) | " Paris, and the capital of France is the capital of the capital of…" |
| `tessera-e8` | yes (`CutlassFP8ScaledMMLinearKernel`) | " 11111111111111111111111" |
| `fp8-rtn` | yes (`CutlassFP8ScaledMMLinearKernel`) | " Paris. The capital of France is also the capital of the European Union…" |

Every format captures and generates. The completions are a different
question, and the Tessera-8 one needed separating from the served path, so
the same prompt was run in HF transformers with each checkpoint's **exact
dequantised weights** (`stock_dequant`, W16A16, no activation quantisation,
`experiments/stock_hf_greedy_check.py`):

| arm | exact-weight greedy completion | top next-token logprobs |
|---|---|---|
| BF16 | " Paris. The capital of France is also the capital of the Republic of France…" | ` Paris` −0.47 |
| `fp8-rtn` | " Paris. The capital of France is also the capital of the Republic of France…" | ` Paris` −0.33 |
| `nvfp4-prod` | " Paris. The capital of France is also the capital of the French Republic…" | ` Paris` −0.34, ` Lyon` −3.03 |
| `tessera-k2` | " Paris, and the capital of Italy is Rome. The capital of the United States is Washington, D.C.…" | ` Paris` −0.45 |
| `tessera-e8` | " 1111. The capital of the United States is 1111. The capital of the United…" | ` ` −1.58, ` a` −1.83, ` the` −2.08; ` Paris` not in the top five |

So the Tessera-8 checkpoint's weights are what answer " 1111"; the FP8 route
(eager and graph) serves them faithfully, exactly as FP8 RTN's are served
faithfully to " Paris". The production W4A4 arm's degenerate *served*
completion is the other case: its exact weights say " Paris", and the A4
activation path on a 0.6 B model is what it loses. KL 0.47 with 63% top-1
agreement is an average over a model that is mostly still coherent and has
lost specific recall — which is what a few catastrophic units do.

**Subnormal block scales: none.** `stock_dequant` decodes a subnormal E4M3
scale byte correctly in software, and `share_global` accepts a binade shift
whenever the byte round-trips, so the identity test alone would not show how
`FlashInferCutlassNvFp4LinearKernel` treats a subnormal block scale. Counted
over every `weight_scale` tensor (27,525,120 bytes per arm): tessera-k2 and
nvfp4-prod both carry zero subnormal and zero zero-valued scale bytes, so the
kernel's subnormal handling never entered the 0.640.

## What this closes, and what it does not

Closed for the **materialised** form: an exporter writes it
(`experiments/export_stock_compressed.py`, from a Tessera grid and rung or a
per-tensor plan), the serving lane is attested on vanilla vLLM with native
kernels for both formats, the tensors are ordinary compressed-tensors tensors
(TP-shardable the way any NVFP4 or FP8 checkpoint is), and a served KL A/B
exists. Quality at equal resident bytes: **worse than the production encoder
on 4-bit and far worse than RTN on 8-bit**, on a dense model — which is the
honest headline of this lane, and the reason its value is the wire.

Not closed: the **size-matched** product. 4.0 bpp exists only on the kernel
lane (`tessera.kernel`), which stock vLLM does not run; a load-time
materialiser plugin or the kernel lane inside vLLM is the remaining
engineering, and MoE (GLM) waits on a dense result — this one — first.

Two leads with a measurement behind them, in the order they should be run:

1. **Activation-aware encoding on the CHANNEL plane** — LDLQ already measured
   1.147× → 1.059× on per-channel Tessera-8 in output space, and the render leg
   refuses `col_weights` rather than consuming them. The H-weighted census
   above is the test: `layers.2.mlp.down_proj` should fall from 1.68e-1 to the
   1e-2 the block-scale arms hold.
2. **Column smoothing folded into the preceding layer** (SmoothQuant-style,
   exact in a standard block: norm weights for q/k/v/gate/up, `v_proj` rows for
   `o_proj`, `up_proj` rows for `down_proj`), which removes the outlier columns
   the row scale cannot see instead of encoding around them.

Neither is default until it wins here and on the GLM experts.
