# GLM5-next NoPE CUSTOM backend correctness, 2026-09-13

Tessera #489; implementation base `76cc9d6c2`. This is a correctness gate
for an explicit research attention backend, not a runtime-contract promotion
or a performance measurement.

Image on sparklina (NVIDIA GB10 / SM121):
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`, FlashInfer `0.6.18`.
Tessera was installed with `pip install --no-deps --no-build-isolation -e /work`.
SHA256 manifests established that all 2,575 stock vLLM `.py` and `.so` files
were unchanged before installation, after installation, and after serving.
The published stock enum continued resolving its own class; CUSTOM resolved
`tessera.serving.glm53_nope.TesseraGLM53NoPEBackend`.

## Native regression and mathematical check

`python3 experiments/glm53_nope_check.py --stock` exited 1 in the unchanged
vLLM cache operator: `cache_kernels.cu:928, pe_dim must be 64 for fp8_ds_mla`.
The custom arm supplies the required reserved zero RoPE bytes and reuses the
same stock writer and native reader.

All four native cases use candidate counts 0, 1, 73, 2,048 and 2,049, including
holes, at capacity 2,176. Prefill repeats those rows 13 times (65 queries).
Padded cases double the physical block stride. The oracle decodes identical
packed FP8 KV values and the native variant's FP8 query representation, then
computes softmax attention in FP32. Split-K and MG quantize queries with
power-of-two-ceiled per-128 scales; SWAPAB uses arbitrary FP32 per-128 scales.
The tolerance is twice BF16 epsilon, for both absolute and relative error.

| Command suffix | Native variant | Max absolute error | RMS error | BF16-query screen max error |
|---|---|---:|---:|---:|
| default | decode split-K, 64 heads | 0.0055804 | 0.0006402 | 0.0612887 |
| `--heads 32 --padded` | decode split-K, 32 heads | 0.0055804 | 0.0006426 | 0.0580368 |
| `--repeat 13 --padded` | prefill SWAPAB, 64 heads | 0.0091524 | 0.0007329 | 0.0935967 |
| `--repeat 13 --heads 32 --padded` | prefill MG, 32 heads | 0.0078125 | 0.0007197 | 0.0985568 |

Repeated calls and CUDA graph replay were bit-exact in all four cases; empty
rows were exactly zero. The BF16-query column is a separate arithmetic screen,
not the represented-operand oracle or a full-model quality result.

An intermediate repeat check failed by 0.00390625: stock compaction uses
racing atomic column tiles for the non-power-of-two width 2,176. Padding
invalid indices to 4,096 selects its deterministic single-program row path;
truncating the compacted output back to 2,176 preserves all valid candidates.
A first 256-token-page check also failed at the native decoder's 64-token
physical page requirement. The backend now views flat physical rows as
64-token native pages without copying the cache. Both fixes reuse stock APIs.

## Four-layer BF16 service

Existing checkpoint `/mnt/shared/models/GLM-5.3-Flash-4layer`, 12 shards,
45.33 GiB checkpoint bytes. Launch:

```sh
TESSERA_RESEARCH_GLM53_NOPE=1 vllm serve MODEL \
  --served-model-name glm53-stub --attention-backend CUSTOM \
  --kv-cache-dtype fp8_ds_mla \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --enforce-eager --gpu-memory-utilization 0.55 \
  --max-model-len 4096 --max-num-seqs 8 --max-logprobs 1024 \
  --trust-remote-code --host 127.0.0.1 --port 8139
```

The service reached READY. Stock hybrid cache sizing selected attention blocks
of 8,704 tokens. Two identical 5-token prompts produced 32 tokens each with
finite log probabilities and bit-identical text/log probabilities. A 3,649-token
prompt produced 16 tokens with finite log probabilities. All returned HTTP 200.
There were no platform, PDL or warmup source patches; the stock public kernel
configuration disabled FlashInfer autotune.

Eight guard/registration tests passed with pytest 8.4.2, pytest-xdist 3.8.0,
`-n 2 --dist worksteal --durations=5`, zero skips. These exercise runtime guards
and registry behavior; the four native checks are the separate GPU population.
All these runs execute vLLM and use Rob's universal vLLM exemption from PB.
No whole-suite or non-vLLM GPU qualification is claimed.

Receipts: `/home/rob/dq-runs/glm-campaign-takeover-20260913/serving/`:
`native-stock.log`, `native-custom.log`, `native-head32-padded.log`,
`native-prefill-padded.log`, `native-prefill-head32.log`, `guard-tests.log`,
`serve-eager.log`, and `receipts/{runtime-identity,stock-after-serve,smoke-summary}.json`.
The exact launch and request scripts are retained beside those receipts.

The four-layer slice cannot establish full-model quality, TP2 collectives,
routed compressed experts, or full-model memory fit. Whole-model compiled/eager
agreement is not established by isolated attention graph equality. No runtime
cell or production ship gate is changed, and no throughput claim is made.
