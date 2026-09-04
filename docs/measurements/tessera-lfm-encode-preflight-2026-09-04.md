# LFM2.5 bounded encode preparation

PrismaBuild action
`95dc64b1e03f6772a890e0ad27b4b227c6a70fe604e5302aa0641060ad613831`
ran one real expert's three projections on Sparky, reserving both of its GPU
tokens, 32 GiB and two CPUs. The model was the unchanged full source
`/mnt/shared/models/LFM2.5-8B-A1B-BF16`, not an expert-count cut.

The action pinned
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
Its checkout snapshot was `28cd059f0b2f33ad31ffac58e026222479787507` and ran
`experiments/moe_source_encode_preflight.py`. The probe derives source
spelling and shapes from the exporter's own `quantizable`, `expert_stacks`
and `plan_expert_stack`, then encodes at E4M3 q256 1024 with verification.

The first projection warmed in 54.227 seconds. The timed projections of
`model.layers.2.feed_forward.experts.0` were:

| Projection | Shape | Seconds | Wire bytes |
| --- | --- | ---: | ---: |
| w1 | 1792 × 2048 | 2.1384 | 1,854,976 |
| w3 | 1792 × 2048 | 2.1513 | 1,854,976 |
| w2 | 2048 × 1792 | 2.2000 | 1,855,488 |

Each timed projection counted four window-Viterbi calls, all four reaching
the fused implementation and none taking the reference. A separate profiled
repeat of w1 recorded `_step` at 1,868,745 microseconds of self CUDA time.
These are weight-only encodes; Hessian-fed LDLQ is not measured.

Peak CUDA allocation was 1,505,394,688 bytes and peak CUDA reservation was
2,619,342,848 bytes. The 72 one-second telemetry samples observed system
MemAvailable no lower than 115,011,672 KiB and SwapFree fixed at
16,776,848 KiB out of 16,777,212 KiB. The maximum sampled board power was
62.99 W against the approximately 140 W envelope. The samples include warmup
and profiling; they are not a steady-state energy-efficiency comparison.

Receipt:
`/mnt/shared/tessera-runs/ts5/lfm25/astra-encode-preflight-r3/encode.json`,
SHA256 `62c7b9949defcfbeff9e415bcb2bf472ece5303495283e92a907757e850033b8`.
The receipt's `host` is the container hostname; PrismaBuild's producer
attestation identifies the physical host as Sparky. Subsequent probe output
uses the explicit name `container_hostname` for that field.

Preparation attempts r1 and r2 assumed an index JSON and failed because the
source is a single `model.safetensors`. The probe now reuses the exporter's
shard map, which covers both layouts. The r2 pre-fix error was
`FileNotFoundError: /models/lfm/model.safetensors.index.json`.

This measurement bounds one-expert working memory and measures per-unit
encode cost. It does not measure full-checkpoint I/O, merge cost, serving,
or quality. Extrapolating these three timings gives about 38 minutes of
arithmetic encoding per 11-layer half; that remains an estimate until the
striped full-model run completes.
