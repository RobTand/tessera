# Native dense receipt producer — GPU qualification on artifact bytes, 2026-09-13

Tessera #376 landed the producer and qualified it on GPU with a synthetic
32 x 32 BF16 R512 encode and basis-vector inputs
(`docs/measurements/native_dense_receipt_20260907.md`). The issue's remaining
item was a GPU qualification after LFM sanity. This records it.

`experiments/qualify_native_artifact_operator.py` freezes a
`tessera.native_dense_panel.v1` panel from retained artifact bytes and emits one
`tessera.native_dense_operator_receipt.v1` receipt. The producer itself is
unchanged; only the driver is new.

## What was measured

LFM2.5-8B-A1B `model.layers.0.feed_forward.w1`, 7,168 x 2,048, format
`TESSERA_BF16_K1_R1792`, body WINDOW, plane CHANNEL, eager/resident/TP1 without
bias, on Sparky GB10 sm121 inside the immutable image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
The wire blob, its `tessera.encoding_inputs.v1` record, the rendered weight and
both phases' reference tensors are the bytes retained at
`/mnt/shared/tessera-measurements/pq267-native-20260907/dense-r1792-01/`, the
same inputs as the untimed preflight
`native-preflight.v2.json`. Nothing was re-encoded or re-rendered.

Predeclared parity tolerance `atol = rtol = 0.015625`, taken from that
preparation's frozen plan and set on the command line before the run.

| Phase | m | QDQ | Output parity | max abs error | Single-apply CUDA-event samples (ms) | Operator scratch bound (bytes) |
|---|---|---|---|---|---|---|
| prefill | 512 | passed | passed | 0.0078125 | 0.3880 - 0.3967 (8 samples) | 36,700,160 |
| decode | 1 | passed | passed | 0.001953125 | 0.1771 - 0.1843 (8 samples) | 71,680 |

Receipt `status = timing_admissible`, `resources.status =
complete_operator_bound`, `timing_scope =
cuda_events_after_resource_collector_stop`. Both phase bounds compose as
`sum_of_independent_peaks_including_output` with
`external_native_peak_bytes = 0` and
`full_model_fixed_resources_complete = false`. The observed route was
`dense / TESSERA_BF16:resident / torch.mm / torch_window / bf16_unquantized`
for both phases, matching the producer's declared route.

Each timing sample is one complete apply, not a loop average. Timings are
collected only after the CUPTI collector stops, so no allocation callback is
registered while a priced call runs.

## Evidence

| Evidence | PB action | Result |
|---|---|---|
| GPU qualification on artifact bytes | `b8adfd50548a25dc24bf4a6617d49c40a6c6024d6d55351e43c9d979b3a2df50` | returncode 0, `timing_admissible` + `complete_operator_bound` |

Done record:
`/mnt/shared/prismabuild-fleet/pb-queue/done/b8adfd50548a25dc24bf4a6617d49c40a6c6024d6d55351e43c9d979b3a2df50.json`.
Host sparky, 19.44 s, cgroup memory peak 2,277,707,776 bytes against a declared
`mem_gb = 12`; the next submission of this shape can declare far less. GPU power
peaked at 10.7% of the SoC reference over the action window. That window is a
short instrumentation invocation and supports no throughput or
energy-efficiency claim.

Retained artifacts, by SHA256 of the file bytes:

- receipt `e1d5376da439d82195885d36623a8a3397cc9603098fe91e736d660d122b1c5a`
- frozen panel `87f86bfa5f27d5d1448e89c78e8e91aa17fc4773e5b0b49d084ebda837c546d9`
- raw CUPTI trace `f980a2c34ff78601f68efe572bd1d5884b03779f600af66eda2594541992dd1c`

The collector binary is the immutable `native_operator_resources-v2.so`, SHA256
`1a893cec8e7386c48b86f69ebb6c677b009f9aa0653afae5b88594e8deaaaf73`, unchanged
from the 2026-09-07 qualification.

## What this is not

The panel's `cost_sha256` and `probe_identity_sha256` are declared fixtures. A
PrismaQuant-frozen panel takes both from a joint AURA cost row
(`prismaquant.joint_aura.operator.v1`); no such row exists for this artifact,
because the joint capture for it failed its forward-parity check and its
replacement requires the regenerated canonical captures tracked on the
PrismaQuant side. This receipt is therefore a producer qualification of
timing, route and operator scratch on real bytes. It is not a joint-bound
runtime price, a render-quality result, a full-model resource measurement, or a
release admission.

The reference tensors were rendered by PrismaQuant before that capture
correction was understood. They are exercised here as fixed bytes for parity
and timing; no quality claim follows from their agreement.

`full_model_fixed_resources_complete = false` on every bound. A PrismaQuant v2
measured-runtime table admits a native row from a receipt like this one, and
then still refuses the table on fixed resources until a recomputable
full-engine resource report exists. That boundary is the consumer's, and it is
open.
