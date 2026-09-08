# Native GLM packed selected-expert lifecycle, 2026-09-08

Issue #422. This is an explicit Python-only eager TP1/EP1/DP1 research owner,
not a production serving selection. The ordinary Tessera config, resident and
streamed gates, packaged cells and release pins do not select it.

The existing GLM factory and `RoutedExperts.load_weights` loaded all 864
projection wires into a 288-expert stack (hidden 4096, intermediate 2048,
top-k 8) on Sparklina/GB10. The image was stock
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`.
Its 4,967 core files were verified unchanged before/after installation and
execution, and loaded Tessera file/spec origins were verified. The container
used four CPUs with native threads bounded to one, a 48 GiB memory limit and
no network. Direct execution follows Rob's vLLM exemption.

Baseline installed source: `2083062de771269a1430593822de8fafde067c79`.
Packed installed source: `b3583bccb7` (implementation `abf9e2445e`). The
resident and packed arms use the same original q512 projection wires and
exact saved inputs (`253031a63500afe6658b7cc501fd72dd135c86f924cba5154dc0b405db18f81c`).
These repeat one trained source projection per role over the expert axis;
they do not establish trained-expert diversity or execute a trained router.

## Ownership and output

| Quantity | Resident | Packed |
|---|---:|---:|
| Persistent expert tensors, excluding common routing bias | 6.758789 GiB | 1.726303 GiB |
| Peak CUDA allocated during create/load/prepare | 17.782466 GiB | 4.164161 GiB |
| CUDA reserved after prepare | 20.226563 GiB | 6.000000 GiB |

The native owner keeps its existing 1,152-byte routing bias in both arms.
Packed preparation removes wire parameters and length references, retains
shared packed FP8/window owners and holds no full FP8 expert parameter pool.
Per-role reference tiles used by shared preparation are transient. The load
peak includes both wire and prepared representations during the transition.

Decode (1 token), all 288 experts (36 tokens), clamp stress (4 tokens, input
multiplier 64), and empty input have exact finite output parity against the
independent full stock FP8 owner. Every selected tile and scale matches the
independent stock representation with reversed requested expert order. Empty
selection returns correctly shaped empty tiles and output.

The separate `diagnostic-01` control canonically reserializes the original
down-projection with 288 distinct positive binade scale signatures. All
non-scale plane content digests remain identical to the original wire, and
the actual loader parses the changed scale bytes. This deliberately changed
fixture makes no encoder-identity or trained-expert-diversity claim. Original
performance wires are untouched. Diagnostic output, tiles and scales are
exact in every case; swapping two compact expert-map entries yields maximum
absolute errors of 0.1875 (decode), 0.875 (all experts) and 32 (clamp stress).
The diagnostic receipt hash is
`4f30afa26a8e452e538e7fbcb025b4ed404ac2034d1b56eaa1be7b3b9969706c`.

## Measured tradeoff

The implementation saves persistent storage but is far too slow for production.
Two warmups precede at least 15 seconds of CUDA-event timing per nonempty case.
The all-expert packed arm completed only three timed samples; these are local
control measurements, not distributional throughput estimates.

| Case | Resident median | Packed median | Packed additional peak CUDA allocation |
|---|---:|---:|---:|
| Decode, 8 selected | 1.235 ms | 150.350 ms | 2.203341 GiB |
| All 288 selected | 32.104 ms | 5,448.947 ms | 10.578341 GiB |
| Clamp stress, 32 selected | 3.881 ms | 607.775 ms | 3.515841 GiB |

The temporary stock kernel and compact scales are not retained. Allocated
memory after profile cleanup returns exactly to its pre-timing value in every
case, in both arms. The independent oracle is allocated after the load receipt
and remains present during timing; incremental peaks subtract that baseline.
Thus a small chunk bound does not bound the entire selected stack, stock
workspace or full-engine memory.

Each arm has CPU/CUDA `torch.profiler` traces and Netdata series on both Sparks.
The packed decode trace attributes about 149 ms of device work to shared eager
selected decoding; bit shifts and ORs dominate, while stock MoE work is small.
The all-expert trace also records substantial command-buffer backpressure.
This is evidence for improving the existing shared selected decoder, not a
reason to introduce a second decoder or promote the research path.

Sparklina mean board power over integer-second interiors was 22.75/27.143/21 W
for resident decode/all/clamp and 29/29.5/24 W for packed, far below the nominal
140 W envelope. Approximate invocation/board-joule was 34.660/1.143/12.105
resident versus 0.2292/0.00622/0.06855 packed. This uses Netdata mean power and
wall invocation rate, with no idle subtraction; the short and coarse power
windows do not establish precise energy consumption. GPU utilization is not
used as a saturation diagnostic. Sparky carried external work in the baseline
window; both boxes' CPU, available-memory and swap-I/O series are preserved.

## Evidence and validation

Evidence root: `/mnt/shared/tessera-measurements/glm-packed-lifecycle-20260908/`.
`resident-01/` and `packed-02/` contain successful launcher/receipt pairs,
inputs, owner/load receipts, core and package identity, traces and `netdata.json`.
`load-netdata.json` retains both boxes during create/load/prepare;
`diagnostic-01/` contains the canonical wire roster and mapping controls.
`derive_summary.py` verifies successful exits/parity and derives `summary.json`.

PrismaBuild CPU regression first: action `4a656f3e5968` failed 11 new tests
against the missing lifecycle API, with the unchanged production gate passing.
Final relevant CPU suite: action `01e7a24ab75a`, 30 passed, zero skips or missing
collection, CPU torch 2.11.0. Compile checks: action `92576ce7164a`, exit 0.
Actual terminal records and CAS payloads were checked, recorded in `pb-audit.json`.
The CPU suite does not cover CUDA; the native controls above do.

Retained failed attempts: two early CPU test assertions were corrected; a
mistyped test filename produced no collection; native `packed-01` stopped at
an overly strict harness assertion that omitted the existing routing bias.
No successful measurement was repeated to fill capacity.

No full-model residency, shared-expert execution, trained-router/model quality,
TP2, heterogeneous expert layouts or release eligibility is claimed.
