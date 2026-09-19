# Tessera

**Compress the weights. Keep the GPU's native math.**

Tessera is a family of formats for storing large language model weights in
fewer bits and reconstructing them directly into the numeric formats GPUs
already compute with: **NVFP4, FP8, and BF16**. It includes the file format,
encoder, decoders, GPU kernels, checkpoint exporter, and its own vLLM plugin.

The key idea is that **storage precision and compute precision can be chosen
separately**. A weight used in an FP8 multiplication need not occupy eight
bits in the checkpoint. A model that computes with BF16 values need not keep
sixteen bits of persistent storage for every encoded weight. Tessera stores a
compact description of a sequence of weights, then reconstructs the values
needed for computation.

That opens a useful middle ground: a finely adjustable memory budget with
hardware-native arithmetic. The encoder does the expensive search once;
the decoder uses the resulting bits, lookup tables, and scales repeatedly.

The results show why this combination is interesting: **Tessera-8 reaches
EXL3-like reconstruction quality while using FP8 activations**, and a
**roughly four-bit Tessera NVFP4 artifact has matched a production NVFP4
encoder's served quality** at equal expanded residency. The comparisons and
their workloads are [below](#measured-results).

[GitHub](https://github.com/RobTand/tessera) · [How it works](#how-it-works) · [The format family](#the-format-family) ·
[Serving and memory](#serving-and-memory) · [Measured results](#measured-results) ·
[Getting started](#getting-started)

## Why this matters

A language model's weights are the billions of learned numbers it uses to
process text. Keeping them in memory and moving them to the GPU's compute
units is a substantial part of the cost of inference. Quantization reduces
that cost by representing weights with fewer bits, at some loss of accuracy.
Tessera gives that tradeoff several connected controls:

- **Spend bits where they help.** The FP8 and BF16 families expose a root-rate
  grid in steps of 1/256 bit per weight. An allocator can propose different
  rates for different weight matrices, with scales, tables, and metadata
  charged in the actual byte budget.
- **Use more than independent rounding.** Trellis coding chooses a sequence
  of reconstructed weights together. It can trade a slightly worse choice at
  one position for a better sequence overall, under the encoder's objective.
- **Choose the arithmetic as well as the size.** NVFP4 and FP8 routes use
  low-precision weights and activations; the BF16 route keeps activations in
  BF16 while compressing the stored weights.
- **Keep compression resident.** The dense routes hold the wire's packed
  representation in GPU memory and decode it inside the GEMM, so the
  checkpoint's compression survives into serving instead of being expanded at
  load.
- **Make the artifact explain itself.** The wire carries its decoding tables,
  scales, layout, and identity. The parser, byte accountant, and serving
  contract make those choices inspectable.

For scale, one billion weights at four bits each occupy **0.5 GB**, versus
**2 GB** at sixteen bits. That is arithmetic for the weights alone: scales,
tables, metadata, unquantized tensors, and inference workspace add to it.
Tessera explicitly accounts for the representation overhead; total serving
memory also includes activations, the KV cache, and temporary buffers.

## How it works

### Encode a path through possible weights

Ordinary scalar quantization picks a nearby representable value for each
weight. Tessera's trellis encoder searches a network of allowed sequences.
Each step stores a few bits; the current bits and a small amount of preceding
state select a reconstruction from a table. The sequence shares information
across positions instead of paying for an independent table index each time.

The **Viterbi algorithm** finds the minimum-cost path for the fixed trellis
and objective. With calibration activations, the encoder can also use
**Hessian-aware LDLQ error feedback**: it accounts for how errors in one
column affect the layer's output and compensates through later columns.
Scale refitting evaluates the scales in the representation actually stored
in the artifact.

The encoder searches; the decoder follows the chosen path. For the window
body, each state can be reconstructed from a short window of packed bits, so
decoding does not require running Viterbi or walking the whole column from
its beginning.

### Land directly on the hardware's numbers

The FP8 and BF16 recipes build a Gaussian-quantile lookup table, snap its
entries to the chosen numeric format, and store that table in the artifact.
The usual window is **14 bits**, giving **16,384 table entries**. Those entries
are reconstruction choices, not 16,384 distinct FP8 numbers; the FP8 entries
are already legal FP8 codes.

For a scalar window stream with `R` bits per position:

```text
state[t] = ((state[t-1] << R) | bits[t]) & ((1 << L) - 1)
code[t]  = table[state[t]]
weight   = grid_value(code[t]) × row_scale × global_scale
```

Here `L` is the window width. The stored table and scales fully determine the
reconstruction. Quantizing the original model is lossy; decoding its Tessera
artifact is deterministic. The compact native load path expands no reference
tile; native decoders are held to the reference reconstruction by the routes'
test suites, not at load ([architecture §3.3](docs/ARCHITECTURE.md)).

The NVFP4 recipe uses a related construction: a **span-2 trellis over pairs
of E2M1 values**, with a lookup-table scale plane. It reconstructs the packed
4-bit values and block scales that the stock NVFP4 matrix multiply consumes.

### Give fractional bit rates an exact meaning

`q256` names the root rate: `q256 / 256`. For example, a scalar window root
rate of **4.25** is represented by a deterministic schedule of four-bit and
five-bit columns. Over a complete schedule period, three quarters of the
columns use four bits and one quarter use five. There are no fractional bits
inside an individual codeword.

The root rate is only the body budget. A window table, row scales, alignment,
headers, and the manifest also occupy bytes. Their cost depends on matrix
shape, so **“four-bit body” and “four-bit file” are different quantities**.
The exact-byte accountant prices the artifact that the exporter writes.

## The format family

All three serving families share one schema, encoder framework, byte
accountant, and plugin. Their reconstruction alphabet and activation
precision differ:

| Family | Stored construction | Matrix-multiply format |
|---|---|---|
| **Tessera NVFP4** | Span-2 trellis over E2M1 pairs; LUT16 block scales | NVFP4 weights and activations (**W4A4**) |
| **Tessera FP8** | Window trellis; E4M3 table; per-row scales | FP8 weights and activations (**W8A8**) |
| **Tessera BF16** | Window trellis; BF16 table; per-row scales | BF16 weights and activations (**W16A16**) |

**W** and **A** describe the compute format of weights and activations, not
the number of bits stored per weight. Compressing onto a BF16 grid still
changes the model's weights; it does not recover the original BF16 model
losslessly.

The packaged [runtime contract](src/tessera/serving/runtime_contract.json)
publishes each family's format limits and its serving-evidence cells; read
those fields rather than a copy here. Present dispatch is described in
[the architecture](docs/ARCHITECTURE.md), and the exporter's
[`wire_recipe`](src/tessera/export.py) selects what is written.

## Serving and memory

A Tessera checkpoint selects `quant_method: "tessera"`. The current dense
routes (**NVFP4, FP8, BF16**) execute packed native weights: the compact
reader holds the wire's planes in GPU memory and the in-forward GEMM decodes
them, with no `[rows, columns]` weight tile materialised. `resident` and
`streamed` prepare the same units; the mode must still be declared explicitly
(`TESSERA_SERVE_MODE`, no default). Routed mixture-of-experts is **resident,
eager mode only**.

## A format you can inspect

The wire schema is **`prismaquant.tessera.v1`**. Each encoded unit carries a
manifest and typed data planes describing its shape, grid, trellis body,
rate schedule, scales, tables, and legal terminal lengths. A decoder reads
the stored tables; it does not depend on rebuilding an encoder's choices.

Three properties make this useful beyond an isolated quantizer:

- **Priced = written = served.** Integer byte accounting distinguishes the
  selected artifact, a bundle of alternatives, compressed residency, and
  expanded residency. Export checks its wire bytes against the plan.
- **Checked identity and integrity.** Content digests cover the declared
  data; canonical padding and legal plane extents are validated. Encoder
  identity records which implementation and settings produced the artifact.
- **A checked route to execution.** Export admission reads the runtime's
  contract; native decoders are held to the reference by the test suites, and
  a route census checks the declared modules against actual dispatch.

The schema also supports verifiable shorter terminal prefixes, and the
sharding machinery can derive rank-local slices from a common encoded unit.
**Today's exporter writes one terminal per unit**, so shipped checkpoints
are not truncatable rate ladders. Route-specific sharding geometry and its
qualification limits are stated in [the architecture](docs/ARCHITECTURE.md)
and the [runtime contract](src/tessera/serving/runtime_contract.json).

See the [byte-level specification](https://github.com/RobTand/tessera/blob/v0.1.0/docs/schema/prismaquant.tessera.v1.md)
and [plan-to-serve architecture](docs/ARCHITECTURE.md)
for the complete contracts.

## Measured results

The design provides a broad space of possible size/quality tradeoffs. Results
belong to a particular model, artifact, encoder, runtime, and workload.
**Tessera does not claim a universal accuracy, speed, or energy advantage.**

### How Tessera compares with EXL3

EXL3 is a useful reference because it also uses trellis coding, but its
reconstructed weights are used with **16-bit activations**. Tessera targets
the FP4 and FP8 arithmetic paths as well. The interesting question is how
much reconstruction quality survives that additional compression of the
computation itself.

On **six GLM-5.3-Flash expert projections**, the production-wire Tessera-8
screen gives the following results. Every Tessera row includes **FP8
activation quantization**. Values are output-error ratios: **below 1.0 is
better for Tessera**.

| Tessera-8 encoded bits/weight | Window | vs EXL3 with 16-bit activations | vs projected EXL3 with 4-bit activations | vs projected EXL3 with 8-bit activations |
|---:|---:|---:|---:|---:|
| **3.009** | 12 bits | **0.973×** | **0.819×** | **0.958×** |
| **4.020** | 14 bits | **1.005×** | **0.618×** | **0.946×** |
| **5.020** | 14 bits | **1.179×** | **0.434×** | **0.964×** |

At about three bits, Tessera-8 has **2.7% less output error** than EXL3's
16-bit-activation reference in this screen, despite using FP8 activations.
At about four bits it is essentially level with that reference, and has
**5.4% less error** than EXL3 projected onto the same 8-bit activation
precision. At five bits it still beats the A8 projection, while EXL3 with
16-bit activations retains the advantage.

“Projected EXL3” means the harness applies the NVFP4 or FP8 activation
quantizer to the same held-out inputs and evaluates EXL3's reconstructed
weights with those inputs. These are calculated output-error comparisons,
not EXL3 serving modes or measured EXL3 throughput. The A16 reference is the
weight-only output-error leg; the harness computes these products in FP32
and does not reproduce a complete 16-bit serving kernel.

The table uses geometric means across six 2048×4096 matrices, with the last
1,024 capture rows held out. EXL3's error is interpolated between measured
rates to each Tessera artifact's byte budget. These Tessera arms use scale
refitting **without LDLQ**; EXL3 uses its reference quantizer with LDLQ.
The results are historical tensor screens, not whole-model KL or a result
for dense layers with different activation distributions. Sources:
[3-bit production-wire results](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/results/tessera_frontier.json),
[4/5-bit production-wire results](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/results/tessera_frontier_L14.json),
and the [measurement harness](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/tessera_frontier.py).

**Tessera-4 has also improved substantially.** The later LDLQ + scale-refit
measurement reduces its weight-only output error to **1.097× EXL3 K4** on
the same six experts, at about four encoded bits per weight. With its
4-bit activation quantizer included, its output error is **0.11319**,
versus EXL3's **0.06736** weight-only reference and **0.10912** A4
projection: about **1.68×** and **1.04×**, respectively. That is close to
the same-activation comparison; the activation cost remains material against
EXL3's A16 reference. [LDLQ measurement](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-ldlq-lut-plane-served-2026-09-02.md#glm-cross-check),
[EXL3 reference values](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/results/tessera_frontier_L14.json).

The computational opportunity is native **W4A4 or W8A8 matrix multiplication**
from a trellis-compressed artifact. Low-precision tensor-core arithmetic
and reduced weight traffic can both help throughput, but the benefit depends
on batch size, decoding cost, and hardware. These quality comparisons do
not measure an end-to-end speedup over EXL3.

### How Tessera compares with Gridbook

Gridbook provides another useful comparison: its FP8-CB family uses vector
codebooks and reconstructs FP8 weights, so both sides can be evaluated with
the **same FP8 activation quantizer**. On the same six GLM expert projections,
Tessera's window trellis gives lower output error:

| Nominal budget | Tessera-8: encoded bits/weight / output error | Gridbook FP8-CB + imatrix: bits/weight / output error | Tessera / Gridbook error |
|---|---:|---:|---:|
| **4 bits/weight** | 4.020 / **0.06732** | 4.008 / 0.08961 (K32) | **0.751× — 24.9% less** |
| **5 bits/weight** | 5.020 / **0.04017** | 5.008 / 0.05154 (K40) | **0.779× — 22.1% less** |

These are geometric means of the recorded per-tensor measurements, using
Gridbook's **imatrix-enabled, LDLQ-off** configuration. Tessera uses the
14-bit window and scale refitting without LDLQ. The budgets are close,
not identical: the extra Tessera storage is about **0.0125 bit/weight**.
These direct comparisons do not interpolate that difference away. Both
columns include activation quantization; neither is a served model score
or a throughput measurement. Sources:
[Gridbook measurements](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/results/tessera_vs_exl3_followups.json)
and [Tessera measurements](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/results/tessera_frontier_L14.json).

### End-to-end serving

The serves below were measured on **NVIDIA GB10 (`sm_121`), single rank**.
KL measures disagreement with a BF16 teacher's next-token probabilities;
lower is better. The cited quality scores use **top-1,024 intersection KL
lower bounds**, not full-vocabulary KL or a general benchmark score.

| Experiment | Recorded result | What it establishes |
|---|---|---|
| **Qwen3-0.6B, NVFP4** | Tessera **0.50997** versus PrismaQuant NVFP4 GPTQ+JSO **0.51058**, at equal expanded residency over 4,088 scored positions | Parity on this corpus with **opt-in `ldlq_block=8`**. A 0.12% margin is not a quality lead. [Receipt](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-dense4-gap-2026-09-03.md) |
| **Qwen3-0.6B, BF16 reconstruction** | **0.004923** at roughly **7.13 encoded bits/weight**; identical scores between resident and streamed modes | A compressed BF16-grid artifact served with low measured divergence on this corpus. Quality was measured in **eager** mode; eager and compiled route censuses each covered all 112 declared modules. [Receipt](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-bf16-route-served-2026-09-02.md) |
| **LFM2.5-8B-A1B, FP8 routed MoE** | All **22 expert stacks / 2,112 projections** covered in prefill and decode; prefill KL lower bound **0.0832**, upper bound **1.179** at the declared probability floor | Dispatch coverage and a bounded prefill comparison over 4,096 positions. No decode KL. A later controlled chat smoke records coherent answers from both source and student. [Quality receipt](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-lfm-campaign-2026-09-04.md), [chat smoke](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/moe-smoke-recorded-2026-09-05.md) |

These measured results are historical artifacts and dispatch: compiled
coverage does not imply compiled quality measurement, and a changed encoder
or dispatch requires fresh qualification. The contract records
[that evidence scope](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/encoder-evidence-scope-2026-09-05.md).

**Per-layer rate allocation is gated.** PrismaQuant can propose rates across
the families, but a multi-rung plan must pass a served comparison with a
byte-matched uniform control. This matters in practice: one surrogate-selected
allocation served about **2× worse KL** than its uniform control. Weight-space
error is useful for screening candidates, but the model's executed output
is the promotion metric. See
[the allocation gates](https://github.com/RobTand/tessera/blob/v0.1.0/docs/ARCHITECTURE.md#4-allocation-and-the-uniform-gate).

## Getting started

The distribution name is **`tessera-quant`**; the Python import is **`tessera`**.
Install the release:

```bash
pip install tessera-quant==0.1.0
pip install 'tessera-quant[serve]==0.1.0'  # also install vLLM and JIT-build tooling
```

Or install from a source checkout:

```bash
git clone https://github.com/RobTand/tessera.git
cd tessera
pip install -e .            # encoder, readers, and plugin entry point
pip install -e '.[serve]'   # also install vLLM and the Python JIT-build tooling
```

The `serve` and `kernels` extras supply `ninja`; route-specific compiler
prerequisites are stated in
[the architecture's JIT-build section](docs/ARCHITECTURE.md). Native dispatch
has no materialising fallback.

For an **already exported, admitted dense Tessera checkpoint**, declare the
residency when starting vLLM (see [Serving and memory](#serving-and-memory)):

```bash
TESSERA_SERVE_MODE=streamed vllm serve /path/to/tessera-checkpoint
```

Installing vLLM from its package index does not reproduce an attested runtime:
the [packaged contract](src/tessera/serving/runtime_contract.json) names the
exact serving images and supported combinations.

For encoding and integration, start with the
[architecture and export pipeline](https://github.com/RobTand/tessera/blob/v0.1.0/docs/ARCHITECTURE.md#2-the-pipeline),
[`export_tessera_serving.py`](https://github.com/RobTand/tessera/blob/v0.1.0/experiments/export_tessera_serving.py),
and the [`tessera.export` API](https://github.com/RobTand/tessera/blob/v0.1.0/src/tessera/export.py).

## Layout

The main implementation lives under `src/tessera/`:

```text
encode.py          trellis search and scale refitting
compensate.py      Hessian-aware error feedback
wire.py            packed bit streams
container.py       serialization, parsing, and integrity checks
footprint.py       exact-byte accounting
decode.py          reference reconstruction and native tile materialization
export.py          encoder recipes and checkpoint export
kernel_window.py   GPU window decoding
kernel_window_gemv.py  compressed matrix-vector multiplication
serving/           vLLM plugin, CUDA sources, and runtime contract
```

## Development and further reading

- [Architecture](https://github.com/RobTand/tessera/blob/v0.1.0/docs/ARCHITECTURE.md): recipes, byte accounting, export, serving, and promotion gates.
- [Wire specification](https://github.com/RobTand/tessera/blob/v0.1.0/docs/schema/prismaquant.tessera.v1.md): container layout and decoding rules.
- [Measurement records](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements): workloads, controls, results, and limitations.
- [Suite population ledger](https://github.com/RobTand/tessera/blob/v0.1.0/docs/status/suite-populations.md): validation by commit, device, run mode, and skips.
- [Working rules](https://github.com/RobTand/tessera/blob/v0.1.0/AGENTS.md): contribution and test requirements. Tests and GPU work run through PrismaBuild.

Hosted CI checks the bytes-only boundary and built-package contents. GPU
serving coverage requires its own measured population; read the GPU and
CPU rows together in the suite ledger.

Links to historical measurements and release documents are pinned to
**v0.1.0**, so they describe the same release on GitHub and the Python package
index. The current architecture and packaged-contract links are relative to
this checkout.

**License:** [MIT](https://github.com/RobTand/tessera/blob/v0.1.0/LICENSE).
