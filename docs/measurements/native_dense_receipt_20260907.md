# Native dense receipt producer qualification — 2026-09-07

The research producer in `experiments/bench_native_operator.py` implements
Tessera #376. It measures a single eager/resident/TP1 dense operator from the
original cached wire after an independently frozen panel passes activation-QDQ
and output parity. It does not establish release admission, full-model quality,
or complete engine memory. No serving kernel or production default changed.

## Verification

All tests and CUDA qualification ran through PrismaBuild. Terminal return codes,
stdout and CAS payload bytes/hashes were independently checked. CPU validation
used Torch 2.11 CPU on dl380g10, four pytest workers with native threads bounded
to one. The final targeted run covered `tests/test_native_operator_receipt.py`
and `tests/test_native_operator_resources.py`: **98 passed, zero skipped,
zero modules missing**. It did not exercise the CUDA-gated serving test surface.

| Evidence | PB action | Result |
|---|---|---|
| Final CPU regression suite | `df3a8c92abcaed3fca8bbb3c22ca6d35f634f5d93af9a4f3f712dbb3b7055546` | 98 passed |
| Real CUDA allocation observer | `60301769b227` | 123,456 external transient bytes + 1,048,576 Torch bytes; complete conservative bound; zero reported drops/errors |
| Original-wire BF16 lifecycle fixture | `d0ba466869063c82b7c0fdddf4375b7e59ffb4bb1af6ae0646584b15d6116154` | Exact QDQ/output parity for both phases; complete operator bounds |
| Actual LFM R1792 preparation | `98a29949a6551387a101bee59c84abb949d5725c4f0569d73baf1950cf9aa617` | Original cached wire and source/render match; untimed native preparation |

The lifecycle fixture is synthetic: a 32 × 32 BF16 R512 wire and basis-vector
inputs. Its prefill and decode operator bounds were 10,240 and 1,536 bytes,
respectively. These are measured instrumentation fixtures, not LFM estimates.
The observer retained 265,469 bytes across 2,902 preexisting device-static
allocations in a separate startup ledger, with raw context IDs; this ledger is
not a complete model-fixed resource measurement.

The actual preparation uses LFM2.5-8B-A1B
`model.layers.0.feed_forward.w1`, shape 7,168 × 2,048, BF16 K1 R1792,
resident/eager/TP1 on Sparky GB10 sm121. The immutable image is
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
Torch 2.13 and vLLM 0.28.1rc1, driver 595.84. The separate PrismaQuant producer
owns the actual PWC cost and numerical panel. Its predeclared parity tolerance
is `atol = rtol = 0.015625`; preparation observes no operator output and does
not set or relax that tolerance.

## Negative results and boundaries

Regression runs first demonstrated mutation of inputs/references after the
numerical gate, execution/route drift, incomplete CUDA API/allocation pairing,
and startup-static ownership mistakes. The fixes reject these states or retain
them in their explicit resource domain. CPU failure fixtures remain in the
suite; the real allocation fixture includes a nonzero external allocation so
that an accidental false zero cannot pass the qualification.

Early native fixture runs refused uninitialized vLLM TP state, inference-mode
loader tensors without version counters, a missing immutable image declaration,
and Torch/NVIDIA UUID spelling differences. The final CLI initializes the
actual TP1 context, loads under `no_grad`, verifies Docker RepoDigests through
the existing image helper, and joins canonical UUIDs. These were harness
integration defects; none warranted bypassing the runtime or identity gates.

CUPTI and Torch's profiler run in separate processes. Allocation observations
produce a conservative sum of independently observed Torch/external peaks,
including output storage; they are not an exact simultaneous peak. Decision
CUDA-event samples are collected only after the allocation observer stops.
Unknown allocator ownership, lost records, changing reservations and unsupported
async/pool/imported/managed allocation domains remain incomplete. Full-model
fixed storage, KV cache, allocator slack, graphs, routed experts and served
quality require independent evidence. No speedup or energy-efficiency result
is claimed by this implementation qualification.

## Durable artifacts

- `/mnt/shared/tessera-native376-resource/final-qualified-actions.json` records
  the independently checked final CPU/lifecycle CAS receipts and payload hashes.
- `/mnt/shared/tessera-native376-resource/verified-actions.json` and
  `resource-contract-review.md` contain collector build, regression and real-CUDA
  evidence. The immutable observer binary is `native_operator_resources-v2.so`,
  SHA256 `1a893cec8e7386c48b86f69ebb6c677b009f9aa0653afae5b88594e8deaaaf73`.
- `/mnt/shared/tessera-measurements/pq267-native-20260907/dense-r1792-01/`
  retains the original PWC/wire/tensors, transport revisions and
  `native-preflight.json` (SHA256
  `5cb86100285f6e20e1b52ef8fc6529cd2c76203984d9cfb94e30571e9e592575`).
- PB's exact submitted commands are retained in
  `/mnt/shared/prismabuild-fleet/cas/requests/<first-two-action-characters>/<action>.json`.
  CUDA runs use the existing `experiments/runtime_image.sh` declaration helper,
  the PB-owned Docker CPU affinity, and a fresh container process per action.

## Subsequent profiler replay qualification

PB `bd152dddfbe4f2711a02fe007845c3c3a1ace6a902cebd747fe6a034cb2edcb2`
returned zero for a fresh generic preparation, frozen synthetic replay panel and
separate-process `--profile`. Both phases again passed exact QDQ/output parity;
prefill recorded cuBLAS MMA/reduction and elementwise kernels, and decode
recorded cuBLAS GEMV and elementwise kernels. The CPU/CUDA profiler output and
Chrome traces are retained under `native-bf16-r7/profile-r2.json*`; their hashes
and PB CAS bytes were independently checked.

The first replay (`2ce1fccf0f44`) refused a runtime mismatch. Diagnostic action
`8f2ed8e00736` established the sole difference: the original combined
encode/prepare fixture had mapped an encoder-only Triton `cuda_utils` binary.
A fresh generic preparation was therefore used to freeze a new synthetic replay
panel. No source, tensor, numerical tolerance or identity-gate change was made.
The actual LFM preflight already used this generic preparation path.

Host-level evidence from both GB10 hosts is retained at
`/mnt/shared/tessera-native376-resource/netdata-qualification-20260907/`:
CPU, RAM, power, clocks and reported framebuffer series, with request URLs and
row counts in `index.json`. These short instrumentation invocations do not
support an energy-efficiency estimate or sustained-throughput comparison.
