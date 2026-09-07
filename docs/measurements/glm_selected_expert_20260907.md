# GLM selected-expert functional control — 2026-09-07

The eager research path passed on the actual GLM E288/top8 owner: selected
packed-window tiles and scales were exact, and complete operator outputs were
bit-exact against the independent full stock FP8 owner. Incorrect expert maps
changed outputs in every case. This is an operator correctness result, not
a throughput, whole-model fit, generation or quality result.

Source: `be0934bbfb`, using `PreparedWindow.stack` and
`PreparedTesseraFp8Module.stack`. The packed owner holds an expert axis and
decodes device IDs into fresh tensors with an explicit eight-expert chunk
bound. The existing window reader and stock vLLM MoE kernels do the work;
there is no CPU weight stream, persistent decoded pool or production route
change. The current batch requires matching window layouts across experts.

The three original expert-0 projection wires from the
[q512 control](glm_q512_control_20260907.md) are repeated across E=288. To
exercise expert identity, the diagnostic down-projection row scale is multiplied
by `1 + bit(global_expert_id, row modulo 9)`. All 288 signatures are distinct.
These are deliberately distinct effective diagnostic weights, not 288 trained
experts. Routing uses `(37 * slot + 19) modulo 288`, with the selected expert
axis reversed before building stock's global-to-compact `expert_map`.

The oracle first compares compact stock tensors to the full independent stock
owner. It then checks every selected packed-decoded tile and scale against
the independent producer materialization and compares the resulting output.
Finally, it swaps two entries in the expert map and requires a changed output.
Both stock legs own their tensors; the result is not a workspace-alias check.

| Case | Tokens | Selected experts | Stock compact map max error | Packed selected max error | Deliberately wrong map max error |
|---|---:|---:|---:|---:|---:|
| Decode | 1 | 8 | 0 | 0 | 0.1875 |
| Every expert | 36 | 288 | 0 | 0 | 0.875 |
| Input multiplied by 64 | 4 | 32 | 0 | 0 | 32 |

The native owner has H=4096, I=2048, top8, clamp=10, TP=1 and EP=1. It is
constructed from the actual GLM model factory configuration and then invokes
the actual `Glm5NextMoE`/`RoutedExperts` class. Trained routing, shared-expert
execution and full-engine generation are outside this control. The original
three repeated-source resident controls also passed in this process.

## Allocator observations

Packed owners occupy 1,236,484,096 bytes for w13 and 617,119,744 for w2,
including their row scales: 1,853,603,840 bytes total. Temporary expansion
remains substantial. Each case synchronizes, preserves the prior global
high-water mark, resets peak counters, and captures decode peak **before**
the independent verification tensors are allocated.

| Case | Fresh decoded weight bytes | Decode peak above existing allocations | Peak including existing allocations and verification |
|---|---:|---:|---:|
| Decode | 201,326,592 | 2,365,816,832 | 18,963,419,648 |
| Every expert | 7,247,757,312 | 11,358,404,608 | 33,519,086,592 |
| Input multiplied by 64 | 805,306,368 | 3,775,102,976 | 20,373,567,488 |

The whole-control peak was 33,519,086,592 allocated bytes and 39,246,102,528
reserved bytes. It includes the resident consumer, full independent stock
oracle, packed research owners and verification temporaries. It is not an
isolated serving footprint. Arithmetic joining the per-case and retained
global counters was checked in the final audit.

No profiler-based speed or work-per-joule comparison was performed. Eager
`torch.unique` synchronizes dynamic cardinality, and the runtime reported
default MoE configurations for E=288, E=8 and E=32. These need profiling and
representative performance measurement before any serving recommendation.

## Execution and validation

The direct vLLM control ran on Sparklina, CPUs 5–8, with a 48 GiB container
cap from 19:28:19 to 19:30:18 UTC. The immutable official image was
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`;
all 4,967 inventoried core files stayed unchanged. Launcher and container
exited zero, no OOM occurred, and harness files stayed unchanged. Exact
container `81f696ffa2464385eff1aa15858a717d91ce1701bf5ec5436923ffd5d7b77629`
and PID 3311074 were independently absent after cleanup; the GPU process roster
was empty. No startup timing is used as a performance claim.

PB CPU validation at `922ea756` selected 66 files across 20 actions, each with
two xdist workers, worksteal distribution, one native thread per worker and
a 2 GiB reservation: **1,532 passed, 156 skipped, zero uncollected modules**.
All used CPU Torch 2.11.0. The skips were 137 GPU-dependent cases and 19 absent
box artifacts; each verbatim reason is retained in the audit. All action
exits, cleanup records, CAS payloads and source bundle hashes were checked.

Pre-change proofs comprise four missing-window-stack failures and two
missing-FP8-stack failures. The first FP8 attempt on Lina lacked Torch and
collected no tests; after installing the official CPU wheel in both Sparks'
scoped test environments, the exact original source snapshot reproduced both
failures. The final targeted window/FP8/reference suite passed 27 tests with
12 CUDA-only skips. Two additional empty-case refusal tests passed after
their demonstrated failures. The final harness compile passed through PB.

One separate nearby fix: prepared windows now clone initial-state provenance
so caller mutation cannot invalidate their private owner. Its regression
failed before the one-line fix and passed afterward. No wire bytes changed.

Evidence root: `/mnt/shared/tessera-glm-native-20260907`. Exact commands and
source/configuration bindings are in the request, launch and audit files.

| Artifact | SHA-256 |
|---|---|
| `selected-functional-audit.json` | `91420270e2d7e34c457bd5183cbdeadde8b1291d59d09ab194ff942f0d0fd2b2` |
| `selected-control-lina-01/selected-expert-receipt.json` | `8a5ef859646c27d463843eaac545aa3f8aa0694d3c734c2614e1eb944f9dd8de` |
| `selected-request-04.json` | `d77d8a27c8e7fc3c3ff4d4eaf5668847281b144815cfb85d74a042779afe4d75` |
| `selected-source-be0934bb.tar` | `22230eec4f6a9d28c017da21deb7a1953070ad318a7b638fc109c801de58dfcd` |
| `selected-impacted-audit.json` | `a047116f6ce1dd5b043b925ae529b8fb74670a20d70286b34aaffd6ab4309d35` |
| `selected-regressions-audit.json` | `cafc1530a43253553bdbf9b021b18d633860af91924af526f62eccc0bdef045e` |

Earlier unexecuted requests 01–03 and their source archives are retained as
superseded, with explicit predecessor bindings. Production remains resident;
packaged runtime cells, release pins and promotion gates are unchanged.
